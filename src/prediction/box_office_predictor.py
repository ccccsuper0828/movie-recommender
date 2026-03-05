"""
Box Office Revenue Predictor — faithful replication of Movie-Analysis
(Kaggle TMDB Box Office Prediction, top-7 %, 95/1400).

Data:  Kaggle competition train.csv (3000 movies) + external features
       (release_dates_per_country, TrainAdditionalFeatures).
Models: CatBoost + XGBoost + LightGBM → exponential weighted average.
"""
# @author 成员 D — 票房预测 & 数据可视化

import ast
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import pickle
from config.settings import get_settings

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LGB_OK = True
except ImportError:
    LGB_OK = False

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    from catboost import CatBoostRegressor, Pool
    CAT_OK = True
except ImportError:
    CAT_OK = False


# ── Utility ──────────────────────────────────────────────────────────
def _safe_eval(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(str(x))
    except (ValueError, SyntaxError):
        return []


def _rmse(y, yp):
    return float(np.sqrt(mean_squared_error(y, yp)))


# ── Manual data corrections (from reference project) ─────────────────
_TRAIN_BUDGET_FIX = {
    90: 30000000, 118: 60000000, 149: 18000000, 464: 20000000,
    470: 13000000, 513: 930000, 797: 8000000, 819: 90000000,
    850: 90000000, 1007: 2, 1112: 7500000, 1131: 4300000,
    1359: 10000000, 1542: 1, 1570: 15800000, 1571: 4000000,
    1714: 46000000, 1721: 17500000, 1885: 12, 2091: 10,
    2268: 17500000, 2491: 6, 2602: 31000000, 2612: 15000000,
    2696: 10000000, 2801: 10000000, 335: 2, 348: 12, 640: 6,
    696: 1, 1199: 5, 1282: 9, 1347: 1, 1755: 2, 1801: 5,
    1918: 592, 2033: 4, 2118: 344, 2252: 130, 2256: 1,
}
_TRAIN_REVENUE_FIX = {16: 192864, 313: 12000000, 451: 12000000, 1865: 25000000}


# ── Feature engineering (matches reference) ──────────────────────────
def _extract_names(col, max_n: int = 5) -> pd.Series:
    return col.apply(_safe_eval).apply(
        lambda x: [d.get("name", "") for d in x if isinstance(d, dict)][:max_n]
    )


def _get_director(crew_col) -> pd.Series:
    def _dir(x):
        for m in _safe_eval(x):
            if isinstance(m, dict) and m.get("job") == "Director":
                return m.get("name", "")
        return ""
    return crew_col.apply(_dir)


class BoxOfficePredictor:
    """
    CatBoost + XGBoost + LightGBM ensemble for revenue prediction.

    Follows the pluggable predictor interface: fit() → predict() → save() → load().
    To replace with a different model, create a new class with the same methods
    and register it: PREDICTOR_REGISTRY.register("new_model", NewPredictor)
    """

    def __init__(self, n_folds: int = 5, seed: int = 2019):
        self.name = "LGB+XGB+CatBoost Ensemble"
        self.n_folds = n_folds
        self.seed = seed

        self.lgb_models: List = []
        self.xgb_models: List = []
        self.cat_models: List = []
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.FEATURE_COLS: List[str] = []
        self._is_fitted = False
        self.cv_results: Optional[Dict[str, Any]] = None
        self.fold_results: List[Dict] = []

    # ──────────────────────────────────────────────────────────────────
    # Data loading (Kaggle competition dataset + extras)
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _kaggle_dir() -> Path:
        return get_settings().data_dir / "raw" / "kaggle_bo"

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train & test with corrections + extra features."""
        kdir = self._kaggle_dir()
        train = pd.read_csv(kdir / "train.csv")
        test = pd.read_csv(kdir / "test.csv")

        # Manual budget / revenue corrections
        for row_id, val in _TRAIN_BUDGET_FIX.items():
            train.loc[train["id"] == row_id, "budget"] = val
        for row_id, val in _TRAIN_REVENUE_FIX.items():
            train.loc[train["id"] == row_id, "revenue"] = val

        # Fill budget=0 with median of same year (critical for performance)
        for df in [train, test]:
            df["_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
            year_medians = df[df["budget"] > 0].groupby("_year")["budget"].median()
            global_median = df.loc[df["budget"] > 0, "budget"].median()
            mask = df["budget"] == 0
            df.loc[mask, "budget"] = df.loc[mask, "_year"].map(year_medians).fillna(global_median)
            df.drop(columns=["_year"], inplace=True)

        # Merge release-date per-country
        rd_path = kdir / "release_dates_per_country.csv"
        if rd_path.exists():
            rd = pd.read_csv(rd_path)
            rd["id"] = range(1, len(rd) + 1)
            rd.drop(columns=["original_title", "title"], errors="ignore", inplace=True)
            train = train.merge(rd, on="id", how="left")
            test = test.merge(rd, on="id", how="left")

        # Merge additional features (popularity2, rating)
        for split, fname in [(train, "TrainAdditionalFeatures.csv"),
                             (test, "TestAdditionalFeatures.csv")]:
            fp = kdir / fname
            if fp.exists():
                extra = pd.read_csv(fp)[["imdb_id", "popularity2", "rating"]]
                idx = split.index  # preserve
                merged = split.merge(extra, on="imdb_id", how="left")
                merged.index = idx
                for c in ["popularity2", "rating"]:
                    if c in merged.columns:
                        split[c] = merged[c].values

        return train, test

    # ──────────────────────────────────────────────────────────────────
    # Feature engineering
    # ──────────────────────────────────────────────────────────────────
    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)

        # Core numeric
        f["budget"] = df["budget"].fillna(0).astype(float)
        f["budget_log"] = np.log1p(f["budget"])
        f["budget_was_zero"] = (df["budget"].fillna(0) <= 1).astype(int)
        f["popularity"] = df["popularity"].fillna(0).astype(float)
        f["popularity_log"] = np.log1p(f["popularity"])
        f["runtime"] = df["runtime"].fillna(0).astype(float)

        # External extras
        if "popularity2" in df.columns:
            f["popularity2"] = df["popularity2"].fillna(0).astype(float)
            f["popularity2_log"] = np.log1p(f["popularity2"])
        if "rating" in df.columns:
            f["ext_rating"] = df["rating"].fillna(0).astype(float)
        if "theatrical" in df.columns:
            f["theatrical"] = df["theatrical"].fillna(0).astype(float)
        if "theatrical_limited" in df.columns:
            f["theatrical_limited"] = df["theatrical_limited"].fillna(0).astype(float)

        # Release date
        f["release_year"] = 2000
        f["release_month"] = 1
        f["release_dow"] = 0
        for i, d in enumerate(df["release_date"]):
            if d and isinstance(d, str):
                try:
                    dt = pd.to_datetime(d)
                    f.iat[i, f.columns.get_loc("release_year")] = dt.year
                    f.iat[i, f.columns.get_loc("release_month")] = dt.month
                    f.iat[i, f.columns.get_loc("release_dow")] = dt.dayofweek
                except Exception:
                    pass
        f["is_summer"] = f["release_month"].isin([5, 6, 7]).astype(int)
        f["is_holiday"] = f["release_month"].isin([11, 12]).astype(int)

        # Language
        lang = df["original_language"].fillna("en")
        f["is_english"] = (lang == "en").astype(int)
        if "lang" not in self.label_encoders:
            le = LabelEncoder()
            f["lang_enc"] = le.fit_transform(lang)
            self.label_encoders["lang"] = le
        else:
            le = self.label_encoders["lang"]
            f["lang_enc"] = lang.apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        # Genres
        gn = _extract_names(df["genres"])
        f["n_genres"] = gn.apply(len)
        for g in ["Drama", "Comedy", "Action", "Thriller", "Adventure",
                   "Science Fiction", "Horror", "Romance", "Animation",
                   "Fantasy", "Family", "Crime"]:
            f[f"g_{g}"] = gn.apply(lambda gs, _g=g: int(_g in gs))

        # Production companies
        if "production_companies" in df.columns:
            pc = _extract_names(df["production_companies"])
            f["n_companies"] = pc.apply(len)
            majors = {"Warner Bros.", "Universal Pictures", "Paramount Pictures",
                      "Twentieth Century Fox Film Corporation", "Columbia Pictures",
                      "Walt Disney Pictures", "New Line Cinema", "Lionsgate",
                      "Metro-Goldwyn-Mayer (MGM)", "Touchstone Pictures"}
            f["has_major"] = pc.apply(lambda ns: int(any(n in majors for n in ns)))

        # Collection (sequel/franchise)
        if "belongs_to_collection" in df.columns:
            f["in_collection"] = df["belongs_to_collection"].apply(
                lambda x: int(bool(x) and str(x).strip() not in ("", "nan", "None"))
            )

        # Keywords
        if "Keywords" in df.columns:
            f["n_kw"] = df["Keywords"].apply(_safe_eval).apply(len)
        elif "keywords" in df.columns:
            f["n_kw"] = df["keywords"].apply(_safe_eval).apply(len)
        else:
            f["n_kw"] = 0

        # Cast / crew counts
        if "cast" in df.columns:
            f["n_cast"] = df["cast"].apply(_safe_eval).apply(len)
        if "crew" in df.columns:
            f["n_crew"] = df["crew"].apply(_safe_eval).apply(len)
            f["has_director"] = _get_director(df["crew"]).apply(lambda x: int(bool(x)))

        # Text lengths
        f["overview_len"] = df["overview"].fillna("").apply(len)
        f["title_len"] = df["title"].fillna("").apply(len)
        if "homepage" in df.columns:
            f["has_homepage"] = df["homepage"].notna().astype(int)
        if "tagline" in df.columns:
            f["has_tagline"] = df["tagline"].notna().astype(int)

        # Country release counts
        country_cols = [c for c in df.columns if len(c) == 2 and c.isupper() and c not in ("id",)]
        if country_cols:
            for c in country_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            f["n_release_countries"] = df[country_cols].sum(axis=1).astype(int)
            f["released_US"] = df.get("US", pd.Series(0, index=df.index)).fillna(0).astype(int)

        # Interactions (key for boosting models)
        f["budget_x_pop"] = f["budget_log"] * f["popularity_log"]
        if "popularity2_log" in f.columns:
            f["budget_x_pop2"] = f["budget_log"] * f["popularity2_log"]
        if "ext_rating" in f.columns:
            f["budget_x_rating"] = f["budget_log"] * f["ext_rating"]
            f["pop_x_rating"] = f["popularity_log"] * f["ext_rating"]
        if "theatrical" in f.columns:
            f["budget_x_theatrical"] = f["budget_log"] * f["theatrical"]
        if "n_release_countries" in f.columns:
            f["budget_x_countries"] = f["budget_log"] * np.log1p(f["n_release_countries"])
        f["runtime_x_budget"] = f["runtime"] * f["budget_log"]

        # Budget percentile within year
        f["budget_year_pct"] = 0.0
        for yr in f["release_year"].unique():
            mask = f["release_year"] == yr
            if mask.sum() > 1:
                f.loc[mask, "budget_year_pct"] = f.loc[mask, "budget"].rank(pct=True)

        # Budget / popularity ratio
        f["budget_per_pop"] = np.where(
            f["popularity"] > 0, f["budget_log"] / np.log1p(f["popularity"]), 0
        )

        self.FEATURE_COLS = list(f.columns)
        return f

    # ──────────────────────────────────────────────────────────────────
    # Fit
    # ──────────────────────────────────────────────────────────────────
    def fit(self, _unused_df=None) -> "BoxOfficePredictor":
        """
        Train models on Kaggle competition data (ignores _unused_df).
        """
        train, _ = self.load_data()
        logger.info(f"Box-office: {len(train)} training movies")

        X = self._build_features(train)
        y = np.log1p(train["revenue"].values.astype(float))

        # ── CV-safe target encodings (director, company, genre combo, language) ──
        global_mean = y.mean()

        def _cv_target_encode(col_series, smoothing=10):
            enc = np.full(len(train), global_mean)
            kf_te = KFold(n_splits=5, shuffle=True, random_state=42)
            for tr_i, va_i in kf_te.split(train):
                means = pd.Series(y[tr_i]).groupby(col_series.iloc[tr_i]).mean()
                counts = col_series.iloc[tr_i].groupby(col_series.iloc[tr_i]).size()
                smooth = (counts * means + smoothing * global_mean) / (counts + smoothing)
                enc[va_i] = col_series.iloc[va_i].map(smooth).fillna(global_mean).values
            return enc

        train["_director"] = _get_director(train.get("crew", pd.Series(dtype=str)))
        X["director_te"] = _cv_target_encode(train["_director"], smoothing=10)

        def _main_company(x):
            items = _safe_eval(x)
            return items[0].get("name", "unknown") if items and isinstance(items[0], dict) else "unknown"
        train["_company"] = train["production_companies"].apply(_main_company)
        X["company_te"] = _cv_target_encode(train["_company"], smoothing=20)

        train["_genre_combo"] = train["genres"].apply(
            lambda x: "|".join(sorted([g.get("name","") for g in _safe_eval(x) if isinstance(g,dict)][:3]))
        )
        X["genre_combo_te"] = _cv_target_encode(train["_genre_combo"], smoothing=15)
        X["lang_te"] = _cv_target_encode(train["original_language"].fillna("en"), smoothing=20)

        # Polynomial features
        X["budget_sq"] = X["budget_log"] ** 2
        if "popularity2_log" in X.columns:
            X["pop2_sq"] = X["popularity2_log"] ** 2

        # Remove extreme outliers (top/bottom 0.5%)
        cap = np.percentile(y, 99.5)
        floor = np.percentile(y, 0.5)
        mask = (y >= floor) & (y <= cap)
        X = X[mask].reset_index(drop=True)
        y = y[mask]
        logger.info(f"After outlier removal: {len(X)} movies")

        self.FEATURE_COLS = list(X.columns)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        oof_lgb = np.zeros(len(X))
        oof_xgb = np.zeros(len(X))
        oof_cat = np.zeros(len(X))

        self.lgb_models, self.xgb_models, self.cat_models = [], [], []
        self.fold_results: List[Dict] = []  # per-fold metrics for visualization

        for fold, (tr, va) in enumerate(kf.split(X)):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y[tr], y[va]
            fold_info = {"fold": fold + 1}

            # ── LightGBM (balanced regularisation) ──
            if LGB_OK:
                dt = lgb.Dataset(Xtr, label=ytr)
                dv = lgb.Dataset(Xva, label=yva, reference=dt)
                evals_result = {}
                m = lgb.train(
                    {"objective": "regression", "metric": "rmse",
                     "learning_rate": 0.03,
                     "num_leaves": 20,
                     "max_depth": 5,
                     "min_child_samples": 60,
                     "feature_fraction": 0.5,
                     "bagging_fraction": 0.6,
                     "bagging_freq": 3,
                     "reg_alpha": 2.0,
                     "reg_lambda": 8.0,
                     "min_gain_to_split": 0.3,
                     "path_smooth": 5.0,
                     "verbose": -1,
                     "seed": self.seed},
                    dt, 3000, valid_sets=[dt, dv], valid_names=["train", "val"],
                    callbacks=[
                        lgb.early_stopping(100),
                        lgb.log_evaluation(0),
                        lgb.record_evaluation(evals_result),
                    ],
                )
                self.lgb_models.append(m)
                oof_lgb[va] = m.predict(Xva)
                fold_info["lgb_rmse"] = float(m.best_score["val"]["rmse"])
                fold_info["lgb_rounds"] = m.best_iteration
                fold_info["lgb_train_curve"] = evals_result.get("train", {}).get("rmse", [])
                fold_info["lgb_val_curve"] = evals_result.get("val", {}).get("rmse", [])

            # ── XGBoost (balanced regularisation) ──
            if XGB_OK:
                dtx = xgb.DMatrix(Xtr, label=ytr)
                dvx = xgb.DMatrix(Xva, label=yva)
                xgb_evals = {}
                m = xgb.train(
                    {"objective": "reg:squarederror", "eval_metric": "rmse",
                     "learning_rate": 0.03,
                     "max_depth": 4,
                     "min_child_weight": 30,
                     "subsample": 0.6,
                     "colsample_bytree": 0.5,
                     "reg_alpha": 2.0,
                     "reg_lambda": 8.0,
                     "gamma": 0.5,
                     "seed": self.seed},
                    dtx, 3000, evals=[(dtx, "train"), (dvx, "val")],
                    early_stopping_rounds=100, verbose_eval=False,
                    evals_result=xgb_evals,
                )
                self.xgb_models.append(m)
                oof_xgb[va] = m.predict(dvx)
                fold_info["xgb_rmse"] = float(min(xgb_evals["val"]["rmse"]))
                fold_info["xgb_rounds"] = int(np.argmin(xgb_evals["val"]["rmse"])) + 1
                fold_info["xgb_train_curve"] = xgb_evals["train"]["rmse"]
                fold_info["xgb_val_curve"] = xgb_evals["val"]["rmse"]

            # ── CatBoost (balanced regularisation) ──
            if CAT_OK:
                m = CatBoostRegressor(
                    iterations=3000, learning_rate=0.03, depth=4,
                    l2_leaf_reg=15, random_seed=self.seed,
                    eval_metric="RMSE", verbose=0,
                    early_stopping_rounds=100,
                    min_child_samples=40,
                    subsample=0.6,
                    colsample_bylevel=0.5,
                    random_strength=1.5,
                )
                m.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=0)
                self.cat_models.append(m)
                oof_cat[va] = m.predict(Xva)
                fold_info["cat_rmse"] = float(m.best_score_["validation"]["RMSE"])
                fold_info["cat_rounds"] = m.best_iteration_
                # Extract CatBoost learning curves
                try:
                    cat_evals = m.evals_result_
                    fold_info["cat_train_curve"] = cat_evals.get("learn", {}).get("RMSE", [])
                    fold_info["cat_val_curve"] = cat_evals.get("validation", {}).get("RMSE", [])
                except Exception:
                    pass

            self.fold_results.append(fold_info)
            logger.info(f"  Fold {fold+1}/{self.n_folds} done")

        # ── Optimised weighted ensemble (CatBoost-heavy based on CV) ──
        parts, weights = [], []
        if LGB_OK and self.lgb_models:
            parts.append(oof_lgb); weights.append(0.25)
        if XGB_OK and self.xgb_models:
            parts.append(oof_xgb); weights.append(0.20)
        if CAT_OK and self.cat_models:
            parts.append(oof_cat); weights.append(0.55)

        if not parts:
            raise RuntimeError("No model available. Install lightgbm / xgboost / catboost.")

        wsum = sum(weights)
        oof = sum(w / wsum * p for w, p in zip(weights, parts))

        rmsle = _rmse(y, oof)  # RMSLE since y = log1p(revenue)
        y_real = np.expm1(y)
        oof_real = np.expm1(oof)
        from sklearn.metrics import mean_absolute_error
        mae = float(mean_absolute_error(y_real, oof_real))

        self.cv_results = {
            "n_movies": len(train),
            "n_folds": self.n_folds,
            "rmsle": rmsle,
            "mae": mae,
            "models": (
                (f"LGB({len(self.lgb_models)})" if self.lgb_models else "")
                + (f" XGB({len(self.xgb_models)})" if self.xgb_models else "")
                + (f" Cat({len(self.cat_models)})" if self.cat_models else "")
            ).strip(),
        }
        logger.info(
            f"CV: RMSLE={rmsle:.4f}, MAE=${mae/1e6:.1f}M"
        )
        self._is_fitted = True
        return self

    # ──────────────────────────────────────────────────────────────────
    # Predict
    # ──────────────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        X = self._build_features(df)

        # Add target-encoding columns with global mean (safe default for unseen data)
        global_mean = 16.0  # approx mean of log1p(revenue)
        for col in self.FEATURE_COLS:
            if col not in X.columns:
                X[col] = global_mean if col.endswith("_te") else 0.0

        # Ensure column order matches training
        X = X.reindex(columns=self.FEATURE_COLS, fill_value=0.0)

        parts, weights = [], []
        if self.lgb_models:
            p = np.mean([m.predict(X) for m in self.lgb_models], axis=0)
            parts.append(p); weights.append(0.25)
        if self.xgb_models:
            dx = xgb.DMatrix(X)
            p = np.mean([m.predict(dx) for m in self.xgb_models], axis=0)
            parts.append(p); weights.append(0.20)
        if self.cat_models:
            p = np.mean([m.predict(X) for m in self.cat_models], axis=0)
            parts.append(p); weights.append(0.55)

        wsum = sum(weights)
        pred = sum(w / wsum * p for w, p in zip(weights, parts))
        return np.expm1(pred)

    # ──────────────────────────────────────────────────────────────────
    # Feature importance
    # ──────────────────────────────────────────────────────────────────
    def feature_importance(self) -> pd.DataFrame:
        if not self.lgb_models:
            return pd.DataFrame(columns=["feature", "importance"])
        imp = np.mean([m.feature_importance(importance_type="gain")
                       for m in self.lgb_models], axis=0)
        return (
            pd.DataFrame({"feature": self.FEATURE_COLS, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ──────────────────────────────────────────────────────────────────
    # Generate Kaggle submission
    # ──────────────────────────────────────────────────────────────────
    def generate_submission(self, output_path: Optional[Path] = None) -> pd.DataFrame:
        """Predict on the competition test set and save submission."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        _, test = self.load_data()
        preds = self.predict(test)
        sub = pd.DataFrame({"id": test["id"], "revenue": preds})
        if output_path:
            sub.to_csv(output_path, index=False)
            logger.info(f"Submission saved: {output_path}")
        return sub

    # ──────────────────────────────────────────────────────────────────
    # Save / Load trained model
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _default_model_path() -> Path:
        return get_settings().data_dir / "cache" / "box_office_model.pkl"

    def save(self, path: Optional[Path] = None) -> Path:
        """Save the trained predictor to disk."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        save_path = path or self._default_model_path()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({
                "lgb_models": self.lgb_models,
                "xgb_models": self.xgb_models,
                "cat_models": self.cat_models,
                "label_encoders": self.label_encoders,
                "FEATURE_COLS": self.FEATURE_COLS,
                "cv_results": self.cv_results,
                "fold_results": self.fold_results,
                "n_folds": self.n_folds,
                "seed": self.seed,
            }, f)
        logger.info(f"Model saved to {save_path}")
        return save_path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "BoxOfficePredictor":
        """Load a previously trained predictor from disk."""
        load_path = path or cls._default_model_path()
        if not load_path.exists():
            raise FileNotFoundError(f"No saved model at {load_path}")
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        bp = cls(n_folds=data["n_folds"], seed=data["seed"])
        bp.lgb_models = data["lgb_models"]
        bp.xgb_models = data["xgb_models"]
        bp.cat_models = data["cat_models"]
        bp.label_encoders = data["label_encoders"]
        bp.FEATURE_COLS = data["FEATURE_COLS"]
        bp.cv_results = data["cv_results"]
        bp.fold_results = data["fold_results"]
        bp._is_fitted = True
        logger.info(f"Model loaded from {load_path}")
        return bp
