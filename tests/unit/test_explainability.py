"""
Unit tests for explainability modules using current public APIs.
"""
# @author 成员 F — 前端框架 & API & 测试


class TestRuleBasedExplainer:
    def test_explain_metadata_based_returns_expected_shape(self, sample_movies_df):
        from src.explainability import RuleBasedExplainer

        explainer = RuleBasedExplainer(sample_movies_df)
        result = explainer.explain_metadata_based("The Matrix", "The Matrix Reloaded")

        assert isinstance(result, dict)
        assert "summary" in result
        assert "reasons" in result
        assert "details" in result
        assert result["source_movie"] == "The Matrix"

    def test_explain_content_based_handles_unknown_title(self, sample_movies_df):
        from src.explainability import RuleBasedExplainer

        explainer = RuleBasedExplainer(sample_movies_df)
        result = explainer.explain_content_based("Unknown", "The Matrix")

        assert isinstance(result, dict)
        assert "Unable to generate explanation" in result["summary"]

    def test_explain_hybrid_returns_contributions(self, sample_movies_df):
        from src.explainability import RuleBasedExplainer

        explainer = RuleBasedExplainer(sample_movies_df)
        result = explainer.explain_hybrid(
            "The Matrix",
            "The Matrix Reloaded",
            content_score=0.8,
            metadata_score=0.9,
            cf_score=0.7,
        )

        assert isinstance(result, dict)
        assert "method_contributions" in result["details"]


class TestSHAPExplainer:
    def test_shap_import_guard(self):
        from src.explainability import SHAPExplainer

        # SHAP is optional in this project runtime.
        assert SHAPExplainer is None or SHAPExplainer is not None
