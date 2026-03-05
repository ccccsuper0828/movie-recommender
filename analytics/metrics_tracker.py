"""
Metrics tracking for recommendation system performance.
"""
# @author 成员 D — 票房预测 & 数据可视化

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
import numpy as np
from collections import defaultdict


@dataclass
class RecommendationEvent:
    """Single recommendation event."""
    timestamp: str
    source_movie: str
    method: str
    num_recommendations: int
    response_time_ms: float
    user_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class UserInteraction:
    """User interaction with recommendations."""
    timestamp: str
    user_id: str
    movie_id: int
    interaction_type: str  # 'view', 'click', 'rating', 'watchlist'
    value: Optional[float] = None  # For ratings
    source_recommendation: Optional[str] = None


@dataclass
class ModelMetrics:
    """Metrics for a recommendation model."""
    model_name: str
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    coverage: float = 0.0
    diversity: float = 0.0
    novelty: float = 0.0
    avg_response_time_ms: float = 0.0
    total_requests: int = 0


class MetricsTracker:
    """
    Tracks and stores recommendation system metrics.

    Provides:
    - Request logging
    - User interaction tracking
    - Model performance metrics
    - System health metrics
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize metrics tracker.

        Parameters
        ----------
        storage_path : Path, optional
            Path to store metrics data
        """
        self.storage_path = storage_path or Path("data/metrics")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory storage
        self.recommendation_events: List[RecommendationEvent] = []
        self.user_interactions: List[UserInteraction] = []
        self.model_metrics: Dict[str, ModelMetrics] = {}

        # Response time tracking
        self.response_times: Dict[str, List[float]] = defaultdict(list)

        # Load existing data
        self._load_data()

    def _load_data(self):
        """Load existing metrics data from storage."""
        events_file = self.storage_path / "recommendation_events.json"
        if events_file.exists():
            try:
                with open(events_file, 'r') as f:
                    data = json.load(f)
                    self.recommendation_events = [
                        RecommendationEvent(**e) for e in data[-1000:]  # Keep last 1000
                    ]
            except Exception:
                pass

        interactions_file = self.storage_path / "user_interactions.json"
        if interactions_file.exists():
            try:
                with open(interactions_file, 'r') as f:
                    data = json.load(f)
                    self.user_interactions = [
                        UserInteraction(**i) for i in data[-5000:]  # Keep last 5000
                    ]
            except Exception:
                pass

    def _save_data(self):
        """Save metrics data to storage."""
        events_file = self.storage_path / "recommendation_events.json"
        with open(events_file, 'w') as f:
            json.dump([e.to_dict() for e in self.recommendation_events], f)

        interactions_file = self.storage_path / "user_interactions.json"
        with open(interactions_file, 'w') as f:
            json.dump([asdict(i) for i in self.user_interactions], f)

    def log_recommendation(
        self,
        source_movie: str,
        method: str,
        num_recommendations: int,
        response_time_ms: float,
        user_id: Optional[str] = None
    ):
        """
        Log a recommendation request.

        Parameters
        ----------
        source_movie : str
            Movie used as input
        method : str
            Recommendation method used
        num_recommendations : int
            Number of recommendations returned
        response_time_ms : float
            Response time in milliseconds
        user_id : str, optional
            User who made the request
        """
        event = RecommendationEvent(
            timestamp=datetime.now().isoformat(),
            source_movie=source_movie,
            method=method,
            num_recommendations=num_recommendations,
            response_time_ms=response_time_ms,
            user_id=user_id
        )

        self.recommendation_events.append(event)
        self.response_times[method].append(response_time_ms)

        # Keep only last 1000 events in memory
        if len(self.recommendation_events) > 1000:
            self.recommendation_events = self.recommendation_events[-1000:]

        # Periodically save
        if len(self.recommendation_events) % 100 == 0:
            self._save_data()

    def log_interaction(
        self,
        user_id: str,
        movie_id: int,
        interaction_type: str,
        value: Optional[float] = None,
        source_recommendation: Optional[str] = None
    ):
        """
        Log a user interaction.

        Parameters
        ----------
        user_id : str
            User identifier
        movie_id : int
            Movie identifier
        interaction_type : str
            Type of interaction ('view', 'click', 'rating', 'watchlist')
        value : float, optional
            Value (e.g., rating score)
        source_recommendation : str, optional
            Source recommendation that led to this interaction
        """
        interaction = UserInteraction(
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            movie_id=movie_id,
            interaction_type=interaction_type,
            value=value,
            source_recommendation=source_recommendation
        )

        self.user_interactions.append(interaction)

        # Keep only last 5000 interactions in memory
        if len(self.user_interactions) > 5000:
            self.user_interactions = self.user_interactions[-5000:]

    def calculate_precision_at_k(
        self,
        recommendations: List[int],
        relevant_items: List[int],
        k: int
    ) -> float:
        """
        Calculate Precision@K.

        Parameters
        ----------
        recommendations : list
            Recommended item IDs
        relevant_items : list
            Actually relevant item IDs
        k : int
            Number of top recommendations to consider

        Returns
        -------
        float
            Precision@K score
        """
        if not recommendations or k == 0:
            return 0.0

        top_k = recommendations[:k]
        relevant_set = set(relevant_items)
        hits = sum(1 for item in top_k if item in relevant_set)

        return hits / k

    def calculate_recall_at_k(
        self,
        recommendations: List[int],
        relevant_items: List[int],
        k: int
    ) -> float:
        """
        Calculate Recall@K.

        Parameters
        ----------
        recommendations : list
            Recommended item IDs
        relevant_items : list
            Actually relevant item IDs
        k : int
            Number of top recommendations to consider

        Returns
        -------
        float
            Recall@K score
        """
        if not relevant_items:
            return 0.0

        top_k = recommendations[:k]
        relevant_set = set(relevant_items)
        hits = sum(1 for item in top_k if item in relevant_set)

        return hits / len(relevant_items)

    def calculate_ndcg_at_k(
        self,
        recommendations: List[int],
        relevant_items: List[int],
        k: int
    ) -> float:
        """
        Calculate NDCG@K (Normalized Discounted Cumulative Gain).

        Parameters
        ----------
        recommendations : list
            Recommended item IDs
        relevant_items : list
            Actually relevant item IDs
        k : int
            Number of top recommendations to consider

        Returns
        -------
        float
            NDCG@K score
        """
        if not recommendations or k == 0:
            return 0.0

        relevant_set = set(relevant_items)
        top_k = recommendations[:k]

        # DCG
        dcg = 0.0
        for i, item in enumerate(top_k):
            if item in relevant_set:
                dcg += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed

        # Ideal DCG
        ideal_hits = min(len(relevant_items), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def calculate_coverage(
        self,
        all_recommendations: List[List[int]],
        total_items: int
    ) -> float:
        """
        Calculate catalog coverage.

        Parameters
        ----------
        all_recommendations : list of lists
            All recommendation lists generated
        total_items : int
            Total number of items in catalog

        Returns
        -------
        float
            Coverage ratio
        """
        if total_items == 0:
            return 0.0

        recommended_items = set()
        for recs in all_recommendations:
            recommended_items.update(recs)

        return len(recommended_items) / total_items

    def calculate_diversity(
        self,
        recommendations: List[int],
        item_features: Dict[int, List[str]]
    ) -> float:
        """
        Calculate intra-list diversity.

        Parameters
        ----------
        recommendations : list
            Recommended item IDs
        item_features : dict
            Mapping of item ID to features (e.g., genres)

        Returns
        -------
        float
            Diversity score (0-1)
        """
        if len(recommendations) < 2:
            return 0.0

        total_distance = 0.0
        pairs = 0

        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                item_i = recommendations[i]
                item_j = recommendations[j]

                features_i = set(item_features.get(item_i, []))
                features_j = set(item_features.get(item_j, []))

                if features_i or features_j:
                    # Jaccard distance
                    intersection = len(features_i & features_j)
                    union = len(features_i | features_j)
                    similarity = intersection / union if union > 0 else 0
                    total_distance += 1 - similarity
                    pairs += 1

        return total_distance / pairs if pairs > 0 else 0.0

    def get_method_stats(self, method: str) -> Dict[str, Any]:
        """
        Get statistics for a specific method.

        Parameters
        ----------
        method : str
            Method name

        Returns
        -------
        dict
            Method statistics
        """
        method_events = [
            e for e in self.recommendation_events
            if e.method == method
        ]

        if not method_events:
            return {
                'total_requests': 0,
                'avg_response_time_ms': 0,
                'min_response_time_ms': 0,
                'max_response_time_ms': 0,
                'avg_recommendations': 0
            }

        response_times = [e.response_time_ms for e in method_events]
        num_recs = [e.num_recommendations for e in method_events]

        return {
            'total_requests': len(method_events),
            'avg_response_time_ms': np.mean(response_times),
            'min_response_time_ms': np.min(response_times),
            'max_response_time_ms': np.max(response_times),
            'avg_recommendations': np.mean(num_recs)
        }

    def get_daily_stats(self, days: int = 7) -> Dict[str, List[Dict]]:
        """
        Get daily statistics for the past N days.

        Parameters
        ----------
        days : int
            Number of days to include

        Returns
        -------
        dict
            Daily statistics by date
        """
        cutoff = datetime.now() - timedelta(days=days)

        daily_events: Dict[str, List[RecommendationEvent]] = defaultdict(list)

        for event in self.recommendation_events:
            event_date = datetime.fromisoformat(event.timestamp).date()
            if datetime.fromisoformat(event.timestamp) >= cutoff:
                daily_events[str(event_date)].append(event)

        stats = {}
        for date, events in sorted(daily_events.items()):
            response_times = [e.response_time_ms for e in events]
            stats[date] = {
                'total_requests': len(events),
                'unique_users': len(set(e.user_id for e in events if e.user_id)),
                'avg_response_time_ms': np.mean(response_times) if response_times else 0,
                'methods': dict(defaultdict(int, {e.method: 0 for e in events}))
            }
            for event in events:
                stats[date]['methods'][event.method] = \
                    stats[date]['methods'].get(event.method, 0) + 1

        return stats

    def get_popular_source_movies(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most popular source movies for recommendations.

        Parameters
        ----------
        limit : int
            Number of movies to return

        Returns
        -------
        list
            Popular source movies with counts
        """
        movie_counts: Dict[str, int] = defaultdict(int)

        for event in self.recommendation_events:
            movie_counts[event.source_movie] += 1

        sorted_movies = sorted(
            movie_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]

        return [
            {'movie': movie, 'count': count}
            for movie, count in sorted_movies
        ]

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get system health metrics.

        Returns
        -------
        dict
            System health status
        """
        recent_events = [
            e for e in self.recommendation_events
            if datetime.fromisoformat(e.timestamp) > datetime.now() - timedelta(hours=1)
        ]

        if not recent_events:
            return {
                'status': 'unknown',
                'requests_last_hour': 0,
                'avg_response_time_ms': 0,
                'error_rate': 0
            }

        response_times = [e.response_time_ms for e in recent_events]
        avg_response_time = np.mean(response_times)

        # Determine health status
        if avg_response_time < 100:
            status = 'healthy'
        elif avg_response_time < 500:
            status = 'degraded'
        else:
            status = 'critical'

        return {
            'status': status,
            'requests_last_hour': len(recent_events),
            'avg_response_time_ms': avg_response_time,
            'p95_response_time_ms': np.percentile(response_times, 95),
            'p99_response_time_ms': np.percentile(response_times, 99)
        }

    def export_metrics(self, filepath: Path) -> None:
        """
        Export all metrics to a JSON file.

        Parameters
        ----------
        filepath : Path
            Output file path
        """
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'recommendation_events': [e.to_dict() for e in self.recommendation_events],
            'user_interactions': [asdict(i) for i in self.user_interactions],
            'method_stats': {
                method: self.get_method_stats(method)
                for method in ['content', 'metadata', 'cf', 'hybrid']
            },
            'system_health': self.get_system_health()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
