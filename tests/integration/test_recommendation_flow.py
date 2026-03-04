"""
Integration tests for core recommendation flow using current interfaces.
"""


class TestEndToEndRecommendationFlow:
    def test_hybrid_recommendation_flow(self, sample_movies_df):
        from src.core import HybridRecommender

        recommender = HybridRecommender()
        recommender.fit(sample_movies_df)
        result = recommender.recommend("The Matrix", top_n=5)

        assert result is not None
        assert len(result) > 0
        assert "hybrid_score" in result.columns

    def test_compare_methods(self, sample_movies_df):
        from src.core import HybridRecommender

        recommender = HybridRecommender()
        recommender.fit(sample_movies_df)
        comparisons = recommender.compare_methods("The Matrix", top_n=3)

        assert comparisons is not None
        assert "hybrid" in comparisons
        assert "content_based" in comparisons


class TestServiceIntegration:
    def test_recommendation_service_returns_structured_payload(self):
        from src.services import RecommendationService

        service = RecommendationService(auto_init=True)
        result = service.get_recommendations("The Matrix", top_n=5, method="hybrid")

        assert isinstance(result, dict)
        assert "source_movie" in result
        assert "recommendations" in result

    def test_explanation_generation(self):
        from src.services import RecommendationService

        service = RecommendationService(auto_init=True)
        explanation = service.explain_recommendation(
            source_title="The Matrix",
            target_title="The Matrix Reloaded",
            method="metadata",
        )

        assert explanation is not None
        assert "summary" in explanation
