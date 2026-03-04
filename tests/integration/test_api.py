"""
Integration tests for FastAPI endpoints using current route structure.
"""


class TestHealthEndpoints:
    def test_health_check(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_readiness_check(self, test_client):
        response = test_client.get("/ready")
        assert response.status_code == 200
        assert "ready" in response.json()

    def test_liveness_check(self, test_client):
        response = test_client.get("/live")
        assert response.status_code == 200
        assert response.json()["alive"] is True


class TestRecommendationEndpoints:
    def test_get_recommendations_by_title(self, test_client):
        response = test_client.get("/api/v1/recommendations/The Matrix?top_n=5&method=hybrid")
        assert response.status_code in [200, 404]

    def test_post_recommendations(self, test_client):
        payload = {"title": "The Matrix", "top_n": 5, "method": "hybrid"}
        response = test_client.post("/api/v1/recommendations/", json=payload)
        assert response.status_code in [200, 404]

    def test_batch_recommendations(self, test_client):
        payload = {"titles": ["The Matrix", "Inception"], "top_n": 3, "method": "hybrid"}
        response = test_client.post("/api/v1/recommendations/batch", json=payload)
        assert response.status_code == 200


class TestMovieEndpoints:
    def test_list_movies(self, test_client):
        response = test_client.get("/api/v1/movies/?query=Matrix&page=1&page_size=5")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    def test_get_suggestions(self, test_client):
        response = test_client.get("/api/v1/movies/suggestions?q=Mat&limit=5")
        assert response.status_code == 200
        assert "suggestions" in response.json()

    def test_get_movie_by_title(self, test_client):
        response = test_client.get("/api/v1/movies/The Matrix")
        assert response.status_code in [200, 404]


class TestUserEndpoints:
    def test_create_user(self, test_client):
        payload = {"username": "testuser_api", "email": "test@api.com"}
        response = test_client.post("/api/v1/users/", json=payload)
        assert response.status_code in [200, 400]

    def test_list_users(self, test_client):
        response = test_client.get("/api/v1/users/")
        assert response.status_code == 200
        data = response.json()
        assert "users" in data
