# CineMatch API Documentation

## Overview

Current API routes are implemented by FastAPI and split into:

- Health routes without prefix: `/health`, `/ready`, `/live`
- Versioned application routes under: `/api/v1`

Base URL: `http://localhost:8000`

## Health Endpoints

### GET `/health`
Returns service health summary.

### GET `/ready`
Returns readiness status.

### GET `/live`
Returns liveness status.

## Recommendation Endpoints (`/api/v1/recommendations`)

### GET `/{movie_title}`
Get recommendations for a movie title.

Query params:
- `top_n` (int, 1-50, default 10)
- `method` (`content|metadata|cf|hybrid`, default `hybrid`)

Example:
```bash
curl "http://localhost:8000/api/v1/recommendations/The%20Matrix?top_n=5&method=hybrid"
```

### POST `/`
Body-based recommendation request.

Request example:
```json
{
  "title": "The Matrix",
  "top_n": 5,
    "method": "hybrid",
  "weights": {
    "content": 0.3,
    "metadata": 0.4,
    "cf": 0.3
        }
}
```

### POST `/batch`
Batch recommendations.

Request example:
```json
{
  "titles": ["The Matrix", "Inception"],
  "top_n": 5,
    "method": "hybrid"
}
```

### GET `/compare/{movie_title}`
Compare outputs from all methods.

### POST `/explain`
Explain one recommendation.

Request example:
```json
{
  "source_title": "The Matrix",
  "target_title": "The Matrix Reloaded",
  "method": "metadata"
}
```

### GET `/similarity/{movie1}/{movie2}`
Get similarity scores for a pair of movies.

## Movie Endpoints (`/api/v1/movies`)

### GET `/`
Search/list movies with pagination.

Query params:
- `query`, `genres`, `year_min`, `year_max`, `rating_min`, `rating_max`, `director`
- `page` (default 1), `page_size` (default 20)

### GET `/popular`
Get popular movies.

### GET `/top-rated`
Get top rated movies.

### GET `/suggestions`
Autocomplete suggestions (`q`, `limit`).

### GET `/genres`
List available genres.

### GET `/years`
List available years.

### GET `/by-genre/{genre}`
Get movies by genre.

### GET `/by-director/{director}`
Get movies by director.

### GET `/{movie_title}`
Get a movie by exact title.

### GET `/analytics/overview`
Get analytics overview.

### GET `/analytics/genres`
Get genre analytics.

### GET `/analytics/directors`
Get director analytics.

## User Endpoints (`/api/v1/users`)

### POST `/`
Create user.

### GET `/`
List users.

### GET `/{user_id}`
Get user profile.

### PUT `/{user_id}/preferences`
Update preferences.

### POST `/{user_id}/ratings`
Add rating.

### GET `/{user_id}/ratings`
List ratings.

### POST `/{user_id}/watchlist/{movie_id}`
Add watchlist item.

### DELETE `/{user_id}/watchlist/{movie_id}`
Remove watchlist item.

### GET `/{user_id}/watchlist`
Get watchlist.

### GET `/{user_id}/recommendations`
Get personalized recommendations.

### DELETE `/{user_id}`
Delete user.

## Notes

- Responses are Pydantic-validated for typed endpoints.
- Some endpoints return direct dictionaries; rely on current OpenAPI docs for exact response shapes.
- Interactive docs: `http://localhost:8000/docs`
