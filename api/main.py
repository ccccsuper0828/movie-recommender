"""
FastAPI main application for the Movie Recommendation System.
"""
# @author 成员 F — 前端框架 & API & 测试

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from config.settings import get_settings
from config.logging_config import setup_logging
from api.routes import (
    recommendations_router,
    movies_router,
    users_router,
    health_router
)
from api.dependencies import get_recommendation_service

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Movie Recommendation API...")

    # Initialize recommendation service (warm up models)
    try:
        service = get_recommendation_service()
        logger.info(f"Loaded {service.movie_count} movies")
    except Exception as e:
        logger.error(f"Failed to initialize recommendation service: {e}")

    yield

    # Shutdown
    logger.info("Shutting down Movie Recommendation API...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="Movie Recommendation System API with multiple ML methods",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    health_router,
    tags=["Health"]
)

app.include_router(
    recommendations_router,
    prefix=f"{settings.api_prefix}/recommendations",
    tags=["Recommendations"]
)

app.include_router(
    movies_router,
    prefix=f"{settings.api_prefix}/movies",
    tags=["Movies"]
)

app.include_router(
    users_router,
    prefix=f"{settings.api_prefix}/users",
    tags=["Users"]
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "api": settings.api_prefix
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
