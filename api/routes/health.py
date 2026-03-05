"""
Health check endpoints.
"""
# @author 成员 F — 前端框架 & API & 测试

from fastapi import APIRouter, Depends
from datetime import datetime

from config.settings import get_settings
from api.dependencies import get_recommendation_service
from src.models.schemas import HealthResponse

router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns the current status of the API and its components.
    """
    components = {}

    # Check recommendation service
    try:
        service = get_recommendation_service()
        if service.is_initialized:
            components["recommendation_service"] = "healthy"
            components["movie_count"] = str(service.movie_count)
        else:
            components["recommendation_service"] = "initializing"
    except Exception as e:
        components["recommendation_service"] = f"unhealthy: {str(e)}"

    # Overall status
    all_healthy = all(
        v == "healthy" or v.isdigit()
        for v in components.values()
    )

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version=settings.app_version,
        timestamp=datetime.now(),
        components=components
    )


@router.get("/ready")
async def readiness_check():
    """
    Readiness check for container orchestration.
    """
    try:
        service = get_recommendation_service()
        if service.is_initialized:
            return {"ready": True}
        return {"ready": False, "reason": "initializing"}
    except Exception as e:
        return {"ready": False, "reason": str(e)}


@router.get("/live")
async def liveness_check():
    """
    Liveness check for container orchestration.
    """
    return {"alive": True}
