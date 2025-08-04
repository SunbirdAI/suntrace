"""
Query-related API routes
"""

from datetime import datetime

from fastapi import APIRouter, Depends

from app.api.deps import get_geospatial_service
from app.core.config import get_settings
from app.models.schemas import (
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.geospatial import GeospatialService

router = APIRouter()
settings = get_settings()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Process Geospatial Query",
    description="Process natural language queries about geospatial data, optionally with polygon regions",
    responses={
        200: {
            "description": "Successfully processed the query",
            "model": QueryResponse,
        },
        422: {
            "description": "Data processing error - invalid input",
            "model": ErrorResponse,
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse,
        },
    },
)
async def process_query(
    request_data: QueryRequest,
    service: GeospatialService = Depends(get_geospatial_service),
):
    """
    Handle LLM queries and process drawn polygons.

    This endpoint processes natural language queries about the geospatial data.
    If a polygon is provided, it will be included in the analysis context.

    Args:
        request_data: The query request containing query and optional polygon

    Returns:
        QueryResponse: AI-generated response to the query with metadata

    Example:
        ```json
        {
            "query": "How many buildings are in this area?",
            "polygon": [
                [-1.2345, 36.1234],
                [-1.2300, 36.1234],
                [-1.2300, 36.1280],
                [-1.2345, 36.1280],
                [-1.2345, 36.1234]
            ]
        }
        ```
    """
    return service.process_query(request_data)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the application and its dependencies",
    include_in_schema=True,
)
async def health_check():
    """Simple health check endpoint with service status"""
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        timestamp=datetime.utcnow().isoformat() + "Z",
        services={
            "geospatial_analyzer": "healthy",
            "openai": "healthy" if settings.OPENAI_API_KEY else "not_configured",
        },
    )
