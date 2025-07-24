"""
Map-related API routes
"""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.api.deps import get_geospatial_service, get_templates
from app.models.schemas import ErrorResponse, MapLayersResponse
from app.services.geospatial import GeospatialService

router = APIRouter()


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request, templates: Jinja2Templates = Depends(get_templates)):
    """
    Serve the main application page.

    Returns the HTML interface for the geospatial analyzer.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@router.get(
    "/get_map_layers",
    response_model=MapLayersResponse,
    summary="Get Map Layers",
    description="Retrieve GeoJSON data for map visualization including candidate minigrids and buildings",
    responses={
        200: {
            "description": "Successfully retrieved map layers",
            "model": MapLayersResponse,
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse,
        },
    },
)
async def get_map_layers(service: GeospatialService = Depends(get_geospatial_service)):
    """
    Return GeoJSON data for the map layers.

    This endpoint provides:
    - Map center coordinates and bounds
    - Candidate minigrids as GeoJSON
    - Building locations (sampled for performance)

    Returns:
        MapLayersResponse: Map data including center, bounds, and GeoJSON layers
    """
    return service.get_map_layers()
