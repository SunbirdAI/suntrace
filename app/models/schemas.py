"""
Pydantic models for request/response validation
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for geospatial queries"""

    query: str = Field(
        ...,
        description="Natural language query about the geospatial data",
        example="How many buildings are in the selected area?",
        min_length=1,
        max_length=1000,
    )
    polygon: Optional[List[List[float]]] = Field(
        None,
        description="Optional polygon coordinates [[lng, lat], [lng, lat], ...] for spatial filtering",
        example=[
            [-1.2345, 36.1234],
            [-1.2300, 36.1234],
            [-1.2300, 36.1280],
            [-1.2345, 36.1280],
            [-1.2345, 36.1234],
        ],
    )

    class Config:
         json_schema_extra = {
            "example": {
                "query": "What is the population density in this region?",
                "polygon": [
                    [-1.2345, 36.1234],
                    [-1.2300, 36.1234],
                    [-1.2300, 36.1280],
                    [-1.2345, 36.1280],
                    [-1.2345, 36.1234],
                ],
            }
        }


class QueryResponse(BaseModel):
    """Response model for query results"""

    response: str = Field(..., description="AI-generated response to the query")
    query_id: Optional[str] = Field(
        None, description="Unique identifier for the query (for tracking)"
    )
    processing_time: Optional[float] = Field(
        None, description="Time taken to process the query in seconds"
    )

    class Config:
         json_schema_extra = {
            "example": {
                "response": "Based on the selected area, there are approximately 1,247 buildings with an estimated population density of 3,200 people per square kilometer.",
                "query_id": "q_12345",
                "processing_time": 2.34,
            }
        }


class Coordinates(BaseModel):
    """Model for coordinate pairs"""

    lat: float = Field(..., description="Latitude")
    lng: float = Field(..., description="Longitude")


class MapBounds(BaseModel):
    """Model for map bounds"""

    min_lat: float = Field(..., description="Minimum latitude")
    min_lng: float = Field(..., description="Minimum longitude")
    max_lat: float = Field(..., description="Maximum latitude")
    max_lng: float = Field(..., description="Maximum longitude")


class GeoJSONFeature(BaseModel):
    """Basic GeoJSON feature model"""

    type: str = Field(default="Feature")
    geometry: Dict[str, Any] = Field(..., description="GeoJSON geometry")
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Feature properties"
    )


class GeoJSONFeatureCollection(BaseModel):
    """GeoJSON feature collection model"""

    type: str = Field(default="FeatureCollection")
    features: List[GeoJSONFeature] = Field(default_factory=list)


class MapLayersResponse(BaseModel):
    """Response model for map layers data"""

    center: List[float] = Field(..., description="Map center coordinates [lat, lng]")
    bounds: List[List[float]] = Field(
        ..., description="Map bounds [[minLat, minLng], [maxLat, maxLng]]"
    )
    # candidate_minigrids: Dict[str, Any]
    # buildings: Dict[str, Any]
    # metadata: Optional[Dict[str, Any]] = None
    candidate_minigrids: GeoJSONFeatureCollection = Field(
        ..., description="GeoJSON data for candidate minigrids"
    )
    buildings: GeoJSONFeatureCollection = Field(
        ..., description="GeoJSON data for buildings (sampled for performance)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata about the map layers"
    )

    class Config:
         json_schema_extra = {
            "example": {
                "center": {"lat": 36.1234, "lng": -1.2345},
                "bounds": {
                    "min_lat": 36.1000,
                    "min_lng": -1.2500,
                    "max_lat": 36.1500,
                    "max_lng": -1.2000,
                },
                "candidate_minigrids": {"type": "FeatureCollection", "features": []},
                "buildings": {"type": "FeatureCollection", "features": []},
                "metadata": {
                    "total_buildings": 5432,
                    "sampled_buildings": 2000,
                    "total_minigrids": 45,
                },
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""

    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(
        None, description="Error code for programmatic handling"
    )
    timestamp: Optional[str] = Field(None, description="Error timestamp")

    class Config:
         json_schema_extra = {
            "example": {
                "error": "Failed to process geospatial data",
                "error_code": "GEOSPATIAL_ERROR",
                "timestamp": "2025-07-24T10:30:00Z",
            }
        }


class HealthResponse(BaseModel):
    """Health check response model"""

    status: str = Field(..., description="Health status")
    version: str = Field(..., description="Application version")
    timestamp: str = Field(..., description="Health check timestamp")
    services: Optional[Dict[str, str]] = Field(
        None, description="Status of dependent services"
    )

    class Config:
         json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2025-07-24T10:30:00Z",
                "services": {
                    "database": "healthy",
                    "openai": "healthy",
                    "geospatial_analyzer": "healthy",
                },
            }
        }
