"""
Geospatial analysis service
"""

import time

from uuid import uuid4


from shapely.geometry import Polygon
from shapely.wkt import dumps as wkt_dumps
from utils.factory import create_geospatial_analyzer
from utils.langraph_function_caller import ask_with_functions
from ..core.logger import logger
from ..core.config import get_settings
from ..core.exceptions import (
    DataProcessingError,
    GeospatialAnalysisError,
    LLMQueryError,
)

from ..models.schemas import (
    Coordinates,
    GeoJSONFeatureCollection,
    MapBounds,
    MapLayersResponse,
    QueryRequest,
    QueryResponse,
)


settings = get_settings()


class GeospatialService:
    """Service for handling geospatial operations"""

    def __init__(self):
        """Initialize the geospatial service"""
        try:
            self.analyzer = create_geospatial_analyzer()
            logger.info("Geospatial analyzer initialized successfully.")

        except Exception as e:
            logger.error("Failed to initialize geospatial analyzer: %s", e)

            raise GeospatialAnalysisError(
                f"Failed to initialize geospatial analyzer: {str(e)}"
            ) from e

    def get_map_layers(self) -> MapLayersResponse:
        """
        Get map layers data including bounds, center, and GeoJSON features.

        Returns:
            MapLayersResponse: Complete map data with layers

        Raises:
            GeospatialAnalysisError: If there's an error processing the data
        """
        try:
            # Get bounds for the map from the plain tiles
            logger.info("Fetching map layers data.")
            bounds = self.analyzer.get_layer_bounds("tiles")

            # Calculate center coordinates
            center = Coordinates(
                lat=(bounds[1] + bounds[3]) / 2, lng=(bounds[0] + bounds[2]) / 2
            )

            # Create bounds object
            map_bounds = MapBounds(
                min_lat=bounds[1],
                min_lng=bounds[0],
                max_lat=bounds[3],
                max_lng=bounds[2],
            )

            # Convert minigrids to GeoJSON
            candidate_minigrids_geo = self.analyzer.get_layer_geojson("candidate_minigrids")

            # Get a sample of buildings to avoid performance issues
            total_buildings = self.analyzer.get_layer_count("buildings")
            buildings_geo = self.analyzer.get_layer_geojson(
                "buildings", sample=settings.BUILDING_SAMPLE_LIMIT
            )
            sampled_buildings = len(buildings_geo.get("features", []))

            # Create metadata
            metadata = {
                "total_buildings": total_buildings,
                "sampled_buildings": sampled_buildings,
                "total_minigrids": self.analyzer.get_layer_count("candidate_minigrids"),
                "coordinate_system": self.analyzer.target_geographic_crs,
            }
            logger.info("Map layers data fetched successfully.")

            return MapLayersResponse(
                center=[center.lat, center.lng],
                bounds=[
                    [map_bounds.min_lat, map_bounds.min_lng],
                    [map_bounds.max_lat, map_bounds.max_lng],
                ],
                candidate_minigrids=GeoJSONFeatureCollection(**candidate_minigrids_geo),
                buildings=GeoJSONFeatureCollection(**buildings_geo),
                metadata=metadata,
            )

        except Exception as e:
            logger.error("Error getting map layers: %s", e)
            raise GeospatialAnalysisError(f"Failed to get map layers: {str(e)}") from e

    def process_query(self, request_data: QueryRequest) -> QueryResponse:
        """
        Process a geospatial query with optional polygon filtering.

        Args:
            request_data: The query request data

        Returns:
            QueryResponse: The processed query response

        Raises:
            DataProcessingError: If there's an error processing the polygon
            LLMQueryError: If there's an error with the LLM query
        """
        logger.info("Processing query: %s...", request_data.query[:50])
       
        start_time = time.time()
        query_id = str(uuid4())[:8]

        try:
            user_query = request_data.query

            # Process the drawn polygon if provided
            if request_data.polygon:
                try:
                    # Validate polygon coordinates
                    if len(request_data.polygon) < 3:
                        raise DataProcessingError(
                            "Polygon must have at least 3 coordinates"
                        )

                    # Create a shapely polygon from the coordinates
                    polygon = Polygon(request_data.polygon)

                    # Validate polygon
                    if not polygon.is_valid:
                        raise DataProcessingError("Invalid polygon geometry")

                    # Convert the polygon to WKT format
                    wkt_polygon = wkt_dumps(polygon)

                    # Append the WKT polygon to the user query
                    user_query = f"{user_query} Here's the region WKT: {wkt_polygon}"

                except Exception as e:
                    raise DataProcessingError(
                        f"Failed to process polygon: {str(e)}"
                    ) from e

            # Call the LLM function to process the query
            try:
                response_text = ask_with_functions(user_query, self.analyzer)
            except Exception as e:
                raise LLMQueryError(f"Failed to process LLM query: {str(e)}") from e

            processing_time = time.time() - start_time
            logger.info("Query processed in %.2f seconds.", processing_time)
        

            return QueryResponse(
                response=response_text,
                query_id=f"q_{query_id}",
                processing_time=round(processing_time, 2),
            )

        except (DataProcessingError, LLMQueryError):
            raise
        except Exception as e:
            raise LLMQueryError(f"Unexpected error processing query: {str(e)}") from e
