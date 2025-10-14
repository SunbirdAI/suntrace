"""Utility helpers for the Lamwo data-gathering pipeline."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import box, mapping
from shapely.geometry.base import BaseGeometry

LOGGER = logging.getLogger(__name__)


def _import_ee():
    """Import Google Earth Engine lazily so non-EE scripts still load."""
    try:
        import ee  # type: ignore
        from ee.ee_exception import EEException  # type: ignore
    except ImportError as exc:  # pragma: no cover - guard for optional dep
        raise ImportError(
            "The earthengine-api package is required for this operation."
        ) from exc
    return ee, EEException


def init_ee(project_id: Optional[str] = None, authenticate_if_needed: bool = True) -> None:
    """Initialise the Earth Engine client, authenticating if credentials are absent."""
    ee, EEException = _import_ee()
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
    except EEException as exc:
        if not authenticate_if_needed:
            raise
        LOGGER.info("EE initialization failed, attempting authentication: %s", exc)
        ee.Authenticate()
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()


def geom_to_ee(geometry: BaseGeometry):
    """Convert a Shapely geometry into an Earth Engine geometry."""
    ee, _ = _import_ee()
    geojson = mapping(geometry)
    geom_type = geojson.get("type")
    coordinates = geojson.get("coordinates")
    if geom_type == "MultiPolygon":
        return ee.Geometry.MultiPolygon(coordinates)
    if geom_type == "Polygon":
        return ee.Geometry.MultiPolygon([coordinates])
    raise ValueError(f"Unsupported geometry type for EE conversion: {geom_type}")


def compute_utm_crs(geometry: BaseGeometry) -> str:
    """Derive an appropriate UTM CRS for a lon/lat geometry."""
    centroid = geometry.centroid
    zone = int((centroid.x + 180) // 6) + 1
    epsg = 32600 + zone if centroid.y >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def create_tile_grid(
    aoi_path: Path,
    *,
    cell_size_m: float = 1000,
    output_path: Optional[Path] = None,
    output_driver: str = "GeoJSON",
) -> gpd.GeoDataFrame:
    """Create a square grid covering the AOI and optionally persist it."""
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive")

    aoi = gpd.read_file(aoi_path)
    if aoi.empty:
        raise ValueError(f"AOI at {aoi_path} is empty")
    if aoi.crs is None:
        raise ValueError("AOI must have a defined CRS")

    aoi_wgs84 = aoi.to_crs(epsg=4326)
    union_geom = aoi_wgs84.unary_union

    utm_crs = compute_utm_crs(union_geom)
    aoi_metric = aoi_wgs84.to_crs(utm_crs)
    union_metric = aoi_metric.unary_union
    minx, miny, maxx, maxy = union_metric.bounds

    cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            candidate = box(x, y, x + cell_size_m, y + cell_size_m)
            intersection = candidate.intersection(union_metric)
            if not intersection.is_empty:
                cells.append(intersection)
            y += cell_size_m
        x += cell_size_m

    grid = gpd.GeoDataFrame({"geometry": cells}, crs=utm_crs).to_crs(epsg=4326)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid.to_file(output_path, driver=output_driver)
        LOGGER.info("Wrote grid to %s", output_path)

    return grid


def geodf_to_feature_collection(gdf: gpd.GeoDataFrame):
    """Convert a GeoDataFrame into an EE FeatureCollection."""
    ee, _ = _import_ee()
    return ee.FeatureCollection(json.loads(gdf.to_json()))


def ee_bytes_to_img(pixel_data) -> np.ndarray:
    """Convert a structured array returned by EE into a standard NumPy array."""
    if isinstance(pixel_data, np.ndarray) and pixel_data.dtype.names:
        return np.stack([pixel_data[name] for name in pixel_data.dtype.names], axis=-1)
    raise ValueError("Expected a structured NumPy array with named bands")


def reshape_monthly_stack(array: np.ndarray, band_count: int) -> np.ndarray:
    """Reshape stacked bands into (months, height, width, bands)."""
    if array.ndim != 3:
        raise ValueError("Array must be 3-D (height, width, bands)")
    height, width, stacked_bands = array.shape
    if stacked_bands % band_count != 0:
        raise ValueError("Stacked band count is not divisible by band_count")
    months = stacked_bands // band_count
    return array.reshape(height, width, months, band_count).transpose(2, 0, 1, 3)
