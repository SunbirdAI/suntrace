"""Merge biomass rasters with Earth Engine tile statistics."""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
from rasterio.warp import Resampling
from rasterstats import zonal_stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lamwo_pipeline import compute_utm_crs

LOGGER = logging.getLogger(__name__)

DEFAULT_RASTER_PATH = Path("PATH/TO/input/aglwb.tif")
DEFAULT_GRID_PATH = Path("PATH/TO/input/lamwo_grid.geojson")
DEFAULT_TILE_STATS_PATH = Path("PATH/TO/input/Lamwo_Tile_Stats_EE.csv")
DEFAULT_OUTPUT_GEOJSON = Path("PATH/TO/output/Lamwo_Tile_Stats_EE_biomass.geojson")
DEFAULT_OUTPUT_CSV = Path("PATH/TO/output/Lamwo_Tile_Stats_EE_biomass.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster-path", type=Path, default=DEFAULT_RASTER_PATH, help="Biomass raster path")
    parser.add_argument("--grid-path", type=Path, default=DEFAULT_GRID_PATH, help="Tile grid GeoJSON")
    parser.add_argument("--tile-stats-path", type=Path, default=DEFAULT_TILE_STATS_PATH, help="Earth Engine per-tile CSV")
    parser.add_argument("--output-geojson", type=Path, default=DEFAULT_OUTPUT_GEOJSON, help="GeoJSON output path")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="CSV output path")
    parser.add_argument(
        "--raster-unit",
        choices=["Mg_per_pixel", "Mg_per_ha"],
        default="Mg_per_pixel",
        help="Interpretation of raster values for biomass",
    )
    parser.add_argument("--utm-crs", help="Optional CRS override for projected calculations (e.g. EPSG:32636)")
    return parser.parse_args()

def compute_zonal_stats(tiles_metric: gpd.GeoDataFrame, raster_path: Path) -> gpd.GeoDataFrame:
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
    stats = zonal_stats(
        tiles_metric.geometry,
        raster_path,
        stats=["sum", "mean", "count"],
        nodata=nodata,
        all_touched=False,
        geojson_out=False,
    )
    tiles_metric = tiles_metric.copy()
    tiles_metric["r_sum"] = [entry.get("sum", np.nan) for entry in stats]
    tiles_metric["r_mean"] = [entry.get("mean", np.nan) for entry in stats]
    tiles_metric["r_count"] = [entry.get("count", 0) for entry in stats]
    return tiles_metric


def aggregate_biomass(tiles_metric: gpd.GeoDataFrame, raster_unit: str) -> gpd.GeoDataFrame:
    if raster_unit == "Mg_per_pixel":
        tiles_metric["tile_total_Mg"] = tiles_metric["r_sum"]
    else:  # Mg_per_ha
        tiles_metric["tile_total_Mg"] = tiles_metric["r_mean"] * tiles_metric["area_ha"]
    return tiles_metric


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    tiles = gpd.read_file(args.grid_path).to_crs(epsg=4326)
    utm_crs = args.utm_crs or compute_utm_crs(tiles.unary_union)
    LOGGER.info("Using projected CRS %s for zonal statistics", utm_crs)

    tiles_metric = tiles.to_crs(utm_crs)
    tiles_metric["area_m2"] = tiles_metric.geometry.area
    tiles_metric["area_ha"] = tiles_metric["area_m2"] / 10000.0

    with tempfile.TemporaryDirectory() as tmpdir:
        projected_raster_path = Path(tmpdir) / "aglwb_utm.tif"
        arr = rxr.open_rasterio(args.raster_path)
        reproj = arr.rio.reproject(utm_crs, resampling=Resampling.nearest)
        reproj.rio.to_raster(projected_raster_path)

        tiles_metric = compute_zonal_stats(tiles_metric, projected_raster_path)

    tiles_metric = aggregate_biomass(tiles_metric, args.raster_unit)

    tiles_wgs84 = tiles_metric.to_crs(epsg=4326)

    tile_stats = pd.read_csv(args.tile_stats_path)
    tile_stats = tile_stats.reset_index(drop=True)

    combined = tile_stats.join(
        tiles_wgs84[["geometry", "area_m2", "area_ha", "tile_total_Mg", "r_sum", "r_mean", "r_count"]]
    )

    result_gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs=tiles_wgs84.crs)

    args.output_geojson.parent.mkdir(parents=True, exist_ok=True)
    result_gdf.to_file(args.output_geojson, driver="GeoJSON")
    result_gdf.drop(columns="geometry").to_csv(args.output_csv, index=False)
    LOGGER.info("Saved %s and %s", args.output_geojson, args.output_csv)


if __name__ == "__main__":
    main()
