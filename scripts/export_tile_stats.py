"""Export per-tile aggregate statistics from Earth Engine."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import geopandas as gpd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ee  # type: ignore

from utils.lamwo_pipeline import geom_to_ee, geodf_to_feature_collection, init_ee

LOGGER = logging.getLogger(__name__)

DEFAULT_PROJECT_ID = "your-ee-project-id"
DEFAULT_AOI_PATH = Path("PATH/TO/lamwo_aoi.gpkg")
DEFAULT_TILES_PATH = Path("PATH/TO/input/lamwo_grid.geojson")
DEFAULT_DRIVE_FOLDER = "ee_exports"
DEFAULT_FILE_PREFIX = "Lamwo_Tile_Stats_EE"
DEFAULT_SELECTORS = [
    "ndvi_mean",
    "ndvi_med",
    "ndvi_std",
    "evi_med",
    "elev_mean",
    "slope_mean",
    "par_mean",
    "rain_total_mm",
    "rain_mean_mm_day",
    "cloud_free_days",
    "bldg_count",
    "bldg_area",
    "bldg_h_max",
    "system:index",
]

DATASETS = {
    "S2_SR": "COPERNICUS/S2_SR_HARMONIZED",
    "SRTM": "USGS/SRTMGL1_003",
    "VIIRS": "NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG",
    "MODIS_PAR": "MODIS/062/MCD18C2",
    "OPEN_BUILDINGS": "GOOGLE/Research/open-buildings-temporal/v1",
    "CHIRPS": "UCSB-CHG/CHIRPS/DAILY",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID, help="Earth Engine project ID (update placeholder)")
    parser.add_argument("--aoi-path", type=Path, default=DEFAULT_AOI_PATH, help="AOI footprint (GeoPackage/GeoJSON)")
    parser.add_argument("--tiles-path", type=Path, default=DEFAULT_TILES_PATH, help="Tile grid GeoJSON")
    parser.add_argument("--year", type=int, default=2023, help="Year to aggregate over")
    parser.add_argument("--drive-folder", default=DEFAULT_DRIVE_FOLDER, help="Drive folder for EE export task")
    parser.add_argument("--file-prefix", default=DEFAULT_FILE_PREFIX, help="Prefix for exported file")
    parser.add_argument(
        "--download-path",
        type=Path,
        help="Optional local path to download GeoJSON instead of creating a Drive task",
    )
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=DEFAULT_SELECTORS,
        help="Feature properties to include in the export",
    )
    parser.add_argument("--cloud-threshold", type=float, default=20.0, help="Cloud threshold for Sentinel-2 filtering")
    return parser.parse_args()


def build_collections(aoi_ee, year: int, cloud_threshold: float):
    s2 = (
        ee.ImageCollection(DATASETS["S2_SR"])
        .filterBounds(aoi_ee)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
    )

    ndvi_collection = s2.map(lambda img: img.normalizedDifference(["B8", "B4"]).rename("NDVI"))
    evi_collection = s2.map(
        lambda img: img.expression(
            "2.5*((NIR - RED)/(NIR + 6*RED - 7.5*BLUE + 1))",
            {"NIR": img.select("B8"), "RED": img.select("B4"), "BLUE": img.select("B2")},
        ).rename("EVI")
    )

    dem = ee.Image(DATASETS["SRTM"]).select("elevation")
    slope = ee.Terrain.slope(dem)

    par = (
        ee.ImageCollection(DATASETS["MODIS_PAR"])
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .select("GMT_1200_PAR")
        .mean()
    )

    chirps = (
        ee.ImageCollection(DATASETS["CHIRPS"])
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .select("precipitation")
    )

    buildings = (
        ee.ImageCollection(DATASETS["OPEN_BUILDINGS"])
        .filterBounds(aoi_ee)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
    ).mosaic()

    return {
        "ndvi_collection": ndvi_collection,
        "evi_collection": evi_collection,
        "dem": dem,
        "slope": slope,
        "par": par,
        "chirps": chirps,
        "buildings": buildings,
        "s2": s2,
    }


def compute_stats_fn(collections, year: int, cloud_threshold: float):
    ndvi_collection = collections["ndvi_collection"]
    evi_collection = collections["evi_collection"]
    dem = collections["dem"]
    slope = collections["slope"]
    par = collections["par"]
    chirps = collections["chirps"]
    buildings = collections["buildings"]
    s2 = collections["s2"]

    def _compute(tile):
        geom = tile.geometry()
        props = {}

        ndvi_mean = ndvi_collection.mean().reduceRegion(
            ee.Reducer.mean(), geometry=geom, scale=30, bestEffort=True
        )
        props["ndvi_mean"] = ndvi_mean.get("NDVI", ee.Number(-9999))

        ndvi_med = ndvi_collection.median().reduceRegion(
            ee.Reducer.median(), geometry=geom, scale=30, bestEffort=True
        )
        props["ndvi_med"] = ndvi_med.get("NDVI", ee.Number(-9999))

        ndvi_std = ndvi_collection.reduce(ee.Reducer.stdDev()).reduceRegion(
            ee.Reducer.mean(), geometry=geom, scale=30, bestEffort=True
        )
        props["ndvi_std"] = ndvi_std.get("NDVI_stdDev", ee.Number(-9999))

        evi_med = evi_collection.median().reduceRegion(
            ee.Reducer.median(), geometry=geom, scale=30, bestEffort=True
        )
        props["evi_med"] = evi_med.get("EVI", ee.Number(-9999))

        elev_mean = dem.reduceRegion(
            ee.Reducer.mean(), geometry=geom, scale=30, bestEffort=True
        )
        props["elev_mean"] = elev_mean.get("elevation", ee.Number(-9999))

        slope_mean = slope.reduceRegion(
            ee.Reducer.mean(), geometry=geom, scale=30, bestEffort=True
        )
        props["slope_mean"] = slope_mean.get("slope", ee.Number(-9999))

        par_mean = par.reduceRegion(
            ee.Reducer.mean(), geometry=geom, scale=1000, bestEffort=True
        )
        props["par_mean"] = par_mean.get("GMT_1200_PAR", ee.Number(-9999))

        rain_sum = chirps.sum().reduceRegion(
            ee.Reducer.sum(), geometry=geom, scale=5566, bestEffort=True
        )
        props["rain_total_mm"] = rain_sum.get("precipitation", ee.Number(0))

        rain_mean = chirps.mean().reduceRegion(
            ee.Reducer.mean(), geometry=geom, scale=5566, bestEffort=True
        )
        props["rain_mean_mm_day"] = rain_mean.get("precipitation", ee.Number(0))

        cf_days = s2.filterBounds(geom).filterDate(
            f"{year}-01-01", f"{year}-12-31"
        ).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold)).size()
        props["cloud_free_days"] = cf_days

        bldg_count = buildings.reduceRegion(
            ee.Reducer.sum(), geometry=geom, scale=1, bestEffort=True
        )
        props["bldg_count"] = bldg_count.get("building_fractional_count", ee.Number(0))

        bldg_presence = buildings.select("building_presence").gt(0.34).reduceRegion(
            ee.Reducer.sum(), geometry=geom, scale=1, bestEffort=True
        )
        props["bldg_area"] = bldg_presence.get("building_presence", ee.Number(0))

        bldg_height = buildings.select("building_height").reduceRegion(
            ee.Reducer.max(), geometry=geom, scale=1, bestEffort=True
        )
        props["bldg_h_max"] = bldg_height.get("building_height", ee.Number(0))

        return tile.set(props)

    return _compute


def start_export(stats_table, selectors: List[str], args: argparse.Namespace) -> None:
    if args.download_path:
        params = {"filetype": "GeoJSON", "selectors": selectors}
        url = stats_table.getDownloadURL(**params)
        response = requests.get(url, timeout=600)
        response.raise_for_status()
        args.download_path.parent.mkdir(parents=True, exist_ok=True)
        args.download_path.write_bytes(response.content)
        LOGGER.info("Downloaded stats to %s", args.download_path)
        return

    task = ee.batch.Export.table.toDrive(
        collection=stats_table,
        description=args.file_prefix,
        folder=args.drive_folder,
        fileNamePrefix=args.file_prefix,
        fileFormat="CSV",
        selectors=selectors,
    )
    task.start()
    LOGGER.info("Started EE Drive export task %s", task.id)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    init_ee(args.project_id)

    aoi = gpd.read_file(args.aoi_path)
    aoi_ee = geom_to_ee(aoi.to_crs(epsg=4326).unary_union)

    tiles = gpd.read_file(args.tiles_path).to_crs(epsg=4326)
    tiles_fc = geodf_to_feature_collection(tiles)

    collections = build_collections(aoi_ee, args.year, args.cloud_threshold)
    stats_table = tiles_fc.map(compute_stats_fn(collections, args.year, args.cloud_threshold))

    start_export(stats_table, args.selectors, args)


if __name__ == "__main__":
    main()
