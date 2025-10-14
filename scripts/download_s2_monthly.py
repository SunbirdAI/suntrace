"""Download monthly Sentinel-2 composites per tile using Earth Engine."""
from __future__ import annotations

import argparse
import io
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Tuple

import geopandas as gpd
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ee  # type: ignore

from utils.lamwo_pipeline import (
    ee_bytes_to_img,
    geom_to_ee,
    init_ee,
    reshape_monthly_stack,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_PROJECT_ID = "your-ee-project-id"
DEFAULT_GRID_PATH = Path("PATH/TO/input/lamwo_grid.geojson")
DEFAULT_OUTPUT_DIR = Path("PATH/TO/output/s2_tiles")
S2_DATASET_ID = "COPERNICUS/S2_SR_HARMONIZED"
DEFAULT_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID, help="Earth Engine project ID (update placeholder)")
    parser.add_argument("--grid-path", type=Path, default=DEFAULT_GRID_PATH, help="Path to the tile grid GeoJSON")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for tile numpy stacks")
    parser.add_argument("--year", type=int, default=2024, help="Calendar year to composite")
    parser.add_argument("--dataset-id", default=S2_DATASET_ID, help="Earth Engine dataset to sample (default: Sentinel-2 SR)")
    parser.add_argument("--bands", nargs="+", default=DEFAULT_BANDS, help="Band list to include per month")
    parser.add_argument("--max-cloud", type=float, default=20.0, help="Cloud percentage threshold")
    parser.add_argument("--scale", type=float, default=10.0, help="Pixel scale in metres for downloads")
    parser.add_argument("--format", default="NPY", choices=["NPY"], help="Download format (NPY only)")
    parser.add_argument("--max-workers", type=int, default=1, help="Concurrent download workers")
    parser.add_argument("--http-timeout", type=int, default=600, help="HTTP timeout per tile download (seconds)")
    return parser.parse_args()


def monthly_stack(region: ee.Geometry, *, year: int, dataset_id: str, bands: Iterable[str], max_cloud: float) -> ee.Image:
    collection = ee.ImageCollection(dataset_id)
    composites = []
    for month in range(1, 13):
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, "month")
        filtered = (
            collection
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
            .select(list(bands))
        )
        image = filtered.median()
        month_str = ee.Number(month).format("%02d")
        renamed = image.rename(image.bandNames().map(lambda b: ee.String(b).cat("_m").cat(month_str)))
        composites.append(renamed)
    return ee.Image.cat(composites)


def resolve_tile_label(row, fallback_index: int) -> str:
    for candidate in ("tile_id", "tileID", "id"):
        value = row.get(candidate)
        if value not in (None, ""):
            return str(value)
    return f"{fallback_index:04d}"


def process_tile(
    idx: int,
    row,
    *,
    args: argparse.Namespace,
    band_names: Iterable[str],
) -> Tuple[str, Path]:
    tile_label = resolve_tile_label(row, idx)
    geom = row.geometry
    region = geom_to_ee(geom)
    image = monthly_stack(
        region,
        year=args.year,
        dataset_id=args.dataset_id,
        bands=args.bands,
        max_cloud=args.max_cloud,
    )
    url = image.getDownloadURL(
        {
            "bands": list(band_names),
            "region": region,
            "scale": args.scale,
            "format": args.format,
        }
    )
    response = requests.get(url, timeout=args.http_timeout)
    response.raise_for_status()
    payload = io.BytesIO(response.content)
    pixel_data = np.load(payload, allow_pickle=True)
    array = ee_bytes_to_img(pixel_data)
    cube = reshape_monthly_stack(array, len(args.bands))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"tile_{tile_label}_{args.year}.npy"
    np.save(output_path, cube)
    LOGGER.info("Saved %s", output_path)
    return tile_label, output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    init_ee(args.project_id)

    tiles = gpd.read_file(args.grid_path).to_crs(epsg=4326).reset_index(drop=True)
    band_names = [f"{band}_m{month:02d}" for month in range(1, 13) for band in args.bands]

    if args.max_workers > 1:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(process_tile, idx, row, args=args, band_names=band_names)
                for idx, row in tiles.iterrows()
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:  # pragma: no cover - surfacing errors
                    LOGGER.error("Tile download failed: %s", exc)
                    raise
    else:
        for idx, row in tiles.iterrows():
            process_tile(idx, row, args=args, band_names=band_names)


if __name__ == "__main__":
    main()
