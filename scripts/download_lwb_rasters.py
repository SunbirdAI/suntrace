"""Download and optionally clip Aboveground Live Woody Biomass rasters per tile."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import geopandas as gpd
import requests
import rasterio
from rasterio.mask import mask

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger(__name__)

DEFAULT_AOI_PATH = Path("PATH/TO/lamwo_aoi.gpkg")
DEFAULT_SERVICE_URL = (
    "https://services2.arcgis.com/g8WusZB13b9OegfU/arcgis/rest/services/"
    "Aboveground_Live_Woody_Biomass_Density/FeatureServer/0/query?where=1%3D1&"
    "outFields=tile_id,Mg_px_1_download,Shape__Area,Shape__Length&outSR=4326&f=json"
)
DEFAULT_DOWNLOAD_DIR = Path("PATH/TO/output/aglwb_tiles")
DEFAULT_CLIP_DIR = Path("PATH/TO/output/aglwb_clipped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi-path", type=Path, default=DEFAULT_AOI_PATH, help="AOI footprint (GeoPackage/GeoJSON)")
    parser.add_argument("--service-url", default=DEFAULT_SERVICE_URL, help="ArcGIS FeatureServer query URL")
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR, help="Directory for downloaded rasters")
    parser.add_argument("--clip-dir", type=Path, default=DEFAULT_CLIP_DIR, help="Directory for AOI-clipped rasters")
    parser.add_argument("--tile-field", default="tile_id", help="Field name containing tile identifier")
    parser.add_argument(
        "--url-field",
        default="Mg_px_1_download",
        help="Field containing download URLs for each tile raster",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--http-timeout", type=int, default=600, help="HTTP timeout (seconds)")
    return parser.parse_args()


def download_raster(url: str, output_path: Path, timeout: int, overwrite: bool) -> bool:
    if output_path.exists() and not overwrite:
        LOGGER.info("Skipping existing %s", output_path)
        return False

    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    LOGGER.info("Downloaded %s", output_path)
    return True


def clip_raster_to_aoi(raster_path: Path, clip_path: Path, geometry) -> Path:
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, [geometry], crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )
        with rasterio.open(clip_path, "w", **out_meta) as dest:
            dest.write(out_image)
    LOGGER.info("Clipped %s", clip_path)
    return clip_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    aoi = gpd.read_file(args.aoi_path)
    if aoi.empty:
        raise ValueError(f"AOI at {args.aoi_path} is empty")
    aoi_geom = aoi.to_crs(epsg=4326).geometry.iloc[0]

    tiles = gpd.read_file(args.service_url)
    tiles = tiles.to_crs(epsg=4326)
    intersects = tiles[tiles.intersects(aoi_geom)]
    LOGGER.info("Found %d biomass tiles intersecting AOI", len(intersects))

    for _, row in intersects.iterrows():
        tile_id = row.get(args.tile_field)
        url = row.get(args.url_field)
        if not url:
            LOGGER.warning("Row %s missing URL field %s", tile_id, args.url_field)
            continue

        download_path = args.download_dir / f"{tile_id}.tif"
        downloaded = download_raster(url, download_path, args.http_timeout, args.overwrite)

        if downloaded or args.overwrite:
            clip_path = args.clip_dir / f"{tile_id}.tif"
            clip_raster_to_aoi(download_path, clip_path, aoi_geom)


if __name__ == "__main__":
    main()
