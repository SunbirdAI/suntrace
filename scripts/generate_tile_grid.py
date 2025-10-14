"""Generate a square tile grid for the Lamwo AOI."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lamwo_pipeline import create_tile_grid

DEFAULT_AOI_PATH = Path("PATH/TO/lamwo_aoi.gpkg")
DEFAULT_OUTPUT_PATH = Path("PATH/TO/output/lamwo_grid.geojson")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aoi-path",
        type=Path,
        default=DEFAULT_AOI_PATH,
        help="Path to the AOI GeoPackage (update placeholder path).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the generated grid GeoJSON (update placeholder path).",
    )
    parser.add_argument(
        "--cell-size-m",
        type=float,
        default=1000,
        help="Tile edge length in metres (default: 1000).",
    )
    parser.add_argument(
        "--driver",
        default="GeoJSON",
        help="OGR driver to use when writing the grid (default: GeoJSON).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    grid = create_tile_grid(
        args.aoi_path,
        cell_size_m=args.cell_size_m,
        output_path=args.output_path,
        output_driver=args.driver,
    )

    logging.info("Generated %d tiles covering %s", len(grid), args.aoi_path)


if __name__ == "__main__":
    main()
