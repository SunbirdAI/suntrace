#!/usr/bin/env python3
"""
Bulk load geospatial files from the repository `data/` folder into PostGIS.

Behavior:
- Scans a data directory for files (GeoJSON, Shapefile, GPKG, CSV).
- For CSVs, tries to detect lat/lon columns or a WKT column and generates a temporary GeoJSON.
- Uses ogr2ogr to import files into PostGIS (robust for many file formats).
- After import, attempts to set SRID to 4326 (if missing) and creates a GiST spatial index.

Usage:
    python3 scripts/load_to_postgis.py --data-dir data --db-uri "postgresql://pguser:pgpass@localhost:5432/suntrace"

Requirements:
- ogr2ogr (GDAL) must be installed and on PATH
- Python packages: geopandas, pandas, sqlalchemy

This script is designed for development deployments where you run a local PostGIS container.
"""

import argparse
import logging
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("load_to_postgis")

DEFAULT_DB_URI = "postgresql+psycopg://pguser:pgpass@localhost:5432/suntrace"

CSV_LON_NAMES = {"lon", "longitude", "long", "x", "lng"}
CSV_LAT_NAMES = {"lat", "latitude", "y"}
WKT_NAMES = {"wkt", "geometry", "geom", "wkb", "wkt_geom"}


def safe_table_name(path: Path) -> str:
    name = path.stem.lower()
    # replace non-alnum with underscore
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    return name


def run_ogr2ogr(src: str, db_uri: str, table: str, ogr2ogr_path: str = "ogr2ogr"):
    # Quote the PG: connection string for use by ogr2ogr
    pg_conn = f'PG:"{db_uri}"'
    cmd = f'{ogr2ogr_path} -f "PostgreSQL" {pg_conn} {shlex.quote(src)} -nln public.{table} -lco GEOMETRY_NAME=geom -t_srs EPSG:4326 -overwrite'
    logger.info("Running: %s", cmd)
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error("ogr2ogr failed for %s: %s", src, res.stderr)
        raise RuntimeError(res.stderr)
    logger.info("Loaded %s -> public.%s", src, table)


def ensure_index_and_srid(engine, table: str, srid: int = 4326):
    # Set SRID where missing and create GIST index
    logger.info("Ensuring SRID and index for %s", table)
    with engine.begin() as conn:
        try:
            # Set SRID where unknown (st_srid = 0)
            conn.execute(text(f"UPDATE public.{table} SET geom = ST_SetSRID(geom, {srid}) WHERE ST_SRID(geom) = 0 OR ST_SRID(geom) IS NULL;"))
        except Exception as e:
            logger.debug("Could not run SRID update for %s: %s", table, e)
        try:
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS {table}_geom_gist ON public.{table} USING GIST(geom);"))
        except Exception as e:
            logger.warning("Could not create spatial index for %s: %s", table, e)


def csv_to_temp_geojson(csv_path: Path) -> str:
    """Detect geometry columns and write a temporary GeoJSON file, return its path."""
    logger.info("Inspecting CSV %s", csv_path)
    df = pd.read_csv(csv_path, nrows=100)
    cols = {c.lower() for c in df.columns}

    # Check for WKT column
    wkt_col = None
    for c in df.columns:
        if c.lower() in WKT_NAMES:
            wkt_col = c
            break

    if wkt_col:
        logger.info("Found WKT column '%s' in %s", wkt_col, csv_path)
        df_full = pd.read_csv(csv_path)
        gdf = gpd.GeoDataFrame(df_full, geometry=gpd.GeoSeries.from_wkt(df_full[wkt_col]), crs="EPSG:4326")
    else:
        # Look for lat/lon pairs
        lon_col = None
        lat_col = None
        for c in df.columns:
            if c.lower() in CSV_LON_NAMES and lon_col is None:
                lon_col = c
            if c.lower() in CSV_LAT_NAMES and lat_col is None:
                lat_col = c
        if lon_col and lat_col:
            logger.info("Found lat/lon columns %s/%s in %s", lat_col, lon_col, csv_path)
            df_full = pd.read_csv(csv_path)
            gdf = gpd.GeoDataFrame(df_full, geometry=gpd.points_from_xy(df_full[lon_col], df_full[lat_col]), crs="EPSG:4326")
        else:
            raise RuntimeError(f"Could not detect geometry in CSV {csv_path}. Provide WKT or lat/lon columns.")

    tmp = tempfile.NamedTemporaryFile(suffix=".geojson", delete=False)
    tmp_path = tmp.name
    tmp.close()
    gdf.to_file(tmp_path, driver="GeoJSON")
    logger.info("Wrote temporary GeoJSON %s", tmp_path)
    return tmp_path


def main(data_dir: str, db_uri: str, ogr2ogr_path: str = "ogr2ogr"):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise RuntimeError(f"Data directory not found: {data_dir}")

    engine = create_engine(db_uri)

    # Collect candidate files in the data directory (non-recursive by default)
    files = [p for p in data_path.iterdir() if p.is_file()]

    # Also include some common subfolders (e.g., lamwo_sentinel_composites)
    for sub in ["lamwo_sentinel_composites", "viz_geojsons", "sample_region_mudu"]:
        subp = data_path / sub
        if subp.exists() and subp.is_dir():
            files.extend([p for p in subp.iterdir() if p.is_file()])

    if not files:
        logger.warning("No data files found in %s", data_dir)
        return

    for f in files:
        try:
            if f.suffix.lower() in {".geojson", ".json", ".gpkg", ".shp"}:
                table = safe_table_name(f)
                run_ogr2ogr(str(f), db_uri, table, ogr2ogr_path=ogr2ogr_path)
                ensure_index_and_srid(engine, table)
            elif f.suffix.lower() == ".csv":
                table = safe_table_name(f)
                try:
                    tmp_geojson = csv_to_temp_geojson(f)
                except Exception as e:
                    logger.error("Skipping CSV %s: %s", f, e)
                    continue
                try:
                    run_ogr2ogr(tmp_geojson, db_uri, table, ogr2ogr_path=ogr2ogr_path)
                    ensure_index_and_srid(engine, table)
                finally:
                    try:
                        os.remove(tmp_geojson)
                    except Exception:
                        pass
            else:
                logger.info("Skipping unsupported file type: %s", f)
        except Exception as e:
            logger.error("Failed to load %s: %s", f, e)

    logger.info("Done loading files into PostGIS.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data", help="Path to data directory to scan and load")
    p.add_argument("--db-uri", default=DEFAULT_DB_URI, help="SQLAlchemy DB URI for PostGIS")
    p.add_argument("--ogr2ogr", default="ogr2ogr", help="Path to ogr2ogr binary")
    args = p.parse_args()

    main(args.data_dir, args.db_uri, ogr2ogr_path=args.ogr2ogr)
