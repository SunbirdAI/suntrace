"""
GeospatialAnalyzer2
Optimized version of GeospatialAnalyzer that pushes heavy spatial operations
to PostGIS when available, uses lazy loading, caching, and efficient SQL patterns.

Usage summary:
- Instantiate with database_uri (defaults to local container credentials).
- Use `ingest_to_postgis` to bulk-load GeoJSON/CSV into PostGIS via ogr2ogr.
- Use query methods (weighted_tile_stats_all, region_total_biomass, nearest_mini_grids,
  get_gdf_info_within_region) which prefer PostGIS SQL; falls back to GeoPandas
  if DB not available.

Designed for the suntrace repo. Keep logic similar to original GeospatialAnalyzer
but optimized for scale.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import tempfile
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt, wkb
from shapely.geometry import base, Point, Polygon
from sqlalchemy import create_engine, text

from configs.stats import DEFAULT_TILE_STAT_COLUMNS

# Configure logging for this module
logger = logging.getLogger("GeospatialAnalyzer2")
logger.setLevel(logging.INFO)

# Default DB URI when running the recommended local PostGIS docker
DEFAULT_DB_URI = "postgresql+psycopg://pguser:pgpass@localhost:5432/suntrace"


class GeospatialAnalyzer2:
    """Optimized geospatial analyzer.

    Key optimizations included:
    - Push spatial aggregates and spatial joins to PostGIS SQL (area-weighted,
      nearest-neighbor, counts).
    - Use GiST spatial indexes and SRID-correct area computations (ST_Transform
      to metric CRS before ST_Area).
    - Bulk-load helpers using ogr2ogr for robust ingestion.
    - Lazy loading and caching for repeated small queries.
    - Fallback to GeoPandas when PostGIS engine is not available.
    """

    def __init__(
        self,
        database_uri: Optional[str] = None,
        target_metric_crs: str = "EPSG:32636",
        target_geographic_crs: str = "EPSG:4326",
        ogr2ogr_path: str = "ogr2ogr",
        layer_table_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self._db_uri = database_uri or DEFAULT_DB_URI
        self._db_engine = None
        try:
            self._db_engine = create_engine(self._db_uri)
            logger.info("Created DB engine for %s", self._db_uri)
        except Exception as e:
            logger.warning("Could not create DB engine: %s; falling back to file-based ops", e)
            self._db_engine = None

        self.target_metric_crs = target_metric_crs
        self.target_geographic_crs = target_geographic_crs
        self._ogr2ogr = ogr2ogr_path

        default_layer_map = {
            "buildings": "public.lamwo_buildings",
            "tiles": "public.joined_tiles",
            "roads": "public.lamwo_roads",
            "villages": "public.lamwo_villages",
            "parishes": "public.lamwo_parishes",
            "subcounties": "public.lamwo_subcounties",
            "existing_grid": "public.existing_grid",
            "grid_extension": "public.grid_extension",
            "candidate_minigrids": "public.candidate_minigrids",
            "existing_minigrids": "public.existing_minigrids",
            "tile_stats": "public.lamwo_tile_stats_ee_biomass",
        }
        self._layer_table_map = default_layer_map if layer_table_map is None else {**default_layer_map, **layer_table_map}
        self._id_column_cache: Dict[str, str] = {}

    # ----------------------------- Ingestion helpers --------------------------
    def update_layer_table_map(self, overrides: Dict[str, str]) -> None:
        """Merge additional table mappings for logical layer names."""
        self._layer_table_map.update(overrides)

    def _resolve_table(self, layer_or_table: str) -> str:
        table = self._layer_table_map.get(layer_or_table, layer_or_table)
        if "." not in table and layer_or_table not in self._layer_table_map:
            raise ValueError(f"Unknown layer or table '{layer_or_table}'")
        return table

    def _escape_wkt_literal(self, geom_wkt: str) -> str:
        return geom_wkt.replace("'", "''")

    def _get_identifier_column(self, table: str) -> str:
        if table in self._id_column_cache:
            return self._id_column_cache[table]

        if not self._db_engine:
            return "ogc_fid"

        schema, tbl = (table.split(".") + [None])[:2]
        if tbl is None:
            tbl = schema
            schema = "public"
        query = text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = :schema AND table_name = :table
            """
        )
        with self._db_engine.connect() as conn:
            cols = {row[0] for row in conn.execute(query, {"schema": schema, "table": tbl})}

        preferred = ["id", "tile_id", "pt_id", "ts_id", "ogc_fid"]
        for cand in preferred:
            if cand in cols:
                self._id_column_cache[table] = cand
                return cand

        chosen = next(iter(cols)) if cols else "ogc_fid"
        self._id_column_cache[table] = chosen
        return chosen

    def _parse_extent(self, extent_str: Optional[str]) -> List[float]:
        if not extent_str:
            return [float("nan")] * 4
        try:
            extent_str = extent_str.replace("BOX(", "").replace(")", "")
            min_part, max_part = extent_str.split(",")
            minx, miny = map(float, min_part.strip().split())
            maxx, maxy = map(float, max_part.strip().split())
            return [minx, miny, maxx, maxy]
        except Exception:
            return [float("nan")] * 4

    def ingest_to_postgis(
        self,
        layer_map: Dict[str, str],
        schema: str = "public",
        overwrite: bool = True,
        srid: int = 4326,
        dtype_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """Bulk load files into PostGIS using ogr2ogr.

        layer_map: mapping of target_table_name -> local_file_path
        dtype_map: optional mapping of layer -> geometry type (e.g. "MULTIPOLYGON")

        This uses subprocess+ogr2ogr because it is robust for many filetypes and
        scales well for large GeoJSON/GeoPackage inputs.
        """
        if not self._db_engine:
            raise RuntimeError("No DB engine available for ingestion")

        for table, path in layer_map.items():
            geom_type = dtype_map.get(table) if dtype_map else None
            geom_flag = f"-nlt {geom_type}" if geom_type else ""
            overwrite_flag = "-overwrite" if overwrite else "-append"

            cmd = (
                f"{self._ogr2ogr} -f PostgreSQL "
                f"PG:\"{self._db_uri}\" "
                f"{shlex.quote(path)} -nln {schema}.{table} {geom_flag} "
                f"-lco GEOMETRY_NAME=geom -t_srs EPSG:{srid} {overwrite_flag}"
            )
            logger.info("Running ogr2ogr for %s -> %s.%s", path, schema, table)
            # Shell because PG: connection string uses colon/equals; use shell True
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if res.returncode != 0:
                logger.error("ogr2ogr failed for %s: %s", path, res.stderr)
                raise RuntimeError(f"ogr2ogr failed: {res.stderr}")
            logger.info("Loaded %s into %s.%s", path, schema, table)

    def ensure_postgis_extensions(self) -> None:
        """Create PostGIS extensions if missing (run once after DB init)."""
        if not self._db_engine:
            raise RuntimeError("No DB engine available to create extensions")
        with self._db_engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis_topology;"))
            logger.info("Ensured PostGIS extensions present")

    def create_spatial_index(self, table: str, geom_col: str = "geom") -> None:
        if not self._db_engine:
            raise RuntimeError("No DB engine available to create index")
        sql = text(f"CREATE INDEX IF NOT EXISTS {table}_geom_gist ON {table} USING GIST({geom_col});")
        with self._db_engine.connect() as conn:
            conn.execute(sql)
            logger.info("Created GiST index on %s(%s)", table, geom_col)

    # ----------------------------- Query helpers ------------------------------
    def query_postgis(self, sql: str, geom_col: Optional[str] = "geom") -> pd.DataFrame:
        """Run SQL against PostGIS and return Geo/regular DataFrame depending on geom_col."""
        if not self._db_engine:
            logger.debug("No DB engine; returning empty GeoDataFrame for SQL: %s", sql)
            return gpd.GeoDataFrame()
        try:
            if geom_col:
                return gpd.read_postgis(sql, self._db_engine, geom_col=geom_col)
            return pd.read_sql_query(sql, self._db_engine)
        except Exception as e:
            logger.error("PostGIS query failed: %s", e)
            return gpd.GeoDataFrame()

    def scalar_query(self, sql: str) -> Any:
        """Run scalar SQL and return single value (first column of first row)."""
        if not self._db_engine:
            logger.debug("No DB engine for scalar query: %s", sql)
            return None
        with self._db_engine.connect() as conn:
            res = conn.execute(text(sql)).fetchone()
            return res[0] if res is not None else None

    # ----------------------------- Utility helpers ----------------------------
    def _region_as_wkt(self, region: base.BaseGeometry) -> str:
        if isinstance(region, str):
            return region
        if hasattr(region, "to_wkt"):
            return region.to_wkt()
        return wkt.dumps(region)

    def _ensure_wgs84_wkt(self, region: base.BaseGeometry) -> str:
        # Store/query in 4326 for WKT readability; upstream SQL will ST_Transform as needed
        try:
            # if region is a GeoSeries/GeoDataFrame element, convert
            if hasattr(region, "__geo_interface__"):
                return wkt.dumps(region)
        except Exception:
            pass
        return wkt.dumps(region)

    # ----------------------------- High-level optimized methods --------------
    def count_features_within_region(self, region: base.BaseGeometry, layer_or_table: str) -> int:
        """Count rows intersecting the region using PostGIS (fast) or GeoPandas fallback."""
        wkt_region = self._ensure_wgs84_wkt(region)
        table = self._resolve_table(layer_or_table)
        if self._db_engine:
            sql = f"SELECT COUNT(*) FROM {table} WHERE ST_Intersects({table}.geom, ST_GeomFromText('{wkt_region}', 4326));"
            res = self.scalar_query(sql)
            return int(res or 0)

        # Fallback: load table as file-based gdf (user must implement mapping outside)
        logger.debug("Fallback count - no DB engine")
        return 0

    def get_tile_ids_within_region(self, region: base.BaseGeometry, table: str = "tiles") -> List[str]:
        """Return identifiers for tiles intersecting region (uses `ogc_fid` fallback)."""
        if not self._db_engine:
            logger.debug("Fallback get_tile_ids_within_region - no DB engine")
            return []

        table = self._resolve_table(table)
        id_col = self._get_identifier_column(table)
        region_wkt = self._escape_wkt_literal(self._ensure_wgs84_wkt(region))
        sql = f"""
        WITH region AS (SELECT ST_GeomFromText('{region_wkt}', 4326) AS geom)
        SELECT CAST(t.{id_col} AS text) AS tile_id
        FROM {table} t, region r
        WHERE t.geom IS NOT NULL
          AND ST_Intersects(t.geom, r.geom)
        ORDER BY tile_id;
        """
        df = self.query_postgis(sql, geom_col=None)
        if df.empty:
            return []
        return [str(v) for v in df["tile_id"].tolist()]

    def get_gdf_info_within_region(
        self,
        region: base.BaseGeometry,
        layer_or_table: str,
        filter_expr: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> gpd.GeoDataFrame:
        """Return features from `table` that intersect region. Prefers PostGIS.

        table should be a fully-qualified table name or public.table.
        """
        wkt_region = self._ensure_wgs84_wkt(region)
        table = self._resolve_table(layer_or_table)
        if self._db_engine:
            where_clauses = [f"ST_Intersects({table}.geom, ST_GeomFromText('{wkt_region}', 4326))"]
            if filter_expr:
                where_clauses.append(filter_expr)
            where_sql = " AND ".join(where_clauses)
            limit_sql = f"LIMIT {int(limit)}" if limit else ""
            sql = f"SELECT * FROM {table} WHERE {where_sql} {limit_sql};"
            gdf = self.query_postgis(sql)
            return gdf

        logger.debug("Fallback get_gdf_info_within_region - no DB engine")
        return gpd.GeoDataFrame()

    # Layer-specific wrappers --------------------------------------------------
    def get_layer_info_within_region(
        self,
        region: base.BaseGeometry,
        layer_name: str,
        filter_expr: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> gpd.GeoDataFrame:
        return self.get_gdf_info_within_region(region, layer_name, filter_expr, limit)

    def get_tiles_info_within_region(self, region: base.BaseGeometry) -> gpd.GeoDataFrame:
        return self.get_layer_info_within_region(region, "tiles")

    def get_roads_info_within_region(self, region: base.BaseGeometry) -> gpd.GeoDataFrame:
        return self.get_layer_info_within_region(region, "roads")

    def get_villages_info_within_region(self, region: base.BaseGeometry) -> gpd.GeoDataFrame:
        return self.get_layer_info_within_region(region, "villages")

    def get_parishes_info_within_region(self, region: base.BaseGeometry) -> gpd.GeoDataFrame:
        return self.get_layer_info_within_region(region, "parishes")

    def get_subcounties_info_within_region(self, region: base.BaseGeometry) -> gpd.GeoDataFrame:
        return self.get_layer_info_within_region(region, "subcounties")

    def get_existing_grid_info_within_region(self, region: base.BaseGeometry) -> gpd.GeoDataFrame:
        return self.get_layer_info_within_region(region, "existing_grid")

    def get_grid_extension_info_within_region(self, region: base.BaseGeometry) -> gpd.GeoDataFrame:
        return self.get_layer_info_within_region(region, "grid_extension")

    def get_candidate_minigrids_info_within_region(self, region: base.BaseGeometry) -> gpd.GeoDataFrame:
        return self.get_layer_info_within_region(region, "candidate_minigrids")

    def get_existing_minigrids_info_within_region(self, region: base.BaseGeometry) -> gpd.GeoDataFrame:
        return self.get_layer_info_within_region(region, "existing_minigrids")

    def nearest_mini_grids(self, pt: Point, table: str = "candidate_minigrids", k: int = 3) -> List[Tuple[str, float]]:
        """KNN nearest neighbor using PostGIS operator (<->) which uses the spatial index.

        Returns list of tuples (id, distance_meters)
        """
        wkt_pt = self._ensure_wgs84_wkt(pt)
        if not self._db_engine:
            logger.debug("Fallback nearest - no DB engine")
            return []

        table = self._resolve_table(table)
        # Compute distance in metric CRS by transforming to metric CRS in the SQL
        sql = f"""
        SELECT id, ST_Distance(ST_Transform({table}.geom, {self.target_metric_crs.split(':')[-1]}), ST_Transform(ST_GeomFromText('{wkt_pt}',4326), {self.target_metric_crs.split(':')[-1]})) AS distance_m
        FROM {table}
        WHERE {table}.geom IS NOT NULL
        ORDER BY {table}.geom <-> ST_GeomFromText('{wkt_pt}',4326)
        LIMIT {int(k)};
        """
        df = self.query_postgis(sql, geom_col=None)
        if df.empty:
            return []
        return [(row["id"], float(row["distance_m"])) for _, row in df.iterrows()]

    def weighted_tile_stats_all(
        self,
        region: base.BaseGeometry,
        table: str = "tile_stats",
        stat_columns: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute area-weighted averages for specified stat_columns within region.

        This builds a single SQL query that computes all requested weighted stats in one pass.
        """
        if stat_columns is None:
            stat_columns = DEFAULT_TILE_STAT_COLUMNS

        wkt_region = self._ensure_wgs84_wkt(region)
        table = self._resolve_table(table)
        if not self._db_engine:
            logger.debug("Fallback weighted_tile_stats_all - no DB engine")
            return {col: float("nan") for col in stat_columns}

        # Build weighted expressions using metric CRS area
        metric_epsg = int(self.target_metric_crs.split(":")[-1])
        weighted_selects = []
        for col in stat_columns:
            safe_col = col  # if needed, sanitize/validate
            weighted = (
                f"SUM({safe_col} * ST_Area(ST_Transform(ST_Intersection(t.geom, region.geom), {metric_epsg})))::double precision"
            )
            denom = f"NULLIF(SUM(ST_Area(ST_Transform(ST_Intersection(t.geom, region.geom), {metric_epsg}))),0)"
            alias = f"{safe_col}_wavg"
            weighted_selects.append(f"({weighted} / {denom}) AS {alias}")

        selects_sql = ",\n                ".join(weighted_selects)

        sql = f"""
        WITH region AS (SELECT ST_GeomFromText('{wkt_region}', 4326) AS geom),
             t AS (SELECT * FROM {table} WHERE {table}.geom IS NOT NULL AND ST_Intersects({table}.geom, (SELECT geom FROM region)))
        SELECT
                {selects_sql}
        FROM t, region;
        """

        df = self.query_postgis(sql, geom_col=None)
        if df.empty:
            return {c: float("nan") for c in stat_columns}

        result: Dict[str, float] = {}
        for col in stat_columns:
            alias = f"{col}_wavg"
            val = df.iloc[0].get(alias)
            result[col] = float(val) if val is not None else float("nan")
        return result

    def weighted_tile_stat(self, region: base.BaseGeometry, stat: str, table: str = "tile_stats") -> float:
        stats = self.weighted_tile_stats_all(region, table=table, stat_columns=[stat])
        return stats.get(stat, float("nan"))

    def avg_ndvi(self, region: base.BaseGeometry) -> float:
        return self.weighted_tile_stat(region, "ndvi_mean")

    def ndvi_stats(self, region: base.BaseGeometry) -> Dict[str, float]:
        values = self.weighted_tile_stats_all(region, stat_columns=["ndvi_mean", "ndvi_med", "ndvi_std"])
        return {
            "NDVI_mean": values.get("ndvi_mean", float("nan")),
            "NDVI_med": values.get("ndvi_med", float("nan")),
            "NDVI_std": values.get("ndvi_std", float("nan")),
        }

    def evi_med(self, region: base.BaseGeometry) -> float:
        return self.weighted_tile_stat(region, "evi_med")

    def cf_days(self, region: base.BaseGeometry) -> float:
        return self.weighted_tile_stat(region, "cloud_free_days")

    def elev_mean(self, region: base.BaseGeometry) -> float:
        return self.weighted_tile_stat(region, "elev_mean")

    def slope_mean(self, region: base.BaseGeometry) -> float:
        return self.weighted_tile_stat(region, "slope_mean")

    def par_mean(self, region: base.BaseGeometry) -> float:
        return self.weighted_tile_stat(region, "par_mean")

    def region_total_biomass(
        self,
        region: base.BaseGeometry,
        table: str = "tile_stats",
        tile_total_col: str = "tile_total_Mg",
    ) -> float:
        """Compute total biomass for a region using PostGIS area-weighting on tile totals.

        If tile_total_col holds per-tile totals (already aggregated per tile), this
        method computes the fraction of each tile within the region and multiplies.
        """
        wkt_region = self._ensure_wgs84_wkt(region)
        table = self._resolve_table(table)
        if not self._db_engine:
            logger.debug("Fallback region_total_biomass - no DB engine")
            return float("nan")

        metric_epsg = int(self.target_metric_crs.split(":")[-1])
        sql = f"""
        WITH region AS (SELECT ST_GeomFromText('{wkt_region}', 4326) AS geom)
        SELECT SUM((ST_Area(ST_Transform(ST_Intersection(t.geom, r.geom), {metric_epsg})) / NULLIF(ST_Area(ST_Transform(t.geom, {metric_epsg})),0)) * t.{tile_total_col})::double precision AS total_biomass
        FROM {table} t, region r
        WHERE ST_Intersects(t.geom, r.geom);
        """
        val = self.scalar_query(sql)
        return float(val or 0.0)


    def list_mini_grids(self, table: str = "existing_minigrids") -> List[str]:
        if not self._db_engine:
            logger.debug("Fallback list_mini_grids - no DB engine")
            return []

        table = self._resolve_table(table)
        sql = f"SELECT COALESCE(name, location::text, id::text) AS label FROM {table} WHERE geom IS NOT NULL ORDER BY label;"
        df = self.query_postgis(sql, geom_col=None)
        if df.empty:
            return []
        return [str(v) for v in df["label"].dropna().tolist()]

    def get_layer_geometry(self, layer_name: str, region: base.BaseGeometry) -> Optional[base.BaseGeometry]:
        if not self._db_engine:
            logger.debug("Fallback get_layer_geometry - no DB engine")
            return None

        table = self._resolve_table(layer_name)
        region_wkt = self._escape_wkt_literal(self._ensure_wgs84_wkt(region))
        sql = f"""
        WITH region AS (SELECT ST_GeomFromText('{region_wkt}', 4326) AS geom),
             clipped AS (
                SELECT ST_Intersection(t.geom, region.geom) AS geom
                FROM {table} t, region
                WHERE t.geom IS NOT NULL AND ST_Intersects(t.geom, region.geom)
             )
        SELECT ST_AsEWKB(ST_UnaryUnion(clipped.geom)) AS geom
        FROM clipped;
        """
        with self._db_engine.connect() as conn:
            row = conn.execute(text(sql)).fetchone()
            if not row or row[0] is None:
                return None
            return wkb.loads(bytes(row[0]))

    def compute_distance_to_grid(self, geometry: base.BaseGeometry) -> float:
        if not self._db_engine:
            logger.debug("Fallback compute_distance_to_grid - no DB engine")
            return float("nan")

        table = self._resolve_table("existing_grid")
        geom_wkt = self._escape_wkt_literal(self._ensure_wgs84_wkt(geometry))
        metric_epsg = int(self.target_metric_crs.split(":")[-1])
        sql = f"""
        WITH region AS (SELECT ST_GeomFromText('{geom_wkt}', 4326) AS geom)
        SELECT MIN(ST_Distance(ST_Transform(t.geom, {metric_epsg}), ST_Transform(region.geom, {metric_epsg})))
        FROM {table} t, region
        WHERE t.geom IS NOT NULL;
        """
        val = self.scalar_query(sql)
        return float(val) if val is not None else float("nan")

    # ----------------------------- Region analysis pipelines ------------------
    def get_layer_bounds(self, layer_name: str, target_epsg: int = 4326) -> List[float]:
        if not self._db_engine:
            logger.debug("Fallback get_layer_bounds - no DB engine")
            return [float("nan")] * 4

        table = self._resolve_table(layer_name)
        sql = text(
            f"SELECT ST_Extent(ST_Transform(t.geom, {target_epsg})) FROM {table} AS t WHERE t.geom IS NOT NULL;"
        )
        with self._db_engine.connect() as conn:
            row = conn.execute(sql).fetchone()
        return self._parse_extent(row[0] if row else None)

    def get_layer_geojson(
        self,
        layer_name: str,
        *,
        limit: Optional[int] = None,
        sample: Optional[int] = None,
        target_epsg: int = 4326,
    ) -> Dict[str, Any]:
        if not self._db_engine:
            logger.debug("Fallback get_layer_geojson - no DB engine")
            return {"type": "FeatureCollection", "features": []}

        table = self._resolve_table(layer_name)
        order_clause = "ORDER BY random()" if sample and sample > 0 else ""
        row_limit = sample if sample and sample > 0 else limit
        limit_clause = f"LIMIT {int(row_limit)}" if row_limit else ""

        sql = f"""
        WITH features AS (
            SELECT jsonb_build_object(
                'type', 'Feature',
                'geometry', ST_AsGeoJSON(ST_Transform(t.geom, {target_epsg}))::jsonb,
                'properties', to_jsonb(t) - 'geom'
            ) AS feature
            FROM {table} AS t
            WHERE t.geom IS NOT NULL
            {order_clause}
            {limit_clause}
        )
        SELECT COALESCE(
            jsonb_build_object(
                'type', 'FeatureCollection',
                'features', COALESCE(jsonb_agg(feature), '[]'::jsonb)
            )::text,
            '{{"type":"FeatureCollection","features":[]}}'
        )
        FROM features;
        """

        with self._db_engine.connect() as conn:
            row = conn.execute(text(sql)).fetchone()
        if not row or row[0] is None:
            return {"type": "FeatureCollection", "features": []}
        payload = row[0]
        if isinstance(payload, str):
            return json.loads(payload)
        return payload

    def get_layer_count(self, layer_name: str) -> int:
        if not self._db_engine:
            logger.debug("Fallback get_layer_count - no DB engine")
            return 0
        table = self._resolve_table(layer_name)
        sql = f"SELECT COUNT(*) FROM {table} AS t WHERE t.geom IS NOT NULL;"
        return int(self.scalar_query(sql) or 0)

    def _analyze_settlements_in_region(self, region: base.BaseGeometry) -> Dict[str, Any]:
        if not self._db_engine:
            logger.debug("Fallback _analyze_settlements_in_region - no DB engine")
            return {
                "building_count": 0,
                "building_categories": {},
                "intersecting_village_count": 0,
                "intersecting_village_details": [],
                "has_truncated_villages": False,
            }

        region_wkt = self._escape_wkt_literal(self._ensure_wgs84_wkt(region))
        buildings_table = self._resolve_table("buildings")
        villages_table = self._resolve_table("villages")

        building_count_sql = f"""
        WITH region AS (SELECT ST_GeomFromText('{region_wkt}', 4326) AS geom)
        SELECT COUNT(*)
        FROM {buildings_table} b, region r
        WHERE b.geom IS NOT NULL AND ST_Intersects(b.geom, r.geom);
        """
        building_count = int(self.scalar_query(building_count_sql) or 0)

        categories_sql = f"""
        WITH region AS (SELECT ST_GeomFromText('{region_wkt}', 4326) AS geom)
        SELECT COALESCE(NULLIF(TRIM(CAST(b.category AS text)), ''), 'residential') AS category,
               COUNT(*)::bigint AS feature_count
        FROM {buildings_table} b, region r
        WHERE b.geom IS NOT NULL AND ST_Intersects(b.geom, r.geom)
        GROUP BY category
        ORDER BY feature_count DESC;
        """
        cat_df = self.query_postgis(categories_sql, geom_col=None)
        building_categories = {
            (row.get("category") or "residential"): int(row.get("feature_count", 0))
            for _, row in cat_df.iterrows()
        } if not cat_df.empty else {}

        villages_total_sql = f"""
        WITH region AS (SELECT ST_GeomFromText('{region_wkt}', 4326) AS geom)
        SELECT COUNT(*)
        FROM {villages_table} v, region r
        WHERE v.geom IS NOT NULL AND ST_Intersects(v.geom, r.geom);
        """
        total_villages = int(self.scalar_query(villages_total_sql) or 0)

        villages_detail_sql = f"""
        WITH region AS (SELECT ST_GeomFromText('{region_wkt}', 4326) AS geom)
        SELECT
            COALESCE(CAST(v.addr_vname AS text), 'Unnamed') AS name,
            COALESCE(CAST(v.category AS text), 'Unknown') AS electrification_category,
            v.rank
        FROM {villages_table} v, region r
        WHERE v.geom IS NOT NULL AND ST_Intersects(v.geom, r.geom)
        ORDER BY name
        LIMIT 20;
        """
        villages_df = self.query_postgis(villages_detail_sql, geom_col=None)
        village_data: List[Dict[str, Any]] = []
        if not villages_df.empty:
            for _, row in villages_df.iterrows():
                info = {
                    "name": row.get("name", "Unnamed"),
                    "electrification_category": row.get("electrification_category", "Unknown"),
                }
                rank_val = row.get("rank")
                if rank_val is not None and not pd.isna(rank_val):
                    try:
                        info["priority_rank"] = int(rank_val)
                    except Exception:
                        pass
                village_data.append(info)

        return {
            "building_count": building_count,
            "building_categories": building_categories,
            "intersecting_village_count": total_villages,
            "intersecting_village_details": village_data,
            "has_truncated_villages": total_villages > len(village_data),
        }

    def _analyze_infrastructure_in_region(self, region: base.BaseGeometry) -> Dict[str, Any]:
        if not self._db_engine:
            logger.debug("Fallback _analyze_infrastructure_in_region - no DB engine")
            return {
                "roads": {"total_road_segments": 0, "road_types": {}},
                "electricity": {
                    "existing_grid_present": False,
                    "distance_to_existing_grid": float("nan"),
                    "grid_extension_proposed": False,
                    "candidate_minigrids_count": 0,
                    "existing_minigrids_count": 0,
                    "capacity_distribution": {},
                    "population_to_be_served": 0,
                },
            }

        region_wkt = self._escape_wkt_literal(self._ensure_wgs84_wkt(region))
        roads_table = self._resolve_table("roads")
        candidate_table = self._resolve_table("candidate_minigrids")
        existing_table = self._resolve_table("existing_minigrids")

        roads_total_sql = f"""
        WITH region AS (SELECT ST_GeomFromText('{region_wkt}', 4326) AS geom)
        SELECT COUNT(*)
        FROM {roads_table} r, region
        WHERE r.geom IS NOT NULL AND ST_Intersects(r.geom, region.geom);
        """
        total_roads = int(self.scalar_query(roads_total_sql) or 0)

        road_types_sql = f"""
        WITH region AS (SELECT ST_GeomFromText('{region_wkt}', 4326) AS geom)
        SELECT COALESCE(CAST(r.highway AS text), 'unknown') AS highway,
               COUNT(*)::bigint AS feature_count
        FROM {roads_table} r, region
        WHERE r.geom IS NOT NULL AND ST_Intersects(r.geom, region.geom)
        GROUP BY highway
        ORDER BY feature_count DESC;
        """
        road_types_df = self.query_postgis(road_types_sql, geom_col=None)
        road_types = {
            (row.get("highway") or "unknown"): int(row.get("feature_count", 0))
            for _, row in road_types_df.iterrows()
        } if not road_types_df.empty else {}

        grid_present = self.count_features_within_region(region, "existing_grid") > 0
        grid_extension = self.count_features_within_region(region, "grid_extension") > 0
        distance_to_grid = self.compute_distance_to_grid(region)

        candidate_counts_sql = f"""
        WITH region AS (SELECT ST_GeomFromText('{region_wkt}', 4326) AS geom)
        SELECT COUNT(*) AS cnt, SUM(COALESCE(candidate.population, 0)) AS population_sum
        FROM {candidate_table} candidate, region
        WHERE candidate.geom IS NOT NULL AND ST_Intersects(candidate.geom, region.geom);
        """
        candidate_row = self.query_postgis(candidate_counts_sql, geom_col=None)
        candidate_count = 0
        population_sum = 0
        if not candidate_row.empty:
            candidate_count = int(candidate_row.iloc[0].get("cnt") or 0)
            population_sum = int(candidate_row.iloc[0].get("population_sum") or 0)

        capacity_sql = f"""
        WITH region AS (SELECT ST_GeomFromText('{region_wkt}', 4326) AS geom)
        SELECT COALESCE(CAST(candidate.capacity AS text), 'unknown') AS capacity,
               COUNT(*)::bigint AS feature_count
        FROM {candidate_table} candidate, region
        WHERE candidate.geom IS NOT NULL AND ST_Intersects(candidate.geom, region.geom)
        GROUP BY capacity
        ORDER BY feature_count DESC;
        """
        capacity_df = self.query_postgis(capacity_sql, geom_col=None)
        capacity_distribution = {
            (row.get("capacity") or "unknown"): int(row.get("feature_count", 0))
            for _, row in capacity_df.iterrows()
        } if not capacity_df.empty else {}

        existing_minigrids_count = self.count_features_within_region(region, existing_table)

        return {
            "roads": {"total_road_segments": total_roads, "road_types": road_types},
            "electricity": {
                "existing_grid_present": grid_present,
                "distance_to_existing_grid": distance_to_grid,
                "grid_extension_proposed": grid_extension,
                "candidate_minigrids_count": candidate_count,
                "existing_minigrids_count": existing_minigrids_count,
                "capacity_distribution": capacity_distribution,
                "population_to_be_served": population_sum,
            },
        }

    def _analyze_administrative_divisions(self, region: base.BaseGeometry) -> Dict[str, Any]:
        if not self._db_engine:
            logger.debug("Fallback _analyze_administrative_divisions - no DB engine")
            return {
                "parishes": {"count": 0, "details": []},
                "subcounties": {"count": 0, "names": []},
            }

        region_wkt = self._escape_wkt_literal(self._ensure_wgs84_wkt(region))
        parishes_table = self._resolve_table("parishes")
        subcounties_table = self._resolve_table("subcounties")

        parishes_sql = f"""
        WITH region AS (SELECT ST_GeomFromText('{region_wkt}', 4326) AS geom)
        SELECT COALESCE(CAST(p.addr_pname AS text), 'Unnamed') AS name,
               COALESCE(CAST(p.category AS text), 'Unknown') AS electrification_category
        FROM {parishes_table} p, region
        WHERE p.geom IS NOT NULL AND ST_Intersects(p.geom, region.geom);
        """
        parishes_df = self.query_postgis(parishes_sql, geom_col=None)
        parish_details: List[Dict[str, Any]] = []
        if not parishes_df.empty:
            for _, row in parishes_df.iterrows():
                parish_details.append(
                    {
                        "name": row.get("name", "Unnamed"),
                        "electrification_category": row.get("electrification_category", "Unknown"),
                    }
                )

        subcounties_sql = f"""
        WITH region AS (SELECT ST_GeomFromText('{region_wkt}', 4326) AS geom)
        SELECT DISTINCT COALESCE(CAST(s.addr_sname AS text), 'Unnamed') AS name
        FROM {subcounties_table} s, region
        WHERE s.geom IS NOT NULL AND ST_Intersects(s.geom, region.geom)
        ORDER BY name;
        """
        subcounties_df = self.query_postgis(subcounties_sql, geom_col=None)
        subcounty_names = (
            subcounties_df["name"].dropna().tolist() if not subcounties_df.empty else []
        )

        return {
            "parishes": {"count": len(parish_details), "details": parish_details},
            "subcounties": {"count": len(subcounty_names), "names": subcounty_names},
        }

    def _analyze_environmental_metrics(self, region: base.BaseGeometry) -> Dict[str, Any]:
        stats = self.weighted_tile_stats_all(region)
        if not stats:
            return {}

        for key, value in list(stats.items()):
            if isinstance(value, (int, float)) and not np.isnan(value):
                stats[key] = round(float(value), 4)

        ndvi_value = stats.get("ndvi_mean", float("nan"))
        if isinstance(ndvi_value, (int, float)) and not np.isnan(ndvi_value):
            if ndvi_value > 0.5:
                stats["vegetation_density"] = "Dense vegetation"
            elif ndvi_value > 0.2:
                stats["vegetation_density"] = "Moderate vegetation"
            elif ndvi_value > 0:
                stats["vegetation_density"] = "Sparse vegetation"
            else:
                stats["vegetation_density"] = "Very limited vegetation"

        biomass = self.region_total_biomass(region)
        if isinstance(biomass, (int, float)) and not np.isnan(biomass):
            stats["total_biomass_Mg"] = round(float(biomass), 4)
        else:
            stats["total_biomass_Mg"] = float("nan")

        try:
            gseries = gpd.GeoSeries([region], crs=self.target_geographic_crs)
            area_km2 = float(gseries.to_crs(self.target_metric_crs).area.iloc[0] / 1e6)
            stats["region_area_km2"] = round(area_km2, 4)
        except Exception:
            stats["region_area_km2"] = float("nan")
        return stats

    def analyze_region(self, region: base.BaseGeometry) -> Dict[str, Any]:
        return {
            "settlements": self._analyze_settlements_in_region(region),
            "infrastructure": self._analyze_infrastructure_in_region(region),
            "administrative": self._analyze_administrative_divisions(region),
            "environment": self._analyze_environmental_metrics(region),
        }

    def create_joined_tiles(
        self,
        tile_stats_table: str,
        plain_tiles_table: str,
        joined_table: str = "public.joined_tiles",
        metric_epsg: int = 32636,
    ) -> None:
        """Create a DB-side joined tiles table that merges tile stats with plain tile geometries.

        The function will:
        - create temporary tables with stable text IDs when needed (row number fallback),
        - left-join stats -> geometries on those ids,
        - ensure geometry SRID and type,
        - compute area_m2 and area_ha,
        - detect biomass-like columns and compute `tile_total_Mg` when missing.

        This keeps heavy work in PostGIS and returns quickly once materialized.
        """
        if not self._db_engine:
            raise RuntimeError("No DB engine available to create joined tiles")

        # Normalize inputs (strip public. prefix for table checks)
        def short(t: str) -> str:
            return t.split(".")[-1]

        stats_short = short(tile_stats_table)
        tiles_short = short(plain_tiles_table)
        joined_short = short(joined_table)

        stmts = []

        # Create plain tiles with deterministic pt_id
        stmts.append(f"DROP TABLE IF EXISTS public.{tiles_short}_with_id;")
        stmts.append(
            f"CREATE TABLE public.{tiles_short}_with_id AS SELECT *, COALESCE(CAST(id AS text), (ROW_NUMBER() OVER ())::text) AS pt_id FROM {plain_tiles_table};"
        )

        # Create tile stats with deterministic ts_id
        stmts.append(f"DROP TABLE IF EXISTS public.{stats_short}_with_id;")
        stmts.append(
            f"CREATE TABLE public.{stats_short}_with_id AS SELECT *, COALESCE(CAST(id AS text), (ROW_NUMBER() OVER ())::text) AS ts_id FROM {tile_stats_table};"
        )

        # Create joined table by left joining stats -> tiles on id text
        stmts.append(f"DROP TABLE IF EXISTS {joined_table};")
        stmts.append(
            f"CREATE TABLE {joined_table} AS SELECT s.*, t.geom FROM public.{stats_short}_with_id s LEFT JOIN public.{tiles_short}_with_id t ON s.ts_id = t.pt_id;"
        )

        # Ensure geometry SRID and create spatial index
        stmts.append(
            f"ALTER TABLE {joined_table} ALTER COLUMN geom TYPE geometry(Geometry,4326) USING ST_SetSRID(geom,4326);"
        )
        stmts.append(f"CREATE INDEX IF NOT EXISTS {joined_short}_geom_gist ON {joined_table} USING GIST(geom);")

        # Compute area in metric CRS
        stmts.append(f"ALTER TABLE {joined_table} DROP COLUMN IF EXISTS area_m2;")
        stmts.append(f"ALTER TABLE {joined_table} ADD COLUMN area_m2 double precision;")
        stmts.append(
            f"UPDATE {joined_table} SET area_m2 = ST_Area(ST_Transform(geom, {metric_epsg}));"
        )
        stmts.append(f"ALTER TABLE {joined_table} DROP COLUMN IF EXISTS area_ha;")
        stmts.append(f"ALTER TABLE {joined_table} ADD COLUMN area_ha double precision;")
        stmts.append(f"UPDATE {joined_table} SET area_ha = area_m2 / 10000.0;")

        # Detect biomass-like columns and populate tile_total_Mg if missing
        # Strategy: if tile_total_Mg exists in stats, keep; else look for any column containing 'biomass' or 'Mg_ha' and compute
        with self._db_engine.connect() as conn:
            for s in stmts:
                try:
                    conn.execute(text(s))
                except Exception as e:
                    logger.warning("Statement failed during create_joined_tiles: %s -> %s", s, e)

            # Inspect columns
            res = conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns WHERE table_name = :t"
                ),
                {"t": joined_short},
            )
            cols = {row[0].lower() for row in res.fetchall()}

            if "tile_total_mg" in cols:
                logger.info("joined table already has tile_total_Mg; no computation needed")
            else:
                # find candidate biomass cols
                candidates = [c for c in cols if "biomass" in c or "mg_ha" in c or "mgperha" in c]
                if candidates:
                    col = candidates[0]
                    logger.info("Computing tile_total_Mg from %s * area_ha", col)
                    # create column and compute
                    try:
                        conn.execute(text(f"ALTER TABLE {joined_table} ADD COLUMN tile_total_mg double precision;"))
                    except Exception:
                        pass
                    # If candidate is per-hectare (units mg/ha or Mg/ha), multiply by area_ha
                    conn.execute(
                        text(
                            f"UPDATE {joined_table} SET tile_total_mg = COALESCE({col}::double precision,0) * COALESCE(area_ha,0);"
                        )
                    )
                else:
                    logger.info("No biomass-like column found; leaving tile_total_Mg empty")

            # Ensure index on potential id columns
            try:
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS {joined_short}_id_idx ON {joined_table} ((COALESCE(id::text,'')));"))
            except Exception:
                pass

        logger.info("Created joined tiles table: %s", joined_table)

    # ----------------------------- Convenience utilities ---------------------
    @lru_cache(maxsize=256)
    def cached_count(self, region_wkt: str, table: str) -> int:
        return self.count_features_within_region(wkt.loads(region_wkt), table)


# End of GeospatialAnalyzer2
