"""Factory helpers for constructing geospatial analyzers."""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, Optional

from configs import paths
from utils.GeospatialAnalyzer import GeospatialAnalyzer
from utils.GeospatialAnalyzer2 import GeospatialAnalyzer2, DEFAULT_DB_URI

logger = logging.getLogger(__name__)

# Ensure repo-relative imports continue to work in legacy contexts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def _resolve_data_path(relative: Optional[str], default_path) -> str:
    if relative:
        return os.path.join(os.path.dirname(__file__), "../../data/" + relative)
    return default_path


def create_geospatial_analyzer(
    bpath: Optional[str] = None,
    tpath: Optional[str] = None,
    ppath: Optional[str] = None,
    cmpath: Optional[str] = None,
    empath: Optional[str] = None,
    egpath: Optional[str] = None,
    gepath: Optional[str] = None,
    rpath: Optional[str] = None,
    vpath: Optional[str] = None,
    papath: Optional[str] = None,
    spath: Optional[str] = None,
    *,
    use_postgis: Optional[bool] = None,
    database_uri: Optional[str] = None,
    layer_table_map: Optional[Dict[str, str]] = None,
):
    """Return a geospatial analyzer tailored to the current environment.

    If PostGIS configuration is detected (either via explicit parameters or
    environment variables), instantiate the PostGIS-first analyzer. Otherwise
    fall back to the legacy GeoPandas implementation that reads local files.
    """

    if use_postgis is None:
        env_flag = os.getenv("SUNTRACE_USE_POSTGIS")
        if env_flag is not None:
            use_postgis = env_flag.lower() in {"1", "true", "yes", "on"}
        else:
            use_postgis = bool(os.getenv("SUNTRACE_DATABASE_URI"))

    if database_uri is None:
        database_uri = os.getenv("SUNTRACE_DATABASE_URI")

    if use_postgis:
        uri = database_uri or DEFAULT_DB_URI
        try:
            analyzer = GeospatialAnalyzer2(database_uri=uri, layer_table_map=layer_table_map)
            logger.info("Initialized PostGIS analyzer at %s", uri)
            return analyzer
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "GeospatialAnalyzer2 init failed (%s); falling back to file-based analyzer",
                exc,
            )

    # Fallback: construct GeoPandas-based analyzer from file paths
    analyzer = GeospatialAnalyzer(
        buildings_path=_resolve_data_path(bpath, paths.BUILDINGS_PATH),
        tile_stats_path=_resolve_data_path(tpath, paths.TILE_STATS_PATH),
        plain_tiles_path=_resolve_data_path(ppath, paths.PLAIN_TILES_PATH),
        candidate_minigrids_path=_resolve_data_path(cmpath, paths.CANDIDATE_MINIGRIDS_PATH),
        existing_minigrids_path=_resolve_data_path(empath, paths.EXISTING_MINIGRIDS_PATH),
        existing_grid_path=_resolve_data_path(egpath, paths.EXISTING_GRID_PATH),
        grid_extension_path=_resolve_data_path(gepath, paths.GRID_EXTENSION_PATH),
        roads_path=_resolve_data_path(rpath, paths.ROADS_PATH),
        villages_path=_resolve_data_path(vpath, paths.VILLAGES_PATH),
        parishes_path=_resolve_data_path(papath, paths.PARISHES_PATH),
        subcounties_path=_resolve_data_path(spath, paths.SUBCOUNTIES_PATH),
    )
    logger.info("Initialized file-based analyzer (GeoPandas)")
    return analyzer
