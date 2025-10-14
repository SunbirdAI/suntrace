# Lamwo Data Pipeline Scripts

These CLI utilities mirror the Lamwo notebook workflow so teammates can refresh inputs without opening Colab. Update the placeholder paths/project IDs before running.

1. `generate_tile_grid.py` — Build a 1 km grid from an AOI GeoPackage. Outputs a GeoJSON grid (`lamwo_grid.geojson`).
2. `download_s2_monthly.py` — Download monthly Sentinel-2 composites per tile as `.npy` cubes using Earth Engine.
3. `export_tile_stats.py` — Start an Earth Engine export (Drive or direct download) with vegetation, terrain, climate, and building stats per tile.
4. `download_lwb_rasters.py` — Fetch Aboveground Live Woody Biomass rasters per tile from the ArcGIS FeatureServer and clip them to the AOI.
5. `merge_lwb_tile_stats.py` — Combine biomass totals with Earth Engine tile statistics and save merged GeoJSON/CSV outputs.

Typical run order follows the notebook sequence: generate the grid → (optional) fetch Sentinel-2 composites → export the EE aggregates → download biomass rasters → merge outputs.
