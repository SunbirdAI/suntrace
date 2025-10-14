# Lamwo Composite Pipeline

This folder contains the Jupyter notebooks used to build the Lamwo district monthly Sentinel-2 composites and related biomass statistics. The notebooks are meant to be run in Google Colab with access to the Sunbird shared Google Drive and Google Earth Engine (EE).

## Prerequisites
- Google account with access to the `Sunbird AI/Projects` shared drive.
- Google Colab runtime with GPU/CPU high-RAM recommended.
- Earth Engine account authorized for `ee-isekalala` (adjust if needed).
- Python packages installed within the Colab session, including `pystac-client`, `rio-tiler`, `boto3`, `python-dateutil`, `geopandas`, `shapely`, `rasterio`, `earthengine-api`, `rasterstats`, and plotting libraries.
- Expected directory structure in Drive:
  - `/content/drive/Shareddrives/Sunbird AI/Projects/suntrace/suntrace-multimodal/data` for outputs.
  - Administrative boundaries in `/content/drive/Shareddrives/Sunbird AI/Projects/GIZ Mini-grid Identification/Phase II/Data/administrative areas/`.

## Recommended Run Order
1. **`Pull S2 imagery for Lamwo_monthly composites.ipynb`**  
   Creates 1 km tile grid, pulls monthly Sentinel-2 L2A composites from AWS STAC, and saves per-tile `npy` cubes plus the grid GeoJSON to Drive.

2. **`Precomputed Aggregates.ipynb`**  
   Authenticates with EE, maps tile geometries over various datasets (NDVI, elevation, climate, buildings), and exports the per-tile statistics as CSV/GeoJSON.

3. **`get_lwb.ipynb`**  
   Downloads Aboveground Live Woody Biomass rasters per tile from ArcGIS, clips them to the Lamwo AOI, and stores GeoTIFFs locally in the Colab session.

4. **`merge_lwb_tile_stats.ipynb`**  
   Loads the biomass rasters and EE tile statistics, computes total biomass per tile, and merges results into GeoJSON/CSV artifacts ready for downstream analysis.

## Inputs and Outputs (by step)
- `Pull S2 imagery …`
  - Inputs: Lamwo AOI GeoPackage, Sentinel-2 STAC (`earth-search.aws.element84.com`).
  - Outputs: `grid.geojson`, monthly `.npy` files per tile/year in Drive.
- `Precomputed Aggregates`
  - Inputs: Tile grid (`grid.geojson`), EE datasets (S2_SR, SRTM, PAR, CHIRPS, VIIRS, building layers).
  - Outputs: EE Drive export (CSV) or downloaded GeoJSON with tile-level statistics.
- `get_lwb`
  - Inputs: Lamwo AOI, ArcGIS feature service `Aboveground_Live_Woody_Biomass_Density`.
  - Outputs: Downloaded `.tif` rasters and clipped versions for intersecting tiles.
- `merge_lwb_tile_stats`
  - Inputs: Biomass raster (`aglwb.tif` or clipped stack), tile grid GeoJSON, tile stats CSV.
  - Outputs: Merged GeoJSON/CSV with biomass totals and calculated metrics per tile.

## Operational Tips
- Verify Drive paths at the top of each notebook; update to your Drive mount if different.
- When running EE exports, monitor the EE Tasks tab for completion before proceeding.
- `merge_lwb_tile_stats.ipynb` requires selecting the correct biomass raster unit (`Mg_per_pixel` vs `Mg_per_ha`). Check dataset metadata before merging.
- Large raster downloads may time out; re-run the `get_lwb` download cell for failed tiles.
- Keep secrets (API keys, service accounts) out of notebooks; rely on EE auth prompts in Colab.

## Troubleshooting Checklist
- Missing imports: re-run the initial install cells or add `!pip install` for any new packages.
- CRS mismatches: the merge notebook reprojects to UTM 36N; ensure custom rasters share this CRS or adjust the code.
- Drive quota: exported artifacts land in `ee_exports` or the `data` folder—clean up old runs to stay within limits.
