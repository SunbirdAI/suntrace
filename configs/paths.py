import os
from pathlib import Path

# Define project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
TILE_STATS_PATH = DATA_DIR / "Lamwo_Tile_Stats_EE.csv"
PLAIN_TILES_PATH = DATA_DIR / "lamwo_sentinel_composites" / "lamwo_grid.geojson"

VIZUALIZATION_DIR = DATA_DIR / "viz_geojsons"
BUILDINGS_PATH = VIZUALIZATION_DIR / "lamwo_buildings.geojson"
CANDIDATE_MINIGRIDS_PATH = VIZUALIZATION_DIR / "candidate_minigrids.geojson"
EXISTING_MINIGRIDS_PATH = VIZUALIZATION_DIR / "existing_minigrids.geojson"
EXISTING_GRID_PATH = VIZUALIZATION_DIR / "existing_grid.geojson"
GRID_EXTENSION_PATH = VIZUALIZATION_DIR / "grid_extension.geojson"
ROADS_PATH = VIZUALIZATION_DIR / "lamwo_roads.geojson"


#Administrative boundaries
VILLAGES_PATH = VIZUALIZATION_DIR / "lamwo_villages.geojson"
PARISHES_PATH = VIZUALIZATION_DIR / "lamwo_parishes.geojson"
SUBCOUNTIES_PATH = VIZUALIZATION_DIR / "lamwo_subcounties.geojson"

#sample for testing
SAMPLE_REGION_PATH = DATA_DIR / "sample_region_mudu" / "mudu_village.gpkg"

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Model directories
# MODELS_DIR = PROJECT_ROOT / "src" / "models"
# CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Output directories
#OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
#LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Example usage:
# from configs.paths import RAW_DATA_DIR
# with open(os.path.join(RAW_DATA_DIR, 'file.csv')) as f:
#     ...

# Add more paths as needed below