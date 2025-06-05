# Create a factory function in a new file (e.g., src/utils/factory.py)
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.GeospatialAnalyzer import GeospatialAnalyzer

# Add the project root (one level up) to the Python path for configs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from configs.paths import BUILDINGS_PATH, MINIGRIDS_PATH, TILE_STATS_PATH, PLAIN_TILES_PATH


def create_geospatial_analyzer(bpath=None, mpath=None, tpath=None, ppath=None):
    buildings_path = os.path.join(os.path.dirname(__file__), "../../data/" + bpath) if bpath else BUILDINGS_PATH
    minigrids_path = os.path.join(os.path.dirname(__file__), "../../data/" + mpath) if mpath else MINIGRIDS_PATH
    tile_stats_path = os.path.join(os.path.dirname(__file__), "../../data/" + tpath) if tpath else TILE_STATS_PATH
    plain_tiles_path = os.path.join(os.path.dirname(__file__), "../../data/" + ppath) if ppath else PLAIN_TILES_PATH
    
    return GeospatialAnalyzer(
        buildings_path=buildings_path,
        minigrids_path=minigrids_path,
        tile_stats_path=tile_stats_path,
        plain_tiles_path=plain_tiles_path
    )