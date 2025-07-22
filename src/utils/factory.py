# Create a factory function in a new file (e.g., src/utils/factory.py)
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.GeospatialAnalyzer import GeospatialAnalyzer

# Add the project root (one level up) to the Python path for configs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from configs import paths

def create_geospatial_analyzer(bpath=None, mpath=None, tpath=None, ppath=None, cmpath=None, empath=None, egpath=None, gepath=None, rpath=None, vdir=None, vpath=None, papath=None, spath=None, sr=None):
    buildings_path = os.path.join(os.path.dirname(__file__), "../../data/" + bpath) if bpath else paths.BUILDINGS_PATH
    tile_stats_path = os.path.join(os.path.dirname(__file__), "../../data/" + tpath) if tpath else paths.TILE_STATS_PATH
    plain_tiles_path = os.path.join(os.path.dirname(__file__), "../../data/" + ppath) if ppath else paths.PLAIN_TILES_PATH
    candidate_minigrids_path = os.path.join(os.path.dirname(__file__), "../../data/" + cmpath) if cmpath else paths.CANDIDATE_MINIGRIDS_PATH
    existing_minigrids_path = os.path.join(os.path.dirname(__file__), "../../data/" + empath) if empath else paths.EXISTING_MINIGRIDS_PATH
    existing_grid_path = os.path.join(os.path.dirname(__file__), "../../data/" + egpath) if egpath else paths.EXISTING_GRID_PATH
    grid_extension_path = os.path.join(os.path.dirname(__file__), "../../data/" + gepath) if gepath else paths.GRID_EXTENSION_PATH
    roads_path = os.path.join(os.path.dirname(__file__), "../../data/" + rpath) if rpath else paths.ROADS_PATH
    villages_path = os.path.join(os.path.dirname(__file__), "../../data/" + vpath) if vpath else paths.VILLAGES_PATH
    parishes_path = os.path.join(os.path.dirname(__file__), "../../data/" + papath) if papath else paths.PARISHES_PATH
    subcounties_path = os.path.join(os.path.dirname(__file__), "../../data/" + spath) if spath else paths.SUBCOUNTIES_PATH
    
    

    return GeospatialAnalyzer(
        buildings_path=buildings_path,
        tile_stats_path=tile_stats_path,
        plain_tiles_path=plain_tiles_path,
        candidate_minigrids_path=candidate_minigrids_path,
        existing_minigrids_path=existing_minigrids_path,
        existing_grid_path= existing_grid_path,
        grid_extension_path=grid_extension_path,
        roads_path= roads_path,
        villages_path=villages_path,
        parishes_path=parishes_path,
        subcounties_path= subcounties_path,

    )