import os
import sys
import pytest
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Add project root to Python path for configs
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root path."""
    return project_root

@pytest.fixture(scope="session")
def data_dir_path(project_root_path):
    """Return the data directory path."""
    return project_root_path / "data"

@pytest.fixture(scope="session")
def sample_data_paths(data_dir_path):
    """Return paths to sample data files."""
    return {
        "buildings": data_dir_path / "lamwo_buildings_V3.gpkg",
        "minigrids": data_dir_path / "updated_candidate_minigrids_merged.gpkg",
        "tile_stats": data_dir_path / "Lamwo_Tile_Stats_EE.csv",
        "plain_tiles": data_dir_path / "lamwo_sentinel_composites" / "lamwo_grid.geojson",
        "sample_region": data_dir_path / "sample_region_mudu" / "mudu_village.gpkg",
    }

@pytest.fixture(scope="session")
def check_data_files(sample_data_paths):
    """Check if required data files exist and skip tests if not."""
    missing_files = []
    for name, path in sample_data_paths.items():
        if not path.exists():
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        pytest.skip(f"Missing data files: {', '.join(missing_files)}")
    
    return sample_data_paths
