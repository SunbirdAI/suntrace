import sys
import os
import geopandas as gpd
from shapely.geometry import Polygon, Point

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils.factory import create_geospatial_analyzer

# Add the project root (one level up) to the Python path for configs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs.paths import SAMPLE_REGION_PATH


# Define a sample region for testing (you may need to adjust coordinates)
# These coordinates are arbitrary and likely outside your actual data extent.
# You should use coordinates that are relevant to your data for meaningful tests.
# using shapely open shapefile from path data/sample_region_mudu/5502888.shp

# Load the sample polygon from the shapefile

sample_gdf = gpd.read_file(SAMPLE_REGION_PATH)
if sample_gdf.empty:
    raise ValueError(f"Sample region shapefile at {SAMPLE_REGION_PATH} is empty or failed to load.")
sample_polygon = sample_gdf.geometry.iloc[0]

def run_tests():
    print("Initializing GeospatialAnalyzer...")
    try:
        analyzer = create_geospatial_analyzer()
        print("GeospatialAnalyzer initialized successfully.")
    except Exception as e:
        print(f"Error initializing GeospatialAnalyzer: {e}")
        return

    # --- Test Fundamental Helper Methods (implicitly tested during init, but we can add specific checks) ---
    print("\n--- Testing Fundamental Helper Methods ---")

    # 1. _load_and_validate_gdf (checked by successful initialization)
    print("\n1. _load_and_validate_gdf:")
    if not analyzer._buildings_gdf.empty:
        print(f"  Buildings loaded: {len(analyzer._buildings_gdf)} features. CRS: {analyzer._buildings_gdf.crs}")
    else:
        print("  Buildings GDF is empty or failed to load.")

    if not analyzer._minigrids_gdf.empty:
        print(f"  Minigrids loaded: {len(analyzer._minigrids_gdf)} features. CRS: {analyzer._minigrids_gdf.crs}")
    else:
        print("  Minigrids GDF is empty or failed to load.")

    if not analyzer._plain_tiles_gdf.empty:
        print(f"  Plain tiles loaded: {len(analyzer._plain_tiles_gdf)} features. CRS: {analyzer._plain_tiles_gdf.crs}")
    else:
        print("  Plain tiles GDF is empty or failed to load.")

    # 2. _load_and_process_tile_stats (checked by successful initialization)
    print("\n2. _load_and_process_tile_stats:")
    if not analyzer._tile_stats_gdf.empty:
        print(f"  Tile stats loaded: {len(analyzer._tile_stats_gdf)} records.")
        if 'ndvi_mean' in analyzer._tile_stats_gdf.columns:
            print(f"  'ndvi_mean' column found in tile_stats_gdf.")
        else:
            print(f"  Warning: 'ndvi_mean' column NOT found in tile_stats_gdf.")
    else:
        print("  Tile stats GDF is empty or failed to load.")

    # 3. _merge_tile_data (checked by successful initialization)
    print("\n3. _merge_tile_data:")
    if not analyzer._joined_tiles_gdf.empty:
        print(f"  Joined tiles created: {len(analyzer._joined_tiles_gdf)} features. CRS: {analyzer._joined_tiles_gdf.crs}")
        if 'ndvi_mean' in analyzer._joined_tiles_gdf.columns and 'geometry' in analyzer._joined_tiles_gdf.columns:
            print(f"  'ndvi_mean' and 'geometry' columns found in joined_tiles_gdf.")
        else:
            print(f"  Warning: 'ndvi_mean' or 'geometry' column NOT found in joined_tiles_gdf.")
    else:
        print("  Joined tiles GDF is empty or failed to merge.")

    # 4. _ensure_gdf_crs_for_calculation
    print("\n4. _ensure_gdf_crs_for_calculation:")
    if not analyzer._buildings_gdf.empty:
        try:
            buildings_metric = analyzer._check_and_reproject_gdf(analyzer._buildings_gdf.copy(), analyzer.target_metric_crs)
            print(f"  Buildings GDF reprojected/ensured to {analyzer.target_metric_crs}: {buildings_metric.crs}")
            buildings_geo = analyzer._check_and_reproject_gdf(analyzer._buildings_gdf.copy(), analyzer.target_geographic_crs)
            print(f"  Buildings GDF reprojected/ensured to {analyzer.target_geographic_crs}: {buildings_geo.crs}")
        except Exception as e:
            print(f"  Error testing _ensure_gdf_crs_for_calculation: {e}")
    else:
        print("  Skipping _ensure_gdf_crs_for_calculation test as buildings GDF is empty.")

    # 5. _ensure_crs_for_calculation (Harder to test in isolation without a GeoSeries/CRS-aware geometry)
    #    This helper is typically used with geometries derived from GDFs.
    #    We'll assume its correct functioning if other methods using it work.
    print("\n5. _ensure_crs_for_calculation: (Primarily tested via other methods)")
    if not analyzer._buildings_gdf.empty and analyzer._buildings_gdf.crs:
        try:
            sample_geom = analyzer._buildings_gdf.geometry.iloc[0]
            # Test reprojecting a single geometry
            reprojected_geom, reprojected_flag = analyzer._prepare_geometry_for_crs(sample_geom, analyzer.target_metric_crs)
            print(f"  Sample geometry reprojected to metric CRS (reprojected: {reprojected_flag}). Original CRS was implicitly {analyzer._buildings_gdf.crs}")

            # Test with a geometry that might already be in the target CRS (less likely for initial geographic load)
            # Create a dummy GeoSeries with the target metric CRS
            import geopandas as gpd
            temp_metric_geom = gpd.GeoSeries([Point(1,1)], crs=analyzer.target_metric_crs).iloc[0]
            ensured_geom, reprojected_flag_metric = analyzer._prepare_geometry_for_crs(temp_metric_geom, analyzer.target_metric_crs)
            print(f"  Sample geometry already in metric CRS (reprojected: {reprojected_flag_metric})")

        except Exception as e:
            print(f"  Error testing _ensure_crs_for_calculation: {e}")
    else:
        print("  Skipping _ensure_crs_for_calculation test as buildings GDF is empty or has no CRS.")


    # --- Test Generic vector-counting primitive ---
    print("\n--- Testing Generic vector-counting primitive ---")
    # print(f"Using sample_polygon: {sample_polygon.wkt} (CRS assumed to be {analyzer.target_geographic_crs} for this test polygon)")

    # It's important that sample_polygon has a CRS that matches what count_features_within expects,
    # or that count_features_within correctly handles reprojection.
    # The current implementation of count_features_within reprojects the region to match the GDF's CRS.
    # Let's assume the sample_polygon is in WGS84 (EPSG:4326) for this test.
    # We can make it more robust by creating it with a CRS if GeoPandas is available here.
    try:
        # Create a GeoSeries for the polygon with a defined CRS
        region_gs = gpd.GeoSeries([sample_polygon], crs=analyzer.target_geographic_crs) # Explicitly WGS84
        test_region_polygon = region_gs.iloc[0]
        print(f"Test region polygon CRS for count_features_within: {region_gs.crs}")
    except ImportError:
        print("Geopandas not available for creating CRS-aware test polygon. Using raw Shapely polygon.")
        test_region_polygon = sample_polygon # Fallback

    # 1. Count buildings
    print("\n1. count_features_within (buildings):")
    try:
        building_count = analyzer.count_buildings_within_region(test_region_polygon)
        print(f"  Number of buildings in sample region: {building_count}")
    except Exception as e:
        print(f"  Error counting buildings: {e}")

    # 2. Count minigrids
    print("\n2. count_features_within (minigrids):")
    try:
        minigrid_count = analyzer.count_features_within_region(test_region_polygon, 'minigrids')
        print(f"  Number of minigrids in sample region: {minigrid_count}")
    except Exception as e:
        print(f"  Error counting minigrids: {e}")

    # 3. Count tiles
    print("\n3. count_features_within (tiles):")
    try:
        tile_count = analyzer.count_features_within_region(test_region_polygon, 'tiles')
        print(f"  Number of tiles in sample region: {tile_count}")
    except Exception as e:
        print(f"  Error counting tiles: {e}")

    # 4. Count tiles with a filter (e.g., NDVI_mean > 0.1)
    print("\n4. count_features_within (tiles with filter):")
    if not analyzer._joined_tiles_gdf.empty and 'ndvi_mean' in analyzer._joined_tiles_gdf.columns:
        try:
            # Ensure the filter is valid for your data. This is an example.
            filtered_tile_count = analyzer.count_features_within_region(test_region_polygon, 'tiles', filter_expr="ndvi_mean > 0.1")
            print(f"  Number of tiles with NDVI > 0.1 in sample region: {filtered_tile_count}")
        except Exception as e:
            print(f"  Error counting filtered tiles: {e}")
    else:
        print("  Skipping filtered tile count as joined_tiles_gdf is empty or missing 'ndvi_mean'.")

    # 5. Test with an invalid layer name
    print("\n5. count_features_within (invalid layer):")
    try:
        invalid_count = analyzer.count_features_within_region(test_region_polygon, 'non_existent_layer')
        print(f"  Count for non_existent_layer: {invalid_count} (expected 0 and an error message)")
    except Exception as e:
        print(f"  Error with invalid layer (as expected): {e}")

    # 6. Return tile ID's of tiles within the region
    print("\n6. get_tile_ids_within_region:")
    try:
        tile_ids = analyzer.get_tile_ids_within_region(test_region_polygon)
        print(f"  Tile IDs within sample region: {tile_ids}")
    except Exception as e:
        print(f"  Error getting tile IDs within region: {e}")
        
    print("\n--- Tests Complete ---")

if __name__ == "__main__":

    run_tests()
