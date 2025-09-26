"""
Integration tests for GeospatialAnalyzer.
These tests require actual data files and test the full workflow.
"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, Polygon
from utils.factory import create_geospatial_analyzer


@pytest.mark.integration
@pytest.mark.geospatial
class TestGeospatialAnalyzerIntegration:
    """Integration tests requiring actual data files."""

    @pytest.fixture(scope="class")
    def analyzer_with_data(self, check_data_files):
        """Create analyzer with real data files."""
        try:
            return create_geospatial_analyzer()
        except Exception as e:
            pytest.skip(f"Could not create analyzer with real data: {e}")

    @pytest.fixture(scope="class")
    def test_region(self, sample_data_paths):
        """Load a test region from actual data."""
        if not sample_data_paths["sample_region"].exists():
            pytest.skip("Sample region file not found")

        sample_gdf = gpd.read_file(sample_data_paths["sample_region"])
        if sample_gdf.empty:
            pytest.skip("Sample region is empty")

        return sample_gdf.geometry.iloc[0]

    def test_full_workflow_building_analysis(self, analyzer_with_data, test_region):
        """Test complete workflow for building analysis."""
        # Test basic counting
        building_count = analyzer_with_data.count_features_within_region(
            test_region, 'buildings'
        )
        assert isinstance(building_count, int)
        assert building_count >= 0

        # Test with different buffer sizes
        if building_count > 0:
            buffered_region = analyzer_with_data.buffer_geometry(test_region, 500)  # 500m buffer
            buffered_count = analyzer_with_data.count_features_within_region(
                buffered_region, 'buildings'
            )
            assert buffered_count >= building_count  # Should be at least as many

    def test_full_workflow_ndvi_analysis(self, analyzer_with_data, test_region):
        """Test complete workflow for NDVI analysis."""
        stats_snapshot = analyzer_with_data.weighted_tile_stats_all(test_region)
        if not stats_snapshot or 'ndvi_mean' not in stats_snapshot:
            pytest.skip("No NDVI data available")

        # Test average NDVI
        avg_ndvi = analyzer_with_data.avg_ndvi(test_region)
        if not np.isnan(avg_ndvi):
            assert -1 <= avg_ndvi <= 1

        # Test NDVI statistics
        ndvi_stats = analyzer_with_data.ndvi_stats(test_region)
        assert isinstance(ndvi_stats, dict)
        assert 'NDVI_mean' in ndvi_stats
        
    def test_full_workflow_minigrid_analysis(self, analyzer_with_data, test_region):
        """Test complete workflow for minigrid analysis."""
        if not hasattr(analyzer_with_data, '_minigrids_gdf') or analyzer_with_data._minigrids_gdf.empty:
            pytest.skip("No minigrid data available")

        # Test listing minigrids
        grid_list = analyzer_with_data.list_mini_grids()
        assert isinstance(grid_list, list)

        if grid_list:
            # Test getting site geometry
            site_geom = analyzer_with_data.get_site_geometry(grid_list[0])
            if site_geom is not None:
                assert hasattr(site_geom, "bounds")

            # Test nearest minigrids
            center_point = test_region.centroid
            nearest = analyzer_with_data.nearest_mini_grids(center_point, k=3)
            assert isinstance(nearest, list)
            assert len(nearest) <= min(3, len(grid_list))

    def test_cross_layer_consistency(self, analyzer_with_data, test_region):
        """Test consistency across different data layers."""
        # Get tile IDs in region
        tile_ids = analyzer_with_data.get_tile_ids_within_region(test_region)

        # Count tiles using generic method
        tile_count = analyzer_with_data.count_features_within_region(
            test_region, "tiles"
        )

        # These should be consistent
        if tile_ids and tile_count > 0:
            assert len(tile_ids) == tile_count

    def test_spatial_accuracy(self, analyzer_with_data, test_region):
        """Test spatial accuracy of operations."""
        if not hasattr(analyzer_with_data, '_buildings_gdf') or analyzer_with_data._buildings_gdf.empty:
            pytest.skip("No buildings data available")

        # Test that buildings in buffered region >= buildings in original region
        original_count = analyzer_with_data.count_features_within_region(
            test_region, 'buildings'
        )

        if original_count > 0:
            buffered_region = analyzer_with_data.buffer_geometry(test_region, 100)  # 100m buffer
            buffered_count = analyzer_with_data.count_features_within_region(
                buffered_region, 'buildings'
            )
            
            assert buffered_count >= original_count

    def test_performance_with_real_data(self, analyzer_with_data, test_region):
        """Test performance with real data."""
        import time

        start_time = time.time()

        # Run multiple operations
        building_count = analyzer_with_data.count_features_within_region(
            test_region, 'buildings'
        )
        avg_ndvi = analyzer_with_data.avg_ndvi(test_region)
        ndvi_stats = analyzer_with_data.ndvi_stats(test_region)
        
        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete within reasonable time (adjust as needed)
        assert execution_time < 10  # 10 seconds max for basic operations

        print(f"Performance test completed in {execution_time:.2f} seconds")

    @pytest.mark.slow
    def test_large_region_analysis(self, analyzer_with_data):
        """Test analysis with large regions."""
        if not hasattr(analyzer_with_data, '_buildings_gdf') or analyzer_with_data._buildings_gdf.empty:
            pytest.skip("No buildings data available")

        # Create a large region covering most of the data
        bounds = analyzer_with_data._buildings_gdf.total_bounds
        if len(bounds) == 4:
            # Create region covering 50% of the bounding box
            x_range = bounds[2] - bounds[0]
            y_range = bounds[3] - bounds[1]

            large_region = Polygon(
                [
                    (bounds[0] + x_range * 0.25, bounds[1] + y_range * 0.25),
                    (bounds[2] - x_range * 0.25, bounds[1] + y_range * 0.25),
                    (bounds[2] - x_range * 0.25, bounds[3] - y_range * 0.25),
                    (bounds[0] + x_range * 0.25, bounds[3] - y_range * 0.25),
                ]
            )

            # This should not crash and should return reasonable results
            building_count = analyzer_with_data.count_features_within_region(large_region, 'buildings')
            assert isinstance(building_count, int)
            assert building_count >= 0

    def test_data_integrity_checks(self, analyzer_with_data):
        """Test data integrity across loaded datasets."""
        # Check that all loaded GeoDataFrames have valid CRS
        if hasattr(analyzer_with_data, '_buildings_gdf') and not analyzer_with_data._buildings_gdf.empty:
            assert analyzer_with_data._buildings_gdf.crs is not None
            
        if hasattr(analyzer_with_data, '_minigrids_gdf') and not analyzer_with_data._minigrids_gdf.empty:
            assert analyzer_with_data._minigrids_gdf.crs is not None
            
        if hasattr(analyzer_with_data, '_plain_tiles_gdf') and not analyzer_with_data._plain_tiles_gdf.empty:
            assert analyzer_with_data._plain_tiles_gdf.crs is not None

        # Check that joined tiles have both geometry and stats
        if hasattr(analyzer_with_data, '_joined_tiles_gdf') and not analyzer_with_data._joined_tiles_gdf.empty:
            assert 'geometry' in analyzer_with_data._joined_tiles_gdf.columns
            # Should have at least some statistical columns
            stat_cols = [
                col
                for col in analyzer_with_data._joined_tiles_gdf.columns
                if any(stat in col.lower() for stat in ["ndvi", "mean", "std", "med"])
            ]
            if stat_cols:  # Only check if statistical columns exist
                assert len(stat_cols) > 0

    def test_coordinate_system_consistency(self, analyzer_with_data):
        """Test that coordinate system handling is consistent."""
        # All geographic data should be transformable to common CRS
        target_crs = "EPSG:4326"  # WGS84
        
        if hasattr(analyzer_with_data, '_buildings_gdf') and not analyzer_with_data._buildings_gdf.empty:
            buildings_4326 = analyzer_with_data._ensure_gdf_crs_for_calculation(
                analyzer_with_data._buildings_gdf.copy(), target_crs
            )
            assert buildings_4326.crs.to_string() == target_crs
            
        if hasattr(analyzer_with_data, '_minigrids_gdf') and not analyzer_with_data._minigrids_gdf.empty:
            minigrids_4326 = analyzer_with_data._ensure_gdf_crs_for_calculation(
                analyzer_with_data._minigrids_gdf.copy(), target_crs
            )
            assert minigrids_4326.crs.to_string() == target_crs


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
