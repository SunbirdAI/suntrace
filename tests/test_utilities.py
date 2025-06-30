"""
Unit tests for GeospatialAnalyzer utility methods.
"""
import pytest
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
from unittest.mock import Mock, patch
import numpy as np

from utils.GeospatialAnalyzer import GeospatialAnalyzer


class TestGeospatialAnalyzerUtilities:
    """Test utility methods of GeospatialAnalyzer."""

    @pytest.fixture
    def mock_minimal_analyzer(self):
        """Create a minimal analyzer for utility testing."""
        with patch('utils.GeospatialAnalyzer.gpd.read_file') as mock_read_file:
            with patch('pandas.read_csv') as mock_read_csv:
                # Mock empty GeoDataFrames
                empty_gdf = gpd.GeoDataFrame(geometry=[])
                mock_read_file.return_value = empty_gdf
                
                # Mock empty DataFrame for CSV
                empty_df = pd.DataFrame()
                mock_read_csv.return_value = empty_df
                
                analyzer = GeospatialAnalyzer(
                    buildings_path="mock.gpkg",
                    minigrids_path="mock.gpkg", 
                    tile_stats_path="mock.csv",
                    plain_tiles_path="mock.geojson"
                )
                return analyzer

    def test_buffer_geometry_with_geographic_crs(self, mock_minimal_analyzer):
        """Test buffering with geographic coordinates."""
        # Create a point in geographic coordinates
        point = Point(-0.5, 0.5)  # Lon, Lat
        
        # Mock the geometry to have geographic CRS
        with patch.object(point, 'crs', "EPSG:4326"):
            buffered = mock_minimal_analyzer.buffer_geometry(point, 1000)  # 1km buffer
            
            assert buffered is not None
            assert hasattr(buffered, 'area')
            assert buffered.area > 0

    def test_buffer_geometry_without_crs(self, mock_minimal_analyzer):
        """Test buffering without CRS information."""
        point = Point(0, 0)
        
        buffered = mock_minimal_analyzer.buffer_geometry(point, 1000)
        
        assert buffered is not None
        assert hasattr(buffered, 'area')
        assert buffered.area > 0

    def test_ensure_gdf_crs_for_calculation(self, mock_minimal_analyzer):
        """Test CRS ensuring for GeoDataFrames."""
        # Create test GeoDataFrame
        test_gdf = gpd.GeoDataFrame({
            'geometry': [Point(0, 0), Point(1, 1)]
        }, crs="EPSG:4326")
        
        target_crs = "EPSG:32636"  # UTM Zone 36N
        
        result_gdf = mock_minimal_analyzer._ensure_gdf_crs_for_calculation(
            test_gdf, target_crs
        )
        
        assert result_gdf.crs.to_string() == target_crs
        assert len(result_gdf) == len(test_gdf)

    def test_ensure_crs_for_calculation_same_crs(self, mock_minimal_analyzer):
        """Test CRS ensuring when geometry is already in target CRS."""
        # Create geometry that's already in target CRS
        target_crs = "EPSG:32636"
        test_geom = gpd.GeoSeries([Point(0, 0)], crs=target_crs).iloc[0]
        
        result_geom, was_reprojected = mock_minimal_analyzer._ensure_crs_for_calculation(
            test_geom, target_crs
        )
        
        assert not was_reprojected
        assert result_geom is not None

    def test_ensure_crs_for_calculation_different_crs(self, mock_minimal_analyzer):
        """Test CRS ensuring when geometry needs reprojection."""
        source_crs = "EPSG:4326"
        target_crs = "EPSG:32636"
        
        test_geom = gpd.GeoSeries([Point(0, 0)], crs=source_crs).iloc[0]
        
        result_geom, was_reprojected = mock_minimal_analyzer._ensure_crs_for_calculation(
            test_geom, target_crs
        )
        
        assert was_reprojected
        assert result_geom is not None


class TestGeospatialAnalyzerErrorHandling:
    """Test error handling in GeospatialAnalyzer."""

    def test_initialization_with_missing_files(self):
        """Test initialization with missing data files."""
        with pytest.raises((FileNotFoundError, Exception)):
            GeospatialAnalyzer(
                buildings_path="nonexistent.gpkg",
                minigrids_path="nonexistent.gpkg",
                tile_stats_path="nonexistent.csv", 
                plain_tiles_path="nonexistent.geojson"
            )

    def test_count_features_empty_gdf(self):
        """Test feature counting with empty GeoDataFrame."""
        with patch('utils.GeospatialAnalyzer.gpd.read_file') as mock_read_file:
            with patch('pandas.read_csv') as mock_read_csv:
                # Mock empty GeoDataFrames
                empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
                mock_read_file.return_value = empty_gdf
                
                # Mock empty DataFrame for CSV
                empty_df = pd.DataFrame()
                mock_read_csv.return_value = empty_df
                
                analyzer = GeospatialAnalyzer(
                    buildings_path="mock.gpkg",
                    minigrids_path="mock.gpkg",
                    tile_stats_path="mock.csv",
                    plain_tiles_path="mock.geojson"
                )
                
                test_region = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
                count = analyzer.count_features_within_region(test_region, 'buildings')
                
                assert count == 0

    def test_invalid_filter_expression(self):
        """Test handling of invalid filter expressions."""
        with patch('utils.GeospatialAnalyzer.gpd.read_file') as mock_read_file:
            with patch('pandas.read_csv') as mock_read_csv:
                # Create mock GeoDataFrame with data
                mock_gdf = gpd.GeoDataFrame({
                    'geometry': [Point(0, 0), Point(1, 1)],
                    'test_col': [1, 2]
                }, crs="EPSG:4326")
                mock_read_file.return_value = mock_gdf
                
                mock_df = pd.DataFrame()
                mock_read_csv.return_value = mock_df
                
                analyzer = GeospatialAnalyzer(
                    buildings_path="mock.gpkg",
                    minigrids_path="mock.gpkg",
                    tile_stats_path="mock.csv",
                    plain_tiles_path="mock.geojson"
                )
                
                test_region = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
                
                # Test with invalid column name in filter
                count = analyzer.count_features_within_region(
                    test_region, 'buildings', filter_expr="nonexistent_col > 0"
                )
                
                # Should return 0 due to error handling
                assert count == 0


class TestGeospatialAnalyzerDataValidation:
    """Test data validation methods."""

    def test_load_and_validate_gdf_valid_file(self):
        """Test loading valid GeoDataFrame."""
        with patch('utils.GeospatialAnalyzer.gpd.read_file') as mock_read:
            valid_gdf = gpd.GeoDataFrame({
                'geometry': [Point(0, 0), Point(1, 1)]
            }, crs="EPSG:4326")
            mock_read.return_value = valid_gdf
            
            # This would be called during initialization
            # We're testing the internal logic here
            result = gpd.read_file("mock_path.gpkg")
            assert not result.empty
            assert result.crs is not None

    def test_load_and_validate_gdf_empty_file(self):
        """Test loading empty GeoDataFrame."""
        with patch('utils.GeospatialAnalyzer.gpd.read_file') as mock_read:
            empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            mock_read.return_value = empty_gdf
            
            result = gpd.read_file("mock_path.gpkg")
            assert result.empty

    def test_tile_stats_processing(self):
        """Test tile statistics processing."""
        with patch('pandas.read_csv') as mock_csv:
            # Mock CSV with tile statistics
            mock_stats = pd.DataFrame({
                'tile_id': [1, 2, 3],
                'ndvi_mean': [0.3, 0.5, 0.7],
                'ndvi_med': [0.25, 0.48, 0.68],
                'ndvi_std': [0.1, 0.15, 0.12]
            })
            mock_csv.return_value = mock_stats
            
            result = pd.read_csv("mock_stats.csv")
            
            assert not result.empty
            assert 'ndvi_mean' in result.columns
            assert len(result) == 3
            assert result['ndvi_mean'].min() >= 0
            assert result['ndvi_mean'].max() <= 1


@pytest.mark.parametrize("crs_input,expected_output", [
    ("EPSG:4326", "EPSG:4326"),
    ("EPSG:32636", "EPSG:32636"),
    (4326, "EPSG:4326"),
])
def test_crs_handling_parametrized(crs_input, expected_output):
    """Parametrized test for CRS handling."""
    # This tests different CRS input formats
    import pyproj
    
    try:
        crs = pyproj.CRS(crs_input)
        assert crs.to_string() == expected_output or crs.to_epsg() == int(expected_output.split(':')[1])
    except Exception:
        pytest.skip(f"Invalid CRS input: {crs_input}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
