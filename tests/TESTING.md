# Testing Setup Guide

This document explains how to set up and run tests for the Suntrace GeospatialAnalyzer project.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run basic tests:**
   ```bash
   make test
   # or
   ./run_tests.sh
   ```

3. **Run tests with coverage:**
   ```bash
   make test-coverage
   # or
   ./run_tests.sh -c
   ```

## Test Structure

The test suite is organized into several files:

- `tests/test_geospatial_analyzer.py` - Main test suite for GeospatialAnalyzer
- `tests/test_utilities.py` - Unit tests for utility methods
- `tests/test_integration.py` - Integration tests requiring actual data files
- `tests/conftest.py` - Shared test fixtures and configuration

## Test Categories

Tests are marked with different categories:

- `@pytest.mark.unit` - Fast unit tests that don't require data files
- `@pytest.mark.integration` - Tests that require actual data files
- `@pytest.mark.geospatial` - Tests that work with geospatial data
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.visualization` - Tests that create visual outputs

## Running Different Test Types

### Basic Tests (Fast)
```bash
make test
# or
./run_tests.sh -f
```

### All Tests (Including Slow)
```bash
make test-all
# or
./run_tests.sh -s -i
```

### Integration Tests Only
```bash
make test-integration
# or
./run_tests.sh -i
```

### With Coverage Report
```bash
make test-coverage
# or
./run_tests.sh -c
```

### Parallel Execution
```bash
./run_tests.sh -p
```

## Test Configuration

Tests are configured via `pyproject.toml`:

- Test discovery patterns
- Coverage settings
- Test markers
- Black and isort configuration

## Data Requirements

Integration tests require data files in the `data/` directory:

- `data/lamwo_buildings_V3.gpkg` - Building polygons
- `data/updated_candidate_minigrids_merged.gpkg` - Minigrid locations
- `data/Lamwo_Tile_Stats_EE.csv` - Tile statistics
- `data/lamwo_sentinel_composites/lamwo_grid.geojson` - Tile geometries
- `data/sample_region_mudu/mudu_village.gpkg` - Sample test region

You can check if data files are present:
```bash
make check-data
```

## Writing New Tests

### Test Class Structure
```python
class TestYourFeature:
    """Test suite for YourFeature."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return create_test_data()
    
    @pytest.mark.unit
    def test_basic_functionality(self, sample_data):
        """Test basic functionality."""
        assert sample_data is not None
        
    @pytest.mark.geospatial
    def test_geospatial_operation(self, analyzer, sample_region):
        """Test geospatial operations."""
        result = analyzer.some_geospatial_method(sample_region)
        assert result is not None
```

### Test Fixtures

Common fixtures available in `conftest.py`:

- `project_root_path` - Path to project root
- `data_dir_path` - Path to data directory
- `sample_data_paths` - Dictionary of data file paths
- `check_data_files` - Ensures required data files exist

### Mocking External Dependencies

For tests that don't need real data:

```python
from unittest.mock import Mock, patch

@pytest.fixture
def mock_analyzer(self):
    """Create analyzer with mocked data."""
    with patch('utils.GeospatialAnalyzer.gpd.read_file') as mock_read:
        mock_read.return_value = gpd.GeoDataFrame(...)
        return GeospatialAnalyzer(...)
```

## Continuous Integration

For CI/CD pipelines, use:
```bash
make ci
```

This runs linting and tests with coverage.

## Troubleshooting

### Common Issues

1. **Import Errors:**
   - Ensure you're running tests from the project root
   - Check that `src/` is in Python path (handled by conftest.py)

2. **Missing Data Files:**
   - Integration tests will be skipped if data files are missing
   - Use `make check-data` to verify data file presence

3. **Slow Tests:**
   - Use `-f` flag to run only fast tests
   - Mark long-running tests with `@pytest.mark.slow`

4. **Memory Issues:**
   - Large geospatial data can consume significant memory
   - Consider using smaller test datasets or mocking

### Performance Tips

- Use `pytest-xdist` for parallel execution: `./run_tests.sh -p`
- Skip slow tests during development: `./run_tests.sh -f`
- Use specific test files: `pytest tests/test_utilities.py`
- Use test selection: `pytest -k "test_count_buildings"`

## Code Coverage

Coverage reports are generated in `htmlcov/` directory when using the `-c` flag.

Target coverage goals:
- Overall: > 80%
- Critical functions: > 90%
- New features: 100%

## Test Data Management

For development, you may want to:

1. Create smaller test datasets
2. Use synthetic data for unit tests
3. Mock external data sources
4. Version control test data separately

## Integration with VS Code

The project includes VS Code configuration for:
- Running tests from the editor
- Debugging test failures
- Coverage visualization

Use the Test Explorer in VS Code or run tests via the command palette.

## Best Practices

1. **Test Organization:**
   - Group related tests in classes
   - Use descriptive test names
   - Follow the Arrange-Act-Assert pattern

2. **Test Data:**
   - Use fixtures for reusable test data
   - Mock external dependencies
   - Clean up after tests

3. **Assertions:**
   - Use specific assertions (`assert isinstance(result, int)`)
   - Test edge cases and error conditions
   - Verify both positive and negative cases

4. **Performance:**
   - Mark slow tests appropriately
   - Use sampling for large datasets
   - Profile test execution when needed
