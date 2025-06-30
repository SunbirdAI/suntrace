# Makefile for Suntrace GeospatialAnalyzer project

.PHONY: help test test-fast test-coverage test-integration test-all clean lint format install dev-install

# Default target
help:
	@echo "Available targets:"
	@echo "  help           - Show this help message"
	@echo "  install        - Install project dependencies"
	@echo "  dev-install    - Install development dependencies"
	@echo "  test           - Run basic test suite"
	@echo "  test-fast      - Run fast tests only"
	@echo "  test-coverage  - Run tests with coverage report"
	@echo "  test-integration - Run integration tests"
	@echo "  test-all       - Run all tests including slow ones"
	@echo "  lint           - Run code linting"
	@echo "  format         - Format code with black and isort"
	@echo "  clean          - Clean up generated files"
	@echo "  jupyter        - Start Jupyter lab"

# Installation targets
install:
	pip install -r requirements.txt

dev-install: install
	pip install -e .

# Test targets
test:
	python -m pytest tests/ -v -m "not slow and not integration"

test-fast:
	python -m pytest tests/ -q -m "not slow and not integration"

test-coverage:
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing -m "not slow and not integration"

test-integration:
	python -m pytest tests/ -v -m "integration" --tb=short

test-all:
	python -m pytest tests/ -v --tb=short

# Code quality targets
lint:
	@echo "Running flake8..."
	flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	@echo "Running black check..."
	black --check src/ tests/
	@echo "Running isort check..."
	isort --check-only src/ tests/

format:
	@echo "Formatting with black..."
	black src/ tests/
	@echo "Sorting imports with isort..."
	isort src/ tests/

# Development targets
jupyter:
	jupyter lab --no-browser --ip=0.0.0.0

# Cleanup targets
clean:
	@echo "Cleaning up generated files..."
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	@echo "Cleanup complete!"

# Data validation (optional - if you want to check data files)
check-data:
	@echo "Checking for required data files..."
	@if [ -f "data/lamwo_buildings_V3.gpkg" ]; then echo "✓ Buildings data found"; else echo "✗ Buildings data missing"; fi
	@if [ -f "data/updated_candidate_minigrids_merged.gpkg" ]; then echo "✓ Minigrids data found"; else echo "✗ Minigrids data missing"; fi
	@if [ -f "data/Lamwo_Tile_Stats_EE.csv" ]; then echo "✓ Tile stats data found"; else echo "✗ Tile stats data missing"; fi
	@if [ -f "data/lamwo_sentinel_composites/lamwo_grid.geojson" ]; then echo "✓ Tiles geometry data found"; else echo "✗ Tiles geometry data missing"; fi

# Quick development workflow
dev-check: format lint test-fast
	@echo "Development checks complete!"

# CI/CD workflow (what you might run in continuous integration)
ci: lint test-coverage
	@echo "CI checks complete!"
