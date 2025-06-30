#!/bin/bash

# Test runner script for Suntrace project
# Usage: ./run_tests.sh [options]

set -e

# Default values
COVERAGE=false
INTEGRATION=false
SLOW=false
VERBOSE=false
PARALLEL=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Run pytest tests for the Suntrace GeospatialAnalyzer project.

OPTIONS:
    -h, --help          Show this help message
    -c, --coverage      Run tests with coverage report
    -i, --integration   Run integration tests (requires data files)
    -s, --slow          Include slow tests
    -v, --verbose       Verbose output
    -p, --parallel      Run tests in parallel
    -u, --unit          Run only unit tests
    -f, --fast          Run only fast tests (exclude slow and integration)
    
EXAMPLES:
    $0                  # Run basic test suite
    $0 -c               # Run with coverage
    $0 -i -s            # Run all tests including slow and integration
    $0 -f               # Run only fast unit tests
    $0 -v -c            # Verbose output with coverage

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -i|--integration)
            INTEGRATION=true
            shift
            ;;
        -s|--slow)
            SLOW=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -u|--unit)
            UNIT_ONLY=true
            shift
            ;;
        -f|--fast)
            FAST_ONLY=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if we're in the project root
if [[ ! -f "pyproject.toml" ]] || [[ ! -d "tests" ]]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Build pytest command
PYTEST_CMD="python -m pytest"

# Add basic options
if [[ "$VERBOSE" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD -v"
else
    PYTEST_CMD="$PYTEST_CMD -q"
fi

# Add parallel execution if requested
if [[ "$PARALLEL" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD -n auto"
fi

# Add coverage if requested
if [[ "$COVERAGE" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src --cov-report=html --cov-report=term-missing"
fi

# Handle test selection
if [[ "$FAST_ONLY" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD -m 'not slow and not integration'"
    print_status "Running fast tests only"
elif [[ "$UNIT_ONLY" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD -m 'unit or not integration'"
    print_status "Running unit tests only"
else
    # Build marker expression
    MARKERS=""
    
    if [[ "$INTEGRATION" == false ]]; then
        if [[ -n "$MARKERS" ]]; then
            MARKERS="$MARKERS and not integration"
        else
            MARKERS="not integration"
        fi
    fi
    
    if [[ "$SLOW" == false ]]; then
        if [[ -n "$MARKERS" ]]; then
            MARKERS="$MARKERS and not slow"
        else
            MARKERS="not slow"
        fi
    fi
    
    if [[ -n "$MARKERS" ]]; then
        PYTEST_CMD="$PYTEST_CMD -m '$MARKERS'"
    fi
fi

# Add test directory
PYTEST_CMD="$PYTEST_CMD tests/"

print_status "Starting test execution..."
print_status "Command: $PYTEST_CMD"

# Check for required dependencies
print_status "Checking dependencies..."
if ! python -c "import pytest" 2>/dev/null; then
    print_error "pytest not found. Please install requirements: pip install -r requirements.txt"
    exit 1
fi

# Check for data files if running integration tests
if [[ "$INTEGRATION" == true ]] || [[ "$FAST_ONLY" != true && "$UNIT_ONLY" != true ]]; then
    print_status "Checking for data files..."
    if [[ ! -d "data" ]]; then
        print_warning "Data directory not found. Integration tests may be skipped."
    fi
fi

# Run the tests
print_status "Running tests..."
if eval $PYTEST_CMD; then
    print_success "All tests passed!"
    
    # Show coverage report if requested
    if [[ "$COVERAGE" == true ]]; then
        print_status "Coverage report generated in htmlcov/"
        if command -v open >/dev/null 2>&1; then
            print_status "Opening coverage report in browser..."
            open htmlcov/index.html
        fi
    fi
    
    exit 0
else
    print_error "Some tests failed!"
    exit 1
fi
