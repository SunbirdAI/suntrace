# Suntrace GeospatialAnalyzer - Project Analysis

## Overview

**Suntrace** is a multimodal geospatial analysis platform for off-grid electrification planning in Uganda. It integrates GIS data, LLM function calling (via OpenAI and Google Gemini), and interactive web visualization using Leaflet maps with a chat interface.

### Purpose
The platform helps government officials, planners, and development practitioners make informed decisions about rural energy infrastructure, focusing on:
- Solar energy potential assessment
- Mini-grid site evaluation
- Settlement density analysis
- Infrastructure connectivity assessment

---

## Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Frontend (templates/index.html)           │
│  ┌─────────────────┐           ┌────────────────────────────────┐│
│  │   Leaflet Map   │           │       Chat Interface          ││
│  │  - Draw regions │           │  - Natural language queries   ││
│  │  - View layers  │           │  - Markdown response display  ││
│  └─────────────────┘           └────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend (main.py)                   │
│  ┌─────────────────┐           ┌────────────────────────────────┐│
│  │   Map Routes    │           │       Query Routes            ││
│  │  /get_map_layers│           │  /query (LLM processing)      ││
│  │  /              │           │  /health                      ││
│  └─────────────────┘           └────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                    GeospatialService Layer                       │
│     (app/services/geospatial.py)                                │
│  - Coordinates map layer retrieval                              │
│  - Processes natural language queries with LLM                  │
│  - Converts polygon coordinates to WKT                          │
└──────────────────────────────────────────────────────────────────┘
                                   │
                         ┌─────────┴─────────┐
                         ▼                   ▼
┌────────────────────────────────┐  ┌────────────────────────────────┐
│    GeospatialAnalyzer          │  │    LangGraph Function Caller   │
│    (utils/GeospatialAnalyzer.py)│  │    (utils/langraph_function_  │
│                                │  │     caller.py)                 │
│  - File-based GeoDataFrame ops │  │  - LLM tool calling via       │
│  - GeoPandas for spatial joins │  │    LangGraph/LangChain        │
│  - Shapely geometry processing │  │  - Function tools for spatial  │
│  - NDVI/environmental metrics  │  │    analysis                   │
└────────────────────────────────┘  └────────────────────────────────┘
           │                                     │
           │                                     ▼
           │                        ┌────────────────────────────────┐
           │                        │    GeospatialAnalyzer2         │
           │                        │    (PostGIS-optimized)         │
           │                        │  - SQL-based spatial queries   │
           │                        │  - Efficient for scale         │
           │                        │  - Falls back to GeoPandas     │
           └───────────────────────►└────────────────────────────────┘
                                                 │
                                                 ▼
                                    ┌────────────────────────────────┐
                                    │        Data Sources            │
                                    │  - GeoJSON/GeoPackage files    │
                                    │  - PostGIS database            │
                                    │  - CSV tile statistics         │
                                    └────────────────────────────────┘
```

---

## Core Components

### 1. GeospatialAnalyzer (File-Based)
**Location:** `utils/GeospatialAnalyzer.py`

The primary analyzer class that loads and processes geospatial data using GeoPandas. It provides:

- **Data Loading**: Reads GeoJSON, GeoPackage, and CSV files
- **Spatial Operations**: Intersection, buffering, distance calculations
- **Environmental Metrics**: NDVI, elevation, slope, rainfall analysis
- **Settlement Analysis**: Building counts, village information
- **Infrastructure Analysis**: Roads, grid infrastructure, minigrids

**Key Methods:**
```python
count_features_within_region(region, layer_name, filter_expr)
analyze_region(region)  # Comprehensive region analysis
weighted_tile_stats_all(region)  # Environmental metrics
compute_distance_to_grid(geometry)
buffer_geometry(geom, radius_m)
```

### 2. GeospatialAnalyzer2 (PostGIS-Optimized)
**Location:** `utils/GeospatialAnalyzer2.py`

An optimized version that pushes spatial operations to PostGIS when available:

- **SQL-based Queries**: Uses ST_Intersects, ST_Area, etc.
- **Spatial Indexing**: Creates GiST indexes for performance
- **Lazy Loading**: Caches results for repeated queries
- **Fallback**: Falls back to GeoPandas when DB unavailable

### 3. Factory Pattern
**Location:** `utils/factory.py`

Creates the appropriate analyzer based on environment:
```python
def create_geospatial_analyzer():
    if use_postgis:
        return GeospatialAnalyzer2(database_uri=uri)
    else:
        return GeospatialAnalyzer(file_paths...)
```

### 4. LLM Integration

**LangGraph Agent** (`utils/langraph_function_caller.py`):
- Uses LangChain and LangGraph for tool-calling
- Supports Google Gemini and OpenAI models
- Maintains conversation state with InMemorySaver
- Provides tools for spatial analysis

**Legacy OpenAI Client** (`utils/llm_function_caller.py`):
- Direct OpenAI API integration
- Manual tool dispatch and response handling

---

## Data Layers

The analyzer supports these geospatial layers:

| Layer | Description | Geometry Type |
|-------|-------------|---------------|
| `buildings` | Building footprints | Polygon |
| `tiles` | Analysis grid tiles with statistics | Polygon |
| `roads` | Road network | LineString |
| `villages` | Village boundaries | Polygon |
| `parishes` | Parish boundaries | Polygon |
| `subcounties` | Subcounty boundaries | Polygon |
| `existing_grid` | Existing electricity grid | LineString |
| `grid_extension` | Proposed grid extensions | LineString |
| `candidate_minigrids` | Candidate minigrid sites | Polygon/Point |
| `existing_minigrids` | Existing minigrids | Polygon/Point |

---

## Strengths 💪

### 1. **Well-Structured Architecture**
- Clear separation of concerns (API, services, core, utilities)
- Factory pattern for flexible analyzer instantiation
- Dependency injection with FastAPI

### 2. **Dual Backend Support**
- File-based GeoPandas for development/small datasets
- PostGIS optimization for production scale
- Automatic fallback mechanism

### 3. **LLM Integration**
- Natural language interface for geospatial queries
- Tool-calling with function schemas
- Support for multiple LLM providers (OpenAI, Google Gemini)

### 4. **Comprehensive Analysis Capabilities**
- Environmental metrics (NDVI, elevation, rainfall)
- Settlement analysis (buildings, villages)
- Infrastructure analysis (roads, grid, minigrids)
- Distance calculations and buffering

### 5. **Interactive Frontend**
- Leaflet-based map with drawing tools
- Chat interface with markdown rendering
- Layer controls and legends
- Responsive design with Bootstrap

### 6. **Container-Ready**
- Dockerfile for easy deployment
- Docker Compose support
- Google Cloud Run deployment scripts

### 7. **Good Documentation**
- Comprehensive README with setup instructions
- Testing guide (TESTING.md)
- Docstrings on major functions

---

## Weaknesses 🔧

### 1. **Outdated/Broken Tests**
The test suite has issues:
```python
# Test uses old signature (minigrids_path)
analyzer = GeospatialAnalyzer(
    buildings_path="mock.gpkg",
    minigrids_path="mock.gpkg",  # This parameter no longer exists!
    ...
)
```
The `GeospatialAnalyzer.__init__` signature was updated but tests weren't.

### 2. **Hard-Coded API Key Initialization**
In `utils/llm_function_caller.py`:
```python
client = OpenAI()  # Fails if OPENAI_API_KEY not set
```
This causes import errors even when using Gemini.

### 3. **Missing Data Handling**
- Tests require actual data files that aren't in the repo
- No mock data or fixtures for unit testing
- Integration tests skip silently when data is missing

### 4. **Incomplete Error Handling**
Some methods return empty results silently:
```python
if gdf.empty:
    return {}  # No indication of why it's empty
```

### 5. **Mixed CRS Handling**
- Multiple places handle CRS conversion differently
- Some methods assume EPSG:4326, others don't
- Potential for coordinate system mismatches

### 6. **Memory Concerns**
- `get_layer_geojson()` can load large datasets into memory
- Building layer sampling helps but may not be sufficient for all layers

### 7. **Dual LLM Systems**
Two different LLM integration systems exist:
- `llm_function_caller.py` (OpenAI direct)
- `langraph_function_caller.py` (LangGraph/LangChain)

The app uses LangGraph but the legacy file still exists and causes issues.

### 8. **PostGIS Dependency Complexity**
- Requires ogr2ogr for data loading
- Complex setup for local development
- Database schema must match expected table names

---

## How It Works

### User Flow

1. **User opens web interface** → Loads `templates/index.html`
2. **Map initializes** → Fetches `/get_map_layers` for GeoJSON data
3. **User draws polygon** → Stores coordinates in JavaScript
4. **User asks question** → POST to `/query` with question + polygon
5. **LLM processes query** → LangGraph agent with geospatial tools
6. **Tools execute** → Analyzer performs spatial analysis
7. **Response rendered** → Markdown parsed and displayed

### Query Processing Flow

```
User Query + Polygon WKT
         │
         ▼
┌─────────────────────────┐
│  GeospatialService      │
│  process_query()        │
│  - Validate polygon     │
│  - Convert to WKT       │
│  - Call LLM             │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  LangGraph Agent        │
│  ask_with_functions()   │
│  - Parse query          │
│  - Select tools         │
│  - Execute analysis     │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  GeospatialAnalyzer     │
│  analyze_region()       │
│  count_features()       │
│  weighted_tile_stats()  │
└─────────────────────────┘
         │
         ▼
   Analysis Results
         │
         ▼
┌─────────────────────────┐
│  LLM Final Response     │
│  - Format results       │
│  - Generate insights    │
│  - Return markdown      │
└─────────────────────────┘
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required for OpenAI |
| `GOOGLE_API_KEY` | Google Gemini API key | Required for Gemini |
| `SUNTRACE_USE_POSTGIS` | Enable PostGIS backend | `false` |
| `SUNTRACE_DATABASE_URI` | PostGIS connection string | Local default |
| `LANGSMITH_API_KEY` | LangSmith tracing | Optional |

### File Paths (configs/paths.py)

Data files are expected in the `data/` directory:
- Buildings, roads, villages, etc. in `data/viz_geojsons/`
- Tile statistics in `data/Lamwo_Tile_Stats_EE_biomass.csv`
- Grid tiles in `data/lamwo_sentinel_composites/`

---

## Testing

### Run Tests
```bash
make test              # Unit tests (non-slow, non-integration)
make test-all          # All tests
make test-integration  # Integration tests with data
make test-coverage     # With coverage report
```

### Current Test Status
- **Passing**: Basic validation, CRS handling, parametrized tests
- **Failing**: Tests using outdated `GeospatialAnalyzer` signature
- **Skipped**: Integration tests (missing data files)

---

## Recommendations

### Immediate Fixes

1. **Fix test signatures** - Update tests to use new `GeospatialAnalyzer` parameters
2. **Lazy OpenAI client** - Don't initialize at import time
3. **Add mock data** - Create fixtures for unit tests

### Short-Term Improvements

1. **Consolidate LLM integrations** - Use only LangGraph, remove legacy
2. **Add validation layer** - Validate inputs before processing
3. **Improve error messages** - Return specific errors, not empty results

### Long-Term Enhancements

1. **Add caching layer** - Cache frequent spatial queries
2. **Streaming responses** - Use LangGraph streaming for real-time updates
3. **User sessions** - Persist conversation state
4. **Multi-region support** - Extend beyond Lamwo district

---

## File Structure Reference

```
suntrace/
├── main.py                 # FastAPI app entry point
├── app/
│   ├── api/
│   │   ├── deps.py         # Dependency injection, app factory
│   │   └── routes/
│   │       ├── map.py      # Map layer endpoints
│   │       └── query.py    # Query processing endpoints
│   ├── core/
│   │   ├── config.py       # Settings with Pydantic
│   │   ├── exceptions.py   # Custom exceptions
│   │   ├── logger.py       # Logging setup
│   │   └── middleware.py   # Request logging
│   ├── models/
│   │   └── schemas.py      # Pydantic request/response models
│   └── services/
│       └── geospatial.py   # Main service class
├── utils/
│   ├── GeospatialAnalyzer.py   # File-based analyzer
│   ├── GeospatialAnalyzer2.py  # PostGIS-optimized analyzer
│   ├── factory.py              # Analyzer factory
│   ├── langraph_function_caller.py  # LangGraph LLM integration
│   ├── llm_function_caller.py      # Legacy OpenAI integration
│   └── tools.py                    # Tool definitions
├── configs/
│   ├── paths.py           # Data file paths
│   ├── stats.py           # Statistics column definitions
│   └── system_prompt.py   # LLM system prompt
├── templates/
│   └── index.html         # Frontend with Leaflet + chat
├── tests/                 # Test suite
├── data/                  # Geospatial data (gitignored)
└── scripts/               # Utility scripts
```

---

## Conclusion

Suntrace is a well-designed geospatial analysis platform with solid architecture and comprehensive functionality. The main areas needing attention are:

1. **Test maintenance** - Update tests to match current API
2. **LLM consolidation** - Remove legacy integration
3. **Error handling** - Improve user feedback
4. **Documentation** - Keep in sync with code changes

The project demonstrates excellent use of modern Python patterns (dependency injection, factory pattern, Pydantic validation) and provides a powerful natural language interface for complex geospatial analysis.
