# Suntrace GeospatialAnalyzer Project

Suntrace is a multimodal geospatial analysis platform integrating GIS data, LLM function calling, and interactive web visualization.

## Features

- Geospatial analysis with [`app/core/GeospatialAnalyzer`](app/core/)
- LLM-powered function calling for spatial queries
- Interactive frontend with map and chat ([templates/index.html](templates/index.html))
- REST API via FastAPI (see [`main.py`](main.py))
- Comprehensive test suite ([tests/](tests/))

## Project Structure

```
app/                # Application modules (API, core logic, models, services)
configs/            # Configuration and path management
data/               # Geospatial datasets
notebooks/          # Jupyter notebooks and experiments
templates/          # Frontend HTML templates
tests/              # Pytest test suite
utils/              # Utility functions and factories
main.py             # FastAPI entrypoint
Dockerfile          # Container build instructions
requirements.txt    # Python dependencies
Makefile            # Dev workflow commands
```

## Quick Start

### 1. Install Dependencies

```sh
pip install -r requirements.txt
```

### 2. (Optional) Stand Up PostGIS via Docker

If you want to use the PostGIS-backed analyzer, bring up a local PostGIS container:

```sh
docker run \
  --name suntrace-postgis \
  -e POSTGRES_DB=suntrace \
  -e POSTGRES_USER=pguser \
  -e POSTGRES_PASSWORD=pgpass \
  -p 5432:5432 \
  -d postgis/postgis:15-3.4
```

Wait for the database to accept connections, then ensure the PostGIS extensions exist (the analyzer can do this for you, or run once inside the container):

```sh
docker exec -it suntrace-postgis psql -U pguser -d suntrace -c "CREATE EXTENSION IF NOT EXISTS postgis;"
docker exec -it suntrace-postgis psql -U pguser -d suntrace -c "CREATE EXTENSION IF NOT EXISTS postgis_topology;"
```

### 3. Load Project Data into PostGIS

With the container running and data available under `data/`, load all vector layers:

```sh
python scripts/load_to_postgis.py \
  --data-dir data \
  --db-uri postgresql://pguser:pgpass@localhost:5432/suntrace
```

Requirements for the loader:

- `ogr2ogr` (GDAL) must be installed on the host running the script.
```sh
brew install gdal
```
- Python deps: `geopandas`, `pandas`, `sqlalchemy`.

> The script scans `data/`, `data/lamwo_sentinel_composites/`, `data/viz_geojsons/`, and `data/sample_region_mudu/`, writing tables such as `public.lamwo_buildings`, `public.lamwo_roads`, `public.lamwo_tile_stats_ee_biomass`, etc. Ensure filenames follow the repository defaults so table names match the analyzer’s expectations.

If you need the joined tile view, run inside a Python shell after the load:

```python
from utils.GeospatialAnalyzer2 import GeospatialAnalyzer2
analyzer = GeospatialAnalyzer2()
analyzer.create_joined_tiles(
    tile_stats_table='public.lamwo_tile_stats_ee_biomass',
    plain_tiles_table='public.lamwo_grid'
)
```

### 4. Configure the App to Use PostGIS

Set the following in `.env` (already present in this repo by default):

```
SUNTRACE_USE_POSTGIS=1
SUNTRACE_DATABASE_URI=postgresql+psycopg://pguser:pgpass@localhost:5432/suntrace
```

Restart the app after changing the env file. The factory will log which analyzer was initialized.

### 5. Run Tests

```sh
make test
```

### 6. Start the Application (Local)

```sh
uvicorn main:app --reload
```

or

```python
python main.py
```

### 4. Run in Docker

```sh
export OPENAI_API_KEY=your_openai_key
docker build -t suntrace .
docker run --rm -d \
  -p 8080:8080 \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  --name suntrace \
  suntrace:latest
```

See logs

```sh
docker logs -f suntracte
```

#### With docker compose

```sh
docker-compose up -d --build
```

### 7. Access Frontend

Open [http://localhost:8080](http://localhost:8080) in your browser.


## Testing

- Tests are in [tests/](tests/)
- Run with coverage: `make test-coverage`
- See [tests/TESTING.md](tests/TESTING.md) for details

## Environment Variables

Create a `.env` file for secrets (e.g., OpenAI API key):

```
OPENAI_API_KEY=your_openai_key
```

## Deployment

Make sure you have [**gcloud cli**](https://cloud.google.com/sdk/docs/install-sdk) installed and setup

The app is deployed using [**Google Cloud Run**](https://cloud.google.com/run?hl=en)

To deploy the application, run the commands below

```sh
chmod +x bin/deploy
chmod +x start.sh
./bin/deploy
```

## Data Requirements

Place required geospatial files in the `data/` directory. See [tests/TESTING.md](tests/TESTING.md) for details.

## License

See [LICENSE](LICENSE) for usage terms.

---
