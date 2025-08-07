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

### 2. Run Tests

```sh
make test
```

### 3. Start the Application (Local)

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

### 5. Access Frontend

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
