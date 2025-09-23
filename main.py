"""Main FastAPI application entry point."""

import logging
import sys
from pathlib import Path

import uvicorn

from app.api.deps import create_application
from app.core.config import get_settings

# Add the src directory to Python path
src_dir = Path(__file__).parent
project_root = src_dir.parent
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(project_root))


logging.basicConfig(level=logging.INFO)
logging.getLogger("GeospatialAnalyzer2").setLevel(logging.INFO)

settings = get_settings()
app = create_application()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
