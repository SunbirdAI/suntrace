"""
Dependency injection and application factory
"""

import os
import sys


import openai
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.core.config import get_settings
from app.services.geospatial import GeospatialService
from app.core.middleware import LoggingMiddleware


# Ensure 'src' is in the Python path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Clear any existing problematic environment variable
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]

# Load environment variables from .env file
load_dotenv()

# Global instances
geospatial_service: GeospatialService = None
templates: Jinja2Templates = None


def get_geospatial_service() -> GeospatialService:
    """Dependency to get geospatial service instance"""
    global geospatial_service
    if geospatial_service is None:
        geospatial_service = GeospatialService()
    return geospatial_service


def get_templates() -> Jinja2Templates:
    """Dependency to get templates instance"""
    if not hasattr(get_templates, "templates"):
        get_templates.templates = Jinja2Templates(directory="templates")
    return get_templates.templates


def create_application() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()

    # Set OpenAI API key
    openai.api_key = settings.OPENAI_API_KEY

    # Create FastAPI app
    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        contact={
            "name": "Geospatial Analysis Team",
            # "email": "support@geospatial-analyzer.com",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # logging
    app.add_middleware(LoggingMiddleware)

    # Mount static files only if directory exists and is accessible
    static_dir = "static"
    if os.path.exists(static_dir) and os.access(static_dir, os.R_OK):
        try:
            app.mount("/static", StaticFiles(directory=static_dir), name="static")
        except Exception as e:
            print(f"Warning: Could not mount static files: {e}")

    from app.api.routes.map import router as map_router
    from app.api.routes.query import router as query_router

    app.include_router(map_router, tags=["Map Data"])
    app.include_router(query_router, tags=["Analysis"])

    return app
