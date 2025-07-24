"""
Application configuration settings
"""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # App settings
    APP_NAME: str = "Geospatial Analyzer API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = """A geospatial analysis API that provides interactive mapping capabilities and LLM-powered queries.

## Features

- **Interactive Mapping**: Get map layers with candidate minigrids and buildings
- **Geospatial Queries**: Process natural language queries with spatial context
- **Polygon Analysis**: Draw polygons on the map and analyze specific regions
- **LLM Integration**: Powered by OpenAI for intelligent geospatial analysis
"""

    # Server settings
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = True

    # API settings
    API_V1_STR: str = "/api/v1"

    # OpenAI settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # CORS settings
    ALLOWED_ORIGINS: list = ["*"]

    # Building sample limit for performance
    BUILDING_SAMPLE_LIMIT: int = 2000

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
