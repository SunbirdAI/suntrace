# Use slim Python 3.12 image
FROM python:3.12-slim

# Don’t write .pyc files and force stdout/stderr to be unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Tell rasterio/geopandas where GDAL lives
    GDAL_CONFIG=/usr/bin/gdal-config \
    CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal

# Set working directory
WORKDIR /app

# Install build tools + GDAL + GEOS + PROJ headers
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

COPY ./start.sh /app/start.sh

# Expose the port FastAPI will run on
ENV PORT 8080
# EXPOSE 8000

# Run the app using Uvicorn
RUN chmod +x /app/start.sh
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
ENTRYPOINT ["/app/start.sh"]
