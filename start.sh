#!/bin/bash

# Start the FastAPI application with Uvicorn
uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1