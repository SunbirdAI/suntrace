"""
Custom exceptions for the application
"""

from fastapi import HTTPException, status


class GeospatialAnalysisError(HTTPException):
    """Custom exception for geospatial analysis errors"""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Geospatial analysis error: {detail}",
        )


class DataProcessingError(HTTPException):
    """Custom exception for data processing errors"""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Data processing error: {detail}",
        )


class LLMQueryError(HTTPException):
    """Custom exception for LLM query errors"""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM query error: {detail}",
        )
