from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.logger import logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """Logging

    Args:
        BaseHTTPMiddleware (_type_): _description_
    """
    async def dispatch(self, request: Request, call_next):
        logger.info("Incoming request: %s %s", request.method, request.url.path)
        response = await call_next(request)
        logger.info(
            "Response status: %s for %s %s", response.status_code, request.method, request.url.path
        )
        return response
