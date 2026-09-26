from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.services.detection import DetectionService

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper()))
logger = logging.getLogger(__name__)

# Shared detection service instance
detection_service = DetectionService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load model on startup."""
    logger.info("Starting ML Service...")
    try:
        detection_service.load_model()
        logger.info("ML Service started successfully")
    except Exception as e:
        logger.error(f"Failed to start ML Service: {e}")
    yield
    logger.info("Shutting down ML Service...")


app = FastAPI(
    title="Dental Caries Detection ML Service",
    description="YOLOv8-based dental caries detection service",
    version=settings.MODEL_VERSION,
    lifespan=lifespan,
)


def get_detection_service() -> DetectionService:
    """Dependency injection for DetectionService."""
    return detection_service


# Register routes after app creation
from app.api.routes import router  # noqa: E402

app.include_router(router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal server error occurred.",
            },
        },
    )
