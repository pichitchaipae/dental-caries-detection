import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import settings
from app.db import get_engine
from app.runner import InferenceRunner

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)

runner = InferenceRunner()

# Create DB engine
db_engine = get_engine(settings.database_url)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ML Service...")
    yield
    logger.info("Shutting down ML Service...")
    runner.cancel()

app = FastAPI(
    title="Dental Caries Detection ML Service",
    description="YOLOv8-based dental caries detection service (Phase 2)",
    lifespan=lifespan,
)

class InferRequest(BaseModel):
    jobId: int

@app.post("/infer", status_code=202)
def infer(req: InferRequest):
    try:
        runner.start(req.jobId, settings.model_dump(by_alias=True), db_engine)
        return {"status": "accepted", "job_id": req.jobId}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to start inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cancel")
def cancel():
    runner.cancel()
    return {"status": "cancelled"}

@app.get("/health")
def health():
    return {
        "ready": True,
        "artifacts": {},
        "active_job_id": runner.active_job_id
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
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
