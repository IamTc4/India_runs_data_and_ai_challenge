"""
api/main.py — FastAPI enterprise server for Redrob AI Ranker.

Provides REST endpoints for the ranking pipeline.
Supports async job submission with status polling for long-running 100K jobs.

Start server:
    uvicorn api.main:app --reload --port 8000

Docs available at:
    http://localhost:8000/docs
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import logger

# In-memory job store (replace with Redis for production)
_job_store: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Redrob API server starting up...")
    yield
    logger.info("Redrob API server shutting down.")


app = FastAPI(
    title="Redrob AI Ranker API",
    description="Enterprise-grade hybrid candidate retrieval system.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class RankRequest(BaseModel):
    candidates_path: str = Field(..., description="Absolute path to candidates.jsonl or .json")
    output_path: str = Field("submission.csv", description="Output CSV path")
    use_models: bool = Field(True, description="Enable ML semantic models (Stage 3 + 4)")
    debug: bool = Field(False, description="Include detailed debug logging")


class JobStatus(BaseModel):
    job_id: str
    status: str    # "pending" | "running" | "completed" | "failed"
    message: str
    output_path: Optional[str] = None
    valid: Optional[bool] = None


class HealthResponse(BaseModel):
    status: str
    version: str


# ── Background Worker ─────────────────────────────────────────────────────────

def _run_pipeline_job(job_id: str, request: RankRequest):
    """Runs the full pipeline in a background thread."""
    _job_store[job_id]["status"] = "running"
    _job_store[job_id]["message"] = "Pipeline running..."

    try:
        from src.pipeline.pipeline import run_pipeline
        valid = run_pipeline(
            candidates_path=request.candidates_path,
            output_path=request.output_path,
            debug=request.debug,
            skip_stage3=not request.use_models,
            skip_stage4=not request.use_models,
        )
        _job_store[job_id].update({
            "status": "completed",
            "message": "Pipeline completed successfully." if valid else "Pipeline completed but validation failed.",
            "output_path": request.output_path,
            "valid": valid,
        })
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        _job_store[job_id].update({
            "status": "failed",
            "message": str(e),
        })


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/api/rank", response_model=JobStatus, status_code=202, tags=["Ranking"])
async def submit_ranking_job(request: RankRequest, background_tasks: BackgroundTasks):
    """
    Submit a candidate ranking job. Returns a job_id for status polling.
    The pipeline runs asynchronously — use GET /api/rank/{job_id} to check progress.
    """
    job_id = str(uuid.uuid4())
    _job_store[job_id] = {
        "status": "pending",
        "message": "Job queued.",
        "output_path": None,
        "valid": None,
    }
    background_tasks.add_task(_run_pipeline_job, job_id, request)
    logger.info(f"Job {job_id} submitted | candidates: {request.candidates_path}")
    return JobStatus(job_id=job_id, **_job_store[job_id])


@app.get("/api/rank/{job_id}", response_model=JobStatus, tags=["Ranking"])
async def get_job_status(job_id: str):
    """Poll ranking job status."""
    if job_id not in _job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return JobStatus(job_id=job_id, **_job_store[job_id])


@app.get("/api/jobs", tags=["Ranking"])
async def list_jobs():
    """List all submitted jobs and their statuses."""
    return [
        {"job_id": jid, **info}
        for jid, info in _job_store.items()
    ]
