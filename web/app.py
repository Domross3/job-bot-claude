"""FastAPI web app wrapping the resume tailoring pipeline."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents import AnalyzerAgent, CriticAgent, FormatterAgent, MapperAgent, PrunerAgent
from src.state import Evaluation, PipelineState
from src.utils.ingestor import IngestionError, ingest_file
from src.utils.resume_normalizer import normalize_project_sections
from src.utils.resume_parser import parse_source_projects
from web.supabase_client import get_all_data, log_feedback, log_run

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ── App setup ────────────────────────────────────────────────────

app = FastAPI(title="ResumeForge", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

ADMIN_KEY = os.getenv("ADMIN_KEY", "dev-admin-key")


# ── In-memory run store ──────────────────────────────────────────

@dataclass
class RunState:
    run_id: str
    events: list[dict] = field(default_factory=list)
    pdf_bytes: bytes | None = None
    critic_data: dict | None = None
    error: str | None = None
    done: bool = False
    created_at: float = field(default_factory=time.time)
    telemetry: dict = field(default_factory=dict)


runs: dict[str, RunState] = {}


def _cleanup_old_runs() -> None:
    """Remove runs older than 1 hour."""
    cutoff = time.time() - 3600
    stale = [rid for rid, r in runs.items() if r.created_at < cutoff]
    for rid in stale:
        del runs[rid]


# ── Web pipeline (automated revision loop replaces HITL gate) ───

MAX_CRITIC_REVISIONS = 2

def _run_pipeline_sync(run_state: RunState, resume_text: str, jd_text: str) -> None:
    """Run the full pipeline synchronously (called via asyncio.to_thread).

    Mirrors the CLI pipeline's revision loop: if the critic doesn't approve,
    feed its suggestions back into the mapper and re-run the core loop.
    """
    start_time = time.time()
    page_fit_attempts = 0

    try:
        state = PipelineState(
            master_resume=resume_text,
            job_description=jd_text,
            source_projects=parse_source_projects(resume_text),
        )

        analyzer = AnalyzerAgent()
        mapper = MapperAgent()
        pruner = PrunerAgent()
        critic = CriticAgent()
        formatter = FormatterAgent()

        # Step 1: Analyze
        run_state.events.append({"step": "analyzing", "detail": "Extracting requirements from job description..."})
        state = analyzer.run(state)

        # Steps 2-4: Core loop with critic-driven revisions
        for revision in range(MAX_CRITIC_REVISIONS + 1):
            # Step 2: Map
            detail = "Selecting and rewriting relevant experience..."
            if revision > 0:
                detail = f"Revision {revision}: re-mapping based on critic feedback..."
            run_state.events.append({"step": "mapping", "detail": detail})
            state = mapper.run(state)
            state = state.model_copy(
                update={"mapped_sections": normalize_project_sections(state.mapped_sections, state.source_projects)}
            )

            # Step 3: Prune
            detail = "Optimizing content for single-page fit..."
            if revision > 0:
                detail = f"Revision {revision}: re-pruning with critic guidance..."
            run_state.events.append({"step": "pruning", "detail": detail})
            state = pruner.run(state)
            state = state.model_copy(
                update={
                    "pruned_sections": normalize_project_sections(
                        state.pruned_sections, state.source_projects, max_bullets_per_project=3
                    )
                }
            )

            # Step 4: Critic
            detail = "Verifying factual accuracy and quality..."
            if revision > 0:
                detail = f"Revision {revision}: re-evaluating improvements..."
            run_state.events.append({"step": "reviewing", "detail": detail})
            state = critic.run(state)

            # Check if critic approved or we've exhausted revisions
            if state.evaluation and state.evaluation.approved:
                logger.info("Critic approved on revision %d", revision)
                break

            if revision < MAX_CRITIC_REVISIONS and state.evaluation:
                # Feed critic's suggestions back so mapper sees them
                logger.info(
                    "Critic rejected (score %.2f), revision %d/%d — feeding back suggestions",
                    state.evaluation.overall_score, revision + 1, MAX_CRITIC_REVISIONS,
                )
            else:
                logger.info("Max revisions reached or no evaluation — proceeding with best draft")

        # Step 5: Render-and-refine loop (overflow + underfill detection)
        run_state.events.append({"step": "rendering", "detail": "Generating polished PDF..."})
        content_floor_attempts = 0
        CONTENT_FLOOR_RATIO = 0.80
        MAX_CONTENT_FLOOR_RETRIES = 2

        for iteration in range(1, state.max_render_iterations + 1):
            page_count = formatter.test_render(state)
            page_fit_attempts += 1

            if page_count <= 1:
                # Check content floor — is the page full enough?
                fill_ratio = formatter.measure_fill_ratio(state)
                if fill_ratio < CONTENT_FLOOR_RATIO and content_floor_attempts < MAX_CONTENT_FLOOR_RETRIES:
                    content_floor_attempts += 1
                    run_state.events.append({
                        "step": "mapping",
                        "detail": f"Page only {fill_ratio:.0%} filled — adding more content (attempt {content_floor_attempts})..."
                    })
                    logger.warning(
                        "Content floor: %.0f%% fill — re-mapping (attempt %d/%d)",
                        fill_ratio * 100, content_floor_attempts, MAX_CONTENT_FLOOR_RETRIES,
                    )
                    from .state import Evaluation
                    synthetic_eval = Evaluation(
                        approved=False,
                        factual_drift_issues=[],
                        missing_keywords=state.evaluation.missing_keywords if state.evaluation else [],
                        suggestions=[
                            f"CRITICAL: The resume is only {fill_ratio:.0%} filled. Add more detailed "
                            f"bullets with specific accomplishments and metrics from the master resume. "
                            f"Ensure all work entries have 3-4 substantial bullets. Include all projects. "
                            f"Expand Education with honors, coursework, or activities."
                        ],
                        overall_score=0.3,
                    )
                    state = state.model_copy(update={"evaluation": synthetic_eval})
                    state = mapper.run(state)
                    state = state.model_copy(
                        update={"mapped_sections": normalize_project_sections(state.mapped_sections, state.source_projects)}
                    )
                    run_state.events.append({"step": "pruning", "detail": "Re-optimizing expanded content..."})
                    state = pruner.run(state)
                    state = state.model_copy(
                        update={
                            "pruned_sections": normalize_project_sections(
                                state.pruned_sections, state.source_projects, max_bullets_per_project=3
                            )
                        }
                    )
                    continue
                break

            # Overflow — re-prune
            overflow_ratio = 1.0 + (0.3 * iteration)
            state = state.model_copy(
                update={"overflow_pages": overflow_ratio, "render_iteration": iteration}
            )
            state = pruner.run(state)
            state = state.model_copy(
                update={
                    "pruned_sections": normalize_project_sections(
                        state.pruned_sections, state.source_projects, max_bullets_per_project=3
                    )
                }
            )

        # Render final PDF to bytes
        buffer = BytesIO()
        formatter._build_pdf(state, buffer)
        run_state.pdf_bytes = buffer.getvalue()
        buffer.close()

        # Build critic response data
        critic_data = None
        if state.evaluation:
            ev = state.evaluation
            critic_data = {
                "approved": ev.approved,
                "overall_score": ev.overall_score,
                "factual_drift_issues": ev.factual_drift_issues,
                "missing_keywords": ev.missing_keywords,
                "suggestions": ev.suggestions,
            }
        run_state.critic_data = critic_data

        # Telemetry
        duration_ms = int((time.time() - start_time) * 1000)
        jd_analysis = None
        if state.analysis:
            jd_analysis = state.analysis.model_dump()

        run_state.telemetry = {
            "id": run_state.run_id,
            "jd_snippet": jd_text[:500],
            "jd_analysis": jd_analysis,
            "critic_eval": critic_data,
            "page_fit_attempts": page_fit_attempts,
            "overall_score": state.evaluation.overall_score if state.evaluation else None,
            "duration_ms": duration_ms,
            "error": None,
        }

        run_state.events.append({
            "step": "complete",
            "detail": "Resume generated successfully!",
            "run_id": run_state.run_id,
            "critic": critic_data,
        })
        logger.info("Pipeline completed for run %s in %dms", run_state.run_id, duration_ms)

    except Exception as exc:
        logger.exception("Pipeline failed for run %s", run_state.run_id)
        duration_ms = int((time.time() - start_time) * 1000)
        run_state.error = str(exc)
        run_state.telemetry = {
            "id": run_state.run_id,
            "jd_snippet": jd_text[:500],
            "jd_analysis": None,
            "critic_eval": None,
            "page_fit_attempts": page_fit_attempts,
            "overall_score": None,
            "duration_ms": duration_ms,
            "error": str(exc),
        }
        run_state.events.append({"step": "error", "detail": str(exc)})

    finally:
        run_state.done = True


# ── Routes ───────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/generate")
async def generate(resume: UploadFile = File(...), jd: str = Form(...)):
    """Start a pipeline run. Returns a run_id for SSE streaming."""
    _cleanup_old_runs()

    # Ingest the uploaded resume
    content = await resume.read()
    suffix = Path(resume.filename or "resume.txt").suffix.lower()

    if suffix == ".pdf":
        # Write to temp file for PyMuPDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            resume_text = ingest_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    elif suffix in (".md", ".txt", ".markdown"):
        resume_text = content.decode("utf-8")
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Resume file is empty.")
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Use PDF, TXT, or MD.",
        )

    if not jd.strip():
        raise HTTPException(status_code=400, detail="Job description is empty.")

    # Create run
    run_id = str(uuid4())
    run_state = RunState(run_id=run_id)
    runs[run_id] = run_state

    # Launch pipeline in background thread
    asyncio.get_event_loop().run_in_executor(
        None, _run_pipeline_sync, run_state, resume_text, jd.strip()
    )

    return {"run_id": run_id}


@app.get("/stream/{run_id}")
async def stream(run_id: str):
    """SSE endpoint — streams pipeline progress events."""
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found.")

    run_state = runs[run_id]

    async def event_generator():
        last_idx = 0
        while True:
            # Yield any new events
            while last_idx < len(run_state.events):
                event = run_state.events[last_idx]
                last_idx += 1
                import json
                yield {"data": json.dumps(event)}

            if run_state.done:
                # Log telemetry to Supabase
                if run_state.telemetry:
                    await log_run(run_state.telemetry)
                break

            await asyncio.sleep(0.3)

    return EventSourceResponse(event_generator())


@app.get("/download/{run_id}")
async def download(run_id: str):
    """Download the generated PDF."""
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found.")

    run_state = runs[run_id]
    if run_state.pdf_bytes is None:
        if run_state.error:
            raise HTTPException(status_code=500, detail=f"Pipeline failed: {run_state.error}")
        raise HTTPException(status_code=202, detail="PDF not ready yet.")

    return Response(
        content=run_state.pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=tailored_resume.pdf"},
    )


class FeedbackRequest(BaseModel):
    run_id: str | None = None
    message: str
    rating: int | None = None


@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Submit user feedback."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message is required.")

    feedback_data = {
        "run_id": req.run_id if req.run_id and req.run_id in runs else None,
        "message": req.message.strip(),
        "rating": req.rating,
    }
    await log_feedback(feedback_data)
    return {"status": "ok"}


@app.get("/admin/data")
async def admin_data(key: str = Query(...)):
    """Export all runs and feedback as JSON. Requires admin key."""
    if key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key.")
    data = await get_all_data()
    return JSONResponse(content=data)
