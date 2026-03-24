"""Lightweight Supabase client using httpx — no supabase-py dependency.

If SUPABASE_URL or SUPABASE_KEY are not set, all functions silently no-op
so the app can run locally without a database.
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

_warned = False


def _headers() -> dict[str, str]:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _available() -> bool:
    global _warned
    if SUPABASE_URL and SUPABASE_KEY:
        return True
    if not _warned:
        logger.warning(
            "Supabase not configured (SUPABASE_URL / SUPABASE_KEY missing). "
            "Telemetry and feedback will not be persisted."
        )
        _warned = True
    return False


async def log_run(run_data: dict) -> None:
    """Insert a row into the 'runs' table."""
    if not _available():
        return
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/runs",
                headers=_headers(),
                json=run_data,
                timeout=10,
            )
            resp.raise_for_status()
    except Exception as exc:
        logger.error("Failed to log run to Supabase: %s", exc)


async def log_feedback(feedback_data: dict) -> None:
    """Insert a row into the 'feedback' table."""
    if not _available():
        return
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/feedback",
                headers=_headers(),
                json=feedback_data,
                timeout=10,
            )
            resp.raise_for_status()
    except Exception as exc:
        logger.error("Failed to log feedback to Supabase: %s", exc)


async def get_all_data() -> dict:
    """Fetch all runs and feedback for admin export."""
    if not _available():
        return {"runs": [], "feedback": []}
    try:
        async with httpx.AsyncClient() as client:
            runs_resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/runs",
                headers=_headers(),
                params={"select": "*", "order": "created_at.desc", "limit": "500"},
                timeout=10,
            )
            runs_resp.raise_for_status()

            feedback_resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/feedback",
                headers=_headers(),
                params={"select": "*", "order": "created_at.desc", "limit": "500"},
                timeout=10,
            )
            feedback_resp.raise_for_status()

            return {
                "runs": runs_resp.json(),
                "feedback": feedback_resp.json(),
            }
    except Exception as exc:
        logger.error("Failed to fetch data from Supabase: %s", exc)
        return {"runs": [], "feedback": []}
