"""CLI entry point for the resume tailoring pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from .pipeline import Pipeline
from .utils.ingestor import IngestionError, ingest_file

# ── Defaults ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-agent resume tailoring system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=INPUTS_DIR / "master_resume.md",
        help="Path to the master resume (.md, .txt, or .pdf)",
    )
    parser.add_argument(
        "--jd",
        type=Path,
        default=INPUTS_DIR / "job_description.md",
        help="Path to the job description (.md, .txt, or .pdf)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: outputs/tailored_YYYYMMDD_HHMMSS.pdf)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    args = parser.parse_args()

    # ── Logging ──────────────────────────────────────────────────
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Ingest inputs (supports .md, .txt, .pdf) ───────────────
    try:
        master_resume = ingest_file(args.resume)
    except FileNotFoundError:
        print(f"✖  Master resume not found: {args.resume}", file=sys.stderr)
        sys.exit(1)
    except IngestionError as exc:
        print(f"✖  Failed to ingest master resume: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        job_description = ingest_file(args.jd)
    except FileNotFoundError:
        print(f"✖  Job description not found: {args.jd}", file=sys.stderr)
        sys.exit(1)
    except IngestionError as exc:
        print(f"✖  Failed to ingest job description: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Output path ──────────────────────────────────────────────
    output_path = args.output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUTS_DIR / f"tailored_{timestamp}.pdf"

    # ── Run pipeline ─────────────────────────────────────────────
    print("\n🚀 Starting resume tailoring pipeline...\n")
    pipeline = Pipeline()
    state = pipeline.run(master_resume, job_description, output_path)

    if state.final_resume:
        print("\n✔  Pipeline complete!")
        print(f"   Open with: open \"{output_path}\"")
    else:
        print("\n⚠  Pipeline finished without producing a final resume.")


if __name__ == "__main__":
    main()
