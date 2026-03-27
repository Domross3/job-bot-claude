"""Pipeline state models — the single source of truth passed between agents."""

from __future__ import annotations

from pydantic import BaseModel, Field


class JDAnalysis(BaseModel):
    """Structured extraction of a Job Description."""

    job_title: str
    company_name: str
    hard_skills: list[str] = Field(
        description='Explicit technical skills, e.g. ["Python", "Kubernetes"]'
    )
    soft_skills: list[str] = Field(
        description='Interpersonal / process skills, e.g. ["leadership"]'
    )
    key_phrases: list[str] = Field(
        description="Verbatim high-value phrases lifted from the JD"
    )
    experience_years: str | None = Field(
        default=None, description='e.g. "5+ years"'
    )
    education_requirements: list[str] = Field(default_factory=list)
    priority_ranking: list[str] = Field(
        description="Skills ranked by emphasis in the JD (most important first)"
    )


class ResumeEntry(BaseModel):
    """A single item within a resume section (role, degree, cert, etc.)."""

    entry_id: str | None = Field(
        default=None,
        description="Stable deterministic identifier for this source entry",
    )
    title: str = Field(description="Role title, degree, or certification name")
    organization: str
    dates: str
    bullets: list[str]
    bullet_ids: list[str] = Field(
        default_factory=list,
        description="Stable source bullet IDs aligned positionally with bullets",
    )
    relevance_score: float | None = Field(
        default=None, description="Set by Mapper (0.0–1.0)"
    )


class ResumeSection(BaseModel):
    """A top-level resume section (e.g. Work Experience, Skills)."""

    heading: str
    entries: list[ResumeEntry] = Field(default_factory=list)


class Evaluation(BaseModel):
    """Output of the Critic agent."""

    approved: bool
    factual_drift_issues: list[str] = Field(
        description="Specific bullets that deviate from the master resume"
    )
    missing_keywords: list[str] = Field(
        description="Target keywords from the JD that were not integrated"
    )
    suggestions: list[str] = Field(description="Actionable fixes for the next revision")
    overall_score: float = Field(
        description="0.0–1.0 quality score for the tailored draft"
    )


class SourceBullet(BaseModel):
    """A single source-truth bullet parsed deterministically from the resume."""

    bullet_id: str
    entry_id: str
    section_kind: str
    section_heading: str
    title: str
    organization: str
    dates: str
    text: str
    order: int


class MappedBullet(BaseModel):
    """Mapper output for one source bullet."""

    bullet_id: str
    rewritten_text: str
    relevance_score: float = Field(description="0.0–1.0")


class PipelineState(BaseModel):
    """Immutable-ish state object threaded through every agent."""

    # ── Raw inputs ──────────────────────────────────────────────
    master_resume: str
    job_description: str

    # ── Agent outputs (populated progressively) ─────────────────
    analysis: JDAnalysis | None = None
    mapped_sections: list[ResumeSection] | None = None
    pruned_sections: list[ResumeSection] | None = None
    evaluation: Evaluation | None = None
    final_resume: str | None = None
    source_sections: list[ResumeSection] = Field(
        default_factory=list,
        description="Deterministic full-resume structure parsed from the source resume",
    )
    source_bullets: list[SourceBullet] = Field(
        default_factory=list,
        description="Flattened source-truth bullet inventory",
    )
    source_projects: list[ResumeEntry] = Field(
        default_factory=list,
        description="Project entries deterministically parsed from the source master resume",
    )
    mapped_bullets: list[MappedBullet] = Field(
        default_factory=list,
        description="All source bullets rewritten/scored by the Mapper",
    )
    selected_bullet_ids: list[str] = Field(
        default_factory=list,
        description="Current deterministic bullet selection used to assemble the draft",
    )
    polished_bullet_texts: dict[str, str] = Field(
        default_factory=dict,
        description="Quality-polished bullet text keyed by source bullet ID",
    )
    pruner_feedback: list[str] = Field(
        default_factory=list,
        description="Programmatic guardrail feedback injected into the Pruner prompt",
    )
    force_source_project_inventory: bool = Field(
        default=False,
        description="When true, Pruner receives the source project inventory to restore missing projects",
    )

    # ── Pipeline metadata ───────────────────────────────────────
    revision_count: int = 0
    max_revisions: int = 2
    critic_history: list[Evaluation] = Field(
        default_factory=list,
        description="All past evaluations (useful for tracking improvement across loops)",
    )

    # ── Render-and-refine loop ────────────────────────────────
    overflow_pages: float | None = Field(
        default=None,
        description="Set by Formatter test render — e.g. 1.3 means 30% over 1 page",
    )
    last_render_page_count: int | None = Field(
        default=None,
        description="Most recent page count observed during formatter test render",
    )
    last_render_fill_ratio: float | None = Field(
        default=None,
        description="Estimated fraction of the printable frame consumed by the latest render",
    )
    render_iteration: int = 0
    max_render_iterations: int = 3
