# Resume Tailoring System — Architecture Plan

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| **Language** | Python 3.12+ | Best LLM ecosystem, typing support |
| **LLM Provider** | Anthropic (hybrid model stack) | Strong instruction-following, low hallucination risk |
| **Orchestration** | Custom pipeline (`Pipeline` class) | No framework overhead — LangGraph/CrewAI add complexity we don't need for a linear pipeline. A simple class-based state machine gives us full control and easy debugging |
| **State** | Single `PipelineState` Pydantic model passed between agents | Type-safe, serializable, inspectable at every step |
| **Input** | Plain text / Markdown only | No PDF/DOCX parsing — keep data pipeline simple |
| **Output** | PDF (strictly required) | Polished, submittable output via `reportlab`. Programmatic rendering — no LLM call needed for formatting |
| **Config** | YAML files for agent prompts + model settings | Swap prompts/models without touching code |
| **Dependencies** | `anthropic`, `pydantic`, `pyyaml`, `reportlab` | Minimal. No bloated frameworks, no system deps |

### Model Allocation Strategy

| Agent | Model Tier | Model | Rationale |
|-------|-----------|-------|-----------|
| **Analyzer** | Fast / Cost-effective | `claude-haiku-4-20250514` | Extraction task — no deep reasoning needed |
| **Semantic Mapper** | High-capability | `claude-sonnet-4-20250514` | Deep semantic understanding required for accurate rewriting without hallucination |
| **Pruner** | Fast / Cost-effective | `claude-haiku-4-20250514` | Rule-based conciseness task — straightforward |
| **Critic** | High-capability | `claude-sonnet-4-20250514` | Factual drift detection demands strong reasoning |
| **Formatter** | No LLM | `reportlab` (programmatic PDF) | Structured data already available — pure code rendering saves cost and gives pixel-level control over layout (flush-right dates, clickable hyperlinks) |

## Data Flow

```
┌─────────────┐   ┌─────────────┐
│ Master Resume│   │     Job     │
│   (text)     │   │ Description │
└──────┬───────┘   └──────┬──────┘
       │                  │
       └──────┬───────────┘
              ▼
  ┌───────────────────────┐
  │   PipelineState       │  ← Created with raw inputs
  │  {                    │
  │    master_resume      │
  │    job_description    │
  │    analysis    : null │
  │    mapped_draft: null │
  │    pruned_draft: null │
  │    evaluation  : null │
  │    final_resume: null │
  │  }                    │
  └───────────┬───────────┘
              │
     ┌────────▼─────────┐
     │  1. ANALYZER      │  Reads: job_description
     │     Agent         │  Writes: analysis (structured JD breakdown)
     └────────┬──────────┘
              │
     ┌────────▼─────────┐
     │  2. SEMANTIC      │  Reads: analysis + master_resume
     │     MAPPER Agent  │  Writes: mapped_draft (selected & reworded bullets)
     └────────┬──────────┘
              │
     ┌────────▼─────────┐
     │  3. PRUNER        │  Reads: mapped_draft + analysis
     │     Agent         │  Writes: pruned_draft (concise, no redundancy)
     └────────┬──────────┘
              │
     ┌────────▼─────────┐
     │  4. CRITIC        │  Reads: pruned_draft + master_resume + analysis
     │     Agent         │  Writes: evaluation {approved: bool, issues: [...]}
     └────────┬──────────┘
              │
     ┌────────▼──────────────────────────┐
     │  5. RENDER-AND-REFINE LOOP        │
     │                                   │
     │  Formatter.test_render(state)     │
     │       ↓                           │
     │  Pages == 1? ──→ proceed          │
     │       ↓ (overflow)                │
     │  Inject overflow_pages into state │
     │  Re-run Pruner (aggressive mode)  │
     │  Loop back to test_render         │
     │       (max 3 iterations)          │
     └────────┬──────────────────────────┘
              │ (1-page confirmed)
              │
     ┌────────▼──────────┐
     │  HUMAN-IN-THE-    │  Pipeline HALTS here.
     │  LOOP GATE        │  User reviews Critic evaluation
     │                   │  + 1-page draft preview.
     │  Actions:         │
     │   [approve] →     │  Save final PDF
     │   [reject]  →     │  Exit with feedback
     │   [revise]  →     │  Re-enter Mapper loop
     └────────┬──────────┘
              │ (approved)
     ┌────────▼──────────┐
     │  6. FORMATTER     │  Reads: pruned_sections + contact header
     │  (reportlab)      │  Writes: polished single-page .pdf
     │  No LLM call      │  Flush-right dates, clickable hyperlinks,
     │                   │  education hierarchy, text sanitization
     └────────┬──────────┘
              │
              ▼
     ┌─────────────────┐
     │  Output: .pdf   │
     │  (exactly 1 pg) │
     └─────────────────┘
```

## Project Structure

```
job_bot_Claude/
├── architecture_plan.md
├── progress.md
├── config/
│   ├── models.yaml          # Model IDs, temperature, max_tokens per agent
│   └── prompts/
│       ├── analyzer.yaml
│       ├── mapper.yaml
│       ├── pruner.yaml
│       └── critic.yaml
├── src/
│   ├── __init__.py
│   ├── main.py              # CLI entry point
│   ├── pipeline.py          # Pipeline orchestrator class
│   ├── state.py             # PipelineState Pydantic model
│   ├── agent.py             # Base Agent class (wraps Anthropic API call)
│   └── agents/
│       ├── __init__.py
│       ├── analyzer.py
│       ├── mapper.py
│       ├── pruner.py
│       ├── critic.py
│       └── formatter.py
├── inputs/
│   ├── master_resume.md     # User places master resume here
│   └── job_description.md   # User places JD here
├── outputs/                 # Generated resumes land here
├── requirements.txt
└── README.md                # (only if requested)
```

## PipelineState Schema

```python
class JDAnalysis(BaseModel):
    job_title: str
    company_name: str
    hard_skills: list[str]          # e.g. ["Python", "Kubernetes", "CI/CD"]
    soft_skills: list[str]          # e.g. ["leadership", "cross-functional"]
    key_phrases: list[str]          # Verbatim high-value phrases from JD
    experience_years: str | None    # e.g. "5+ years"
    education_requirements: list[str]
    priority_ranking: list[str]     # Skills ranked by emphasis in JD

class ResumeSection(BaseModel):
    heading: str                    # e.g. "Work Experience", "Skills"
    entries: list[ResumeEntry]

class ResumeEntry(BaseModel):
    title: str                      # Role / degree / cert
    organization: str
    dates: str
    bullets: list[str]
    relevance_score: float | None   # Set by Mapper (0.0–1.0)

class Evaluation(BaseModel):
    approved: bool
    factual_drift_issues: list[str] # Specific bullets that deviate from source
    missing_keywords: list[str]     # Target keywords not integrated
    suggestions: list[str]          # Actionable fixes

class PipelineState(BaseModel):
    master_resume: str              # Raw input
    job_description: str            # Raw input
    analysis: JDAnalysis | None
    mapped_sections: list[ResumeSection] | None
    pruned_sections: list[ResumeSection] | None
    evaluation: Evaluation | None
    final_resume: str | None        # Formatted output
    revision_count: int = 0         # Track critic loops (max 2)
```

## Agent Design Pattern

Each agent follows the same contract:

```python
class BaseAgent:
    def __init__(self, config: AgentConfig):
        self.client = anthropic.Anthropic()
        self.model = config.model        # from models.yaml
        self.temperature = config.temperature
        self.system_prompt = config.system_prompt  # from prompts/<name>.yaml

    def run(self, state: PipelineState) -> PipelineState:
        """Each subclass builds its user_message from state,
           calls the API, parses the response, and mutates state."""
        user_message = self._build_prompt(state)
        response = self._call_llm(user_message)
        return self._parse_and_update(state, response)
```

Key design choices:
- **Structured output**: Each agent's prompt instructs the LLM to return JSON matching the expected Pydantic schema. We parse with `model_validate_json()` for type safety.
- **No shared memory/vector DB**: The state object IS the memory. Each agent sees exactly what it needs — no more, no less.
- **Retry with feedback**: If the Critic rejects, its `issues` and `suggestions` are injected into the Mapper's next prompt as explicit constraints.

## Agent Prompt Strategy (Summary)

| Agent | Input from State | Core Instruction | Output |
|-------|-----------------|------------------|--------|
| **Analyzer** | `job_description` | Extract structured requirements. Do NOT infer — only extract what is explicitly stated or strongly implied. | `JDAnalysis` JSON |
| **Mapper** | `analysis` + `master_resume` | Select and reword bullets to mirror JD vocabulary. You MUST preserve factual accuracy — change phrasing, never change facts. Every bullet must trace back to a source bullet in the master resume. | `list[ResumeSection]` JSON |
| **Pruner** | `mapped_sections` + `analysis` | Cut filler, compress, enforce action-verb starts. Remove sections/bullets with low relevance scores. Target ≤ 1 page density. | `list[ResumeSection]` JSON |
| **Critic** | `pruned_sections` + `master_resume` + `analysis` | Compare each bullet to its source for factual drift. Check keyword coverage against `analysis.hard_skills`. Flag issues or approve. | `Evaluation` JSON |
| **Formatter** | `pruned_sections` (approved) + contact header from `master_resume` | **No LLM call.** Programmatically renders a styled PDF using `reportlab` with Helvetica fonts, flush-right date alignment, clickable hyperlinks, section rules, and tight margins. | `.pdf` file |

## Render-and-Refine Loop

After the Critic, the pipeline enters a **closed-loop page-fit check**:

1. **Test Render**: The Formatter builds the PDF to an in-memory buffer (`BytesIO`) and counts pages via a page callback.
2. **Overflow Check**: If pages > 1, the Pruner is re-invoked with `overflow_pages` injected into state (e.g., `1.3` means 30% over). The Pruner's prompt includes aggressive condensing instructions proportional to the overflow.
3. **Iterate**: Steps 1–2 repeat (max 3 iterations) until the content fits in exactly 1 page.
4. **Constraint Injection**: The Mapper and Pruner prompts include spatial budgets (max bullets per role, max roles, character limits) so initial output is close to 1 page, minimizing refine iterations.

## Human-in-the-Loop Protocol

The pipeline implements a **"Propose → Render → Preview → Approve → Execute"** pattern:

1. **Propose**: Agents 1–4 (Analyzer → Mapper → Pruner → Critic) run automatically.
2. **Render-and-Refine**: The Formatter test-renders the PDF; if it overflows, the Pruner condenses and re-renders until it fits in 1 page.
3. **Preview**: The pipeline halts and displays:
   - The Critic's `Evaluation` (approval status, flagged issues, suggestions)
   - A text preview of the 1-page-confirmed draft
4. **Approve**: The user chooses one of three actions:
   - `approve` — Save the final PDF
   - `reject` — Abort the pipeline; no output is produced
   - `revise` — Re-enter the full Mapper → Pruner → Critic → Render loop
5. **Execute**: Only after human approval does the Formatter save the final `.pdf` file.

This gate is implemented in `pipeline.py` as a blocking input prompt during CLI execution.

## Extensibility Points

1. **Swap models**: Change `model` in `config/models.yaml` per agent — Haiku/Sonnet/Opus are all configurable per-agent
2. **Swap prompts**: Edit YAML files in `config/prompts/` — no code changes needed
3. **Add agents**: Create a new file in `src/agents/`, register it in the pipeline's agent list
4. **Add output formats**: Currently PDF only (strictly required). Future: add DOCX or Markdown fallback if needed
5. **Parallel execution**: Analyzer is the only agent with no resume dependency — future optimization could pre-process resume structure in parallel
6. **Input format expansion**: Currently plain text/Markdown. Future: add PDF/DOCX parsing as a preprocessing step

## Error Handling

- LLM returns invalid JSON → Retry once with a "fix your JSON" appended prompt
- Critic rejects after 2 revision cycles → Output the best draft with warnings attached
- API rate limit / failure → Exponential backoff (3 retries)
- Missing input files → Fail fast with clear error message

## Design Decisions Log

| # | Decision | Rationale | Date |
|---|----------|-----------|------|
| 1 | PDF-only output via ReportLab | Submittable, polished output with flush-right dates, clickable hyperlinks, sanitized name. Purely programmatic (no LLM cost) | 2026-03-22 |
| 2 | Hybrid model stack (Haiku + Sonnet) | Cost-effective for simple tasks; high-capability where semantic accuracy is critical | 2026-03-22 |
| 3 | Plain text/Markdown input only | Skip PDF/DOCX parsing to keep MVP data pipeline simple | 2026-03-22 |
| 4 | Human-in-the-loop after Critic | "Propose, preview, approve, execute" pattern — user reviews before final formatting | 2026-03-22 |
