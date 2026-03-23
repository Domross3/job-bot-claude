# Progress Tracker

## Phase 1: Architecture & Design
- [x] Define tech stack
- [x] Design data flow pipeline
- [x] Define PipelineState schema
- [x] Define agent contracts and prompt strategy
- [x] Write architecture_plan.md
- [x] User decisions integrated (model allocation, HITL gate, Markdown-only, text input)
- [x] **Architecture approved — ready for implementation**

## Phase 2: Core Infrastructure
- [x] Set up project structure (dirs, requirements.txt)
- [x] Implement PipelineState + all data models (state.py)
- [x] Implement BaseAgent class with retry/JSON parsing (agent.py)
- [x] Create config YAML — models.yaml (hybrid Haiku/Sonnet stack)
- [x] Create config YAML — all 5 agent prompt files
- [x] Implement Pipeline orchestrator with HITL gate (pipeline.py)
- [x] Implement CLI entry point (main.py)
- [x] Install dependencies & verify all imports
- [x] **Phase 2 complete**

## Phase 3: Agent Implementation
- [x] Analyzer Agent (Haiku) — src/agents/analyzer.py
- [x] Semantic Mapper Agent (Sonnet) — src/agents/mapper.py
- [x] Pruner Agent (Haiku) — src/agents/pruner.py
- [x] Critic/Evaluator Agent (Sonnet) — src/agents/critic.py
- [x] Formatter Agent (Haiku) — src/agents/formatter.py
- [x] **Phase 3 complete**

## Phase 4: Integration & End-to-End Test
- [ ] Place sample master_resume.md + job_description.md in inputs/
- [ ] End-to-end run with real API calls
- [ ] Verify HITL gate works (approve/revise/reject flows)
- [ ] Verify output quality

## Phase 5: Polish
- [ ] Error handling edge cases
- [ ] Logging / verbose mode improvements
- [ ] Optional HTML/PDF output (deferred)
