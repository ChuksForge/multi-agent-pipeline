# Multi-Agent Data Pipeline

> **Production-grade agentic data infrastructure** — ETL, anomaly detection, and report generation orchestrated by Claude through a LangGraph supervisor pattern.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-purple)](https://langchain-ai.github.io/langgraph/)
[![Claude](https://img.shields.io/badge/Claude-Sonnet%204-orange)](https://anthropic.com)
[![Tests](https://img.shields.io/badge/Tests-511%20passing-green)]()
[![Coverage](https://img.shields.io/badge/Coverage-84%25-yellowgreen)]()

---

## What This Solves

**The problem:** Most data teams spend 60–80% of their analyst time on the same three tasks — loading and cleaning data, checking it for anomalies, and writing the summary for someone else to read. These tasks require domain expertise to do well but are largely mechanical once the patterns are established. They're expensive to staff, slow to run, and inconsistent across analysts.

**The solution:** A multi-agent pipeline that accepts a natural language instruction — *"analyse last month's sales data and flag anything unusual"* — and autonomously handles the full workflow: ingesting the data, validating its quality, running statistical anomaly detection, generating charts, and producing a structured report. A human reviews the output, not the process.

**Who this is for:**
- **Data teams** who need consistent, automated reporting on recurring datasets (daily metrics, weekly sales, monthly financials)
- **ML platform teams** building internal tooling where data quality checks and anomaly detection need to be embedded in pipelines, not bolted on after
- **Engineering leaders** evaluating LLM-based automation for analyst workflows before committing to a larger build
- **Founders** in the data/AI space assessing what's genuinely possible with multi-agent orchestration today

---

## Business Impact

| Metric | Manual Process | This Pipeline |
|--------|---------------|---------------|
| Time from data to report | 2–4 hours (analyst) | 45–90 seconds |
| Cost per report (fully loaded) | $80–$200 (analyst time) | $0.003–$0.04 (API cost) |
| Consistency | Varies by analyst | Deterministic schema |
| Anomaly detection | Ad-hoc, often missed | Ensemble (IsolationForest + Z-score) every run |
| Audit trail | Email threads | Full typed state log per run |
| Scale | 1 analyst = N reports/day | Parallel execution, no bottleneck |

**Conservative ROI estimate:** A team running 20 routine reports per week at $80 average analyst cost = $83k/year. This pipeline at mixed model pricing ≈ $30/year in API costs. The real value is the analyst hours freed for higher-leverage work.

---

## Architecture

```
User Task (natural language)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Task Planner  (Claude Sonnet 4)                        │
│  Decomposes task → typed TaskPlan (Pydantic v2)         │
│  One structured-output LLM call                         │
└────────────────────┬────────────────────────────────────┘
                     │ TaskPlan
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Orchestrator  (LangGraph StateGraph Supervisor)        │
│  Routes subtasks · Manages retries · Signals END        │
│  Pure Python — zero LLM calls, deterministic routing    │
└──────┬──────────────┬──────────────┬────────────────────┘
       │              │              │
       ▼              ▼              ▼
┌──────────┐  ┌───────────────┐  ┌──────────────────┐
│ETL Agent │  │Analysis Agent │  │Report Agent      │
│Haiku     │  │Haiku          │  │Haiku             │
│          │  │               │  │                  │
│• DuckDB  │  │• polars stats │  │• MD formatter    │
│• polars  │  │• IsolationFst │  │• JSON emitter    │
│• Schema  │  │• Z-score      │  │• PDF renderer    │
│  infer   │  │• Vega-Lite    │  │• Executive       │
│• Validator│ │  chart specs  │  │  summary (LLM)   │
└──────────┘  └───────────────┘  └──────────────────┘
       │              │              │
       └──────────────┴──────────────┘
                     │
         ┌───────────▼───────────┐
         │  TokenTracker         │
         │  (LangChain callback) │
         │  Cost · Latency · Log │
         └───────────────────────┘
                     │
         ┌───────────▼───────────┐
         │  FastAPI Layer        │
         │  POST /api/v1/run     │
         │  GET  /api/v1/result  │
         │  GET  /api/v1/status  │
         └───────────────────────┘
```

**Key architectural decisions:**

- **Typed state over message passing** — All agents share a single `PipelineState` TypedDict with LangGraph reducers. Any agent can inspect prior results. Partial retries don't re-run earlier stages.
- **Planner separate from router** — Task decomposition (one Sonnet call) is distinct from routing logic (pure Python). The orchestrator is stateless and deterministic.
- **Three-tier failure recovery** — Every agent degrades gracefully: retry → simplified (sampled data) → degraded skeleton. The pipeline never hard-fails; it always produces annotated output.
- **Cost as first-class concern** — `TokenTracker` is a LangChain callback wired at the LLM constructor level. Cost is logged on every call, not as an afterthought.
- **Tools decoupled from agents** — Each tool is independently testable. DuckDB, polars, IsolationForest, and Vega-Lite are all invoked directly — no LLM generates the data or the numbers.

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Orchestration | LangGraph 0.2+ | Typed state, conditional edges, built-in checkpointing |
| LLM (planning) | Claude Sonnet 4 | Reliable structured JSON output, 200K context |
| LLM (execution) | Claude Haiku 4.5 | 6–8× cheaper than Sonnet for tool-call-heavy agents |
| Data ingestion | DuckDB + polars | DuckDB reads CSV/Parquet/JSON/Postgres through one SQL interface; polars for fast in-memory transforms |
| Anomaly detection | scikit-learn (IsolationForest) + scipy (Z-score) | Ensemble catches both multivariate and univariate outliers |
| Charts | Vega-Lite v5 specs | JSON specs are testable, portable, renderable anywhere |
| API | FastAPI + uvicorn | Async, typed, auto-documented |
| Validation | Pydantic v2 | All inter-agent contracts are validated models |
| Logging | structlog | Structured JSON logs with per-agent context binding |
| Testing | pytest + pytest-asyncio | 511 tests, 84% coverage |

---

## Model Tier Strategy & Cost

The system routes different agents to different model tiers based on reasoning requirements:

```
Planner      → Sonnet 4   (needs reliable structured output reasoning)
Orchestrator → Sonnet 4   (routing logic benefits from reasoning)
ETL Agent    → Haiku 4.5  (tool calls, not reasoning-heavy)
Analysis     → Haiku 4.5  (tool calls + light interpretation)
Report       → Haiku 4.5  (template-driven, low reasoning need)
```

**Benchmarked cost per full pipeline run:**

| Scenario | Avg Cost | vs All-Sonnet |
|----------|----------|---------------|
| All Sonnet 4 | ~$0.042 | baseline |
| **Mixed routing (default)** | **~$0.006** | **~7× cheaper** |
| All Haiku 4.5 | ~$0.001 | ~42× cheaper (lower quality) |

Run `python scripts/cost_benchmark.py --dry-run` to estimate costs before committing API budget.

---

## Project Structure

```
multi-agent-pipeline/
├── src/pipeline/
│   ├── core/               # Shared contracts — import from here, never from agents
│   │   ├── config.py       # pydantic-settings singleton
│   │   ├── exceptions.py   # 15 typed exception classes
│   │   ├── schemas.py      # All Pydantic v2 domain models
│   │   └── state.py        # PipelineState TypedDict + helpers
│   ├── middleware/
│   │   ├── cost_model.py   # Pricing table + compute_cost()
│   │   ├── logger.py       # structlog JSON/console
│   │   └── token_tracker.py # LangChain callback handler
│   ├── agents/
│   │   ├── etl/            # DuckDB, file reader, schema inference, validation
│   │   ├── analysis/       # Stats engine, anomaly detector, chart builder
│   │   └── report/         # MD formatter, JSON emitter, PDF renderer
│   ├── planner/            # Task decomposition → typed TaskPlan
│   ├── orchestrator/       # LangGraph StateGraph + supervisor routing
│   └── api/                # FastAPI — 5 endpoints, dependency injection
├── evals/
│   ├── fixtures/           # 15 predefined task JSONs (4 categories)
│   ├── harness.py          # Parallel task runner + JSONL result writer
│   └── metrics.py          # Completion rate, correctness score, cost/run
├── data/samples/           # 8 sample datasets for dev/demo
├── scripts/
│   ├── run_demo.py         # One-command demo with rich terminal output
│   ├── run_evals.py        # Full eval harness (--category, --parallel flags)
│   └── cost_benchmark.py   # Model tier cost comparison (--dry-run mode)
└── tests/                  # 511 tests: unit (tools) + integration (agents)
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/chuksforge/multi-agent-pipeline.git
cd multi-agent-pipeline
uv sync --extra dev          # or: pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

# 3. Run tests (no API key needed)
pytest tests/unit/ -v

# 4. Run the demo
python scripts/run_demo.py

# 5. Start the API
uvicorn pipeline.api.main:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

---

## API Reference

```bash
# Submit a task
curl -X POST http://localhost:8000/api/v1/run \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Load data/samples/sales_monthly.csv, detect revenue anomalies, output a markdown report",
    "output_format": "markdown"
  }'

# Response: {"task_id": "abc-123", "status": "complete", "message": "Pipeline completed in 42.3s"}

# Get the full result (includes report content)
curl http://localhost:8000/api/v1/result/abc-123

# Get lightweight status
curl http://localhost:8000/api/v1/status/abc-123

# List recent runs
curl http://localhost:8000/api/v1/runs

# Health check
curl http://localhost:8000/health
```

---

## Evaluation

The project ships with a production-style eval harness:

```bash
# Run all 15 fixture tasks
python scripts/run_evals.py

# Run by category
python scripts/run_evals.py --category full_pipeline
python scripts/run_evals.py --category etl_only

# Cost benchmark (estimate without API calls)
python scripts/cost_benchmark.py --dry-run

# Cost benchmark (live, 3 runs per scenario)
python scripts/cost_benchmark.py --runs 3
```

**Fixture coverage:**

| Category | Count | What it tests |
|----------|-------|---------------|
| ETL only | 4 | CSV, Parquet, dirty data, JSON ingestion |
| Analysis only | 4 | Stats, anomaly detection, clean/null data |
| Full pipeline | 4 | End-to-end, dirty data + recovery, multi-source |
| Edge cases | 3 | Empty file, malformed JSON, unsupported task |

---

## Failure Recovery

The pipeline never hard-fails. Every agent implements three-tier degradation:

| Tier | Trigger | Behaviour |
|------|---------|-----------|
| **Retry** | Transient error (timeout, rate limit) | Retry with exponential backoff, up to `MAX_RETRIES` |
| **Simplified** | Persistent data error | ETL: 5k-row sample. Analysis: Z-score only, 10% sample |
| **Degraded** | Both tiers exhausted | Return minimal valid result, annotate errors in output |

`status=PARTIAL` signals degraded output. `status=FAILED` only occurs when the Planner itself cannot produce a TaskPlan.

---

## Development

```bash
# Run tests with coverage
pytest --cov=src/pipeline --cov-report=html

# Lint
ruff check src/

# Type check
mypy src/

# Run specific phase tests
pytest tests/unit/test_etl_tools.py -v
pytest tests/integration/test_full_pipeline.py -v
```

---

## Roadmap

- [ ] **Async task queue** — Replace synchronous API execution with ARQ/Celery for true async job submission and polling
- [ ] **Postgres persistence** — Replace in-memory run store with a real database for result durability across restarts
- [ ] **Streaming responses** — Stream report content as it's generated via SSE
- [ ] **LangSmith tracing** — Add full trace visibility per pipeline run
- [ ] **Multi-source joins** — ETL agent support for joining across multiple data sources in a single DuckDB query
- [ ] **Dashboard** — React frontend consuming the existing API endpoints

---

## Built by ChuksForge AI Solutions Ltd.

Production-grade AI agents, tools, and applications.

**Website:** [chuksforge.com](https://chuksforge.com)
**GitHub:** [@ChuksForge](https://github.com/ChuksForge) · **Email:** [hello@chuksforge.com](mailto:hello@chuksforge.com) · **Telegram:** [@ChuksForge](https://t.me/ChuksForge)


## License

MIT

---
