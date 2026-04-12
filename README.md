# agentic-runtime

A LangGraph-assisted runtime for production-grade multi-agent systems with orchestration-level guardrails, typed tool contracts, structured run state, and reproducible evaluation.

## Why this exists

Most agent demos prove that a model can complete a task once. They do not prove that the system can be operated reliably with bounded steps, typed tool usage, risk controls, or repeatable evaluation. This project focuses on that runtime layer.

Core thesis:

> Guardrails should be integrated into orchestration as runtime policy checkpoints, not treated as edge-only filters.

## Initial scope

The first release cycle is scoped to two benchmark workloads:

- support triage
- research and retrieval

The runtime is built on LangGraph where graph execution helps, but the main contribution lives in:

- run-state design
- guardrail checkpoint placement
- typed tool boundaries
- reliability and cost controls
- benchmark methodology

## Repository layout

```text
src/agentic_runtime/
  orchestrator/
  agents/
  guardrails/
  memory/
  tools/
  evaluation/
  observability/
examples/
  support_triage/
  research_assistant/
benchmarks/
  datasets/
  configs/
  reports/
tests/
  unit/
  integration/
  regression/
```

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .[dev]
pytest
```

## Status

Research prototype stage. The repository now includes:

- runtime-backed Pair B benchmark workloads
- benchmark variant comparisons across `single_agent`, `multi_agent_baseline`, and `multi_agent_guarded`
- saved result artifacts under `benchmarks/reports/`
- a manuscript draft under `../docs/agentic-runtime-paper/`

## Research Artifacts

- Manuscript draft: `docs/research-paper/paper.md`
- Benchmark methodology: `docs/research-paper/benchmark-methodology.md`
- Submission abstract: `docs/research-paper/submission-abstract.md`
- Variant comparison report: `benchmarks/reports/pair-b-variant-comparison.json`
- Runtime baseline report: `benchmarks/reports/pair-b-runtime-baseline.json`
