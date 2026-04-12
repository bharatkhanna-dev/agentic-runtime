# agentic-runtime

A LangGraph-assisted runtime for reliable multi-agent systems. Integrates guardrails as orchestration-level policy checkpoints, with typed tool contracts, explicit run state, and a reproducible evaluation framework.

## Core idea

Most agent stacks treat safety controls as input/output wrappers. This project embeds guardrail decisions inside the execution graph so that every tool invocation is evaluated against the current run state before it proceeds.

> Guardrails should be runtime policy checkpoints, not edge-only filters.

## What this implements

- **Orchestrator** — owns RunState, node lifecycle, step limits, retries, and termination
- **Guardrail layer** — evaluates named checkpoints (`before_tool_call`, action routing, retrieval discipline) and records allow/deny decisions as structured events
- **Tool registry** — typed ToolSpec entries with explicit permission levels: `allow`, `approval_required`, `deny`
- **Workload executors** — support triage (approval gating, escalation, tool-order enforcement) and research-and-retrieval (retrieval budgets, citation recall, grounding)
- **Evaluation harness** — deterministic scorers, Agent Reliability Score (ARS), benchmark runner

## Repository layout

```
src/agentic_runtime/
    orchestrator/       # RunState, AgenticRuntime, models
    agents/             # support_triage, research_assistant executors
    evaluation/         # ARS, benchmark_runner, scorers, CLI
benchmarks/
    reports/            # saved JSON benchmark results
tests/
    unit/
    integration/
examples/
```

## Benchmarks

Three runtime variants evaluated on two workloads (Pair B):

| Variant | Task Success | ARS |
|---|---:|---:|
| Single-agent | 0.00 | 0.4395 |
| Multi-agent baseline | 0.50 | 0.5765 |
| Multi-agent guarded | 1.00 | 0.8545 |

Run benchmarks:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .[dev]
python -m agentic_runtime.evaluation.cli run-pair-b-runtime
```

## Tests

```powershell
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
