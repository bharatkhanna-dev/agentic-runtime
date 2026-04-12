# agentic-runtime

A LangGraph-assisted runtime for reliable multi-agent systems. Integrates guardrails as orchestration-level policy checkpoints, with typed tool contracts, explicit run state, and a reproducible evaluation framework.

## Core idea

Most agent stacks treat safety controls as input/output wrappers. This project embeds guardrail decisions inside the execution graph so that every tool invocation is evaluated against the current run state before it proceeds.

> Guardrails should be runtime policy checkpoints, not edge-only filters.

## What this implements

- **Orchestrator** — owns RunState, node lifecycle, step limits, retries, and termination; records wall-clock timing
- **Guardrail layer** — evaluates named checkpoints (`input_validation`, `reasoning_check`, `before_tool_call`, action routing, retrieval discipline) and records allow/deny decisions as structured events
- **Tool registry** — typed ToolSpec entries with explicit permission levels: `allow`, `approval_required`, `deny`
- **Workload executors** — support triage (approval gating, escalation, tool-order enforcement) and research-and-retrieval (retrieval budgets, citation recall, grounding)
- **Evaluation harness** — deterministic scorers, configurable Agent Reliability Score (ARS), benchmark runner

## Repository layout

```
src/agentic_runtime/
    orchestrator/       # RunState, AgenticRuntime, models
    agents/             # support_triage, research_assistant executors
    guardrails/         # InputGuardrail, ReasoningGuardrail
    evaluation/         # ARS, benchmark_runner, scorers, CLI
benchmarks/
    datasets/           # support_triage_cases.json, research_assistant_cases.json
    reports/            # saved JSON benchmark results
tests/
    unit/
    integration/
```

## Benchmarks

Three runtime variants evaluated on two workloads (Pair B), six cases each:

| Variant | Task Success | ARS |
|---|---:|---:|
| Single-agent | 0.00 | 0.4458 |
| Multi-agent baseline | 0.42 | 0.5473 |
| Multi-agent guarded | 1.00 | 0.8545 |

Run benchmarks:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .[dev]
python -m agentic_runtime.evaluation.cli run-pair-b-runtime
python -m agentic_runtime.evaluation.cli run-pair-b-variants
```

## Tests

```powershell
pytest
```

## Status

The repository contains:

- Runtime-backed Pair B benchmark workloads (6 cases per workload including adversarial cases)
- Benchmark variant comparisons across `single_agent`, `multi_agent_baseline`, and `multi_agent_guarded`
- `InputGuardrail` and `ReasoningGuardrail` classes for pre-execution and mid-run policy enforcement
- Wall-clock timing and token-count instrumentation in `RunState`
- Configurable `ARSWeights` for weight-sensitivity analysis
- Saved result artifacts under `benchmarks/reports/`

## Research Paper

**Guardrails as Runtime Policy: An Orchestration Architecture and Evaluation Framework for Reliable Multi-Agent Systems**  
Bharat Khanna, Independent Researcher, Phoenix, United States  
Manuscript under review.

