# Benchmark Methodology

This note supports the manuscript and records the current evaluation setup for `agentic-runtime`.

## Workload Pair

The current benchmark program uses Pair B:

- support triage
- research and retrieval

The goal is to test transfer across two different runtime failure surfaces.

## Variants

Three variants are evaluated:

1. `single_agent`
2. `multi_agent_baseline`
3. `multi_agent_guarded`

## Support Triage Metrics

Each support triage case is scored on:

- severity correctness
- queue correctness
- action correctness
- approval-routing correctness
- tool-order correctness

A case counts as successful only if all of the above are correct.

## Research-and-Retrieval Metrics

Each research case is scored on:

- required-keyword coverage
- forbidden-keyword violations
- citation recall
- retrieval-budget compliance

A case counts as successful only if keyword coverage is full, forbidden-keyword violations are zero, citation recall is full, and the retrieval budget is respected.

## ARS Weights

The current prototype uses:

- task success: 0.35
- cost efficiency: 0.20
- latency score: 0.15
- guardrail compliance: 0.30

## Saved Report Artifacts

The current report artifacts are:

- `benchmarks/reports/pair-b-runtime-baseline.json`
- `benchmarks/reports/pair-b-variant-comparison.json`

## Current Interpretation

The saved comparison artifacts support a narrow but defensible conclusion:

- guarded orchestration materially improves benchmark performance over weaker baselines in the current Pair B suite
- the improvement transfers across both an operational workflow and a retrieval-heavy workflow
- the current prototype still needs more cases and richer live measurements before stronger generalization claims are warranted
