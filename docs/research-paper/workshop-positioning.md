# Workshop Positioning Notes

## Core Positioning

This paper should be framed as a systems-oriented runtime and evaluation paper, not as a prompt engineering paper.

## Strongest Claims

- guardrails are more effective when integrated into orchestration checkpoints
- runtime structure changes agent behavior in measurable ways
- reproducible benchmark variants reveal reliability differences across agent configurations
- support triage and research-and-retrieval provide complementary workload evidence

## What To Emphasize

- public code artifact
- executable benchmark reports
- runtime-policy framing
- deterministic evaluation design
- practical relevance to production agent systems

## What Not To Overclaim

- do not claim broad generalization from six benchmark cases
- do not present ARS as a universal metric
- do not present LangGraph integration as a novel graph engine contribution

## Best-Fit Venue Language

Use phrasing such as:

- practical runtime architecture for reliable agent systems
- orchestration-level guardrail design
- reproducible evaluation framework for multi-agent reliability
- executable research prototype

## Reviewer Risk Areas

Prepare concise answers for:

- small benchmark size
- synthetic or deterministic workload construction
- normalized latency and cost rather than live operational measurements
- limited related work breadth in the current draft

## Suggested Near-Term Improvement Before Workshop Submission

1. add at least one non-happy-path case to each workload
2. strengthen the related work section with more recent agent systems and evaluation papers
3. export one architecture figure and one results figure
