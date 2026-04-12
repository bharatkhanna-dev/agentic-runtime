# arXiv Metadata

## Title

Guardrails as Runtime Policy: An Orchestration Architecture and Evaluation Framework for Reliable Multi-Agent Systems

## Authors

Bharat Khanna

## Affiliation

Independent Researcher, Phoenix, United States

## Contact

khanna.bharat@gmail.com

## Abstract

Large language model agents are increasingly deployed as multi-step systems that plan, invoke tools, retrieve external context, and produce user-facing actions. In practice, however, many agent stacks still treat safety and reliability controls as wrappers around model input and output rather than as first-class runtime behavior. This separation creates a systems problem: an agent may produce a plausible final answer while still taking an unsafe, wasteful, or operationally incorrect trajectory. This paper presents `agentic-runtime`, a LangGraph-assisted runtime architecture that integrates guardrails into orchestration checkpoints rather than attaching them only at the edges of the application. The runtime centers on typed tool contracts, explicit run state, bounded execution, and policy decisions recorded during graph execution. We evaluate the design on two workload families chosen to expose different failure surfaces: support triage and research-and-retrieval. Across three benchmark variants, single-agent, multi-agent baseline, and multi-agent guarded, the guarded runtime achieves a mean task success rate of 1.0 and mean Agent Reliability Score (ARS) of 0.8545, compared with 0.5 and 0.5765 for the multi-agent baseline and 0.0 and 0.4395 for the single-agent baseline. The results suggest that orchestration-level guardrails materially improve approval routing, evidence discipline, and end-to-end reliability, while introducing measurable but bounded efficiency trade-offs. The paper contributes both a practical runtime architecture and an evaluation method for comparing agent systems on reliability, guardrail compliance, latency, and cost.

## Suggested Primary Category

- cs.AI

## Suggested Cross-Listings

- cs.SE
- cs.CL

## Suggested Comments

15 pages, 3 tables, public code artifact with executable benchmark reports.

## Suggested Keywords

- agent systems
- multi-agent orchestration
- guardrails
- runtime policy
- tool use
- evaluation
- retrieval-augmented generation
- LangGraph

## Code Artifact

- https://github.com/bharatkhanna-dev/agentic-runtime
