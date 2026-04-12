# Runtime Architecture

`agentic-runtime` is organized around one central idea: the runtime owns control, while models and graph execution remain replaceable parts.

## Layers

1. Orchestrator
   Manages run state, graph progress, retries, and termination.
2. Guardrails
   Applies policy at checkpoints such as planning, tool execution, and final response.
3. Tools
   Exposes typed contracts and permission levels.
4. Memory
   Stores bounded execution context and summaries.
5. Evaluation
   Scores task results, execution trajectories, safety events, latency, and cost.
6. Observability
   Emits logs, traces, and metrics for every run.

## Initial v0.1 design

The scaffold establishes:

- structured run state
- runtime node definitions
- typed tool specs
- guardrail event recording
- a minimal orchestration shell

LangGraph integration is intentionally deferred until the state model and guardrail boundaries are stable.
