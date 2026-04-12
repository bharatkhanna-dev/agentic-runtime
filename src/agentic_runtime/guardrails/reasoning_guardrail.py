from __future__ import annotations

from pydantic import BaseModel

from agentic_runtime.orchestrator.models import RunState


class ReasoningGuardrailResult(BaseModel):
    """Result from evaluating whether a run should continue executing."""

    should_continue: bool
    risk_score: float = 0.0
    reason: str = ""
    anomaly_type: str | None = None


class ReasoningGuardrail:
    """Detects execution anomalies in an in-progress run.

    Checks performed:
    - Step-limit enforcement: halts when steps_taken >= max_steps.
    - Token-budget enforcement: halts when total_tokens >= token_budget.
    - Loop detection: halts when the same guardrail checkpoint has been
      denied loop_detection_threshold or more times, indicating the agent
      is stuck retrying a blocked path.
    """

    def __init__(
        self,
        *,
        loop_detection_threshold: int = 3,
        token_budget: int = 500,
    ) -> None:
        self.loop_detection_threshold = loop_detection_threshold
        self.token_budget = token_budget

    def evaluate(self, run_state: RunState) -> ReasoningGuardrailResult:
        """Evaluate whether execution should continue based on current run state."""
        if run_state.steps_taken >= run_state.max_steps:
            return ReasoningGuardrailResult(
                should_continue=False,
                risk_score=0.7,
                reason=f"Step limit of {run_state.max_steps} reached.",
                anomaly_type="step_limit",
            )

        if run_state.total_tokens >= self.token_budget:
            return ReasoningGuardrailResult(
                should_continue=False,
                risk_score=0.8,
                reason=(
                    f"Token budget of {self.token_budget} exhausted "
                    f"({run_state.total_tokens} tokens used)."
                ),
                anomaly_type="token_budget",
            )

        checkpoint_blocks: dict[str, int] = {}
        for event in run_state.guardrail_events:
            if not event.allowed:
                checkpoint_blocks[event.checkpoint] = checkpoint_blocks.get(event.checkpoint, 0) + 1

        for checkpoint, count in checkpoint_blocks.items():
            if count >= self.loop_detection_threshold:
                return ReasoningGuardrailResult(
                    should_continue=False,
                    risk_score=0.9,
                    reason=(
                        f"Loop detected: checkpoint '{checkpoint}' has been blocked "
                        f"{count} consecutive times."
                    ),
                    anomaly_type="loop_detected",
                )

        return ReasoningGuardrailResult(
            should_continue=True,
            risk_score=0.0,
            reason="Execution may continue.",
        )
