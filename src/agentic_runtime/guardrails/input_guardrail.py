from __future__ import annotations

from pydantic import BaseModel

from agentic_runtime.orchestrator.models import RunState


class InputGuardrailResult(BaseModel):
    """Result from validating a run's input before the first node executes."""

    valid: bool
    risk_score: float = 0.0
    reason: str = ""
    injection_detected: bool = False


class InputGuardrail:
    """Validates task structure and detects prompt-injection patterns in the objective.

    Rules:
    - Returns valid=False for structurally malformed inputs (empty run_id, empty or
      oversized objective). The runtime records these as denied guardrail events and the
      run transitions to BLOCKED.
    - Returns valid=True with a high risk_score when injection patterns are detected.
      The run continues so the legitimate question can still be answered, but the
      detection is logged as a guardrail event for audit purposes.
    """

    INJECTION_PATTERNS: tuple[str, ...] = (
        "ignore previous instructions",
        "disregard safety rules",
        "override runtime policy",
    )
    MAX_OBJECTIVE_LENGTH: int = 4096

    def validate(self, run_state: RunState) -> InputGuardrailResult:
        """Validate the run's input fields before execution begins."""
        if not run_state.run_id.strip():
            return InputGuardrailResult(
                valid=False,
                risk_score=1.0,
                reason="run_id must not be empty.",
            )

        if not run_state.objective.strip():
            return InputGuardrailResult(
                valid=False,
                risk_score=1.0,
                reason="Objective must not be empty.",
            )

        if len(run_state.objective) > self.MAX_OBJECTIVE_LENGTH:
            return InputGuardrailResult(
                valid=False,
                risk_score=0.8,
                reason=(
                    f"Objective length {len(run_state.objective)} exceeds maximum of "
                    f"{self.MAX_OBJECTIVE_LENGTH} characters."
                ),
            )

        lower_objective = run_state.objective.lower()
        for pattern in self.INJECTION_PATTERNS:
            if pattern in lower_objective:
                return InputGuardrailResult(
                    valid=True,
                    risk_score=0.9,
                    reason=(
                        f"Potential prompt injection detected: '{pattern}'. "
                        "Task will proceed with sanitized context."
                    ),
                    injection_detected=True,
                )

        return InputGuardrailResult(
            valid=True,
            risk_score=0.0,
            reason="Input validation passed.",
        )
