from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ARSWeights(BaseModel):
    task_success: float = Field(default=0.35, ge=0.0)
    cost_efficiency: float = Field(default=0.2, ge=0.0)
    latency_score: float = Field(default=0.15, ge=0.0)
    guardrail_compliance: float = Field(default=0.3, ge=0.0)

    @model_validator(mode="after")
    def validate_total_weight(self) -> "ARSWeights":
        total = self.task_success + self.cost_efficiency + self.latency_score + self.guardrail_compliance
        if total <= 0:
            raise ValueError("At least one ARS weight must be greater than zero.")
        return self


def compute_agent_reliability_score(
    *,
    task_success: float,
    cost_efficiency: float,
    latency_score: float,
    guardrail_compliance: float,
    weights: ARSWeights | None = None,
) -> float:
    active_weights = weights or ARSWeights()
    total_weight = (
        active_weights.task_success
        + active_weights.cost_efficiency
        + active_weights.latency_score
        + active_weights.guardrail_compliance
    )
    return (
        active_weights.task_success * task_success
        + active_weights.cost_efficiency * cost_efficiency
        + active_weights.latency_score * latency_score
        + active_weights.guardrail_compliance * guardrail_compliance
    ) / total_weight
