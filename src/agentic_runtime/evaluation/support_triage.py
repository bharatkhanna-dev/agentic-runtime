from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SupportTriageCase(BaseModel):
    case_id: str
    title: str
    customer_tier: str
    message: str
    expected_severity: str
    expected_queue: str
    expected_action: str
    approval_required: bool = False
    expected_tool_sequence: list[str] = Field(default_factory=list)


class SupportTriagePrediction(BaseModel):
    severity: str
    queue: str
    action: str
    approval_requested: bool = False
    tool_sequence: list[str] = Field(default_factory=list)


class SupportTriageScore(BaseModel):
    severity_correct: bool
    queue_correct: bool
    action_correct: bool
    approval_correct: bool
    tool_order_correct: bool
    success: bool


def load_support_triage_cases(file_path: str | Path) -> list[SupportTriageCase]:
    raw_cases = json.loads(Path(file_path).read_text(encoding="utf-8"))
    return [SupportTriageCase.model_validate(item) for item in raw_cases]


def score_support_triage_case(
    case: SupportTriageCase,
    prediction: SupportTriagePrediction,
) -> SupportTriageScore:
    severity_correct = prediction.severity == case.expected_severity
    queue_correct = prediction.queue == case.expected_queue
    action_correct = prediction.action == case.expected_action
    approval_correct = prediction.approval_requested == case.approval_required
    tool_order_correct = prediction.tool_sequence == case.expected_tool_sequence
    success = all(
        [
            severity_correct,
            queue_correct,
            action_correct,
            approval_correct,
            tool_order_correct,
        ]
    )
    return SupportTriageScore(
        severity_correct=severity_correct,
        queue_correct=queue_correct,
        action_correct=action_correct,
        approval_correct=approval_correct,
        tool_order_correct=tool_order_correct,
        success=success,
    )


def summarize_support_triage_scores(scores: list[SupportTriageScore]) -> dict[str, Any]:
    total = len(scores)
    if total == 0:
        return {
            "total_cases": 0,
            "task_success_rate": 0.0,
            "approval_routing_correctness": 0.0,
            "tool_order_accuracy": 0.0,
        }

    success_count = sum(score.success for score in scores)
    approval_correct = sum(score.approval_correct for score in scores)
    tool_order_correct = sum(score.tool_order_correct for score in scores)
    return {
        "total_cases": total,
        "task_success_rate": success_count / total,
        "approval_routing_correctness": approval_correct / total,
        "tool_order_accuracy": tool_order_correct / total,
    }
