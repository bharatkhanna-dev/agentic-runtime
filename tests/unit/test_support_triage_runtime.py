from pathlib import Path

from agentic_runtime.agents.support_triage import (
    execute_support_triage_case,
    build_support_triage_run_state,
    default_support_triage_tools,
    evaluate_support_triage_action,
)
from agentic_runtime.evaluation.support_triage import load_support_triage_cases
from agentic_runtime.orchestrator.models import RunStatus
from agentic_runtime.orchestrator.runtime import AgenticRuntime


DATASET_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "datasets" / "support_triage_cases.json"


def test_support_triage_action_requires_approval_for_refund() -> None:
    runtime = AgenticRuntime()
    for tool in default_support_triage_tools():
        runtime.register_tool(tool)
    run_state = build_support_triage_run_state(
        run_id="triage-run-1",
        objective="Handle duplicate charge refund request.",
    )

    allowed = evaluate_support_triage_action(
        runtime,
        run_state,
        action="issue_refund",
        approval_granted=False,
    )

    assert allowed is False
    assert run_state.runtime_observations["guardrail_blocks"] == 1.0


def test_support_triage_action_allows_safe_ticket_creation() -> None:
    runtime = AgenticRuntime()
    for tool in default_support_triage_tools():
        runtime.register_tool(tool)
    run_state = build_support_triage_run_state(
        run_id="triage-run-2",
        objective="Handle password reset support request.",
    )

    allowed = evaluate_support_triage_action(
        runtime,
        run_state,
        action="create_ticket",
        approval_granted=False,
    )

    assert allowed is True
    assert run_state.runtime_observations["guardrail_allows"] == 1.0


def test_execute_support_triage_case_requests_approval_for_critical_incident() -> None:
    case = load_support_triage_cases(DATASET_PATH)[0]

    prediction, run_state = execute_support_triage_case(case, approval_granted=False)

    assert prediction.severity == "critical"
    assert prediction.queue == "incident-response"
    assert prediction.action == "escalate_incident"
    assert prediction.approval_requested is True
    assert prediction.tool_sequence == ["lookup_customer", "escalate_incident"]
    assert run_state.runtime_observations["guardrail_blocks"] == 1.0
    assert run_state.runtime_observations["cost_efficiency"] > 0.0


def test_execute_support_triage_case_completes_safe_ticket_path() -> None:
    case = load_support_triage_cases(DATASET_PATH)[2]

    prediction, run_state = execute_support_triage_case(case, approval_granted=False)

    assert prediction.severity == "low"
    assert prediction.queue == "general-support"
    assert prediction.action == "create_ticket"
    assert prediction.approval_requested is False
    assert run_state.status == RunStatus.COMPLETED
    assert run_state.runtime_observations["guardrail_allows"] >= 2.0