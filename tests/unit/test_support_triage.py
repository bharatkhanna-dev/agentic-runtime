from pathlib import Path

from agentic_runtime.agents.support_triage import build_support_triage_nodes, default_support_triage_tools
from agentic_runtime.evaluation.support_triage import (
    SupportTriagePrediction,
    load_support_triage_cases,
    score_support_triage_case,
    summarize_support_triage_scores,
)
from agentic_runtime.orchestrator.models import ToolPermission


DATASET_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "datasets" / "support_triage_cases.json"


def test_support_triage_dataset_loads_with_expected_case_count() -> None:
    cases = load_support_triage_cases(DATASET_PATH)

    assert len(cases) == 3
    assert cases[0].expected_action == "escalate_incident"


def test_support_triage_scoring_marks_perfect_prediction_as_success() -> None:
    case = load_support_triage_cases(DATASET_PATH)[0]
    prediction = SupportTriagePrediction(
        severity="critical",
        queue="incident-response",
        action="escalate_incident",
        approval_requested=True,
        tool_sequence=["lookup_customer", "escalate_incident"],
    )

    score = score_support_triage_case(case, prediction)

    assert score.success is True
    assert score.tool_order_correct is True


def test_support_triage_summary_tracks_success_and_approval_accuracy() -> None:
    cases = load_support_triage_cases(DATASET_PATH)
    scores = [
        score_support_triage_case(
            cases[0],
            SupportTriagePrediction(
                severity="critical",
                queue="incident-response",
                action="escalate_incident",
                approval_requested=True,
                tool_sequence=["lookup_customer", "escalate_incident"],
            ),
        ),
        score_support_triage_case(
            cases[1],
            SupportTriagePrediction(
                severity="medium",
                queue="billing-operations",
                action="issue_refund",
                approval_requested=False,
                tool_sequence=["lookup_customer", "issue_refund"],
            ),
        ),
    ]

    summary = summarize_support_triage_scores(scores)

    assert summary["total_cases"] == 2
    assert summary["task_success_rate"] == 0.5
    assert summary["approval_routing_correctness"] == 0.5


def test_support_triage_workflow_contains_approval_required_tools() -> None:
    tools = {tool.name: tool for tool in default_support_triage_tools()}
    nodes = build_support_triage_nodes()

    assert tools["escalate_incident"].permission == ToolPermission.APPROVAL_REQUIRED
    assert tools["issue_refund"].permission == ToolPermission.APPROVAL_REQUIRED
    assert [node.node_id for node in nodes] == [
        "classify_issue",
        "decide_route",
        "finalize_triage",
    ]
