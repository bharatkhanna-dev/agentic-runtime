from pathlib import Path

from agentic_runtime.agents.research_assistant import build_research_assistant_run_state, execute_research_case
from agentic_runtime.evaluation.research_assistant import load_research_cases
from agentic_runtime.orchestrator.models import RunStatus


DATASET_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "datasets" / "research_assistant_cases.json"


def test_research_runtime_builds_started_run_state() -> None:
    run_state = build_research_assistant_run_state(
        run_id="research-run-1",
        objective="Answer a grounded runtime question.",
    )

    assert run_state.status == RunStatus.RUNNING
    assert set(run_state.nodes.keys()) == {
        "retrieve_candidates",
        "select_evidence",
        "draft_grounded_answer",
    }


def test_execute_research_case_produces_grounded_prediction_and_runtime_observations() -> None:
    case = load_research_cases(DATASET_PATH)[1]

    prediction, run_state = execute_research_case(case)

    assert "memory compaction" in prediction.answer.lower()
    assert prediction.cited_document_ids == ["doc-memory", "doc-cost"]
    assert run_state.status == RunStatus.COMPLETED
    assert run_state.runtime_observations["guardrail_allows"] == 3.0
    assert run_state.runtime_observations["cost_efficiency"] > 0.0