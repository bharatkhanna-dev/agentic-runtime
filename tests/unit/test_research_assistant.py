from pathlib import Path

from agentic_runtime.agents.research_assistant import build_research_assistant_nodes, default_research_assistant_tools
from agentic_runtime.evaluation.research_assistant import (
    ResearchPrediction,
    load_research_cases,
    score_research_case,
    summarize_research_scores,
)
from agentic_runtime.orchestrator.models import ToolPermission


DATASET_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "datasets" / "research_assistant_cases.json"


def test_research_dataset_loads_with_expected_case_count() -> None:
    cases = load_research_cases(DATASET_PATH)

    assert len(cases) == 6
    assert cases[0].expected_citation_ids == ["doc-guardrails", "doc-runtime-policy"]


def test_research_scoring_marks_fully_grounded_prediction_as_success() -> None:
    case = load_research_cases(DATASET_PATH)[0]
    prediction = ResearchPrediction(
        answer="Prompt injection should be handled through runtime policy checks before tool execution, not only with edge filtering.",
        retrieved_document_ids=["doc-guardrails", "doc-runtime-policy"],
        cited_document_ids=["doc-guardrails", "doc-runtime-policy"],
    )

    score = score_research_case(case, prediction)

    assert score.success is True
    assert score.keyword_coverage == 1.0
    assert score.citation_recall == 1.0


def test_research_summary_tracks_grounding_and_budget() -> None:
    cases = load_research_cases(DATASET_PATH)
    scores = [
        score_research_case(
            cases[0],
            ResearchPrediction(
                answer="Prompt injection needs runtime policy checks before tool execution.",
                retrieved_document_ids=["doc-guardrails", "doc-runtime-policy"],
                cited_document_ids=["doc-guardrails", "doc-runtime-policy"],
            ),
        ),
        score_research_case(
            cases[1],
            ResearchPrediction(
                answer="Memory compaction reduces token cost but leaves out context pruning.",
                retrieved_document_ids=["doc-memory", "doc-cost", "doc-extra-1", "doc-extra-2", "doc-extra-3"],
                cited_document_ids=["doc-memory"],
            ),
        ),
    ]

    summary = summarize_research_scores(scores)

    assert summary["total_cases"] == 2
    assert summary["task_success_rate"] == 0.5
    assert summary["retrieval_budget_respected_rate"] == 0.5


def test_research_workflow_nodes_are_sequential_and_allow_only_safe_tools() -> None:
    tools = {tool.name: tool for tool in default_research_assistant_tools()}
    nodes = build_research_assistant_nodes()

    assert all(tool.permission == ToolPermission.ALLOW for tool in tools.values())
    assert [node.node_id for node in nodes] == [
        "retrieve_candidates",
        "select_evidence",
        "draft_grounded_answer",
    ]
