from pathlib import Path

from agentic_runtime.evaluation.ars import ARSWeights, compute_agent_reliability_score
from agentic_runtime.evaluation.benchmark_runner import (
    run_pair_b_benchmarks,
    run_research_benchmark,
    run_support_triage_benchmark,
)
from agentic_runtime.evaluation.research_assistant import ResearchPrediction, load_research_cases
from agentic_runtime.evaluation.support_triage import SupportTriagePrediction, load_support_triage_cases


SUPPORT_DATASET_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "datasets" / "support_triage_cases.json"
RESEARCH_DATASET_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "datasets" / "research_assistant_cases.json"


def test_ars_respects_custom_weights() -> None:
    score = compute_agent_reliability_score(
        task_success=1.0,
        cost_efficiency=0.5,
        latency_score=0.5,
        guardrail_compliance=1.0,
        weights=ARSWeights(task_success=0.5, cost_efficiency=0.1, latency_score=0.1, guardrail_compliance=0.3),
    )

    assert round(score, 3) == 0.9


def test_support_triage_benchmark_runner_returns_expected_summary() -> None:
    cases = load_support_triage_cases(SUPPORT_DATASET_PATH)[:3]  # use first 3 cases for isolated scoring test
    predictions = {
        "triage-001": SupportTriagePrediction(
            severity="critical",
            queue="incident-response",
            action="escalate_incident",
            approval_requested=True,
            tool_sequence=["lookup_customer", "escalate_incident"],
        ),
        "triage-002": SupportTriagePrediction(
            severity="medium",
            queue="billing-operations",
            action="issue_refund",
            approval_requested=False,
            tool_sequence=["lookup_customer", "issue_refund"],
        ),
        "triage-003": SupportTriagePrediction(
            severity="low",
            queue="general-support",
            action="create_ticket",
            approval_requested=False,
            tool_sequence=["lookup_customer", "create_ticket"],
        ),
    }

    summary = run_support_triage_benchmark(
        cases=cases,
        predictions=predictions,
        runtime_observations={"cost_efficiency": 0.8, "latency_score": 0.9},
    )

    assert summary["workload"] == "support_triage"
    assert summary["task_success_rate"] == 2 / 3
    assert summary["approval_routing_correctness"] == 2 / 3
    assert 0.0 <= summary["ars"] <= 1.0


def test_research_benchmark_runner_returns_expected_summary() -> None:
    cases = load_research_cases(RESEARCH_DATASET_PATH)[:3]  # use first 3 cases for isolated scoring test
    predictions = {
        "research-001": ResearchPrediction(
            answer="Prompt injection should be handled with runtime policy before tool execution.",
            retrieved_document_ids=["doc-guardrails", "doc-runtime-policy"],
            cited_document_ids=["doc-guardrails", "doc-runtime-policy"],
        ),
        "research-002": ResearchPrediction(
            answer="Memory compaction reduces token cost and supports context pruning.",
            retrieved_document_ids=["doc-memory", "doc-cost"],
            cited_document_ids=["doc-memory", "doc-cost"],
        ),
        "research-003": ResearchPrediction(
            answer="Typed tool contracts and schema improve runtime reliability.",
            retrieved_document_ids=["doc-tools", "doc-reliability"],
            cited_document_ids=["doc-tools", "doc-reliability"],
        ),
    }

    summary = run_research_benchmark(
        cases=cases,
        predictions=predictions,
        runtime_observations={"cost_efficiency": 0.75, "latency_score": 0.85},
    )

    assert summary["workload"] == "research_assistant"
    assert summary["task_success_rate"] == 1.0
    assert summary["citation_recall"] == 1.0
    assert 0.0 <= summary["ars"] <= 1.0


def test_pair_b_runner_combines_both_workloads() -> None:
    support_cases = load_support_triage_cases(SUPPORT_DATASET_PATH)
    research_cases = load_research_cases(RESEARCH_DATASET_PATH)
    result = run_pair_b_benchmarks(
        support_cases=support_cases,
        support_predictions={
            case.case_id: SupportTriagePrediction(
                severity=case.expected_severity,
                queue=case.expected_queue,
                action=case.expected_action,
                approval_requested=case.approval_required,
                tool_sequence=case.expected_tool_sequence,
            )
            for case in support_cases
        },
        research_cases=research_cases,
        research_predictions={
            "research-001": ResearchPrediction(
                answer="Prompt injection requires runtime policy before tool execution.",
                retrieved_document_ids=["doc-guardrails", "doc-runtime-policy"],
                cited_document_ids=["doc-guardrails", "doc-runtime-policy"],
            ),
            "research-002": ResearchPrediction(
                answer="Memory compaction reduces token cost and supports context pruning.",
                retrieved_document_ids=["doc-memory", "doc-cost"],
                cited_document_ids=["doc-memory", "doc-cost"],
            ),
            "research-003": ResearchPrediction(
                answer="Typed tool contracts and schema improve runtime reliability.",
                retrieved_document_ids=["doc-tools", "doc-reliability"],
                cited_document_ids=["doc-tools", "doc-reliability"],
            ),
            "research-004": ResearchPrediction(
                answer="Race condition and state isolation prevent concurrent access violations.",
                retrieved_document_ids=["doc-concurrency", "doc-isolation"],
                cited_document_ids=["doc-concurrency", "doc-isolation"],
            ),
            "research-005": ResearchPrediction(
                answer="Enforce retrieval budget and pre-execution check within context window.",
                retrieved_document_ids=["doc-rag-checks"],
                cited_document_ids=["doc-rag-checks"],
            ),
            "research-006": ResearchPrediction(
                answer="Input validation guardrail enforces runtime policy before tool calls.",
                retrieved_document_ids=["doc-guardrails", "doc-input-validation"],
                cited_document_ids=["doc-guardrails", "doc-input-validation"],
            ),
            "research-007": ResearchPrediction(
                answer="An approval workflow adds a human-in-the-loop checkpoint before any irreversible action.",
                retrieved_document_ids=["doc-approval", "doc-governance"],
                cited_document_ids=["doc-approval", "doc-governance"],
            ),
            "research-008": ResearchPrediction(
                answer="A retrieval budget limits context bloat and improves token efficiency.",
                retrieved_document_ids=["doc-budget", "doc-token-efficiency"],
                cited_document_ids=["doc-budget", "doc-token-efficiency"],
            ),
            "research-009": ResearchPrediction(
                answer="Input validation at the runtime boundary should detect instruction override attempts.",
                retrieved_document_ids=["doc-input-validation", "doc-runtime-boundary"],
                cited_document_ids=["doc-input-validation", "doc-runtime-boundary"],
            ),
            "research-010": ResearchPrediction(
                answer="An audit log of each guardrail event supports post-incident review.",
                retrieved_document_ids=["doc-audit", "doc-guardrail-event"],
                cited_document_ids=["doc-audit", "doc-guardrail-event"],
            ),
            "research-011": ResearchPrediction(
                answer="Runtime state should track approval expiry and delegated authority for approval-sensitive actions.",
                retrieved_document_ids=["doc-approval-expiry", "doc-delegated-authority"],
                cited_document_ids=["doc-approval-expiry", "doc-delegated-authority"],
            ),
            "research-012": ResearchPrediction(
                answer="A tool deny list strengthens runtime authorization at the action boundary.",
                retrieved_document_ids=["doc-deny-list", "doc-runtime-authorization"],
                cited_document_ids=["doc-deny-list", "doc-runtime-authorization"],
            ),
            "research-013": ResearchPrediction(
                answer="Loop detection and a reasoning check enforce a step budget before repeated retries continue.",
                retrieved_document_ids=["doc-loop-detection", "doc-step-budget"],
                cited_document_ids=["doc-loop-detection", "doc-step-budget"],
            ),
            "research-014": ResearchPrediction(
                answer="A grounded answer should maximize citation recall by citing selected evidence.",
                retrieved_document_ids=["doc-grounding", "doc-selected-evidence"],
                cited_document_ids=["doc-grounding", "doc-selected-evidence"],
            ),
            "research-015": ResearchPrediction(
                answer="After a blocked action, the runtime should route to a fallback path or human escalation.",
                retrieved_document_ids=["doc-blocked-action", "doc-fallback"],
                cited_document_ids=["doc-blocked-action", "doc-fallback"],
            ),
        },
    )

    assert result["workload_count"] == 2
    assert result["support_triage"]["task_success_rate"] == 1.0
    assert result["research_assistant"]["task_success_rate"] == 1.0
    assert result["mean_ars"] == 1.0
