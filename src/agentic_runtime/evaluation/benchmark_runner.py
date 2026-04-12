from __future__ import annotations

from typing import Any

from agentic_runtime.evaluation.ars import ARSWeights, compute_agent_reliability_score
from agentic_runtime.evaluation.research_assistant import (
    ResearchCase,
    ResearchPrediction,
    score_research_case,
    summarize_research_scores,
)
from agentic_runtime.evaluation.support_triage import (
    SupportTriageCase,
    SupportTriagePrediction,
    score_support_triage_case,
    summarize_support_triage_scores,
)


def run_support_triage_benchmark(
    *,
    cases: list[SupportTriageCase],
    predictions: dict[str, SupportTriagePrediction],
    weights: ARSWeights | None = None,
    runtime_observations: dict[str, float] | None = None,
) -> dict[str, Any]:
    scores = [score_support_triage_case(case, predictions[case.case_id]) for case in cases]
    summary = summarize_support_triage_scores(scores)
    observations = runtime_observations or {}
    cost_efficiency = observations.get("cost_efficiency", 1.0)
    latency_score = observations.get("latency_score", 1.0)
    guardrail_compliance = summary["approval_routing_correctness"]
    summary["guardrail_compliance"] = guardrail_compliance
    summary["cost_efficiency"] = cost_efficiency
    summary["latency_score"] = latency_score
    summary["workload"] = "support_triage"
    summary["ars"] = compute_agent_reliability_score(
        task_success=summary["task_success_rate"],
        cost_efficiency=cost_efficiency,
        latency_score=latency_score,
        guardrail_compliance=guardrail_compliance,
        weights=weights,
    )
    return summary


def run_research_benchmark(
    *,
    cases: list[ResearchCase],
    predictions: dict[str, ResearchPrediction],
    weights: ARSWeights | None = None,
    runtime_observations: dict[str, float] | None = None,
) -> dict[str, Any]:
    scores = [score_research_case(case, predictions[case.case_id]) for case in cases]
    summary = summarize_research_scores(scores)
    observations = runtime_observations or {}
    cost_efficiency = observations.get("cost_efficiency", 1.0)
    latency_score = observations.get("latency_score", 1.0)
    guardrail_compliance = (summary["citation_recall"] + summary["retrieval_budget_respected_rate"]) / 2
    summary["guardrail_compliance"] = guardrail_compliance
    summary["cost_efficiency"] = cost_efficiency
    summary["latency_score"] = latency_score
    summary["workload"] = "research_assistant"
    summary["ars"] = compute_agent_reliability_score(
        task_success=summary["task_success_rate"],
        cost_efficiency=cost_efficiency,
        latency_score=latency_score,
        guardrail_compliance=guardrail_compliance,
        weights=weights,
    )
    return summary


def run_pair_b_benchmarks(
    *,
    support_cases: list[SupportTriageCase],
    support_predictions: dict[str, SupportTriagePrediction],
    research_cases: list[ResearchCase],
    research_predictions: dict[str, ResearchPrediction],
    weights: ARSWeights | None = None,
    support_runtime_observations: dict[str, float] | None = None,
    research_runtime_observations: dict[str, float] | None = None,
) -> dict[str, Any]:
    support_summary = run_support_triage_benchmark(
        cases=support_cases,
        predictions=support_predictions,
        weights=weights,
        runtime_observations=support_runtime_observations,
    )
    research_summary = run_research_benchmark(
        cases=research_cases,
        predictions=research_predictions,
        weights=weights,
        runtime_observations=research_runtime_observations,
    )
    return {
        "workload_count": 2,
        "support_triage": support_summary,
        "research_assistant": research_summary,
        "mean_task_success_rate": (
            support_summary["task_success_rate"] + research_summary["task_success_rate"]
        ) / 2,
        "mean_ars": (support_summary["ars"] + research_summary["ars"]) / 2,
    }
