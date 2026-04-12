from __future__ import annotations

import json
from pathlib import Path

from agentic_runtime.agents.research_assistant import execute_research_case
from agentic_runtime.agents.support_triage import execute_support_triage_case
from agentic_runtime.evaluation.benchmark_runner import run_pair_b_benchmarks
from agentic_runtime.evaluation.research_assistant import ResearchPrediction, load_research_cases
from agentic_runtime.evaluation.support_triage import SupportTriagePrediction, load_support_triage_cases
from agentic_runtime.evaluation.variants import BenchmarkVariant


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _support_dataset_path() -> Path:
    return _project_root() / "benchmarks" / "datasets" / "support_triage_cases.json"


def _research_dataset_path() -> Path:
    return _project_root() / "benchmarks" / "datasets" / "research_assistant_cases.json"


def build_gold_support_predictions() -> dict[str, SupportTriagePrediction]:
    return {
        case.case_id: SupportTriagePrediction(
            severity=case.expected_severity,
            queue=case.expected_queue,
            action=case.expected_action,
            approval_requested=case.approval_required,
            tool_sequence=case.expected_tool_sequence,
        )
        for case in load_support_triage_cases(_support_dataset_path())
    }


def build_runtime_support_benchmark_inputs(
    variant: BenchmarkVariant = BenchmarkVariant.MULTI_AGENT_GUARDED,
) -> tuple[dict[str, SupportTriagePrediction], dict[str, float]]:
    predictions: dict[str, SupportTriagePrediction] = {}
    total_cost_efficiency = 0.0
    total_latency_score = 0.0
    cases = load_support_triage_cases(_support_dataset_path())

    for case in cases:
        prediction, run_state = execute_support_triage_case(case, approval_granted=False, variant=variant)
        predictions[case.case_id] = prediction
        total_cost_efficiency += run_state.runtime_observations.get("cost_efficiency", 1.0)
        total_latency_score += run_state.runtime_observations.get("latency_score", 1.0)

    case_count = len(cases) or 1
    observations = {
        "cost_efficiency": total_cost_efficiency / case_count,
        "latency_score": total_latency_score / case_count,
    }
    return predictions, observations


def build_gold_research_predictions() -> dict[str, ResearchPrediction]:
    cases = load_research_cases(_research_dataset_path())
    return {
        "research-001": ResearchPrediction(
            answer="Prompt injection needs runtime policy before tool execution.",
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


def build_runtime_research_benchmark_inputs(
    variant: BenchmarkVariant = BenchmarkVariant.MULTI_AGENT_GUARDED,
) -> tuple[dict[str, ResearchPrediction], dict[str, float]]:
    predictions: dict[str, ResearchPrediction] = {}
    total_cost_efficiency = 0.0
    total_latency_score = 0.0
    cases = load_research_cases(_research_dataset_path())

    for case in cases:
        prediction, run_state = execute_research_case(case, variant=variant)
        predictions[case.case_id] = prediction
        total_cost_efficiency += run_state.runtime_observations.get("cost_efficiency", 1.0)
        total_latency_score += run_state.runtime_observations.get("latency_score", 1.0)

    case_count = len(cases) or 1
    observations = {
        "cost_efficiency": total_cost_efficiency / case_count,
        "latency_score": total_latency_score / case_count,
    }
    return predictions, observations


def run_pair_b_gold_benchmark() -> dict[str, object]:
    support_cases = load_support_triage_cases(_support_dataset_path())
    research_cases = load_research_cases(_research_dataset_path())
    return run_pair_b_benchmarks(
        support_cases=support_cases,
        support_predictions=build_gold_support_predictions(),
        research_cases=research_cases,
        research_predictions=build_gold_research_predictions(),
    )


def run_pair_b_runtime_benchmark() -> dict[str, object]:
    return run_pair_b_variant_benchmark(BenchmarkVariant.MULTI_AGENT_GUARDED)


def run_pair_b_variant_benchmark(variant: BenchmarkVariant) -> dict[str, object]:
    support_cases = load_support_triage_cases(_support_dataset_path())
    research_cases = load_research_cases(_research_dataset_path())
    support_predictions, support_observations = build_runtime_support_benchmark_inputs(variant)
    research_predictions, research_observations = build_runtime_research_benchmark_inputs(variant)
    result = run_pair_b_benchmarks(
        support_cases=support_cases,
        support_predictions=support_predictions,
        research_cases=research_cases,
        research_predictions=research_predictions,
        support_runtime_observations=support_observations,
        research_runtime_observations=research_observations,
    )
    result["variant"] = variant.value
    return result


def main() -> None:
    result = run_pair_b_runtime_benchmark()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()