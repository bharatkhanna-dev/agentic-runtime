from agentic_runtime.evaluation.cli import run_pair_b_ablation_benchmarks, run_pair_b_variant_benchmark
from agentic_runtime.evaluation.variants import BenchmarkVariant


def test_guarded_variant_outperforms_multi_agent_baseline_on_mean_ars() -> None:
    baseline = run_pair_b_variant_benchmark(BenchmarkVariant.MULTI_AGENT_BASELINE)
    guarded = run_pair_b_variant_benchmark(BenchmarkVariant.MULTI_AGENT_GUARDED)

    assert guarded["mean_ars"] > baseline["mean_ars"]
    assert guarded["support_triage"]["guardrail_compliance"] > baseline["support_triage"]["guardrail_compliance"]


def test_guarded_variant_outperforms_single_agent_on_mean_task_success() -> None:
    single_agent = run_pair_b_variant_benchmark(BenchmarkVariant.SINGLE_AGENT)
    guarded = run_pair_b_variant_benchmark(BenchmarkVariant.MULTI_AGENT_GUARDED)

    assert guarded["mean_task_success_rate"] > single_agent["mean_task_success_rate"]
    assert guarded["research_assistant"]["citation_recall"] > single_agent["research_assistant"]["citation_recall"]


def test_full_guardrails_outperform_ablations_on_target_metrics() -> None:
    results = run_pair_b_ablation_benchmarks()

    assert results["full_guardrails"]["mean_ars"] > results["action_authorization_off"]["mean_ars"]
    assert (
        results["full_guardrails"]["support_triage"]["guardrail_compliance"]
        > results["action_authorization_off"]["support_triage"]["guardrail_compliance"]
    )
    assert (
        results["full_guardrails"]["research_assistant"]["retrieval_budget_respected_rate"]
        > results["reasoning_check_off"]["research_assistant"]["retrieval_budget_respected_rate"]
    )