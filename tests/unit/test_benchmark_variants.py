from agentic_runtime.evaluation.cli import run_pair_b_variant_benchmark
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