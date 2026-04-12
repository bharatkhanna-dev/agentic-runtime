import json

from agentic_runtime.evaluation.cli import run_pair_b_gold_benchmark, run_pair_b_runtime_benchmark


def test_pair_b_cli_runner_returns_json_serializable_result() -> None:
    result = run_pair_b_gold_benchmark()

    payload = json.dumps(result)

    assert isinstance(payload, str)
    assert result["workload_count"] == 2
    assert result["support_triage"]["task_success_rate"] == 1.0
    assert result["research_assistant"]["task_success_rate"] == 1.0


def test_pair_b_runtime_benchmark_uses_runtime_backed_support_triage() -> None:
    result = run_pair_b_runtime_benchmark()

    payload = json.dumps(result)

    assert isinstance(payload, str)
    assert result["workload_count"] == 2
    assert result["support_triage"]["task_success_rate"] == 1.0
    assert result["research_assistant"]["task_success_rate"] == 1.0
    assert result["support_triage"]["cost_efficiency"] < 1.0
    assert result["support_triage"]["latency_score"] < 1.0
    assert result["research_assistant"]["cost_efficiency"] < 1.0
    assert result["research_assistant"]["latency_score"] < 1.0