import json
from pathlib import Path

from agentic_runtime.evaluation.artifact_manifest import build_artifact_manifest, write_artifact_manifest
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


def test_artifact_manifest_includes_saved_reports(tmp_path: Path) -> None:
    manifest = build_artifact_manifest()

    assert manifest["artifact"]["name"] == "agentic-runtime"
    assert manifest["benchmark"]["workloads"]["support_triage_cases"] == 15
    assert manifest["benchmark_summary"]["variants"]["multi_agent_guarded"]["mean_ars"] == 0.8545
    assert any(
        record["path"] == "benchmarks/reports/pair-b-variant-comparison.json"
        for record in manifest["saved_files"]["reports"]
    )

    output_path = write_artifact_manifest(tmp_path / "artifact-manifest.json")
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.exists()
    assert payload["benchmark_summary"]["ablations"]["action_authorization_off"]["mean_task_success_rate"] == 20 / 30