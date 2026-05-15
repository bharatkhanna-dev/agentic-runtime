from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    git_dir = root / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return None

    head_value = head_path.read_text(encoding="utf-8").strip()
    if head_value.startswith("ref:"):
        ref_path = git_dir / head_value.split(" ", 1)[1]
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
        return None

    return head_value or None


def _package_version() -> str:
    try:
        return version("agentic-runtime")
    except PackageNotFoundError:
        return "0.1.0"


def _file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": _relative(path, root),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def build_artifact_manifest(project_root: Path | None = None) -> dict[str, object]:
    root = project_root or _project_root()
    report_dir = root / "benchmarks" / "reports"
    dataset_dir = root / "benchmarks" / "datasets"

    reports = [
        report_dir / "pair-b-runtime-baseline.json",
        report_dir / "pair-b-variant-comparison.json",
        report_dir / "pair-b-ablation-comparison.json",
    ]
    datasets = sorted(dataset_dir.glob("*.json"))
    supporting_files = [root / "README.md", root / "pyproject.toml"]
    dataset_counts = {
        path.stem: len(json.loads(path.read_text(encoding="utf-8")))
        for path in datasets
    }

    variant_report = json.loads((report_dir / "pair-b-variant-comparison.json").read_text(encoding="utf-8"))
    ablation_report = json.loads((report_dir / "pair-b-ablation-comparison.json").read_text(encoding="utf-8"))

    return {
        "artifact": {
            "name": "agentic-runtime",
            "version": _package_version(),
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "git_commit": _git_commit(root),
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "benchmark": {
            "name": "Pair B",
            "workloads": dataset_counts,
            "commands": [
                "python -m agentic_runtime.evaluation.cli run-pair-b-runtime",
                "python -m agentic_runtime.evaluation.cli run-pair-b-variants",
                "python -m agentic_runtime.evaluation.cli run-pair-b-ablations",
            ],
            "optional_live_baseline": {
                "command": "python -m agentic_runtime.evaluation.cli run-support-triage-live-openai --model gpt-5",
                "requires_env": ["OPENAI_API_KEY"],
                "status": "not_included_in_saved_artifact",
            },
        },
        "saved_files": {
            "reports": [_file_record(path, root) for path in reports],
            "datasets": [_file_record(path, root) for path in datasets],
            "supporting_files": [_file_record(path, root) for path in supporting_files],
        },
        "benchmark_summary": {
            "variants": {
                name: {
                    "mean_task_success_rate": payload["mean_task_success_rate"],
                    "mean_ars": payload["mean_ars"],
                }
                for name, payload in variant_report.items()
            },
            "ablations": {
                name: {
                    "mean_task_success_rate": payload["mean_task_success_rate"],
                    "mean_ars": payload["mean_ars"],
                }
                for name, payload in ablation_report.items()
            },
        },
    }


def write_artifact_manifest(output_path: Path | None = None, project_root: Path | None = None) -> Path:
    root = project_root or _project_root()
    destination = output_path or root / "benchmarks" / "reports" / "artifact-manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_artifact_manifest(root)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination