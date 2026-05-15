from __future__ import annotations

import json
import os
from pathlib import Path
from urllib import request

from agentic_runtime.evaluation.benchmark_runner import run_support_triage_benchmark
from agentic_runtime.evaluation.support_triage import SupportTriagePrediction, load_support_triage_cases


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _support_dataset_path() -> Path:
    return _project_root() / "benchmarks" / "datasets" / "support_triage_cases.json"


def run_support_triage_live_openai_benchmark(*, model: str, api_key: str | None = None) -> dict[str, object]:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required to run the live OpenAI support-triage baseline.")

    cases = load_support_triage_cases(_support_dataset_path())
    predictions: dict[str, SupportTriagePrediction] = {}

    for case in cases:
        response_text = _responses_text(
            api_key=key,
            model=model,
            instructions=(
                "You are a support triage assistant. Return only valid JSON with keys severity, queue, "
                "action, approval_requested, tool_sequence. Use severity in {critical, medium, low}; "
                "queue in {incident-response, billing-operations, general-support}; action in "
                "{escalate_incident, issue_refund, create_ticket}; approval_requested as true or false; "
                "and tool_sequence as an array of tool names."
            ),
            input_text=(
                f"case_id: {case.case_id}\n"
                f"title: {case.title}\n"
                f"customer_tier: {case.customer_tier}\n"
                f"message: {case.message}\n"
            ),
        )
        predictions[case.case_id] = SupportTriagePrediction.model_validate(_extract_json_object(response_text))

    result = run_support_triage_benchmark(cases=cases, predictions=predictions)
    result["variant"] = "support_triage_live_openai"
    result["model"] = model
    return result


def _responses_text(*, api_key: str, model: str, instructions: str, input_text: str) -> str:
    payload = json.dumps(
        {
            "model": model,
            "instructions": instructions,
            "input": input_text,
        }
    ).encode("utf-8")
    req = request.Request(
        OPENAI_RESPONSES_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=90) as response:
        body = json.loads(response.read().decode("utf-8"))

    if isinstance(body.get("output_text"), str) and body["output_text"].strip():
        return body["output_text"]

    output_parts: list[str] = []
    for item in body.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                output_parts.append(content["text"])
    if output_parts:
        return "\n".join(output_parts)

    raise RuntimeError("OpenAI Responses API returned no parseable text output.")


def _extract_json_object(raw_text: str) -> dict[str, object]:
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise RuntimeError("Model output did not contain a valid JSON object.") from None
        return json.loads(raw_text[start : end + 1])