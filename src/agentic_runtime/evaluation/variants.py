from __future__ import annotations

from enum import Enum


class BenchmarkVariant(str, Enum):
    SINGLE_AGENT = "single_agent"
    MULTI_AGENT_BASELINE = "multi_agent_baseline"
    MULTI_AGENT_GUARDED = "multi_agent_guarded"
