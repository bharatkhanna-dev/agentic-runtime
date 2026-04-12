from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    BLOCKED = "blocked"
    FAILED = "failed"
    COMPLETED = "completed"


class NodeStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    BLOCKED = "blocked"
    FAILED = "failed"
    COMPLETED = "completed"


class ToolPermission(str, Enum):
    ALLOW = "allow"
    APPROVAL_REQUIRED = "approval_required"
    DENY = "deny"


class ToolSpec(BaseModel):
    name: str
    description: str
    permission: ToolPermission = ToolPermission.ALLOW
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)


class RuntimeNode(BaseModel):
    node_id: str
    title: str
    description: str
    depends_on: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING


class NodeResult(BaseModel):
    node_id: str
    status: NodeStatus
    output: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    tokens_used: int = 0


class GuardrailDecision(BaseModel):
    checkpoint: str
    allowed: bool
    risk_score: float = 0.0
    confidence_score: float = 1.0
    rationale: str = ""


class RunState(BaseModel):
    run_id: str
    objective: str
    status: RunStatus = RunStatus.PENDING
    steps_taken: int = 0
    max_steps: int = 12
    total_tokens: int = 0
    start_time: float | None = None
    nodes: dict[str, RuntimeNode] = Field(default_factory=dict)
    results: dict[str, NodeResult] = Field(default_factory=dict)
    guardrail_events: list[GuardrailDecision] = Field(default_factory=list)
    runtime_observations: dict[str, float] = Field(default_factory=dict)
    summary: str = ""

    def ready_node_ids(self) -> list[str]:
        ready: list[str] = []
        for node in self.nodes.values():
            if node.status not in {NodeStatus.PENDING, NodeStatus.READY}:
                continue
            if all(
                self.results.get(dependency, NodeResult(node_id=dependency, status=NodeStatus.PENDING)).status
                == NodeStatus.COMPLETED
                for dependency in node.depends_on
            ):
                ready.append(node.node_id)
        return ready

    def can_continue(self) -> bool:
        return self.status in {RunStatus.PENDING, RunStatus.RUNNING} and self.steps_taken < self.max_steps
