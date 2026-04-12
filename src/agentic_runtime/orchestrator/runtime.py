from __future__ import annotations

from agentic_runtime.orchestrator.models import (
    GuardrailDecision,
    NodeResult,
    NodeStatus,
    RunState,
    RunStatus,
    RuntimeNode,
    ToolPermission,
    ToolSpec,
)


class AgenticRuntime:
    """Small orchestration shell around structured run state.

    This intentionally keeps graph execution lightweight at scaffold stage.
    LangGraph integration will plug into this state model rather than replace it.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    @property
    def tools(self) -> dict[str, ToolSpec]:
        return dict(self._tools)

    def register_tool(self, tool: ToolSpec) -> None:
        self._tools[tool.name] = tool

    def add_node(self, run_state: RunState, node: RuntimeNode) -> None:
        run_state.nodes[node.node_id] = node
        if not node.depends_on:
            run_state.nodes[node.node_id].status = NodeStatus.READY

    def start(self, run_state: RunState) -> RunState:
        run_state.status = RunStatus.RUNNING
        for node_id in run_state.ready_node_ids():
            run_state.nodes[node_id].status = NodeStatus.READY
        return run_state

    def record_guardrail(self, run_state: RunState, decision: GuardrailDecision) -> None:
        run_state.guardrail_events.append(decision)
        run_state.runtime_observations["guardrail_checks"] = run_state.runtime_observations.get("guardrail_checks", 0.0) + 1
        if decision.allowed:
            run_state.runtime_observations["guardrail_allows"] = run_state.runtime_observations.get("guardrail_allows", 0.0) + 1
        else:
            run_state.runtime_observations["guardrail_blocks"] = run_state.runtime_observations.get("guardrail_blocks", 0.0) + 1
        if not decision.allowed:
            run_state.status = RunStatus.BLOCKED

    def evaluate_tool_call(
        self,
        run_state: RunState,
        *,
        tool_name: str,
        checkpoint: str = "before_tool_call",
        approval_granted: bool = False,
    ) -> GuardrailDecision:
        tool = self._tools.get(tool_name)
        if tool is None:
            decision = GuardrailDecision(
                checkpoint=checkpoint,
                allowed=False,
                risk_score=1.0,
                confidence_score=1.0,
                rationale=f"Tool '{tool_name}' is not registered.",
            )
            self.record_guardrail(run_state, decision)
            return decision

        if tool.permission == ToolPermission.DENY:
            decision = GuardrailDecision(
                checkpoint=checkpoint,
                allowed=False,
                risk_score=1.0,
                confidence_score=1.0,
                rationale=f"Tool '{tool_name}' is denied by runtime policy.",
            )
            self.record_guardrail(run_state, decision)
            return decision

        if tool.permission == ToolPermission.APPROVAL_REQUIRED and not approval_granted:
            decision = GuardrailDecision(
                checkpoint=checkpoint,
                allowed=False,
                risk_score=0.9,
                confidence_score=0.95,
                rationale=f"Tool '{tool_name}' requires approval before execution.",
            )
            self.record_guardrail(run_state, decision)
            return decision

        decision = GuardrailDecision(
            checkpoint=checkpoint,
            allowed=True,
            risk_score=0.1 if tool.permission == ToolPermission.ALLOW else 0.4,
            confidence_score=0.95,
            rationale=f"Tool '{tool_name}' is permitted for execution.",
        )
        self.record_guardrail(run_state, decision)
        return decision

    def record_result(self, run_state: RunState, result: NodeResult) -> None:
        run_state.results[result.node_id] = result
        run_state.total_tokens += result.tokens_used
        run_state.steps_taken += 1

        if result.node_id in run_state.nodes:
            run_state.nodes[result.node_id].status = result.status

        if result.status == NodeStatus.FAILED:
            run_state.status = RunStatus.FAILED
            return

        if all(node.status == NodeStatus.COMPLETED for node in run_state.nodes.values()):
            run_state.status = RunStatus.COMPLETED
            return

        run_state.status = RunStatus.RUNNING
        for node_id in run_state.ready_node_ids():
            if run_state.nodes[node_id].status == NodeStatus.PENDING:
                run_state.nodes[node_id].status = NodeStatus.READY
