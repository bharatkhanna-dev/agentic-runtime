from __future__ import annotations

import time

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
        self._run_start: float | None = None

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
        self._run_start = time.perf_counter()
        run_state.start_time = self._run_start
        run_state.status = RunStatus.RUNNING
        for node_id in run_state.ready_node_ids():
            run_state.nodes[node_id].status = NodeStatus.READY
        return run_state

    def validate_input(self, run_state: RunState) -> "InputGuardrailResult":
        """Run input validation before the first node executes.

        Records the result as a guardrail event. Invalid inputs (empty objective,
        oversized payload) transition the run to BLOCKED. Suspicious inputs
        (injection patterns) are logged with a high risk score but do not halt
        the run so the legitimate question can still be answered.
        """
        from agentic_runtime.guardrails.input_guardrail import InputGuardrail, InputGuardrailResult  # noqa: F401

        result = InputGuardrail().validate(run_state)
        decision = GuardrailDecision(
            checkpoint="input_validation",
            allowed=result.valid,
            risk_score=result.risk_score,
            confidence_score=0.95,
            rationale=result.reason,
        )
        self.record_guardrail(run_state, decision)
        return result

    def check_reasoning(self, run_state: RunState, *, token_budget: int = 500) -> "ReasoningGuardrailResult":
        """Evaluate whether the run should continue executing.

        Checks step limits, token budgets, and repeated checkpoint denials.
        Records the result as a guardrail event. Returns a result the caller
        can inspect to decide whether to dispatch the next node.
        """
        from agentic_runtime.guardrails.reasoning_guardrail import ReasoningGuardrail, ReasoningGuardrailResult  # noqa: F401

        result = ReasoningGuardrail(token_budget=token_budget).evaluate(run_state)
        decision = GuardrailDecision(
            checkpoint="reasoning_check",
            allowed=result.should_continue,
            risk_score=result.risk_score,
            confidence_score=0.95,
            rationale=result.reason,
        )
        self.record_guardrail(run_state, decision)
        return result

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
            # Use start_time stored on run_state so timing works across runtime instances
            t0 = self._run_start if self._run_start is not None else run_state.start_time
            if t0 is not None:
                run_state.runtime_observations["wall_clock_seconds"] = (
                    time.perf_counter() - t0
                )
            return

        run_state.status = RunStatus.RUNNING
        for node_id in run_state.ready_node_ids():
            if run_state.nodes[node_id].status == NodeStatus.PENDING:
                run_state.nodes[node_id].status = NodeStatus.READY
