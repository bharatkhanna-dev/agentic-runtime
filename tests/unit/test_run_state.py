from agentic_runtime.orchestrator.models import GuardrailDecision, NodeResult, NodeStatus, RunState, RunStatus, RuntimeNode, ToolPermission, ToolSpec
from agentic_runtime.orchestrator.runtime import AgenticRuntime


def test_run_state_marks_dependent_node_ready_after_parent_completes() -> None:
    runtime = AgenticRuntime()
    run_state = RunState(run_id="run-1", objective="triage an incoming support issue")

    runtime.add_node(
        run_state,
        RuntimeNode(
            node_id="classify",
            title="Classify issue",
            description="Classify the incoming issue.",
        ),
    )
    runtime.add_node(
        run_state,
        RuntimeNode(
            node_id="route",
            title="Route issue",
            description="Route the issue to the right queue.",
            depends_on=["classify"],
        ),
    )

    runtime.start(run_state)
    runtime.record_result(
        run_state,
        NodeResult(node_id="classify", status=NodeStatus.COMPLETED, output={"severity": "high"}),
    )

    assert run_state.nodes["route"].status == NodeStatus.READY
    assert run_state.status == RunStatus.RUNNING


def test_guardrail_decision_can_block_run() -> None:
    runtime = AgenticRuntime()
    run_state = RunState(run_id="run-2", objective="respond to a sensitive request")

    runtime.record_guardrail(
        run_state,
        GuardrailDecision(
            checkpoint="before_tool_call",
            allowed=False,
            risk_score=0.9,
            confidence_score=0.8,
            rationale="External action requires human approval.",
        ),
    )

    assert run_state.status == RunStatus.BLOCKED
    assert run_state.guardrail_events[-1].checkpoint == "before_tool_call"


def test_tool_registration_is_typed_and_discoverable() -> None:
    runtime = AgenticRuntime()
    runtime.register_tool(
        ToolSpec(
            name="create_ticket",
            description="Create a support ticket.",
            permission=ToolPermission.APPROVAL_REQUIRED,
        )
    )

    assert runtime.tools["create_ticket"].permission == ToolPermission.APPROVAL_REQUIRED


def test_runtime_blocks_approval_required_tool_without_approval() -> None:
    runtime = AgenticRuntime()
    run_state = RunState(run_id="run-3", objective="issue refund for billing error")
    runtime.register_tool(
        ToolSpec(
            name="issue_refund",
            description="Issue a refund to the customer.",
            permission=ToolPermission.APPROVAL_REQUIRED,
        )
    )

    decision = runtime.evaluate_tool_call(run_state, tool_name="issue_refund", approval_granted=False)

    assert decision.allowed is False
    assert run_state.status == RunStatus.BLOCKED
    assert run_state.runtime_observations["guardrail_blocks"] == 1.0


def test_runtime_allows_registered_tool_when_policy_is_satisfied() -> None:
    runtime = AgenticRuntime()
    run_state = RunState(run_id="run-4", objective="escalate outage correctly")
    runtime.register_tool(
        ToolSpec(
            name="escalate_incident",
            description="Escalate an outage.",
            permission=ToolPermission.APPROVAL_REQUIRED,
        )
    )

    decision = runtime.evaluate_tool_call(run_state, tool_name="escalate_incident", approval_granted=True)

    assert decision.allowed is True
    assert run_state.status == RunStatus.PENDING
    assert run_state.runtime_observations["guardrail_allows"] == 1.0
