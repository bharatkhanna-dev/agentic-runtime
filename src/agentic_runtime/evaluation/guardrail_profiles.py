from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailProfile:
    name: str
    validate_input: bool = True
    reasoning_check: bool = True
    authorize_tools: bool = True
    retrieval_discipline: bool = True


FULL_GUARDRAIL_PROFILE = GuardrailProfile(name="full_guardrails")

INPUT_VALIDATION_OFF_PROFILE = GuardrailProfile(
    name="input_validation_off",
    validate_input=False,
)

REASONING_CHECK_OFF_PROFILE = GuardrailProfile(
    name="reasoning_check_off",
    reasoning_check=False,
    retrieval_discipline=False,
)

ACTION_AUTHORIZATION_OFF_PROFILE = GuardrailProfile(
    name="action_authorization_off",
    authorize_tools=False,
)


ABLATION_PROFILES: tuple[GuardrailProfile, ...] = (
    FULL_GUARDRAIL_PROFILE,
    INPUT_VALIDATION_OFF_PROFILE,
    REASONING_CHECK_OFF_PROFILE,
    ACTION_AUTHORIZATION_OFF_PROFILE,
)