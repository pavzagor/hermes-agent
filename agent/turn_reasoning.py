"""Host-owned per-turn effort policy; plugins propose data, never mutate agents.

Explicit request/session provenance wins even when its value equals config.
Only session choices are durable; automatic effort is reset before every hook.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict

from hermes_constants import resolve_reasoning_config, resolve_per_model_reasoning_effort
from agent import turn_routing

logger = logging.getLogger(__name__)
ReasoningProvenance = Literal['request', 'session', 'automatic', 'config']


class ReasoningDirective(TypedDict):
    effort: Literal['low', 'medium', 'high', 'xhigh']


class PreLLMReasoningContext(TypedDict, total=False):
    reasoning_routing_enabled: bool
    reasoning_pinned: bool


class ModelDirective(TypedDict):
    model: str


class PreLLMResult(TypedDict, total=False):
    context: str
    route: ModelDirective
    reasoning: ReasoningDirective
    skills: list[str]


def _config() -> dict:
    from hermes_cli.config_effective import load_user_config_effective
    return load_user_config_effective()


def _safe_config() -> dict:
    try:
        return _config() or {}
    except Exception:
        logger.warning('Reasoning config unavailable; automation disabled', exc_info=True)
        return {}


def routing_configured() -> bool:
    return (_safe_config().get('reasoning_routing') or {}).get('enabled') is True


def set_session_reasoning(session_id: str | None, config: dict | None) -> bool:
    if not isinstance(session_id, str) or not session_id:
        return False
    state = turn_routing._read_state(session_id) or {}
    state['reasoning_pin'] = dict(config) if config is not None else None
    turn_routing._state_overrides[turn_routing._path(session_id)] = state
    return turn_routing._write_state(session_id, state)


def reasoning_pinned(agent: Any) -> bool:
    overrides = getattr(agent, 'request_overrides', None) or {}
    extra = overrides.get('extra_body') or {}
    if any(k in overrides or k in extra for k in ('reasoning', 'reasoning_effort', 'thinking')):
        return True
    return getattr(agent, 'reasoning_provenance', 'config') in ('request', 'session')


def refresh_reasoning(agent: Any) -> None:
    """Model/fallback changes re-resolve config, never an explicit effort choice."""
    if reasoning_pinned(agent):
        return
    agent.reasoning_config = resolve_reasoning_config(_safe_config(), agent.model)
    agent.reasoning_provenance = 'config'


def prepare_turn(agent: Any) -> None:
    if getattr(agent, 'reasoning_provenance', 'config') == 'request':
        return
    state = turn_routing._read_state(agent.session_id) or {}
    pin = state.get('reasoning_pin')
    if isinstance(pin, dict):
        agent.reasoning_config = dict(pin)
        agent.reasoning_provenance = 'session'
    else:
        agent.reasoning_provenance = 'config'
        refresh_reasoning(agent)


def apply_reasoning(agent: Any, directive: Any) -> bool:
    if not isinstance(directive, dict) or set(directive) != {'effort'}:
        return False
    effort = directive['effort']
    if not isinstance(effort, str) or effort not in ('low', 'medium', 'high', 'xhigh'):
        return False
    cfg = _safe_config()
    policy = cfg.get('reasoning_routing') or {}
    if policy.get('enabled') is not True or reasoning_pinned(agent):
        return False
    # Initial verified capability set. Unknown providers/models fail closed.
    from agent.model_metadata import strip_codex_context_variant_suffix
    model = strip_codex_context_variant_suffix(agent.model)
    if agent.provider != 'openai-codex' or model not in ('gpt-6-astra', 'gpt-5.6-sol', 'gpt-5.6-luna'):
        return False
    explicit = resolve_per_model_reasoning_effort(agent.model, (cfg.get('agent') or {}).get('reasoning_overrides'))
    allowed_overrides = (policy.get('override_models') or {}).get(agent.provider, [])
    if explicit is not None and agent.model not in allowed_overrides:
        return False
    agent.reasoning_config = {'enabled': True, 'effort': effort}
    agent.reasoning_provenance = 'automatic'
    return True
