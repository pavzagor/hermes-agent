"""Opt-in, same-provider model directives at the user-turn boundary.

Plugins return data; only the host mutates the runtime. Routing state lives
outside transcripts so agent recreation and compression preserve explicit
manual pins and the last successful automatic route.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SKILLS = 8
_MAX_SKILLS_HARD_LIMIT = 32


def _config() -> dict:
    from hermes_cli.config_effective import load_user_config_effective
    cfg = load_user_config_effective().get("turn_routing", {})
    return cfg if isinstance(cfg, dict) else {}


def _safe_config() -> dict:
    try:
        return _config()
    except Exception:
        logger.warning("Turn-routing config could not be read; routing disabled", exc_info=True)
        return {}


def _path(session_id: str):
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Routing requires a session ID")
    digest = hashlib.sha256(session_id.encode()).hexdigest()
    return get_hermes_home() / "routing" / f"{digest}.json"


def _read_state(session_id: str) -> dict | None:
    try:
        raw = json.loads(_path(session_id).read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else None
    except FileNotFoundError:
        return None
    except Exception:
        logger.warning("Turn-routing state could not be read; routing disabled", exc_info=True)
        return {}


def _write_state(session_id: str, state: dict) -> bool:
    try:
        path = _path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(path, state)
        return True
    except Exception:
        logger.warning("Turn-routing state could not be written", exc_info=True)
        return False


def routing_configured() -> bool:
    return _safe_config().get("enabled") is True


def set_routing(session_id: str, enabled: bool) -> bool:
    """Explicit conversation control; credentials are never persisted."""
    state = _read_state(session_id) or {}
    state["enabled"] = bool(enabled)
    return _write_state(session_id, state)


def pin_model(session_id: str | None) -> bool:
    """Record an explicit manual choice and discard stale automatic lineage."""
    if not session_id:
        return False
    return _write_state(session_id, {"enabled": False})


def routing_enabled(session_id: str, *, is_first_turn: bool = False) -> bool:
    if not routing_configured():
        return False
    state = _read_state(session_id)
    if state is not None:
        return state.get("enabled") is True
    # Never reinterpret an existing conversation's model choice on rollout.
    if is_first_turn:
        return set_routing(session_id, True)
    return False


def snapshot_routing_state(session_id: str) -> dict:
    """Capture exact state, including absence, for a one-turn model override."""
    state = _read_state(session_id)
    return {"exists": state is not None, "state": state}


def restore_routing_state(session_id: str, snapshot: Any) -> bool:
    """Best-effort restoration after a one-turn override."""
    if not isinstance(snapshot, dict):
        return False
    if snapshot.get("exists") is True and isinstance(snapshot.get("state"), dict):
        return _write_state(session_id, snapshot["state"])
    if snapshot.get("exists") is False:
        try:
            _path(session_id).unlink(missing_ok=True)
            return True
        except Exception:
            logger.warning("Turn-routing state could not be restored", exc_info=True)
    return False


def migrate_routing_state(old_session_id: str, new_session_id: str) -> bool:
    """Copy routing provenance across a compression-created session id."""
    state = _read_state(old_session_id)
    return bool(state is not None and _write_state(new_session_id, state))


def _allowed_route(agent: Any, model: Any, *, is_first_turn: bool) -> bool:
    if not isinstance(model, str):
        return False
    if not routing_enabled(agent.session_id, is_first_turn=is_first_turn):
        return False
    models = _safe_config().get("models", {})
    if not isinstance(models, dict):
        return False
    allowed = models.get(getattr(agent, "provider", ""), [])
    return isinstance(allowed, list) and model in allowed


def _switch_preserving_prefix(agent: Any, model: str, provenance: str) -> bool:
    prefix = getattr(agent, "_cached_system_prompt", None)
    try:
        agent.switch_model(
            new_model=model, new_provider=agent.provider, provenance=provenance)
        return True
    except Exception:
        logger.warning("Turn model directive failed; retaining current route", exc_info=True)
        return False
    finally:
        # Routing never rebuilds memories/tools/system instructions mid-conversation.
        agent._cached_system_prompt = prefix


def apply_route(agent: Any, directive: Any, *, is_first_turn: bool = False) -> bool:
    """Consume one allowlisted same-provider directive and persist its provenance."""
    if not isinstance(directive, dict) or set(directive) != {"model"}:
        return False
    model = directive["model"]
    if not _allowed_route(agent, model, is_first_turn=is_first_turn):
        return False
    changed = model != agent.model
    if changed and not _switch_preserving_prefix(agent, model, "automatic_routing"):
        return False
    state = {
        "enabled": True,
        "automatic_route": {"model": model, "provider": agent.provider},
    }
    _write_state(agent.session_id, state)
    return changed


def restore_automatic_route(agent: Any, *, is_first_turn: bool = False) -> bool:
    """Restore a persisted automatic route before invoking this turn's classifier."""
    if not routing_enabled(agent.session_id, is_first_turn=is_first_turn):
        return False
    state = _read_state(agent.session_id) or {}
    route = state.get("automatic_route")
    if not isinstance(route, dict) or route.get("provider") != getattr(agent, "provider", None):
        return False
    model = route.get("model")
    if not _allowed_route(agent, model, is_first_turn=is_first_turn) or model == agent.model:
        return False
    return _switch_preserving_prefix(agent, model, "automatic_restore")


def _skill_limit() -> int:
    value = _safe_config().get("max_skills_per_turn", _DEFAULT_MAX_SKILLS)
    try:
        return max(1, min(int(value), _MAX_SKILLS_HARD_LIMIT))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_SKILLS


def load_selected_skills(results: Any, task_id: str) -> list[str]:
    """Load a bounded number of complete skills through the canonical loader."""
    from tools.skills_tool import skill_view

    loaded: set[str] = set()
    output: list[str] = []
    skipped = False
    selected = 0
    limit = _skill_limit()
    for result in results if isinstance(results, (list, tuple)) else ():
        names = result.get("skills", []) if isinstance(result, dict) else []
        if not isinstance(names, list):
            continue
        for name in names:
            if not isinstance(name, str) or name in loaded:
                continue
            loaded.add(name)
            if selected >= limit:
                skipped = True
                continue
            selected += 1
            try:
                value = skill_view(name=name, task_id=task_id)
                if json.loads(value).get("success"):
                    output.append(value)
            except Exception:
                logger.warning("Automatic skill loading failed; use normal discovery", exc_info=True)
    if output:
        output.insert(0, "[Automatically loaded skills, full content. This supplements, never "
                         "replaces, mandatory skill discovery. Follow all relevant skills and load "
                         "linked references as needed.]")
    if skipped:
        output.append("[Additional automatic skill selections were skipped by the per-turn limit; "
                      "normal skill discovery remains available.]")
    return output
