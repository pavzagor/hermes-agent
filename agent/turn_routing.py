"""Opt-in, same-provider model directives at the user-turn boundary.

Plugins return data; only the host mutates the runtime. Routing state lives
outside transcripts so agent recreation and compression preserve explicit
manual pins and the last successful automatic route.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

# Disk failure must not undo user intent when the gateway evicts an agent.
# Keys are profile-scoped paths; None represents an absent pre--once state.
_state_overrides: dict[Path, dict | None] = {}


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
        path = _path(session_id)
        if path in _state_overrides:
            state = _state_overrides[path]
            return dict(state) if state is not None else None
        raw = json.loads(path.read_text(encoding="utf-8"))
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
        _state_overrides.pop(path, None)
        return True
    except Exception:
        logger.warning("Turn-routing state could not be saved; persistence unavailable")
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
    if not isinstance(session_id, str) or not session_id:
        return False
    _state_overrides[_path(session_id)] = {"enabled": False}
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
    path = _path(session_id)
    if snapshot.get("exists") is True and isinstance(snapshot.get("state"), dict):
        _state_overrides[path] = dict(snapshot["state"])
        return _write_state(session_id, snapshot["state"])
    if snapshot.get("exists") is False:
        _state_overrides[path] = None
        try:
            path.unlink(missing_ok=True)
            _state_overrides.pop(path, None)
            return True
        except Exception:
            logger.warning("Turn-routing state could not be restored on disk; restored in memory")
    return False


def migrate_routing_state(old_session_id: str, new_session_id: str) -> bool:
    """Copy routing provenance across a compression-created session id."""
    state = _read_state(old_session_id)
    if state is None:
        return False
    _state_overrides[_path(new_session_id)] = dict(state)
    return _write_state(new_session_id, state)


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


def load_selected_skills(results: Any, task_id: str) -> list[str]:
    """Load every distinct selected skill fully through the canonical loader."""
    from tools.skills_tool import skill_view

    loaded: set[str] = set()
    output: list[str] = []

    for result in results if isinstance(results, (list, tuple)) else ():
        names = result.get("skills", []) if isinstance(result, dict) else []
        if not isinstance(names, list):
            continue
        for name in names:
            if not isinstance(name, str) or name in loaded:
                continue
            loaded.add(name)

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

    return output
