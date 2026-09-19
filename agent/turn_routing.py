"""Opt-in, same-provider model directives at the user-turn boundary.

Plugins return data; only the host may mutate the runtime. Persist manual pins
outside transcripts so rehydrating an agent cannot silently unpin a conversation.
"""
from __future__ import annotations
import hashlib
import json
import logging
from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)


def load_selected_skills(results, task_id):
    """Use the canonical loader, fully; instructional content must never be spilled."""
    from tools.skills_tool import skill_view
    loaded = set()
    output = []
    for result in results:
        names = result.get('skills', []) if isinstance(result, dict) else []
        if not isinstance(names, list):
            continue
        for name in names:
            if not isinstance(name, str) or name in loaded:
                continue
            loaded.add(name)
            try:
                value = skill_view(name=name, task_id=task_id)
                if json.loads(value).get('success'):
                    output.append(value)
            except Exception:
                logger.warning('Automatic skill loading failed; use normal discovery')
    if output:
        output.insert(0, '[Automatically loaded skills, full content. This supplements, never replaces, mandatory skill discovery. Follow all relevant skills and load linked references as needed.]')
    return output


def _config():
    from hermes_cli.config_effective import load_user_config_effective
    return load_user_config_effective().get('turn_routing', {})


def _path(session_id):
    if not isinstance(session_id, str) or not session_id:
        raise ValueError('Routing requires a session ID')
    return get_hermes_home() / 'routing' / (hashlib.sha256(session_id.encode()).hexdigest() + '.json')


def routing_configured():
    return _config().get('enabled') is True


def set_routing(session_id, enabled):
    """Explicit conversation control; no model/provider credentials are persisted."""
    path = _path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, {'enabled': bool(enabled)})


def pin_model(session_id):
    if _config().get('enabled') and session_id:
        set_routing(session_id, False)


def routing_enabled(session_id, *, is_first_turn=False):
    if not _config().get('enabled'):
        return False
    try:
        return json.loads(_path(session_id).read_text()).get('enabled') is True
    except FileNotFoundError:
        # Never reinterpret an existing conversation's model choice on rollout.
        if is_first_turn:
            set_routing(session_id, True)
        return bool(is_first_turn)
    except (ValueError, OSError, TypeError):
        return False


def apply_route(agent, directive, *, is_first_turn=False):
    """Consume at most one allowlisted directive. Fail-open without leaking errors."""
    if not isinstance(directive, dict) or set(directive) != {'model'}:
        return False
    model = directive['model']
    if not isinstance(model, str):
        return False
    if not routing_enabled(agent.session_id, is_first_turn=is_first_turn):
        return False
    allowed = _config().get('models', {}).get(agent.provider, [])
    if model not in allowed or model == agent.model:
        return False
    prefix = getattr(agent, '_cached_system_prompt', None)
    try:
        agent.switch_model(new_model=model, new_provider=agent.provider, automatic=True)
        return True
    except Exception:
        logger.warning('Turn model directive failed; retaining current route')
        return False
    finally:
        # A route decision must not rebuild memories/tools/system instructions.
        agent._cached_system_prompt = prefix
