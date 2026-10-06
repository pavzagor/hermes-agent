"""What a finite one-shot session (``hermes chat -q`` / ``--oneshot``, ``hermes -z``) does NOT do.

A one-shot run has no later session in its HERMES_HOME to learn for: the process answers one query and
exits. The interactive self-improvement loop is pure overhead there, and a measured one — across 21
one-shot benchmark trajectories the agent authored 7 new skills and patched a bundled one mid-task, 37 of
~215 tool calls were ``skill_view``/``skill_manage``, and skill text was 34% of every tool-result byte fed
back into context. Three rules follow, all keyed on the same session marker the approval gate and the
delegation dispatcher already read (``HERMES_SINGLE_QUERY_SESSION``), so interactive sessions are untouched:

* ``skill_manage`` is not offered (``skills_list``/``skill_view`` stay: reading a domain skill can still win);
* the ## Skills prompt drops the "record it / patch it / offer to save" coaching while retaining
  operation-and-branch selection, user conventions, and reloads for missing or pruned instructions;
* delegation is capped per session (``delegation.oneshot_max_children``): subagents each re-pay a cold
  system prompt and re-explore the repo, and the observed spawns were mostly "independent review of my own
  work" rather than parallel work.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

ONESHOT_HIDDEN_TOOLS = frozenset({"skill_manage"})


def is_single_query_session() -> bool:
    """The finite ``-q`` marker, read through the session env so gateway-bound sessions never see it."""
    try:
        from gateway.session_context import get_session_env
    except Exception:
        import os
        get_session_env = os.environ.get
    return str(get_session_env("HERMES_SINGLE_QUERY_SESSION", "") or "") == "1"


def prune_oneshot_tools(tools: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """*tools* minus ``ONESHOT_HIDDEN_TOOLS``; identity when the session is not one-shot."""
    tools = list(tools)
    if not is_single_query_session():
        return tools
    return [t for t in tools if (t.get("function") or {}).get("name") not in ONESHOT_HIDDEN_TOOLS]


ONESHOT_SKILLS_LOAD_GUIDANCE = (
    "## Skills\n"
    "Before acting, scan the skills below for the governing operation. You MUST load its skill with "
    "skill_view(name), then add skills or linked references required by a concrete action, safety, or proof "
    "branch. Follow their instructions, including the user's conventions and quality standards even for "
    "familiar work. Do not load adjacent skills for topical overlap alone. If the operation or branch is "
    "uncertain, expand discovery before acting; do not guess away a safety or verification requirement. "
    "A skill's full current body already in context, including auto-loaded content, counts as loaded — "
    "do not reload it just to satisfy this rule. A name, index entry, summary, or `[SKILL_PRUNED]` placeholder "
    "does not count: reload missing or stale instructions before relying on it. Selection does not relax "
    "tool-use, authorization, scope, or verification requirements. If no skill governs the operation or "
    "a required branch, proceed without loading one. Do not create or edit skills: this is a one-shot run "
    "with no later session to reuse them.\n"
)
