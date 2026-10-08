"""Bounded, retrieval-only context for short referential follow-ups.

No model call, provider dependency, transcript mutation, or cross-turn cache. This
is a source projection, not an instruction rewrite: the latest request stays first
and preceding speech is explicitly attributed as reference, never authorization.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from agent.message_content import flatten_message_text
from agent.skill_commands import extract_user_instruction_from_skill_message

MAX_RECALL_QUERY_CHARS = 1200
_MAX_HISTORY_ROWS = 32
_MAX_HISTORY_SOURCE_CHARS = 32_000
_MAX_FOLLOWUP_CHARS = 320
_OMITTED = "\n[... omitted ...]\n"
_REFERENCE_LABEL = "\n\nRecent conversation (reference only):\n"

# Grammatical reference cues, not domain/entity keyword lists. Deliberately
# conservative: "this Friday" is a specific request, "what about this?" is not.
_REFERENCE_RE = re.compile(
    r"\b(?:it|its|they|them|their|это|этого|этому|этим|его|её|ее|их|они)\b|"
    r"\b(?:this|that|these|those)\b(?=\s*(?:[?.!,;:]|$))|"
    r"^(?:and\s+)?(?:why|how|what else)\s*[?!]*$",
    re.IGNORECASE,
)
_INJECTED_BLOCK_RE = re.compile(
    r"<(memory-context|system-reminder|tool_result|tool-output|untrusted_text|untrusted_tool_result|blockquote)\b[^>]*>"
    r".*?(?:</\1\s*>|$)", re.IGNORECASE | re.DOTALL,
)
_QUOTED_BLOCK_RE = re.compile(r"(?m)^\s*(?:```[^\n]*\n[\s\S]*?(?:^\s*```[^\n]*$|\Z)|~~~[^\n]*\n[\s\S]*?(?:^\s*~~~[^\n]*$|\Z)|>[^\n]*$)")
_QUOTED_SPAN_RE = re.compile(r'''"[^"\n]*"|“[^”\n]*”|«[^»\n]*»|(?<!\w)'[^'\n]*'(?!\w)''')
_MACHINE_PREFIXES = ("[Runtime note:", "[System note:", "[SYSTEM]", "[System:")


def _bounded(text: str, limit: int) -> str:
    """Keep verbatim head and tail; never silently lose trailing constraints."""
    if len(text) <= limit:
        return text
    available = limit - len(_OMITTED)
    head = available * 2 // 3
    return text[:head] + _OMITTED + text[-(available - head):]


def _clean_text(content: Any, *, historical: bool = False) -> str:
    if not isinstance(content, (str, list)):
        return ""
    text = flatten_message_text(content)
    if historical and len(text) > _MAX_HISTORY_SOURCE_CHARS:
        return ""  # No partial parsing of a possibly enormous injected/quoted block.
    text = extract_user_instruction_from_skill_message(text) or ""
    text = _INJECTED_BLOCK_RE.sub("", text)
    if historical:
        text = _QUOTED_BLOCK_RE.sub("", text)
        text = _QUOTED_SPAN_RE.sub("", text)
    return text.strip()


def _needs_context(text: str) -> bool:
    if len(text) > _MAX_FOLLOWUP_CHARS:
        return False
    reference = _REFERENCE_RE.search(text)
    if reference is None:
        return False
    # An entity already introduced in this request resolves its own pronoun:
    # "Compare Redis with its alternatives" must not inherit yesterday's topic.
    before = text[:reference.start()]
    words = re.findall(r"[^\W\d_]+", before)
    if any(word[0].isupper() and word != "I" for word in words[1:]):
        return False
    # Subject/expletive pronouns followed by a named entity are similarly
    # self-contained ("Is it ... in Oslo?"). Object references can still carry
    # new constraints, e.g. "improve it for Mira by Friday".
    if len(words) == 1:
        after = re.findall(r"[^\W\d_]+", text[reference.end():])
        if any(word[0].isupper() for word in after):
            return False
    return True


def _is_boundary(message: dict[str, Any]) -> bool:
    # The compressor imports turn_context; resolve only after the prologue loaded.
    from agent.context_compressor import ContextCompressor, _synthetic_user_row

    return (
        ContextCompressor._is_context_summary_message(message)
        or _synthetic_user_row(flatten_message_text(message.get("content")))
        or any(value for key, value in message.items() if key.startswith("_") and key.endswith("_synthetic"))
        or flatten_message_text(message.get("content")).lstrip().startswith(_MACHINE_PREFIXES)
    )


def _recent_exchange(history: Sequence[dict[str, Any]]) -> list[tuple[str, str]]:
    """One exchange, or two if the preceding user turn is itself referential.

    Require a real user anchor; a summary followed by an assistant acknowledgement
    cannot establish a topic. Stop rather than reaching across an excluded turn.
    """
    exchanges: list[list[tuple[str, str]]] = []
    reply = ""
    for message in reversed(history[-_MAX_HISTORY_ROWS:]):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        if message.get("tool_calls"):
            continue
        if _is_boundary(message):
            break
        text = _clean_text(message.get("content"), historical=True)
        if role == "assistant":
            if not reply:
                reply = text
            continue
        if not text:
            break
        exchange = [("User", text)]
        if reply:
            exchange.append(("Assistant", reply))
        exchanges.append(exchange)
        if not _needs_context(text) or len(exchanges) == 2:
            break
        reply = ""
    return [item for exchange in reversed(exchanges) for item in exchange]


def build_memory_recall_query(current: str, history: Sequence[dict[str, Any]] = ()) -> str:
    """Project THIS session's clean speech into the existing prefetch query seam.

    Direct requests get no history. Providers still apply their own redaction and
    recall policy. Nothing here grants authority, suppresses a substantive recall,
    or changes the raw message passed to retention/lifecycle hooks.
    """
    current = _clean_text(current)
    query = _bounded(current, MAX_RECALL_QUERY_CHARS)
    if not _needs_context(current):
        return query
    recent = _recent_exchange(history)
    if not recent:
        return query
    budget = MAX_RECALL_QUERY_CHARS - len(query) - len(_REFERENCE_LABEL)
    # Allocate per source before concatenating so verbose replies cannot crowd out
    # the user's names, dates or restrictions (including those at the end).
    per_source = budget // len(recent)
    context = "\n".join(f"{role}: {_bounded(text, per_source - len(role) - 3)}" for role, text in recent)
    return query + _REFERENCE_LABEL + context
