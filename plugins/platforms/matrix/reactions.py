"""Ordinary reaction intake, adapted from karthee85's #37308 and liuhao1024's #43745.

Use the current text pipeline rather than the old PR's direct MessageEvent path.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)
CONTROL_PROMPT_KEY = "org.hermes.control_prompt"


def is_control_prompt(content: dict) -> bool:
    # Durable marker survives retirement/restart. Recognize pre-marker prompts too;
    # conservative false positives are preferable to turning expired controls into consent.
    body = content.get("body") or ""
    return bool(content.get(CONTROL_PROMPT_KEY)) or any(phrase in body for phrase in (
        "You can also click the reaction to approve:", "React to choose a model:",
        "React to choose.", "Choose an option, or type an answer:",
        "Reaction controls unavailable.",
    )) or body.startswith("❓ ")


async def forward_reaction(adapter, event: Any, room_id: str, sender: str,
                           event_id: str, target_id: str, key: str) -> None:
    from .adapter import _matrix_event_timestamp_seconds, _STARTUP_GRACE_SECONDS

    if not getattr(adapter, "_forward_reactions", False) or not adapter._client:
        return
    if (not adapter._is_authorized_user(sender)
            or adapter._is_system_or_bridge_sender(sender)
            or any(pattern.search(sender) for pattern in adapter._ignored_user_patterns)):
        return
    if not await adapter._is_allowed_matrix_room_event(room_id):
        return
    timestamp = _matrix_event_timestamp_seconds(event)
    if timestamp and timestamp < adapter._startup_ts - _STARTUP_GRACE_SECONDS:
        return
    if not isinstance(target_id, str) or not target_id or not isinstance(key, str) or not key:
        return
    content = event.content
    content = content if isinstance(content, dict) else content.serialize()
    if (content.get("m.relates_to") or {}).get("rel_type") != "m.annotation":
        return
    try:
        # Fetch, not a sent-message cache: targets from before a restart still work.
        target = await asyncio.wait_for(adapter._client.get_event(room_id, target_id), timeout=10)
        if (str(getattr(target, "sender", "")) != adapter._user_id
                or str(getattr(target, "type", "")) != "m.room.message"
                or str(getattr(target, "room_id", room_id)) != room_id):
            return
        target_content = target.content
        target_content = target_content if isinstance(target_content, dict) else target_content.serialize()
        if not target_content.get("body") or is_control_prompt(target_content):
            return
        target_relation = target_content.get("m.relates_to") or {}
        if target_relation.get("rel_type") == "m.replace":
            return
        relation: dict[str, Any] = {"m.in_reply_to": {"event_id": target_id}}
        if target_relation.get("rel_type") == "m.thread":
            relation.update(rel_type="m.thread", event_id=target_relation["event_id"])
        synthetic = {
            "msgtype": "m.text",
            # Quoting prevents arbitrary annotation keys becoming commands/markup.
            "body": f"[reaction] reacted with {json.dumps(key, ensure_ascii=False)} "
                    "to your message (feedback, not authorization).",
            "m.relates_to": relation,
        }
        await adapter._handle_text_message(
            room_id, sender, event_id, timestamp, synthetic, relation,
            synthetic_thread_root=target_id)
    except Exception as exc:
        logger.debug("Matrix: failed to forward reaction %s: %s", event_id, exc)
