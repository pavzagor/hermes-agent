"""Matrix clarify controls; reuse adapter reaction authorization and redaction."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from gateway.platforms.base import SendResult

logger = logging.getLogger(__name__)


@dataclass
class _MatrixClarifyPrompt:
    """Tracks a pending Matrix reaction-based generic clarify prompt."""

    chat_id: str
    message_id: str
    clarify_id: str
    session_key: str
    choices: dict[str, str]
    requester_user_id: str | None = None
    expires_at: float | None = None
    resolved: bool = False
    bot_reaction_events: dict[str, str] = field(default_factory=dict)


_MATRIX_CLARIFY_FALLBACK_REACTIONS = (
    "1\ufe0f\u20e3",
    "2\ufe0f\u20e3",
    "3\ufe0f\u20e3",
    "4\ufe0f\u20e3",
)


_MATRIX_CLARIFY_OTHER_REACTION = "\u270d\ufe0f"


def _matrix_logical_emoji(emoji: str) -> str:
    """Collapse text/emoji presentation selectors for reaction comparisons."""
    return str(emoji).replace("\ufe0e", "").replace("\ufe0f", "")


def _matrix_leading_emoji_token(choice: str) -> str | None:
    """Return a safe leading emoji token, excluding ordinary text tokens."""
    text = str(choice or "").lstrip()
    if not text:
        return None
    token = text.split(maxsplit=1)[0]

    # Keycap emoji contain an ASCII base plus combining keycap, unlike other
    # emoji sequences, so recognize them explicitly.
    if re.fullmatch(r"[0-9#*]\ufe0f?\u20e3", token):
        return token

    def _is_emoji_codepoint(char: str) -> bool:
        codepoint = ord(char)
        return (
            0x1F000 <= codepoint <= 0x1FAFF
            or 0x2600 <= codepoint <= 0x27BF
            or 0x2300 <= codepoint <= 0x23FF
            or 0x2B00 <= codepoint <= 0x2BFF
            or 0x2190 <= codepoint <= 0x21FF
        )

    if not any(_is_emoji_codepoint(char) for char in token):
        return None
    for char in token:
        codepoint = ord(char)
        if (
            _is_emoji_codepoint(char)
            or codepoint in {0x200D, 0xFE0E, 0xFE0F, 0x20E3}
            or 0x1F3FB <= codepoint <= 0x1F3FF
        ):
            continue
        return None
    return token


def _matrix_clarify_choice_reactions(choices: list) -> dict[str, str]:
    """Map unique reaction keys to canonical clarify choice strings."""
    mapped: dict[str, str] = {}
    used = {_matrix_logical_emoji(_MATRIX_CLARIFY_OTHER_REACTION)}
    for index, raw_choice in enumerate(
        choices[: len(_MATRIX_CLARIFY_FALLBACK_REACTIONS)]
    ):
        choice = str(raw_choice)
        leading = _matrix_leading_emoji_token(choice)
        if leading and _matrix_logical_emoji(leading) not in used:
            reaction = leading
        else:
            preferred = _MATRIX_CLARIFY_FALLBACK_REACTIONS[index]
            reaction = preferred
            if _matrix_logical_emoji(reaction) in used:
                reaction = next(
                    fallback
                    for fallback in _MATRIX_CLARIFY_FALLBACK_REACTIONS
                    if _matrix_logical_emoji(fallback) not in used
                )
        mapped[reaction] = choice
        used.add(_matrix_logical_emoji(reaction))
    return mapped


class MatrixClarifyMixin:
    async def send_clarify(
        self,
        chat_id: str,
        question: str,
        choices: Optional[list],
        clarify_id: str,
        session_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a generic clarify prompt with one-tap Matrix reactions."""
        if not choices:
            return await self.send(
                chat_id=chat_id,
                content=f"❓ {question}",
                metadata=metadata,
            )

        # Preserve Matrix's existing free-text fallback alongside reactions.
        from tools.clarify_gateway import mark_awaiting_text

        mark_awaiting_text(clarify_id)
        reaction_choices = _matrix_clarify_choice_reactions(choices)
        lines = [f"❓ {question}", "", "Choose an option, or type an answer:"]
        lines.extend(
            f"{reaction} {choice}" for reaction, choice in reaction_choices.items()
        )
        lines.append(f"{_MATRIX_CLARIFY_OTHER_REACTION} Other / type an answer")

        result = await self.send(chat_id, "\n".join(lines), metadata=metadata)
        if not result.success or not result.message_id:
            return result

        from tools.clarify_gateway import get_clarify_timeout

        timeout = get_clarify_timeout()
        prompt = _MatrixClarifyPrompt(
            chat_id=chat_id,
            message_id=result.message_id,
            clarify_id=clarify_id,
            session_key=session_key,
            choices=reaction_choices,
            requester_user_id=(
                str((metadata or {}).get("requester_user_id") or "") or None
            ),
            expires_at=time.monotonic() + timeout if timeout > 0 else None,
        )
        self._clarify_prompts_by_event[result.message_id] = prompt
        self._clarify_prompt_event_by_id[clarify_id] = result.message_id

        for reaction in (*reaction_choices, _MATRIX_CLARIFY_OTHER_REACTION):
            if prompt.resolved:
                break
            try:
                reaction_event_id = await self._send_reaction(
                    chat_id, result.message_id, reaction
                )
                if reaction_event_id:
                    if prompt.resolved:
                        self._schedule_reaction_redaction(
                            chat_id, str(reaction_event_id), "clarify resolved"
                        )
                    else:
                        prompt.bot_reaction_events[reaction] = str(reaction_event_id)
            except Exception as exc:
                logger.debug("Matrix: failed to add clarify reaction: %s", exc)

        if (
            not prompt.resolved
            and len(prompt.bot_reaction_events) != len(reaction_choices) + 1
        ):
            await self.cleanup_clarify(clarify_id)
            fallback = "\n".join([
                f"❓ {question}",
                "",
                *(f"{i}. {choice}" for i, choice in enumerate(choices, 1)),
                "",
                "Reaction controls unavailable. Reply with an option or your own answer.",
            ])
            await self.edit_message(
                chat_id=chat_id, message_id=result.message_id, content=fallback
            )
        return result

    async def _handle_clarify_reaction(
        self, room_id: str, reacts_to: str, key: str, sender: str
    ) -> bool:
        from tools.clarify_gateway import mark_awaiting_text, resolve_gateway_clarify

        prompt = self._clarify_prompts_by_event.get(reacts_to)
        if prompt is None:
            return False
        choices = {_matrix_logical_emoji(k): v for k, v in prompt.choices.items()}
        other = _matrix_logical_emoji(_MATRIX_CLARIFY_OTHER_REACTION)
        choices[other] = other
        handled, prompt, choice = await self._claim_reaction_prompt(
            self._clarify_prompts_by_event,
            room_id,
            reacts_to,
            _matrix_logical_emoji(key),
            sender,
            "clarify",
            "That reaction is not one of the available choices.",
            self._expire_matrix_clarify_prompt,
            choices=choices,
        )
        if choice is None:
            return handled
        resolved = (
            mark_awaiting_text(prompt.clarify_id)
            if _matrix_logical_emoji(key) == other
            else resolve_gateway_clarify(prompt.clarify_id, choice)
        )
        await self.cleanup_clarify(prompt.clarify_id)
        if not resolved:
            await self._send_invalid_reaction_feedback(
                room_id, reacts_to, "This clarify prompt is no longer active."
            )
        return True

    def _retire_clarify_prompt(self, clarify_id: str) -> _MatrixClarifyPrompt | None:
        """Atomically remove one clarify prompt from both local indexes."""
        event_id = self._clarify_prompt_event_by_id.pop(clarify_id, None)
        if event_id is None:
            return None

        prompt = self._clarify_prompts_by_event.pop(event_id, None)
        if prompt is None:
            return None
        self._clarify_prompt_event_by_id.pop(prompt.clarify_id, None)
        prompt.resolved = True
        return prompt

    async def cleanup_clarify(self, clarify_id: str) -> None:
        """Retire a completed Matrix clarify prompt and its bot seed reactions."""
        prompt = self._retire_clarify_prompt(clarify_id)
        if prompt is None:
            return
        await self._redact_bot_clarify_reactions(prompt.chat_id, prompt)

    async def _expire_matrix_clarify_prompt(
        self,
        room_id: str,
        target_event_id: str,
        prompt: "_MatrixClarifyPrompt",
    ) -> None:
        """Resolve local clarify state and retire an expired reaction prompt."""
        await self.cleanup_clarify(prompt.clarify_id)
        await self._send_invalid_reaction_feedback(
            room_id,
            target_event_id,
            "This clarify prompt has expired. Ask again to choose an answer.",
        )

    async def _redact_bot_clarify_reactions(
        self,
        room_id: str,
        prompt: "_MatrixClarifyPrompt",
    ) -> None:
        """Schedule redaction of bot clarify seeds, leaving user reactions intact."""
        for reaction, event_id in prompt.bot_reaction_events.items():
            self._schedule_reaction_redaction(
                room_id,
                event_id,
                "clarify resolved",
            )
            logger.debug(
                "Matrix: scheduled clarify seed redaction %s (%s)",
                reaction,
                event_id,
            )
