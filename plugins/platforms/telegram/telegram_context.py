"""Telegram group addressing helpers, kept out of the adapter facade.

The identity line lives in ``channel_prompt`` and therefore in the cached-agent signature: it
must be stable for the life of a session (username only — never a per-message fact).
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from telegram import Message
    from plugins.platforms.telegram.adapter import TelegramAdapter


def is_mention_only_chat(adapter: "TelegramAdapter", message: "Message") -> bool:
    """An opt-in chat gate overrides reply, wake-word and free-response fallbacks."""
    configured = adapter.config.extra.get("mention_only_chats", [])
    if not isinstance(configured, list):
        raise ValueError("telegram.extra.mention_only_chats must be a list of numeric chat IDs")
    return adapter._chat_id_str(message) in {str(peer) for peer in configured}


def is_mention_only_reply_allowed(adapter: "TelegramAdapter", message: "Message") -> bool:
    """Selected human senders may reply to the receiving bot without a tag."""
    configured = adapter.config.extra.get("mention_only_reply_users", {})
    if not isinstance(configured, dict):
        raise ValueError("telegram.extra.mention_only_reply_users must be a chat-to-user-list mapping")
    users = configured.get(adapter._chat_id_str(message), [])
    if not isinstance(users, list):
        raise ValueError("telegram.extra.mention_only_reply_users values must be user-ID lists")
    sender = getattr(message, "from_user", None)
    bot_id = getattr(adapter._bot, "id", None)
    return bool(
        sender and not getattr(sender, "is_bot", False) and bot_id is not None
        and str(getattr(sender, "id", None)) in {str(user) for user in users}
        and adapter._is_reply_to_bot(message)
    )


def mentions_other_participants(adapter: "TelegramAdapter", message: "Message") -> bool:
    """True when a ``mention``/``text_mention`` entity names someone other than this bot."""
    own = adapter._current_bot_username()
    bot_id = getattr(adapter._bot, "id", None) if adapter._bot else None
    for source_text, entities in adapter._entity_sources(message):
        for entity in entities:
            entity_type = adapter._entity_type(entity)
            if entity_type == "mention":
                handle = (adapter._entity_span(source_text, entity) or "").strip().lstrip("@").lower()
                if handle and handle != own:
                    return True
            elif entity_type == "text_mention":
                user = getattr(entity, "user", None)
                if user is not None and getattr(user, "id", None) != bot_id:
                    return True
    return False


def group_trigger_text(adapter: "TelegramAdapter", message: "Message", text: Optional[str]) -> Optional[str]:
    """Strip our own handle only when we are the sole addressee. With other participants named,
    ``@research_bot , @ops_bot are you both listening?`` must not reach us as ``, @ops_bot …``."""
    if adapter._is_group_chat(message) and mentions_other_participants(adapter, message):
        return text
    return adapter._clean_bot_trigger_text(text)


def group_identity_prompt(
    adapter: "TelegramAdapter", message: "Message", channel_prompt: Optional[str],
) -> Optional[str]:
    """Session-stable identity line so the model can read retained @mentions as itself or not."""
    if not adapter._is_group_chat(message) or not getattr(adapter, "_bot", None):
        return channel_prompt
    username = adapter._current_bot_username()
    if not username:
        return channel_prompt
    identity = (
        f"Your Telegram bot username in this group: @{username}. "
        "Mentions of other bots are not requests for you to relay the message."
    )
    return f"{channel_prompt}\n\n{identity}" if channel_prompt else identity
