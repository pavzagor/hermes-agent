"""Final-answer reasoning rendering and exact-chat display suppression."""
from __future__ import annotations

from typing import Any

from gateway.config import Platform
from agent.i18n import t

_QUOTE_STYLES = {
    "subtext": ("gateway.reasoning.quote_label_discord", "-# ", "-#"),
    "blockquote": ("gateway.reasoning.quote_label_md", "> ", ">"),
}


def prepend_reasoning(
    agent_result: dict[str, Any], response: str, source: Any,
    intentional_silence: bool, *, default: bool = False,
) -> str:
    """Hide scratch text for configured chats without changing reasoning effort."""
    from gateway.run import _load_gateway_config, _platform_config_key, _resolve_gateway_display_bool
    from gateway.display_config import resolve_display_setting
    from gateway.stream_consumer_fences import escape_code_fences_for_display

    config = _load_gateway_config()
    platform_key = _platform_config_key(source.platform)
    display = config.get("display") or {}
    platform_display = (display.get("platforms") or {}).get(platform_key) or {}
    hidden_chats = platform_display.get("hide_reasoning_chats") or []
    if not isinstance(hidden_chats, list):
        raise ValueError(f"display.platforms.{platform_key}.hide_reasoning_chats must be a list")
    if str(getattr(source, "chat_id", "")) in {str(chat) for chat in hidden_chats}:
        return response
    show = _resolve_gateway_display_bool(
        config, platform_key, "show_reasoning", default=default, platform=source.platform,
        require_platform_override_for={Platform.MATTERMOST},
    )
    last_reasoning = agent_result.get("last_reasoning")
    if not (show and response and not intentional_silence and last_reasoning):
        return response
    lines = last_reasoning.strip().splitlines()
    display_reasoning = (
        "\n".join(lines[:15]) + t("gateway.reasoning.more_lines", count=len(lines) - 15)
        if len(lines) > 15 else last_reasoning.strip()
    )
    style = resolve_display_setting(config, platform_key, "reasoning_style", "code")
    quote = _QUOTE_STYLES.get(style)
    if quote:
        header_key, prefix, empty = quote
        quoted = "\n".join([f"{prefix}{line}" if line else empty for line in display_reasoning.splitlines()])
        return f"{t(header_key)}\n{quoted}\n\n{response}"
    display_reasoning = escape_code_fences_for_display(display_reasoning)
    return t("gateway.reasoning.block", reasoning=display_reasoning, response=response)
