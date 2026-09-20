"""Slack routing addresses, before command parsing or display-name enrichment."""

import re


def bot_mention_pattern(bot_user_id: str) -> re.Pattern:
    # Slack's optional display label is not identity; only the exact user ID is.
    return re.compile(r"<@" + re.escape(bot_user_id) + r"(?:\|[^>\r\n]*)?>")


def strip_typed_command_address(text: str, bot_name: str) -> str:
    """Accept a literal @bot-name before a known command, not arbitrary prose.

    This is command normalization only, not mention admission: channels that
    require a real Slack mention still require it. Never apply this to humanized
    text, where a different participant can have the bot's display name.
    """
    if not bot_name:
        return text
    match = re.match(r"^@" + re.escape(bot_name) + r"\s+([/!]\S+)", text)
    if not match:
        return text
    from hermes_cli.commands import is_gateway_known_command

    name = match[1][1:].split("@", 1)[0].lower()
    if not name or "/" in name or not is_gateway_known_command(name):
        return text
    return text[match.start(1):]
