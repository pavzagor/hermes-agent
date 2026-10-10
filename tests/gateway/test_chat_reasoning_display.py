"""Reasoning visibility is a display policy, scoped to an exact platform/chat."""
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gateway.config import Platform
from gateway.run import GatewayRunner


@pytest.mark.parametrize(
    ("platform", "chat_id", "thread_id", "hidden"),
    [
        (Platform.TELEGRAM, "-1001234567890", None, True),
        (Platform.TELEGRAM, "-1001234567890", "150", True),
        (Platform.TELEGRAM, -1001234567890, None, True),
        (Platform.TELEGRAM, "25415780", None, False),
        (Platform.SLACK, "-1001234567890", None, False),
    ],
)
def test_hidden_chat_never_prepends_reasoning(
    tmp_path: Any, monkeypatch: Any, platform: Platform, chat_id: str | int,
    thread_id: str | None, hidden: bool,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    config = {
        "display": {
            "show_reasoning": True,
            "platforms": {"telegram": {"hide_reasoning_chats": ["-1001234567890"]}},
        },
        "agent": {"reasoning_effort": "high"},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._show_reasoning = True
    source = SimpleNamespace(platform=platform, chat_id=chat_id, thread_id=thread_id)
    result = runner._hmwa_prepend_reasoning(
        {"last_reasoning": "Private scratch text"}, "Useful final answer", source, False,
    )
    assert result.endswith("Useful final answer")
    assert ("Private scratch text" not in result) is hidden
    assert yaml.safe_load((tmp_path / "config.yaml").read_text()) == config


@pytest.mark.parametrize("style", ["code", "blockquote", "subtext"])
def test_existing_render_style_and_truncation_are_preserved(
    tmp_path: Any, monkeypatch: Any, style: str,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "display": {"show_reasoning": True, "reasoning_style": style},
    }))
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._show_reasoning = True
    source = SimpleNamespace(platform=Platform.TELEGRAM, chat_id="25415780")
    result = runner._hmwa_prepend_reasoning(
        {"last_reasoning": "\n".join([f"line {n}" for n in range(20)])},
        "Useful final answer", source, False,
    )
    assert "line 14" in result
    assert "line 15" not in result
    assert result.endswith("Useful final answer")


@pytest.mark.parametrize(
    ("platform", "show", "silence", "expected_scratch"),
    [(Platform.TELEGRAM, False, False, False),
     (Platform.TELEGRAM, True, True, False),
     (Platform.MATTERMOST, True, False, False)],
)
def test_other_visibility_guards_are_preserved(
    tmp_path: Any, monkeypatch: Any, platform: Platform, show: bool,
    silence: bool, expected_scratch: bool,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"display": {"show_reasoning": show}}))
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._show_reasoning = True
    source = SimpleNamespace(platform=platform, chat_id="25415780")
    result = runner._hmwa_prepend_reasoning(
        {"last_reasoning": "Scratch text"}, "Final answer", source, silence,
    )
    assert ("Scratch text" in result) is expected_scratch
