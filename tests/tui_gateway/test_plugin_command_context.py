"""TUI/API plugin commands receive the session's persistent key."""

import tui_gateway.server as server


def test_tui_plugin_dispatch_binds_session_context(monkeypatch):
    from hermes_cli import plugins

    seen = {}
    def handler(raw_args):
        seen["args"] = raw_args
        seen["context"] = plugins.current_plugin_command_context()
        return "ok"
    monkeypatch.setattr(server, "_plugin_command_handler", lambda name: handler)

    result = server._dispatch_plugin(
        "rid", {"session_id": "ui-id"}, {"session_key": "db-session"}, "routing", "off")

    assert result["result"]["output"] == "ok"
    assert (seen["args"], seen["context"].session_id, seen["context"].surface) == (
        "off", "db-session", "tui_gateway")
    assert plugins.current_plugin_command_context() is None


def test_plugin_command_uses_session_profile_and_actual_platform(monkeypatch, tmp_path):
    from agent import turn_routing
    from hermes_cli import plugins

    launch_home = tmp_path / "launch"
    profile_home = tmp_path / "profile"
    launch_home.mkdir()
    profile_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setattr(turn_routing, "_config", lambda: {"enabled": True})
    seen = {}
    def handler(raw_args):
        context = plugins.current_plugin_command_context()
        seen["context"] = context
        assert turn_routing.set_routing(context.session_id, True)
        return "ok"
    session = {
        "session_key": "profile-session", "source": "desktop",
        "profile_home": str(profile_home),
    }

    assert server._run_plugin_command(handler, "on", session) == "ok"
    assert seen["context"].surface == "tui_gateway"
    assert seen["context"].platform == "desktop"
    digest = turn_routing._path("profile-session").name
    assert (profile_home / "routing" / digest).exists()
    assert not (launch_home / "routing" / digest).exists()
