"""Codex auxiliary credentials follow the main runtime resolver on exhausted pools."""

import json
import time
from pathlib import Path

from agent.auxiliary_client import _build_codex_client, _resolve_codex_credential_and_base
from agent.secret_scope import (
    build_profile_secret_scope, reset_secret_scope, set_multiplex_active, set_secret_scope,
)
from hermes_cli.auth import resolve_codex_runtime_credentials
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _write_pool_only(home: Path, token: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "credential_pool": {"openai-codex": [{
            "id": f"pool-{token}", "label": "exhausted", "auth_type": "oauth",
            "source": "manual:device_code", "priority": 0,
            "access_token": token, "refresh_token": "fixture-refresh",
            "last_status": "exhausted", "last_status_at": time.time(),
            "last_error_code": 429, "last_error_reason": "usage_limit_reached",
        }]},
    }))


def test_compression_uses_main_runtime_token_when_pool_is_exhausted(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    _write_pool_only(home, "fixture-codex-token")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing-external-login"))

    expected = resolve_codex_runtime_credentials()["api_key"]
    assert expected == "fixture-codex-token"
    assert _resolve_codex_credential_and_base()[0] == expected
    client, model = _build_codex_client("gpt-6-luna")
    assert client is not None
    try:
        assert model == "gpt-6-luna"
        assert client.api_key == expected
    finally:
        client.close()


def test_exhausted_codex_fallback_keeps_profile_scope_a_b_a(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing-external-login"))
    a, b = root / "profiles" / "a", root / "profiles" / "b"
    _write_pool_only(a, "fixture-a-token")
    _write_pool_only(b, "fixture-b-token")

    def read_for(home):
        home_scope = set_hermes_home_override(home)
        secret_scope = set_secret_scope(build_profile_secret_scope(home))
        try:
            assert resolve_codex_runtime_credentials()["api_key"] == _resolve_codex_credential_and_base()[0]
            return _resolve_codex_credential_and_base()[0]
        finally:
            reset_secret_scope(secret_scope)
            reset_hermes_home_override(home_scope)

    set_multiplex_active(True)
    try:
        assert [read_for(home) for home in (a, b, a)] == [
            "fixture-a-token", "fixture-b-token", "fixture-a-token",
        ]
    finally:
        set_multiplex_active(False)
