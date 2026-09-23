"""Real command-source → profile scope → native browser vault, synthetic I/O only."""
import json
import os
import shlex
import sys
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml

from agent.secret_scope import (build_profile_secret_scope, set_secret_scope,
                                reset_secret_scope, set_multiplex_active)
from agent.vault_backends.base import enabled_backends
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from hermes_cli.env_loader import hydrate_profile_secret_sources, reset_secret_source_cache

V, I = 'a' * 26, 'b' * 26
HANDLE = f'op:connect:{V}:{I}'


def test_command_source_connect_without_cli_is_profile_isolated(tmp_path, monkeypatch):
    from tools import browser_vault_tool as vault
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_GET(self):
            auth = self.headers.get('Authorization')
            calls.append((self.path, auth))
            profile = auth.removeprefix('Bearer synthetic-')
            item = {'id': I, 'vault': {'id': V}, 'category': 'LOGIN',
                    'title': profile, 'urls': [{'href': f'https://{profile}.example/login'}]}
            routes = {'/v1/vaults': [{'id': V}], f'/v1/vaults/{V}/items': [item],
                      f'/v1/vaults/{V}/items/{I}': dict(item, fields=[
                          {'purpose': 'USERNAME', 'value': profile + '@example.test'},
                          {'purpose': 'PASSWORD', 'value': 'synthetic-password-' + profile}])}
            self.send_response(200 if profile in ('a', 'b') else 403)
            self.end_headers()
            self.wfile.write(json.dumps(routes.get(self.path, {})).encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setattr('agent.secret_sources.onepassword.find_op', lambda *_: None)
    monkeypatch.setenv('OP_CONNECT_TOKEN', 'launch-profile-must-not-leak')
    monkeypatch.setenv('OP_CONNECT_HOST', 'http://127.0.0.1:1')
    monkeypatch.setenv('OP_SERVICE_ACCOUNT_TOKEN', 'launch-service-must-not-leak')
    monkeypatch.setattr(vault, '_focus_bound_origin', lambda *_: None)
    monkeypatch.setattr(vault, '_eval_js', lambda *_: {'success': True, 'result': [
        {'type': 'password', 'autocomplete': 'current-password', 'index': 0}]})
    fills = []
    def fill(_, source):
        fills.append(source)
        return {'success': True, 'result': {'filled': 1}}
    monkeypatch.setattr(vault, '_eval_js_secret', fill)
    homes = {}
    for name in ('a', 'b', 'empty'):
        home = tmp_path / name
        home.mkdir()
        cfg = {'vault': {'bitwarden': {'enabled': False}}}
        if name != 'empty':
            # Stand-in for the existing Keychain helper; exercise the documented adapter.
            helper = home / 'helper.py'
            helper.write_text('import sys\nassert sys.argv[1:] == '
                              + repr(['get', 'synthetic-connect-service', name])
                              + '\nprint(' + repr('synthetic-' + name) + ')\n')
            command = shlex.join([sys.executable, str(helper), 'get', 'synthetic-connect-service', name])
            cfg['secrets'] = {'command': {'enabled': True, 'command':
                f'token=$({command}) || exit 1; test -n "$token" || exit 1; '
                f"printf 'OP_CONNECT_HOST=http://127.0.0.1:{server.server_port}\\nOP_CONNECT_TOKEN=%s\\n' \"$token\""}}
        (home / 'config.yaml').write_text(yaml.safe_dump(cfg))
        homes[name] = home
    set_multiplex_active(True)
    try:
        for name in ('a', 'b', 'a', 'empty', 'a'):
            home = homes[name]
            home_token = set_hermes_home_override(home)
            hydrate_profile_secret_sources(home)
            scope_token = set_secret_scope(build_profile_secret_scope(home))
            try:
                selected = [b for b in enabled_backends() if b.name == 'onepassword']
                if name == 'empty':
                    assert selected == []
                    continue
                assert len(selected) == 1, 'Connect must not require an installed op CLI'
                listed = json.loads(vault.browser_vault_list())
                assert listed['items'][0]['handle'] == HANDLE
                assert listed['items'][0]['origin'] == f'https://{name}.example'
                assert 'synthetic-password' not in json.dumps(listed)
                monkeypatch.setattr(vault, '_current_page_origin', lambda _: 'https://evil.example')
                before = len(fills)
                denied = json.loads(vault.browser_vault_fill(HANDLE, task_id='test'))
                assert not denied['success'] and len(fills) == before
                monkeypatch.setattr(vault, '_current_page_origin', lambda _, n=name: f'https://{n}.example')
                result = json.loads(vault.browser_vault_fill(HANDLE, task_id='test'))
                assert result['success'] and 'synthetic-password' not in json.dumps(result)
                assert 'synthetic-password-' + name in fills[-1]
                assert 'synthetic-password-' + ('b' if name == 'a' else 'a') not in fills[-1]
            finally:
                reset_secret_scope(scope_token)
                reset_hermes_home_override(home_token)
        assert {auth for _, auth in calls} == {'Bearer synthetic-a', 'Bearer synthetic-b'}
        assert os.environ['OP_CONNECT_TOKEN'] == 'launch-profile-must-not-leak'
    finally:
        set_multiplex_active(False)
        for home in homes.values():
            reset_secret_source_cache(home)
        server.shutdown(); server.server_close(); thread.join()


def test_connect_scope_errors_are_not_disguised_as_a_locked_manager():
    from agent.secret_scope import UnscopedSecretError
    from agent.vault_backends.onepassword import OnePasswordLoginBackend
    backend = OnePasswordLoginBackend()
    token = set_secret_scope(None)
    set_multiplex_active(True)
    try:
        with pytest.raises(UnscopedSecretError):
            backend.is_unlocked()
    finally:
        set_multiplex_active(False)
        reset_secret_scope(token)
