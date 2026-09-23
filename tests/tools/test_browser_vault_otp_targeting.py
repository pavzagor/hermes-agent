"""OTP origin/tab contracts on installed Chrome, driven by Browser Use only.

HERMES_E2E_BROWSER=1 scripts/run_tests.sh tests/tools/test_browser_vault_otp_targeting.py -m integration
Put uvx on PATH and use a short scratch --basetemp (macOS AF_UNIX path limit).
Uses a fresh Chrome profile, loopback fixtures and synthetic codes; no live accounts.
No Playwright or downloaded browser. The real supervisor is the production fill transport.
"""
import json
import os
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import pytest

from agent.vault_store import VaultStore
from tools import browser_vault_tool as vault
from tools.browser_supervisor import SUPERVISOR_REGISTRY
from tools.registry import registry

pytestmark = [pytest.mark.integration, pytest.mark.macos_only,
              pytest.mark.skipif(os.environ.get("HERMES_E2E_BROWSER") != "1", reason="opt-in installed Chrome / Browser Use fixture")]
OTP = '<form><input name=code autocomplete=one-time-code></form>'


@pytest.fixture(scope="module")
def chrome(tmp_path_factory):
    root = tmp_path_factory.mktemp("otp-chrome")
    binary = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
    if not binary.is_file():
        pytest.skip("installed Google Chrome is required")
    uvx = shutil.which("uvx")
    assert uvx, "put the installed uvx on PATH"
    pages = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = pages.get(self.path, '<html><title>OTP fixture</title></html>').encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f'http://127.0.0.1:{server.server_port}'
    other = f'http://localhost:{server.server_port}'
    profile = root / 'chrome'
    process = subprocess.Popen([str(binary), '--headless=new', '--no-first-run', '--no-default-browser-check',
                                '--disable-background-networking', '--remote-debugging-port=0',
                                f'--user-data-dir={profile}', 'about:blank'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    task = 'otp-synthetic-fixture'
    env = {'PATH': os.environ['PATH'], 'HOME': str(root), 'BU_NAME': 'otp-fixture',
           'BU_AUTOSPAWN': '0', 'ANONYMIZED_TELEMETRY': 'false', 'BU_TELEMETRY': '0',
           'BH_AGENT_WORKSPACE': str(root), 'TMPDIR': str(root),
           'BH_RUNTIME_DIR': str(root / 'ipc'), 'BH_TMP_DIR': str(root / 'artifacts'),
           'UV_CACHE_DIR': str(Path.home() / '.cache' / 'uv')}
    try:
        port_file = profile / 'DevToolsActivePort'
        deadline = time.monotonic() + 20
        while not port_file.exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(.1)
        port = port_file.read_text().splitlines()[0]
        endpoint = f'http://127.0.0.1:{port}'
        env['BU_CDP_URL'] = endpoint

        def run(code):
            result = subprocess.run([uvx, 'browser-use'], input=code, text=True, capture_output=True, env=env, timeout=45)
            assert result.returncode == 0, result.stderr
            return result.stdout

        run('new_tab("about:blank")\nprint("fixture ready")')
        with urlopen(endpoint + '/json/version') as response:
            websocket = json.load(response)['webSocketDebuggerUrl']
        sup = SUPERVISOR_REGISTRY.get_or_start(task_id=task, cdp_url=websocket)
        yield SimpleNamespace(run=run, pages=pages, origin=origin, other=other, sup=sup, task=task)
    finally:
        SUPERVISOR_REGISTRY.stop(task)
        if 'BU_CDP_URL' in env:
            subprocess.run([uvx, 'browser-use', '--reload'], env=env, capture_output=True, timeout=15)
        process.terminate()
        process.wait(timeout=15)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def page(chrome, tmp_path, monkeypatch):
    # Close only targets in this fresh, fixture-owned Chrome; never the user's browser.
    chrome.run('for t in cdp("Target.getTargets")["targetInfos"]:\n    if t["type"] == "page": cdp("Target.closeTarget", targetId=t["targetId"])\nnew_tab("about:blank")')
    chrome.pages.clear()
    store = VaultStore(base_dir=tmp_path / 'vault')
    meta = store.add_item('login', 'synthetic', {'identifier_type': 'username', 'identifier': 'fixture',
                                               'password': 'not-a-real-password'}, origin=chrome.origin)
    monkeypatch.setattr('agent.vault_store.get_vault_store', lambda: store)
    asked = []
    mutation = [None]

    def prompt(site, hint):
        asked.append(site)
        if mutation[0]:
            mutation[0]()
        return '246 810'

    monkeypatch.setattr('agent.vault_backends.unlock.can_prompt_here', lambda: True)
    monkeypatch.setattr('agent.vault_backends.unlock.get_code_prompt_callback', lambda: prompt)

    def setup(tabs):
        code = []
        for i, (origin, html) in enumerate(tabs):
            path = f'/fixture-{i}'
            chrome.pages[path] = html
            code.append(f'new_tab({json.dumps(origin + path)})\nwait_for_load()')
        chrome.run('\n'.join(code))

    def fill(handle=meta.id):
        entry = next(e for e in registry.get_all_entries() if e.name == 'browser_vault_enter_code')
        raw = entry.handler({'handle': handle}, task_id=chrome.task)
        assert '246810' not in raw
        return json.loads(raw)

    return SimpleNamespace(**vars(chrome), setup=setup, fill=fill, handle=meta.id, asked=asked,
                           mutation=mutation, store=store)


@pytest.mark.parametrize('case', ['bound', 'hidden', 'label', 'no_handle', 'wrong_origin', 'unknown',
                                    'ambiguous_tabs', 'ambiguous_no_handle', 'ambiguous_fields', 'no_code', 'multi_origin',
                                    'ambiguous_multi_origin', 'wrong_kind', 'unbound', 'ineligible', 'automatic', 'split'])
def test_otp_targets_only_one_eligible_page_on_saved_origins(page, monkeypatch, case):
    p = page
    decoy = '<input name=postal_code><input name=otp hidden><input name=otp disabled><input name=otp readonly><div style="opacity:0"><input autocomplete=one-time-code></div>'
    tabs = [(p.other, OTP), (p.origin, OTP)]
    handle = p.handle
    expected = None
    if case in ('hidden', 'label', 'no_handle', 'split'):
        target = OTP
        if case == 'label': target = '<label>Authentication code<input name=challenge></label>'
        if case == 'split': target = '<form>' + '<input autocomplete=one-time-code maxlength=1>' * 6 + '</form>'
        tabs = [(p.other, decoy), (p.origin, target)]
    if case == 'no_handle': handle = ''
    if case == 'wrong_origin':
        tabs, expected = [(p.other, OTP)], 'no_code_field'
    if case == 'unknown': handle, expected = 'vault_missing', 'invalid_handle'
    if case == 'ineligible': tabs, expected = [(p.origin, '<input type=email autocomplete=one-time-code>')], 'no_code_field'
    if case == 'ambiguous_no_handle': handle, expected = '', 'ambiguous_code_target'
    if case in ('wrong_kind', 'unbound'):
        from agent.vault_store import VaultItemMeta
        meta = VaultItemMeta(id='op:fixture', kind='payment' if case == 'wrong_kind' else 'login',
                            label='synthetic', origin=p.origin if case == 'wrong_kind' else None, created_at='')
        backend = SimpleNamespace(get_meta=lambda _: meta, resolve_otp=lambda _: pytest.fail('must not resolve'))
        monkeypatch.setattr('agent.vault_backends.backend_for_handle', lambda _: backend)
        handle, expected = meta.id, 'invalid_handle' if case == 'wrong_kind' else 'no_origin'
    if case == 'automatic':
        meta = p.store.add_item('login', 'synthetic TOTP', {'identifier_type': 'username', 'identifier': 'fixture',
                               'password': 'not-real', 'otp_secret': 'JBSWY3DPEHPK3PXP'}, origin=p.origin)
        handle = meta.id
        monkeypatch.setattr('agent.vault_store.totp_now', lambda _: '246810')
    if case == 'ambiguous_tabs':
        tabs, expected = [(p.origin, OTP), (p.origin, OTP)], 'ambiguous_code_target'
    if case == 'ambiguous_fields':
        tabs, expected = [(p.origin, OTP + OTP)], 'ambiguous_code_target'
    if case == 'no_code': tabs, expected = [(p.origin, decoy)], 'no_code_field'
    if case in ('multi_origin', 'ambiguous_multi_origin'):
        from agent.vault_store import VaultItemMeta
        meta = VaultItemMeta(id='op:fixture', kind='login', label='synthetic', origin=p.other,
                            allowed_origins=(p.other, p.origin), created_at='')
        backend = SimpleNamespace(get_meta=lambda _: meta, resolve_otp=lambda _: None)
        monkeypatch.setattr('agent.vault_backends.backend_for_handle', lambda _: backend)
        tabs, handle = [(p.other, decoy), (p.origin, OTP)], meta.id
        if case == 'ambiguous_multi_origin':
            tabs, expected = [(p.other, OTP), (p.origin, OTP)], 'ambiguous_code_target'
    p.setup(tabs)
    result = p.fill(handle)
    if expected:
        assert result.get('error_type') == expected, result
        assert not result['success'] and not p.asked
    else:
        assert result['success'] and result['origin'] == p.origin, result
        assert result['filled_fields'] == (6 if case == 'split' else 1)
        assert p.asked == ([] if case == 'automatic' else [p.origin.split('://')[1]])
        # Boolean-only readback from the exact production supervisor's selected page.
        assert p.sup.evaluate_runtime("Array.from(document.querySelectorAll('input')).map(e=>e.value).join('') === '246810'")['result']
    # Inspect every fixture-owned tab, not just the selected supervisor page.
    p.run("written=[]\nfor t in cdp('Target.getTargets')['targetInfos']:\n"
          "    if t['type'] == 'page' and t['url'].startswith('http'):\n"
          "        if js(\"Array.from(document.querySelectorAll('input')).some(e=>Boolean(e.value))\", target_id=t['targetId']): written.append(t['url'])\n"
          f"assert len(written) == {0 if expected else 1}\n"
          f"assert not written or written[0].startswith({p.origin + '/'!r})")
    if case == 'bound':
        # Shared write guards must preserve the previously fixed signup pair.
        p.setup([(p.origin, '<form action=/users/register/><input type=password name=password><input type=password name=password-repeat></form>')])
        entry = next(e for e in registry.get_all_entries() if e.name == 'browser_vault_fill')
        result = json.loads(entry.handler({'handle': p.handle}, task_id=p.task))
        assert result['success'] and result['filled_fields'] == 2
        assert p.sup.evaluate_runtime("Array.from(document.querySelectorAll('input')).every(e=>Boolean(e.value))")['result']


@pytest.mark.parametrize('mutation', ['origin', 'same_origin_navigation', 'other_tab', 'hidden', 'focus_hidden',
                                     'type', 'name', 'form', 'transport_error'])
def test_otp_prompt_cannot_redirect_or_echo_the_code(page, monkeypatch, mutation):
    p = page
    p.setup([(p.origin, OTP)])
    if mutation == 'origin':
        p.mutation[0] = lambda: p.run(f'goto_url({json.dumps(p.other + "/fixture-0")})\nwait_for_load()')
    elif mutation == 'same_origin_navigation':
        p.mutation[0] = lambda: p.run(f'goto_url({json.dumps(p.origin + "/fixture-0?new")})\nwait_for_load()')
    elif mutation == 'other_tab':
        def switch():
            p.run(f'new_tab({json.dumps(p.origin + "/fixture-0")})\nwait_for_load()')
            # A concurrent focus to another same-origin page must not inherit inspection stamps.
            p.sup.focus_page(p.origin, accept="!document.querySelector('[data-hermes-vault-slot]')")
        p.mutation[0] = switch
    elif mutation == 'hidden':
        p.mutation[0] = lambda: p.run("js(\"document.querySelector('input').hidden=true\")")
    elif mutation == 'focus_hidden':
        p.mutation[0] = lambda: p.run("cdp('Page.bringToFront')\njs(\"(() => { const e=document.querySelector('input'); e.blur(); e.onfocus = event => {event.target.hidden=true}; return true; })()\")")
    elif mutation in ('type', 'name', 'form'):
        script = {'type': "document.querySelector('input').type='email'",
                  'name': "document.querySelector('input').name='postal_code'",
                  'form': "document.body.appendChild(document.querySelector('input'))"}[mutation]
        p.mutation[0] = lambda: p.run(f'js({json.dumps(script)})')
    else:
        def fail(*args):
            raise RuntimeError('fixture transport error includes 246810')
        monkeypatch.setattr(vault, '_eval_js_secret', fail)
    result = p.fill()
    assert not result['success'], result
    assert p.asked == [p.origin.split('://')[1]]
    assert p.sup.evaluate_runtime("!Array.from(document.querySelectorAll('input')).some(e=>e.value)")['result']
