import importlib.util
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest


PROVIDERS = ['openai-codex', 'claude-subscription-directsdk-experimental', 'vertex']
BASE = 'https://chatgpt.com/backend-api/codex'

@pytest.mark.parametrize('settings', [None, True, [], {}, {'enabled': 'true'}, {'enabled': 1}])
def test_host_opt_in_requires_exact_boolean(tmp_path, monkeypatch, settings):
    m = usage_module()
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text(json.dumps({'account_usage_routing': settings}))
    monkeypatch.setattr(m, 'get_subscription_usage', lambda *a, **kw: pytest.fail('must not collect'))
    assert m.usage_for_turn(SimpleNamespace()) == {}


def test_http_response_cap(monkeypatch):
    import httpx
    m = usage_module()
    original = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kw: original(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=b' ' * 65537)), **kw))
    assert m._fetch_runtime_usage(base_url=BASE, api_key='PRIVATE') is None



def usage_module():
    spec = importlib.util.spec_from_file_location('isolated_usage', Path(__file__).parents[2] / 'agent' / 'account_usage_routing.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def snapshot(**overrides):
    values = dict(provider='openai-codex', fetched_at=datetime.now(timezone.utc),
                  windows=[SimpleNamespace(label='Weekly', used_percent=10, reset_at=None)],
                  unavailable_reason=None, raw={'account_id': 'PRIVATE'})
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize('windows', [[], None, 'PRIVATE', [SimpleNamespace(label='PRIVATE', used_percent=10)],
    [SimpleNamespace(label='Weekly', used_percent=float('nan'))], [SimpleNamespace(label='Weekly', used_percent=True)]])
def test_bad_snapshot_unknown(windows):
    m = usage_module()
    result = m.build_subscription_usage(PROVIDERS, fetcher=lambda p: snapshot(windows=windows))
    assert result['openai-codex']['status'] == 'unknown'
    assert 'PRIVATE' not in json.dumps(result)


@pytest.mark.parametrize('delta', [-1000, 1000])
def test_stale_or_future_snapshot_unknown(delta):
    m = usage_module()
    result = m.build_subscription_usage(PROVIDERS, fetcher=lambda p: snapshot(
        fetched_at=datetime.now(timezone.utc) + timedelta(seconds=delta)))
    assert result['openai-codex']['status'] == 'unknown'


@pytest.mark.parametrize('overrides', [{'unavailable_reason': 'PRIVATE'}, {'available': False}])
def test_unavailable_snapshot_unknown(overrides):
    m = usage_module()
    result = m.build_subscription_usage(PROVIDERS, fetcher=lambda p: snapshot(**overrides))
    assert result['openai-codex']['status'] == 'unknown'
    assert 'PRIVATE' not in json.dumps(result)


def test_runtime_binding_nonblocking_singleflight_and_no_worker_buildup(monkeypatch):
    m = usage_module()
    entered, release = threading.Event(), threading.Event()
    calls = []
    def fetch(**kw):
        calls.append(kw)
        entered.set()
        release.wait(3)
        return snapshot()
    monkeypatch.setattr(m, '_fetch_runtime_usage', fetch)
    def get(key='PRIVATE', profile='profile-a'):
        return m.get_subscription_usage(PROVIDERS, provider='openai-codex', base_url=BASE,
                                        api_key=key, profile_home=profile)
    start = time.monotonic()
    assert get()['openai-codex']['status'] == 'unknown'
    assert entered.wait(1)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: get(), range(30)))
        for i in range(30):
            assert get(str(i), str(i))['openai-codex']['status'] == 'unknown'
        assert time.monotonic() - start < 1
        assert len(calls) == 1
        assert calls[0]['api_key'] == 'PRIVATE'
        assert calls[0]['base_url'] == BASE
        assert 'PRIVATE' not in repr(m._cache)
        assert all(r['openai-codex']['status'] == 'unknown' for r in results)
    finally:
        release.set()
    for _ in range(100):
        result = get()
        if result['openai-codex']['status'] == 'available':
            break
        time.sleep(.01)
    assert result['openai-codex']['status'] == 'available'
    result['openai-codex']['windows'][0]['remaining_percent'] = -100
    assert get()['openai-codex']['windows'][0]['remaining_percent'] == 90
    assert get(profile='other')['openai-codex']['status'] == 'unknown'
    assert get(key='other')['openai-codex']['status'] == 'unknown'


def test_no_runtime_credentials_never_resolves_or_scrapes(monkeypatch):
    m = usage_module()
    monkeypatch.setattr(m, '_fetch_runtime_usage', lambda **kw: pytest.fail('must not fetch'))
    for kwargs in ({}, {'provider': 'vertex', 'api_key': 'PRIVATE', 'base_url': BASE},
                   {'provider': 'openai-codex', 'api_key': 'PRIVATE', 'base_url': 'https://evil.invalid'}):
        result = m.get_subscription_usage(PROVIDERS, profile_home='profile', **kwargs)
        assert all(v['status'] == 'unknown' for v in result.values())


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True, -1, 101, 10 ** 1000, [], {}])
def test_boundary_rejects_malformed_percent(bad):
    m = usage_module()
    value = {'openai-codex': {'status': 'available', 'fetched_at': datetime.now(timezone.utc).isoformat(),
                             'windows': [{'label': 'Weekly', 'remaining_percent': bad}]}}
    assert m.sanitize_subscription_usage(value)['openai-codex']['status'] == 'unknown'


@pytest.mark.parametrize('reset', ['PRIVATE', '2026-09-21T00:00:00', [], {}, float('nan')])
def test_boundary_malformed_reset_is_unknown(reset):
    m = usage_module()
    value = {'openai-codex': {'status': 'available', 'fetched_at': datetime.now(timezone.utc).isoformat(),
                             'windows': [{'label': 'Weekly', 'remaining_percent': 10, 'reset_at': reset}]}}
    result = m.sanitize_subscription_usage(value)
    assert result['openai-codex']['status'] == 'unknown'
    assert 'PRIVATE' not in json.dumps(result)


def test_runtime_fetch_uses_only_supplied_credential_no_redirect_retry(monkeypatch):
    import httpx
    m = usage_module()
    calls = []
    def transport(request):
        calls.append(request)
        assert str(request.url) == 'https://chatgpt.com/backend-api/wham/usage'
        assert request.headers['Authorization'] == 'Bearer PRIVATE'
        assert 'chatgpt-account-id' not in request.headers
        return httpx.Response(200, json={'account_id': 'PRIVATE', 'rate_limit': {
            'primary_window': {'used_percent': 25, 'limit_window_seconds': 604800}}})
    original = httpx.Client
    def client(**kw):
        assert kw == {'timeout': 1.0, 'follow_redirects': False, 'trust_env': False}
        return original(transport=httpx.MockTransport(transport), **kw)
    monkeypatch.setattr(httpx, 'Client', client)
    result = m.build_subscription_usage(PROVIDERS, fetcher=lambda _: m._fetch_runtime_usage(base_url=BASE, api_key='PRIVATE'))
    assert result['openai-codex']['windows'] == [{'label': 'Weekly', 'remaining_percent': 75.0}]
    assert len(calls) == 1 and 'PRIVATE' not in json.dumps(result)


@pytest.mark.parametrize('reset', ['PRIVATE', True, float('nan'), 10 ** 1000])
def test_provider_malformed_reset_is_unknown(monkeypatch, reset):
    import httpx
    m = usage_module()
    original = httpx.Client
    body = json.dumps({'rate_limit': {'primary_window': {'used_percent': 25, 'reset_at': reset}}})
    monkeypatch.setattr(httpx, 'Client', lambda **kw: original(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text=body)), **kw))
    result = m.build_subscription_usage(PROVIDERS, fetcher=lambda _: m._fetch_runtime_usage(base_url=BASE, api_key='PRIVATE'))
    assert result['openai-codex']['status'] == 'unknown'


@pytest.mark.parametrize('status', [302, 401, 429, 500])
def test_runtime_http_failure_is_unknown_without_retry(monkeypatch, status):
    import httpx
    m = usage_module()
    calls = []
    def transport(request):
        calls.append(request)
        return httpx.Response(status, headers={'Location': 'https://evil.invalid'}, text='PRIVATE')
    original = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kw: original(transport=httpx.MockTransport(transport), **kw))
    result = m.build_subscription_usage(PROVIDERS, fetcher=lambda _: m._fetch_runtime_usage(base_url=BASE, api_key='PRIVATE'))
    assert result['openai-codex']['status'] == 'unknown'
    assert len(calls) == 1 and 'PRIVATE' not in json.dumps(result)


def test_cache_is_bounded_and_expiry_rechecks(monkeypatch):
    m = usage_module()
    monkeypatch.setattr(m, '_fetch_runtime_usage', lambda **kw: snapshot())
    for i in range(m._MAX_CACHE_ENTRIES + 5):
        kwargs = dict(provider='openai-codex', api_key=str(i), base_url=BASE, profile_home='profile')
        m.get_subscription_usage(PROVIDERS, **kwargs)
        for _ in range(100):
            if not m._inflight:
                break
            time.sleep(.001)
    assert len(m._cache) == m._MAX_CACHE_ENTRIES
    monkeypatch.setattr(m, '_CACHE_TTL_SECONDS', -1)
    assert m.get_subscription_usage(PROVIDERS, **kwargs)['openai-codex']['status'] == 'unknown'
