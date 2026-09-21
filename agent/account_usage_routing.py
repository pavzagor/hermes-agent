"""Nonblocking, account-bound quota hints; only a whitelisted schema may leave here."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import secrets
import threading
import time
from types import SimpleNamespace
from typing import Any

_PROVIDERS = ('openai-codex', 'claude-subscription-directsdk-experimental', 'vertex')
_LABELS = frozenset({'Session', 'Weekly'})
_REASONS = frozenset({'no_supported_quota_endpoint', 'usage_unavailable', 'account_unproven', 'refresh_pending'})
_CACHE_TTL_SECONDS = 60.0
_MAX_CACHE_ENTRIES = 64
_MAX_AGE_SECONDS = 120.0
_BASE_URLS = frozenset({'https://chatgpt.com/backend-api/codex', 'https://chatgpt.com/backend-api'})
_cache = OrderedDict()
_lock = threading.Lock()
_inflight = set()
_fingerprint_key = secrets.token_bytes(32)


def _unknown(reason='usage_unavailable'):
    return {'status': 'unknown', 'reason': reason if isinstance(reason, str) and reason in _REASONS else 'usage_unavailable'}


def _date(value):
    try:
        dt = datetime.fromisoformat(value) if isinstance(value, str) else value
        if not isinstance(dt, datetime) or dt.tzinfo is None:
            return None
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError):
        return None


def _percent(value):
    return type(value) in (int, float) and 0 <= value <= 100 and math.isfinite(value)


def sanitize_subscription_usage(value):
    """Final egress boundary: never trust callers, provider labels or raw objects."""
    if not isinstance(value, dict):
        return {}
    result = {}
    for provider in _PROVIDERS:
        if provider not in value:
            continue
        if provider != 'openai-codex':
            result[provider] = _unknown('no_supported_quota_endpoint')
            continue
        item = value[provider]
        result[provider] = _unknown()
        if not isinstance(item, dict):
            continue
        if item.get('status') != 'available':
            result[provider] = _unknown(item.get('reason'))
            continue
        stamp = _date(item.get('fetched_at'))
        if stamp is None or not 0 <= (datetime.now(timezone.utc) - stamp).total_seconds() <= _MAX_AGE_SECONDS:
            continue
        windows = item.get('windows')
        if not isinstance(windows, (tuple, list)) or not 1 <= len(windows) <= 2:
            continue
        clean = []
        for window in windows:
            if (not isinstance(window, dict) or not isinstance(window.get('label'), str) or window['label'] not in _LABELS
                    or not _percent(window.get('remaining_percent'))):
                break
            if window['label'] in {w['label'] for w in clean}:
                break
            entry = {'label': window['label'], 'remaining_percent': float(window['remaining_percent'])}
            reset = _date(window.get('reset_at'))
            if window.get('reset_at') is not None and reset is None:
                break
            if reset is not None:
                entry['reset_at'] = reset.isoformat()
            clean.append(entry)
        else:
            result[provider] = {'status': 'available', 'fetched_at': stamp.isoformat(), 'windows': clean}
    return result


def build_subscription_usage(providers, *, fetcher=None):
    """Project injected snapshots. No implicit credential resolution or file access."""
    result = {}
    for provider in _PROVIDERS:
        if provider not in providers:
            continue
        if provider != 'openai-codex':
            result[provider] = _unknown('no_supported_quota_endpoint')
            continue
        result[provider] = _unknown()
        try:
            snapshot = fetcher(provider) if fetcher is not None else None
            if (snapshot is None or getattr(snapshot, 'unavailable_reason', None)
                    or getattr(snapshot, 'available', True) is not True):
                continue
            if getattr(snapshot, 'provider', provider) != provider:
                continue
            windows = getattr(snapshot, 'windows', None)
            if not isinstance(windows, (tuple, list)) or not 1 <= len(windows) <= 2:
                continue
            projected = []
            for window in windows:
                used = getattr(window, 'used_percent', None)
                if not _percent(used):
                    break
                projected.append({'label': getattr(window, 'label', None),
                                  'remaining_percent': round(100.0 - used, 2),
                                  'reset_at': getattr(window, 'reset_at', None)})
            else:
                result[provider] = {'status': 'available', 'fetched_at': getattr(snapshot, 'fetched_at', None),
                                    'windows': projected}
        except Exception:
            pass  # Provider exceptions may contain secrets; never log them.
    return sanitize_subscription_usage(result)


def _fetch_runtime_usage(*, base_url, api_key):
    """Read only the known Codex endpoint using this runtime's exact credential.

    No auth-store lookup, refresh, retry, proxy, redirect or account fallback.
    Called in the single background slot, never on the routing critical path.
    """
    import httpx
    if base_url not in _BASE_URLS:
        return None
    url = 'https://chatgpt.com/backend-api/wham/usage'
    started = time.monotonic()
    with httpx.Client(timeout=1.0, follow_redirects=False, trust_env=False) as client:
        from agent.codex_headers import codex_account_headers
        headers = {'Authorization': 'Bearer ' + api_key, **codex_account_headers(api_key)}
        with client.stream('GET', url, headers=headers) as response:
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_bytes():
                if time.monotonic() - started > 2 or len(body) + len(chunk) > 65536:
                    return None
                body.extend(chunk)
    payload = json.loads(body)
    rate_limit = payload.get('rate_limit')
    if not isinstance(rate_limit, dict):
        return None
    windows = []
    for key, fallback in (('primary_window', 'Session'), ('secondary_window', 'Weekly')):
        window = rate_limit.get(key)
        if window is None:
            continue
        if not isinstance(window, dict):
            return None
        seconds = window.get('limit_window_seconds')
        label = {18000: 'Session', 604800: 'Weekly'}.get(seconds, fallback)
        reset = window.get('reset_at')
        if reset is not None:
            if type(reset) not in (int, float) or not math.isfinite(reset):
                return None
            reset = datetime.fromtimestamp(reset, timezone.utc)
        windows.append(SimpleNamespace(label=label, used_percent=window.get('used_percent'), reset_at=reset))
    return SimpleNamespace(provider='openai-codex', windows=windows, fetched_at=datetime.now(timezone.utc))


def get_subscription_usage(providers, *, provider='', base_url='', api_key='', profile_home=''):
    """Return a defensive cache copy immediately; cold/stale/busy means unknown.

    One global in-flight slot, no queue: a stuck I/O worker cannot accumulate
    abandoned threads. Cache keys are process-salted opaque fingerprints of the
    actual runtime credential, endpoint and profile. No raw identities persist.
    """
    requested = tuple(providers)
    providers = tuple(p for p in _PROVIDERS if p in requested)
    result = {p: _unknown('account_unproven' if p == 'openai-codex' else 'no_supported_quota_endpoint')
              for p in providers}
    if ('openai-codex' not in result or provider != 'openai-codex'
            or not isinstance(api_key, str) or not api_key
            or not isinstance(base_url, str) or base_url not in _BASE_URLS
            or not isinstance(profile_home, str) or not profile_home):
        return result
    identity = json.dumps([profile_home, provider, base_url, api_key]).encode()
    key = hmac.new(_fingerprint_key, identity, hashlib.sha256).hexdigest()
    now = time.monotonic()
    with _lock:
        cached = _cache.get(key)
        if cached is not None and now - cached[0] < _CACHE_TTL_SECONDS:
            _cache.move_to_end(key)
            result['openai-codex'] = deepcopy(cached[1])
            return sanitize_subscription_usage(result)
        result['openai-codex'] = _unknown('refresh_pending')
        if _inflight:
            return result
        _inflight.add(key)

    def refresh():
        try:
            value = build_subscription_usage(('openai-codex',), fetcher=lambda _: _fetch_runtime_usage(
                base_url=base_url, api_key=api_key))['openai-codex']
            with _lock:
                _cache[key] = (time.monotonic(), value)
                _cache.move_to_end(key)
                while len(_cache) > _MAX_CACHE_ENTRIES:
                    _cache.popitem(last=False)
        finally:
            with _lock:
                _inflight.discard(key)
    try:
        threading.Thread(target=refresh, name='account-usage-refresh', daemon=True).start()
    except RuntimeError:
        with _lock:
            _inflight.discard(key)
    return result


def usage_for_turn(agent):
    """Opt-in internal adapter. Only the returned plain snapshot reaches hooks.

    Read no credential store, resolve no account, and capture the owning profile
    before starting I/O. Unsupported runtime identities remain unknown.
    """
    try:
        from hermes_cli.config_effective import load_user_config_effective
        from hermes_constants import get_hermes_home
        config = load_user_config_effective()
        settings = config.get('account_usage_routing', {}) if isinstance(config, dict) else {}
        if not isinstance(settings, dict) or settings.get('enabled') is not True:
            return {}
        return get_subscription_usage(
            _PROVIDERS, provider=getattr(agent, 'provider', ''),
            base_url=getattr(agent, 'base_url', ''), api_key=getattr(agent, 'api_key', ''),
            profile_home=str(get_hermes_home().resolve()))
    except Exception:
        return {}  # Advisory only; never expose transport/config exceptions.
