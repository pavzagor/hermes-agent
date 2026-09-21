"""Exercise real API selection/construction and turn hooks with an offline runtime."""
import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent import turn_routing
from agent.turn_context import _collect_pre_llm_call_context
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _request_agent_overrides


@pytest.fixture
def adapter(monkeypatch):
    import gateway.run as gateway_run
    import hermes_cli.lifecycle
    import run_agent

    monkeypatch.setattr(turn_routing, '_config', lambda: {
        'enabled': True, 'models': {'p': ['big', 'small']}})
    monkeypatch.setattr(gateway_run, '_resolve_runtime_agent_kwargs', lambda: {'provider': 'p'})
    monkeypatch.setattr(gateway_run, '_resolve_gateway_model', lambda *a: 'big')
    monkeypatch.setattr(gateway_run, '_load_gateway_config', lambda: {})
    monkeypatch.setattr(gateway_run, '_current_max_iterations', lambda: 2)
    monkeypatch.setattr(gateway_run.GatewayRunner, '_load_reasoning_config', lambda *a: None)
    monkeypatch.setattr(gateway_run.GatewayRunner, '_load_fallback_model', lambda: None)
    monkeypatch.setattr(hermes_cli.lifecycle, 'invoke_hook', lambda *a, **kw: [
        {'route': {'model': 'small'}, 'context': 'unrelated hook still runs'}])

    class Runtime:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self._cached_system_prompt = 'PREFIX'
            self.session_prompt_tokens = self.session_completion_tokens = self.session_total_tokens = 0

        def switch_model(self, **kwargs):
            self.model = kwargs['new_model']

        def run_conversation(self, user_message=None, conversation_history=None, **kwargs):
            context = _collect_pre_llm_call_context(
                self, effective_task_id=self.session_id, turn_id='turn',
                original_user_message=user_message, messages=[], conversation_history=conversation_history or [])
            assert 'unrelated hook still runs' in context
            assert self._cached_system_prompt == 'PREFIX'
            return {'final_response': self.model, 'messages': []}

    monkeypatch.setattr(run_agent, 'AIAgent', Runtime)
    instance = APIServerAdapter(PlatformConfig())
    monkeypatch.setattr(instance, '_apply_provider_runtime', lambda runtime, provider, **kw: True)
    return instance


@pytest.mark.asyncio
@pytest.mark.parametrize('selection', [
    {'requested_model': 'big'}, {'requested_provider': 'p'},
    {'requested_model': 'big', 'requested_provider': 'p', 'confirmed_runtime_lock': True},
    {'route': {'model': 'big', 'provider': 'p'}}, {'session_model': 'big'},
    {},
])
@pytest.mark.parametrize('storage_failure', [False, True])
async def test_run_agent_selection_precedes_restore_and_directives(adapter, selection, storage_failure, monkeypatch):
    turn_routing._write_state('session', {
        'enabled': True, 'automatic_route': {'model': 'small', 'provider': 'p'}})
    if storage_failure:
        def fail(*args, **kwargs):
            raise OSError('storage unavailable')
        monkeypatch.setattr(turn_routing, 'atomic_json_write', fail)
    result, _ = await adapter._run_agent(
        user_message='hello', conversation_history=[], session_id='session',
        requested_runtime={'model': 'big', 'provider': 'p'} if selection else {}, **selection)
    assert result['final_response'] == ('big' if selection else 'small')
    if storage_failure and selection:
        # A fresh runtime with no request selection still sees the failed-write pin.
        recreated, _ = await adapter._run_agent(
            user_message='next', conversation_history=[], session_id='session')
        assert recreated['final_response'] == 'big'


@pytest.mark.asyncio
@pytest.mark.parametrize('body, expected', [
    ({'model': 'big', 'provider': 'p'}, 'big'),
    ({'model': 'hermes-agent'}, 'small'),
])
async def test_runs_http_selection_precedes_automatic_route(adapter, body, expected):
    turn_routing._write_state('session', {
        'enabled': True, 'automatic_route': {'model': 'small', 'provider': 'p'}})
    app = web.Application()
    app.router.add_post('/v1/runs', adapter._handle_runs)
    app.router.add_get('/v1/runs/{run_id}', adapter._handle_get_run)
    async with TestClient(TestServer(app)) as client:
        response = await client.post('/v1/runs', json={
            'input': 'hello', 'session_id': 'session', **body})
        assert response.status == 202, await response.text()
        payload = await response.json()
        run_id = payload['run_id']
        for _ in range(200):
            status_response = await client.get(f'/v1/runs/{run_id}')
            status = await status_response.json()
            if status.get('status') in {'completed', 'failed'}:
                break
            await asyncio.sleep(.01)
        assert status.get('status') == 'completed', status
        assert status.get('output') == expected


def test_compatibility_placeholder_is_not_a_manual_selection(adapter):
    overrides = _request_agent_overrides(
        {'model': 'gpt-4o'}, virtual_model='hermes-agent', allow_bare_model=False)
    agent = adapter._create_agent(session_id='placeholder', **overrides)
    assert agent.run_conversation(user_message='hello')['final_response'] == 'small'
