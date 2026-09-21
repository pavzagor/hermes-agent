"""Automatic routes mutate the cached runtime, not its input configuration signature."""
from types import SimpleNamespace as NS
from unittest.mock import MagicMock
import threading


def test_warm_cross_provider_route_reuses_agent_and_frozen_prefix(monkeypatch, tmp_path):
    from agent import turn_routing as routing
    from gateway.run_agent_cache import GatewayAgentCacheMixin
    from gateway.run_turn_runner import TurnRunner
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(routing, '_config', lambda: {
        'enabled': True, 'models': {'other': ['small']}})
    monkeypatch.setattr('hermes_cli.runtime_provider.resolve_runtime_provider', lambda **kw: {
        'provider': 'other', 'base_url': 'https://other.example/v1', 'api_key': 'test'})
    agent = NS(session_id='s', provider='initial', model='big', _cached_system_prompt='PREFIX')
    def switch(**kw):
        agent.provider, agent.model = kw['new_provider'], kw['new_model']
        agent._cached_system_prompt = None
    agent.switch_model = switch
    runtime = {'provider': 'initial', 'api_key': 'test', 'base_url': 'https://initial.example'}
    signature = GatewayAgentCacheMixin._agent_config_signature('big', runtime, [], '')
    runner = MagicMock()
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache = {'key': (agent, signature, 2, 's')}
    runner._agent_config_signature = GatewayAgentCacheMixin._agent_config_signature
    runner._extract_cache_busting_config.return_value = {}
    ctx = NS(session_key='key', session_id='s', enabled_toolsets=[], user_config={},
             source=NS(user_id=None, user_id_alt=None), _interrupt_depth=0)
    turn = TurnRunner(runner, ctx)
    monkeypatch.setattr(turn, '_skip_context_files', lambda *a: False)
    monkeypatch.setattr(turn, '_cached_sid_is_dead', lambda *a: ('s', False))
    monkeypatch.setattr(turn, '_current_message_count', lambda: 2)
    monkeypatch.setattr(turn, '_build_fresh_agent', lambda *a: (_ for _ in ()).throw(AssertionError('rebuilt')))
    assert routing.apply_route(agent, {'provider': 'other', 'model': 'small'}, is_first_turn=True)
    selected, reused = turn._resolve_turn_agent(
        {'model': 'big', 'runtime': runtime}, 'telegram', '', 5, None, None)
    assert reused and selected is agent
    assert selected.provider == 'other' and selected.model == 'small'
    assert selected._cached_system_prompt == 'PREFIX'
