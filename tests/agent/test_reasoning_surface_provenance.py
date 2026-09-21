"""Real surface handlers retain explicit intent; display toggles do not pin."""
from types import SimpleNamespace as NS
from unittest.mock import patch

def test_cli_model_reresolve_retains_explicit_effort():
    from hermes_cli.cli_model_switch_mixin import _resolve_cli_reasoning
    for provenance in ('request', 'session'):
        cli = NS(model='gpt-6-astra', reasoning_config={'enabled': True, 'effort': 'low'},
                 reasoning_provenance=provenance)
        _resolve_cli_reasoning(cli)
        assert cli.reasoning_config == {'enabled': True, 'effort': 'low'}
        assert cli.reasoning_provenance == provenance



def test_cli_auto_and_cold_once_restore(tmp_path, monkeypatch):
    from cli import HermesCLI
    from agent.turn_reasoning import prepare_turn, apply_reasoning, set_session_reasoning
    from agent import turn_reasoning
    monkeypatch.setattr(turn_reasoning, '_config', lambda: {
        'agent': {'reasoning_effort': 'high'}, 'reasoning_routing': {'enabled': True}})
    cli = NS(session_id='cold-cli', model='gpt-6-astra', reasoning_config=None,
             reasoning_provenance='config', agent=None)
    with patch('cli._cprint'):
        HermesCLI._handle_reasoning_command(cli, '/reasoning high')
        snapshot = HermesCLI._snapshot_model_runtime(cli)
        HermesCLI._handle_reasoning_command(cli, '/reasoning xhigh')
        HermesCLI._restore_model_runtime_snapshot(cli, snapshot)
        fresh = NS(session_id='cold-cli', model='gpt-6-astra', provider='openai-codex')
        prepare_turn(fresh)
        assert fresh.reasoning_config['effort'] == 'high'
        HermesCLI._handle_reasoning_command(cli, '/reasoning auto')
        prepare_turn(fresh)
        assert apply_reasoning(fresh, {'effort': 'low'})


def test_tui_auto_display_and_once_restore(monkeypatch):
    import tui_gateway.server as server
    from agent.turn_reasoning import prepare_turn
    from tui_gateway.model_switch import _snapshot_agent_model_runtime, _restore_agent_model_runtime
    a = NS(session_id='tui', model='gpt-6-astra', provider='openai-codex', reasoning_config=None)
    session = {'agent': a, 'session_key': 'tui'}
    with patch.dict(server._sessions, {'ui': session}), patch.object(server, '_write_config_key'), \
         patch.object(server, '_persist_live_session_runtime'), patch.object(server, '_emit'):
        def command(value):
            return server._methods['config.set']('r', {'key': 'reasoning', 'session_id': 'ui', 'value': value})
        assert 'error' not in command('xhigh')
        snapshot = _snapshot_agent_model_runtime(a)
        assert 'error' not in command('low')
        _restore_agent_model_runtime(a, snapshot)
        prepare_turn(a)
        assert a.reasoning_config['effort'] == 'xhigh'
        assert 'error' not in command('hide')
        prepare_turn(a)
        assert a.reasoning_provenance == 'session'
        assert 'error' not in command('auto')
        prepare_turn(a)
        assert a.reasoning_provenance == 'config'
        assert 'create_reasoning_override' not in session


def test_gateway_durable_pin_auto_and_model_independence(tmp_path):
    from gateway.run import GatewayRunner
    from agent.turn_reasoning import prepare_turn
    from agent.turn_routing import set_routing
    runner = object.__new__(GatewayRunner)
    runner.session_store = NS(peek_session_id=lambda key: 'gateway-effort')
    runner._evict_cached_agent = lambda key: None
    runner._load_reasoning_config = lambda *a: None
    runner._save_gateway_config_key = lambda *a: True
    runner._apply_reasoning_selection('key', 'telegram', 'high')
    set_routing('gateway-effort', True)
    fresh = NS(session_id='gateway-effort', model='gpt-6-astra')
    prepare_turn(fresh)
    assert fresh.reasoning_provenance == 'session'
    runner._apply_reasoning_selection('key', 'telegram', 'hide')
    prepare_turn(fresh)
    assert fresh.reasoning_provenance == 'session'
    runner._apply_reasoning_selection('key', 'telegram', 'auto')
    prepare_turn(fresh)
    assert fresh.reasoning_provenance == 'config'


def test_gateway_once_reasoning_is_request_scoped():
    from tests.gateway.test_display_null_turn_wiring import _wire
    from agent.turn_reasoning import prepare_turn
    a = _wire({}, once_reasoning={'enabled': True, 'effort': 'xhigh'})
    prepare_turn(a)
    assert a.reasoning_provenance == 'request'
    assert a.reasoning_config['effort'] == 'xhigh'


def test_gateway_once_record_stages_effort_without_persisting_it():
    import asyncio
    from unittest.mock import AsyncMock
    from gateway.slash_commands_model import GatewayModelCommandsMixin, _ModelSwitchContext
    from agent.turn_reasoning import prepare_turn
    runner = NS(async_session_store=NS(get_or_create_session=AsyncMock(return_value=NS(session_id='once'))),
                _cached_agent_for=lambda key: None, _session_db=None, _pending_model_notes={},
                _session_model_overrides={}, _claim_one_turn_restore=lambda *a: None,
                _evict_cached_agent=lambda *a: None)
    ctx = _ModelSwitchContext(session_key='key', source=None, config_path=None, persist_global=False,
                              one_turn=True, reasoning_effort='xhigh', restore_snapshot={})
    result = NS(new_model='gpt-6-astra', target_provider='openai-codex', provider_label='Codex',
                api_key='', base_url='', api_mode='', request_overrides={}, runtime_capabilities={})
    asyncio.run(GatewayModelCommandsMixin._record_model_switch(
        runner, result, ctx, source=None, one_turn=True, picker=False))
    assert runner._session_model_overrides['key']['reasoning_config']['effort'] == 'xhigh'
    fresh = NS(session_id='once', model='gpt-6-astra')
    prepare_turn(fresh)
    assert fresh.reasoning_provenance == 'config'
