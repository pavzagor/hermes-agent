"""Turn directives must preserve prompt prefixes and explicit model provenance."""
import json
from types import SimpleNamespace
import pytest


def test_directive_is_allowlisted_pinned_and_cache_stable(tmp_path, monkeypatch):
    from agent import turn_routing as routing
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    cfg = {'enabled': True, 'models': {'p': ['big', 'small']}}
    monkeypatch.setattr(routing, '_config', lambda: cfg)
    calls = []
    agent = SimpleNamespace(session_id='s', model='big', provider='p', _cached_system_prompt='PREFIX')
    def switch(**kw):
        calls.append(kw)
        agent.model = kw['new_model']
        agent._cached_system_prompt = None
    agent.switch_model = switch
    assert not routing.apply_route(agent, {'model':'small'}, is_first_turn=False)
    routing.set_routing('s', True)
    assert not routing.apply_route(agent, {'model':'alien'}, is_first_turn=False)
    assert routing.apply_route(agent, {'model':'small'}, is_first_turn=False)
    assert agent._cached_system_prompt == 'PREFIX'
    assert len(calls) == 1 and calls[0]['provenance'] == 'automatic_routing'
    routing.pin_model('s')
    assert not routing.apply_route(agent, {'model':'big'}, is_first_turn=False)
    assert len(calls) == 1


def test_hook_directive_only_consumed_at_turn_boundary(monkeypatch, tmp_path):
    from agent import turn_routing as routing
    from agent.turn_context import _collect_pre_llm_call_context
    import hermes_cli.lifecycle
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(routing, '_config', lambda: {'enabled':True,'models':{'p':['big','small']}})
    agent = SimpleNamespace(session_id='new',model='big',provider='p',_cached_system_prompt='PREFIX')
    calls = []
    def switch(**kw):
        calls.append(kw)
        agent.model = kw['new_model']
    agent.switch_model = switch
    monkeypatch.setattr(hermes_cli.lifecycle, 'invoke_hook', lambda *a, **kw: [{'route':{'model':'small'},'context':'FULL SKILLS'}])
    result = _collect_pre_llm_call_context(agent,effective_task_id='t',turn_id='turn',original_user_message='request',messages=[],conversation_history=[])
    assert result == 'FULL SKILLS'
    assert agent.model == 'small' and len(calls) == 1
    assert routing.routing_enabled('new', is_first_turn=False)


def test_selected_skills_are_loaded_fully_not_spilled(monkeypatch, tmp_path):
    from agent.turn_context import _collect_pre_llm_call_context
    import hermes_cli.lifecycle
    import tools.skills_tool
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    loaded = []
    def view(name, **kwargs):
        loaded.append(name)
        return json.dumps({'success': True, 'content': name * (40000 if name in {'one','two'} else 1)})
    monkeypatch.setattr(tools.skills_tool, 'skill_view', view)
    names = ['one','two'] + [f'skill_{i}' for i in range(38)]
    monkeypatch.setattr(hermes_cli.lifecycle, 'invoke_hook', lambda *a, **kw: [{'skills':names + ['one']}])
    agent = SimpleNamespace(session_id='s',model='big',provider='p')
    context = _collect_pre_llm_call_context(agent,effective_task_id='t',turn_id='turn',original_user_message='request',messages=[],conversation_history=[])
    assert loaded == names
    assert 'one'*40000 in context and 'two'*40000 in context
    assert 'skill_37' in context


def test_persisted_automatic_route_restores_before_a_failed_classifier(monkeypatch, tmp_path):
    from agent import turn_routing as routing
    from agent.turn_context import _collect_pre_llm_call_context
    import hermes_cli.lifecycle

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(routing, '_config', lambda: {
        'enabled': True, 'models': {'p': ['big', 'small']},
    })
    first = SimpleNamespace(
        session_id='persisted', model='big', provider='p', _cached_system_prompt='PREFIX')
    first.switch_model = lambda **kw: setattr(first, 'model', kw['new_model'])
    routing.set_routing('persisted', True)
    assert routing.apply_route(first, {'model': 'small'}, is_first_turn=False)

    calls = []
    recreated = SimpleNamespace(
        session_id='persisted', model='big', provider='p', _cached_system_prompt='PREFIX')
    def switch(**kwargs):
        calls.append(kwargs)
        recreated.model = kwargs['new_model']
        recreated._cached_system_prompt = None
    recreated.switch_model = switch
    monkeypatch.setattr(
        hermes_cli.lifecycle, 'invoke_hook',
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('classifier unavailable')),
    )

    assert _collect_pre_llm_call_context(
        recreated, effective_task_id='t', turn_id='turn',
        original_user_message='request', messages=[],
        conversation_history=[{'role': 'user', 'content': 'earlier'}],
    ) == ''
    assert recreated.model == 'small'
    assert recreated._cached_system_prompt == 'PREFIX'
    assert calls == [{
        'new_model': 'small', 'new_provider': 'p',
        'provenance': 'automatic_restore',
    }]


def test_manual_pin_wins_over_persisted_automatic_route(monkeypatch, tmp_path):
    from agent import turn_routing as routing

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(routing, '_config', lambda: {
        'enabled': True, 'models': {'p': ['big', 'small']},
    })
    routed = SimpleNamespace(session_id='s', model='big', provider='p', _cached_system_prompt='P')
    routed.switch_model = lambda **kw: setattr(routed, 'model', kw['new_model'])
    routing.set_routing('s', True)
    assert routing.apply_route(routed, {'model': 'small'})
    routing.pin_model('s')

    recreated = SimpleNamespace(session_id='s', model='big', provider='p', _cached_system_prompt='P')
    recreated.switch_model = lambda **kw: pytest.fail('manual pin must prevent restore')
    assert not routing.restore_automatic_route(recreated, is_first_turn=False)
    assert recreated.model == 'big'


def test_corrupt_state_does_not_skip_unrelated_pre_llm_hook(monkeypatch, tmp_path):
    from agent import turn_routing as routing
    from agent.turn_context import _collect_pre_llm_call_context
    import hermes_cli.lifecycle

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(routing, '_config', lambda: {'enabled': True, 'models': {'p': ['big']}})
    state_path = routing._path('s')
    state_path.parent.mkdir(parents=True)
    state_path.write_text('{broken', encoding='utf-8')
    monkeypatch.setattr(
        hermes_cli.lifecycle, 'invoke_hook', lambda *a, **kw: [{'context': 'still runs'}])
    agent = SimpleNamespace(session_id='s', model='big', provider='p')

    assert _collect_pre_llm_call_context(
        agent, effective_task_id='t', turn_id='turn', original_user_message='request',
        messages=[], conversation_history=[{'role': 'user', 'content': 'earlier'}],
    ) == 'still runs'


def test_one_turn_snapshot_restores_prior_automatic_state(monkeypatch, tmp_path):
    from agent import turn_routing as routing

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(routing, '_config', lambda: {
        'enabled': True, 'models': {'p': ['big', 'small']},
    })
    routing.set_routing('s', True)
    agent = SimpleNamespace(session_id='s', model='big', provider='p', _cached_system_prompt='P')
    agent.switch_model = lambda **kw: setattr(agent, 'model', kw['new_model'])
    assert routing.apply_route(agent, {'model': 'small'})
    snapshot = routing.snapshot_routing_state('s')
    routing.pin_model('s')
    assert not routing.routing_enabled('s')
    assert routing.restore_routing_state('s', snapshot)
    assert routing.routing_enabled('s')
    assert json.loads(routing._path('s').read_text())['automatic_route'] == {
        'model': 'small', 'provider': 'p',
    }


def test_compression_carries_pin_and_automatic_route_to_child(monkeypatch, tmp_path):
    from agent import turn_routing as routing
    from agent.conversation_compression import _carry_session_state_to_child
    import hermes_cli.goals
    import hermes_cli.heartbeat
    import hermes_cli.loops

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(routing, '_config', lambda: {
        'enabled': True, 'models': {'p': ['big', 'small']},
    })
    monkeypatch.setattr(hermes_cli.goals, 'migrate_goal_to_session', lambda *a, **k: None)
    monkeypatch.setattr(hermes_cli.heartbeat, 'migrate_heartbeat_to_session', lambda *a, **k: None)
    monkeypatch.setattr(hermes_cli.loops, 'migrate_loop_to_session', lambda *a, **k: None)
    parent = SimpleNamespace(session_id='parent', model='big', provider='p', _cached_system_prompt='P')
    parent.switch_model = lambda **kw: setattr(parent, 'model', kw['new_model'])
    routing.set_routing('parent', True)
    assert routing.apply_route(parent, {'model': 'small'})

    child = SimpleNamespace(session_id='child')
    _carry_session_state_to_child(child, 'parent', None)

    assert routing.routing_enabled('child')
    assert json.loads(routing._path('child').read_text())['automatic_route']['model'] == 'small'


def test_restore_cross_provider_route_is_exact_and_allowlisted(monkeypatch, tmp_path):
    from agent import turn_routing as routing
    import hermes_cli.runtime_provider as runtime_provider

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(routing, '_config', lambda: {
        'enabled': True, 'models': {'p': ['small'], 'other': ['other-small']},
    })
    monkeypatch.setattr(runtime_provider, 'resolve_runtime_provider', lambda **kwargs: {
        'provider': 'p', 'model': 'small', 'api_key': 'secret',
        'base_url': 'https://p.example/v1', 'api_mode': 'chat_completions',
        'capabilities': {},
    })
    assert routing._write_state('s', {
        'enabled': True, 'automatic_route': {'model': 'small', 'provider': 'p'},
    })
    agent = SimpleNamespace(
        session_id='s', model='other-small', provider='other', _cached_system_prompt='P')
    calls = []

    def switch(**kwargs):
        calls.append(kwargs)
        agent.provider = kwargs['new_provider']
        agent.model = kwargs['new_model']
        agent._cached_system_prompt = None

    agent.switch_model = switch

    assert routing.restore_automatic_route(agent)
    assert agent.provider == 'p' and agent.model == 'small'
    assert calls[0]['provenance'] == 'automatic_restore'
    assert agent._cached_system_prompt == 'P'


def test_cross_provider_directive_resolves_exact_runtime_and_persists_no_credentials(
    monkeypatch, tmp_path,
):
    from agent import turn_routing as routing
    import hermes_cli.runtime_provider as runtime_provider

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(routing, '_config', lambda: {
        'enabled': True,
        'models': {'openai-codex': ['astra'], 'claude-sub': ['opus']},
    })
    monkeypatch.setattr(runtime_provider, 'resolve_runtime_provider', lambda **kwargs: {
        'provider': 'claude-sub', 'model': 'opus', 'api_key': 'SECRET',
        'base_url': 'process://claude-sub', 'api_mode': 'chat_completions',
        'capabilities': {'vision': True},
    })
    calls = []
    agent = SimpleNamespace(
        session_id='cross', model='astra', provider='openai-codex',
        _cached_system_prompt='OLD_PREFIX')

    def switch(**kwargs):
        calls.append(kwargs)
        agent.model = kwargs['new_model']
        agent.provider = kwargs['new_provider']
        agent._cached_system_prompt = None

    agent.switch_model = switch
    routing.set_routing('cross', True)

    assert routing.apply_route(
        agent, {'provider': 'claude-sub', 'model': 'opus'}, is_first_turn=False)
    assert calls == [{
        'new_model': 'opus', 'new_provider': 'claude-sub', 'api_key': 'SECRET',
        'base_url': 'process://claude-sub', 'api_mode': 'chat_completions',
        'capabilities': {'vision': True}, 'provenance': 'automatic_routing',
    }]
    assert agent._cached_system_prompt == 'OLD_PREFIX'
    state = json.loads(routing._path('cross').read_text())
    assert state == {
        'enabled': True,
        'automatic_route': {'provider': 'claude-sub', 'model': 'opus'},
    }
    assert 'SECRET' not in routing._path('cross').read_text()


def test_failed_state_write_never_breaks_regular_hook_context(monkeypatch, tmp_path):
    from agent import turn_routing as routing
    from agent.turn_context import _collect_pre_llm_call_context
    import hermes_cli.lifecycle

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(routing, '_config', lambda: {'enabled': True, 'models': {'p': ['small']}})
    monkeypatch.setattr(
        routing, 'atomic_json_write',
        lambda *a, **k: (_ for _ in ()).throw(OSError('read-only filesystem')),
    )
    monkeypatch.setattr(
        hermes_cli.lifecycle, 'invoke_hook', lambda *a, **kw: [{'context': 'unrelated hook'}])
    agent = SimpleNamespace(session_id='s', model='big', provider='p')

    assert not routing.set_routing('s', True)
    assert _collect_pre_llm_call_context(
        agent, effective_task_id='t', turn_id='turn', original_user_message='request',
        messages=[], conversation_history=[],
    ) == 'unrelated hook'


def test_manual_pin_invalidates_stale_auto_route_while_global_routing_is_off(
    monkeypatch, tmp_path
):
    from agent import turn_routing as routing

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    config = {'enabled': True, 'models': {'p': ['small']}}
    monkeypatch.setattr(routing, '_config', lambda: config)
    routing._write_state('s', {
        'enabled': True, 'automatic_route': {'model': 'small', 'provider': 'p'},
    })
    config['enabled'] = False

    assert routing.pin_model('s')
    config['enabled'] = True
    assert routing.snapshot_routing_state('s')['state'] == {'enabled': False}


@pytest.mark.parametrize('provider', ['named', 'custom:named'])
def test_named_custom_runtime_keeps_exact_identity(provider, monkeypatch, tmp_path):
    from agent import turn_routing as routing
    import hermes_cli.runtime_provider as rp
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(routing, '_config', lambda: {
        'enabled': True, 'models': {provider: ['small']}})
    monkeypatch.setattr(rp, 'load_config', lambda: {
        'providers': {'named': {'base_url': 'https://named.example/v1', 'api_key': 'test'}}})
    monkeypatch.setattr(rp, '_try_resolve_from_custom_pool', lambda *a, **kw: None)
    agent = SimpleNamespace(session_id='named', provider='old', model='big')
    calls = []
    agent.switch_model = lambda **kw: calls.append(kw)
    assert routing.apply_route(agent, {'provider': provider, 'model': 'small'}, is_first_turn=True)
    assert calls[0]['new_provider'] == provider
    assert calls[0]['base_url'] == 'https://named.example/v1'


@pytest.mark.parametrize('restore', [False, True])
def test_switch_rebinds_image_cost_only_after_success(restore, monkeypatch, tmp_path):
    from agent import turn_routing as routing, image_token_cost as images
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(routing, '_config', lambda: {
        'enabled': True, 'models': {'p': ['small', 'big']}})
    monkeypatch.setattr(images, 'learned_image_token_cost', lambda model, url: 111 if model == 'big' else 222)
    agent = SimpleNamespace(session_id='images', model='big', provider='p', base_url='https://p.example')
    def switch(**kw):
        agent.model = kw['new_model']
    agent.switch_model = switch
    routing._write_state('images', {'enabled': True, 'automatic_route': {'provider': 'p', 'model': 'small'}})
    with images.image_cost_context(None):
        images.bind_image_token_cost(agent)
        if restore:
            assert routing.restore_automatic_route(agent)
        else:
            assert routing.apply_route(agent, {'model': 'small'})
        assert images.current_image_token_cost() == 222
        agent.switch_model = lambda **kw: (_ for _ in ()).throw(RuntimeError('failed'))
        assert not routing.apply_route(agent, {'model': 'big'})
        assert images.current_image_token_cost() == 222


@pytest.mark.parametrize('directive', [None, {}, {'model':'small','provider':'evil'}, {'model':123}])
def test_bad_routes_fail_open(directive, monkeypatch, tmp_path):
    from agent import turn_routing as routing
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(routing, '_config', lambda: {'enabled':True,'models':{'p':['small']}})
    agent = SimpleNamespace(session_id='s',model='big',provider='p')
    assert not routing.apply_route(agent,directive,is_first_turn=True)


@pytest.mark.parametrize('prior_pin', [False, True])
def test_failed_pin_survives_recreation_restore_compression_and_profile_scope(
    monkeypatch, tmp_path, caplog, prior_pin,
):
    from agent import turn_routing as routing
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'one'))
    monkeypatch.setattr(routing, '_config', lambda: {
        'enabled': True, 'models': {'p': ['small', 'big']}})
    assert routing._write_state('s', {
        'enabled': True, 'automatic_route': {'model': 'small', 'provider': 'p'}})
    def fail(*args, **kwargs):
        raise OSError('SECRET storage detail')
    with monkeypatch.context() as broken:
        broken.setattr(routing, 'atomic_json_write', fail)
        if prior_pin:
            assert not routing.pin_model('s')
        snapshot = routing.snapshot_routing_state('s')
        assert not routing.pin_model('s')
        recreated = SimpleNamespace(session_id='s', model='big', provider='p')
        recreated.switch_model = lambda **kw: pytest.fail('manual model was overwritten')
        assert not routing.restore_automatic_route(recreated)
        assert not routing.apply_route(recreated, {'model': 'small'})
        assert recreated.model == 'big'
        routing.migrate_routing_state('s', 'child')
        assert not routing.routing_enabled('child', is_first_turn=True)
        routing.restore_routing_state('s', snapshot)
        assert routing.routing_enabled('s') is (not prior_pin)
        assert 'SECRET' not in caplog.text
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'two'))
    assert routing.set_routing('s', True)
    assert routing.routing_enabled('s')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'one'))
    assert not routing.routing_enabled('child')
    assert routing.set_routing('child', True)
    assert routing.routing_enabled('child')
    assert 'automatic_route' not in routing.snapshot_routing_state('child')['state']
