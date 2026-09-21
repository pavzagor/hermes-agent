"""Independent effort directives and explicit provenance at turn boundaries."""
from types import SimpleNamespace
import pytest


@pytest.fixture
def policy(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    from agent import turn_reasoning as r
    cfg = {'reasoning_routing': {'enabled': True}, 'agent': {'reasoning_effort': 'high'}}
    monkeypatch.setattr(r, '_config', lambda: cfg)
    return r, cfg


def agent(**kw):
    return SimpleNamespace(**dict(dict(session_id='s', model='gpt-6-astra-900k', provider='openai-codex',
        reasoning_config={'enabled': True, 'effort': 'high'}, reasoning_provenance='config'), **kw))


def test_default_off_and_strict_validation(policy):
    r, cfg = policy
    a = agent()
    cfg['reasoning_routing']['enabled'] = False
    assert not r.apply_reasoning(a, {'effort': 'low'})
    cfg['reasoning_routing']['enabled'] = True
    for bad in [None, {'effort': 'none'}, {'effort': 'low', 'model': 'x'}, {'effort': []}]:
        assert not r.apply_reasoning(a, bad)
    for model, provider in [('unknown', 'openai-codex'), ('gpt-6-astra', 'custom')]:
        assert not r.apply_reasoning(agent(model=model, provider=provider), {'effort': 'low'})
    assert r.apply_reasoning(a, {'effort': 'low'})
    assert a.reasoning_provenance == 'automatic'


def test_model_pin_does_not_pin_effort_and_routing_keeps_effort_pin(policy):
    r, _ = policy
    from agent import turn_routing
    turn_routing.pin_model('s')
    a = agent()
    assert r.apply_reasoning(a, {'effort': 'low'})
    r.set_session_reasoning('s', {'enabled': True, 'effort': 'high'})
    turn_routing.set_routing('s', True)
    turn_routing.pin_model('s')
    r.prepare_turn(a)
    assert r.reasoning_pinned(a)
    assert not r.apply_reasoning(a, {'effort': 'low'})
    assert a.reasoning_config['effort'] == 'high'


def test_explicit_same_as_baseline_survives_rebind_and_cold_recreation(policy):
    r, _ = policy
    a = agent(reasoning_provenance='request')
    r.refresh_reasoning(a)
    assert a.reasoning_config['effort'] == 'high'
    assert not r.apply_reasoning(a, {'effort': 'low'})
    r.set_session_reasoning('s', {'enabled': True, 'effort': 'xhigh'})
    fresh = agent()
    r.prepare_turn(fresh)
    assert fresh.reasoning_config['effort'] == 'xhigh'
    r.set_session_reasoning('s', None)
    r.prepare_turn(fresh)
    assert fresh.reasoning_provenance == 'config'
    assert fresh.reasoning_config['effort'] == 'high'


def test_reset_and_per_model_override(policy):
    r, cfg = policy
    a = agent()
    assert r.apply_reasoning(a, {'effort': 'low'})
    r.prepare_turn(a)
    assert a.reasoning_config['effort'] == 'high'
    cfg['agent']['reasoning_overrides'] = {'gpt-6-astra-900k': 'xhigh'}
    r.prepare_turn(a)
    assert a.reasoning_config['effort'] == 'xhigh'
    assert not r.apply_reasoning(a, {'effort': 'low'})
    cfg['reasoning_routing']['override_models'] = {'openai-codex': ['gpt-6-astra-900k']}
    assert r.apply_reasoning(a, {'effort': 'low'})


def test_hook_flags_and_failure_reset(policy, monkeypatch):
    r, _ = policy
    from agent.turn_context import _collect_pre_llm_call_context
    import hermes_cli.lifecycle
    a = agent()
    seen = []
    def hook(name, **kw):
        seen.append(kw)
        return [{'reasoning': {'effort': 'low'}}]
    monkeypatch.setattr(hermes_cli.lifecycle, 'invoke_hook', hook)
    args = dict(effective_task_id='t', turn_id='t', original_user_message='hi', messages=[], conversation_history=[])
    _collect_pre_llm_call_context(a, **args)
    assert seen[0]['reasoning_routing_enabled'] is True
    assert seen[0]['reasoning_pinned'] is False
    assert a.reasoning_config['effort'] == 'low'
    monkeypatch.setattr(hermes_cli.lifecycle, 'invoke_hook', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError()))
    _collect_pre_llm_call_context(a, **args)
    assert a.reasoning_config['effort'] == 'high'


@pytest.mark.parametrize('model', ['gpt-6-astra', 'gpt-6-astra-900k', 'gpt-5.6-sol', 'gpt-5.6-sol-900k', 'gpt-5.6-luna', 'gpt-5.6-luna-900k'])
@pytest.mark.parametrize('effort', ['low', 'medium', 'high', 'xhigh'])
def test_verified_capabilities(policy, model, effort):
    r, _ = policy
    a = agent(model=model)
    assert r.apply_reasoning(a, {'effort': effort})
    assert a.reasoning_config['effort'] == effort


def test_raw_request_pin_and_compression(policy):
    r, _ = policy
    from agent.turn_routing import migrate_routing_state
    a = agent(request_overrides={'extra_body': {'reasoning': {'effort': 'high'}}})
    assert r.reasoning_pinned(a)
    assert not r.apply_reasoning(a, {'effort': 'low'})
    r.set_session_reasoning('s', {'enabled': False})
    assert migrate_routing_state('s', 'child')
    child = agent(session_id='child')
    r.prepare_turn(child)
    assert child.reasoning_config == {'enabled': False}
    assert child.reasoning_provenance == 'session'


def test_old_narrow_hook_signatures_remain_compatible():
    from hermes_cli.plugins_dispatch import PluginDispatchMixin
    assert PluginDispatchMixin._invoke_hook_callback(lambda user_message: user_message, {
        'user_message': 'hello', 'reasoning_routing_enabled': True, 'reasoning_pinned': False,
    }) == 'hello'
