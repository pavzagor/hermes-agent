"""Turn directives must preserve prompt prefixes and manual conversation pins."""
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
    assert len(calls) == 1 and calls[0]['automatic'] is True
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
        return '{"success":true,"content":"' + name * 40000 + '"}'
    monkeypatch.setattr(tools.skills_tool, 'skill_view', view)
    monkeypatch.setattr(hermes_cli.lifecycle, 'invoke_hook', lambda *a, **kw: [{'skills':['one','two','one']}])
    agent = SimpleNamespace(session_id='s',model='big',provider='p')
    context = _collect_pre_llm_call_context(agent,effective_task_id='t',turn_id='turn',original_user_message='request',messages=[],conversation_history=[])
    assert loaded == ['one','two']
    assert 'one'*40000 in context and 'two'*40000 in context


@pytest.mark.parametrize('directive', [None, {}, {'model':'small','provider':'evil'}, {'model':123}])
def test_bad_routes_fail_open(directive, monkeypatch, tmp_path):
    from agent import turn_routing as routing
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(routing, '_config', lambda: {'enabled':True,'models':{'p':['small']}})
    agent = SimpleNamespace(session_id='s',model='big',provider='p')
    assert not routing.apply_route(agent,directive,is_first_turn=True)
