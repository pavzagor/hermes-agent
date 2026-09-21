"""Real AIAgent -> Codex transport -> OpenAI SDK, with only HTTP/tool I/O fake."""
import json
from unittest.mock import patch

import httpx
import pytest
import yaml


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    from run_agent import AIAgent
    from agent import turn_reasoning
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    cfg = {'agent': {'reasoning_effort': 'high'}, 'reasoning_routing': {'enabled': True},
           'turn_routing': {'enabled': True, 'models': {'openai-codex': ['gpt-6-astra-900k', 'gpt-5.6-sol-900k']}}}
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump(cfg))
    bodies = []
    def respond(request):
        body = json.loads(request.content)
        bodies.append(body)
        # Two tool rounds per turn, then final answer; reset across turns.
        tool = len(bodies) % 3 != 0
        item = ({'type': 'function_call', 'id': f'fc_{len(bodies)}', 'call_id': f'call_{len(bodies)}',
                 'name': 'web_search', 'arguments': '{"query":"test"}', 'status': 'completed'} if tool else
                {'type': 'message', 'id': 'msg', 'role': 'assistant', 'status': 'completed',
                 'content': [{'type': 'output_text', 'text': 'done', 'annotations': []}]})
        response = {'id': f'resp_{len(bodies)}', 'object': 'response', 'created_at': 0,
                    'status': 'completed', 'model': body['model'], 'output': [item]}
        events = [{'type': 'response.output_item.done', 'output_index': 0, 'item': item},
                  {'type': 'response.completed', 'response': response}]
        return httpx.Response(200, headers={'Content-Type': 'text/event-stream'},
                              content=''.join(f'data: {json.dumps(e)}\n\n' for e in events) + 'data: [DONE]\n\n')
    monkeypatch.setattr(AIAgent, '_build_keepalive_http_client', staticmethod(
        lambda *a, **kw: httpx.Client(transport=httpx.MockTransport(respond))))
    monkeypatch.setattr('model_tools.get_tool_definitions', lambda **kw: [
        {'type': 'function', 'function': {'name': 'web_search', 'description': 'search',
         'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}}}}])
    monkeypatch.setattr('model_tools.check_toolset_requirements', lambda *a, **kw: {})
    monkeypatch.setattr('model_tools.handle_function_call', lambda *a, **kw: '{"ok":true}')
    def make(**kw):
        a = AIAgent(model='gpt-6-astra-900k', provider='openai-codex', api_key='test-offline',
                    base_url='https://chatgpt.com/backend-api/codex', api_mode='codex_responses',
                    skip_memory=True, skip_context_files=True, quiet_mode=True, max_iterations=5,
                    session_id='effort-loop', **kw)
        a._cached_system_prompt = 'FIXED PREFIX'
        a.compression_enabled = False
        a._cleanup_task_resources = lambda *a, **kw: None
        return a
    return make, bodies, cfg


def test_real_fallback_inside_tool_loop_preserves_request_effort(runtime, monkeypatch):
    make, bodies, _ = runtime
    from agent import auxiliary_client
    from run_agent import AIAgent
    monkeypatch.setenv('OPENAI_API_KEY', 'test-offline')
    monkeypatch.setattr(auxiliary_client, '_openai_http_client_kwargs',
                        lambda *a, **kw: {'http_client': AIAgent._build_keepalive_http_client()})
    a = make(reasoning_config={'enabled': True, 'effort': 'xhigh'}, fallback_model={
        'provider': 'custom', 'model': 'gpt-5.6-sol', 'api_key': 'test-offline',
        'base_url': 'https://api.openai.com/v1', 'api_mode': 'codex_responses'})
    activated = []
    def tool(*args, **kw):
        if not activated:
            activated.append(a._try_activate_fallback())
        return '{"ok":true}'
    monkeypatch.setattr('model_tools.handle_function_call', tool)
    monkeypatch.setattr('hermes_cli.lifecycle.invoke_hook', lambda name, **kw:
                        [{'reasoning': {'effort': 'low'}}] if name == 'pre_llm_call' else [])
    result = a.run_conversation('search twice')
    assert result['final_response'] == 'done'
    assert activated == [True]
    assert [b['model'] for b in bodies] == ['gpt-6-astra', 'gpt-5.6-sol', 'gpt-5.6-sol']
    assert all(b['reasoning']['effort'] == 'xhigh' for b in bodies)
    assert a.reasoning_provenance == 'request'
    a.client.close()


@pytest.mark.parametrize('pin', [None, 'request', 'session'])
def test_every_tool_round_uses_selected_effort_and_wire_alias(runtime, monkeypatch, pin):
    make, bodies, cfg = runtime
    args = {} if pin is None else {'reasoning_config': {'enabled': True, 'effort': 'high'}, 'reasoning_provenance': pin}
    a = make(**args)
    calls = []
    def hook(name, **kw):
        if name == 'pre_llm_call':
            calls.append(kw)
            return [{'route': {'model': 'gpt-5.6-sol-900k'}, 'reasoning': {'effort': 'low'}}]
        return []
    monkeypatch.setattr('hermes_cli.lifecycle.invoke_hook', hook)
    result = a.run_conversation('search twice')
    assert result['final_response'] == 'done'
    assert len(calls) == 1
    assert len(bodies) == 3
    assert all(b['model'] == 'gpt-5.6-sol' for b in bodies)
    assert all(b['reasoning']['effort'] == ('low' if pin is None else 'high') for b in bodies)
    assert all(b['instructions'] == 'FIXED PREFIX' for b in bodies)
    assert len({b['prompt_cache_key'] for b in bodies}) == 1
    assert calls[0]['reasoning_pinned'] is bool(pin)
    a.client.close()


def test_next_turn_failure_cold_recreation_and_manual_fallback(runtime, monkeypatch):
    from agent.chat_completion_helpers import _reresolve_fallback_reasoning_config
    from agent.turn_reasoning import set_session_reasoning
    make, bodies, _ = runtime
    a = make()
    monkeypatch.setattr('hermes_cli.lifecycle.invoke_hook', lambda name, **kw:
                        [{'reasoning': {'effort': 'low'}}] if name == 'pre_llm_call' else [])
    first = a.run_conversation('search twice')
    assert len(bodies) == 3 and all(b['reasoning']['effort'] == 'low' for b in bodies)
    monkeypatch.setattr('hermes_cli.lifecycle.invoke_hook', lambda *a, **kw: [])
    a.run_conversation('again', conversation_history=first['messages'])
    assert len(bodies) == 6 and all(b['reasoning']['effort'] == 'high' for b in bodies[3:])
    set_session_reasoning(a.session_id, {'enabled': True, 'effort': 'xhigh'})
    fresh = make()
    fresh.run_conversation('resumed', conversation_history=first['messages'])
    assert len(bodies) == 9 and all(b['reasoning']['effort'] == 'xhigh' for b in bodies[6:])
    fresh.model = 'gpt-5.6-luna-900k'
    _reresolve_fallback_reasoning_config(fresh)
    assert fresh.reasoning_config['effort'] == 'xhigh'
    a.client.close()
    fresh.client.close()


def test_config_read_failure_resets_automatic_effort_on_every_wire_round(runtime, monkeypatch):
    from agent import turn_reasoning
    make, bodies, _ = runtime
    a = make()
    seen = []
    def hook(name, **kw):
        if name == 'pre_llm_call':
            seen.append(kw)
            return [{'reasoning': {'effort': 'low'}, 'context': 'unrelated hook context'}]
        return []
    monkeypatch.setattr('hermes_cli.lifecycle.invoke_hook', hook)
    first = a.run_conversation('search twice')
    assert all(b['reasoning']['effort'] == 'low' for b in bodies)
    def unavailable():
        raise OSError('config temporarily unreadable')
    monkeypatch.setattr(turn_reasoning, '_config', unavailable)
    result = a.run_conversation('again', conversation_history=first['messages'])
    assert result['final_response'] == 'done'
    assert len(seen) == 2 and seen[-1]['reasoning_routing_enabled'] is False
    assert len(bodies) == 6
    assert all(b['reasoning']['effort'] == 'high' for b in bodies[3:])
    assert all(b['instructions'] == 'FIXED PREFIX' for b in bodies)
    assert a.reasoning_provenance == 'config'
    a.client.close()


@pytest.mark.parametrize('options, expected', [({}, 'low'), ({'reasoning': {'effort': 'high'}}, 'high'),
                                             ({'reasoning_effort': 'xhigh'}, 'xhigh')])
def test_api_constructor_request_and_model_pins_are_independent(runtime, monkeypatch, options, expected):
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.config import PlatformConfig
    import gateway.run as gateway
    make, bodies, cfg = runtime
    monkeypatch.setattr(gateway, '_resolve_runtime_agent_kwargs', lambda: {
        'provider': 'openai-codex', 'api_mode': 'codex_responses', 'api_key': 'test-offline',
        'base_url': 'https://chatgpt.com/backend-api/codex'})
    monkeypatch.setattr(gateway, '_resolve_gateway_model', lambda *a: 'gpt-6-astra-900k')
    monkeypatch.setattr(gateway, '_load_gateway_config', lambda: cfg)
    monkeypatch.setattr(gateway, '_current_max_iterations', lambda: 5)
    monkeypatch.setattr(gateway.GatewayRunner, '_load_fallback_model', lambda: None)
    adapter = APIServerAdapter(PlatformConfig())
    a = adapter._create_agent(session_id='api-effort', requested_model='gpt-6-astra-900k', model_options=options)
    a._cached_system_prompt = 'FIXED PREFIX'
    a.compression_enabled = False
    a._cleanup_task_resources = lambda *a, **kw: None
    monkeypatch.setattr('hermes_cli.lifecycle.invoke_hook', lambda name, **kw:
                        [{'reasoning': {'effort': 'low'}}] if name == 'pre_llm_call' else [])
    result = a.run_conversation('search twice')
    assert result['final_response'] == 'done'
    assert len(bodies) == 3
    assert all(b['model'] == 'gpt-6-astra' and b['reasoning']['effort'] == expected for b in bodies)
    a.client.close()
