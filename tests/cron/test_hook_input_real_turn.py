"""Exercise builder -> real turn setup -> discovered plugin, with egress forbidden.

No provider replies or Jev verdicts are manufactured. Stop before serving inference.
"""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import pytest


class StopBeforeInference(BaseException):
    pass


@pytest.mark.parametrize('text', ['Public instructions.', 'Public instructions. Привет. 😀'])
def test_cron_input_reaches_discovered_plugin_through_real_turn_setup(tmp_path, monkeypatch, text):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_DISABLE_LAZY_INSTALLS', '1')
    source = Path(__file__).resolve().parents[3] / 'candidate'
    shutil.copytree(source, tmp_path / 'plugins' / 'jev-router', ignore=shutil.ignore_patterns('.git', '__pycache__', '.pytest_cache'))
    config = {'plugins': {'enabled': ['jev-router'], 'entries': {'jev-router': {'settings': {'enabled': True, 'profile_home': str(tmp_path)}}}},
              'turn_routing': {'enabled': True, 'routes': {'luna': {'provider': 'openai', 'model': 'gpt-4o-mini'}}}}
    (tmp_path / 'config.yaml').write_text(json.dumps(config))
    skill = tmp_path / 'skills' / 'public-review' / 'SKILL.md'
    skill.parent.mkdir(parents=True)
    skill.write_text('---\nname: public-review\ndescription: Review public release notes.\n---\n' + text * 5000)
    from hermes_cli.plugins import get_plugin_manager
    manager = get_plugin_manager()
    manager.discover_and_load(force=True)
    hooks = [h for h in manager._hooks['pre_llm_call'] if h.__name__ == 'before_turn']
    assert len(hooks) == 1
    router = importlib.import_module(hooks[0].__module__ + '.router')
    payloads = []
    def forbid_classification(payload, wrapper):
        payloads.append(copy.deepcopy(payload))
        raise router.InferenceError('transport_unavailable')
    monkeypatch.setattr(router, 'classify', forbid_classification)
    monkeypatch.setattr(router, 'connect_key', lambda *a: pytest.fail('Offline test must not resolve credentials'))
    from cron.scheduler_prompt import _build_job_prompt
    request = 'Summarize public release notes. Read only. Never restart services. ' + text
    message = _build_job_prompt({'id': 'public', 'prompt': request, 'skills': ['public-review']}, extra_prompt='Keep [SILENT] exactly.')
    from agent.hook_input import input_envelope_snapshot
    assert input_envelope_snapshot(message) is not None
    original = str(message)
    from run_agent import AIAgent
    with patch('model_tools.get_tool_definitions', return_value=[]), patch('model_tools.check_toolset_requirements', return_value={}):
        agent = AIAgent(model='gpt-4o-mini', provider='openai', api_key='offline-constructor-only',
                        base_url='https://api.openai.com/v1', quiet_mode=True, skip_memory=True,
                        skip_context_files=True, enabled_toolsets=[], max_iterations=1, session_id='offline-cron')
    agent._cached_system_prompt = 'FROZEN PREFIX'
    from agent import turn_context
    collect = turn_context._collect_pre_llm_call_context
    captured = []
    def stop_after_collection(*args, **kwargs):
        assert input_envelope_snapshot(kwargs['original_user_message']) is not None, 'sanitization lost provenance'
        captured.append(copy.deepcopy(kwargs['messages']))
        collect(*args, **kwargs)
        raise StopBeforeInference()
    monkeypatch.setattr(turn_context, '_collect_pre_llm_call_context', stop_after_collection)
    with pytest.raises(StopBeforeInference):
        agent.run_conversation(message, conversation_history=[])
    assert len(payloads) == 1, 'The actual turn preparation must carry source-bound provenance to the plugin'
    assert request in payloads[0]['state']['request']
    assert 'Keep [SILENT] exactly.' in payloads[0]['state']['request']
    assert payloads[0]['state']['projection']['omitted_host_context']['kind'] == 'skill_instructions'
    assert captured[0][-1]['content'] == original
    assert str(message) == original and agent._cached_system_prompt == 'FROZEN PREFIX'
