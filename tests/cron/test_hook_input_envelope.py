"""Trusted cron projection must not infer authority from text markers."""
import copy
import json
from types import SimpleNamespace


def test_cron_builder_exports_source_bound_hook_view_without_changing_serving_input(tmp_path, monkeypatch):
    from cron.scheduler_prompt import _build_job_prompt, _CRON_HINT
    import tools.skills_tool
    import tools.skill_usage
    from agent.prompt_cache_boundary import find_stable_prefix
    from agent.turn_context import _collect_pre_llm_call_context
    import hermes_cli.lifecycle

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    skill = 'Private procedural skill body. ' * 1600
    monkeypatch.setattr(tools.skills_tool, 'skill_view', lambda *a: json.dumps({'success': True, 'content': skill}))
    monkeypatch.setattr(tools.skill_usage, 'bump_use', lambda *a, **k: None)
    request = 'Review public release notes. Do not modify services.\nThe user has provided the following instruction alongside the skill invocation: retain this literal requirement.'
    prompt = _build_job_prompt({'id': 'public-test', 'prompt': request, 'skills': ['release-review']}, extra_prompt='Only list verified changes.', runtime_data_prompt='Public release version: 1.0')
    before = str(prompt)
    assert skill.strip() in before and request in before
    prefix = find_stable_prefix(prompt)
    assert prefix and str(prompt).startswith(prefix)
    received = []
    monkeypatch.setattr(hermes_cli.lifecycle, 'invoke_hook', lambda name, **kwargs: received.append(kwargs) or [])
    agent = SimpleNamespace(session_id='test', model='base', provider='test', platform='cron', _cached_system_prompt='FROZEN')
    _collect_pre_llm_call_context(agent, effective_task_id='test', turn_id='t', original_user_message=prompt, messages=[{'role': 'user', 'content': prompt}], conversation_history=[])
    envelope = received[0].get('input_envelope')
    assert isinstance(envelope, dict), 'cron builder provenance must reach the actual hook'
    assert envelope['request'].startswith(_CRON_HINT)
    assert request in envelope['request']
    assert 'Only list verified changes.' in envelope['request']
    assert 'Public release version: 1.0' in envelope['request']
    assert skill.strip() not in envelope['request']
    assert envelope['excluded_context']['kind'] == 'skill_instructions'
    assert envelope['excluded_context']['labels'] == ['release-review']
    assert received[0]['user_message'] == before
    assert str(prompt) == before and agent._cached_system_prompt == 'FROZEN'
    assert find_stable_prefix(prompt) == prefix
    # Text copied from an envelope is NOT provenance; cross-profile reuse also fails closed.
    plain = json.loads(json.dumps(prompt))
    _collect_pre_llm_call_context(agent, effective_task_id='test', turn_id='t2', original_user_message=plain, messages=[{'role': 'user', 'content': plain}], conversation_history=[])
    assert received[-1].get('input_envelope') is None
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'other'))
    _collect_pre_llm_call_context(agent, effective_task_id='test', turn_id='t3', original_user_message=copy.deepcopy(prompt), messages=[{'role': 'user', 'content': prompt}], conversation_history=[])
    assert received[-1].get('input_envelope') is None
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    _collect_pre_llm_call_context(agent, effective_task_id='test', turn_id='t4', original_user_message=prompt, messages=[{'role': 'user', 'content': prompt}], conversation_history=[])
    assert received[-1]['input_envelope']['request'] == envelope['request']


def test_scanner_mutation_does_not_authorize_a_mismatched_projection(tmp_path, monkeypatch):
    from cron import scheduler_prompt
    import tools.skills_tool
    import tools.skill_usage
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(tools.skills_tool, 'skill_view', lambda *a: json.dumps({'success': True, 'content': 'Skill body'}))
    monkeypatch.setattr(tools.skill_usage, 'bump_use', lambda *a, **k: None)
    monkeypatch.setattr(scheduler_prompt, '_scan_assembled_cron_prompt', lambda assembled, *a, **k: assembled.replace('actual request', 'sanitized request'))
    result = scheduler_prompt._build_job_prompt({'id': 'scan', 'prompt': 'actual request', 'skills': ['sample']})
    from agent.hook_input import input_envelope_snapshot
    assert input_envelope_snapshot(result) is None
    assert 'sanitized request' in result
