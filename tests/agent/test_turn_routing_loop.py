"""Exercise the real AIAgent loop and runtime switch (only remote I/O is fake)."""
import json
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch


def response(tool=False):
    calls = [NS(id='call-1',type='function',function=NS(name='web_search',arguments='{"query":"x"}'))] if tool else None
    return NS(choices=[NS(message=NS(content=None if tool else 'done',tool_calls=calls,reasoning=None,reasoning_content=None),finish_reason='tool_calls' if tool else 'stop')], model='gpt-4o-mini',usage=None)


def test_real_switch_once_across_tool_loop_and_manual_pin(tmp_path, monkeypatch):
    from run_agent import AIAgent
    from agent import turn_routing
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / 'state.db')
    monkeypatch.setattr(turn_routing,'_config',lambda:{'enabled':True,'models':{'openai':['gpt-4o','gpt-4o-mini']}})
    schema=[{'type':'function','function':{'name':'web_search','description':'search','parameters':{'type':'object','properties':{'query':{'type':'string'}}}}}]
    with patch('model_tools.get_tool_definitions',return_value=schema), patch('model_tools.check_toolset_requirements',return_value={}), patch('agent.process_bootstrap.OpenAI'):
        agent=AIAgent(model='gpt-4o',provider='openai',api_key='synthetic-test',base_url='https://api.openai.com/v1',quiet_mode=True,skip_memory=True,skip_context_files=True,max_iterations=5,session_id='route-loop',session_db=db)
    agent._cached_system_prompt='FIXED SYSTEM PREFIX'
    agent._disable_streaming=True
    agent.compression_enabled=False
    agent.save_trajectories=False
    client=MagicMock()
    client.chat.completions.create.side_effect=[response(True),response()]
    agent.client=client
    hook_calls=[]
    def hook(name, **kw):
        if name == 'pre_llm_call':
            hook_calls.append(kw)
            return [{'route':{'model':'gpt-4o-mini'}}]
        return []
    with patch('hermes_cli.lifecycle.invoke_hook',side_effect=hook), patch.object(agent,'_create_openai_client',return_value=client), patch('model_tools.handle_function_call',return_value=json.dumps({'ok':True})), patch.object(agent,'_cleanup_task_resources'):
        result=agent.run_conversation('search',conversation_history=[])
        assert result['final_response']=='done'
        assert len(hook_calls)==1
        assert client.chat.completions.create.call_count==2
        assert all(c.kwargs['model']=='gpt-4o-mini' for c in client.chat.completions.create.call_args_list)
        assert agent._cached_system_prompt=='FIXED SYSTEM PREFIX'
        row = db.get_session('route-loop')
        assert row is not None and row['system_prompt'] == 'FIXED SYSTEM PREFIX'
        # A recreated API agent starts without an in-memory prompt. Exercise the
        # exact restore path and prove it does not rebuild the prefix.
        from agent.conversation_loop import _restore_or_build_system_prompt
        agent._cached_system_prompt = None
        with patch.object(agent, '_build_system_prompt', side_effect=AssertionError('unexpected rebuild')):
            _restore_or_build_system_prompt(agent, None, result['messages'])
        assert agent._cached_system_prompt == 'FIXED SYSTEM PREFIX'
        agent.switch_model('gpt-4o-mini','openai')
        assert not turn_routing.routing_enabled('route-loop',is_first_turn=True)
        assert db.get_session('route-loop')['system_prompt'] is None
    db.close()
