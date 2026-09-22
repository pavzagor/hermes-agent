"""Restart regressions: durable history, real recovery and inbound preparation.

Only transport/model-side enrichment is stubbed; no live home or Slack I/O.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner, _prepare_resume_pending_message
from gateway.session import SessionSource, SessionStore
from hermes_state import SessionDB


@pytest.fixture
def lane(tmp_path):
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    if store._db is not None:
        store._db.close()
    store._db = SessionDB(db_path=tmp_path / "state.db")
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="group",
                           user_id="U1", user_name="Pavel", thread_id="100.1", scope_id="T1")
    entry = store.get_or_create_session(source)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = store
    runner._enrich_inbound_images = AsyncMock(return_value="enriched image")
    runner._expand_inbound_context_references = AsyncMock(side_effect=lambda s, k, text: text)
    yield runner, store, entry, source
    store._db.close()


def append(store, entry, role, content, **kwargs):
    store.append_to_transcript(entry.session_id, {"role": role, "content": content, **kwargs})


@pytest.mark.asyncio
async def test_completed_recent_turn_not_rearmed_by_unclean_recovery(lane):
    runner, store, entry, source = lane
    token = store.mark_turn_active(entry.session_key)
    append(store, entry, "user", "old image complaint", message_id="100.1")
    append(store, entry, "assistant", "The answer is complete.")
    store.clear_turn_active(entry.session_key, token)
    # Reload durable routing before taking the actual boot recovery path.
    store._loaded = False
    assert await runner._recover_unclean_sessions() == (0, 0)
    assert not store._entries[entry.session_key].resume_pending


@pytest.mark.asyncio
@pytest.mark.parametrize("internal", [False, True])
async def test_unclean_recovery_preserves_exact_unfinished_turn(lane, internal):
    runner, store, entry, source = lane
    append(store, entry, "user", "old request")
    append(store, entry, "assistant", "Earlier final")
    # A new user/delegation turn may die before writing its first transcript row.
    event = MessageEvent(text="delegation result" if internal else "new request",
                         source=source, internal=internal)
    assert await runner._mark_durable_active_turn(event, entry.session_key)
    assert await runner._recover_unclean_sessions() == (1, 0)
    assert store._entries[entry.session_key].resume_pending


@pytest.mark.asyncio
async def test_legacy_unfinished_tool_tail_remains_recoverable(lane):
    runner, store, entry, source = lane
    append(store, entry, "user", "work")
    append(store, entry, "assistant", None, tool_calls=[{"id": "call1", "type": "function",
           "function": {"name": "work", "arguments": "{}"}}])
    append(store, entry, "tool", "result", tool_call_id="call1")
    assert await runner._recover_unclean_sessions() == (0, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True])
async def test_same_completed_platform_id_dropped_before_image_enrichment(lane, queued):
    runner, store, entry, source = lane
    append(store, entry, "user", "Double delivery bullshit again", message_id="100.1")
    append(store, entry, "assistant", "Completed answer")
    history = store.load_transcript(entry.session_id)
    if queued:
        # The real agent-history builder removes platform IDs. Queued drains
        # must consult the durable transcript, not just their in-memory result.
        from gateway.run import _build_gateway_agent_history
        history, _ = _build_gateway_agent_history(history)
        history += [{"role": "user", "content": "new request"},
                    {"role": "assistant", "content": "new final"}]
    event = MessageEvent(text="Double delivery bullshit again", source=source,
                         message_id="100.1", message_type=MessageType.PHOTO,
                         media_urls=["redownloaded.jpeg"], media_types=["image/jpeg"])
    if queued:
        runner._run_agent = AsyncMock(side_effect=AssertionError("replayed model turn"))
        runner._run_agent_deliver_first_response = AsyncMock()
        ctx = SimpleNamespace(source=source, session_id=entry.session_id,
                              session_key=entry.session_key, run_generation=1,
                              _interrupt_depth=0, history=[], _status_thread_metadata=None)
        previous = {"messages": history, "final_response": "new final"}
        result = await runner._run_agent_queued_followup(
            ctx, None, event.text, event, previous, previous, None)
        assert result is previous
        runner._run_agent.assert_not_awaited()
    else:
        result = await runner._prepare_profile_scoped_inbound_message_text(
            event=event, source=source, history=history, session_key=entry.session_key)
        assert result is None
    runner._enrich_inbound_images.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unfinished", "new-id", "internal"])
async def test_replay_guard_preserves_unfinished_new_and_internal_inputs(lane, kind):
    runner, store, entry, source = lane
    append(store, entry, "user", "work", message_id="100.1")
    if kind != "unfinished":
        append(store, entry, "assistant", "Complete")
    event = MessageEvent(text="work", source=source,
                         message_id="100.2" if kind == "new-id" else "100.1",
                         internal=kind == "internal")
    assert await runner._prepare_profile_scoped_inbound_message_text(
        event=event, source=source, history=store.load_transcript(entry.session_id),
        session_key=entry.session_key) is not None


@pytest.mark.asyncio
async def test_empty_recovery_never_becomes_attributed_new_user_text(lane):
    runner, store, entry, source = lane
    store.mark_resume_pending(entry.session_key, "restart_interrupted")
    event = MessageEvent(text="", source=source, internal=True)
    text = await runner._prepare_profile_scoped_inbound_message_text(
        event=event, source=source, history=[], session_key=entry.session_key)
    assert text == ""
    note, persisted = _prepare_resume_pending_message(entry.resume_reason, text)
    assert "NEW message" not in note
    assert "CONTINUE the interrupted task" in note
    assert "do NOT ask the user to resend" in note
    assert persisted == note


@pytest.mark.asyncio
async def test_empty_internal_event_without_recovery_work_is_dropped(lane):
    runner, store, entry, source = lane
    event = MessageEvent(text="", source=source, internal=True)
    assert await runner._prepare_profile_scoped_inbound_message_text(
        event=event, source=source, history=[], session_key=entry.session_key) is None


@pytest.mark.asyncio
async def test_boot_scheduler_to_turn_preparation_keeps_recovery_internal(lane, monkeypatch):
    import asyncio
    from gateway.run_turn_runner import TurnRunner

    runner, store, entry, source = lane
    store.mark_turn_active(entry.session_key)
    assert await runner._recover_unclean_sessions() == (1, 0)
    captured = []

    async def handle(event):
        assert event.internal and event.text == "" and not event.media_urls
        history = store.load_transcript(entry.session_id)
        text = await runner._prepare_profile_scoped_inbound_message_text(
            event=event, source=event.source, history=history, session_key=entry.session_key)
        ctx = SimpleNamespace(message=text, persist_user_message=None, persist_user_timestamp=None,
                              history=history, session_key=entry.session_key, source=event.source)
        persisted, _ = TurnRunner(runner, ctx)._prepare_turn_message(history)
        captured.append((ctx.message, persisted))

    adapter = SimpleNamespace(handle_message=handle, _session_tasks={}, interactive_resume=True)
    runner._delivery_adapter_for = lambda source: adapter
    runner._is_user_authorized_for_source = lambda source: True
    runner._persist_active_agents = lambda: None
    runner._background_tasks = set()
    runner._startup_restore_in_progress = True
    monkeypatch.setattr("gateway.restart_loop_guard.check_and_record", lambda *a, **kw: False)
    assert runner._schedule_resume_pending_sessions() == 1
    await asyncio.gather(*runner._startup_restore_tasks)
    assert len(captured) == 1
    note, persisted = captured[0]
    assert "NEW message" not in note
    assert "CONTINUE the interrupted task" in note
    assert "do NOT ask the user to resend" in note
    assert persisted == note


async def assert_replay_admission(lane, history, queued, completed):
    """Exercise both entry paths; only model execution and transport are stubbed."""
    runner, store, entry, source = lane
    event = MessageEvent(text="work", source=source, message_id="100.1",
                         message_type=MessageType.PHOTO,
                         media_urls=["redownloaded.jpeg"], media_types=["image/jpeg"])
    if queued:
        from gateway.run import _build_gateway_agent_history
        history, _ = _build_gateway_agent_history(history)
        runner._delivery_adapter_for = lambda source: None
        runner._intake_adapter_for = lambda source: None
        runner._refresh_agent_cache_message_count = AsyncMock()
        runner._run_agent = AsyncMock(return_value={"messages": history, "final_response": "resumed"})
        runner._run_agent_deliver_first_response = AsyncMock()
        ctx = SimpleNamespace(source=source, session_id=entry.session_id,
                              session_key=entry.session_key, run_generation=1,
                              _interrupt_depth=0, history=[], _status_thread_metadata=None,
                              context_prompt="")
        previous = {"messages": history, "final_response": "previous"}
        result = await runner._run_agent_queued_followup(
            ctx, None, event.text, event, previous, previous, None)
        if completed:
            assert result is previous
            runner._run_agent.assert_not_awaited()
        else:
            runner._run_agent.assert_awaited_once()
            assert runner._run_agent.call_args.kwargs["inbound_message_id"] == "100.1"
    else:
        result = await runner._prepare_profile_scoped_inbound_message_text(
            event=event, source=source, history=history, session_key=entry.session_key)
        assert (result is None) is completed
    assert runner._enrich_inbound_images.await_count == (0 if completed else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("gate", ["verification_required", "verify_hook_continue"])
@pytest.mark.parametrize("stage", ["candidate", "nudge", "tool-tail", "final"])
async def test_stop_gate_attempt_stays_open_until_true_final(lane, monkeypatch, queued, gate, stage):
    from agent import turn_stop_gates
    from agent.session_persistence import SessionPersistenceMixin

    runner, store, entry, source = lane
    # Run the real producer AND agent persistence: candidates are durable, the
    # flagged nudge is in-memory only, and finish_reason survives SessionStore.
    agent = SessionPersistenceMixin()
    agent._session_db = store._db
    agent._session_db_created = True
    agent.session_id = entry.session_id
    agent._emit_interim_assistant_message = lambda msg: None
    agent._interim_content_was_streamed = lambda text: False
    monkeypatch.setattr(turn_stop_gates, "_verify_on_stop_nudge",
                        lambda agent: "verify" if gate == "verification_required" else None)
    monkeypatch.setattr(turn_stop_gates, "_pre_verify_nudge", lambda *args: "verify")
    messages = [{"role": "user", "content": "work", "platform_message_id": "100.1"}]
    verdict = turn_stop_gates.apply_stop_gates(
        agent, {"role": "assistant", "content": "candidate", "finish_reason": "stop"},
        final_response="candidate", messages=messages, conversation_history=[],
        pending_verification_response=None, pending_verification_response_previewed=False)
    assert verdict.continue_turn
    durable = store.load_transcript(entry.session_id)
    assert [row["role"] for row in durable] == ["user", "assistant"]
    assert durable[-1]["finish_reason"] == gate
    if stage == "candidate":
        history = durable
    else:
        if stage in {"tool-tail", "final"}:
            messages += [
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "verify1", "type": "function",
                     "function": {"name": "terminal", "arguments": "{}"}}]},
                {"role": "tool", "content": "verified", "tool_call_id": "verify1"},
            ]
        if stage == "final":
            messages.append({"role": "assistant", "content": "true final", "finish_reason": "stop"})
        assert agent._flush_messages_to_session_db(messages, []) is True
        history = messages
    from gateway.session_transcript import transcript_turn_completed
    # Check the live history as well: the durable fallback must not hide a
    # synthetic nudge incorrectly severing the attempt before its true final.
    assert transcript_turn_completed(history, "100.1") is (stage == "final")
    assert await runner._recover_unclean_sessions() == ((0, 0) if stage == "final" else (0, 1))
    await assert_replay_admission(lane, history, queued, completed=stage == "final")


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("latest_completed", [False, True])
async def test_repeated_platform_id_uses_latest_attempt(lane, queued, latest_completed):
    runner, store, entry, source = lane
    for completed in (not latest_completed, latest_completed):
        append(store, entry, "user", "work", message_id="100.1")
        if completed:
            append(store, entry, "assistant", "final", finish_reason="stop")
        else:
            append(store, entry, "assistant", None, tool_calls=[
                {"id": "work1", "type": "function",
                 "function": {"name": "terminal", "arguments": "{}"}}])
            append(store, entry, "tool", "result", tool_call_id="work1")
    # A different human turn must neither close nor reopen the requested attempt.
    append(store, entry, "user", "other work", message_id="100.2")
    append(store, entry, "assistant", "other final")
    await assert_replay_admission(lane, store.load_transcript(entry.session_id),
                                 queued, completed=latest_completed)
