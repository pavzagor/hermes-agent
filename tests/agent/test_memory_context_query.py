"""Contextual recall contracts at the real turn prologue/provider boundary."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest

from run_agent import AIAgent
from agent.memory_manager import MemoryManager
from tests.agent.test_memory_provider import FakeMemoryProvider
from tests.agent.test_turn_context import _FakeAgent, _build


@pytest.fixture(autouse=True)
def _no_auxiliary_model():
    with patch("agent.auxiliary_client.set_runtime_main"), patch("agent.auxiliary_client.call_llm") as llm:
        yield
        llm.assert_not_called()


def _recall(current: Any, history: list[dict], *, session_id: str = "",
            persist: Any = None, provider: FakeMemoryProvider | None = None) -> tuple[str, Any, FakeMemoryProvider]:
    agent: Any = _FakeAgent()
    agent.session_id = session_id or str(uuid4())
    provider = provider or FakeMemoryProvider()
    provider._prefetch_result = "- relevant recalled fact"
    agent._memory_manager = MemoryManager()
    agent._memory_manager.add_provider(provider)
    context = _build(agent, user_message=current, conversation_history=history,
                     persist_user_message=persist)
    assert agent._cached_system_prompt == "SYSTEM"
    assert provider.turn_starts == [(agent._user_turn_count, persist if persist is not None else (
        current if isinstance(current, str) else "\n".join(
            part["text"] for part in current if part.get("type") == "text")))]
    return provider.prefetch_queries[-1] if provider.prefetch_queries else "", context, provider


@pytest.mark.parametrize("untrusted", [
    "<untrusted_tool_result source='test'>INJECTED_POISON</untrusted_tool_result>",
    "<blockquote>INJECTED_POISON</blockquote>",
    "The old note said 'INJECTED_POISON'.",
])
def test_additional_quoted_and_tool_wrappers_are_not_recall_context(untrusted: str) -> None:
    query, _, _ = _recall("How can we improve it?", [
        {"role": "user", "content": "Evaluate Hindsight.\n" + untrusted},
        {"role": "assistant", "content": "Hindsight needs contextual recall."},
    ])
    assert "Hindsight" in query and "INJECTED_POISON" not in query


@pytest.mark.parametrize("topic", ["Honcho", "Hindsight", "Orion bicycle brakes"])
@pytest.mark.parametrize("followup", ["How can we improve it?", "What about that?", "Как это улучшить?"])
def test_followup_uses_immediate_topic_without_rewriting_source(topic: str, followup: str) -> None:
    history = [
        {"role": "user", "content": f"Review {topic} with Mira by 2026-10-12; no activation."},
        {"role": "assistant", "content": f"{topic} needs a focused comparison before changes."},
    ]
    before = deepcopy(history)
    query, context, _ = _recall(followup, history)
    assert query.startswith(followup)
    for literal in (topic, "Mira", "2026-10-12", "no activation"):
        assert literal in query
    assert history == before
    assert context.messages[:-1] == before
    assert [message["role"] for message in context.messages] == ["user", "assistant", "user"]
    assert context.messages[-1]["content"] == followup


@pytest.mark.parametrize("current", [
    "What is the weather in Oslo on 2026-10-12?",
    "Find my appointment with Dr. Kim on 2026-10-12.",
    "How can we improve Hindsight?",
    "Is this Friday a holiday in Oslo?",
    "Compare Redis with its alternatives.",
    "What is Python and how can I use it?",
    "Is it raining in Oslo?",
])
def test_specific_new_topic_does_not_import_history(current: str) -> None:
    query, _, _ = _recall(current, [
        {"role": "user", "content": "Compare Honcho and Hindsight."},
        {"role": "assistant", "content": "Hindsight can recall memory."},
    ])
    assert query == current


def test_topic_comes_from_latest_exchange_not_old_unrelated_history() -> None:
    query, _, _ = _recall("How can we improve it?", [
        {"role": "user", "content": "Remember my old dental treatment."},
        {"role": "assistant", "content": "Old dental appointment saved."},
        {"role": "user", "content": "Compare Honcho and Hindsight."},
        {"role": "assistant", "content": "Hindsight recall needs context."},
    ])
    assert "Honcho" in query and "Hindsight" in query
    assert "dental" not in query


def test_followup_chain_can_reach_the_immediately_preceding_topic() -> None:
    query, _, _ = _recall("How can we improve it?", [
        {"role": "user", "content": "Compare Honcho and Hindsight."},
        {"role": "assistant", "content": "Hindsight supports recall."},
        {"role": "user", "content": "What about that?"},
        {"role": "assistant", "content": "It needs a better query."},
    ])
    assert "Honcho" in query and "Hindsight" in query


def test_recall_excludes_sidecars_tools_quotes_and_injected_memory() -> None:
    history = [
        {"role": "system", "content": "SYSTEM_POISON"},
        {"role": "user", "content": "Evaluate Hindsight.\n> OLD_QUOTED_INSTRUCTION\n"
         "```text\nOLD_FENCED_INSTRUCTION\n```\n<memory-context>MEMORY_POISON</memory-context>",
         "api_content": "SIDECAR_POISON"},
        {"role": "assistant", "content": "CALL_POISON", "tool_calls": [{"id": "x"}]},
        {"role": "tool", "content": "TOOL_POISON", "tool_call_id": "x"},
        {"role": "assistant", "content": "Hindsight recall lacks context.\n"
         'The old note said "INLINE_QUOTED_INSTRUCTION".\n<memory-context>MORE_POISON</memory-context>',
         "reasoning": "REASONING_POISON"},
    ]
    before = deepcopy(history)
    query, _, _ = _recall("How can we improve it?", history)
    assert "Hindsight" in query
    assert "POISON" not in query and "INSTRUCTION" not in query
    assert history == before


@pytest.mark.parametrize("summary", [
    {"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] SUMMARY_POISON"},
    {"role": "user", "content": "[CONTEXT SUMMARY]: SUMMARY_POISON"},
    {"role": "user", "content": "SUMMARY_POISON", "_compressed_summary": True},
])
def test_summary_is_a_boundary_not_a_topic_source(summary: dict) -> None:
    query, _, _ = _recall("How can we improve it?", [
        {"role": "user", "content": "STALE_TOPIC"},
        {"role": "assistant", "content": "STALE_REPLY"},
        summary,
        {"role": "assistant", "content": "Understood; continuing."},
    ])
    assert query == "How can we improve it?"


def test_multimodal_history_uses_text_not_memory_or_media() -> None:
    query, _, _ = _recall("How can we improve it?", [
        {"role": "user", "content": [
            {"type": "text", "text": "Review Hindsight with Mira."},
            {"type": "image_url", "image_url": {"url": "MEDIA_POISON"}},
            {"type": "text", "text": "<memory-context>MEMORY_POISON</memory-context>"},
        ]},
        {"role": "assistant", "content": "The recall query needs context."},
    ])
    assert "Hindsight" in query and "Mira" in query
    assert "POISON" not in query


def test_current_persisted_source_wins_over_model_facing_scaffolding() -> None:
    current = "How can we improve it without activation?"
    query, _, _ = _recall("MODEL_FACING_POISON", [
        {"role": "user", "content": "Compare Honcho and Hindsight."},
        {"role": "assistant", "content": "Hindsight could use contextual recall."},
    ], persist=current)
    assert query.startswith(current)
    assert "Hindsight" in query and "POISON" not in query


def test_long_query_keeps_head_and_tail_under_provider_input_budget() -> None:
    current = "Review Project-Orion with Mira. " + "details " * 2000 + "Deadline 2026-10-12; do not activate."
    query, _, _ = _recall(current, [{"role": "assistant", "content": "STALE_TOPIC"}])
    assert len(query) <= 1200
    assert "Project-Orion" in query and "Mira" in query
    assert "2026-10-12; do not activate." in query
    assert "STALE_TOPIC" not in query


def test_long_context_keeps_latest_request_and_context_endpoints_bounded() -> None:
    current = "How can we improve it? Do not activate."
    query, _, _ = _recall(current, [
        {"role": "user", "content": "Review Project-Orion. " + "detail " * 2000 + "Owner Mira; deadline 2026-10-12."},
        {"role": "assistant", "content": "Tradeoffs " * 2000},
    ])
    assert query.startswith(current) and len(query) <= 1200
    assert "Project-Orion" in query and "Mira" in query and "2026-10-12" in query


def test_parallel_sessions_have_no_context_cache_or_cross_session_state() -> None:
    def run(topic: str) -> str:
        query, _, _ = _recall("How can we improve it?", [
            {"role": "user", "content": f"Review {topic}."},
            {"role": "assistant", "content": f"{topic} needs work."},
        ], session_id=topic)
        return query

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(run, ["Session-A-Orion", "Session-B-Vega"]))
    assert "Session-A-Orion" in a and "Session-B-Vega" not in a
    assert "Session-B-Vega" in b and "Session-A-Orion" not in b
    fresh, _, _ = _recall("How can we improve it?", [], session_id="fresh")
    assert fresh == "How can we improve it?"


@pytest.mark.parametrize("current", ["hi!", "thanks", "do it", "/new"])
def test_existing_trivial_recall_gate_remains_closed(current: str) -> None:
    query, context, provider = _recall(current, [
        {"role": "user", "content": "Deploy Hindsight?"},
        {"role": "assistant", "content": "Awaiting authorization."},
    ])
    assert query == "" and provider.prefetch_queries == []
    assert context.ext_prefetch_cache == ""


def test_compacted_history_including_current_row_does_not_duplicate_request(monkeypatch) -> None:
    from agent.turn_context_compaction import run_turn_start_compaction

    def compact(*args, **kwargs):
        outcome = run_turn_start_compaction(*args, **kwargs)
        # The in-place compression path can return the full active transcript,
        # including the current user row, as its persistence history.
        outcome.conversation_history = list(outcome.messages)
        return outcome

    monkeypatch.setattr("agent.turn_context_compaction.run_turn_start_compaction", compact)
    current = "How can we improve it?"
    query, _, _ = _recall(current, [
        {"role": "user", "content": "Compare Honcho and Hindsight."},
        {"role": "assistant", "content": "Hindsight needs context."},
    ])
    assert query.count(current) == 1
    assert "Hindsight" in query


def test_parallel_profile_context_reaches_correct_provider(tmp_path) -> None:
    from threading import Barrier
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

    barrier = Barrier(2)

    class ScopedProvider(FakeMemoryProvider):
        def prefetch(self, query: str, *, session_id: str = "") -> str:
            self.observed = (get_hermes_home(), session_id)
            barrier.wait(timeout=5)
            return super().prefetch(query, session_id=session_id)

    def run(topic: str) -> str:
        home = tmp_path / topic
        token = set_hermes_home_override(home)
        provider = ScopedProvider()
        try:
            query, _, _ = _recall("How can we improve it?", [
                {"role": "user", "content": f"Review {topic}."},
                {"role": "assistant", "content": f"{topic} needs work."},
            ], session_id=topic, provider=provider)
            assert provider.observed == (home, topic)
            return query
        finally:
            reset_hermes_home_override(token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(run, ["Profile-Orion", "Profile-Vega"]))
    assert "Profile-Orion" in a and "Profile-Vega" not in a
    assert "Profile-Vega" in b and "Profile-Orion" not in b


def test_real_agent_turn_passes_context_to_recall_but_retains_original(monkeypatch) -> None:
    from types import SimpleNamespace

    class Completions:
        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="A focused answer.", tool_calls=None,
                                        reasoning=None, reasoning_content=None),
                finish_reason="stop")], usage=None, model="test-model")

    monkeypatch.setattr("agent.process_bootstrap.OpenAI", lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())))
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *args, **kwargs: [])
    agent = AIAgent(model="test-model", api_key="test", base_url="http://localhost:1/v1",
                    platform="cli", max_iterations=2, quiet_mode=True, skip_memory=True,
                    skip_context_files=True, save_trajectories=False)
    agent._disable_streaming = True
    agent._session_db = None
    manager = MemoryManager()
    provider = FakeMemoryProvider()
    manager.add_provider(provider)
    agent._memory_manager = manager
    current = "How can we improve it?"
    history = [
        {"role": "user", "content": "Compare Honcho and Hindsight."},
        {"role": "assistant", "content": "Hindsight needs contextual recall."},
    ]
    result = agent.run_conversation(current, conversation_history=history)
    assert result["final_response"] == "A focused answer."
    assert len(provider.prefetch_queries) == 1
    assert provider.prefetch_queries[0].startswith(current)
    assert "Hindsight" in provider.prefetch_queries[0]
    assert manager.flush_pending(timeout=5)
    assert provider.synced_turns == [(current, "A focused answer.")]
    manager.shutdown_all()


def test_long_history_and_wire_prefix_are_immutable_across_recall() -> None:
    from agent.turn_context import build_api_messages

    history = [row for n in range(100) for row in (
        {"role": "user", "content": f"Unrelated old topic {n}",
         "api_content": f"Unrelated old topic {n}\n<memory-context>OLD_MEMORY_{n}</memory-context>"},
        {"role": "assistant", "content": f"Old answer {n}"},
    )]
    history.extend([
        {"role": "user", "content": "Review Hindsight with Mira by 2026-10-12.",
         "api_content": "Review Hindsight with Mira by 2026-10-12.\n<memory-context>STALE_MEMORY</memory-context>"},
        {"role": "assistant", "content": "Hindsight needs contextual retrieval."},
    ])
    before = deepcopy(history)
    query, context, _ = _recall("How can we improve it?", history)
    assert "Hindsight" in query and "OLD_MEMORY" not in query and "STALE_MEMORY" not in query
    assert "Unrelated old topic" not in query
    agent: Any = _FakeAgent()
    agent._current_turn_timestamp = 1.0
    agent._copy_reasoning_content_for_api = lambda *args: None
    agent._should_sanitize_tool_calls = lambda: False
    agent.ephemeral_system_prompt = ""
    wire, system = build_api_messages(
        agent, context.messages, current_turn_user_idx=context.current_turn_user_idx,
        ext_prefetch_cache=context.ext_prefetch_cache, plugin_user_context="",
        moa_config=None, active_system_prompt="SYSTEM",
    )
    assert system == "SYSTEM"
    assert [row["content"] for row in wire[1:-1]] == [
        row.get("api_content", row["content"]) for row in before
    ]
    assert history == before and context.messages[:-1] == before
    assert [row["role"] for row in wire] == ["system"] + [row["role"] for row in context.messages]


@pytest.mark.parametrize("row", [
    {"role": "user", "content": "[System: continue the old task] POISON"},
    {"role": "user", "content": "POISON", "_verification_stop_synthetic": True},
    {"role": "user", "content": "> POISON\n```\nMORE_POISON\n```"},
])
def test_excluded_recent_turn_does_not_reach_back_to_stale_topic(row: dict) -> None:
    query, _, _ = _recall("How can we improve it?", [
        {"role": "user", "content": "STALE_TOPIC"},
        {"role": "assistant", "content": "STALE_REPLY"},
        row,
        {"role": "assistant", "content": "Continuing."},
    ])
    assert query == "How can we improve it?"
