"""Final turn-context budget includes the production host framing, not just recall text."""
from __future__ import annotations

import copy
from uuid import uuid4

import pytest

from agent.memory_manager import MemoryManager, build_memory_context_block
from agent.model_metadata import estimate_tokens_rough
from tests.agent.test_turn_context import _FakeAgent, _build
from tests.plugins.memory.test_hindsight_recall_selection import Fact, make_provider  # noqa: F401


def _turn(provider, current="What is the recorded status?"):
    agent = _FakeAgent()
    agent.session_id = str(uuid4())
    manager = MemoryManager()
    manager.add_provider(provider)
    agent._memory_manager = manager
    context = _build(agent, user_message=current)
    row = context.messages[context.current_turn_user_idx]
    assert row["content"] == current
    assert agent._cached_system_prompt == "SYSTEM"
    wire = row.get("api_content", current)
    block = wire[len(current) + 2:] if wire != current else ""
    assert block == build_memory_context_block(context.ext_prefetch_cache)
    return block, context


@pytest.mark.parametrize("budget", [300, 400, 1200])
def test_turn_sidecar_full_block_fits_and_preserves_latest_raw_qualification(make_provider, budget):
    correction = Fact("new", "Correction: migration proposed; not approved. | When: October 8 | Involving: Assistant | Not Pavel's decision.",
                      mentioned_at="2026-10-08", metadata={"speaker": "assistant", "status": "proposed"})
    observations = [Fact(str(i), f"Older observation {i} " + "details " * 28, "observation") for i in range(30)]
    provider = make_provider(observations, [correction], recall_max_tokens=budget)
    before = copy.deepcopy(provider._client.raw)
    block, context = _turn(provider)
    assert estimate_tokens_rough(block) <= budget
    for literal in ("Correction: migration proposed; not approved.", "October 8", "Assistant", "Not Pavel's decision.",
                    '"id":"new"', '"speaker":"assistant"', '"status":"proposed"'):
        assert literal in block
    assert "omitted" in block and "..." not in block
    assert len(provider._client.calls) == 1
    assert provider._client.calls[0]["max_tokens"] == budget
    assert provider._client.raw == before
    assert context.ext_prefetch_cache


@pytest.mark.parametrize("empty", [False, True])
def test_unfittable_complete_correction_or_empty_recall_injects_nothing(make_provider, empty):
    records = [] if empty else [Fact("correction", "Migration approved " + "detail " * 400 + "; actually NOT approved.")]
    provider = make_provider(raw=records, recall_max_tokens=150)
    block, context = _turn(provider)
    assert block == "" and context.ext_prefetch_cache == ""
    assert provider.recall_status() is None
    assert len(provider._client.calls) == 1


def test_wrapper_only_boundary_abstains_without_losing_qualification(make_provider):
    provider = make_provider(raw=[Fact("correction", "Approved, but not by Pavel.")])
    payload = provider.prefetch("Recorded status?")
    # A full record fits the provider payload but not its final host wrapper.
    provider._recall_max_tokens = estimate_tokens_rough(payload) + 25
    block, _ = _turn(provider)
    assert block == ""
    # Explicit recovery is a tool result, not a memory-context injection: no wrapper tax.
    explicit = provider.handle_tool_call("hindsight_recall", {"query": "Recorded status?"})
    assert "Approved, but not by Pavel." in explicit
