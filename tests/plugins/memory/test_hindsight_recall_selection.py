"""Offline provider-boundary contracts for opt-in provenance-safe recall.

Fixture fields/signature are the hindsight-client 0.6.1 RecallResult/arecall
contract (PyPI wheel); also accepted by server 0.10.1's /openapi.json.
No daemon, retention, bank setup, dependency installation or model calls.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from agent.model_metadata import estimate_tokens_rough
from plugins.memory.hindsight import HindsightMemoryProvider


@dataclass
class Fact:
    id: str
    text: str
    type: str = "world"
    entities: list[str] | None = None
    context: str | None = None
    occurred_start: str | None = None
    occurred_end: str | None = None
    mentioned_at: str | None = None
    document_id: str | None = None
    metadata: dict[str, str] | None = None
    chunk_id: str | None = None
    tags: list[str] | None = None
    source_fact_ids: list[str] | None = None
    scores: SimpleNamespace | dict[str, float] | None = None


class RecallClient061:
    def __init__(self, observations: list[Fact], raw: list[Fact]) -> None:
        self.observations, self.raw = observations, raw
        self.calls: list[dict] = []
        self.fail_types: list[str] = []

    async def arecall(
        self, bank_id: str, query: str, types: list[str] | None = None,
        max_tokens: int = 4096, budget: str = "mid", trace: bool = False,
        query_timestamp: str | None = None, include_entities: bool = False,
        max_entity_tokens: int = 500, include_chunks: bool = False,
        max_chunk_tokens: int = 8192, include_source_facts: bool = False,
        max_source_facts_tokens: int = 4096, tags: list[str] | None = None,
        tags_match: str = "any", tag_groups: list[dict] | None = None,
    ) -> SimpleNamespace:
        self.calls.append({"types": types, "query": query, "max_tokens": max_tokens,
                           "bank_id": bank_id, "budget": budget, "tags": tags,
                           "tags_match": tags_match})
        if types == self.fail_types:
            raise RuntimeError("recall unavailable")
        results = [r for r in self.observations + self.raw if types is None or r.type in types]
        return SimpleNamespace(results=results, source_facts=None)


@pytest.fixture
def make_provider(tmp_path, monkeypatch):
    monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("plugins.memory.hindsight.dependencies.ensure", lambda *a, **kw: None)

    def make(observations=(), raw=(), **config):
        cfg = {"mode": "local_external", "api_url": "http://invalid.test",
               "bank_id": "test-bank", "recall_sync": True, "auto_retain": False,
               "recall_selection_policy": "observation_first", "recall_max_tokens": 1200,
               **config}
        path = tmp_path / "hindsight" / "config.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(cfg))
        provider = HindsightMemoryProvider()
        provider.initialize(session_id="test", hermes_home=str(tmp_path), platform="cli")
        provider._client = RecallClient061(list(observations), list(raw))
        return provider
    return make


def test_auto_score_floor_omits_noise_but_explicit_recall_keeps_access(make_provider):
    noise = Fact("noise", "An unrelated historical incident.")
    noise.scores = SimpleNamespace(reranker=0.002)
    p = make_provider(raw=[noise], recall_auto_min_reranker_score=0.01)
    assert p.prefetch("Explain the sky color.") == ""
    assert p.recall_diagnostics()["low_score_records_omitted"] == 1
    assert "An unrelated historical incident." in p.handle_tool_call(
        "hindsight_recall", {"query": "Find that historical incident."},
    )
    assert len(p._client.calls) == 2


def test_auto_score_floor_preserves_relevant_and_unscored_evidence(make_provider):
    relevant = Fact("relevant", "The user confirmed the issue resolved.")
    relevant.scores = SimpleNamespace(reranker=0.9)
    unscored = Fact("legacy", "A sparse legacy fact with no scoring metadata.")
    noise = Fact("noise", "Unrelated.")
    noise.scores = {"reranker": 0.0001}
    p = make_provider(raw=[relevant, unscored, noise], recall_auto_min_reranker_score=0.01)
    output = p.prefetch("Is the issue resolved?")
    assert relevant.text in output and unscored.text in output
    assert noise.text not in output
    assert len(p._client.calls) == 1


@pytest.mark.parametrize("value", [-0.1, 1.1, True, "invalid", float("nan"), float("inf")])
def test_invalid_auto_score_floor_is_rejected(make_provider, value):
    with pytest.raises(ValueError, match="recall_auto_min_reranker_score"):
        make_provider(recall_auto_min_reranker_score=value)


def test_mixed_recall_avoids_a_second_network_round_trip(make_provider):
    p = make_provider([Fact("o", "Old summary.", "observation")], [Fact("r", "New correction.")])
    output = p.prefetch("What changed?")
    assert "Old summary." in output and "New correction." in output
    assert [c["types"] for c in p._client.calls] == [["observation", "world", "experience"]]


def test_coalesced_old_observation_cannot_steal_newest_raw_slot(make_provider):
    old = "Older claim " + "earlier " * 450
    correction = "Latest correction: not approved. " + "detail " * 120
    p = make_provider([Fact("obs", old, "observation", source_fact_ids=["old"], mentioned_at="2026-01-01")],
                      [Fact("old", old, mentioned_at="2026-01-01"),
                       Fact("new", correction, mentioned_at="2026-10-08")])
    output = p.prefetch("What is approved?")
    assert correction.strip() in output
    assert estimate_tokens_rough(output) <= 1200


def test_exact_text_distinct_sources_share_body_with_compact_provenance(make_provider):
    text = "Pavel prefers English; Russian only if explicitly requested."
    p = make_provider([Fact("a", text, "observation", mentioned_at="2026-10-01",
                            metadata={"session_id": "verbose-scaffold-" * 100, "speaker": "user"})],
                      [Fact("b", text, mentioned_at="2026-10-02", metadata={"speaker": "user"})])
    output = p.prefetch("Language preference?")
    assert output.count(text) == 1
    assert '"id":"a"' in output and '"id":"b"' in output
    assert "2026-10-01" in output and "2026-10-02" in output
    assert '"speaker":"user"' in output
    assert "verbose-scaffold" not in output
    assert p.recall_diagnostics()["duplicate_records_coalesced"] == 1


def test_modern_sdk_uses_server_observation_preference_in_one_call(make_provider):
    p = make_provider([Fact("o", "Old summary.", "observation")], [Fact("r", "Unconsolidated correction.")])
    calls = []
    old_recall = p._client.arecall

    async def modern_recall(*, prefer_observations=False, **kwargs):
        calls.append(prefer_observations)
        return await old_recall(**kwargs)

    p._client.arecall = modern_recall
    assert "Unconsolidated correction." in p.prefetch("What changed?")
    assert calls == [True]


def test_scaffold_grouping_keeps_each_source_date_and_attribution(make_provider):
    body = "The migration was proposed, not approved."
    p = make_provider([Fact("o", body, "observation")],
                      [Fact("a", body + " | When: Monday | Involving: Assistant | Unconfirmed."),
                       Fact("b", body + " | When: Tuesday | Involving: Pavel | Rejected.")])
    output = p.prefetch("Was it approved?")
    assert output.count(body) == 1
    for detail in ("Monday", "Tuesday", "Assistant", "Pavel", "Unconfirmed.", "Rejected."):
        assert detail in output
    assert all(f'"id":"{ident}"' in output for ident in ("o", "a", "b"))


def test_host_deadline_delivers_single_recall_not_two_sequential_calls(make_provider):
    import asyncio
    from agent.memory_manager import MemoryManager

    p = make_provider([Fact("o", "Old summary.", "observation")], [Fact("r", "Latest correction.")])
    client_recall = p._client.arecall

    async def delayed_recall(**kwargs):
        await asyncio.sleep(4.1)
        return await client_recall(**kwargs)

    p._client.arecall = delayed_recall
    manager = MemoryManager(external_prefetch_timeout=8.0)
    manager._providers = [p]
    output = manager.prefetch_all("What changed?")
    assert "Latest correction." in output
    assert p.recall_diagnostics()["selected_claims"] == 2
    assert "injected_claims" not in p.recall_diagnostics()


def test_observation_does_not_suppress_unconsolidated_correction(make_provider):
    old = Fact("obs", "Pavel lives in Berlin.", "observation",
               mentioned_at="2026-10-08T15:00:00Z", source_fact_ids=["old"])
    corrected = Fact("new", "Pavel no longer lives in Berlin; he moved to Ankara.",
                     mentioned_at="2026-10-08T12:00:00Z", metadata={"speaker": "user"})
    p = make_provider([old], [corrected], recall_types="observation,world")
    output = p.prefetch("Where does Pavel live?")
    assert "Ankara" in output
    assert "Berlin" in output
    assert [c["types"] for c in p._client.calls] == [["observation", "world", "experience"]]
    assert "2026-10-08T15:00:00Z" in output
    assert "2026-10-08T12:00:00Z" in output
    assert "not proof" in output
    assert "superseded" not in output


@pytest.mark.parametrize("query", ["Exactly when did it happen?", "Show the source quotation",
    "What changed recently?", "What happened on 2025-02-03?", "Когда случилась авария?",
    "Tell me about this", "What do you know?"])
def test_raw_fallback_for_detail_and_ambiguous_queries(make_provider, query):
    p = make_provider([Fact("obs", "There was an incident.", "observation")],
                      [Fact("raw", "The outage happened on 2025-02-03, later resolved.",
                            occurred_start="2025-02-03", metadata={"status": "resolved"})])
    output = p.prefetch(query)
    assert "2025-02-03" in output
    assert '"status":"resolved"' in output
    assert p._client.calls[-1]["query"] == query


def test_empty_observations_fall_back_and_empty_bank_abstains(make_provider):
    p = make_provider([], [Fact("r", "Only retained fact.")])
    assert "Only retained fact." in p.prefetch("What happened?")
    p._client.raw = []
    assert p.prefetch("Nothing known") == ""
    assert p.recall_status() is None


def test_duplicate_text_coalesces_without_mutating_sources(make_provider):
    a = Fact("a", "Pavel uses dark mode.", "observation", document_id="doc", chunk_id="chunk")
    b = Fact("b", "Pavel  uses dark mode.", "observation", document_id="doc", chunk_id="chunk")
    p = make_provider([a, b], [])
    before = copy.deepcopy(p._client.observations)
    output = p.prefetch("Preferences?")
    assert output.count("Pavel uses dark mode.") == 1
    assert '"id":"a"' in output and '"id":"b"' in output
    assert p.recall_status().count == 1
    assert p._client.observations == before


@pytest.mark.parametrize("change", [
    {"text": "Pavel does not use dark mode."}, {"text": "Anna uses dark mode."},
    {"text": "Pavel uses light mode."}, {"mentioned_at": "2026-10-01T00:00:00Z"},
    {"occurred_start": "2025-01-01"}, {"metadata": {"speaker": "assistant"}},
    {"metadata": {"status": "proposed"}}, {"document_id": "another-document"},
    {"text": "Pavel uses Dark Mode."},
])
def test_different_subject_value_time_speaker_status_not_lost(make_provider, change):
    first = Fact("a", "Pavel uses dark mode.", "observation", document_id="doc")
    second = Fact(**{**first.__dict__, "id": "b", **change})
    p = make_provider([first, second], [])
    output = p.prefetch("What is known?")
    assert '"id":"a"' in output and '"id":"b"' in output
    if "text" in change:
        assert p.recall_status().count == 2
    else:
        assert p.recall_status().count == 1
        for key in ("mentioned_at", "occurred_start"):
            if key in change:
                assert change[key] in output
        for value in change.get("metadata", {}).values():
            assert value in output


def test_source_link_alone_does_not_erase_negation_or_extra_detail(make_provider):
    p = make_provider([Fact("obs", "Pavel chose a migration.", "observation", source_fact_ids=["r"])],
                      [Fact("r", "Assistant recommended a migration; user has not approved it.",
                            metadata={"speaker": "assistant", "authority": "recommendation"})])
    output = p.prefetch("Did I approve?")
    assert "has not approved" in output
    assert '"authority":"recommendation"' in output
    assert "user decisions" in output
    assert '"source_fact_ids":["r"]' in output
    assert p.recall_status().count == 2


def test_same_text_keeps_distinct_unknown_sources(make_provider):
    p = make_provider([Fact("a", "Ready.", "observation"), Fact("b", "Ready.", "observation")], [])
    output = p.prefetch("Status?")
    assert '"id":"a"' in output and '"id":"b"' in output
    assert p.recall_status().count == 1


def test_raw_and_observation_same_claim_same_source_group_provenance(make_provider):
    p = make_provider([Fact("obs", "Pavel uses dark mode.", "observation", source_fact_ids=["r"])],
                      [Fact("r", "Pavel uses dark mode.")])
    output = p.prefetch("Preferences?")
    assert output.count("Pavel uses dark mode.") == 1
    assert '"id":"r"' in output and '"id":"obs"' in output
    assert p.recall_status().count == 1


def test_budget_reserves_raw_evidence_and_never_slices_claims(make_provider):
    p = make_provider([Fact(str(i), "Older observation " + ("word " * 30), "observation") for i in range(30)],
                      [Fact("new", "Correction: use Ankara, not Berlin.", mentioned_at="2026-10-08")],
                      recall_max_tokens=400)
    output = p.prefetch("Where?")
    assert "Correction: use Ankara, not Berlin." in output
    assert "omitted" in output
    assert estimate_tokens_rough(output) <= 400
    assert all(c["max_tokens"] <= 400 for c in p._client.calls)
    assert "..." not in output


def test_raw_order_uses_mention_not_observation_creation_as_truth(make_provider):
    p = make_provider([Fact("o", "Old summary", "observation", mentioned_at="2026-10-09")],
                      [Fact("old", "Old raw", mentioned_at="2026-01-01"),
                       Fact("new", "New raw", mentioned_at="2026-10-08"),
                       Fact("unknown", "Undated raw")])
    output = p.prefetch("Current status?")
    assert output.index("New raw") < output.index("Old raw")
    assert "Undated raw" in output
    assert "Old summary" in output


def test_historical_top_match_survives_newer_lower_ranked_raw(make_provider):
    incident = Fact("historical", "The outage on 2024-02-05 was resolved the next day.",
                    mentioned_at="2024-02-06", occurred_start="2024-02-05",
                    metadata={"status": "resolved"})
    recent = [Fact(f"recent-{i}", "A later update " + ("detail " * 25),
                   mentioned_at=f"2026-10-{i + 1:02d}") for i in range(15)]
    p = make_provider([Fact("obs", "Service now works.", "observation")], [incident, *recent])
    output = p.prefetch("What happened during the outage on 2024-02-05?")
    assert incident.text in output
    assert '"status":"resolved"' in output
    assert estimate_tokens_rough(output) <= 1200


def test_failed_mixed_recall_abstains_without_a_second_slow_probe(make_provider):
    p = make_provider([Fact("o", "Observation", "observation")], [Fact("r", "Raw fact")])
    p._client.fail_types = ["observation", "world", "experience"]
    assert p.prefetch("Status?") == ""
    assert p.recall_status() is None
    assert len(p._client.calls) == 1
    assert p.recall_diagnostics()["failed_lanes"] == ["mixed"]


def test_legacy_is_default_and_preserves_one_request(make_provider):
    p = make_provider([Fact("o", "Observation", "observation")], [Fact("r", "Raw fact")],
                      recall_selection_policy="legacy")
    assert "Observation" in p.prefetch("Status?")
    assert len(p._client.calls) == 1
    assert "evidence=" not in p.prefetch("Status?")


@pytest.mark.parametrize("value", [True, None, "observation-first", {}, 1])
def test_invalid_policy_is_rejected_not_truthy_enabled(make_provider, value):
    with pytest.raises(ValueError, match="recall_selection_policy"):
        make_provider(recall_selection_policy=value)


def test_schema_exposes_opt_in_and_manual_tool_uses_policy(make_provider):
    p = make_provider([], [Fact("r", "Exact historical fact.", occurred_start="2024-02-05")],
                      recall_tags=["scope"], recall_tags_match="all_strict")
    schema = {s["key"]: s for s in p.get_config_schema()}
    assert schema["recall_selection_policy"]["default"] == "legacy"
    result = json.loads(p.handle_tool_call("hindsight_recall", {"query": "History?"}))
    assert "2024-02-05" in result["result"]
    assert all(c["tags"] == ["scope"] and c["tags_match"] == "all_strict" for c in p._client.calls)


def test_diagnostics_explain_selection_without_query_or_fact_content(make_provider):
    p = make_provider([Fact("secret-id", "Secret claim.", "observation", source_fact_ids=["raw-id"])],
                      [Fact("raw-id", "Secret claim.")])
    output = p.prefetch("Private query about a secret")
    metrics = p.recall_diagnostics()
    assert metrics["policy"] == "observation_first"
    assert metrics["path"] == "mixed_recall"
    assert metrics["requests"] == 1
    assert metrics["observations_returned"] == 1
    assert metrics["raw_returned"] == 1
    assert metrics["duplicate_records_coalesced"] == 1
    assert metrics["selected_claims"] == 1
    assert metrics["omitted_claims"] == 0
    assert metrics["estimated_tokens"] == estimate_tokens_rough(output)
    assert metrics["requested_max_tokens_total"] == 1200
    assert metrics["elapsed_ms"] >= metrics["retrieval_elapsed_ms"] >= 0
    assert not any(s in json.dumps(metrics) for s in ("secret-id", "raw-id", "Secret claim", "Private query"))
    metrics["requests"] = -1
    assert p.recall_diagnostics()["requests"] == 1


def test_invalid_policy_save_does_not_mutate_persisted_config(make_provider, tmp_path):
    p = make_provider()
    path = tmp_path / "hindsight" / "config.json"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="recall_selection_policy"):
        p.save_config({"recall_selection_policy": True}, str(tmp_path))
    assert path.read_bytes() == before


def test_diagnostics_count_preserved_embedded_reconnect_retry(make_provider, monkeypatch):
    p = make_provider([Fact("o", "Observation", "observation")], [Fact("r", "Raw")])
    healthy = p._client

    class StaleClient:
        async def arecall(self, **kwargs):
            raise ConnectionError("Connection refused")

    p._mode = "local_embedded"
    p._client = StaleClient()
    monkeypatch.setattr(p, "_new_embedded_client", lambda: healthy)
    assert "Observation" in p.prefetch("Status?")
    metrics = p.recall_diagnostics()
    assert metrics["requests"] == 2
    assert metrics["requested_max_tokens_total"] == 2400


def test_real_sdk_request_serialization_and_response_models(make_provider, monkeypatch):
    sdk = pytest.importorskip("hindsight_client")
    models = pytest.importorskip("hindsight_client_api.models.recall_response")
    p = make_provider()
    real_client = sdk.Hindsight(base_url="http://invalid.test")
    requests = []

    async def transport(bank_id, request, **kwargs):
        requests.append(request.to_dict())
        payloads = [
            {"id": "o", "type": "observation", "text": "Old summary.", "source_fact_ids": ["s"]},
            {"id": "r", "type": "world", "text": "User corrected the summary.",
             "mentioned_at": "2026-10-08", "metadata": {"speaker": "user", "status": "corrected"}},
        ]
        return models.RecallResponse.from_dict({"results": payloads})

    monkeypatch.setattr(real_client._memory_api, "recall_memories", transport)
    p._client = real_client
    try:
        output = p.prefetch("Current statement?")
        assert "User corrected the summary." in output
        assert "Old summary." in output
        assert '"status":"corrected"' in output
        assert [r["types"] for r in requests] == [["observation", "world", "experience"]]
        assert all(r["max_tokens"] == 1200 for r in requests)
    finally:
        p.shutdown()
