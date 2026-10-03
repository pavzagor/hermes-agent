"""Behavior contracts for a read-only cron proposal and public task checks."""
from __future__ import annotations

import copy
import importlib
import importlib.util
from typing import Any
from pathlib import Path
import json

import pytest


def audit() -> Any:
    name = "evals.cron_speed_quality.audit"
    assert importlib.util.find_spec(name) is not None, "Read-only proposal validator is missing"
    return importlib.import_module(name)


def job() -> dict[str, Any]:
    return {"id": "123456abcdef", "enabled": True, "no_agent": False,
            "model": None, "provider": None, "prompt": "Keep every requirement. Return [SILENT] on no change.",
            "deliver": "local", "skills": [], "schedule": {"kind": "interval", "minutes": 15},
            "last_run_at": None, "repeat": {"times": None, "completed": 0}}


def proposal(j: dict[str, Any]) -> dict[str, Any]:
    m = audit()
    return {"schema_version": 1, "profile": "default", "review": "pending_parent_astra",
            "eligible_job_ids": [j["id"]], "unchanged_job_ids": [],
            "changes": [{"job_id": j["id"], "settings_sha256": m.settings_hash(j),
                         "current": {"model": j.get("model"), "provider": j.get("provider")},
                         "recommended": {"model": "public-sol", "provider": "public-provider"},
                         "rollback": {"model": j.get("model"), "provider": j.get("provider")},
                         "rationale": "Known-output routine task; parent must review task evidence.",
                         "task_evidence": "public_fixture_v1", "intentional_pin_override": False}]}


@pytest.mark.parametrize("mutation", ["none", "no_agent", "disabled", "intentional_pin", "stale_request",
                                    "rollback", "unauthorized", "duplicate", "unrelated_write", "missing_evidence"])
def test_validator_rejects_unsafe_proposals_without_mutating_inputs(mutation: str) -> None:
    m = audit()
    j = job()
    p = proposal(j)
    if mutation == "no_agent":
        j["no_agent"] = True
        p["changes"][0]["settings_sha256"] = m.settings_hash(j)
    elif mutation == "disabled":
        j["enabled"] = False
        p["changes"][0]["settings_sha256"] = m.settings_hash(j)
    elif mutation == "intentional_pin":
        j["model"] = "public-astra"
        p = proposal(j)
    elif mutation == "stale_request":
        j["prompt"] += " Never accept altered remedies."
    elif mutation == "rollback":
        p["changes"][0]["rollback"]["model"] = "wrong"
    elif mutation == "unauthorized":
        p["changes"][0]["recommended"]["model"] = "public-luna"
    elif mutation == "duplicate":
        p["changes"].append(copy.deepcopy(p["changes"][0]))
    elif mutation == "unrelated_write":
        p["changes"][0]["recommended"]["prompt"] = "Drop constraints"
    elif mutation == "missing_evidence":
        p["changes"][0].pop("task_evidence")
    before = copy.deepcopy((j, p))
    errors = m.validate_proposal(p, [j], {"sol": {"provider": "public-provider", "model": "public-sol"}})
    assert bool(errors) == (mutation != "none"), errors
    assert (j, p) == before
    if mutation == "none":
        j["last_run_at"] = "2026-10-03T10:00:00Z"
        j["repeat"]["completed"] = 2
        assert m.validate_proposal(p, [j], {"sol": {"provider": "public-provider", "model": "public-sol"}}) == []


@pytest.mark.parametrize("field", ["eligible_job_ids", "unchanged_job_ids"])
@pytest.mark.parametrize("mutation", ["duplicate", "unknown", "missing", "misclassified", "absent", "not_list", "nonstring"])
def test_validator_reconciles_complete_scope(field: str, mutation: str) -> None:
    m = audit()
    j = job()
    other = job() | {"id": "abcdef123456", "no_agent": True}
    p = proposal(j) | {"unchanged_job_ids": [other["id"]]}
    if mutation == "duplicate":
        p[field] *= 2
    elif mutation == "unknown":
        p[field].append("unknown")
    elif mutation == "missing":
        p[field] = []
    elif mutation == "misclassified":
        p[field] = [other["id"] if field == "eligible_job_ids" else j["id"]]
    elif mutation == "absent":
        p.pop(field)
    elif mutation == "not_list":
        p[field] = j["id"]
    else:
        p[field].append({"id": "not-a-string"})
    before = copy.deepcopy((p, j, other))
    assert m.validate_proposal(p, [j, other], {"sol": {"provider": "public-provider", "model": "public-sol"}})
    assert (p, j, other) == before


@pytest.mark.parametrize("bad_id", [None, "", 7, {"invalid": "id"}])
def test_validator_rejects_invalid_source_ids(bad_id: Any) -> None:
    m = audit()
    j = job()
    p = proposal(j)
    invalid = job() | {"id": bad_id}
    assert m.validate_proposal(p, [j, invalid], {"sol": {"provider": "public-provider", "model": "public-sol"}})


@pytest.mark.parametrize("mutation", ["unknown_change", "missing_change", "expanded_change", "source_duplicate", "overlap", "none"])
def test_change_scope_cannot_expand_or_hide_missing_changes(mutation: str) -> None:
    m = audit()
    j = job()
    other = job() | {"id": "abcdef123456"}
    routes = {"sol": {"provider": "public-provider", "model": "public-sol"}}
    p = proposal(j) | {"unchanged_job_ids": [other["id"]]}
    source = [j, other]
    if mutation == "unknown_change":
        p["changes"][0]["job_id"] = "unknown"
    elif mutation == "missing_change":
        p["changes"] = []
    elif mutation == "expanded_change":
        p["changes"].append(proposal(other)["changes"][0])
    elif mutation == "source_duplicate":
        source.append(copy.deepcopy(other))
    elif mutation == "overlap":
        p["unchanged_job_ids"].append(j["id"])
    original = copy.deepcopy((p, source))
    errors = m.validate_proposal(p, source, routes)
    assert bool(errors) is (mutation != "none")
    assert (p, source) == original
    # A fully accounted no-op is valid; eligibility means selected changes, not
    # every enabled agent that could theoretically be considered.
    if mutation == "none":
        p.update(changes=[], eligible_job_ids=[], unchanged_job_ids=[j["id"], other["id"]])
        assert m.validate_proposal(p, source, routes) == []


@pytest.mark.parametrize("name,answer,expected", [
    ("no_change", "[SILENT]", True), ("no_change", "Nothing changed", False),
    ("status_exception", '{"report_ids":["evt-new"],"requires_approval":true}', True),
    ("status_exception", '{"report_ids":["evt-old","evt-new"],"requires_approval":false}', False),
    ("fixed_reminder", "Check the quality rubric.", True),
    ("fixed_reminder", "Check the quality rubric. I updated the task.", False),
])
def test_public_checks_enforce_silence_duplicates_approval_and_no_fabricated_writes(
    name: str, answer: str, expected: bool,
) -> None:
    assert audit().check_answer(name, answer) is expected


@pytest.mark.parametrize("name", ["no_change", "status_exception", "fixed_reminder"])
def test_public_benchmark_preserves_prompt_and_disables_tools(name: str) -> None:
    module = "evals.cron_speed_quality.benchmark"
    assert importlib.util.find_spec(module) is not None, "Bounded public provider harness is missing"
    benchmark = importlib.import_module(module)
    routes = {"sol": {"provider": "openai-codex", "model": "authorized-sol"},
              "astra": {"provider": "openai-codex", "model": "authorized-astra"}}
    sol = benchmark.request_for_task(name, "sol", routes)
    astra = benchmark.request_for_task(name, "astra", routes)
    assert sol["model"] == routes["sol"]["model"]
    assert astra["model"] == routes["astra"]["model"]
    assert {k: v for k, v in sol.items() if k != "model"} == {k: v for k, v in astra.items() if k != "model"}
    assert sol["tools"] == [] and sol["store"] is False
    assert sol["input"][0]["content"] == audit().PUBLIC_PROMPTS[name]
    assert benchmark.request_for_task(name, "sol", routes) == sol
    with pytest.raises(ValueError):
        benchmark.request_for_task(name, "luna", routes)
    with pytest.raises(ValueError):
        benchmark.request_for_task("private-transcript", "sol", routes)


def test_stream_text_is_not_lost_when_real_sdk_final_is_empty() -> None:
    benchmark = importlib.import_module("evals.cron_speed_quality.benchmark")
    fixture = json.loads((Path(__file__).parent / "fixtures/cron_public_response.json").read_text())
    assert hasattr(benchmark, "answer_from_stream"), "Genuine provider stream text is currently lost"
    assert benchmark.answer_from_stream(fixture["events"], fixture["output_text"]) == "[SILENT]"


def test_proposal_builder_has_exact_scope_and_rollback_without_runtime_writes() -> None:
    m = audit()
    assert hasattr(m, "build_proposal"), "Deterministic proposal builder is missing"
    j = job()
    other = job() | {"id": "abcdef123456", "no_agent": True}
    routes = {"sol": {"provider": "public-provider", "model": "public-sol"}}
    selection = [{"job_id": j["id"], "rationale": "Known-output public test", "task_evidence": "public_fixture_v1"}]
    original = copy.deepcopy([j, other])
    p = m.build_proposal([j, other], routes, selection)
    assert m.validate_proposal(p, [j, other], routes) == []
    assert p["eligible_job_ids"] == [j["id"]]
    assert p["unchanged_job_ids"] == [other["id"]]
    assert p["changes"][0]["rollback"] == {"model": None, "provider": None}
    assert m.build_proposal([j, other], routes, selection) == p
    assert [j, other] == original
    with pytest.raises(ValueError):
        m.build_proposal([j, other], routes, [{"job_id": other["id"], "rationale": "No-agent", "task_evidence": "x"}])


def test_collection_uses_real_instants_not_lexical_local_timestamps(tmp_path: Path, monkeypatch: Any) -> None:
    import sqlite3
    from datetime import datetime, timedelta, timezone
    m = audit()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    (home / "cron").mkdir(parents=True)
    (home / "cron/jobs.json").write_text(json.dumps({"jobs": [job()]}))
    (home / "config.yaml").write_text("model:\n  default: public-astra\n  provider: public-provider\n")
    (home / "cron/usage_audit.jsonl").write_text("")
    with sqlite3.connect(home / "state.db") as db:
        db.execute("CREATE TABLE session_model_usage (session_id,model,billing_provider,task,api_call_count,input_tokens,output_tokens,cache_read_tokens,cache_write_tokens,reasoning_tokens,first_seen,last_seen)")
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    local = timezone(timedelta(hours=3))
    with sqlite3.connect(home / "cron/executions.db") as db:
        db.execute("CREATE TABLE executions (job_id,status,claimed_at,started_at,finished_at)")
        for instant in (cutoff - timedelta(minutes=1), cutoff + timedelta(minutes=1)):
            db.execute("INSERT INTO executions VALUES (?,?,?,?,?)", (job()["id"], "completed", instant.astimezone(local).isoformat(), None, None))
    result = m.collect(home, 7)
    assert result["history_rows"]["executions"] == 1
    assert result["inventory"][0]["execution_statuses"] == {"completed": 1}
    assert "prompt" not in result["inventory"][0]
