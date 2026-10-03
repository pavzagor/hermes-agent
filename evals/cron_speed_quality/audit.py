"""Read-only default-profile cron evidence and fail-closed proposal validation.

No job runner, state writer, network API, or message delivery is imported here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import statistics
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

# Scheduler-owned fields may change between proposal review and activation.
RUNTIME_FIELDS = frozenset({
    "updated_at", "next_run_at", "last_run_at", "last_status", "last_error", "last_delivery_error",
    "last_delivery_unverified", "failure_streak", "last_dispatch", "fire_claim", "pending_slot",
    "monitor_state", "preflight_alerted", "run_claim",
})
CHANGE_FIELDS = frozenset({"job_id", "settings_sha256", "current", "recommended", "rollback",
                           "rationale", "task_evidence", "intentional_pin_override"})
PUBLIC_PROMPTS = {
    "no_change": "Read-only monitor. Previously reported IDs: [evt-old]. Latest events: [{id:evt-old,status:unchanged}]. Report only new material changes. Do not send messages or change data. If there is no material change, return exactly [SILENT].",
    "status_exception": "Read-only monitor. Previously reported IDs: [evt-old]. Latest events: [{id:evt-old,status:unchanged},{id:evt-new,status:seller proposes a replacement fee}]. Do not accept fees or altered remedies. Report only new material changes. Return only JSON with exactly report_ids (array of event IDs to report) and requires_approval (boolean). No tools or writes are available.",
    "fixed_reminder": "Return exactly this single sentence and nothing else: Check the quality rubric. Do not claim to update anything.",
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def settings_hash(job: dict[str, Any]) -> str:
    settings = {k: v for k, v in job.items() if k not in RUNTIME_FIELDS}
    if isinstance(settings.get("repeat"), dict):
        settings["repeat"] = {k: v for k, v in settings["repeat"].items() if k != "completed"}
    return hashlib.sha256(canonical(settings)).hexdigest()


def validate_proposal(proposal: Any, jobs: list[dict[str, Any]], routes: dict[str, Any]) -> list[str]:
    """Validate proposals only; deliberately has no apply mode or approval resolver."""
    if not isinstance(proposal, dict) or proposal.get("schema_version") != 1:
        return ["schema_version must equal 1"]
    if proposal.get("profile") != "default" or proposal.get("review") not in {"pending_parent_astra", "parent_reviewed"}:
        return ["default scope and explicit review state required"]
    changes = proposal.get("changes")
    if not isinstance(changes, list):
        return ["changes must be a list"]
    by_id = {j["id"]: j for j in jobs}
    if len(by_id) != len(jobs):
        return ["duplicate IDs in source jobs"]
    authorized_sol = routes.get("sol")
    errors: list[str] = []
    seen: set[str] = set()
    for index, change in enumerate(changes):
        prefix = f"change[{index}]"
        if not isinstance(change, dict) or set(change) != CHANGE_FIELDS:
            errors.append(f"{prefix}: exact change fields required")
            continue
        job_id = change["job_id"]
        if not isinstance(job_id, str) or job_id not in by_id or job_id in seen:
            errors.append(f"{prefix}: unknown or duplicate job ID")
            continue
        seen.add(job_id)
        job = by_id[job_id]
        if job.get("enabled") is not True or job.get("no_agent") is not False:
            errors.append(f"{prefix}: disabled/no_agent jobs must remain unchanged")
        if any(job.get(k) for k in ("model", "provider", "base_url")):
            errors.append(f"{prefix}: explicit pins require separate parent evidence/review; not eligible here")
        if change["settings_sha256"] != settings_hash(job):
            errors.append(f"{prefix}: stale authored settings")
        current = {"model": job.get("model"), "provider": job.get("provider")}
        if change["current"] != current or change["rollback"] != current:
            errors.append(f"{prefix}: exact current and rollback fields required")
        recommended = change["recommended"]
        if not isinstance(authorized_sol, dict) or recommended != authorized_sol or set(recommended) != {"model", "provider"}:
            errors.append(f"{prefix}: only exact currently authorized Sol pair is eligible")
        if change["intentional_pin_override"] is not False:
            errors.append(f"{prefix}: intentional pins cannot be overridden")
        for field in ("rationale", "task_evidence"):
            if not isinstance(change[field], str) or not change[field].strip():
                errors.append(f"{prefix}: {field} is required")
    return errors


def build_proposal(jobs: list[dict[str, Any]], routes: dict[str, Any], selection: list[dict[str, str]]) -> dict[str, Any]:
    """Selection is a reviewed input file, never inferred from model names or regexes."""
    by_id = {j["id"]: j for j in jobs}
    changes: list[dict[str, Any]] = []
    for evidence in sorted(selection, key=lambda row: row["job_id"]):
        jid = evidence["job_id"]
        if jid not in by_id:
            raise ValueError("Unknown selected job")
        job = by_id[jid]
        current = {"model": job.get("model"), "provider": job.get("provider")}
        changes.append({"job_id": jid, "settings_sha256": settings_hash(job), "current": current,
                        "recommended": dict(routes.get("sol") or {}), "rollback": dict(current),
                        "rationale": evidence["rationale"], "task_evidence": evidence["task_evidence"],
                        "intentional_pin_override": False})
    chosen = {change["job_id"] for change in changes}
    proposal = {"schema_version": 1, "profile": "default", "review": "pending_parent_astra",
                "eligible_job_ids": sorted(chosen),
                "unchanged_job_ids": sorted(set(by_id) - chosen), "changes": changes,
                "activation_blockers": ["Parent Astra must independently review task sufficiency and exact prompts.",
                                        "Scheduler pins must be authoritative through the actual pre-LLM route path before activation.",
                                        "Public microbenchmarks do not establish live cron latency or tool correctness."],
                "activation_owner": "parent_astra", "live_jobs_modified": False}
    errors = validate_proposal(proposal, jobs, routes)
    if errors:
        raise ValueError("; ".join(errors))
    return proposal


def check_answer(name: str, answer: str) -> bool:
    if name == "no_change":
        return answer.strip() == "[SILENT]"
    if name == "fixed_reminder":
        return answer.strip() == "Check the quality rubric."
    if name != "status_exception":
        raise ValueError("Unknown public fixture")
    try:
        result = json.loads(answer)
    except (ValueError, TypeError):
        return False
    return (isinstance(result, dict) and set(result) == {"report_ids", "requires_approval"}
            and result["report_ids"] == ["evt-new"] and result["requires_approval"] is True)


def default_home(home: Path) -> Path:
    home = home.expanduser().resolve()
    if home != (Path.home() / ".hermes").resolve():
        raise ValueError("This audit is authorized for the default profile only")
    return home


def read_inputs(home: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    home = default_home(home)
    jobs = json.loads((home / "cron/jobs.json").read_text())["jobs"]
    cfg = yaml.safe_load((home / "config.yaml").read_text())
    # Never serialize credential-bearing config sections.
    model = cfg.get("model") or {}
    main_model = model if isinstance(model, str) else model.get("default")
    return jobs, {"main_model": main_model,
                  "main_provider": model.get("provider") if isinstance(model, dict) else None,
                  "cron_model": (cfg.get("cron") or {}).get("model"),
                  "cron_model_provider": (cfg.get("cron") or {}).get("model_provider"),
                  "routes": (cfg.get("turn_routing") or {}).get("routes", {})}


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def collect(home: Path, days: int) -> dict[str, Any]:
    """Only usage rows and scheduler history; no sessions/messages or provider calls."""
    home = default_home(home)
    jobs, config = read_inputs(home)
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    audit_rows: list[dict[str, Any]] = []
    malformed = 0
    for line in (home / "cron/usage_audit.jsonl").read_text().splitlines():
        try:
            row = json.loads(line)
            if datetime.fromisoformat(row["ts"].replace("Z", "+00:00")) >= cutoff:
                audit_rows.append(row)
        except (ValueError, KeyError, TypeError):
            malformed += 1
    with sqlite3.connect(f"file:{home / 'state.db'}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        # Named query: cron_usage_window; do not query sessions or raw messages.
        usage_rows = [dict(row) for row in db.execute(
            "SELECT session_id,model,billing_provider,task,api_call_count,input_tokens,output_tokens,"
            "cache_read_tokens,cache_write_tokens,reasoning_tokens,first_seen,last_seen "
            "FROM session_model_usage WHERE session_id LIKE 'cron_%' AND last_seen>=?",
            (cutoff.timestamp(),))]
    with sqlite3.connect(f"file:{home / 'cron/executions.db'}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        # Named query: default_cron_execution_window; exclude private errors and output paths.
        executions = [dict(row) for row in db.execute(
            "SELECT job_id,status,claimed_at,started_at,finished_at FROM executions WHERE julianday(claimed_at)>=julianday(?)",
            (cutoff.isoformat(),))]
    inventory: list[dict[str, Any]] = []
    for job in jobs:
        jid = job["id"]
        runs = [r for r in audit_rows if r.get("job_id") == jid]
        usage = [r for r in usage_rows if r["session_id"].startswith(f"cron_{jid}_")]
        ledger = [r for r in executions if r["job_id"] == jid]
        serving: dict[str, Any] = {}
        for row in usage:
            key = f"{row['billing_provider']}|{row['model']}|{row['task']}"
            totals = serving.setdefault(key, {"usage_rows": 0, "api_calls": 0, "input_tokens": 0,
                                               "output_tokens": 0, "cache_read_tokens": 0,
                                               "cache_write_tokens": 0, "reasoning_tokens": 0})
            totals["usage_rows"] += 1
            totals["api_calls"] += row["api_call_count"]
            for name in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens"):
                totals[name] += row[name]
        pinned = any(job.get(k) for k in ("model", "provider", "base_url"))
        model_source = "no_agent" if job.get("no_agent") else ("explicit_job_pin" if pinned else
                       "cron_default" if config["cron_model"] else "main_default_unpinned")
        inventory.append({
            "job_id": jid, "enabled": job.get("enabled"), "no_agent": job.get("no_agent"),
            "current": {k: job.get(k) for k in ("model", "provider", "model_snapshot", "provider_snapshot", "reasoning_effort")},
            "settings_sha256": settings_hash(job), "authored_prompt_characters": len(job.get("prompt") or ""),
            "skill_count": len(job.get("skills") or []), "has_monitor_gate": bool(job.get("monitor_script") or job.get("monitor_url")),
            "model_source": model_source, "envelope_dominated": "not_established",
            "audit_runs": len(runs), "audit_errors": sum(bool(r.get("error")) for r in runs),
            "silent_runs": sum(r.get("response_silent") is True for r in runs),
            "run_duration_ms_median": median([r["duration_ms"] for r in runs if isinstance(r.get("duration_ms"), (int, float))]),
            "audit_prompt_tokens_median": median([r["prompt_tokens"] for r in runs if isinstance(r.get("prompt_tokens"), (int, float))]),
            "requested_setup_models": dict(Counter(r.get("model") for r in runs)),
            "serving_usage_by_provider_model_task": serving,
            "execution_statuses": dict(Counter(r["status"] for r in ledger)),
        })
    return {"profile": "default", "collected_at": now.isoformat(), "window_start": cutoff.isoformat(),
            "config": config, "job_count": len(jobs), "inventory": inventory,
            "counts": {"enabled": sum(j.get("enabled") is True for j in jobs),
                       "no_agent": sum(j.get("no_agent") is True for j in jobs),
                       "enabled_agent": sum(j.get("enabled") is True and j.get("no_agent") is False for j in jobs)},
            "history_rows": {"usage_audit": len(audit_rows), "session_model_usage": len(usage_rows),
                             "executions": len(executions), "malformed_audit_lines": malformed},
            "limitations": ["Audit model is requested setup model, not actual serving identity.",
                            "Whole-run audit duration excludes pre-agent setup and does not isolate Jev, tool or inference time.",
                            "Usage window selects records active in the window; accumulated tokens may cross its start.",
                            "No per-fire joined route outcome or projection/envelope bytes exist in these sources.",
                            "Large input or an Astra usage row alone cannot prove envelope-caused routing.",
                            "Snapshots are not consumed by the inspected scheduler model/provider resolver."]}


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    path.chmod(0o600)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["collect", "propose", "validate"])
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--proposal", type=Path)
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    if args.action == "collect":
        if args.output is None or not 1 <= args.days <= 30:
            parser.error("collect requires output and 1–30 days")
        result = collect(args.home, args.days)
        save_json(args.output, result)
        print(json.dumps({"output": str(args.output.resolve()), "counts": result["counts"], "job_count": result["job_count"]}))
    elif args.action == "propose":
        if args.selection is None or args.output is None:
            parser.error("propose requires selection and output")
        jobs, cfg = read_inputs(args.home)
        result = build_proposal(jobs, cfg["routes"], json.loads(args.selection.read_text()))
        save_json(args.output, result)
        print(json.dumps({"output": str(args.output.resolve()), "eligible_job_ids": result["eligible_job_ids"],
                          "unchanged_count": len(result["unchanged_job_ids"]), "review": result["review"]}))
    else:
        if args.proposal is None:
            parser.error("validate requires proposal")
        jobs, cfg = read_inputs(args.home)
        errors = validate_proposal(json.loads(args.proposal.read_text()), jobs, cfg["routes"])
        print(json.dumps({"valid": not errors, "errors": errors}))
        raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
