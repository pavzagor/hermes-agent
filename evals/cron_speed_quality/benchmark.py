"""Bounded genuine Codex comparisons on fixed public tasks; no agent/tool loop."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.cron_speed_quality.audit import PUBLIC_PROMPTS, canonical, check_answer, read_inputs, save_json

INSTRUCTIONS = "Complete the supplied public test task. Do not call tools, send messages, change data, or claim external actions. Follow exact output requirements."


def request_for_task(name: str, tier: str, routes: dict[str, Any]) -> dict[str, Any]:
    if name not in PUBLIC_PROMPTS or tier not in {"sol", "astra"}:
        raise ValueError("Only allowlisted public tasks and Sol/Astra tiers may be tested")
    route = routes.get(tier)
    if not isinstance(route, dict) or route.get("provider") != "openai-codex" or not isinstance(route.get("model"), str):
        raise ValueError("Exact authorized Codex route is required")
    return {"model": route["model"], "instructions": INSTRUCTIONS,
            "input": [{"role": "user", "content": PUBLIC_PROMPTS[name]}],
            "store": False, "tools": [], "reasoning": {"effort": "low"},
            "prompt_cache_key": hashlib.sha256(canonical({"instructions": INSTRUCTIONS, "task": name})).hexdigest()}


def answer_from_stream(events: list[dict[str, Any]], final_text: str) -> str:
    """The live Codex SDK may leave final.output_text empty despite real text deltas."""
    return final_text or "".join(event["delta"] for event in events
                                 if event.get("type") == "response.output_text.delta" and isinstance(event.get("delta"), str))


def deadline_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("Public probe wall-clock deadline reached")


def run(home: Path, output: Path, repeats: int = 2) -> dict[str, Any]:
    if not 1 <= repeats <= 2:
        raise ValueError("At most two passes are authorized by this harness")
    os.environ["HERMES_DISABLE_LAZY_INSTALLS"] = "1"
    _, config = read_inputs(home)
    routes = config["routes"]
    for tier in ("sol", "astra"):
        request_for_task("no_change", tier, routes)
    from agent.auxiliary_client import resolve_provider_client
    from contextlib import ExitStack
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    clients: dict[str, Any] = {}
    resolutions: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    previous_handler = signal.signal(signal.SIGALRM, deadline_handler)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ExitStack() as scope:
            token = set_hermes_home_override(home)
            scope.callback(reset_hermes_home_override, token)
            for tier in ("sol", "astra"):
                started = time.perf_counter()
                signal.alarm(75)
                try:
                    client, resolved = resolve_provider_client("openai-codex", model=routes[tier]["model"], raw_codex=True)
                    if client is None or resolved != routes[tier]["model"]:
                        raise RuntimeError("Exact model resolution unavailable")
                    clients[tier] = client.with_options(max_retries=0, timeout=65.0)
                    resolutions[tier] = {"ok": True, "resolved_model": resolved,
                                         "resolve_ms": (time.perf_counter() - started) * 1000}
                except Exception as exc:
                    resolutions[tier] = {"ok": False, "error_type": type(exc).__name__,
                                         "resolve_ms": (time.perf_counter() - started) * 1000}
                finally:
                    signal.alarm(0)
            for pass_index in range(repeats):
                order = ("sol", "astra") if pass_index == 0 else ("astra", "sol")
                for name in PUBLIC_PROMPTS:
                    for tier in order:
                        request = request_for_task(name, tier, routes)
                        receipt: dict[str, Any] = {
                            "task": name, "tier": tier, "pass": pass_index + 1,
                            "requested_model": request["model"], "requested_provider": routes[tier]["provider"],
                            "prompt_sha256": hashlib.sha256(canonical({k: v for k, v in request.items() if k != "model"})).hexdigest(),
                            "condition": "first_task_request" if pass_index == 0 else "repeated_task_same_client",
                            "server_cold_state": "unknown", "actual_serving_model": None,
                            "passed": False, "tools_disabled": True, "fallback_enabled": False,
                        }
                        if tier not in clients:
                            receipt["failure"] = "resolution_unavailable"
                            results.append(receipt)
                            continue
                        started = time.perf_counter()
                        first_event: float | None = None
                        first_text: float | None = None
                        public_events: list[dict[str, Any]] = []
                        signal.alarm(75)
                        try:
                            with clients[tier].responses.stream(**request) as stream:
                                for event in stream:
                                    elapsed = (time.perf_counter() - started) * 1000
                                    if first_event is None:
                                        first_event = elapsed
                                    if event.type == "response.output_text.delta":
                                        public_events.append({"type": event.type, "delta": event.delta})
                                        if first_text is None:
                                            first_text = elapsed
                                response = stream.get_final_response()
                            text = answer_from_stream(public_events, response.output_text or "")
                            usage = response.usage
                            usage_counts = ({"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                                             "cached_input_tokens": getattr(usage.input_tokens_details, "cached_tokens", None),
                                             "reasoning_tokens": getattr(usage.output_tokens_details, "reasoning_tokens", None)}
                                            if usage else None)
                            receipt.update({"actual_serving_model": response.model, "response_status": response.status,
                                            "public_answer": text, "public_text_events": public_events, "usage": usage_counts,
                                            "passed": response.model == request["model"] and response.status == "completed" and check_answer(name, text)})
                            if not receipt["passed"]:
                                receipt["failure"] = "model_status_or_correctness_check_failed"
                        except Exception as exc:
                            receipt.update({"failure": type(exc).__name__, "http_status": getattr(exc, "status_code", None)})
                        finally:
                            signal.alarm(0)
                            receipt.update({"inference_ms": (time.perf_counter() - started) * 1000,
                                            "first_event_ms": first_event, "first_text_ms": first_text})
                        results.append(receipt)
                        save_json(output, {"profile": "default", "resolutions": resolutions, "results": results,
                                           "complete": False, "recorded_at": datetime.now(timezone.utc).isoformat()})
                        print(json.dumps({k: receipt.get(k) for k in ["task", "tier", "pass", "actual_serving_model", "passed", "inference_ms", "failure"]}), flush=True)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        for client in clients.values():
            client.close()
    evidence = {"profile": "default", "resolutions": resolutions, "results": results,
                "complete": True, "recorded_at": datetime.now(timezone.utc).isoformat(),
                "limitations": ["Short public prompts do not test live cron tool execution, task auth, browser or delivery.",
                                "First request is not proof of a cold provider/cache; repeat is not proof of cache hit.",
                                "Low effort is held constant only for these probes; no live reasoning setting changed.",
                                "Two passes per task/model cannot establish a speed or quality distribution.",
                                "No Jev or live system prompt is used in this direct-provider microbenchmark."]}
    save_json(output, evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    result = run(args.home, args.output, args.repeats)
    raise SystemExit(0 if len(result["results"]) == 2 * len(PUBLIC_PROMPTS) * args.repeats
                     and all(receipt["passed"] for receipt in result["results"]) else 1)


if __name__ == "__main__":
    main()
