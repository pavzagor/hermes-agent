"""Exercise scheduler precedence and host acceptance without inference or delivery."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from cron import scheduler
from agent import turn_routing
from hermes_constants import get_hermes_home


@pytest.mark.parametrize("model,cron_model,snapshot,expected", [
    ("intentional-astra", "fleet-sol", "old-snapshot", "intentional-astra"),
    (None, "fleet-sol", "old-snapshot", "fleet-sol"),
    (None, None, "old-snapshot", "main-astra"),
    (None, None, None, "main-astra"),
])
def test_real_scheduler_model_selection_ignores_nonpin_snapshots(
    model: str | None, cron_model: str | None, snapshot: str | None, expected: str,
) -> None:
    home = get_hermes_home()
    cfg = {"model": {"default": "main-astra", "provider": "openai-codex"},
           "cron": {"model": cron_model, "model_provider": "openai-codex"}}
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))
    job = {"id": "public-fixture", "model": model, "model_snapshot": snapshot}
    selected = scheduler._load_cron_job_config(job, "public-fixture", "public fixture")
    assert selected.model == expected
    assert selected.cron_default_provider == "openai-codex"


def test_scheduler_constructor_propagates_job_pin_to_turn_routing() -> None:
    home = get_hermes_home()
    cfg = {"turn_routing": {"enabled": True, "routes": {
        "sol": {"provider": "openai-codex", "model": "public-sol"}}}}
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))
    job = {"id": "public-pin", "model": "public-astra", "provider": "openai-codex",
           "no_agent": False, "enabled_toolsets": []}
    setup = scheduler._CronAgentSetup(model="public-astra", runtime={"provider": "openai-codex"})

    class ConstructorCapture:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.session_id = kwargs["session_id"]
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]

    agent = scheduler._construct_cron_agent(ConstructorCapture, job, cfg, setup,
                                           workdir=None, session_id="cron_public_pin", session_db=None)
    assert scheduler._job_route_pinned(job) is True
    assert agent.model == "public-astra"
    # Explicit job identity must win at the real host acceptance boundary.
    assert turn_routing.routing_enabled(agent.session_id, is_first_turn=True) is False
    assert turn_routing.apply_route(agent, {"provider": "openai-codex", "model": "public-sol"},
                                    is_first_turn=True) is False
    assert agent.model == "public-astra"
