# Jev turn routing Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Once-per-user-turn opt-in model routing plus full multi-skill loading, fail-open and cache-safe.

**Architecture:** Standalone Jev plugin, not vendor code in core. Extend the existing once-per-turn pre_llm_call return contract with an allowlisted same-provider model directive. Core owns switches and manual pins; plugin owns inference, privacy filtering and skill selection. Existing sessions remain pinned until explicitly enabled.

**Tech Stack:** Python stdlib, existing Hermes plugin hooks, local Connect wrapper, pytest canonical runner.

## Tasks
1. Inspect real runtime and supported plugin boundaries; record config route IDs (no credentials).
2. Write behavioral tests for generic directive allowlist, cache stability, pin controls and failure. Run canonical runner RED. Implement narrow core helper and connect at turn-start, before compaction. Never switch from an iteration hook.
3. Add persistent manual pin on successful manual switch; expose explicit routing control, preserving existing sessions by default. Cover manual re-selection and cold gateway overrides.
4. Create standalone plugin repository. Tests first for typed payload, confidence fallback, multi-skill loading, secret redaction, overflow and transport failure. Fetch credential only in process via local Connect; no retries, no raw exception logging.
5. Route against all text task context; exclude system memories, raw reasoning, credentials and multimodal data. Conservatively refuse unsupported/overbudget input rather than silently truncate evidence. Send every skill as independent Noul question; load all positive/relevant selections, preserving mandatory discovery instructions.
6. Execute authenticated inference with synthetic tasks using exact production client. Run focused core/plugin integration and canonical suite. Record actual outputs, not benchmark estimates.
7. Commit artifacts. Document staged activation/rollback; no live changes/restart. Parent independently reviews before activation.

## Verification
`scripts/run_tests.sh tests/agent/test_turn_routing.py`; adjacent pre_llm_call, model switch and gateway model tests; `scripts/run_tests.sh`; `git diff --check`; production smoke via standalone plugin client. Full suite failures must be disclosed and compared to unchanged base where practical.
