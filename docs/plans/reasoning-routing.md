# Per-turn reasoning routing implementation plan

**Goal:** Independently consume typed effort suggestions while explicit user effort always wins.
**Architecture:** Add a host-owned reasoning policy next to turn_routing; preserve explicit provenance at constructors and runtime switches. Reuse routing sidecar for durable session pins and compression/--once snapshots. Never persist automatic effort or change prompts/tools.
**Tech stack:** Python, pytest, existing lifecycle hooks and Codex transport.

1. Write failing tests in tests/agent/test_turn_reasoning.py for default-off gate, model/effort independence, session/request pins, reset on hook failure, provider/model validation and per-model override protection. Run scripts/run_tests.sh on that file.
2. Implement agent/turn_reasoning.py, wire turn_context after model routing, guard model/fallback config resolution; preserve provenance in runtime snapshots.
3. Trace CLI/oneshot, gateway, API and TUI constructors and commands: distinguish configuration from explicit values, add auto unpin, preserve show/hide semantics and --once restoration.
4. Add typed hook docs and canonical config defaults. Add real AIAgent tool-loop and actual Codex payload tests plus cold recreation coverage.
5. Run canonical focused suites, inspect diff and commit. No production configuration, installation or restart.

Upstream discovery: searched current upstream PRs for reasoning; related #116513 (omitted constructor config), #93378 (per-skill effort), #114033 (supported efforts) are not this independent turn-hook/provenance contract. No wholesale main update.
