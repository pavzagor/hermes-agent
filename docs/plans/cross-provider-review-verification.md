# Cross-provider routing review verification

## Implemented

- Successful ACP manual model rebuilds pin routing; failed rebuilds leave prior provenance untouched.
- Exact named/custom identities survive runtime resolution through the generic `custom` transport. The requested identity, not the shared transport label, is compared with the allowlisted provider.
- Automatic switches/restores retain existing system-prompt bytes across providers. Destination client/capability/context-window changes remain active; prompt rebuilding is not required for transport adaptation.
- Image-token cost is rebound after a successful route switch, before downstream estimates and compaction.
- Updated same-provider-only documentation.

## Evidence

Canonical runner: `scripts/run_tests.sh` using the existing main-checkout venv.

- Red: routing + ACP tests: 24 passed, 7 failed, demonstrating the named/custom rejection, both prompt invalidations, stale image cost on apply/restore, and missing ACP pin.
- Green focused routing/surface tests: 53 passed before the additional cross-provider two-turn and warm-cache regressions.
- Expanded run over 19 files: 101 passed, 2 failed. Both failures are in `tests/agent/test_switch_model_reasoning_override.py`; rerunning that exact file on unchanged `dcdb2a49d8` reproduced both failures. The restore failure reports an installed SDK `httpx.Timeout` versus `httpx2.Timeout` incompatibility; the other test swallows client construction exceptions and then lacks a primary runtime snapshot.
- Real AIAgent loop test exercises same- and cross-provider routes through a tool loop and a second user turn, asserting identical system-prefix messages. Remote model/tool I/O is stubbed.
- Gateway test invokes the real cache signature and `_resolve_turn_agent` / `_lookup_cached_agent` path after an automatic cross-provider switch, forbidding reconstruction and verifying the same agent and prefix survive.
- Existing API/manual-pin, TUI reopen, gateway override persistence, switch rollback/context/capability/credential tests retained and exercised. No wholesale historical test restoration.

## Unresolved independent reasoning routing

The supplied starting tree does **not** contain `agent/turn_reasoning.py` or the `test_turn_reasoning*` tests. Historical commits `68adfca86f` and `8bdd6e0381` contain that separate feature, but it is absent from this branch, rather than regressed by this review patch. Its preservation therefore cannot be certified. Reconcile that feature separately against current upstream before claiming parity with the earlier local stack; do not silently restore old whole files over newer code/tests.

No live installation, profile configuration, plugin checkout edits, or restarts were performed. The full repository suite and authenticated cross-provider serving were not exercised here.
