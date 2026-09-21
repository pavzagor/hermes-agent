# Quota and independent reasoning candidate verification

No installation, restart, merge or push. Source changes apply to the isolated
cross-provider candidates; live configuration and installed plugins were untouched.

## Implementation

- Generic host `account_usage_routing.enabled` opt-in, default false. Runtime-only
  Codex usage collection, same-token account/residency headers, singleflight
  background slot, TTL/size bounds, no auth resolution or credential refresh.
- Hooks receive only sanitized plain subscription_usage; the standalone plugin
  no longer accepts/collects runtime credentials. Egress remains validated.
- Historical 68adfca86f + 8bdd6e0381 reconciled as diffs, retaining newer exact-pair
  provider identity, image rebinding and frozen-prefix behavior. Effort applies
  after routing, supports Codex Astra/Sol/Luna only, and preserves explicit pins.
- CLI/API/gateway/TUI provenance and --once restoration; routing writes retain
  reasoning pins. CLI model re-resolution now preserves explicit effort, with
  a regression that failed before the fix.

## Test receipts

`quota-reasoning-verification.json` contains the exact canonical command and output:
35 discovered files, 472 passed, zero failed: 301 targeted core regressions and
all 171 standalone plugin tests. This includes gateway session-override persistence
and ACP model validation at their verified current paths.

Red phase: new host collector tests failed for missing host module (35 failures);
real discovered-plugin lifecycle test failed for missing module; restored historical
reasoning tests failed for missing helper/provenance. Green: these and existing
route/pin/prefix/effort/surface suites pass. HTTP is stubbed, not live inference.
Full core suite was not run.

## SDK mismatch and operational incident

The shared venv contained anthropic 1.7.0 (httpx2); native adapter builds
httpx.Timeout. With HERMES_DISABLE_LAZY_INSTALLS=1, a synthetic-token, no-network
build_anthropic_client call reproduces the production constructor TypeError.
The source lazy-dependency declaration instead pins anthropic==0.87.0 (httpx).
Thus this is a real constructor incompatibility with the installed 1.7.0 SDK,
not a reasoning regression; normal lazy resolution may hide it by installing the
source-pinned version. No compatibility/dependency migration is included.

The two reasoning-switch unit tests incorrectly left native SDK construction
unmocked (and one swallowed all exceptions). They now mock the actual client
builder and no longer swallow failures; both pass while the separate constructor
probe still exposes the version mismatch.

An initial probe accidentally allowed lazy installation and downgraded the shared
Anthropic package to 0.87.0. This was outside scope. It was immediately restored
with `python -m pip install --no-deps anthropic==1.7.0`; version 1.7.0 and the same
constructor failure were then verified with lazy installation disabled. No service
was restarted. Do not treat this as an authorized dependency change or a live
provider validation; all future probes must disable lazy installs explicitly.
