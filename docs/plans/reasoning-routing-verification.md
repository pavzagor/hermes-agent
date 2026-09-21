# Reasoning routing verification

Base: `cf908be1f4624caad38485bb74e39994f614f4dd`.
Implementation worktree: `/Users/pavelzagorodnikh/.hermes/effort-routing`.
Unchanged comparison worktree: `/Users/pavelzagorodnikh/.hermes/effort-routing-base`.
No install, production configuration write, network model probe or restart.

## Entry-point audit

- Direct `AIAgent`: explicit reasoning dict defaults to request provenance; resolved configuration callers mark `config`. No value-difference inference.
- CLI constructor/oneshot: `--reasoning` is request provenance; global/per-model defaults are config. Slash selections and model picker effort are session pins. Background construction forwards provenance.
- CLI/TUI `--once`: snapshot includes provenance and routing sidecar, including cold CLI state without a live agent.
- Gateway: config construction is marked config; per-turn session override gets session provenance. `--once --reasoning` is stored only on the temporary model override and wired as request provenance. Reset/auto clears the durable pin.
- API: structured `reasoning` and legacy `reasoning_effort` are request provenance; absent fields use config. Raw request wire overrides block effort automation.
- TUI/Desktop: constructor overrides are session selections; runtime persistence includes provenance so automatic values cannot become pins on resume. Config values are re-resolved. Display commands do not pin.
- Compression: the existing routing-state migration carries the independent reasoning pin. Automatic effort is not stored in that sidecar.
- Model switch, fallback activation and primary restoration retain explicit effort. Config/automatic effort is resolved for the destination model.

## Execution evidence

Canonical runner: `scripts/run_tests.sh` (isolated HERMES_HOME, clean env).

- Final focused run: **196 passed, 0 failed**, 19 files; `/tmp/effort-green.log`.
- Expanded regression rerun: **749 passed, 1 failed**, 9 files; `/tmp/effort-final-focused.log`. The remaining failure, `test_model_options_preserves_canonical_custom_row_after_agent_init`, also fails on the unchanged base.
- Broad run during implementation: **31,722 passed, 62 failed, 282 skipped**, 2,918 files across agent/CLI/gateway/TUI; `/tmp/effort-routing-tests.log`. This was not a final all-green run. All newly introduced failures from that run were repaired and their files rerun. Other failing files were compared on the unchanged base: `/tmp/effort-baseline-tests.log` and `/tmp/effort-baseline-extra.log`. Missing Anthropic dependencies and existing platform/config/update failures remain outside this change. Scheduler and auxiliary-stream timing failures passed on isolated reruns; auxiliary rerun had 205 passed and one base-reproduced missing-Anthropic failure.
- Real AIAgent tests use real runtime model switching, fallback resolution, Codex transport and OpenAI SDK serialization, replacing HTTP/tool I/O only. They exercise two tool rounds plus final answer, assert every request's effort, strip `-900k` on the actual wire, and preserve system instructions/cache key.
- Other tests cover default-off policy, all six supported model/alias forms at all four allowed levels, malformed/unsupported directives, protected per-model overrides, request/session pins equal to configured defaults, failed hook baseline reset, cold reconstruction, compression state carry, legacy hook signatures, auto/show/hide, and once restoration.

## Activation (owner decision, not performed)

Set `reasoning_routing.enabled: true` in the intended profile after deploying the commit. Independently enable the consuming plugin's `effort_enabled` option. No `turn_routing.enabled` dependency exists. Existing per-model reasoning overrides remain protected; to automate them deliberately, add exact model IDs under:

```yaml
reasoning_routing:
  enabled: true
  override_models:
    openai-codex:
      - gpt-6-astra-900k
      - gpt-5.6-sol-900k
      - gpt-5.6-luna-900k
```

Only list aliases actually used in the profile. Explicit session/request choices still win; `/reasoning auto` clears the session pin without enabling automation. Plugin contract: optional context booleans `reasoning_routing_enabled` / `reasoning_pinned`, result top-level `reasoning: {effort: low|medium|high|xhigh}`; model route schema unchanged.
