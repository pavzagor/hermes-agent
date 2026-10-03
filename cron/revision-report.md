# Explicit cron pins are fixed in the candidate; both Sol proposals remain deferred

Candidate only. Nothing was deployed or activated. Parent Astra owns independent review, integration, and activation.

Worktree: /Users/pavelzagorodnikh/.hermes/plans/jev-speed-quality/cron/worktree
Branch: sol/jev-cron-audit
Starting commit: 54b2f64fffe9fb501ce02f8f0ed93aae3016a81b
Implementation/evidence commit: efc99df61d4862b6b46b6fa5d5c1bddb7d62f3dc
This report and the final rerun receipt are committed separately; the final report commit is available in git log.

The starting branch and HEAD were verified. Both staged and unstaged diffs were empty before work. Existing implementation was extended, not rebuilt. Relevant routing/Hermes skills and the typed-turn-router review reference were read.

## The scheduler now publishes explicit job identity to host acceptance

cron/scheduler.py::_construct_cron_agent now retains the constructed agent, calls the existing host pin_model(session_id) only when _job_route_pinned(job) is true, and returns the agent. It pins after successful construction and before the first pre_llm_call hook. Construction failure publishes no pin.

An authored nonblank model, provider, or base_url field pins the resolved runtime identity. Inherited cron/main defaults and legacy snapshot fields do not. No blanket turn_routing disable was added. No classifier or generic route-choice policy was changed. All serving constructor arguments, toolset rules, approvals, credentials, memory behavior, prompt loading, reasoning configuration, fallback selection, and continuation/completeness budgets remain as they were in the candidate.

The host pin is durable, session-scoped routing state, using the existing host pin_model API. Agent teardown closes resources; it deliberately does not erase the pin and thereby make a rebuilt historical session routable. Other sessions still route. Failed pin persistence preserves the host's existing in-memory authoritative pin and the session's manual reasoning pin.

Real AIAgent tests use synthetic credentials, an unreachable loopback endpoint, tools disabled through the existing no_mcp sentinel, isolated test homes, and a socket-connect prohibition. No inference or account integration runs. Tests exercise actual scheduler construction, host apply_route, real pre_llm_call consumption, canonical skill-loader consumption with synthetic skill content, independent effort acceptance, prefix stability, real client close, and pinned -> unpinned -> pinned session isolation. Only the classifier result and synthetic skill content are substituted; route acceptance and agent model switching are real.

Covered forms: model-only, provider-only, endpoint-only, combined provider/model, null/missing/blank fields, ignored snapshots, failed construction, and failed persistence. The effort fixture uses IDs supported by the candidate host's existing capability gate; it does not change a model catalog or live route configuration.

## Proposal ID sets are reconciled against the exact source

validate_proposal requires eligible_job_ids and unchanged_job_ids to be lists of unique nonempty source IDs. It rejects unknown IDs, malformed entries, duplicate source IDs, overlaps, missing IDs, misclassified sets, missing changes, and changes outside the declared eligible set. Eligible IDs must equal the exact changed source IDs. Unchanged IDs must equal every remaining source ID. A fully accounted no-op remains valid.

Existing stale-settings fingerprints, disabled/no_agent and intentional-pin exclusions, exact authorized-pair checks, rollback equality, and evidence requirements remain. Inputs are not mutated. The existing external proposal validated successfully against the read-only live source with the revised structural validator; structural validity is not approval to apply it.

## The timezone-corrected read-only baseline was recollected

File: cron/baseline-revision.json
Collected at: 2026-10-03T10:43:45.903623+00:00
Window start: 2026-09-26T10:43:45.903623+00:00
SHA256: fc4d7f16c65147549fca3bbf00b07c9e747d8253d07a2cbd346730e09b17c435
Local mode verified: 0600.

51 jobs; 42 enabled; 22 no_agent; 21 enabled agent jobs. Of enabled agent jobs, five have explicit pins and 16 inherit defaults. Current history selection: 814 audit rows, 814 usage rows, 1000 execution rows, zero malformed audit lines. The 1000 selected execution rows are completed. The collector uses the previously staged SQLite julianday comparison; the timezone boundary regression passed on this final code tree.

The older REPORT.txt execution count is not a valid comparison baseline. The rolling window moved, so 815 -> 814 audit/usage rows is not an optimization result. Usage records are selected by last_seen and contain accumulated counters that can cross the window start. There is still no per-fire joined Jev decision, host acceptance, wire route, envelope/projection size, or component latency attribution. Envelope-caused routing remains not_established.

The output was checked for prompt/message/history/credential/error-output keys; none were present. Inventory IDs were counted and deduplicated programmatically. No raw job prompt, transcript, account response, or secret was exported. A subsequent read-only comparison found no authored-settings drift since collection. Both proposal targets still had model=null and provider=null.

Read-only collection command:

    HERMES_DISABLE_LAZY_INSTALLS=1 /Users/pavelzagorodnikh/.hermes/hermes-agent/venv/bin/python -m evals.cron_speed_quality.audit collect --home /Users/pavelzagorodnikh/.hermes --output cron/baseline-revision.json --days 7

## Neither proposed Sol pin should be applied

cron/pin-proposals-deferred.json explicitly records apply_allowed=false and deferred_pending_task_specific_evidence for 27cfa82397e1 and 4de19e2dfa5c. The original external proposal/selection/REPORT.txt are retained as historical artifacts, not edited or converted. This decision supersedes their conditional pin recommendation.

The initial tiny benchmark did NOT show Sol faster: both models passed six public checks, while the pooled medians were Sol 3243.47 ms and Astra 2312.20 ms. That sample is too small and too unlike actual jobs to justify the two pins. Public probes are not real job correctness proof. No new provider benchmark or real job was executed in this revision. Both jobs retain their existing model/provider fields.

## Red/green evidence and final-tree affected suites

1. Converted the old pin-gap characterization into the desired regression. RED: one failed, four passed; routing incorrectly remained enabled. GREEN after the narrow scheduler change: five passed.
2. Added 14 scope-set mutation cases. RED: all 14 failed because invalid scope sets were accepted. GREEN with reconciliation: all passed.
3. Added malformed-source-ID coverage. RED: an unhashable source ID raised TypeError rather than being rejected. GREEN with source-ID validation: all passed.
4. Real-host sensitivity check: temporarily removed only the new pin call in this candidate. All four authored-pin forms failed because the real host switched to the proposed model. Restored the change; the complete real-host file passed all ten cases.
5. Initial host fixture setup failed because enabled_toolsets=[] follows cron defaults. Corrected the test setup to the existing no_mcp sentinel and asserted that agent.tools is empty. This was a fixture correction, not a toolset-policy change.

Final canonical affected suite: 142 files, 1504 passed, zero failed, nine skipped, 100% complete. The skipped cases are host-specific Linux/Windows tests on macOS. The final rerun receipt is cron/revision-tests.log. Canonical runner selected the required shared interpreter, /Users/pavelzagorodnikh/.hermes/hermes-agent/venv/bin/python (Python 3.11.16). No dependency install was performed.

Command from this candidate:

    HERMES_DISABLE_LAZY_INSTALLS=1 HERMES_TEST_FILE_RETRIES=0 scripts/run_tests.sh tests/evals/test_cron_speed_quality.py tests/evals/test_cron_routing_characterization.py tests/evals/test_cron_job_pin_host.py tests/cron/ tests/agent/test_turn_routing.py tests/agent/test_turn_reasoning.py

This is affected-suite evidence, not the repository-wide suite. git diff --check passed before the scoped commit.

## Astra must reconcile the shared scheduler boundary and remaining proof gaps

Overlap with the router worker: cron/scheduler.py is shared. This worker changes only _construct_cron_agent: return AIAgent(...) becomes construct -> publish explicit pin -> return. The router worker's scheduler changes are in another worktree. They were not edited or imported here. Integration must keep pin publication before host turn hooks, distinguish authored pins from inherited defaults, retain skill discovery and independent effort routing, and preserve all serving constructor arguments. Do not infer that cherry-pick order alone resolves the overlap.

Open gaps:

- No task-specific inference replays prove blocked/missing source != [SILENT] for either real job. The offline host fixtures prove routing, effort, skill loading, and cleanup, not model judgment about unavailable sources. A mocked final answer would not close that gap, so none is presented as correctness evidence. Adding real semantic host replays was deferred rather than widening this revision into a provider/integration exercise.
- No live gateway-owned verification, naturally scheduled post-activation observation, or latency improvement is claimed. No new projection/envelope/timing instrumentation was added.
- Full-repository and non-macOS test lanes were not run.
- Parent Astra must review the integrated router/cron tree and recheck concurrent production drift before any separately authorized activation. Both Sol pin proposals remain deferred, not queued for automatic application.

No production edits, job/model setting changes, other-profile changes, restarts, pushes, Slack APIs, WACLI operations, live task executions, or account integration actions were performed.

## Exact files changed relative to 54b2f64fff

- cron/scheduler.py
- evals/cron_speed_quality/audit.py
- tests/evals/test_cron_routing_characterization.py
- tests/evals/test_cron_speed_quality.py
- tests/evals/test_cron_job_pin_host.py
- cron/baseline-revision.json
- cron/pin-proposals-deferred.json
- cron/revision-tests.log
- cron/revision-report.md
