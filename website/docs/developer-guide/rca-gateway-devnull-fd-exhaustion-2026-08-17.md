# RCA: gateway `/dev/null` descriptor exhaustion on 2026-08-17

**Status:** resolved by loading the fix already shipped in v2026.8.16

**Impact:** the local gateway stayed alive but became unable to open files,
SQLite databases, sockets, and provider connections.

## Summary

Hermes did not kill itself. The launchd-managed gateway process (PID 995) had
been running continuously since 2026-08-10 17:44 MSK and exhausted its macOS
soft limit of 256 file descriptors. At diagnosis it held 253 numeric file
descriptors. The largest class was 131 separate writable handles to
`/dev/null`; `state.db` and `state.db-wal` accounted for another 54.

Once the table filled, unrelated operations failed with `OSError: [Errno 24]
Too many open files`. That surfaced as auth-file read failures, browser and
sandbox failures, Matrix delivery failures, provider connection errors, and
`sqlite3.OperationalError: unable to open database file`. Because the gateway
process was still running, launchd had no exit to detect and restart.

The checkout had been advanced to v2026.8.16 at 2026-08-17 11:32 MSK, but the
week-old interpreter was still executing its previously imported code. The
tagged release already contains commit `3d73821e9d` (`fix(agent): stop thread
output descriptor leaks`), committed after PID 995 started. Restarting the job
loaded that repair.

## Timeline

- **2026-08-10 17:44 MSK:** PID 995 started under
  `com.hermes.gateway.op`.
- **2026-08-15:** upstream committed `3d73821e9d`, bounding thread-output
  sinks to one per routed stream.
- **2026-08-17 11:32 MSK:** the checkout moved to the v2026.8.16 merge, but
  PID 995 was not restarted.
- **2026-08-17 18:23 MSK:** the first retained timestamped `EMFILE` evidence
  appears in `errors.log`; failures then spread across database, auth,
  delivery, browser, and provider paths.
- **2026-08-17 19:57 MSK:** launchd was force-restarted to recover a process
  that could no longer perform its own normal drain/restart path.
- **2026-08-17 19:58 MSK:** replacement PID 44993 served both health endpoints
  as Hermes v0.20.2 with 46 numeric descriptors and one `/dev/null` handle.

The `SIGKILL`/unclean-exit record emitted at 19:57 was caused by the deliberate
`launchctl kickstart -k` recovery. It was not the initiating failure.

## Root cause

The runtime-era implementation of `agent.thread_scoped_output` installed
per-thread stdout/stderr routing proxies. When another context temporarily
replaced process-global `sys.stdout` or `sys.stderr`, `_ensure_installed()`
created a new routing proxy and opened a new `/dev/null` sink. Restored older
proxies retained their sinks, so repeated background review/output-redirection
cycles accumulated permanent descriptors.

A controlled reproduction against the runtime-era module at `d8441e16e9`
performed 20 temporary stdout/stderr redirect cycles. It opened 44 sinks where
the bounded contract permits two. The same scenario is pinned by
`test_temporary_global_redirects_do_not_allocate_new_sinks` in the current
test suite.

SQLite handles were a secondary contributor, not the primary leak in this
incident. The live process held 32 `state.db` and 22 `state.db-wal` handles,
but the 131 `/dev/null` handles dominated the descriptor table. v2026.8.16 also
contains bounded read-pool and abandoned-`SessionDB` lifecycle repairs; the
replacement process started with four handles to each SQLite file.

## Durable fix

Commit `3d73821e9d` changes the routing layer to keep one process-lifetime sink
per stream and share routing state across proxy generations. If a temporary
redirect restores an older routing proxy, Hermes adopts that proxy instead of
wrapping it again. The fix is included in v2026.8.16, so no duplicate code
change was necessary during this incident.

Operational recovery was:

```text
launchctl kickstart -k gui/501/com.hermes.gateway.op
```

Future source updates that change long-lived gateway code must be followed by
a gateway restart; updating the checkout alone does not alter modules already
loaded in the running interpreter.

## Verification

- Runtime-era reproduction: **failed as expected**, opening 44 sinks for a
  contract of two.
- Focused output/SQLite lifecycle regression set — **32 passed**.
- Ruff on the exercised production modules and regression files — **passed**.
- Real-descriptor probe over 200 temporary stdout/stderr redirect cycles —
  **+2 descriptors**, the two bounded process-lifetime sinks.
- launchd: replacement PID 44993 is **running**.
- `hermes status`: gateway is **running** under launchd.
- `http://127.0.0.1:8644/health`: `{"status":"ok","platform":"webhook"}`.
- `http://127.0.0.1:8642/health`:
  `{"status":"ok","platform":"hermes-agent","version":"0.20.2"}`.
- Descriptor snapshot after restart: **46 numeric**, **1 `/dev/null`**, four
  `state.db`, and four `state.db-wal` handles, versus 253/131/32/22 before.
- Post-start check after 19 gateway forks: **48 numeric**, still **1
  `/dev/null`**, with zero new `EMFILE`, database-open, or read-connection-close
  errors after the restart marker.

## Follow-up

- Treat process age plus checkout revision as part of update verification;
  source-on-disk being fixed is insufficient for a long-lived Python service.
- Consider exposing the loaded Git revision in `hermes status` or the health
  response so code/runtime drift is directly observable.
- Keep total-FD and path-class probes in gateway incident diagnostics. A raw
  `EMFILE` error identifies exhaustion, but descriptor composition is what
  distinguishes `/dev/null`, SQLite, socket, and subprocess leaks.
