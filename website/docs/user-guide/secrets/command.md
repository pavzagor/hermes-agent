# Command Helper Secret Source

Resolve credentials by running your own helper command at startup — any secret store with a CLI works: `keepassxc-cli`, `secret-tool` (GNOME Keyring), `pass`, `gpg`, Vaultwarden's CLI, or a script that cats a tmpfs env file. The helper prints `KEY=VALUE` lines on stdout; Hermes applies them through the same orchestrator as [Bitwarden](./bitwarden) and [1Password](./onepassword), so you can enable any combination of sources simultaneously.

## How it works

1. You configure a helper command in `config.yaml` (never in `.env` — the command is configuration, `.env` holds values).
2. At startup, after `.env` loads, Hermes runs the helper ONCE via `/bin/sh -c` and parses its stdout as a dotenv blob.
3. The parsed keys flow through the standard precedence ladder: `.env`/shell win unless `override_existing: true`; mapped sources beat this bulk source on contested vars; first claim wins.

```yaml
secrets:
  command:
    enabled: true
    command: "cat /run/user/1000/hermes-secrets.env"
    # or any vault CLI that dumps KEY=VALUE lines:
    # command: "pass show hermes/env"
    # command: "secret-tool lookup service hermes-env"
```

## Config

| Key | Default | What it does |
|---|---|---|
| `enabled` | `false` | Master switch. |
| `command` | `""` | Helper run via `/bin/sh -c`; must print `KEY=VALUE` lines on stdout. |
| `helper_timeout_seconds` | `3` | Hard timeout for one helper run. Deliberately tight — the helper must be fast and NON-interactive (no unlock prompts, no touch/PIN). |
| `override_existing` | `false` | Helper values overwrite `.env`/shell values. Off by default (unlike Bitwarden/1Password) since a local helper is not a central rotation authority. |

## Security model

- The helper command string is YOUR configuration — same trust level as the `.env` file you control.
- Output is hard-capped at 1 MiB; a runaway helper can't wedge startup (process group killed on timeout).
- The helper's **stderr is discarded** — vault CLI diagnostics can carry secret material, so they never reach Hermes' output. Failures log structured fields only (exit code / signal / errno), never the command string.
- Whitespace-only values are treated as "no value" — a placeholder entry never flows into an Authorization header.
- POSIX-only (needs `/bin/sh`). On Windows the source reports itself unconfigured and startup continues.

## Failure modes

Startup is never blocked. Errors print one line plus a `→` remediation hint:

| Symptom | Cause | Fix |
|---|---|---|
| `secrets.command.command is empty` | Enabled without a command | Set `secrets.command.command` in config.yaml |
| `helper command failed` | Non-zero exit, timeout, spawn failure | Run the helper manually in a shell to see its real error (Hermes discards its stderr on purpose) |
| `helper output was not a KEY=VALUE map` | Helper printed a bare value or garbage | Make the helper emit dotenv-shaped lines |

## Native browser logins with 1Password Connect and Keychain

The browser vault can use `OP_CONNECT_HOST` and `OP_CONNECT_TOKEN` from this
source without installing `op`, signing in interactively, or copying website
passwords into the local Hermes vault. Connect uses its read-only HTTP API;
`op item list` is not a supported Connect transport.

Keep the Connect token in the existing OS secret store. Configure a fast,
non-interactive helper in **only the owning profile's** `config.yaml`. For a
Keychain helper with the interface `helper get SERVICE ACCOUNT`:

```yaml
secrets:
  command:
    enabled: true
    override_existing: false
    command: >-
      token=$(/absolute/path/to/keychain-helper get CONNECT_SERVICE ACCOUNT) || exit 1;
      test -n "$token" || exit 1;
      printf 'OP_CONNECT_HOST=http://127.0.0.1:8080\nOP_CONNECT_TOKEN=%s\n' "$token"
```

This is configuration for Hermes to consume privately, **not a command to run
in an agent terminal**: stdout contains the Connect token. Use the same scoped
Keychain service/account as the existing Connect wrapper; do not add a second
token, plaintext token file, or broader vault grant. Merge with an existing
command source rather than replacing unrelated secret outputs. Remove conflicting
Connect entries from other sources deliberately: host and token must come from
the same approved connection. Do not set `vault.onepassword.binary_path` to a
Connect wrapper — its command and JSON contracts differ from `op`.

The native backend reads the active profile's secret scope on each request.
Connect takes precedence over service/session auth, and partial Connect settings
fail closed. Requests do not follow redirects, use environment proxies, or send
credentials to non-loopback HTTP. Server error bodies and credentials are not
returned to the model. Handles include the vault and item IDs; filling still
requires an exact saved web origin and the browser's atomic origin check.
A configured token means no interactive unlock is needed, not that server
health or token authorization has already been verified; reads can still fail.
Connect summaries without fields do not advertise a username or automatic 2FA;
item resolution reads authoritative metadata and mints supported TOTP internally.

After an approved code/config activation, start a fresh runtime and use
`browser_vault_list`, then `browser_vault_fill` on an authorized matching page.
Do not test by printing item details or passwords, and do not treat unit tests
as proof of a live account login. No special Keychain auto-discovery runs in
other profiles; each profile must explicitly configure its own secret source.

## When to use this vs a plugin

The command source is the escape hatch for vaults without a bundled integration. If you find yourself wrapping a complex CLI dance in a long script, consider a proper [secret-source plugin](../../developer-guide/secret-source-plugin.md) instead — plugins get caching, provenance labels, and typed config.
