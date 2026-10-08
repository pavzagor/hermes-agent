# Hindsight Memory Provider

Long-term memory with knowledge graph, entity resolution, and multi-strategy retrieval. Supports cloud, local embedded, and local external modes.

## Requirements

- **Cloud:** API key from [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io)
- **Local Embedded:** API key for a supported LLM provider (OpenAI, Anthropic, Gemini, Groq, OpenRouter, MiniMax, Ollama, or any OpenAI-compatible endpoint). Embeddings and reranking run locally — no additional API keys needed.
- **Local External:** A running Hindsight instance (Docker or self-hosted) reachable over HTTP.

## Setup

```bash
hermes memory setup    # select "hindsight"
```

The setup wizard installs dependencies automatically via `uv`, walks you through configuration, and offers to seed the bank with a **starter memory template** (a curated set of dispositions/instructions for common agent roles) — you can skip it, and it warns before overwriting an already-configured bank.

Or manually (cloud mode with defaults):
```bash
hermes config set memory.provider hindsight
echo "HINDSIGHT_API_KEY=your-key" >> ~/.hermes/.env
```

### Cloud

Connects to the Hindsight Cloud API. Requires an API key from [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io).

### Local Embedded

Hermes spins up a local Hindsight daemon with built-in PostgreSQL. Requires an LLM API key for memory extraction and synthesis. The daemon starts automatically in the background on first use and stops after 5 minutes of inactivity.

Supports any OpenAI-compatible LLM endpoint (llama.cpp, vLLM, LM Studio, etc.) — pick `openai_compatible` as the provider and enter the base URL.

Daemon startup logs: `~/.hermes/logs/hindsight-embed.log`
Daemon runtime logs: `~/.hindsight/profiles/<profile>.log`

To open the Hindsight web UI (local embedded mode only):
```bash
hindsight-embed -p hermes ui start
```

### Local External

Points the plugin at an existing Hindsight instance you're already running (Docker, self-hosted, etc.). No daemon management — just a URL and an optional API key.

## Config

Config file: `~/.hermes/hindsight/config.json`

### Connection

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `cloud` | `cloud`, `local_embedded`, or `local_external` |
| `api_url` | `https://api.hindsight.vectorize.io` | API URL (cloud and local_external modes) |

### Memory Bank

| Key | Default | Description |
|-----|---------|-------------|
| `bank_id` | `hermes` | Memory bank name (static fallback used when `bank_id_template` is unset or resolves empty) |
| `bank_id_template` | — | Optional template to derive the bank name dynamically. Placeholders: `{profile}`, `{workspace}`, `{platform}`, `{user}`, `{session}`. Example: `hermes-{profile}` isolates memory per active Hermes profile. Empty placeholders collapse cleanly (e.g. `hermes-{user}` with no user becomes `hermes`). |
| `bank_mission` | — | Reflect mission (identity/framing for reflect reasoning). Applied via Banks API. |
| `bank_retain_mission` | — | Retain mission (steers what gets extracted). Applied via Banks API. |

### Recall

| Key | Default | Description |
|-----|---------|-------------|
| `recall_budget` | `mid` | Recall thoroughness: `low` / `mid` / `high` |
| `recall_selection_policy` | `legacy` | `legacy` preserves existing behavior; opt-in `observation_first` uses one mixed recall, native observation preference when supported, a recent-raw safety slot and compact source-aware grouping. Overrides `recall_types` for recall, not reflect. Invalid values are rejected. |
| `recall_prefetch_method` | `recall` | Auto-recall method: `recall` (raw facts) or `reflect` (LLM synthesis) |
| `recall_max_tokens` | `4096` | Maximum tokens for recall results |
| `recall_max_input_chars` | `800` | Maximum input query length for auto-recall |
| `recall_prompt_preamble` | — | Custom preamble for recalled memories in context |
| `recall_tags` | — | Tags to filter when searching memories |
| `recall_tags_match` | `any` | Tag matching mode: `any` / `all` / `any_strict` / `all_strict` |
| `recall_types` | `observation` | Fact types surfaced by recall (both auto-recall and the `hindsight_recall` tool). Comma-separated string or JSON list. **Default narrowed to `observation` only** (see "Behavior change" below). Set to `observation,world,experience` to also include raw facts. |
| `auto_recall` | `true` | Automatically recall memories before each turn |
| `recall_sync` | `false` | Recall synchronously against the *current* message each turn (higher relevance, adds recall latency). Default off: recall runs in the background and is injected on the next turn. |
| `recall_indicator` | `true` | Show a `👁️ Hindsight — recalled N memories` status line when auto-recall injects memory. Turn off for customer-facing agents. |

> **Legacy policy — `recall_types` defaults to `observation` only.**
>
> Previously recall returned all three fact types. It now returns only observations.
>
> Per [Hindsight's docs](https://hindsight.vectorize.io/developer/observations), observations are the **consolidated** knowledge layer Hindsight builds on top of raw facts: deduplicated beliefs grounded in evidence, refined as new facts arrive, with proof counts and freshness signals. Raw `world` / `experience` facts are the individual supporting evidence that feeds them. For per-turn context injection, observations are denser per token and avoid feeding the model multiple raw facts that one observation already summarizes.
>
> Restore the broad recall with `"recall_types": "observation,world,experience"` (string or JSON list) in `~/.hermes/hindsight/config.json`. This applies to **both** auto-recall and the `hindsight_recall` tool — both read the same `recall_types` setting (the tool schema has no per-call `types` argument), so narrowing the default narrows both paths.

### Opt-in provenance-safe selection

Merge these settings into the active profile's `hindsight/config.json`; this is
an enable payload, **not** a deployment or service-mode migration:

```json
{
  "recall_selection_policy": "observation_first",
  "recall_prefetch_method": "recall",
  "recall_sync": true,
  "recall_max_tokens": 1200
}
```

Selection is provider-local, read-only and used by both automatic recall and
`hindsight_recall`. It adds no model tools or LLM synthesis. Existing retain,
shared observation scope, capture, scoped workers and embedded reconnect logic
are unchanged. Leaving out the policy or setting `legacy` preserves the previous
one-request path. `reflect` remains a separate, model-backed operation and does
not use this selection policy.

1. Make **one mixed recall** for `observation,world,experience`, using the original
   query, bank, tags and token cap. The historical `observation_first` option name
   is retained for compatibility; it no longer means two sequential requests.
   A second unconditional probe exceeded the host's 8-second prefetch deadline.
2. Feature-detect the SDK's `prefer_observations` keyword. With current SDK 0.10.3
   and server 0.10.1, request native observation preference: the server drops raw
   facts referenced by returned observations and backfills the freed slots.
   SDK 0.6.1 lacks this keyword and uses ordinary mixed recall, still in one call.
   This server policy is not proof of summary completeness or correctness: a
   covered raw fact can contain details absent from its observation. Explicit
   source inspection remains necessary for exact quotations or contested details.
3. Reserve half the usable context for returned raw evidence before filling
   observations. The newest **raw mention** has first selection priority, even
   when an older raw claim coalesces with an observation earlier in display order.
   Other raw facts keep server relevance order; historical matches are not sorted
   away by recency. Selection priority does not establish truth or authority.
4. Share only an exact, whitespace-normalized factual body. The ` | When:` and
   ` | Involving:` presentation suffix is excluded from the grouping key, but its
   **entire text is retained per source**, including trailing context/negations.
   Different subjects, values, negations and factual body dates remain separate.
   Identical text from different sources can share a body, while every grouped
   source keeps its ID, mention/event times and supplied qualifications. No
   embedding, LLM, substring or paraphrase deduper is used. At most three source
   records share a displayed body; additional records form another group rather
   than silently dropping their qualifications.
5. Render compact references with fact IDs, type, `mentioned` time, supplied
   `occurred`/`until` event times, context and explicit speaker/status/authority
   metadata. Mention time is **not** an observed-truth timestamp. Source links are
   bounded to three per record with an additional-source count. Full operational
   metadata (session/chat/worker IDs, tags, entities, document/chunk IDs) stays out
   of the hot context; retrieve source records by fact ID when needed. Never infer
   a fact's speaker from a session's `user_name`, approval from a recommendation,
   or current truth from observation creation time. Historical incidents remain.

**Budgets and limits:** the one SDK call requests at most `recall_max_tokens`
results; the final combined text, including guidance, preamble and references,
is capped at the same number of **estimated tokens** using Hermes's multilingual
estimator. This is not an exact model-specific tokenizer guarantee. Whole claims
and their compact references are omitted, never sliced; omissions are disclosed.
If no whole claim fits, the result is empty, or retrieval fails, automatic recall
abstains. A missing raw result does not trigger another potentially slow request
and does not establish that every correction was consolidated. Retrieval, native
source suppression and packing can still miss facts. No arbitrary score floor is
applied: reranker scores are not calibrated and sparse relevant results can score
low. This policy does not establish semantic abstention for irrelevant queries.

**Compatibility evidence:** SDK 0.6.1 request/response compatibility is covered by
an offline transport smoke and provider-boundary tests; current SDK 0.10.3 and
server 0.10.1 `/openapi.json` were inspected and exercised read-only. The newer
server also supports `min_scores` and `temporal_window`; this policy does not use
them. No server upgrade, memory writes, observation rebuild or deployment occurs.

### Selection diagnostics and bounded live comparison

`provider.recall_diagnostics()` reports the last policy selection's content-free
counters: path, SDK request count, requested token caps, observation/raw counts,
coalesced records, **selected_claims**, omitted claims, estimated tokens,
retrieval/total milliseconds, failed lanes and whether native observation
preference was requested. It contains no query, facts, IDs, tags or secrets.
Selection is not confirmed injection: the host can time out or drop a completed
provider result. Background mode describes the latest completed selection, not
necessarily this turn. Legacy/reflect paths report `path: not_run`.

Normal cost is **one SDK call**. Counts include the existing embedded reconnect
retry; SDK-internal HTTP retries are opaque. Requested token caps are ceilings,
not billed cost. Local tests cover the 450-word old/coalesced claim versus a
120-word recent correction at a 1200-token cap, and the actual host's 8-second
prefetch deadline with a 4.1-second SDK response.

A read-only, sequential live comparison on English preference, Slack repair,
Honcho recommendation attribution and Useful Idea Radar queries returned the
revised candidate in 2.31–5.10 seconds (one request each), at 1148–1200 estimated
output tokens. The corresponding legacy mixed output was 1519–1626 tokens.
English preference, repair confirmation, recommendation-not-adoption and radar
format evidence survived. These unfrozen samples support a latency/budget fix,
**not** a general quality claim: paraphrased English/radar repetitions remain,
and no exact-body groups coalesced in those four final server-filtered samples.
Tests are regression evidence, not proof that all relevant corrections survive
server retrieval. Activation and real user-visible answer verification remain
separate steps.

### Retain

| Key | Default | Description |
|-----|---------|-------------|
| `auto_retain` | `true` | Automatically retain conversation turns |
| `observation_scopes` | `combined` | Use `shared` when per-session source tags vary: observations consolidate globally while source facts keep provenance tags. |
| `retain_async` | `true` | Process retain asynchronously on the Hindsight server |
| `retain_every_n_turns` | `1` | Retain every N turns (1 = every turn) |
| `retain_context` | `conversation between Hermes Agent and the User` | Context label for retained memories |
| `retain_tags` | — | Default tags applied to retained memories; merged with per-call tool tags |
| `retain_source` | — | Opt-in `metadata.source` attached to retained memories (identifies the storing client, e.g. `hermes`). Empty by default — no attribution tag ships unless you set it. |
| `retain_indicator` | `true` | Show a `👁️ Hindsight — saving to memory…` status line when a turn is saved. Turn off for customer-facing agents. |
| `retain_user_prefix` | `User` | Label used before user turns in auto-retained transcripts |
| `retain_assistant_prefix` | `Assistant` | Label used before assistant turns in auto-retained transcripts |

### Integration

| Key | Default | Description |
|-----|---------|-------------|
| `memory_mode` | `hybrid` | How memories are integrated into the agent |

**memory_mode:**
- `hybrid` — automatic context injection + tools available to the LLM
- `context` — automatic injection only, no tools exposed
- `tools` — tools only, no automatic injection

### Local Embedded LLM

| Key | Default | Description |
|-----|---------|-------------|
| `llm_provider` | `openai` | `openai`, `anthropic`, `gemini`, `groq`, `openrouter`, `minimax`, `ollama`, `lmstudio`, `openai_compatible` |
| `llm_model` | per-provider | Model name (e.g. `gpt-4o-mini`, `qwen/qwen3.5-9b`) |
| `llm_base_url` | — | Endpoint URL for `openai_compatible` (e.g. `http://192.168.1.10:8080/v1`) |

The LLM API key is stored in `~/.hermes/.env` as `HINDSIGHT_LLM_API_KEY`.

The embedded daemon is a subprocess that cannot see the per-turn secret
scope, so it reads the key from `~/.hindsight/profiles/<profile>.env`
(materialized owner-only at setup and on config change). Key resolution
order is explicit config → secret scope → the on-disk profile env, and the
rewrite path is fail-closed: a build with no key never clobbers a profile
file that already holds one.

## Tools

Available in `hybrid` and `tools` memory modes:

| Tool | Description |
|------|-------------|
| `hindsight_retain` | Store information with auto entity extraction; supports optional per-call `tags` |
| `hindsight_recall` | Multi-strategy search (semantic + entity graph) |
| `hindsight_reflect` | Cross-memory synthesis (LLM-powered) |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HINDSIGHT_API_KEY` | API key for Hindsight Cloud |
| `HINDSIGHT_LLM_API_KEY` | LLM API key for local mode |
| `HINDSIGHT_API_LLM_BASE_URL` | LLM Base URL for local mode (e.g. OpenRouter) |
| `HINDSIGHT_API_URL` | Override API endpoint |
| `HINDSIGHT_BANK_ID` | Override bank name |
| `HINDSIGHT_BUDGET` | Override recall budget |
| `HINDSIGHT_MODE` | Override mode (`cloud`, `local_embedded`, `local_external`) |

## Client Version

Requires `hindsight-client >= 0.6.1`. The plugin auto-upgrades on session start if an older version is detected.
