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
| `recall_selection_policy` | `legacy` | `legacy` preserves existing behavior; opt-in `observation_first` adds a raw-evidence safety lane, conservative injection deduplication and provenance. Overrides `recall_types` for recall, not reflect. Invalid values are rejected. |
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

1. Recall observations first, then `world,experience` with the **same query,
   bank, tag filter and budget**. Both raw types are deliberate even if legacy
   `recall_types` excluded one: corrections and assistant-origin evidence may be
   either type. No date filter or English-only intent classifier is used.
2. Always probe raw evidence, including when observations are nonempty. SDK
   0.6.1 exposes no consolidation-completeness watermark; even a newly created
   observation cannot prove that it includes the latest correction. This handles
   empty/insufficient observations and source/exact-date/recent/history queries
   without guessing whether a summary is sufficient.
3. Reserve half the usable injection budget for raw evidence before filling
   observations. Give the newest **raw mention** a safety slot, then preserve
   server relevance order so a historical top match is not buried by newer
   material. This is selection priority, **not** a truth/authority ranking.
4. Coalesce only whitespace-equivalent claims with identical time, context,
   metadata, tag and entity qualifiers **and** a positive source link (same fact
   ID, direct `source_fact_ids` relationship, or same nonempty document+chunk).
   Keep every coalesced record's provenance. Different values, negations, dates,
   speakers, status assertions, or unknown source identity remain separate.
   A source link by itself never proves a paraphrase equivalent. There is no
   embedding/LLM paraphrase deduper and no database deletion or observation rebuild.
5. Render supplied IDs, source links, document/chunk IDs, times, context, tags,
   entities and metadata as evidence. Metadata is reported literally, not promoted
   to verified status or authority. Missing speaker/status remains unknown;
   a session's `user_name` is not the speaker of every extracted fact. Never infer
   approval from an assistant recommendation, or current truth from observation
   creation time. Resolved historical incidents are not suppressed.

**Budgets and limits:** each of the two SDK recall calls requests at most
`recall_max_tokens` results, so a normal 1200-token setting permits **2400 tokens
of server candidate retrieval**, but the final combined injection (including
preamble, provenance and safety guidance) is capped at **1200 estimated tokens**
using Hermes's multilingual token estimator. This is not an exact model-specific
tokenizer guarantee. Whole records are omitted, never sliced to lose a negation
or its provenance; omissions are disclosed. If no whole record fits, or both
lanes are empty, automatic injection abstains. One failed lane leaves the other
usable with an incompleteness warning. Retrieval/packing can still miss facts:
this does not prove that every relevant correction was retrieved, and SDK 0.6.1
does not expose calibrated relevance floors for semantic abstention.

**Compatibility evidence:** inspected the PyPI `hindsight-client==0.6.1` wheel
and server `0.10.1` `/version` + `/openapi.json`, alongside the
[current recall docs](https://hindsight.vectorize.io/developer/api/recall).
Both expose `RecallResult.id/text/type/entities/context/occurred_start/
occurred_end/mentioned_at/document_id/metadata/chunk_id/tags/source_fact_ids`.
There is no top-level speaker, status or authority field in that contract.
The newer server supports `prefer_observations`, `min_scores`, and
`temporal_window`, but the 0.6.1 Python client **does not**; this policy sends none
of those kwargs. The server's newer scores/truncation fields are not required.
An offline real-0.6.1 SDK transport smoke and provider-boundary tests exercise
the compatible request and response shape; no server upgrade is necessary.

### Selection diagnostics and A/B evaluation

After `provider.prefetch(query)` (or manual recall),
`provider.recall_diagnostics()` returns a copy of the **last policy selection's**
content-free counters: policy/path, SDK call count, total requested token caps,
observation/raw result counts, coalesced record count, injected/omitted claim
counts, estimated output tokens, per-lane/total elapsed milliseconds, and failed
lane names. It contains no query text, fact text, IDs, bank IDs, tags or secrets.
It is a Python diagnostic method, not a model tool; background mode describes
the latest completed selection, not necessarily the current turn's injection.
Legacy/reflect paths do not produce selection timings (`path: not_run`).

Normal cost is **two sequential SDK calls instead of one**. `requests` counts
SDK calls, including the provider's embedded reconnect retry; HTTP retries hidden
inside the SDK are not observable here. `requested_max_tokens_total` is a request
ceiling, **not** actual consumed tokens or billed cost. `raw_elapsed_ms` measures
the added raw lane; `elapsed_ms` includes retrieval, grouping and formatting.
Measure service latency separately from fixture-only CPU timings.

Use frozen development and held-out retrieval fixtures through the real provider
boundary at the same final 1200-token context cap. Compare evidence recall,
contradiction/negation/speaker preservation, historical exact-date recovery,
duplicate density, abstention, output tokens and latency. Inspect negative pairs
before accepting a duplicate-metric gain; never improve it by deleting differing
claims. Unit tests are compatibility/regression proof, **not** evidence of a
production quality uplift. Live activation remains a separate approval/verification
step, and a change to external service mode must preserve the existing DB/version.

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
