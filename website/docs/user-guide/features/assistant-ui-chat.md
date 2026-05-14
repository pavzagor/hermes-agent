---
sidebar_position: 15
title: "Assistant UI Chat"
description: "Private assistant-ui web chat for Hermes via the API server"
---

# Assistant UI Chat

Hermes Dashboard includes a browser-native chat page powered by [assistant-ui](https://www.assistant-ui.com/). It is separate from the embedded terminal chat: the browser talks to the dashboard backend, and the dashboard backend forwards requests to the localhost-only Hermes API server.

```text
Browser
  → Hermes dashboard /assistant
  → POST /api/assistant-chat with dashboard session token
  → http://127.0.0.1:8642/v1/chat/completions with server-side bearer token
  → Hermes agent
```

The browser never receives `API_SERVER_KEY`.

## Enable the Hermes API server

Add this to `~/.hermes/.env`:

```bash
API_SERVER_ENABLED=true
API_SERVER_HOST=127.0.0.1
API_SERVER_PORT=8642
API_SERVER_KEY=<strong-random-secret>
```

Start or restart the gateway:

```bash
hermes gateway restart
```

Then start the dashboard:

```bash
hermes dashboard
```

Open `/assistant` in the dashboard. The sidebar shows the current browser thread plus recent Hermes sessions from the dashboard session store; click a previous chat to load its transcript into the assistant-ui surface.

## Optional dashboard proxy settings

The assistant-ui dashboard proxy reads these environment variables from `~/.hermes/.env`:

- `HERMES_ASSISTANT_API_BASE`: Defaults to `http://127.0.0.1:8642/v1`.
- `HERMES_ASSISTANT_API_KEY`: Optional override for the API server bearer token. If omitted, the proxy uses `API_SERVER_KEY`.
- `HERMES_ASSISTANT_MODEL`: Cosmetic model name sent to the OpenAI-compatible API. Defaults to `hermes-agent`.
- `HERMES_DASHBOARD_ALLOWED_EMAILS`: Optional comma-separated Cloudflare Access email allowlist for `/api/assistant-chat`.
- `CLOUDFLARE_ACCESS_ALLOWED_EMAILS`: Alias for the same allowlist.

## Cloudflare Tunnel + Access

For remote access, expose the dashboard through your own Cloudflare Tunnel and protect it with Cloudflare Access.

Recommended shape:

```text
https://hermes.example.com
  → Cloudflare Access policy allowing only your email
  → cloudflared tunnel
  → http://127.0.0.1:9119
```

Keep the Hermes API server bound to localhost:

```bash
API_SERVER_HOST=127.0.0.1
```

Do not expose `http://127.0.0.1:8642` directly through the tunnel.

For defense in depth, set an email allowlist in `~/.hermes/.env`:

```bash
HERMES_DASHBOARD_ALLOWED_EMAILS=you@example.com
```

When this is set, `/api/assistant-chat` requires Cloudflare Access to pass `Cf-Access-Authenticated-User-Email` matching the allowlist.

## Security model

The assistant-ui chat endpoint is protected by two layers:

1. The dashboard's ephemeral `X-Hermes-Session-Token`, injected into the page by the dashboard server.
2. Optional Cloudflare Access email allowlist for public tunnel deployments.

The dashboard backend then injects the Hermes API bearer token server-side.

## Limitations

- Assistant-ui replies stream through the dashboard proxy as browser-safe NDJSON. If the API server is unavailable, the message displays a backend-unavailable error in the thread.
- Text messages and image attachments are supported. The browser converts images to `data:image/...` parts, and the dashboard proxy forwards those as OpenAI-compatible `image_url` content. Non-image file uploads are still represented as text/placeholders.
- Loading a previous chat restores its transcript into the browser runtime for viewing and follow-up context. New assistant-ui turns are still sent as normal chat history rather than using server-side `previous_response_id` chaining.
