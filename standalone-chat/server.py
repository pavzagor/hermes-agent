#!/usr/bin/env python3
"""Standalone mobile-first Hermes chat server.

Serves the standalone assistant-ui build and proxies chat requests to the local
Hermes OpenAI-compatible API. This deliberately does not import or modify the
Hermes dashboard web app.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
DIST_DIR = APP_DIR / "dist"
HERMES_HOME = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        values[key.strip()] = value
    return values


ENV = {**load_dotenv(HERMES_HOME / ".env"), **load_dotenv(HERMES_HOME / "standalone-chat.env"), **os.environ}


def env(name: str, default: str = "") -> str:
    return str(ENV.get(name) or default)


def unauthorized() -> Response:
    return Response(
        "Authentication required\n",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Hermes Chat"'},
        media_type="text/plain",
    )


def allowed_access_emails() -> set[str]:
    raw = env("HERMES_CHAT_ALLOWED_EMAILS") or env("CLOUDFLARE_ACCESS_ALLOWED_EMAILS")
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


async def enforce_auth(request: Request) -> Optional[Response]:
    # Health checks from localhost are allowed so launchd/cloudflared can probe without secrets.
    if request.url.path == "/healthz" and request.client and request.client.host in {"127.0.0.1", "::1"}:
        return None

    access_email = request.headers.get("cf-access-authenticated-user-email", "").strip().lower()
    allowed_emails = allowed_access_emails()
    if access_email:
        if not allowed_emails or access_email in allowed_emails:
            return None
        raise HTTPException(status_code=403, detail="Cloudflare Access user is not allowed")

    expected_user = env("HERMES_CHAT_BASIC_USER", "pavel")
    expected_password = env("HERMES_CHAT_BASIC_PASSWORD")
    if expected_password:
        header = request.headers.get("authorization", "")
        if header.lower().startswith("basic "):
            try:
                decoded = base64.b64decode(header.split(" ", 1)[1]).decode("utf-8")
                user, password = decoded.split(":", 1)
                if secrets.compare_digest(user, expected_user) and secrets.compare_digest(password, expected_password):
                    return None
            except Exception:
                pass
        return unauthorized()

    # Safe default for first local bring-up only. Public deployments should set either
    # Cloudflare Access headers or HERMES_CHAT_BASIC_PASSWORD.
    if request.client and request.client.host in {"127.0.0.1", "::1"}:
        return None
    return unauthorized()


app = FastAPI(title="Hermes Standalone Chat", docs_url=None, redoc_url=None)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    result = await enforce_auth(request)
    if result is not None:
        return result
    return await call_next(request)


def assistant_part_to_openai(part: Any) -> Optional[Dict[str, Any]]:
    if isinstance(part, str):
        return {"type": "text", "text": part} if part else None
    if not isinstance(part, dict):
        return None
    part_type = str(part.get("type") or "").lower()
    if part_type in {"text", "input_text", "output_text"}:
        text = str(part.get("text") or "")
        return {"type": "text", "text": text} if text else None
    if part_type in {"image", "image_url", "input_image"}:
        image_ref = part.get("image_url")
        if isinstance(image_ref, dict):
            url_value = image_ref.get("url")
            detail = image_ref.get("detail")
        else:
            url_value = image_ref or part.get("image")
            detail = part.get("detail")
        if not isinstance(url_value, str) or not url_value.strip():
            return None
        image_part: Dict[str, Any] = {"type": "image_url", "image_url": {"url": url_value.strip()}}
        if isinstance(detail, str) and detail.strip():
            image_part["image_url"]["detail"] = detail.strip()
        return image_part
    if part_type in {"file", "input_file"}:
        name = part.get("filename") or part.get("name") or "file"
        return {"type": "text", "text": f"[File attachment omitted by standalone chat proxy: {name}]"}
    return None


def assistant_message_to_openai(message: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(message, dict):
        return None
    role = str(message.get("role") or "user").lower()
    if role not in {"system", "user", "assistant"}:
        return None
    content = message.get("content", "")
    if isinstance(content, list):
        parts = [part for part in (assistant_part_to_openai(item) for item in content) if part]
        if any(part.get("type") == "image_url" for part in parts):
            normalized: Any = parts
        else:
            normalized = "\n".join(str(part.get("text") or "") for part in parts if part.get("text"))
    else:
        part = assistant_part_to_openai(content)
        normalized = str(part.get("text") or "") if part else ""
    return {"role": role, "content": normalized}


def api_key() -> str:
    key = env("HERMES_ASSISTANT_API_KEY") or env("API_SERVER_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="Hermes API key is not configured for standalone chat")
    return key


def stream_hermes_chat(payload: Dict[str, Any]):
    base = env("HERMES_ASSISTANT_API_BASE", "http://127.0.0.1:8642/v1").rstrip("/")
    data = json.dumps({**payload, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key()}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            emitted_done = False
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                payload_text = line[5:].strip()
                if payload_text == "[DONE]":
                    break
                try:
                    event = json.loads(payload_text)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices") if isinstance(event, dict) else None
                if not isinstance(choices, list) or not choices:
                    continue
                first = choices[0]
                if not isinstance(first, dict):
                    continue
                delta = first.get("delta")
                if isinstance(delta, dict):
                    text = delta.get("content")
                    if isinstance(text, str) and text:
                        yield json.dumps({"type": "delta", "text": text}) + "\n"
                finish_reason = first.get("finish_reason")
                if finish_reason:
                    emitted_done = True
                    yield json.dumps({"type": "done", "reason": finish_reason}) + "\n"
                    break
            if not emitted_done:
                yield json.dumps({"type": "done", "reason": "stop"}) + "\n"
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1200]
        yield json.dumps({"type": "error", "status": exc.code, "message": body}) + "\n"
    except Exception as exc:
        yield json.dumps({"type": "error", "status": 502, "message": f"Hermes API server unavailable: {exc}"}) + "\n"


@app.get("/healthz")
async def healthz():
    return {"ok": True, "service": "hermes-standalone-chat"}


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    raw_messages = body.get("messages", []) if isinstance(body, dict) else []
    messages = [msg for msg in (assistant_message_to_openai(item) for item in raw_messages) if msg and msg.get("content")]
    if not messages:
        raise HTTPException(status_code=400, detail="No chat messages provided")
    payload: Dict[str, Any] = {
        "model": env("HERMES_ASSISTANT_MODEL", "hermes-agent"),
        "messages": messages,
        "stream": True,
    }
    return StreamingResponse(
        stream_hermes_chat(payload),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/sessions")
async def sessions(limit: int = 40, offset: int = 0):
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        rows = db.list_sessions_rich(limit=limit, offset=offset)
        total = db.session_count()
        now = time.time()
        for row in rows:
            row["is_active"] = row.get("ended_at") is None and (now - row.get("last_active", row.get("started_at", 0))) < 300
        return {"sessions": rows, "total": total, "limit": limit, "offset": offset}
    finally:
        db.close()


@app.get("/api/sessions/{session_id}/messages")
async def session_messages(session_id: str):
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        sid = db.resolve_session_id(session_id)
        if not sid:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"session_id": sid, "messages": db.get_messages(sid)}
    finally:
        db.close()


if DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")


@app.get("/{path:path}")
async def spa(path: str):
    index = DIST_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=503, detail="Standalone chat frontend is not built. Run bun run build in standalone-chat.")
    target = (DIST_DIR / path).resolve()
    if path and target.is_file() and DIST_DIR.resolve() in target.parents:
        return FileResponse(target)
    return FileResponse(index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=env("HERMES_CHAT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(env("HERMES_CHAT_PORT", "9120")))
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
