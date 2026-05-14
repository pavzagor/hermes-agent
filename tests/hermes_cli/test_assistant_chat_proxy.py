import json

from hermes_cli import web_server
from hermes_cli.web_server import _assistant_message_to_openai, _stream_hermes_chat_api


def test_assistant_message_preserves_image_data_url_parts():
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the image"},
            {"type": "image", "image": "data:image/png;base64,abc123"},
        ],
    }

    assert _assistant_message_to_openai(msg) == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
        ],
    }


def test_assistant_message_preserves_openai_image_url_parts():
    msg = {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "What is this?"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png", "detail": "low"},
            },
        ],
    }

    assert _assistant_message_to_openai(msg) == {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is this?"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png", "detail": "low"},
            },
        ],
    }


def test_assistant_message_keeps_text_only_content_as_string():
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ],
    }

    assert _assistant_message_to_openai(msg) == {"role": "user", "content": "hello\nworld"}


def test_assistant_stream_proxy_converts_sse_to_ndjson(monkeypatch):
    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return iter(
                [
                    b'data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}\n',
                    b'\n',
                    b'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}\n',
                    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
                    b'data: [DONE]\n',
                ]
            )

    captured = {}

    def fake_urlopen(req, timeout):
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["accept"] = req.headers.get("Accept")
        return FakeResponse()

    monkeypatch.setenv("API_SERVER_KEY", "test-key")
    monkeypatch.setattr(web_server.urllib.request, "urlopen", fake_urlopen)

    lines = [json.loads(line) for line in _stream_hermes_chat_api({"messages": []})]

    assert captured == {
        "timeout": 300,
        "body": {"messages": [], "stream": True},
        "accept": "text/event-stream",
    }
    assert lines == [
        {"type": "delta", "text": "Hel"},
        {"type": "delta", "text": "lo"},
        {"type": "done", "reason": "stop"},
    ]
