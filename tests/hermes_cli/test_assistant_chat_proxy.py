from hermes_cli.web_server import _assistant_message_to_openai


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
