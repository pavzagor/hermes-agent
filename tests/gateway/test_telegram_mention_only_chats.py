"""A strict per-chat @mention gate must not alter other groups or DMs."""
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter

TARGET = -1001234567890


def make_adapter() -> TelegramAdapter:
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token='test', extra={
        'mention_only_chats': [str(TARGET)], 'require_mention': False,
        'allowed_chats': [], 'allowed_topics': [], 'group_allowed_chats': [],
        'guest_mode': True, 'free_response_chats': [str(TARGET)],
        'mention_patterns': ['^Hermes'],
    })
    adapter._bot = SimpleNamespace(id=999, username='hermes_bot')
    adapter._mention_patterns = adapter._compile_mention_patterns()
    return adapter


def message(text: str, *, chat_id: int = TARGET, private: bool = False,
            reply: bool = False, caption: bool = False) -> SimpleNamespace:
    data = {'message_id': 1, 'date': 1,
            'chat': {'id': chat_id, 'type': 'private' if private else 'supergroup', 'title': 'Test'},
            'from': {'id': 111, 'is_bot': False, 'first_name': 'Sender'}}
    field = 'caption' if caption else 'text'
    data[field] = text
    if '@hermes_bot' in text:
        data['caption_entities' if caption else 'entities'] = [
            {'type': 'mention', 'offset': text.index('@hermes_bot'), 'length': len('@hermes_bot')},
        ]
    elif text.startswith('/'):
        data['entities'] = [{'type': 'bot_command', 'offset': 0, 'length': len(text.split()[0])}]
    if reply:
        data['reply_to_message'] = {'message_id': 2, 'date': 1, 'chat': data['chat'],
                                    'from': {'id': 999, 'is_bot': True, 'first_name': 'Hermes'},
                                    'text': 'Earlier bot reply'}
    return SimpleNamespace(
        message_id=1, date=None, text=None if caption else text, caption=text if caption else None,
        entities=[] if caption else [SimpleNamespace(**e) for e in data.get('entities', [])],
        caption_entities=[SimpleNamespace(**e) for e in data.get('caption_entities', [])],
        chat=SimpleNamespace(id=chat_id, type='private' if private else 'supergroup', is_forum=False),
        from_user=SimpleNamespace(id=111, is_bot=False), message_thread_id=None, is_topic_message=False,
        reply_to_message=SimpleNamespace(from_user=SimpleNamespace(id=999), message_id=2) if reply else None,
    )


@pytest.mark.parametrize('text,reply,caption,expected', [
    ('ordinary conversation', False, False, False),
    ('Thanks', True, False, False),
    ('Hermes answer this', False, False, False),
    ('/status', False, False, False),
    ('@other_bot hello', False, False, False),
    ('@hermes_bot hello', False, False, True),
    ('@hermes_bot photo', False, True, True),
    ('/status@hermes_bot', False, False, True),
])
def test_selected_chat_requires_explicit_address_even_with_fallbacks(
    text: str, reply: bool, caption: bool, expected: bool,
) -> None:
    adapter = make_adapter()
    assert adapter._should_process_message(message(text, reply=reply, caption=caption)) is expected


def test_other_groups_and_direct_messages_keep_open_response() -> None:
    adapter = make_adapter()
    assert adapter._should_process_message(message('hello', chat_id=-1009876543210))
    assert adapter._should_process_message(message('hello', chat_id=111, private=True))


def test_real_yaml_loader_carries_per_chat_gate(tmp_path, monkeypatch) -> None:
    from gateway.config import load_gateway_config
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text(
        'telegram:\n  extra:\n    mention_only_chats:\n      - "-1001234567890"\n',
        encoding='utf-8',
    )
    config = load_gateway_config()
    assert config.platforms[Platform.TELEGRAM].extra['mention_only_chats'] == [str(TARGET)]
