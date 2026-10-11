"""Selected human reply triggers do not weaken strict chat gates or sender authorization."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import Platform, load_gateway_config
from plugins.platforms.telegram.adapter import TelegramAdapter

CHAT = -1004373539793
PAVEL = 25415780
BOT = 8794916470
HANDLE = 'pavzagor_hermes_bot'
OTHER_CHAT = -1001234567890


@pytest.fixture
def adapter(tmp_path, monkeypatch) -> TelegramAdapter:
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text(
        'telegram:\n  extra:\n'
        f'    mention_only_chats: ["{CHAT}", "{OTHER_CHAT}"]\n'
        '    mention_only_reply_users:\n'
        f'      "{CHAT}": ["{PAVEL}"]\n'
        '    require_mention: false\n'
        f'    free_response_chats: ["{CHAT}"]\n'
        '    mention_patterns: ["^Hermes"]\n', encoding='utf-8',
    )
    config = load_gateway_config().platforms[Platform.TELEGRAM]
    result = TelegramAdapter(config)
    result._bot = SimpleNamespace(id=BOT, username=HANDLE)
    return result


def message(text: str = 'Thanks', *, sender: int = PAVEL, reply_to: int | None = BOT,
            chat: int = CHAT, caption: bool = False, bot_sender: bool = False) -> SimpleNamespace:
    entities = []
    if text.startswith('@'):
        entities = [SimpleNamespace(type='mention', offset=0, length=len(text.split()[0]))]
    elif text.startswith('/'):
        entities = [SimpleNamespace(type='bot_command', offset=0, length=len(text.split()[0]))]
    return SimpleNamespace(
        message_id=42, date=None, text=None if caption else text, caption=text if caption else None,
        entities=[] if caption else entities, caption_entities=entities if caption else [],
        chat=SimpleNamespace(id=chat, type='supergroup', is_forum=False, title='Test'),
        from_user=SimpleNamespace(id=sender, is_bot=bot_sender, username='sender', first_name='Sender', full_name='Sender'),
        message_thread_id=None, is_topic_message=False,
        reply_to_message=SimpleNamespace(
            from_user=SimpleNamespace(id=reply_to, is_bot=reply_to in (BOT, 888)),
            message_id=10, text='Parent message', caption=None, entities=[],
        ) if reply_to is not None else None,
    )


@pytest.mark.parametrize('kwargs,expected', [
    ({}, True),
    ({'caption': True}, True),
    ({'reply_to': None}, False),
    ({'reply_to': 777}, False),
    ({'reply_to': 888}, False),
    ({'sender': 7541188094}, False),
    ({'sender': 123}, False),
    ({'bot_sender': True}, False),
    ({'text': '@other_bot Thanks'}, False),
    ({'chat': OTHER_CHAT}, False),
    ({'text': 'Hermes answer this', 'reply_to': None}, False),
    ({'text': '/status', 'reply_to': None}, False),
    ({'text': f'@{HANDLE} answer', 'reply_to': 777}, True),
    ({'text': f'/status@{HANDLE}', 'reply_to': None}, True),
])
def test_reply_exception_is_sender_chat_and_receiving_bot_scoped(
    adapter: TelegramAdapter, kwargs: dict, expected: bool,
) -> None:
    assert adapter._should_process_message(message(**kwargs)) is expected


def test_absent_setting_and_other_profile_keep_strict_reply_gate(
    adapter: TelegramAdapter, tmp_path, monkeypatch,
) -> None:
    second_home = tmp_path / 'second'; second_home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(second_home))
    (second_home / 'config.yaml').write_text(
        f'telegram:\n  extra:\n    mention_only_chats: ["{CHAT}"]\n', encoding='utf-8',
    )
    other = TelegramAdapter(load_gateway_config().platforms[Platform.TELEGRAM])
    other._bot = adapter._bot
    assert adapter._should_process_message(message()) is True
    assert other._should_process_message(message()) is False


@pytest.mark.parametrize('restriction', ['allowed_chats', 'allowed_topics', 'ignored_threads'])
def test_reply_does_not_bypass_chat_or_topic_restrictions(
    adapter: TelegramAdapter, restriction: str,
) -> None:
    adapter.config.extra[restriction] = ['123'] if restriction != 'ignored_threads' else [1]
    msg = message()
    # A real forum General topic has effective thread id 1.
    if restriction == 'ignored_threads':
        msg.chat.is_forum = True
    assert adapter._should_process_message(msg) is False


def test_missing_bot_identity_does_not_admit_reply(adapter: TelegramAdapter) -> None:
    adapter._bot = SimpleNamespace(id=None, username=HANDLE)
    assert adapter._should_process_message(message()) is False


@pytest.mark.asyncio
@pytest.mark.parametrize('authorized', [True, False])
async def test_reply_text_intake_still_requires_existing_sender_authorization(
    adapter: TelegramAdapter, authorized: bool,
) -> None:
    adapter.config.extra['group_allow_from'] = [str(PAVEL if authorized else 222)]
    adapter._ensure_forum_commands = AsyncMock()
    adapter._cache_replied_media = AsyncMock()
    adapter._enqueue_text_event = Mock()
    msg = message()
    assert adapter._should_process_message(msg) is True
    await adapter._handle_text_message(
        SimpleNamespace(effective_message=msg, message=msg, update_id=1), SimpleNamespace(),
    )
    assert adapter._enqueue_text_event.call_count == int(authorized)
    if authorized:
        event = adapter._enqueue_text_event.call_args.args[0]
        assert event.source.user_id == str(PAVEL)
        assert event.source.chat_id == str(CHAT)
        assert event.reply_to_message_id == '10'
        assert event.text == 'Thanks'


def test_non_strict_chat_retains_existing_open_behavior(adapter: TelegramAdapter) -> None:
    assert adapter._should_process_message(message(chat=-100999, reply_to=None)) is True
