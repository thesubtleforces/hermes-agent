from types import SimpleNamespace

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from gateway.platforms.telegram import TelegramAdapter


def _make_adapter() -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))


def test_media_message_type_classifies_video_note_as_video():
    platform = _make_adapter()
    msg = SimpleNamespace(
        sticker=None,
        photo=None,
        video=None,
        video_note=object(),
        audio=None,
        voice=None,
    )

    assert platform._media_message_type(msg) is MessageType.VIDEO


def test_observed_media_source_supports_video_note():
    platform = _make_adapter()
    source = object()
    msg = SimpleNamespace(
        photo=None,
        video=None,
        video_note=source,
        voice=None,
        audio=None,
        document=None,
    )

    assert platform._observed_media_source(msg) == (
        source,
        "video_note.mp4",
        "video/mp4",
        "video",
    )
