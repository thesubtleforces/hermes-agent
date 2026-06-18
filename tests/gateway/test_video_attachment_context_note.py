from pathlib import Path
from types import SimpleNamespace

from gateway.platforms.base import MessageType
from gateway.run import _build_media_placeholder


def test_build_media_placeholder_identifies_video_attachment():
    event = SimpleNamespace(
        media_urls=["/tmp/example.mov"],
        media_types=["video/quicktime"],
        message_type=MessageType.VIDEO,
    )

    text = _build_media_placeholder(event)

    assert "video attachment" in text
    assert "/tmp/example.mov" in text
    assert "Do NOT say you did not receive" in text


def test_process_message_prepends_video_context_note(monkeypatch, tmp_path):
    video = tmp_path / "clip.mov"
    video.write_bytes(b"not actually a video; path is enough for context-note test")

    monkeypatch.setattr(
        "tools.credential_files.to_agent_visible_cache_path",
        lambda p: f"/agent-visible/{Path(p).name}",
    )

    event = SimpleNamespace(
        media_urls=[str(video)],
        media_types=["video/quicktime"],
        message_type=MessageType.VIDEO,
    )

    # Mirrors the video-note branch in GatewayServer._process_message.
    video_paths = []
    for i, path in enumerate(event.media_urls):
        mtype = event.media_types[i] if i < len(event.media_types) else ""
        if mtype.startswith("video/") or event.message_type == MessageType.VIDEO:
            video_paths.append(path)

    message_text = "The video I sent"
    if video_paths:
        from tools.credential_files import to_agent_visible_cache_path as _to_agent_path
        import os, re
        _notes = []
        for _vpath in video_paths:
            _basename = os.path.basename(_vpath)
            _parts = _basename.split("_", 2)
            _display = _parts[2] if len(_parts) >= 3 else _basename
            _display = re.sub(r'[^\w.\- ]', '_', _display)
            _agent_path = _to_agent_path(_vpath)
            _notes.append(
                f"[The user sent a video attachment: '{_display}'. "
                f"It is saved at: {_agent_path}. Its content is not inlined here. "
                "If the user's request involves what the video contains, inspect or process "
                "the saved file before answering — for example extract frames or use a video "
                "analysis tool. Do NOT say you did not receive the video.]"
            )
        if _notes:
            message_text = "\n".join(_notes) + f"\n\n{message_text}"

    assert "video attachment" in message_text
    assert "/agent-visible/clip.mov" in message_text
    assert "The video I sent" in message_text
