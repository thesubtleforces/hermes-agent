import asyncio
import json
from unittest.mock import AsyncMock, patch

from hermes_constants import set_hermes_home_override, reset_hermes_home_override


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _runner_stub():
    from gateway.run import GatewayRunner

    class _Stub:
        _enrich_message_with_vision = GatewayRunner._enrich_message_with_vision

    return _Stub()


def test_vision_enrichment_reuses_cached_auto_assessment(tmp_path):
    from gateway.media_ledger import record_auto_assessment, record_received

    token = set_hermes_home_override(tmp_path)
    try:
        record_received(
            platform="telegram",
            chat_id="100",
            message_id="42",
            path="/tmp/reused.jpg",
            media_type="image/jpeg",
            sha256="b" * 64,
        )
        record_auto_assessment("/tmp/reused.jpg", "A reused family photo description.")

        mock_tool = AsyncMock(return_value=json.dumps({"success": True, "analysis": "fresh"}))
        with patch("tools.vision_tools.vision_analyze_tool", new=mock_tool):
            out = _run(_runner_stub()._enrich_message_with_vision("caption", ["/tmp/reused.jpg"]))

        mock_tool.assert_not_called()
        assert "A reused family photo description" in out
        assert "image_url: /tmp/reused.jpg" in out
    finally:
        reset_hermes_home_override(token)


def test_vision_enrichment_records_sanitized_auto_assessment_on_miss(tmp_path):
    from gateway.media_ledger import get_auto_assessment, record_received

    token = set_hermes_home_override(tmp_path)
    try:
        record_received(
            platform="telegram",
            chat_id="100",
            message_id="42",
            path="/tmp/new.jpg",
            media_type="image/jpeg",
            sha256="c" * 64,
        )
        leaked = "<memory-context>secret</memory-context> A clean description."
        mock_tool = AsyncMock(return_value=json.dumps({"success": True, "analysis": leaked}))
        with patch("tools.vision_tools.vision_analyze_tool", new=mock_tool):
            out = _run(_runner_stub()._enrich_message_with_vision("caption", ["/tmp/new.jpg"]))

        mock_tool.assert_awaited_once()
        assert "A clean description" in out
        assert "secret" not in out
        assert get_auto_assessment("/tmp/new.jpg") == "A clean description."
    finally:
        reset_hermes_home_override(token)
