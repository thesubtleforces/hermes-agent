import json

from hermes_constants import set_hermes_home_override, reset_hermes_home_override


def test_record_received_merges_same_sha_across_paths(tmp_path):
    from gateway.media_ledger import record_received, sha256_bytes

    token = set_hermes_home_override(tmp_path)
    try:
        digest = sha256_bytes(b"same image bytes")
        first = record_received(
            platform="telegram",
            chat_id="100",
            message_id="42",
            media_index=0,
            path="/tmp/img_a.jpg",
            media_type="image/jpeg",
            sha256=digest,
            media_group_id="album-1",
            file_unique_id="uniq-a",
        )
        second = record_received(
            platform="telegram",
            chat_id="100",
            message_id="43",
            media_index=0,
            path="/tmp/img_b.jpg",
            media_type="image/jpeg",
            sha256=digest,
            media_group_id="album-1",
            file_unique_id="uniq-a",
        )

        assert first["artifact_key"] == second["artifact_key"] == f"sha256:{digest}"
        assert second["paths"] == ["/tmp/img_a.jpg", "/tmp/img_b.jpg"]
        assert second["message_ids"] == ["42", "43"]
        assert second["media_group_ids"] == ["album-1"]
        assert second["telegram_file_unique_ids"] == ["uniq-a"]
        assert second["states"]["received"] is True

        ledger = json.loads((tmp_path / "state" / "media_ledger.json").read_text())
        assert ledger["path_index"]["/tmp/img_b.jpg"] == f"sha256:{digest}"
        assert ledger["platform_index"]["telegram:100:43:0"] == f"sha256:{digest}"
    finally:
        reset_hermes_home_override(token)


def test_auto_assessment_retrievable_from_later_path_with_same_sha(tmp_path):
    from gateway.media_ledger import (
        get_auto_assessment,
        record_auto_assessment,
        record_received,
        sha256_bytes,
    )

    token = set_hermes_home_override(tmp_path)
    try:
        digest = sha256_bytes(b"same bytes")
        record_received(
            platform="telegram",
            chat_id="100",
            message_id="42",
            path="/tmp/first.jpg",
            media_type="image/jpeg",
            sha256=digest,
        )
        record_auto_assessment("/tmp/first.jpg", "A child holding a shell.")
        record_received(
            platform="telegram",
            chat_id="100",
            message_id="43",
            path="/tmp/second.jpg",
            media_type="image/jpeg",
            sha256=digest,
        )

        assert get_auto_assessment("/tmp/second.jpg") == "A child holding a shell."
    finally:
        reset_hermes_home_override(token)


def test_corrupt_ledger_falls_back_to_empty(tmp_path):
    from gateway.media_ledger import record_received, find_by_path

    token = set_hermes_home_override(tmp_path)
    try:
        state = tmp_path / "state"
        state.mkdir()
        (state / "media_ledger.json").write_text("not-json")

        assert find_by_path("/tmp/missing.jpg") is None
        artifact = record_received(
            platform="telegram",
            chat_id="100",
            message_id="42",
            path="/tmp/recovered.jpg",
            media_type="image/jpeg",
            sha256="a" * 64,
        )
        assert artifact["artifact_key"] == "sha256:" + "a" * 64
    finally:
        reset_hermes_home_override(token)
