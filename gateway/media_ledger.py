"""Durable media artifact ledger for gateway-cached attachments.

The ledger gives gateway/runtime code a small idempotency primitive: record
that a media artifact was received, remember any generic auto-vision
assessment already performed for it, and look that assessment up even if the
same bytes later arrive at a different cache path.

This module deliberately uses stdlib + Hermes' existing atomic JSON write. It
is not a model tool and does not add prompt/tool-schema footprint.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def ledger_path() -> Path:
    """Return the media ledger path under the active Hermes home."""
    return get_hermes_home() / "state" / "media_ledger.json"


def _empty_ledger() -> dict[str, Any]:
    return {
        "version": _SCHEMA_VERSION,
        "artifacts": {},
        "path_index": {},
        "platform_index": {},
    }


def _load_ledger() -> dict[str, Any]:
    path = ledger_path()
    if not path.exists():
        return _empty_ledger()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("media_ledger: ignoring unreadable/corrupt ledger %s: %s", path, exc)
        return _empty_ledger()
    if not isinstance(raw, dict):
        return _empty_ledger()
    raw.setdefault("version", _SCHEMA_VERSION)
    raw.setdefault("artifacts", {})
    raw.setdefault("path_index", {})
    raw.setdefault("platform_index", {})
    if not isinstance(raw["artifacts"], dict):
        raw["artifacts"] = {}
    if not isinstance(raw["path_index"], dict):
        raw["path_index"] = {}
    if not isinstance(raw["platform_index"], dict):
        raw["platform_index"] = {}
    return raw


def _save_ledger(ledger: dict[str, Any]) -> None:
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, ledger, indent=2)


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest for raw media bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> Optional[str]:
    """Return SHA-256 hex digest for a file, or None if it cannot be read."""
    try:
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _append_unique(target: list[Any], value: Any) -> None:
    if value is None:
        return
    s = str(value)
    if not s:
        return
    if s not in target:
        target.append(s)


def _platform_key(platform: str, chat_id: str, message_id: str, media_index: int) -> str:
    return f"{platform}:{chat_id}:{message_id}:{media_index}"


def record_received(
    *,
    platform: str,
    chat_id: str,
    message_id: str,
    path: str | Path,
    media_type: str,
    media_index: int = 0,
    sha256: str | None = None,
    media_group_id: str | None = None,
    file_unique_id: str | None = None,
) -> dict[str, Any]:
    """Record that a media artifact was received/cached.

    Repeated calls with the same SHA merge paths/platform IDs into one artifact
    instead of creating duplicate records.
    """
    path_s = str(path)
    digest = sha256 or sha256_file(path_s)
    if not digest:
        raise ValueError(f"media_ledger: cannot record media without sha256/path bytes: {path_s}")

    ledger = _load_ledger()
    artifacts = ledger["artifacts"]
    artifact_key = f"sha256:{digest}"
    ts = _now()

    artifact = artifacts.get(artifact_key)
    if not isinstance(artifact, dict):
        artifact = {
            "artifact_key": artifact_key,
            "sha256": digest,
            "platform": str(platform),
            "chat_id": str(chat_id),
            "message_ids": [],
            "media_group_ids": [],
            "telegram_file_unique_ids": [],
            "paths": [],
            "media_type": str(media_type or ""),
            "states": {
                "received": True,
                "auto_assessed": False,
                "chronicle_logged": False,
                "needs_human_context": False,
            },
            "auto_assessment": None,
            "chronicle_refs": [],
            "created_at": ts,
            "updated_at": ts,
        }
        artifacts[artifact_key] = artifact

    artifact.setdefault("message_ids", [])
    artifact.setdefault("media_group_ids", [])
    artifact.setdefault("telegram_file_unique_ids", [])
    artifact.setdefault("paths", [])
    artifact.setdefault("states", {})
    artifact.setdefault("chronicle_refs", [])
    artifact["states"]["received"] = True
    artifact["updated_at"] = ts
    if media_type and not artifact.get("media_type"):
        artifact["media_type"] = str(media_type)

    _append_unique(artifact["paths"], path_s)
    _append_unique(artifact["message_ids"], message_id)
    _append_unique(artifact["media_group_ids"], media_group_id)
    _append_unique(artifact["telegram_file_unique_ids"], file_unique_id)

    ledger["path_index"][path_s] = artifact_key
    ledger["platform_index"][_platform_key(str(platform), str(chat_id), str(message_id), int(media_index))] = artifact_key
    _save_ledger(ledger)
    return artifact


def _find_artifact_key_for_path(ledger: dict[str, Any], path: str | Path) -> str | None:
    path_s = str(path)
    key = ledger.get("path_index", {}).get(path_s)
    if key:
        return key
    digest = sha256_file(path_s)
    if digest:
        return f"sha256:{digest}"
    return None


def find_by_sha256(sha256: str) -> dict[str, Any] | None:
    ledger = _load_ledger()
    artifact = ledger["artifacts"].get(f"sha256:{sha256}")
    return artifact if isinstance(artifact, dict) else None


def find_by_path(path: str | Path) -> dict[str, Any] | None:
    ledger = _load_ledger()
    key = _find_artifact_key_for_path(ledger, path)
    if not key:
        return None
    artifact = ledger["artifacts"].get(key)
    return artifact if isinstance(artifact, dict) else None


def record_auto_assessment(path: str | Path, assessment: str) -> dict[str, Any]:
    ledger = _load_ledger()
    key = _find_artifact_key_for_path(ledger, path)
    if not key:
        digest = sha256_file(path)
        if not digest:
            raise ValueError(f"media_ledger: cannot record assessment for unknown path: {path}")
        key = f"sha256:{digest}"
    artifact = ledger["artifacts"].setdefault(
        key,
        {
            "artifact_key": key,
            "sha256": key.removeprefix("sha256:"),
            "platform": "",
            "chat_id": "",
            "message_ids": [],
            "media_group_ids": [],
            "telegram_file_unique_ids": [],
            "paths": [],
            "media_type": "",
            "states": {"received": False, "auto_assessed": False, "chronicle_logged": False, "needs_human_context": False},
            "auto_assessment": None,
            "chronicle_refs": [],
            "created_at": _now(),
        },
    )
    artifact.setdefault("paths", [])
    _append_unique(artifact["paths"], path)
    ledger["path_index"][str(path)] = key
    artifact.setdefault("states", {})["auto_assessed"] = True
    artifact["auto_assessment"] = assessment
    artifact["updated_at"] = _now()
    _save_ledger(ledger)
    return artifact


def get_auto_assessment(path: str | Path) -> str | None:
    artifact = find_by_path(path)
    if not artifact:
        return None
    if not artifact.get("states", {}).get("auto_assessed"):
        return None
    assessment = artifact.get("auto_assessment")
    return assessment if isinstance(assessment, str) and assessment else None


def mark_chronicle_logged(path_or_sha: str | Path, note_ref: str | None = None) -> dict[str, Any]:
    ledger = _load_ledger()
    raw = str(path_or_sha)
    key = raw if raw.startswith("sha256:") else None
    if not key and len(raw) == 64 and all(c in "0123456789abcdefABCDEF" for c in raw):
        key = f"sha256:{raw.lower()}"
    if not key:
        key = _find_artifact_key_for_path(ledger, raw)
    if not key or key not in ledger["artifacts"]:
        raise KeyError(f"media_ledger: artifact not found: {path_or_sha}")
    artifact = ledger["artifacts"][key]
    artifact.setdefault("states", {})["chronicle_logged"] = True
    artifact.setdefault("chronicle_refs", [])
    _append_unique(artifact["chronicle_refs"], note_ref)
    artifact["updated_at"] = _now()
    _save_ledger(ledger)
    return artifact
