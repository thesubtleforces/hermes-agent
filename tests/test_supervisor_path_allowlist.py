"""Supervisor path-allowlist matcher tests.

Covers the security-critical containment plus fail-closed handling of every
field. Run from the worktree root:

    /home/transversed/.hermes/hermes-agent/venv/bin/python -m pytest \
        tests/test_supervisor_path_allowlist.py -q
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model_tools  # noqa: E402

match = model_tools._supervisor_path_allowlist_match

STAGING = "/home/transversed/.hermes/staging/lorenzo/"


def _entry(**over):
    """A fully-valid baseline allowlist entry; override one field to test it."""
    e = {
        "initiative_id": "safe-staging-2026-0618",
        "enabled": True,
        "tools": ["write_file"],
        "actions": ["edit_handler"],
        "path_prefix": STAGING,
        "expires_at": "2999-01-01T00:00:00Z",
        "max_bytes_per_file": 25000,
        "forbid_targets": ["config.yaml", ".env", "AGENTS.md"],
        "reason": "test",
    }
    e.update(over)
    return e


def _call(entries, path, content="x", *, function_name="write_file", action="edit_handler"):
    return match(
        function_name=function_name,
        action=action,
        function_args={"path": path, "content": content},
        entries=entries,
    )


# ---- happy path -----------------------------------------------------------
def test_match_under_prefix():
    r = _call([_entry()], STAGING + "foo.py")
    assert r is not None
    assert r["initiative_id"] == "safe-staging-2026-0618"
    assert r["payload_bytes"] == 1


def test_multiple_entries_first_valid_match_wins():
    bad = _entry(enabled=False)
    good = _entry(initiative_id="second")
    r = _call([bad, good], STAGING + "foo.py")
    assert r is not None and r["initiative_id"] == "second"


# ---- containment (the part that must never be wrong) ----------------------
def test_dotdot_traversal_blocked():
    assert _call([_entry()], STAGING + "../../.hermes/config.yaml") is None


def test_outside_prefix_blocked():
    assert _call([_entry()], "/home/transversed/.hermes/scripts/x.py") is None


def test_sibling_prefix_substring_blocked():
    # /staging/lorenzo-evil must NOT count as under /staging/lorenzo
    assert _call([_entry()], "/home/transversed/.hermes/staging/lorenzo-evil/x.py") is None


def test_prefix_directory_itself_not_writable():
    assert _call([_entry()], STAGING.rstrip("/")) is None


def test_symlink_escape_blocked(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = staging / "escape"
    link.symlink_to(outside, target_is_directory=True)
    entry = _entry(path_prefix=str(staging))
    # writing "through" the symlink resolves outside staging -> blocked
    assert _call([entry], str(link / "stolen.txt")) is None


def test_real_file_under_staging_tmp(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    entry = _entry(path_prefix=str(staging))
    assert _call([entry], str(staging / "ok.py")) is not None


# ---- enable / tool / action gates -----------------------------------------
def test_disabled_entry_blocked():
    assert _call([_entry(enabled=False)], STAGING + "foo.py") is None


def test_missing_enabled_blocked():
    e = _entry()
    del e["enabled"]
    assert _call([e], STAGING + "foo.py") is None


def test_enabled_must_be_true_not_truthy():
    assert _call([_entry(enabled=1)], STAGING + "foo.py") is None


def test_wrong_attempted_tool_blocked():
    assert _call([_entry()], STAGING + "foo.py", function_name="run_command") is None


def test_tool_not_in_entry_blocked():
    assert _call([_entry(tools=["patch"])], STAGING + "foo.py") is None


def test_wrong_attempted_action_blocked():
    assert _call([_entry()], STAGING + "foo.py", action="run_command") is None


def test_action_not_in_entry_blocked():
    assert _call([_entry(actions=["something_else"])], STAGING + "foo.py") is None


# ---- expiry (required) ----------------------------------------------------
def test_expired_blocked():
    assert _call([_entry(expires_at="2000-01-01T00:00:00Z")], STAGING + "foo.py") is None


def test_missing_expiry_blocked():
    e = _entry()
    del e["expires_at"]
    assert _call([e], STAGING + "foo.py") is None


def test_unparseable_expiry_blocked():
    assert _call([_entry(expires_at="whenever")], STAGING + "foo.py") is None


# ---- byte cap (required + enforced) ---------------------------------------
def test_over_cap_blocked():
    assert _call([_entry(max_bytes_per_file=10)], STAGING + "foo.py", content="A" * 11) is None


def test_at_cap_allowed():
    assert _call([_entry(max_bytes_per_file=10)], STAGING + "foo.py", content="A" * 10) is not None


def test_missing_cap_blocked_fail_closed():
    e = _entry()
    del e["max_bytes_per_file"]
    assert _call([e], STAGING + "foo.py") is None


def test_nonint_cap_blocked():
    assert _call([_entry(max_bytes_per_file="lots")], STAGING + "foo.py") is None


def test_bytes_content_counted_by_length():
    # 11 raw bytes over a 10 cap -> blocked (not counted as the b'...' repr)
    assert _call([_entry(max_bytes_per_file=10)], STAGING + "f", content=b"A" * 11) is None


# ---- forbid targets -------------------------------------------------------
def test_entry_forbidden_basename_blocked():
    assert _call([_entry()], STAGING + "config.yaml") is None


def test_always_forbid_floor_even_without_entry_list():
    e = _entry()
    del e["forbid_targets"]
    assert _call([e], STAGING + ".env") is None  # floor still denies it


def test_soul_md_floor_blocked():
    e = _entry()
    del e["forbid_targets"]
    assert _call([e], STAGING + "SOUL.md") is None


# ---- malformed input fails closed, never crashes --------------------------
def test_entries_not_a_list():
    assert match(function_name="write_file", action="edit_handler",
                 function_args={"path": STAGING + "f", "content": "x"}, entries={"a": 1}) is None


def test_non_dict_entry_skipped():
    assert _call([None, "nope", _entry()], STAGING + "foo.py") is not None  # skips junk, matches real


def test_function_args_not_dict():
    assert match(function_name="write_file", action="edit_handler",
                 function_args="not a dict", entries=[_entry()]) is None


def test_path_missing():
    assert match(function_name="write_file", action="edit_handler",
                 function_args={"content": "x"}, entries=[_entry()]) is None


def test_empty_entries():
    assert _call([], STAGING + "foo.py") is None

# ---- runtime supervisor wrapper wiring ------------------------------------
class _FakeSupervisor:
    class Risk:
        MEDIUM = "medium"
        HIGH = "high"

    @staticmethod
    def risk_for(action):
        return _FakeSupervisor.Risk.HIGH


class _FakeReview:
    verdict = "REJECT"
    blocking = True
    reasoning = "fake reject"
    issues = ["fake issue"]


class _FakeGateBlocked(Exception):
    def __init__(self):
        self.review = _FakeReview()


def _install_fake_gate(monkeypatch, calls):
    class FakeGate:
        GateBlocked = _FakeGateBlocked

        @staticmethod
        def gate(**kwargs):
            calls.append(kwargs)
            raise _FakeGateBlocked()

    monkeypatch.setattr(
        model_tools,
        "_supervisor_import_modules",
        lambda: (_FakeSupervisor, FakeGate),
    )


def _runtime_call(monkeypatch, *, path=STAGING + "runtime.py", allowlist=None):
    calls = []
    decisions = []
    _install_fake_gate(monkeypatch, calls)
    monkeypatch.setenv("SUPERVISOR_GATE_ENABLED", "1")
    monkeypatch.setattr(model_tools, "_supervisor_append_decision", decisions.append)
    if allowlist is None:
        monkeypatch.delenv("SUPERVISOR_PATH_ALLOWLIST", raising=False)
    else:
        monkeypatch.setenv("SUPERVISOR_PATH_ALLOWLIST", json.dumps(allowlist))
    result = model_tools._maybe_apply_supervisor_gate(
        function_name="write_file",
        function_args={"path": path, "content": "x"},
        user_task="approved staging write",
        task_id="task",
        session_id="session",
        tool_call_id="tool-call",
        turn_id="turn",
        api_request_id="api",
    )
    return result, calls, decisions


def test_runtime_gate_disabled_is_noop(monkeypatch):
    calls = []
    _install_fake_gate(monkeypatch, calls)
    monkeypatch.delenv("SUPERVISOR_GATE_ENABLED", raising=False)
    result = model_tools._maybe_apply_supervisor_gate(
        function_name="write_file",
        function_args={"path": STAGING + "runtime.py", "content": "x"},
        user_task="approved staging write",
        task_id=None,
        session_id=None,
        tool_call_id=None,
        turn_id=None,
        api_request_id=None,
    )
    assert result is None
    assert calls == []


def test_runtime_allowlisted_write_skips_external_gate(monkeypatch):
    result, calls, decisions = _runtime_call(monkeypatch, allowlist=[_entry()])
    assert result is None
    assert calls == []
    assert decisions[0]["verdict"] == "APPROVE"
    assert decisions[0]["blocking"] is False
    assert decisions[0]["allowlist_match"]["resolved_path"].endswith("runtime.py")


def test_runtime_nonallowlisted_write_calls_external_gate_and_blocks(monkeypatch):
    result, calls, decisions = _runtime_call(
        monkeypatch,
        path="/home/transversed/.hermes/scripts/not-allowed.py",
        allowlist=[_entry()],
    )
    assert calls and calls[0]["action"] == "edit_handler"
    assert '"status": "blocked"' in result
    assert decisions[0]["verdict"] == "REJECT"
    assert decisions[0]["blocking"] is True


def test_runtime_malformed_allowlist_fails_closed_to_external_gate(monkeypatch):
    calls = []
    decisions = []
    _install_fake_gate(monkeypatch, calls)
    monkeypatch.setenv("SUPERVISOR_GATE_ENABLED", "1")
    monkeypatch.setenv("SUPERVISOR_PATH_ALLOWLIST", "not json")
    monkeypatch.setattr(model_tools, "_supervisor_append_decision", decisions.append)
    result = model_tools._maybe_apply_supervisor_gate(
        function_name="write_file",
        function_args={"path": STAGING + "runtime.py", "content": "x"},
        user_task="approved staging write",
        task_id=None,
        session_id=None,
        tool_call_id=None,
        turn_id=None,
        api_request_id=None,
    )
    assert calls and calls[0]["action"] == "edit_handler"
    assert '"status": "blocked"' in result
    assert decisions[0]["verdict"] == "REJECT"


def test_runtime_allowlist_does_not_apply_to_other_tools(monkeypatch):
    calls = []
    decisions = []
    _install_fake_gate(monkeypatch, calls)
    monkeypatch.setenv("SUPERVISOR_GATE_ENABLED", "1")
    monkeypatch.setenv(
        "SUPERVISOR_PATH_ALLOWLIST",
        json.dumps([_entry(tools=["terminal"], actions=["run_command"])]),
    )
    monkeypatch.setattr(model_tools, "_supervisor_append_decision", decisions.append)
    result = model_tools._maybe_apply_supervisor_gate(
        function_name="terminal",
        function_args={"command": "printf ok"},
        user_task="run a command",
        task_id=None,
        session_id=None,
        tool_call_id=None,
        turn_id=None,
        api_request_id=None,
    )
    assert calls and calls[0]["action"] == "run_command"
    assert '"status": "blocked"' in result
    assert decisions[0]["verdict"] == "REJECT"
