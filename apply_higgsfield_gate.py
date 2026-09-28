#!/usr/bin/env python3
"""
apply_higgsfield_gate.py — surgical patcher for model_tools.py.

Two edits, both anchored on exact existing text (str-replace; fails LOUDLY and
changes nothing if an anchor isn't found — safe on a security file):

  1. _SUPERVISOR_TOOL_ACTION_MAP gains the Higgsfield generation + upload keys.
  2. _maybe_apply_supervisor_gate gains the code spend/upload gate, inserted
     right after action-resolution and before the supervisor import, reusing the
     file's own _supervisor_append_decision / _supervisor_block_result helpers.

Usage:
    python3 apply_higgsfield_gate.py /path/to/model_tools.py        # writes .new beside it
    python3 apply_higgsfield_gate.py /path/to/model_tools.py --inplace
Then DIFF the .new before moving it into place.
"""
import sys

# ---- Edit 1: action map -----------------------------------------------------
MAP_ANCHOR = '''    "mcp_era_knowledge__reset_pack_questions": "edit_config",
}'''

MAP_REPLACEMENT = '''    "mcp_era_knowledge__reset_pack_questions": "edit_config",
    # --- Higgsfield generation: private; only references media already inside
    # Higgsfield. MEDIUM-reviewed (supervisor reads the prompt for consent/scope)
    # + code spend gate (artifact_spend). Explicit map wins over the fragment set.
    "mcp_higgsfield_generate_image": "external_generation",
    "mcp_higgsfield_generate_video": "external_generation",
    "mcp_higgsfield_generate_audio": "external_generation",
    "mcp_higgsfield_generate_3d": "external_generation",
    "mcp_higgsfield_outpaint_image": "external_generation",
    "mcp_higgsfield_upscale_image": "external_generation",
    "mcp_higgsfield_upscale_video": "external_generation",
    "mcp_higgsfield_reframe": "external_generation",
    "mcp_higgsfield_remove_background": "external_generation",
    "mcp_higgsfield_motion_control": "external_generation",
    "mcp_higgsfield_dubbing": "external_generation",
    "mcp_higgsfield_voice_change": "external_generation",
    # --- Fresh media ENTERING Higgsfield: hard-blocked in code below. Lorenzo
    # never uploads; Sean uploads and supplies the media_id.
    "mcp_higgsfield_media_upload": "external_media_upload",
    "mcp_higgsfield_media_import_url": "external_media_upload",
    "mcp_higgsfield_media_upload_widget": "external_media_upload",
    "mcp_higgsfield_media_confirm": "external_media_upload",
}'''

# ---- Edit 2: code gate in _maybe_apply_supervisor_gate ----------------------
GATE_ANCHOR = '''        if action is None:
            return None

        # From here onward, the tool is in a potentially risky bucket.  Import'''

GATE_BLOCK = '''        if action is None:
            return None

        # --- Higgsfield artifact gate: code-enforced, runs BEFORE the LLM review.
        # Uploads hard-block; generation passes the spend ceiling, then the
        # MEDIUM supervisor review reads the prompt against the SOUL consent ledger.
        if action == "external_media_upload":
            _supervisor_append_decision({
                "action": action, "risk": "high", "tool": function_name,
                "verdict": "REJECT", "blocking": True,
                "reasoning": "Lorenzo does not upload media; Sean uploads and supplies the media_id.",
                "issues": ["external_media_upload hard-blocked in code"],
                "wrapper_enforced": True,
            })
            return _supervisor_block_result(
                function_name=function_name, action=action, risk="high",
                verdict="REJECT",
                reason=("Lorenzo doesn't upload media. Describe it in text, or Sean "
                        "uploads it and gives you the media_id."),
            )
        if action == "external_generation":
            try:
                import artifact_spend
                _spend = artifact_spend.evaluate_and_record(
                    tool_name=function_name, function_args=function_args,
                )
                _spend_allowed = bool(_spend.allowed)
                _spend_reason = _spend.reason
                _spend_log = _spend.as_log()
            except Exception as _spend_exc:  # fail closed on spend-gate error
                _spend_allowed = False
                _spend_reason = f"spend gate error: {type(_spend_exc).__name__}: {_spend_exc}"
                _spend_log = None
            if not _spend_allowed:
                _supervisor_append_decision({
                    "action": action, "risk": "medium", "tool": function_name,
                    "verdict": "REJECT", "blocking": True,
                    "reasoning": _spend_reason, "issues": [_spend_reason],
                    "wrapper_enforced": True, "spend": _spend_log,
                })
                return _supervisor_block_result(
                    function_name=function_name, action=action, risk="medium",
                    verdict="REJECT", reason=_spend_reason,
                )
            # allowed -> fall through to the MEDIUM supervisor review below.

        # From here onward, the tool is in a potentially risky bucket.  Import'''


def patch(text: str) -> str:
    if MAP_ANCHOR not in text:
        raise SystemExit("ERROR: action-map anchor not found — model_tools.py differs; aborting (no change).")
    if text.count(MAP_ANCHOR) != 1:
        raise SystemExit("ERROR: action-map anchor not unique; aborting.")
    if GATE_ANCHOR not in text:
        raise SystemExit("ERROR: gate anchor not found — model_tools.py differs; aborting (no change).")
    if text.count(GATE_ANCHOR) != 1:
        raise SystemExit("ERROR: gate anchor not unique; aborting.")
    if '"external_generation"' in text:
        raise SystemExit("ERROR: 'external_generation' already present — looks already patched; aborting.")
    text = text.replace(MAP_ANCHOR, MAP_REPLACEMENT, 1)
    text = text.replace(GATE_ANCHOR, GATE_BLOCK, 1)
    return text


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: apply_higgsfield_gate.py /path/to/model_tools.py [--inplace]")
    path = sys.argv[1]
    inplace = "--inplace" in sys.argv[2:]
    with open(path, "r") as f:
        original = f.read()
    patched = patch(original)
    out = path if inplace else path + ".new"
    with open(out, "w") as f:
        f.write(patched)
    print(f"OK: wrote {out}  (+{patched.count(chr(10)) - original.count(chr(10))} lines)")
    print("Next: diff it, then move into place.")


if __name__ == "__main__":
    main()
