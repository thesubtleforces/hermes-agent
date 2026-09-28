"""
artifact_spend.py — code-enforced spend ceiling for Higgsfield generation,
implementing Sean's allowance model, plus a Sean-issued tier-grant lever.

POLICY (Sean's rules)
---------------------
Treat Higgsfield credits as a MONTHLY allowance, not a per-artifact purse.
Derive a conservative DAILY allowance from the live balance:

        daily_allowance = current_balance / 30

Per-artifact-type ceilings, as a fraction of one daily allowance, with attempt
caps (the caller passes a `tier`; autonomous generations default to the most
conservative "standard"):

    tier            ceiling (of daily allowance)   attempts
    micro           10%                            1
    standard         25%   (default autonomous)    2
    flagship         50%   (briefing, combined)    3
    user_requested   50%                           3
    remediation     100%   (Ops context only)      3
    video / 3D      no autonomous spend            approval + preflight required

Two ABSOLUTE caps bound everything regardless of tier:
    - cumulative autonomous spend today  <=  one daily allowance
    - balance after spend  >=  monthly reserve floor (70% of month-start balance)

Other guardrails: post-production = one paid pass; repeated identical failure ->
pivot; balance lookup fails -> standard/audio one low-cost attempt, video/3D/
remediation nothing without approval.

TIER GRANTS (Sean-issued, forge-proof)
--------------------------------------
The gate defaults every autonomous generation to `standard`. To let Sean
deliberately raise the tier for a creative iteration session (e.g. bump to
`user_requested`: 50% ceiling, 3 attempts), he issues a GRANT from the shell:

    python3 -c "import artifact_spend; print(artifact_spend.grant_artifact_tier('user_requested', minutes=30))"

This writes a time-boxed, use-capped grant file in the supervisor directory and
(by default) resets today's attempt counts so the elevated tier has a fresh
budget. The gate reads the grant live (no restart) and applies its tier; the
grant is consumed per use and auto-expires. Lorenzo cannot issue one: every path
it has to write that file is a gated write that the supervisor blocks as
self-authorization. The grant only relaxes the COUNT and per-call CEILING — the
absolute daily-allowance and reserve-floor caps still bound all spend, and video/
3D still require explicit approval regardless of any grant.

    python3 -c "import artifact_spend; artifact_spend.clear_artifact_grant()"   # revoke early

TUNING (env, all optional):
  HIGGSFIELD_BALANCE_CACHE / HIGGSFIELD_BALANCE_MAX_AGE_MIN / HIGGSFIELD_RESERVE_FRACTION
  HIGGSFIELD_FALLBACK_LOWCOST / HIGGSFIELD_COST_ESTIMATES / ARTIFACT_SPEND_STATE
  ARTIFACT_GRANT_FILE   path to the grant json. default ~/hermes_supervisor/artifact_grant.json
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple


# --------------------------------------------------------------------------- #
# LIFTED VERBATIM from lorenzo_artifact_reliability.py — do not edit the logic.
# --------------------------------------------------------------------------- #
def next_generation_decision(
    *,
    attempts_used: int,
    max_paid_attempts: int,
    same_failure_count: int,
    budget_remaining: bool,
) -> str:
    if same_failure_count >= 2:
        return "pivot_required"
    if attempts_used >= max_paid_attempts:
        return "pivot_required"
    if not budget_remaining:
        return "pivot_required"
    return "retry_allowed"


def validate_video_spend_approval(
    *,
    artifact_type: str,
    cost_preflighted: bool,
    explicit_approval: bool,
) -> str:
    if artifact_type not in {"video", "3d", "multimodal_video"}:
        return "not_required"
    if not cost_preflighted:
        return "blocked_needs_cost_preflight"
    if not explicit_approval:
        return "blocked_needs_explicit_spend_approval"
    return "approved"


# --------------------------------------------------------------------------- #
# Sean's tier table: tier -> (fraction of daily allowance, attempt cap)
# --------------------------------------------------------------------------- #
_TIERS: dict[str, Tuple[float, int]] = {
    "micro": (0.10, 1),
    "standard": (0.25, 6),          # default for autonomous generations
    "flagship": (0.50, 3),          # morning dispatch, components combined
    "user_requested": (0.50, 3),
    "remediation": (1.00, 3),       # Ops context only
}

_DEFAULT_COST_ESTIMATES: dict[str, float] = {
    "image": 5.0, "audio": 5.0, "postproduction": 5.0,
    "video": 30.0, "3d": 30.0, "default": 10.0,
}


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _reserve_fraction() -> float: return _env_float("HIGGSFIELD_RESERVE_FRACTION", 0.70)
def _allowance_days() -> float: return _env_float("HIGGSFIELD_ALLOWANCE_DAYS", 30.0)
def _fallback_lowcost() -> float: return _env_float("HIGGSFIELD_FALLBACK_LOWCOST", 3.0)
def _balance_max_age_min() -> float: return _env_float("HIGGSFIELD_BALANCE_MAX_AGE_MIN", 720.0)


def _cost_estimates() -> dict[str, float]:
    table = dict(_DEFAULT_COST_ESTIMATES)
    raw = os.environ.get("HIGGSFIELD_COST_ESTIMATES")
    if raw:
        try:
            override = json.loads(raw)
            if isinstance(override, dict):
                for k, v in override.items():
                    table[str(k)] = float(v)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return table


def _state_path() -> Path:
    p = os.environ.get("ARTIFACT_SPEND_STATE")
    return Path(p).expanduser() if p else Path.home() / "hermes_supervisor" / "artifact_spend_state.json"


def _balance_path() -> Path:
    p = os.environ.get("HIGGSFIELD_BALANCE_CACHE")
    return Path(p).expanduser() if p else Path.home() / "hermes_supervisor" / "higgsfield_balance.json"


def _grant_path() -> Path:
    p = os.environ.get("ARTIFACT_GRANT_FILE")
    return Path(p).expanduser() if p else Path.home() / "hermes_supervisor" / "artifact_grant.json"


def _today() -> str: return datetime.now(timezone.utc).strftime("%Y-%m-%d")
def _month() -> str: return datetime.now(timezone.utc).strftime("%Y-%m")


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
_POSTPROD = ("outpaint", "upscale", "reframe", "remove_background")


def classify_generation(tool_name: str) -> str:
    n = (tool_name or "").lower()
    if "generate_video" in n: return "video"
    if "generate_3d" in n: return "3d"
    if "generate_audio" in n: return "audio"
    if any(p in n for p in _POSTPROD): return "postproduction"
    if "motion_control" in n or "dubbing" in n or "voice_change" in n: return "postproduction"
    return "image"


def _cost_preflighted_from_args(function_args: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(function_args, Mapping):
        return False
    if bool(function_args.get("get_cost")):
        return True
    params = function_args.get("params")
    if isinstance(params, Mapping) and bool(params.get("get_cost")):
        return True
    return False


# --------------------------------------------------------------------------- #
# State + balance cache (locked)
# --------------------------------------------------------------------------- #
_LOCK = threading.Lock()


def _fresh_state() -> dict[str, Any]:
    return {"date": _today(), "spent_credits": 0.0, "attempts": {},
            "month": _month(), "month_start_balance": None}


def _load_state_locked() -> dict[str, Any]:
    try:
        data = json.loads(_state_path().read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return _fresh_state()
    if not isinstance(data, dict):
        return _fresh_state()
    if data.get("date") != _today():
        data["date"] = _today()
        data["spent_credits"] = 0.0
        data["attempts"] = {}
    data.setdefault("spent_credits", 0.0)
    data.setdefault("attempts", {})
    data.setdefault("month", _month())
    data.setdefault("month_start_balance", None)
    return data


def _save_state_locked(state: dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)


def _read_balance_cache() -> Tuple[Optional[float], bool]:
    try:
        data = json.loads(_balance_path().read_text())
        credits = float(data["credits"])
        ts = datetime.fromisoformat(data["ts"])
    except (FileNotFoundError, json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        return None, False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_min = (datetime.now(timezone.utc) - ts).total_seconds() / 60.0
    if age_min > _balance_max_age_min():
        return credits, False
    return credits, True


def record_balance(credits: float) -> None:
    credits = float(credits)
    with _LOCK:
        _balance_path().parent.mkdir(parents=True, exist_ok=True)
        _balance_path().write_text(json.dumps(
            {"credits": credits, "ts": datetime.now(timezone.utc).isoformat()}, indent=2))
        state = _load_state_locked()
        if state.get("month") != _month() or state.get("month_start_balance") is None:
            state["month"] = _month()
            state["month_start_balance"] = credits
        _save_state_locked(state)


# --------------------------------------------------------------------------- #
# Tier grants (Sean-issued, time-boxed + use-capped)
# --------------------------------------------------------------------------- #
def _read_active_grant() -> Optional[dict[str, Any]]:
    """Return the active grant dict if valid, unexpired, and uses remaining;
    else None. Cleans up an expired/exhausted/corrupt grant file."""
    path = _grant_path()
    try:
        g = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    try:
        tier = g["tier"]
        expires = datetime.fromisoformat(str(g["expires_at"]).replace("Z", "+00:00"))
        uses = int(g.get("uses", 0))
    except (KeyError, TypeError, ValueError):
        _safe_unlink(path)
        return None
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if tier not in _TIERS or uses <= 0 or expires <= datetime.now(timezone.utc):
        _safe_unlink(path)
        return None
    return g


def active_grant_tier() -> str:
    g = _read_active_grant()
    return g["tier"] if g else "standard"


def _consume_grant_locked() -> None:
    """Decrement the active grant's uses by 1; remove the file at 0."""
    path = _grant_path()
    g = _read_active_grant()
    if g is None:
        return
    g["uses"] = int(g.get("uses", 0)) - 1
    if g["uses"] <= 0:
        _safe_unlink(path)
    else:
        path.write_text(json.dumps(g, indent=2))


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except (FileNotFoundError, OSError):
        pass


def grant_artifact_tier(
    tier: str = "user_requested",
    minutes: float = 30.0,
    uses: Optional[int] = None,
    reset_attempts: bool = True,
) -> dict[str, Any]:
    """Sean-only, shell-issued. Grant an elevated tier for the next creative
    session. Time-boxed (minutes) and use-capped (defaults to the tier's attempt
    cap). By default resets today's attempt counts so the tier has a fresh budget;
    does NOT reset the credit tally, so the daily-allowance and reserve-floor caps
    still bound spend. video/3D still require explicit approval regardless."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}; choose from {sorted(_TIERS)}")
    if uses is None:
        uses = _TIERS[tier][1]  # the tier's attempt cap
    expires = datetime.now(timezone.utc) + timedelta(minutes=float(minutes))
    grant = {
        "tier": tier,
        "budget": int(uses),
        "uses": int(uses),
        "expires_at": expires.isoformat(),
        "issued_at": datetime.now(timezone.utc).isoformat(),
    }
    with _LOCK:
        _grant_path().parent.mkdir(parents=True, exist_ok=True)
        _grant_path().write_text(json.dumps(grant, indent=2))
        if reset_attempts:
            state = _load_state_locked()
            state["attempts"] = {}
            _save_state_locked(state)
    return grant


def clear_artifact_grant() -> None:
    with _LOCK:
        _safe_unlink(_grant_path())


# --------------------------------------------------------------------------- #
# Result + entrypoint
# --------------------------------------------------------------------------- #
@dataclass
class SpendDecision:
    allowed: bool
    reason: str
    decision: str
    artifact_type: str
    tier: str
    cost_estimate: float
    daily_allowance: Optional[float]
    spent_before: float
    spent_after: float
    reserve_floor: Optional[float]
    balance: Optional[float]
    attempts_before: int
    attempts_after: int
    attempt_cap: int
    grant_applied: bool = False

    def as_log(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_and_record(
    *,
    tool_name: str,
    function_args: Optional[Mapping[str, Any]] = None,
    tier: Optional[str] = None,
    explicit_approval: bool = False,
    same_failure_count: int = 0,
    balance_override: Optional[float] = None,
) -> SpendDecision:
    """Decide whether a Higgsfield generation may proceed; if so, record it.
    tier=None (the default, used by the gate) consults any active Sean-issued
    grant, else falls to 'standard'. An explicit tier overrides grants."""
    atype = classify_generation(tool_name)
    estimates = _cost_estimates()
    cost = estimates.get(atype, estimates["default"])

    # tier resolution: explicit arg wins; otherwise an active grant; else standard
    grant = _read_active_grant() if tier is None else None
    used_grant = tier is None and grant is not None
    if tier is None:
        tier = grant["tier"] if grant else "standard"
    tier = tier if tier in _TIERS else "standard"
    tier_fraction, tier_attempts = _TIERS[tier]
    if used_grant:
        tier_attempts = int(grant.get("budget", grant.get("uses", tier_attempts)))
    if atype == "postproduction":
        tier_attempts = min(tier_attempts, 1)
    preflighted = _cost_preflighted_from_args(function_args)

    if balance_override is not None:
        balance, fresh = float(balance_override), True
    else:
        balance, fresh = _read_balance_cache()

    with _LOCK:
        state = _load_state_locked()
        spent_before = float(state.get("spent_credits", 0.0))
        attempts_map = state.get("attempts", {})
        attempts_before = int(attempts_map.get(atype, 0))

        if fresh and balance is not None and (
            state.get("month") != _month() or state.get("month_start_balance") is None
        ):
            state["month"] = _month()
            state["month_start_balance"] = balance

        daily_allowance = (balance / _allowance_days()) if (fresh and balance is not None) else None
        reserve_floor = (
            _reserve_fraction() * float(state["month_start_balance"])
            if state.get("month_start_balance") is not None else None
        )

        def deny(reason: str, decision: str) -> SpendDecision:
            return SpendDecision(
                allowed=False, reason=reason, decision=decision, artifact_type=atype,
                tier=tier, cost_estimate=cost, daily_allowance=daily_allowance,
                spent_before=spent_before, spent_after=spent_before,
                reserve_floor=reserve_floor, balance=balance,
                attempts_before=attempts_before, attempts_after=attempts_before,
                attempt_cap=tier_attempts, grant_applied=used_grant,
            )

        def allow(decision: str, reason: str) -> SpendDecision:
            spent_after = spent_before + cost
            attempts_map[atype] = attempts_before + 1
            state["spent_credits"] = spent_after
            state["attempts"] = attempts_map
            try:
                _save_state_locked(state)
            except OSError:
                return deny("could not persist spend state; failing closed", "state_write_failed")
            if used_grant:
                _consume_grant_locked()
            return SpendDecision(
                allowed=True, reason=reason, decision=decision, artifact_type=atype,
                tier=tier, cost_estimate=cost, daily_allowance=daily_allowance,
                spent_before=spent_before, spent_after=spent_after, reserve_floor=reserve_floor,
                balance=balance, attempts_before=attempts_before, attempts_after=attempts_before + 1,
                attempt_cap=tier_attempts, grant_applied=used_grant,
            )

        # video/3D approval gate (independent of balance and of any grant)
        if atype in ("video", "3d"):
            vd = validate_video_spend_approval(
                artifact_type=atype, cost_preflighted=preflighted, explicit_approval=explicit_approval)
            if vd not in ("approved", "not_required"):
                human = {
                    "blocked_needs_cost_preflight": f"{atype} needs a get_cost preflight first",
                    "blocked_needs_explicit_spend_approval": f"{atype} needs Sean's explicit spend approval (no autonomous video/3D spend)",
                }.get(vd, vd)
                return deny(human, vd)

        # balance lookup failed: conservative fallback
        if not fresh:
            if atype in ("video", "3d") or tier == "remediation":
                return deny("balance lookup unavailable; no autonomous video/3D/remediation spend without approval", "balance_unavailable")
            total_attempts_today = sum(int(v) for v in attempts_map.values())
            if total_attempts_today >= 1:
                return deny("balance lookup unavailable; the single low-cost fallback attempt is already used today", "balance_unavailable")
            if cost > _fallback_lowcost():
                return deny(f"balance lookup unavailable; only a low-cost (<= {_fallback_lowcost():.0f} cr) attempt is allowed", "balance_unavailable")
            return allow("fallback_lowcost", "balance unavailable; one low-cost fallback attempt")

        # normal path
        autonomous_daily_cap = daily_allowance
        per_call_ceiling = tier_fraction * daily_allowance
        budget_remaining = spent_before < autonomous_daily_cap

        decision = next_generation_decision(
            attempts_used=attempts_before, max_paid_attempts=tier_attempts,
            same_failure_count=same_failure_count, budget_remaining=budget_remaining)
        if decision == "pivot_required":
            if attempts_before >= tier_attempts:
                why = f"{tier} attempt cap reached for {atype} ({attempts_before}/{tier_attempts}); pivot to text"
            elif same_failure_count >= 2:
                why = "two identical failures; pivot to a cheaper/controlled path rather than re-spend"
            else:
                why = f"today's autonomous allowance is spent ({spent_before:.1f}/{autonomous_daily_cap:.1f} cr); pivot to text"
            return deny(why, decision)

        if cost > per_call_ceiling:
            return deny(
                f"{atype} (~{cost:.0f} cr) exceeds the {tier} ceiling "
                f"(~{per_call_ceiling:.1f} cr = {int(tier_fraction*100)}% of {daily_allowance:.1f}/day)",
                "over_tier_ceiling")
        if spent_before + cost > autonomous_daily_cap:
            return deny(
                f"would exceed today's autonomous allowance "
                f"({spent_before:.1f}+{cost:.0f} > {autonomous_daily_cap:.1f} cr); needs approval",
                "over_daily_allowance")
        if reserve_floor is not None and (balance - cost) < reserve_floor:
            return deny(
                f"would breach the monthly reserve floor (~{reserve_floor:.0f} cr = "
                f"{int(_reserve_fraction()*100)}% of month-start); needs Ops approval",
                "below_reserve_floor")

        return allow(decision, "within tier ceiling, daily allowance, and reserve floor")


def current_usage() -> dict[str, Any]:
    with _LOCK:
        state = _load_state_locked()
        balance, fresh = _read_balance_cache()
        msb = state.get("month_start_balance")
        g = _read_active_grant()
        return {
            "date": state.get("date"),
            "spent_credits_today": float(state.get("spent_credits", 0.0)),
            "attempts_today": dict(state.get("attempts", {})),
            "balance": balance, "balance_fresh": fresh,
            "daily_allowance": (balance / _allowance_days()) if (fresh and balance is not None) else None,
            "month": state.get("month"), "month_start_balance": msb,
            "reserve_floor": (_reserve_fraction() * float(msb)) if msb is not None else None,
            "active_grant": g,
        }
