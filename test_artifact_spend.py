"""
test_artifact_spend.py — proves Sean's allowance/tier/reserve-floor model.

Run:  python3 -m unittest test_artifact_spend -v

Covers the deterministic half (what code decides). The LLM half (consent ledger
/ out-of-scope depiction) is proven by the two-direction live test on deploy.
"""

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = str(Path(self.tmp.name) / "spend.json")
        self.balance = str(Path(self.tmp.name) / "balance.json")
        self.grant = str(Path(self.tmp.name) / "grant.json")
        os.environ["ARTIFACT_SPEND_STATE"] = self.state
        os.environ["HIGGSFIELD_BALANCE_CACHE"] = self.balance
        os.environ["ARTIFACT_GRANT_FILE"] = self.grant
        os.environ["HIGGSFIELD_RESERVE_FRACTION"] = "0.70"
        os.environ["HIGGSFIELD_FALLBACK_LOWCOST"] = "3"
        os.environ["HIGGSFIELD_COST_ESTIMATES"] = json.dumps(
            {"image": 5, "audio": 5, "postproduction": 5, "video": 30, "3d": 30}
        )
        import artifact_spend
        self.mod = importlib.reload(artifact_spend)

    def tearDown(self):
        for k in ("ARTIFACT_SPEND_STATE", "HIGGSFIELD_BALANCE_CACHE",
                  "HIGGSFIELD_RESERVE_FRACTION", "HIGGSFIELD_FALLBACK_LOWCOST",
                  "HIGGSFIELD_COST_ESTIMATES", "HIGGSFIELD_BALANCE_MAX_AGE_MIN",
                  "ARTIFACT_GRANT_FILE"):
            os.environ.pop(k, None)
        self.tmp.cleanup()

    def seed_state(self, **kw):
        s = {"date": self.mod._today(), "spent_credits": 0.0, "attempts": {},
             "month": self.mod._month(), "month_start_balance": None}
        s.update(kw)
        Path(self.state).write_text(json.dumps(s))


class TestPureFunctions(Base):
    def test_next_generation_decision(self):
        f = self.mod.next_generation_decision
        self.assertEqual(f(attempts_used=0, max_paid_attempts=2, same_failure_count=0, budget_remaining=True), "retry_allowed")
        self.assertEqual(f(attempts_used=2, max_paid_attempts=2, same_failure_count=0, budget_remaining=True), "pivot_required")
        self.assertEqual(f(attempts_used=0, max_paid_attempts=2, same_failure_count=2, budget_remaining=True), "pivot_required")
        self.assertEqual(f(attempts_used=0, max_paid_attempts=2, same_failure_count=0, budget_remaining=False), "pivot_required")

    def test_validate_video(self):
        f = self.mod.validate_video_spend_approval
        self.assertEqual(f(artifact_type="image", cost_preflighted=False, explicit_approval=False), "not_required")
        self.assertEqual(f(artifact_type="video", cost_preflighted=False, explicit_approval=True), "blocked_needs_cost_preflight")
        self.assertEqual(f(artifact_type="video", cost_preflighted=True, explicit_approval=False), "blocked_needs_explicit_spend_approval")
        self.assertEqual(f(artifact_type="3d", cost_preflighted=True, explicit_approval=True), "approved")


class TestClassify(Base):
    def test_mapping(self):
        c = self.mod.classify_generation
        self.assertEqual(c("mcp_higgsfield_generate_image"), "image")
        self.assertEqual(c("mcp_higgsfield_generate_video"), "video")
        self.assertEqual(c("mcp_higgsfield_generate_3d"), "3d")
        self.assertEqual(c("mcp_higgsfield_generate_audio"), "audio")
        self.assertEqual(c("mcp_higgsfield_upscale_image"), "postproduction")
        self.assertEqual(c("mcp_higgsfield_reframe"), "postproduction")
        self.assertEqual(c("mcp_higgsfield_remove_background"), "postproduction")
        self.assertEqual(c("mcp_higgsfield_motion_control"), "postproduction")


class TestStandardImage(Base):
    def test_healthy_balance_allows_and_records(self):
        # balance 853.87 -> daily 28.46 -> standard 25% = 7.12; image 5 fits
        d = self.mod.evaluate_and_record(tool_name="mcp_higgsfield_generate_image",
                                         function_args={}, balance_override=853.87)
        self.assertTrue(d.allowed, d.reason)
        self.assertAlmostEqual(d.daily_allowance, 853.87 / 30.0, places=2)
        self.assertEqual(d.attempts_after, 1)
        self.assertEqual(d.spent_after, 5)

    def test_attempt_cap_two(self):
        for _ in range(2):
            self.assertTrue(self.mod.evaluate_and_record(
                tool_name="mcp_higgsfield_generate_image", function_args={}, balance_override=853.87).allowed)
        third = self.mod.evaluate_and_record(
            tool_name="mcp_higgsfield_generate_image", function_args={}, balance_override=853.87)
        self.assertFalse(third.allowed)
        self.assertIn("attempt cap", third.reason)


class TestCeilings(Base):
    def test_over_tier_ceiling_at_low_balance(self):
        # balance 100 -> daily 3.33 -> standard 25% = 0.83; image 5 too big
        d = self.mod.evaluate_and_record(tool_name="mcp_higgsfield_generate_image",
                                         function_args={}, balance_override=100)
        self.assertFalse(d.allowed)
        self.assertEqual(d.decision, "over_tier_ceiling")

    def test_over_daily_allowance_cumulative(self):
        # remediation (ceiling=allowance), balance 350 -> daily 11.67; image 5
        # gen1=5, gen2=10 (both under, budget remains), gen3=15 > 11.67 -> over_daily_allowance
        a = self.mod.evaluate_and_record(tool_name="mcp_higgsfield_generate_image", function_args={}, tier="remediation", balance_override=350)
        b = self.mod.evaluate_and_record(tool_name="mcp_higgsfield_generate_image", function_args={}, tier="remediation", balance_override=350)
        self.assertTrue(a.allowed and b.allowed)
        c = self.mod.evaluate_and_record(tool_name="mcp_higgsfield_generate_image", function_args={}, tier="remediation", balance_override=350)
        self.assertFalse(c.allowed)
        self.assertEqual(c.decision, "over_daily_allowance")

    def test_below_reserve_floor(self):
        # month started at 1000 -> floor 700; current dropped to 600
        self.seed_state(month_start_balance=1000.0)
        d = self.mod.evaluate_and_record(tool_name="mcp_higgsfield_generate_image",
                                         function_args={}, tier="remediation", balance_override=600)
        self.assertFalse(d.allowed)
        self.assertEqual(d.decision, "below_reserve_floor")


class TestBalanceUnavailable(Base):
    def test_video_denied_without_balance(self):
        d = self.mod.evaluate_and_record(tool_name="mcp_higgsfield_generate_video",
                                         function_args={"get_cost": True}, explicit_approval=True)
        self.assertFalse(d.allowed)
        self.assertEqual(d.decision, "balance_unavailable")

    def test_one_lowcost_fallback_then_blocked(self):
        os.environ["HIGGSFIELD_COST_ESTIMATES"] = json.dumps({"image": 2})  # under fallback 3
        self.mod = importlib.reload(self.mod)
        a = self.mod.evaluate_and_record(tool_name="mcp_higgsfield_generate_image", function_args={})
        self.assertTrue(a.allowed, a.reason)
        self.assertEqual(a.decision, "fallback_lowcost")
        b = self.mod.evaluate_and_record(tool_name="mcp_higgsfield_generate_image", function_args={})
        self.assertFalse(b.allowed)

    def test_expensive_image_blocked_without_balance(self):
        # default image cost 5 > fallback 3 -> not allowed even as fallback
        d = self.mod.evaluate_and_record(tool_name="mcp_higgsfield_generate_image", function_args={})
        self.assertFalse(d.allowed)
        self.assertEqual(d.decision, "balance_unavailable")


class TestVideoApproval(Base):
    def test_video_no_approval_blocked(self):
        d = self.mod.evaluate_and_record(tool_name="mcp_higgsfield_generate_video",
                                         function_args={"get_cost": True}, explicit_approval=False,
                                         tier="remediation", balance_override=3000)
        self.assertFalse(d.allowed)
        self.assertIn("approval", d.reason.lower())

    def test_video_approved_and_affordable_allowed(self):
        # balance 3000 -> daily 100; remediation ceiling 100 >= 30
        d = self.mod.evaluate_and_record(tool_name="mcp_higgsfield_generate_video",
                                         function_args={"params": {"get_cost": True}}, explicit_approval=True,
                                         tier="remediation", balance_override=3000)
        self.assertTrue(d.allowed, d.reason)
        self.assertEqual(d.artifact_type, "video")
        self.assertEqual(d.spent_after, 30)


class TestPostProduction(Base):
    def test_one_paid_pass(self):
        a = self.mod.evaluate_and_record(tool_name="mcp_higgsfield_upscale_image", function_args={}, balance_override=853.87)
        self.assertTrue(a.allowed, a.reason)
        self.assertEqual(a.attempt_cap, 1)
        b = self.mod.evaluate_and_record(tool_name="mcp_higgsfield_upscale_image", function_args={}, balance_override=853.87)
        self.assertFalse(b.allowed)
        self.assertIn("attempt cap", b.reason)


class TestRecordBalance(Base):
    def test_record_sets_cache_and_month_basis(self):
        self.mod.record_balance(853.87)
        cache = json.loads(Path(self.balance).read_text())
        self.assertEqual(cache["credits"], 853.87)
        usage = self.mod.current_usage()
        self.assertTrue(usage["balance_fresh"])
        self.assertAlmostEqual(usage["daily_allowance"], 853.87 / 30.0, places=2)
        self.assertEqual(usage["month_start_balance"], 853.87)
        self.assertAlmostEqual(usage["reserve_floor"], 0.70 * 853.87, places=2)

    def test_stale_cache_is_lookup_failure(self):
        os.environ["HIGGSFIELD_BALANCE_MAX_AGE_MIN"] = "1"
        self.mod = importlib.reload(self.mod)
        # write a cache timestamped in the past
        from datetime import datetime, timezone, timedelta
        old = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        Path(self.balance).write_text(json.dumps({"credits": 853.87, "ts": old}))
        _, fresh = self.mod._read_balance_cache()
        self.assertFalse(fresh)



class TestGrants(Base):
    def test_no_grant_defaults_standard(self):
        self.assertEqual(self.mod.active_grant_tier(), "standard")
        d = self.mod.evaluate_and_record(tool_name="mcp_higgsfield_generate_image",
                                         function_args={}, balance_override=853.87)
        self.assertEqual(d.tier, "standard")
        self.assertFalse(d.grant_applied)

    def test_grant_bumps_tier_and_resets_attempts(self):
        # burn the 2 standard attempts
        for _ in range(2):
            self.assertTrue(self.mod.evaluate_and_record(
                tool_name="mcp_higgsfield_generate_image", function_args={}, balance_override=853.87).allowed)
        blocked = self.mod.evaluate_and_record(
            tool_name="mcp_higgsfield_generate_image", function_args={}, balance_override=853.87)
        self.assertFalse(blocked.allowed)  # 2/2 standard
        # Sean grants user_requested (resets attempts, fresh budget of 3)
        g = self.mod.grant_artifact_tier("user_requested", minutes=30)
        self.assertEqual(g["uses"], 3)
        self.assertEqual(self.mod.active_grant_tier(), "user_requested")
        d = self.mod.evaluate_and_record(
            tool_name="mcp_higgsfield_generate_image", function_args={}, balance_override=853.87)
        self.assertTrue(d.allowed, d.reason)
        self.assertEqual(d.tier, "user_requested")
        self.assertTrue(d.grant_applied)

    def test_grant_is_use_capped(self):
        self.mod.grant_artifact_tier("user_requested", minutes=30, uses=1)
        a = self.mod.evaluate_and_record(tool_name="mcp_higgsfield_generate_image", function_args={}, balance_override=853.87)
        self.assertTrue(a.allowed and a.grant_applied)
        # grant consumed -> back to standard tier for the next call
        self.assertEqual(self.mod.active_grant_tier(), "standard")

    def test_grant_expires(self):
        self.mod.grant_artifact_tier("user_requested", minutes=-1)  # already expired
        self.assertEqual(self.mod.active_grant_tier(), "standard")

    def test_explicit_tier_overrides_grant(self):
        self.mod.grant_artifact_tier("remediation", minutes=30)
        d = self.mod.evaluate_and_record(tool_name="mcp_higgsfield_generate_image",
                                         function_args={}, tier="micro", balance_override=853.87)
        self.assertEqual(d.tier, "micro")
        self.assertFalse(d.grant_applied)  # explicit caller tier, grant untouched

    def test_clear_grant(self):
        self.mod.grant_artifact_tier("user_requested", minutes=30)
        self.assertEqual(self.mod.active_grant_tier(), "user_requested")
        self.mod.clear_artifact_grant()
        self.assertEqual(self.mod.active_grant_tier(), "standard")

    def test_grant_does_not_bypass_video_approval(self):
        # even with a remediation grant, video without approval is denied
        self.mod.grant_artifact_tier("remediation", minutes=30)
        d = self.mod.evaluate_and_record(tool_name="mcp_higgsfield_generate_video",
                                         function_args={"get_cost": True}, explicit_approval=False,
                                         balance_override=3000)
        self.assertFalse(d.allowed)
        self.assertIn("approval", d.reason.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
