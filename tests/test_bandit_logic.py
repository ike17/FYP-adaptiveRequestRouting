import math
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "smart-gateway"))

from algorithms.bandit import CBState, ThompsonSamplingBandit


def _plain(window_size=5):
    return ThompsonSamplingBandit(
        target_latency_ms=5000,
        window_size=window_size,
        enable_regime_detection=False,
        enable_cb=False,
    )


def _regime(window_size=5):
    return ThompsonSamplingBandit(
        target_latency_ms=5000,
        window_size=window_size,
        enable_regime_detection=True,
        enable_cb=False,
    )


def _adaptive(window_size=3, cb_open_duration_s=15.0, cb_half_open_duration_s=15.0):
    return ThompsonSamplingBandit(
        target_latency_ms=5000,
        window_size=window_size,
        enable_regime_detection=True,
        enable_cb=True,
        cb_open_duration_s=cb_open_duration_s,
        cb_half_open_duration_s=cb_half_open_duration_s,
    )


class TestRewardFunction(unittest.TestCase):
    def test_at_target_reward_is_one(self):
        b = _plain()
        self.assertAlmostEqual(b._compute_reward(5000), 1.0, places=3)

    def test_below_target_clamps_to_one(self):
        b = _plain()
        self.assertAlmostEqual(b._compute_reward(100), 1.0, places=3)

    def test_double_target_decays_exponentially(self):
        b = _plain()
        self.assertAlmostEqual(b._compute_reward(10000), math.exp(-1), places=3)

    def test_extreme_latency_stays_above_zero(self):
        b = _plain()
        self.assertGreater(b._compute_reward(999_999), 0.0)

    def test_timeout_gives_zero_reward(self):
        b = _plain()
        b.update(arm=0, latency_ms=5000, timed_out=True)
        self.assertAlmostEqual(b._alpha[0], 1.0, places=6)

    def test_good_latency_increments_alpha(self):
        b = _plain()
        b.update(arm=0, latency_ms=5000)
        self.assertGreater(b._alpha[0], 1.0)


class TestRegimeDetection(unittest.TestCase):
    def _feed_stable(self, b, n, arm=0, latency_ms=5000):
        for _ in range(n):
            b.update(arm=arm, latency_ms=latency_ms)

    def test_baseline_established_after_burn_in(self):
        b = _regime(window_size=3)
        self._feed_stable(b, 9)
        self.assertIsNotNone(b._baseline_reward)

    def test_no_regime_change_on_stable_rewards(self):
        b = _regime(window_size=3)
        self._feed_stable(b, 20)
        self.assertEqual(b._reset_count, 0)

    def test_regime_change_increments_reset_count(self):
        b = _regime(window_size=3)
        self._feed_stable(b, 9)
        for _ in range(5):
            b.update(arm=0, latency_ms=25000)
        self.assertGreater(b._reset_count, 0)

    def test_soft_reset_decays_but_preserves_posterior_direction(self):
        b = _regime(window_size=3)
        self._feed_stable(b, 20)
        alpha_before = b._alpha[0]
        for _ in range(5):
            b.update(arm=0, latency_ms=25000)
        self.assertGreater(b._reset_count, 0)
        self.assertGreater(b._alpha[0], 1.0)
        self.assertLess(b._alpha[0], alpha_before)


class TestCircuitBreaker(unittest.TestCase):
    def _trigger_reset(self, b, arm=0):
        for _ in range(b.window_size * 3):
            b.update(arm=arm, latency_ms=5000)
        for _ in range(b.window_size + 2):
            b.update(arm=arm, latency_ms=25000)

    def test_cb_opens_worse_arm_after_regime_change(self):
        b = _adaptive()
        self._trigger_reset(b, arm=0)
        self.assertEqual(b._cb_state[0], CBState.OPEN)

    def test_cb_transitions_open_to_half_open_after_duration(self):
        b = _adaptive()
        b._cb_transition(0, CBState.OPEN)
        b._cb_state_start[0] = time.time() - 20
        b._cb_tick(0)
        self.assertEqual(b._cb_state[0], CBState.HALF_OPEN)

    def test_good_probes_in_half_open_close_circuit(self):
        b = _adaptive()
        b._cb_transition(0, CBState.HALF_OPEN)
        b._cb_probe_results[0] = [0.9, 0.9, 0.9]
        b._cb_state_start[0] = time.time() - (b._cb_half_open_duration_s + 1)
        b._cb_tick(0)
        self.assertEqual(b._cb_state[0], CBState.CLOSED)

    def test_bad_probes_reopen_circuit_immediately(self):
        b = _adaptive()
        b._cb_transition(0, CBState.HALF_OPEN)
        for _ in range(3):
            b._cb_record_probe(0, 0.02)
        self.assertEqual(b._cb_state[0], CBState.OPEN)

    def test_half_open_without_probes_reopens_cleanly(self):
        b = _adaptive()
        b._cb_transition(0, CBState.HALF_OPEN)
        b._cb_state_start[0] = time.time() - (b._cb_half_open_duration_s + 1)
        b._cb_tick(0)
        self.assertEqual(b._cb_state[0], CBState.OPEN)

    def test_open_arm_excluded_from_selection(self):
        b = _adaptive()
        b._cb_transition(0, CBState.OPEN)
        b._cb_state_start[0] = time.time() + 9999
        for _ in range(30):
            self.assertEqual(b.select_arm(), 1)

    def test_deadlock_guard_returns_an_arm_when_all_blocked(self):
        b = _adaptive()
        future = time.time() + 9999
        b._cb_transition(0, CBState.OPEN)
        b._cb_state_start[0] = future
        b._cb_transition(1, CBState.OPEN)
        b._cb_state_start[1] = future
        arm = b.select_arm()
        self.assertIn(arm, (0, 1))

    def test_unreachable_triggers_immediate_cb_open(self):
        b = _adaptive()
        b.update(arm=0, latency_ms=0, unreachable=True)
        self.assertEqual(b._cb_state[0], CBState.OPEN)


class TestConvergence(unittest.TestCase):
    def test_select_arm_converges_on_dominant_arm(self):
        b = _plain()
        target = b.target_latency_ms
        for _ in range(50):
            b.update(arm=0, latency_ms=target)
        for _ in range(50):
            b.update(arm=1, latency_ms=target * 10)
        selections = [b.select_arm() for _ in range(200)]
        arm0_count = selections.count(0)
        self.assertGreaterEqual(arm0_count, 190)


if __name__ == "__main__":
    unittest.main()
