# smart-gateway/algorithms/bandit.py
# Thompson Sampling (Beta-Bernoulli) bandit for routing between GPU (arm 0) and CPU (arm 1).
#
# Key design choices vs the old scheduler-based bandit:
#   - Reward function: exp(-k * excess / target) instead of linear 1 - (lat / max).
#     This gives reward=1.0 at exactly target latency and decays exponentially as
#     latency exceeds the target — much more sensitive to small degradations.
#   - Updates happen inline in the request path (the gateway owns the full
#     request lifecycle) rather than via a background polling thread.
#   - Soft-reset regime detection preserves directional knowledge on shock.
#   - Adaptive mode adds a circuit-breaker layer on top: after a regime change the
#     worse-performing arm is blocked for recovery_window_s seconds, forcing all
#     traffic to the surviving arm while the degraded node recovers.

import math
import time
from collections import deque

import numpy as np

_INITIAL_ALPHA = 1.0
_INITIAL_BETA = 1.0


class ThompsonSamplingBandit:
    """
    Beta-Bernoulli Thompson Sampling bandit with sliding-window regime detection.

    Two arms:
        0 — GPU Ollama (ollama-gpu-service)
        1 — CPU Ollama (ollama-cpu-service)

    When is_adaptive=True a circuit-breaker is layered on top: on regime change
    the worse arm is excluded from selection for recovery_window_s seconds.
    """

    def __init__(
        self,
        target_latency_ms: float = 5000.0,
        k: float = 1.0,
        window_size: int = 10,
        reset_threshold: float = 0.4,
        is_adaptive: bool = False,
        recovery_window_s: float = 120.0,
    ):
        self.target_latency_ms = target_latency_ms
        self.k = k
        self.window_size = window_size
        self.reset_threshold = reset_threshold
        self._is_adaptive = is_adaptive
        self._recovery_window_s = recovery_window_s

        # Per-arm Beta posteriors
        self._alpha = [_INITIAL_ALPHA, _INITIAL_ALPHA]
        self._beta = [_INITIAL_BETA, _INITIAL_BETA]

        self._reward_window: deque = deque(maxlen=window_size)
        self._arm_reward_window: dict[int, deque] = {
            0: deque(maxlen=window_size),
            1: deque(maxlen=window_size),
        }
        self._baseline_reward: float | None = None
        self._total_pulls = 0
        self._reset_count = 0

        # Wall-clock time until which each arm is blocked (adaptive mode only)
        self._blocked_until: dict[int, float] = {0: 0.0, 1: 0.0}

    # ------------------------------------------------------------------
    # Core bandit operations
    # ------------------------------------------------------------------

    def select_arm(self) -> int:
        """
        Sample from each posterior and return the arm with the highest draw.
        In adaptive mode, arms that are still within their recovery block window
        are excluded. Falls back to all arms if none are available (safety).
        """
        if self._is_adaptive:
            now = time.time()
            available = [a for a in range(2) if now >= self._blocked_until[a]]
            if not available:
                available = list(range(2))  # safety: never deadlock
        else:
            available = list(range(2))

        samples = [np.random.beta(self._alpha[a], self._beta[a]) for a in available]
        return available[int(np.argmax(samples))]

    def update(
        self, arm: int, latency_ms: float, timed_out: bool = False, unreachable: bool = False,
    ) -> None:
        """
        Update the posterior for `arm` based on observed latency.

        Reward function: exp(-k * max(0, latency - target) / target)
            - latency == target  →  reward = 1.0
            - latency >> target  →  reward → 0  (exponential decay)
            - timeout            →  reward = 0.0 (heavy penalty)
            - unreachable (503)  →  reward = 0.0 + immediate arm block in adaptive mode

        When unreachable=True the node is confirmed dead (503 ConnectError).
        No statistical evidence window is needed — block immediately.
        """
        r = 0.0 if (timed_out or unreachable) else self._compute_reward(latency_ms)

        self._alpha[arm] += r
        self._beta[arm] += 1.0 - r

        self._total_pulls += 1
        self._reward_window.append(r)
        self._arm_reward_window[arm].append(r)

        # 503 = node confirmed dead — bypass regime detection, block immediately
        if unreachable and self._is_adaptive:
            other = 1 - arm
            now = time.time()
            if now >= self._blocked_until[other]:
                self._blocked_until[arm] = now + self._recovery_window_s

        self._check_regime_change()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_reward(self, latency_ms: float) -> float:
        excess = max(0.0, latency_ms - self.target_latency_ms)
        return math.exp(-self.k * excess / self.target_latency_ms)

    def _check_regime_change(self) -> None:
        if len(self._reward_window) < self.window_size:
            return
        # Burn-in: don't fire the circuit breaker until we have at least
        # window_size * 3 pulls. With window_size=5 this is 15 queries —
        # enough to build a stable baseline before the detector is trusted.
        if self._total_pulls < self.window_size * 3:
            return

        current_avg = float(np.mean(self._reward_window))

        if self._baseline_reward is None:
            self._baseline_reward = current_avg
            return

        if current_avg < self._baseline_reward * self.reset_threshold:
            self._soft_reset()
            return

        # Slowly drift the baseline to track gradual improvement
        if self._total_pulls % (self.window_size * 2) == 0:
            self._baseline_reward = 0.9 * self._baseline_reward + 0.1 * current_avg

    def _soft_reset(self) -> None:
        """
        Decay posteriors towards uniform priors, preserving directional bias.
        In adaptive mode, additionally blocks the worse arm for recovery_window_s.
        """
        self._alpha = [a * 0.3 + _INITIAL_ALPHA * 0.7 for a in self._alpha]
        self._beta  = [b * 0.3 + _INITIAL_BETA  * 0.7 for b in self._beta]
        self._reward_window.clear()
        self._baseline_reward = None
        self._reset_count += 1

        if self._is_adaptive:
            # Identify the bad arm from per-arm recent rewards, not historical posteriors.
            # An untouched arm defaults to 1.0 (assume healthy until proven otherwise).
            arm_avgs = [
                float(np.mean(self._arm_reward_window[a])) if self._arm_reward_window[a] else 1.0
                for a in range(2)
            ]
            worse_arm = int(np.argmin(arm_avgs))
            other_arm = 1 - worse_arm
            self._arm_reward_window[0].clear()
            self._arm_reward_window[1].clear()
            now = time.time()
            # Only block the worse arm if the other arm is currently available.
            # During a cluster-wide blackout both arms fail simultaneously; blocking
            # the worse arm while the other is already blocked causes a deadlock where
            # no arm can be selected.  The safety fallback in select_arm() handles
            # temporary dual-failure without needing an explicit block.
            if now >= self._blocked_until[other_arm]:
                self._blocked_until[worse_arm] = now + self._recovery_window_s
            # Reset baseline to 1.0 so the detector stays sensitive after a shock;
            # leaving it at 0.0 (all-timeout window) permanently flatlines the check.
            self._baseline_reward = 1.0

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset to initial prior distribution — call before each benchmark run."""
        self._alpha = [_INITIAL_ALPHA, _INITIAL_ALPHA]
        self._beta  = [_INITIAL_BETA,  _INITIAL_BETA]
        self._reward_window.clear()
        self._arm_reward_window[0].clear()
        self._arm_reward_window[1].clear()
        self._baseline_reward = None
        self._total_pulls = 0
        self._reset_count = 0
        self._blocked_until = {0: 0.0, 1: 0.0}

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        now = time.time()
        return {
            "arms": [
                {
                    "name": name,
                    "alpha": round(self._alpha[i], 3),
                    "beta": round(self._beta[i], 3),
                    "expected_reward": round(
                        self._alpha[i] / (self._alpha[i] + self._beta[i]), 4
                    ),
                }
                for i, name in enumerate(["gpu", "cpu"])
            ],
            "total_pulls": self._total_pulls,
            "reset_count": self._reset_count,
            "window_avg": round(float(np.mean(self._reward_window)), 4)
            if self._reward_window
            else None,
            "baseline_reward": round(self._baseline_reward, 4)
            if self._baseline_reward is not None
            else None,
            "is_adaptive": self._is_adaptive,
            "recovery_remaining": {
                0: max(0, round(self._blocked_until[0] - now, 1)),
                1: max(0, round(self._blocked_until[1] - now, 1)),
            },
        }
