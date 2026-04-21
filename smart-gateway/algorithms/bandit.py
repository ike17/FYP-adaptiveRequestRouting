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
#
# Three bandit variants share this class, controlled by constructor flags:
#   bandit_plain  — Thompson Sampling only (no regime detection, no CB)
#   bandit_regime — Thompson Sampling + regime detection soft-reset (no CB)
#   adaptive      — Thompson Sampling + regime detection + 3-state circuit breaker

import math
import time
from collections import deque
from enum import Enum

import numpy as np

_INITIAL_ALPHA = 1.0
_INITIAL_BETA = 1.0


class CBState(Enum):
    CLOSED = "closed"       # arm fully available
    OPEN = "open"           # arm blocked
    HALF_OPEN = "half_open" # graduated admission (25% → 50%)


class ThompsonSamplingBandit:
    """
    Beta-Bernoulli Thompson Sampling bandit with optional regime detection
    and 3-state circuit breaker.

    Two arms:
        0 — GPU Ollama (ollama-gpu-service)
        1 — CPU Ollama (ollama-cpu-service)

    Modes (set via constructor flags):
        enable_regime_detection=False, enable_cb=False → bandit_plain
        enable_regime_detection=True,  enable_cb=False → bandit_regime
        enable_regime_detection=True,  enable_cb=True  → adaptive
    """

    def __init__(
        self,
        target_latency_ms: float = 5000.0,
        k: float = 1.0,
        window_size: int = 10,
        reset_threshold: float = 0.4,
        enable_regime_detection: bool = False,
        enable_cb: bool = False,
        cb_open_duration_s: float = 15.0,
        cb_half_open_duration_s: float = 30.0,
        cb_half_open_admit_1: float = 0.25,
        cb_half_open_admit_2: float = 0.50,
    ):
        self.target_latency_ms = target_latency_ms
        self.k = k
        self.window_size = window_size
        self.reset_threshold = reset_threshold
        self._enable_regime_detection = enable_regime_detection
        self._enable_cb = enable_cb
        self._cb_open_duration_s = cb_open_duration_s
        self._cb_half_open_duration_s = cb_half_open_duration_s
        self._cb_half_open_admit_1 = cb_half_open_admit_1
        self._cb_half_open_admit_2 = cb_half_open_admit_2

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

        # Circuit breaker state per arm (adaptive mode only)
        self._cb_state: dict[int, CBState] = {0: CBState.CLOSED, 1: CBState.CLOSED}
        self._cb_state_start: dict[int, float] = {0: 0.0, 1: 0.0}
        # Track probe outcomes in HALF_OPEN to decide graduation vs reopening
        self._cb_probe_results: dict[int, list[float]] = {0: [], 1: []}

    # ------------------------------------------------------------------
    # Circuit breaker helpers
    # ------------------------------------------------------------------

    def _cb_transition(self, arm: int, new_state: CBState) -> None:
        self._cb_state[arm] = new_state
        self._cb_state_start[arm] = time.time()
        self._cb_probe_results[arm] = []

    def _cb_get_admission_rate(self, arm: int) -> float:
        """Return the current admission probability for an arm under CB control."""
        state = self._cb_state[arm]
        if state == CBState.CLOSED:
            return 1.0
        if state == CBState.OPEN:
            return 0.0
        # HALF_OPEN: graduated admission
        elapsed = time.time() - self._cb_state_start[arm]
        half = self._cb_half_open_duration_s / 2.0
        if elapsed < half:
            return self._cb_half_open_admit_1  # 25%
        return self._cb_half_open_admit_2      # 50%

    def _cb_tick(self, arm: int) -> None:
        """Advance CB state machine based on elapsed time and probe outcomes."""
        state = self._cb_state[arm]
        now = time.time()
        elapsed = now - self._cb_state_start[arm]

        if state == CBState.OPEN:
            if elapsed >= self._cb_open_duration_s:
                self._cb_transition(arm, CBState.HALF_OPEN)

        elif state == CBState.HALF_OPEN:
            if elapsed >= self._cb_half_open_duration_s:
                # Graduation complete — check if probes were healthy.
                # If no probes were recorded (zero traffic), we cautiously CLOSE
                # to allow rediscovery, rather than re-blocking the arm.
                probes = self._cb_probe_results[arm]
                if not probes:
                    self._cb_transition(arm, CBState.CLOSED)
                elif np.mean(probes) >= 0.3:
                    self._cb_transition(arm, CBState.CLOSED)
                else:
                    # Probes failed — reopen
                    self._cb_transition(arm, CBState.OPEN)

    def _cb_record_probe(self, arm: int, reward: float) -> None:
        """Record a probe result during HALF_OPEN state."""
        if self._cb_state[arm] == CBState.HALF_OPEN:
            self._cb_probe_results[arm].append(reward)
            # Early failure: if we have enough probes and they're bad, reopen immediately
            probes = self._cb_probe_results[arm]
            if len(probes) >= 3 and np.mean(probes) < 0.1:
                self._cb_transition(arm, CBState.OPEN)

    # ------------------------------------------------------------------
    # Core bandit operations
    # ------------------------------------------------------------------

    def select_arm(self) -> int:
        """
        Sample from each posterior and return the arm with the highest draw.
        In CB mode, arms are filtered by circuit breaker admission probability.
        Falls back to all arms if none are available (safety).
        """
        available = list(range(2))

        if self._enable_cb:
            # Advance CB state machines
            for a in range(2):
                self._cb_tick(a)

            # Filter by admission probability
            admitted = []
            for a in range(2):
                rate = self._cb_get_admission_rate(a)
                if rate >= 1.0 or np.random.random() < rate:
                    admitted.append(a)
            if admitted:
                available = admitted
            # else: safety fallback — all arms available

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
            - unreachable (503)  →  reward = 0.0

        When unreachable=True the node is confirmed dead (503 ConnectError).
        In CB mode this triggers an immediate OPEN transition.
        """
        r = 0.0 if (timed_out or unreachable) else self._compute_reward(latency_ms)

        self._alpha[arm] += r
        self._beta[arm] += 1.0 - r

        self._total_pulls += 1
        self._reward_window.append(r)
        self._arm_reward_window[arm].append(r)

        # Record probe outcome for CB half-open evaluation
        if self._enable_cb:
            self._cb_record_probe(arm, r)

        # 503 = node confirmed dead — immediate CB open (no regime detection needed)
        if unreachable and self._enable_cb:
            other = 1 - arm
            if self._cb_state[other] != CBState.OPEN:
                self._cb_transition(arm, CBState.OPEN)

        if self._enable_regime_detection:
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
        # Burn-in: don't fire until we have at least window_size * 3 pulls.
        # With window_size=5 this is 15 queries — enough to build a stable
        # baseline before the detector is trusted.
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
        In CB mode, additionally opens the circuit breaker on the worse arm.
        """
        self._alpha = [a * 0.3 + _INITIAL_ALPHA * 0.7 for a in self._alpha]
        self._beta  = [b * 0.3 + _INITIAL_BETA  * 0.7 for b in self._beta]
        self._reward_window.clear()
        self._baseline_reward = None
        self._reset_count += 1

        if self._enable_cb:
            # Identify the bad arm from per-arm recent rewards
            arm_avgs = [
                float(np.mean(self._arm_reward_window[a])) if self._arm_reward_window[a] else 1.0
                for a in range(2)
            ]
            worse_arm = int(np.argmin(arm_avgs))
            other_arm = 1 - worse_arm
            self._arm_reward_window[0].clear()
            self._arm_reward_window[1].clear()
            # Only open CB on worse arm if the other arm is not already open.
            # During a cluster-wide blackout both arms fail simultaneously;
            # opening both causes a deadlock.
            if self._cb_state[other_arm] != CBState.OPEN:
                self._cb_transition(worse_arm, CBState.OPEN)
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
        self._cb_state = {0: CBState.CLOSED, 1: CBState.CLOSED}
        self._cb_state_start = {0: 0.0, 1: 0.0}
        self._cb_probe_results = {0: [], 1: []}

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
            "enable_regime_detection": self._enable_regime_detection,
            "enable_cb": self._enable_cb,
            "cb_state": {
                i: {
                    "state": self._cb_state[i].value,
                    "elapsed_s": round(now - self._cb_state_start[i], 1) if self._cb_state_start[i] > 0 else 0,
                    "admission_rate": self._cb_get_admission_rate(i),
                }
                for i in range(2)
            },
        }
