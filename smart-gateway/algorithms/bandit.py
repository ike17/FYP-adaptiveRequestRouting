import math
import time
from collections import deque
from enum import Enum

import numpy as np

_INITIAL_ALPHA = 1.0
_INITIAL_BETA = 1.0


class CBState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ThompsonSamplingBandit:
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

        self._cb_state: dict[int, CBState] = {0: CBState.CLOSED, 1: CBState.CLOSED}
        self._cb_state_start: dict[int, float] = {0: 0.0, 1: 0.0}
        self._cb_probe_results: dict[int, list[float]] = {0: [], 1: []}

    def _cb_transition(self, arm: int, new_state: CBState) -> None:
        self._cb_state[arm] = new_state
        self._cb_state_start[arm] = time.time()
        self._cb_probe_results[arm] = []

    def _cb_get_admission_rate(self, arm: int) -> float:
        state = self._cb_state[arm]
        if state == CBState.CLOSED:
            return 1.0
        if state == CBState.OPEN:
            return 0.0
        elapsed = time.time() - self._cb_state_start[arm]
        half = self._cb_half_open_duration_s / 2.0
        if elapsed < half:
            return self._cb_half_open_admit_1
        return self._cb_half_open_admit_2

    def _cb_tick(self, arm: int) -> None:
        state = self._cb_state[arm]
        now = time.time()
        elapsed = now - self._cb_state_start[arm]

        if state == CBState.OPEN:
            if elapsed >= self._cb_open_duration_s:
                self._cb_transition(arm, CBState.HALF_OPEN)

        elif state == CBState.HALF_OPEN:
            if elapsed >= self._cb_half_open_duration_s:
                probes = self._cb_probe_results[arm]
                if probes and np.mean(probes) >= 0.3:
                    self._cb_transition(arm, CBState.CLOSED)
                else:
                    self._cb_transition(arm, CBState.OPEN)

    def _cb_record_probe(self, arm: int, reward: float) -> None:
        if self._cb_state[arm] == CBState.HALF_OPEN:
            self._cb_probe_results[arm].append(reward)
            probes = self._cb_probe_results[arm]
            if len(probes) >= 3 and np.mean(probes) < 0.1:
                self._cb_transition(arm, CBState.OPEN)

    def select_arm(self) -> int:
        available = list(range(2))

        if self._enable_cb:
            for a in range(2):
                self._cb_tick(a)

            admitted = []
            for a in range(2):
                rate = self._cb_get_admission_rate(a)
                if rate >= 1.0 or np.random.random() < rate:
                    admitted.append(a)
            if admitted:
                available = admitted

        samples = [np.random.beta(self._alpha[a], self._beta[a]) for a in available]
        return available[int(np.argmax(samples))]

    def update(
        self, arm: int, latency_ms: float, timed_out: bool = False, unreachable: bool = False,
    ) -> None:
        r = 0.0 if (timed_out or unreachable) else self._compute_reward(latency_ms)

        self._alpha[arm] += r
        self._beta[arm] += 1.0 - r

        self._total_pulls += 1
        self._reward_window.append(r)
        self._arm_reward_window[arm].append(r)

        if self._enable_cb:
            self._cb_record_probe(arm, r)

        if unreachable and self._enable_cb:
            other = 1 - arm
            if self._cb_state[other] != CBState.OPEN:
                self._cb_transition(arm, CBState.OPEN)

        if self._enable_regime_detection:
            self._check_regime_change()

    def _compute_reward(self, latency_ms: float) -> float:
        excess = max(0.0, latency_ms - self.target_latency_ms)
        return math.exp(-self.k * excess / self.target_latency_ms)

    def _check_regime_change(self) -> None:
        if len(self._reward_window) < self.window_size:
            return
        if self._total_pulls < self.window_size * 3:
            return

        current_avg = float(np.mean(self._reward_window))

        if self._baseline_reward is None:
            self._baseline_reward = current_avg
            return

        if current_avg < self._baseline_reward * self.reset_threshold:
            self._soft_reset()
            return

        if self._total_pulls % (self.window_size * 2) == 0:
            self._baseline_reward = 0.9 * self._baseline_reward + 0.1 * current_avg

    def _soft_reset(self) -> None:
        self._alpha = [a * 0.3 + _INITIAL_ALPHA * 0.7 for a in self._alpha]
        self._beta  = [b * 0.3 + _INITIAL_BETA  * 0.7 for b in self._beta]
        self._reward_window.clear()
        self._baseline_reward = None
        self._reset_count += 1

        if self._enable_cb:
            arm_avgs = [
                float(np.mean(self._arm_reward_window[a])) if self._arm_reward_window[a] else 1.0
                for a in range(2)
            ]
            worse_arm = int(np.argmin(arm_avgs))
            other_arm = 1 - worse_arm
            self._arm_reward_window[0].clear()
            self._arm_reward_window[1].clear()
            if self._cb_state[other_arm] != CBState.OPEN:
                self._cb_transition(worse_arm, CBState.OPEN)
            self._baseline_reward = 1.0

    def reset(self) -> None:
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
