import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _install_kubernetes_stubs():
    """Install minimal kubernetes module stubs for unit testing."""
    if "kubernetes" in sys.modules:
        return

    kubernetes_mod = types.ModuleType("kubernetes")
    client_mod = types.ModuleType("kubernetes.client")
    config_mod = types.ModuleType("kubernetes.config")
    watch_mod = types.ModuleType("kubernetes.watch")
    rest_mod = types.ModuleType("kubernetes.client.rest")

    class DummyApiException(Exception):
        def __init__(self, status=None, reason=None):
            super().__init__(f"ApiException(status={status}, reason={reason})")
            self.status = status
            self.reason = reason

    class DummyConfigException(Exception):
        pass

    def _noop(*_args, **_kwargs):
        return None

    class DummyCoreV1Api:
        pass

    class DummyWatch:
        def stream(self, *_args, **_kwargs):
            return iter(())

    client_mod.CoreV1Api = DummyCoreV1Api
    client_mod.V1Binding = object
    client_mod.V1ObjectMeta = object
    client_mod.V1ObjectReference = object
    config_mod.load_incluster_config = _noop
    config_mod.load_kube_config = _noop
    config_mod.ConfigException = DummyConfigException
    watch_mod.Watch = DummyWatch
    rest_mod.ApiException = DummyApiException

    kubernetes_mod.client = client_mod
    kubernetes_mod.config = config_mod
    kubernetes_mod.watch = watch_mod

    sys.modules["kubernetes"] = kubernetes_mod
    sys.modules["kubernetes.client"] = client_mod
    sys.modules["kubernetes.config"] = config_mod
    sys.modules["kubernetes.watch"] = watch_mod
    sys.modules["kubernetes.client.rest"] = rest_mod


def _load_module(module_name: str, relative_path: str):
    _install_kubernetes_stubs()
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


adaptive_module = _load_module(
    "adaptive_bandit_scheduler_under_test",
    "ml-schedulers/bandit/adaptive_bandit_scheduler.py",
)
bandit_module = _load_module(
    "bandit_scheduler_under_test",
    "ml-schedulers/bandit/bandit_scheduler.py",
)


class TestAdaptiveBanditCore(unittest.TestCase):
    def test_establishes_baseline_after_full_window(self):
        bandit = adaptive_module.ThompsonSamplingBandit(
            n_arms=2,
            window_size=3,
            reset_threshold=0.5,
        )
        bandit.update(arm=0, reward=0.9)
        bandit.update(arm=0, reward=0.9)
        result = bandit.update(arm=0, reward=0.9)

        self.assertFalse(result["regime_change_detected"])
        self.assertIsNotNone(bandit.baseline_reward)
        self.assertAlmostEqual(bandit.baseline_reward, 0.9, places=3)

    def test_detects_regime_change_without_inline_reset(self):
        bandit = adaptive_module.ThompsonSamplingBandit(
            n_arms=2,
            window_size=3,
            reset_threshold=0.5,
        )
        bandit.update(arm=1, reward=0.9)
        bandit.update(arm=1, reward=0.9)
        bandit.update(arm=1, reward=0.9)

        bandit.update(arm=1, reward=0.1)
        result = bandit.update(arm=1, reward=0.1)

        self.assertTrue(result["regime_change_detected"])
        self.assertEqual(bandit.reset_count, 0)
        self.assertIsNotNone(bandit.baseline_reward)

    def test_latency_to_reward_clamps_to_closed_range(self):
        metrics = adaptive_module.MetricsCollector("http://example")
        self.assertEqual(metrics.latency_to_reward(0), 0.99)
        self.assertEqual(metrics.latency_to_reward(100000), 0.01)


class TestLegacyBanditCore(unittest.TestCase):
    def test_regime_change_triggers_internal_reset(self):
        bandit = bandit_module.ThompsonSamplingBandit(
            n_arms=2,
            window_size=3,
            reset_threshold=0.5,
        )
        bandit.update(arm=0, reward=0.9)
        bandit.update(arm=0, reward=0.9)
        bandit.update(arm=0, reward=0.9)

        bandit.update(arm=0, reward=0.1)
        result = bandit.update(arm=0, reward=0.1)

        self.assertTrue(result["regime_change_detected"])
        self.assertEqual(bandit.reset_count, 1)
        self.assertIsNone(bandit.baseline_reward)
        self.assertEqual(len(bandit.reward_window), 0)


if __name__ == "__main__":
    unittest.main()
