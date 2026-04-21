import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "smart-gateway"))

from algorithms.static_ml import StaticMLRouter

PKL_PATH = ROOT / "smart-gateway" / "model" / "gateway_model.pkl"


class TestStaticMLRouter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.router = StaticMLRouter(str(PKL_PATH))

    def test_short_prompt_zero_inflight_returns_valid_node(self):
        result = self.router.predict("kubernetes", 0, 0)
        self.assertIn(result, (0, 1))

    def test_long_prompt_high_gpu_inflight_returns_valid_node(self):
        prompt = "What is Kubernetes scheduling? " * 20
        result = self.router.predict(prompt, 10, 0)
        self.assertIn(result, (0, 1))

    def test_medium_prompt_mixed_inflight_returns_valid_node(self):
        prompt = "Explain RAG retrieval augmented generation in detail."
        result = self.router.predict(prompt, 2, 3)
        self.assertIn(result, (0, 1))


if __name__ == "__main__":
    unittest.main()
