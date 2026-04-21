import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import analyze_results as ar


class TestCohensD(unittest.TestCase):
    def test_cohens_d_known_value(self):
        rng = np.random.default_rng(42)
        a = rng.normal(loc=1.0, scale=1.0, size=100).tolist()
        b = rng.normal(loc=3.0, scale=1.0, size=100).tolist()
        # Expected d ≈ -2.0 (mean difference / pooled std ≈ (1-3)/1)
        d = ar.cohens_d(a, b)
        self.assertAlmostEqual(d, -2.307, delta=0.05)


class TestRunComparison(unittest.TestCase):
    def test_detects_clear_difference(self):
        rng = np.random.default_rng(42)
        lat1 = rng.normal(3000, 200, 50).tolist()
        lat2 = rng.normal(6000, 200, 50).tolist()
        result = ar.run_comparison("full", lat1, lat2, "mode_a", "mode_b", 15)
        self.assertIsNotNone(result)
        self.assertLess(result["p_mwu"], 0.001)
        self.assertGreater(abs(result["cohens_d"]), 0.8)

    def test_identical_distributions_not_significant(self):
        lat = [3000.0 + i for i in range(50)]
        result = ar.run_comparison("full", lat, lat[:], "mode_a", "mode_b", 15)
        self.assertIsNotNone(result)
        self.assertGreater(result["p_mwu"], 0.05)
        self.assertLess(abs(result["cohens_d"]), 0.05)

    def test_empty_input_returns_none(self):
        result = ar.run_comparison("full", [], [1000.0, 2000.0], "mode_a", "mode_b", 15)
        self.assertIsNone(result)


class TestGetLatencies(unittest.TestCase):
    def setUp(self):
        # 10 successful records with query_number 1–10
        self.data = [
            {"success": True, "query_number": i, "total_time_ms": float(i * 100)}
            for i in range(1, 11)
        ]
        ar.OVERLOAD_QUERY = 6  # queries 1–5 are pre, 6–10 are post

    def test_phase_pre_returns_correct_records(self):
        pre = ar.get_latencies(self.data, "pre")
        self.assertEqual(len(pre), 5)
        self.assertEqual(pre, [100.0, 200.0, 300.0, 400.0, 500.0])

    def test_phase_post_returns_correct_records(self):
        post = ar.get_latencies(self.data, "post")
        self.assertEqual(len(post), 5)
        self.assertEqual(post, [600.0, 700.0, 800.0, 900.0, 1000.0])


if __name__ == "__main__":
    unittest.main()
