import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import generate_report as report


class TestReportMath(unittest.TestCase):
    def test_theoretical_ber_decreases_with_photons(self):
        values = [report.theoretical_4ppm_ber(ns, 0.1) for ns in (0.5, 2.0, 5.0, 10.0)]
        self.assertTrue(all(left > right for left, right in zip(values, values[1:])))

    def test_count_snr_definition(self):
        self.assertAlmostEqual(report.count_snr_linear(5.0, 0.1), 25.0 / 5.2)

    def test_wilson_interval_contains_observed_rate(self):
        low, high = report.wilson_interval(10, 1000)
        self.assertLessEqual(low, 0.01)
        self.assertGreaterEqual(high, 0.01)


if __name__ == "__main__":
    unittest.main()

