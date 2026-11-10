import itertools
import math
import unittest
from fractions import Fraction

from podlab.audit_policy import (
    DriftGroup, calibrate_request_max, exact_detection_probability, plan_audit,
)


class CalibrationTests(unittest.TestCase):
    def test_correlated_steps_do_not_inflate_sample_size(self):
        groups = [DriftGroup(str(i), (float(i),) * 25) for i in range(32)]
        result = calibrate_request_max(groups, steps=25, alpha=0.01)
        self.assertEqual(result.n_groups, 32)
        self.assertEqual(result.rank, 33)
        self.assertFalse(result.usable)
        self.assertEqual(result.assess({0: 0.0}), "inconclusive")

    def test_exact_finite_sample_boundary(self):
        groups = [DriftGroup(str(i), (float(i), float(2 * i))) for i in range(99)]
        result = calibrate_request_max(groups, steps=2, alpha=0.01)
        self.assertEqual(result.rank, 99)
        self.assertEqual(result.thresholds, (196.0, 196.0))
        self.assertEqual(result.assess({1: 196.0}), "within_threshold")
        self.assertEqual(result.assess({1: 196.001}), "flagged")
        self.assertEqual(result.assess({}), "inconclusive")

    def test_scale_fit_is_disjoint_and_does_not_count_as_calibration(self):
        fit = [DriftGroup("fit", (2.0, 10.0))]
        cal = [DriftGroup(str(i), (2.0 * i, 10.0 * i)) for i in range(1, 10)]
        result = calibrate_request_max(cal, steps=2, alpha=0.1, scale_fit=fit)
        self.assertEqual(result.n_groups, 9)
        self.assertEqual(result.scales, (2.0, 10.0))
        self.assertEqual(result.thresholds, (18.0, 90.0))
        with self.assertRaises(ValueError):
            calibrate_request_max(cal, steps=2, scale_fit=cal[:1])

    def test_exhaustive_exchangeable_rank_coverage(self):
        # Every point is equally likely to be the test point. Correlated steps
        # have different scales, but only request maxima determine coverage.
        groups = [DriftGroup(str(i), (float(i), float(i * i))) for i in range(10)]
        flags = 0
        for i, test in enumerate(groups):
            result = calibrate_request_max(groups[:i] + groups[i + 1:], steps=2, alpha=0.2)
            flags += result.assess(dict(enumerate(test.ratios))) == "flagged"
        self.assertEqual(flags, 2)

    def test_ties_are_conservative_and_zero_drift_is_supported(self):
        groups = [DriftGroup(str(i), (0.0, 0.0)) for i in range(9)]
        result = calibrate_request_max(groups, steps=2, alpha=0.1)
        self.assertEqual(result.assess({0: 0.0}), "within_threshold")
        self.assertEqual(result.assess({0: 1e-30}), "flagged")

    def test_rejects_incomplete_duplicate_and_nonfinite_groups(self):
        invalid = [
            [DriftGroup("", (1.0,))],
            [DriftGroup("x", (1.0,)), DriftGroup("x", (2.0,))],
            [DriftGroup("x", ())],
            [DriftGroup("x", (-1.0,))],
            [DriftGroup("x", (math.nan,))],
            [DriftGroup("x", (math.inf,))],
        ]
        for groups in invalid:
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                calibrate_request_max(groups, steps=1)
        result = calibrate_request_max([], steps=1)
        for observed in ({-1: 0.0}, {1: 0.0}, {True: 0.0}, {0: math.nan}, {0: -1.0}):
            with self.subTest(observed=observed), self.assertRaises(ValueError):
                result.assess(observed)


class AuditBudgetTests(unittest.TestCase):
    def test_hypergeometric_matches_exhaustive_sampling(self):
        for steps in range(1, 9):
            for m in range(steps + 1):
                for k in range(steps + 1):
                    subsets = list(itertools.combinations(range(steps), k))
                    caught = sum(any(i < m for i in sample) for sample in subsets)
                    self.assertEqual(exact_detection_probability(steps, m, k),
                                     Fraction(caught, len(subsets)))

    def test_sparse_deviation_cannot_support_two_check_claim(self):
        self.assertEqual(float(exact_detection_probability(25, 1, 2)), 0.08)
        result = plan_audit(25, 1, 0.99)
        self.assertEqual(result.checks, 25)
        self.assertEqual(result.network_evaluation_fraction, 1.0)
        self.assertEqual(plan_audit(25, 25, 0.99).checks, 1)
        self.assertIsNone(plan_audit(25, 0, 0.99).checks)

    def test_plans_are_minimal(self):
        for m in range(1, 26):
            for target in (0.5, 0.9, 0.99, 1.0):
                result = plan_audit(25, m, target)
                self.assertGreaterEqual(result.achieved, target)
                self.assertLess(float(exact_detection_probability(25, m, result.checks - 1)), target)

    def test_invalid_inputs_never_silently_clamp(self):
        for args in ((0, 0, 0), (25, -1, 2), (25, 26, 2), (25, 1, 26), (25, 1, -1), (25, 1, True)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                exact_detection_probability(*args)
        for target in (0, -1, 1.1, math.nan, math.inf, True):
            with self.subTest(target=target), self.assertRaises(ValueError):
                plan_audit(25, 1, target)


if __name__ == "__main__":
    unittest.main()
