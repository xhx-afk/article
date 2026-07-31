from __future__ import annotations

import unittest

import torch
from scipy.stats import spearmanr

from tests.test_unified_evaluator import evaluator


class TieAwareSpearmanTest(unittest.TestCase):
    def test_matches_scipy_with_tied_targets(self):
        values = torch.tensor([0.2, 0.1, 0.3, 0.7, 0.6, 0.9])
        targets = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 2.0])
        expected = float(spearmanr(values.numpy(), targets.numpy()).statistic)
        self.assertAlmostEqual(evaluator.tie_aware_spearman(values, targets), expected, places=12)

    def test_constant_or_too_short_target_is_null(self):
        self.assertIsNone(
            evaluator.tie_aware_spearman(torch.tensor([0.1, 0.2, 0.3]), torch.zeros(3))
        )
        self.assertIsNone(
            evaluator.tie_aware_spearman(torch.tensor([0.1, 0.2]), torch.tensor([0.0, 1.0]))
        )


if __name__ == "__main__":
    unittest.main()
