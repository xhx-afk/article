from __future__ import annotations

import unittest
from types import SimpleNamespace

from tests.test_unified_evaluator import evaluator


class SemanticGateSwitchTest(unittest.TestCase):
    @staticmethod
    def args(enabled=None, gate_lambda=None, gate_gamma=None):
        return SimpleNamespace(
            semantic_gate_enabled=enabled,
            semantic_gate_lambda=gate_lambda,
            semantic_gate_gamma=gate_gamma,
        )

    def test_config_off_cli_unspecified_is_off(self):
        resolved = evaluator.resolve_semantic_gate(
            self.args(), {"PostProcessor": {"semantic_gate_enabled": False}}
        )
        self.assertEqual(resolved, (False, 0.10, 2.0, "config"))

    def test_cli_enabled_overrides_config_off(self):
        resolved = evaluator.resolve_semantic_gate(
            self.args(True), {"PostProcessor": {"semantic_gate_enabled": False}}
        )
        self.assertTrue(resolved[0])
        self.assertEqual(resolved[3], "command_line")

    def test_cli_disabled_overrides_config_on_and_cli_parameters_win(self):
        resolved = evaluator.resolve_semantic_gate(
            self.args(False, 0.3, 4.0),
            {"PostProcessor": {"semantic_gate_enabled": True, "semantic_gate_lambda": 0.2}},
        )
        self.assertEqual(resolved, (False, 0.3, 4.0, "command_line"))

    def test_missing_config_defaults_off(self):
        self.assertEqual(
            evaluator.resolve_semantic_gate(self.args(), {}),
            (False, 0.10, 2.0, "default"),
        )


if __name__ == "__main__":
    unittest.main()
