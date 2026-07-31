#!/usr/bin/env python3
"""Static and converter contracts for the default elastic solver."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_converter():
    path = ROOT / "Unit_Psedobinary.py"
    spec = importlib.util.spec_from_file_location("unit_psedobinary", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ElasticWarmStartDefaultsTest(unittest.TestCase):
    def test_converter_emits_qualified_defaults(self) -> None:
        module = load_converter()
        self.assertEqual(
            module.ELASTIC_SOLVER_DEFAULTS,
            {
                "elastic_warm_start_enabled": 1,
                "elastic_residual_control_enabled": 1,
                "elastic_iter_min": 2,
                "elastic_iter_max": 32,
                "elastic_residual_tolerance": 1.0e-6,
                "elastic_residual_absolute_floor": 1.0e-30,
                "elastic_fail_on_nonconvergence": 1,
            },
        )

    def test_cuda_defaults_and_general_mode_contract(self) -> None:
        source = (ROOT / "main_cuda.cu").read_text(encoding="utf-8")
        for line in (
            "P->elastic_iter_max = 32;",
            "P->elastic_warm_start_enabled = 1;",
            "P->elastic_residual_control_enabled = 1;",
            "P->elastic_iter_min = 2;",
            "P->elastic_residual_tolerance = 1.0e-6;",
            "P->elastic_fail_on_nonconvergence = 1;",
        ):
            self.assertIn(line, source)
        self.assertNotIn(
            "accelerated elastic warm-start requires %s",
            source,
        )
        self.assertNotIn(
            "!P.elastic_enabled || P.mode != 0",
            source,
        )
        self.assertIn(
            "P.elastic_enabled &&\n        (P.elastic_warm_start_enabled != 0 ||",
            source,
        )
        self.assertIn(
            'pf_zero_mode_enabled ? "V4" : "NOT_APPLICABLE"',
            source,
        )

    def test_legacy_fallback_remains_explicit(self) -> None:
        runner = (
            ROOT / "jobs" / "run_pf_elastic_warm_start_residual_v1.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("elastic_warm_start_enabled = 0", runner)
        self.assertIn("elastic_residual_control_enabled = 0", runner)
        self.assertIn("elastic_iter_max = 20", runner)


if __name__ == "__main__":
    unittest.main()
