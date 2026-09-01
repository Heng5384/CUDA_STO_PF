"""Regression qualification for the conservative beta-only positivity repair."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

from coupling.fixture_conditioned_handoff_v2 import load_host_96cube_fixture, load_validation_contract


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "run_beta_only_same_contract_control.py"
CONTRACT_PATH = ROOT / "contracts" / "pf_kwn_validation_contract_v1.json"
FIXTURE_SPEC = (
    ROOT
    / "data"
    / "qualification"
    / "pf_mass_conserving_library_handoff_v1"
    / "six_particle_96cube_spec.json"
)
PROFILE_ROOT = (
    ROOT.parent
    / "CUDA_STO_PF"
    / "data"
    / "qualification"
    / "pf_elastic_target_profile_quarter_nm_v2"
    / "profiles"
)
LEGACY_FAILURE_TIME_S = 0.39317699499770825 * 3600.0


def _load_control_module():
    specification = importlib.util.spec_from_file_location(
        "conservative_positivity_control_module", SCRIPT_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load beta-only control module")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class ConservativePositivityRepairTest(unittest.TestCase):
    """The frozen failure must be crossed without negative-bin correction."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.control = _load_control_module()
        cls.contract = load_validation_contract(CONTRACT_PATH)
        cls.fixture = load_host_96cube_fixture(FIXTURE_SPEC, PROFILE_ROOT, cls.contract)

    def test_exact_failure_window_is_crossed_conservatively(self) -> None:
        config, _ = self.control.build_same_contract_config(
            fixture=self.fixture,
            contract=self.contract,
        )
        self.assertLess(config.positivity_safety, 1.0)
        self.assertEqual(config.cfl_active_inventory_relative_threshold, 1.0e-6)
        solver = self.control.KWNSolver(config)
        solver.run_to_time(0.5 * 3600.0)
        self.assertGreater(solver.time_s, LEGACY_FAILURE_TIME_S)
        self.assertGreaterEqual(np.min(solver.population("beta").number_density_per_m4), 0.0)
        self.assertEqual(solver.roundoff_zeroed_bin_count, 0)
        self.assertLessEqual(
            max(item.positivity_utilization for item in solver.history),
            config.positivity_safety * (1.0 + 1.0e-12),
        )
        snapshot = solver.ledger.snapshot(
            matrix_xb=solver.matrix_xb, populations=solver.population_list()
        )
        self.assertLessEqual(snapshot.relative_residual, 1.0e-10)


if __name__ == "__main__":
    unittest.main()
