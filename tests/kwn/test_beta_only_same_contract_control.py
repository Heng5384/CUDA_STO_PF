"""Focused checks for the host-only beta same-contract direction control."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from coupling.fixture_conditioned_handoff_v2 import make_synthetic_fixture_control


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


def _load_control_module():
    specification = importlib.util.spec_from_file_location(
        "beta_only_same_contract_control_test_module", SCRIPT_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load beta-only same-contract control script")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class BetaOnlySameContractControlTests(unittest.TestCase):
    """The direction gate must be contract-owned and fixture-inventory closed."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.control = _load_control_module()
        cls.contract = cls.control.load_validation_contract(CONTRACT_PATH)
        cls.fixture = make_synthetic_fixture_control(cls.contract, FIXTURE_SPEC)

    def test_contract_owned_beta_parameters_and_t0_direction_gate(self) -> None:
        config, construction = self.control.build_same_contract_config(
            fixture=self.fixture,
            contract=self.contract,
        )
        beta = config.populations[1]
        self.assertEqual(config.thermo_mode, "pf_contract")
        self.assertEqual(config.validation_contract_hash, self.contract.contract_hash)
        self.assertEqual(beta.x_b, self.contract.v_B)
        self.assertEqual(beta.molar_volume_m3_mol, self.contract.vm_beta_m3_mol)
        self.assertEqual(
            beta.diffusivity_m2_s,
            self.contract.canonical.matrix_diffusivity_m2_s(self.contract.temperature_K),
        )
        self.assertEqual(beta.gamma_j_m2, self.contract.canonical.gamma_j_m2)
        self.assertEqual(beta.elastic_penalty_j_m3, self.contract.canonical.kwn_elastic_penalty_j_m3)
        self.assertEqual(config.populations[0].nucleation["mode"], "off")
        self.assertEqual(beta.nucleation["mode"], "off")

        solver = self.control.KWNSolver(config)
        entries = construction["config_mapping"]["populations"]["beta"]["initial"]["entries"]
        rows, direction_pass = self.control._direction_rows(
            solver=solver,
            contract=self.contract,
            entries=entries,
        )
        metrics = self.control._snapshot_metrics(solver)
        self.assertTrue(direction_pass)
        self.assertTrue(all(row["direction_match"] == "true" for row in rows))
        self.assertLessEqual(metrics["kwn_inventory_relative_residual"], 1.0e-12)

        domain = construction["numerical_domain_gate"]
        self.assertGreater(domain["contract_valid_minimum_m"], domain["template_minimum_m"])
        self.assertLess(
            domain["contract_valid_minimum_m"], domain["minimum_initial_fixture_radius_m"])
        self.assertFalse(domain["initial_fixture_radii_changed"])
        self.assertFalse(domain["physical_parameters_changed"])


if __name__ == "__main__":
    unittest.main()
