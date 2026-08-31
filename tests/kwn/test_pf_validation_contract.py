"""Regression tests for the hash-bound PF--KWN validation thermodynamic path."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kwn_mvp.contract import (
    ValidationContractError,
    ValidationContractHashMismatch,
    canonical_contract_json,
    load_validation_contract,
)
from kwn_mvp.populations import PopulationParameters
from kwn_mvp.solver import KWNSolver, SolverConfig
from kwn_mvp.thermo_adapter import (
    PFContractReference,
    ThermodynamicContractBlocked,
    ValidationContractEquilibriumAdapter,
    build_equilibrium_adapter,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPOSITORY_ROOT / "contracts" / "pf_kwn_validation_contract_v1.json"


def _beta_parameters() -> PopulationParameters:
    """Build beta parameters strictly from the loaded contract, not copied literals."""

    contract = load_validation_contract(CONTRACT_PATH)
    temperature = contract.temperature_k
    return PopulationParameters(
        name="beta",
        x_b=contract.beta_xb,
        molar_volume_m3_mol=contract.vm_beta_m3_mol,
        diffusivity_m2_s=contract.matrix_diffusivity_m2_s(temperature),
        gamma_j_m2=contract.gamma_j_m2,
        xeq_infinity=contract.planar_solvus_xb(temperature),
        elastic_penalty_j_m3=contract.kwn_elastic_penalty_j_m3,
        nucleation={"mode": "off"},
    )


class PFValidationContractTest(unittest.TestCase):
    """Confirm Python reads and evaluates the single validation contract exactly."""

    def test_canonical_hash_and_validation_only_gate(self) -> None:
        """The loader uses the frozen canonical byte rule and rejects hash drift."""

        raw = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        contract = load_validation_contract(CONTRACT_PATH)
        expected = hashlib.sha256(canonical_contract_json(raw).encode("utf-8")).hexdigest()
        self.assertEqual(contract.sha256, expected)
        self.assertIs(raw["historical_as_run_claim"], False)
        with self.assertRaises(ValidationContractHashMismatch):
            load_validation_contract(CONTRACT_PATH, expected_hash="0" * 64)

    def test_validation_only_schema_rejects_historical_relabelling(self) -> None:
        """A contract cannot be loaded after claiming historical as-run authority."""

        raw = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        raw["historical_as_run_claim"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid_contract.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValidationContractError):
                load_validation_contract(path)

    def test_full_thermodynamic_and_kinetic_evaluation(self) -> None:
        """G, mu, curvature, solvus, D, and raw h(phi) share one JSON source."""

        contract = load_validation_contract(CONTRACT_PATH)
        temperature = contract.temperature_k
        x_b = 0.01
        self.assertAlmostEqual(
            contract.dg_alpha_dx_j_mol(temperature, x_b),
            contract.chemical_potential_b_j_mol(temperature, x_b)
            - contract.chemical_potential_a_j_mol(temperature, x_b),
            places=10,
        )
        self.assertAlmostEqual(
            contract.d2g_alpha_dx2_j_mol(temperature, x_b),
            contract.dchemical_potential_b_dx_j_mol(temperature, x_b)
            - contract.dchemical_potential_a_dx_j_mol(temperature, x_b),
            places=10,
        )
        step = 1.0e-7
        finite_difference = (
            contract.g_alpha_j_mol(temperature, x_b + step)
            - contract.g_alpha_j_mol(temperature, x_b - step)
        ) / (2.0 * step)
        self.assertAlmostEqual(
            finite_difference,
            contract.dg_alpha_dx_j_mol(temperature, x_b),
            delta=1.0e-4,
        )

        solvus = contract.planar_solvus_xb(temperature)
        self.assertAlmostEqual(solvus, 0.004649261005504821, places=15)
        self.assertLess(contract.beta_driving_force_j_mol(temperature, 0.004), 0.0)
        self.assertGreater(contract.beta_driving_force_j_mol(temperature, 0.006), 0.0)
        self.assertGreater(contract.matrix_diffusivity_m2_s(temperature), 0.0)

        radii = (2.0e-9, 5.0e-9, 10.0e-9, 20.0e-9, 50.0e-9)
        curvature = [contract.curvature_equilibrium_xb(temperature, radius) for radius in radii]
        self.assertTrue(all(left > right for left, right in zip(curvature, curvature[1:])))
        self.assertGreater(curvature[-1], solvus)

        # PF h(phi) intentionally evaluates the raw quintic, even when a
        # time-stepping overshoot carries phi outside [0, 1].
        for phi in (-0.1, 0.0, 0.5, 1.0, 1.1):
            expected = 6.0 * phi**5 - 15.0 * phi**4 + 10.0 * phi**3
            self.assertEqual(contract.h_of_phi(phi), expected)
        self.assertNotEqual(contract.h_of_phi(-0.1), 0.0)

    def test_explicit_path_constructs_exact_beta_adapter(self) -> None:
        """An explicit contract path activates the executable validation control."""

        contract = load_validation_contract(CONTRACT_PATH)
        adapter = build_equilibrium_adapter(
            mode="pf_contract",
            temperature_k=contract.temperature_k,
            contract_path=CONTRACT_PATH,
            expected_contract_hash=contract.sha256,
            planar_reference_xb=contract.planar_solvus_xb(contract.temperature_k),
        )
        self.assertIsInstance(adapter, ValidationContractEquilibriumAdapter)
        self.assertEqual(adapter.contract_hash, contract.sha256)
        radii = np.asarray([2.0e-9, 10.0e-9, 50.0e-9])
        values = adapter.equilibrium_xb(radii, _beta_parameters())
        expected = np.asarray(
            [contract.curvature_equilibrium_xb(contract.temperature_k, float(radius)) for radius in radii]
        )
        np.testing.assert_allclose(values, expected, rtol=0.0, atol=0.0)

    def test_reference_path_and_parameter_mismatch_fail_closed(self) -> None:
        """Hash-bound references work, but a KWN-side D retune is rejected."""

        contract = load_validation_contract(CONTRACT_PATH)
        reference = PFContractReference(
            status="VALIDATION_CONTRACT_RESOLVED",
            source="contracts/pf_kwn_validation_contract_v1.json",
            source_commit="f228cd45638279413ea62edd774d5b2315dbcbf6",
            planar_xb=contract.planar_solvus_xb(contract.temperature_k),
            contract_path=str(CONTRACT_PATH),
            contract_hash=contract.sha256,
        )
        adapter = build_equilibrium_adapter(
            mode="pf_contract",
            temperature_k=contract.temperature_k,
            pf_contract=reference,
        )
        parameters = _beta_parameters()
        self.assertIsInstance(adapter, ValidationContractEquilibriumAdapter)
        adapter.equilibrium_xb(np.asarray([10.0e-9]), parameters)
        retuned = PopulationParameters(
            **{**parameters.__dict__, "diffusivity_m2_s": parameters.diffusivity_m2_s * 1.01}
        )
        with self.assertRaises(ThermodynamicContractBlocked):
            adapter.equilibrium_xb(np.asarray([10.0e-9]), retuned)

    def test_explicit_path_does_not_override_blocked_legacy_reference(self) -> None:
        """Historical conflict provenance remains blocked even if a new path exists."""

        contract = load_validation_contract(CONTRACT_PATH)
        blocked = PFContractReference(
            status="BLOCKED_CONTRACT_CONFLICT",
            source="thermodynamic_contract_freeze_v1/07_frozen_thermodynamic_contract.yaml",
            source_commit="1c08f9ee011b31e0cd4d82749e58a8f69ebd2204",
        )
        with self.assertRaises(ThermodynamicContractBlocked):
            build_equilibrium_adapter(
                mode="pf_contract",
                temperature_k=contract.temperature_k,
                pf_contract=blocked,
                contract_path=CONTRACT_PATH,
            )

    def test_solver_plumbs_the_hash_bound_adapter_and_checkpoint_metadata(self) -> None:
        """A KWN run records the JSON hash and restores only the same contract."""

        contract = load_validation_contract(CONTRACT_PATH)
        beta = _beta_parameters()
        data = {
            "simulation": {
                "temperature_K": contract.temperature_k,
                "max_dt_s": 1.0,
                "min_dt_s": 1.0e-12,
                "size_cfl": 0.35,
                "rmax_outflow_relative_tolerance": 1.0e-10,
            },
            "matrix": {
                "molar_volume_m3_mol": contract.vm_alpha_m3_mol,
                "initial_xB": 0.01,
                "total_b_mol_m3": None,
                "inventory_tolerance_relative": 1.0e-10,
            },
            "radius_grid": {"minimum_m": 1.0e-9, "maximum_m": 1.0e-7, "bins": 32},
            "thermodynamics": {
                "mode": "pf_contract",
                "contract_path": str(CONTRACT_PATH),
                "contract_hash": contract.sha256,
            },
            "populations": {
                "g": {
                    "xB": 0.02,
                    "molar_volume_m3_mol": contract.vm_alpha_m3_mol,
                    "diffusivity_m2_s": contract.matrix_diffusivity_m2_s(contract.temperature_k),
                    "gamma_j_m2": 0.0,
                    "xeq_infinity": contract.planar_solvus_xb(contract.temperature_k),
                    "initial": {"kind": "empty"},
                    "nucleation": {"mode": "off"},
                },
                "beta": {
                    "xB": beta.x_b,
                    "molar_volume_m3_mol": beta.molar_volume_m3_mol,
                    "diffusivity_m2_s": beta.diffusivity_m2_s,
                    "gamma_j_m2": beta.gamma_j_m2,
                    "xeq_infinity": beta.xeq_infinity,
                    "elastic_penalty_j_m3": beta.elastic_penalty_j_m3,
                    "initial": {
                        "kind": "monodisperse",
                        "radius_m": 1.0e-8,
                        "number_density_m3": 1.0e18,
                    },
                    "nucleation": {"mode": "off"},
                },
            },
        }
        config = SolverConfig.from_mapping(data)
        solver = KWNSolver(config)
        self.assertEqual(solver.contract_hash, contract.sha256)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kwn_contract_checkpoint.npz"
            solver.save_checkpoint(path)
            restored = KWNSolver.load_checkpoint(config=config, path=path)
        self.assertEqual(restored.contract_hash, contract.sha256)
        self.assertEqual(restored.matrix_xb, solver.matrix_xb)
