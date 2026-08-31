"""Focused unittest coverage for the v1 KWN--PF snapshot handoff contract."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from coupling.handoff_audit import PASS_HANDOFF_LEDGER_CLOSED, audit_handoff
from coupling.kwn_pf_schema import (
    HandoffPackage,
    SchemaValidationError,
    assert_roundtrip_equivalent,
    make_handoff_package,
    read_handoff_package,
    validate_handoff_package,
    write_handoff_package,
)
from coupling.kwn_to_pf import (
    PARTIAL_PF_STATE_NOT_CLOSED,
    READY_FOR_PF_HANDOFF,
    KwnHandoffExportError,
    PFStateNotClosedError,
    adapt_kwn_to_pf,
    build_handoff_from_kwn_solver,
)
from coupling.pf_to_kwn import PfResolvedSnapshot, pf_snapshot_to_handoff
from coupling.spatial_sampler import (
    SpatialSamplingError,
    sample_resolved_beta_geometry,
    validate_periodic_nonoverlap,
)
from kwn_mvp.populations import Population, PopulationParameters
from kwn_mvp.radius_grid import RadiusGrid


def valid_package(
    matrix_inventory: float = 5.0,
    gp_inventory: float = 2.0,
    subgrid_inventory: float = 1.0,
    resolved_inventory: float = 2.0,
) -> HandoffPackage:
    """Build a compact package with all four ledger buckets populated."""

    total = matrix_inventory + gp_inventory + subgrid_inventory + resolved_inventory
    metadata = {
        "schema_version": "kwn_pf_handoff_v1",
        "source": {
            "git_commit": "a" * 40,
            "config_hash": "b" * 64,
            "kwn_backend": "internal_kwn_mvp",
            "kwn_backend_version": "v1",
        },
        "temperature_K": 653.15,
        "handoff_time_s": 6.0 * 3600.0,
        "pf_box": {"lengths_m": [1.0e-6, 1.0e-6, 1.0e-6], "periodic": True},
        "unit_system": "SI",
        "inventory_basis": "mol_B_per_m3",
        "total_pseudo_binary_inventory": total,
        "assumptions": ["offline one-way snapshot", "xB_g is an effective parameter"],
        "observation_dataset_role": "Sheskin_2018_primary",
        "matrix": {
            "xB_alpha": 0.006219279767278563,
            "Vm_m3_per_mol": 2.0e-5,
            "B_inventory": matrix_inventory,
        },
        "gp_population": {
            "xB_g": 0.35,
            "Vm_m3_per_mol": 2.0e-5,
            "volume_fraction": 0.01,
            "B_inventory": gp_inventory,
        },
        "beta_subgrid_population": {
            "xB_beta": 1.0,
            "Vm_m3_per_mol": 2.1e-5,
            "volume_fraction": 0.005,
            "B_inventory": subgrid_inventory,
        },
        "beta_resolved_population": {
            "xB_beta": 1.0,
            "Vm_m3_per_mol": 2.1e-5,
            "volume_fraction": 0.002,
            "B_inventory": resolved_inventory,
            "shape_orientation_metadata": {"shape": "existing_pf_geometry", "orientation": "identity"},
        },
        "ledger": {
            "C_B_total": total,
            "C_B_matrix": matrix_inventory,
            "C_B_GP": gp_inventory,
            "C_B_beta_subgrid": subgrid_inventory,
            "C_B_beta_resolved": resolved_inventory,
            "residual": 0.0,
        },
    }
    arrays = {
        "gp_radius_bin_edges_m": np.array([0.5e-9, 1.0e-9, 2.0e-9]),
        "gp_number_density_per_m4": np.array([2.0e30, 1.0e30]),
        "beta_subgrid_radius_bin_edges_m": np.array([1.0e-9, 2.0e-9, 4.0e-9]),
        "beta_subgrid_number_density_per_m4": np.array([5.0e29, 2.0e29]),
        "beta_resolved_radius_bin_edges_m": np.array([4.0e-9, 8.0e-9, 16.0e-9]),
        "beta_resolved_expected_count": np.array([2.0, 1.0]),
        "beta_resolved_sampled_radii_m": np.array([6.0e-9, 12.0e-9]),
        "beta_resolved_centers_m": np.array([[1.0e-7, 2.0e-7, 3.0e-7], [7.0e-7, 8.0e-7, 9.0e-7]]),
    }
    return make_handoff_package(metadata, arrays)


class SchemaRoundTripTests(unittest.TestCase):
    def test_json_npz_roundtrip_preserves_four_bucket_ledger(self) -> None:
        package = valid_package()
        validation = validate_handoff_package(package)
        self.assertEqual(validation.C_B_total, 10.0)
        self.assertEqual(validation.C_B_bucket_sum, 10.0)
        self.assertEqual(validation.relative_residual, 0.0)
        with tempfile.TemporaryDirectory() as raw:
            paths = write_handoff_package(package, Path(raw))
            self.assertTrue(paths.metadata_path.is_file())
            self.assertTrue(paths.arrays_path.is_file())
            restored = read_handoff_package(Path(raw))
        assert_roundtrip_equivalent(package, restored)

    def test_validator_rejects_a_nonclosing_ledger(self) -> None:
        package = valid_package()
        broken_metadata = copy.deepcopy(package.metadata)
        broken_metadata["ledger"]["C_B_total"] = 10.1
        broken_metadata["total_pseudo_binary_inventory"] = 10.1
        broken = HandoffPackage(metadata=broken_metadata, arrays=package.arrays)
        with self.assertRaises(SchemaValidationError):
            validate_handoff_package(broken)

    def test_validator_rejects_unknown_array_keys(self) -> None:
        package = valid_package()
        arrays = dict(package.arrays)
        arrays["untracked_inventory"] = np.array([1.0])
        with self.assertRaises(SchemaValidationError):
            make_handoff_package(package.metadata, arrays)


class CurrentPfAdapterTests(unittest.TestCase):
    def test_nonzero_gp_or_subgrid_is_partial_and_is_not_added_to_matrix(self) -> None:
        package = valid_package()
        plan = adapt_kwn_to_pf(package)
        self.assertEqual(plan.status, PARTIAL_PF_STATE_NOT_CLOSED)
        self.assertFalse(plan.pf_state_closed)
        self.assertEqual(plan.materialized_state["matrix"]["B_inventory"], 5.0)
        self.assertNotIn("gp_population", plan.materialized_state)
        self.assertNotIn("beta_subgrid_population", plan.materialized_state)
        self.assertEqual(plan.unmapped_inventory, {"C_B_GP": 2.0, "C_B_beta_subgrid": 1.0})
        with self.assertRaises(PFStateNotClosedError):
            plan.require_pf_state_closed()
        audit = audit_handoff(package, plan)
        self.assertEqual(audit.status, PARTIAL_PF_STATE_NOT_CLOSED)
        self.assertFalse(audit.pf_raw_initialization_emitted)
        self.assertFalse(audit.double_counting_detected)
        self.assertEqual(audit.accounted_total, 10.0)

    def test_each_unrepresented_bucket_independently_forces_partial_state(self) -> None:
        for label, gp_inventory, subgrid_inventory in (
            ("gp_only", 2.0, 0.0),
            ("subgrid_only", 0.0, 1.0),
        ):
            with self.subTest(label=label):
                package = valid_package(
                    gp_inventory=gp_inventory,
                    subgrid_inventory=subgrid_inventory,
                )
                plan = adapt_kwn_to_pf(package)
                self.assertEqual(plan.status, PARTIAL_PF_STATE_NOT_CLOSED)
                self.assertEqual(plan.materialized_state["matrix"]["B_inventory"], 5.0)
                self.assertNotIn("gp_population", plan.materialized_state)
                self.assertNotIn("beta_subgrid_population", plan.materialized_state)

    def test_zero_unrepresented_buckets_makes_a_closed_pf_plan(self) -> None:
        package = valid_package(gp_inventory=0.0, subgrid_inventory=0.0)
        metadata = copy.deepcopy(package.metadata)
        metadata["gp_population"]["B_inventory"] = 0.0
        metadata["beta_subgrid_population"]["B_inventory"] = 0.0
        metadata["ledger"].update(
            {"C_B_total": 7.0, "C_B_GP": 0.0, "C_B_beta_subgrid": 0.0, "residual": 0.0}
        )
        metadata["total_pseudo_binary_inventory"] = 7.0
        arrays = dict(package.arrays)
        arrays["gp_number_density_per_m4"] = np.zeros(2)
        arrays["beta_subgrid_number_density_per_m4"] = np.zeros(2)
        closed_package = make_handoff_package(metadata, arrays)
        plan = adapt_kwn_to_pf(closed_package)
        self.assertEqual(plan.status, READY_FOR_PF_HANDOFF)
        plan.require_pf_state_closed()
        audit = audit_handoff(closed_package, plan)
        self.assertEqual(audit.status, PASS_HANDOFF_LEDGER_CLOSED)
        self.assertFalse(audit.pf_raw_initialization_emitted)
        self.assertTrue(audit.pf_raw_initialization_allowed)


class KwnExportTests(unittest.TestCase):
    def test_kwn_export_preserves_the_beta_subgrid_resolved_split(self) -> None:
        grid = RadiusGrid(np.array([1.0e-9, 2.0e-9, 4.0e-9]))
        gp_parameters = PopulationParameters(
            name="g",
            x_b=0.35,
            molar_volume_m3_mol=2.0e-5,
            diffusivity_m2_s=1.0e-18,
            gamma_j_m2=0.1,
            xeq_infinity=0.005,
        )
        beta_parameters = PopulationParameters(
            name="beta",
            x_b=1.0,
            molar_volume_m3_mol=2.1e-5,
            diffusivity_m2_s=1.0e-18,
            gamma_j_m2=0.17,
            xeq_infinity=0.005,
        )
        gp = Population(gp_parameters, grid, np.array([1.0e25, 0.0]))
        beta = Population(beta_parameters, grid, np.array([2.0e25, 3.0e25]))
        matrix_xb = 0.0062
        matrix_vm = 2.0e-5
        matrix_fraction = 1.0 - gp.volume_fraction() - beta.volume_fraction()
        total = (
            matrix_fraction * matrix_xb / matrix_vm
            + gp.b_inventory_mol_m3()
            + beta.b_inventory_mol_m3()
        )
        solver = SimpleNamespace(
            config=SimpleNamespace(
                matrix_molar_volume_m3_mol=matrix_vm,
                temperature_k=653.15,
                source_config_hash="e" * 64,
            ),
            ledger=SimpleNamespace(total_b_mol_m3=total),
            matrix_xb=matrix_xb,
            time_s=6.0 * 3600.0,
            solver_version="internal_kwn_finite_volume_v1",
        )
        solver.population = lambda name: {"g": gp, "beta": beta}[name]
        package = build_handoff_from_kwn_solver(
            solver,
            source_git_commit="f" * 40,
            pf_box_lengths_m=[1.0e-6, 1.0e-6, 1.0e-6],
            observation_dataset_role="Sheskin_2018_primary",
            beta_handoff_radius_m=2.0e-9,
            assumptions=["test fixture"],
        )
        self.assertGreater(package.arrays["beta_subgrid_number_density_per_m4"][0], 0.0)
        self.assertEqual(package.arrays["beta_subgrid_number_density_per_m4"][1], 0.0)
        self.assertEqual(package.arrays["beta_resolved_expected_count"][0], 0.0)
        self.assertGreater(package.arrays["beta_resolved_expected_count"][1], 0.0)
        self.assertEqual(adapt_kwn_to_pf(package).status, PARTIAL_PF_STATE_NOT_CLOSED)
        with self.assertRaises(KwnHandoffExportError):
            build_handoff_from_kwn_solver(
                solver,
                source_git_commit="f" * 40,
                pf_box_lengths_m=[1.0e-6, 1.0e-6, 1.0e-6],
                observation_dataset_role="Sheskin_2018_primary",
                beta_handoff_radius_m=3.0e-9,
                assumptions=["test fixture"],
            )


class PfSnapshotAndSpatialSamplingTests(unittest.TestCase):
    def test_pf_snapshot_import_never_invents_unresolved_populations(self) -> None:
        snapshot = PfResolvedSnapshot(
            source_git_commit="c" * 40,
            config_hash="d" * 64,
            temperature_K=653.15,
            handoff_time_s=12.0 * 3600.0,
            pf_box_lengths_m=[1.0e-6, 1.0e-6, 1.0e-6],
            matrix_xB_alpha=0.0062,
            matrix_B_inventory=3.0,
            matrix_Vm_m3_per_mol=2.0e-5,
            resolved_xB_beta=1.0,
            resolved_Vm_m3_per_mol=2.1e-5,
            resolved_volume_fraction=0.02,
            resolved_B_inventory=4.0,
            resolved_radius_bin_edges_m=np.array([4.0e-9, 8.0e-9]),
            resolved_expected_count=np.array([6.0]),
            gp_xB_g=0.35,
            gp_Vm_m3_per_mol=2.0e-5,
            beta_subgrid_Vm_m3_per_mol=2.1e-5,
            gp_radius_bin_edges_m=np.array([0.5e-9, 1.0e-9]),
            beta_subgrid_radius_bin_edges_m=np.array([1.0e-9, 2.0e-9]),
            observation_dataset_role="PF_authoritative_snapshot",
            assumptions=["source inventory audit completed"],
        )
        package = pf_snapshot_to_handoff(snapshot)
        self.assertEqual(package.metadata["ledger"]["C_B_GP"], 0.0)
        self.assertEqual(package.metadata["ledger"]["C_B_beta_subgrid"], 0.0)
        self.assertEqual(adapt_kwn_to_pf(package).status, READY_FOR_PF_HANDOFF)

    def test_periodic_sampler_is_reproducible_and_rejects_subparticle_loss(self) -> None:
        kwargs = {
            "radius_bin_edges_m": [4.0e-9, 8.0e-9, 16.0e-9],
            "expected_count": [2.0, 1.0],
            "box_lengths_m": [1.0e-6, 1.0e-6, 1.0e-6],
            "seed": 19,
        }
        first = sample_resolved_beta_geometry(**kwargs)
        second = sample_resolved_beta_geometry(**kwargs)
        self.assertEqual(first.radii_m.tolist(), second.radii_m.tolist())
        self.assertEqual(first.centers_m.tolist(), second.centers_m.tolist())
        self.assertEqual(first.radius_scale_relative_change, 0.0)
        validate_periodic_nonoverlap(first.radii_m, first.centers_m, kwargs["box_lengths_m"])
        with self.assertRaises(SpatialSamplingError):
            sample_resolved_beta_geometry(
                radius_bin_edges_m=[4.0e-9, 8.0e-9],
                expected_count=[0.3],
                box_lengths_m=[1.0e-6, 1.0e-6, 1.0e-6],
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
