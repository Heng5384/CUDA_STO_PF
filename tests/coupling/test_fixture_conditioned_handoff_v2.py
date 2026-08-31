"""Focused regression tests for the validation-only fixture-conditioned v2 handoff."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from coupling.fixture_conditioned_handoff_v2 import (
    AUXILIARY_SIDECAR_FILENAME,
    AUXILIARY_SIDECAR_SCHEMA_VERSION,
    AUXILIARY_SIDECAR_SEMANTICS,
    FixtureConditionedHandoffError,
    InfeasibleFixtureConditionedHandoff,
    build_fixture_conditioned_handoff_v2,
    field_matrix_inventory_mol,
    load_validation_contract,
    make_synthetic_fixture_control,
    map_matrix_inventory_preserving_fixture,
    read_fixture_conditioned_handoff_v2,
    render_auxiliary_population_sidecar_v1,
    validate_fixture_conditioned_handoff_v2,
    write_fixture_conditioned_handoff_v2,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "contracts" / "pf_kwn_validation_contract_v1.json"
FIXTURE_SPEC = (
    ROOT
    / "data"
    / "qualification"
    / "pf_mass_conserving_library_handoff_v1"
    / "six_particle_96cube_spec.json"
)


class FixtureConditionedHandoffV2Tests(unittest.TestCase):
    """The suite uses only a labelled synthetic control, never historical fields."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_validation_contract(CONTRACT_PATH)
        cls.fixture = make_synthetic_fixture_control(cls.contract, FIXTURE_SPEC)

    def test_four_buckets_close_with_nonzero_frozen_gp(self) -> None:
        metadata, arrays, _ = build_fixture_conditioned_handoff_v2(
            self.fixture, self.contract
        )
        report = validate_fixture_conditioned_handoff_v2(
            metadata, arrays, self.fixture, self.contract
        )
        self.assertEqual(report["status"], "PASS_FIXTURE_CONDITIONED_HANDOFF_V2")
        self.assertGreater(report["ledger"]["Q_B_GP_mol"], 0.0)
        self.assertGreater(
            report["ledger"]["Q_B_beta_resolved_fixed_mol"], 0.0
        )
        self.assertLessEqual(report["relative_residual"], 1.0e-12)
        self.assertEqual(
            self.fixture.source_kind,
            "SYNTHETIC_STORAGE_CONTROL_NOT_HISTORICAL_NOT_PRODUCTION",
        )

    def test_baseline_inverse_preserves_geometry_and_relaxation(self) -> None:
        phi_before = self.fixture.phi.copy()
        delta_before = self.fixture.delta_C_relaxation.copy()
        target = self.fixture.source_matrix_inventory_mol * 0.98
        x_b, audit = map_matrix_inventory_preserving_fixture(
            self.fixture, self.contract, target
        )
        q_matrix = field_matrix_inventory_mol(
            self.fixture.alpha,
            x_b,
            self.fixture.voxel_volume_m3,
            self.contract.vm_alpha_m3_mol,
        )
        self.assertTrue(np.array_equal(phi_before, self.fixture.phi))
        self.assertTrue(
            np.array_equal(delta_before, self.fixture.delta_C_relaxation)
        )
        self.assertLessEqual(audit["relative_residual"], 1.0e-12)
        self.assertAlmostEqual(q_matrix, target, delta=abs(target) * 1.0e-12)
        self.assertEqual(audit["clipping_used"], 0.0)

    def test_validator_rejects_resolved_double_count(self) -> None:
        metadata, arrays, _ = build_fixture_conditioned_handoff_v2(
            self.fixture, self.contract
        )
        broken = copy.deepcopy(metadata)
        broken["ledger"]["Q_B_beta_resolved_fixed_mol"] *= 2.0
        with self.assertRaises(FixtureConditionedHandoffError):
            validate_fixture_conditioned_handoff_v2(
                broken, arrays, self.fixture, self.contract
            )

    def test_validator_rejects_contract_hash_mismatch(self) -> None:
        """A package cannot be read under a different validation contract."""

        metadata, arrays, _ = build_fixture_conditioned_handoff_v2(
            self.fixture, self.contract
        )
        broken = copy.deepcopy(metadata)
        broken["contract_hash"] = "0" * 64
        with self.assertRaises(FixtureConditionedHandoffError):
            validate_fixture_conditioned_handoff_v2(
                broken, arrays, self.fixture, self.contract
            )

    def test_package_roundtrip_preserves_psd_and_matrix_payload(self) -> None:
        metadata, arrays, _ = build_fixture_conditioned_handoff_v2(
            self.fixture, self.contract
        )
        with tempfile.TemporaryDirectory(prefix="fixture_conditioned_v2_") as raw:
            root = Path(raw) / "package"
            write_fixture_conditioned_handoff_v2(
                root, metadata, arrays, self.fixture, self.contract
            )
            restored, restored_arrays, report = read_fixture_conditioned_handoff_v2(
                root, self.fixture, self.contract
            )
        self.assertEqual(restored["contract_hash"], self.contract.contract_hash)
        self.assertEqual(report["double_count_check"]["status"], "PASS_NO_DOUBLE_COUNT")
        for name, value in arrays.items():
            self.assertTrue(np.array_equal(value, restored_arrays[name]), name)

    def test_package_emits_compact_deterministic_auxiliary_sidecar(self) -> None:
        metadata, arrays, _ = build_fixture_conditioned_handoff_v2(
            self.fixture, self.contract
        )
        with tempfile.TemporaryDirectory(prefix="fixture_conditioned_sidecar_") as raw:
            root = Path(raw) / "package"
            paths = write_fixture_conditioned_handoff_v2(
                root, metadata, arrays, self.fixture, self.contract
            )
            restored, restored_arrays, _ = read_fixture_conditioned_handoff_v2(
                root, self.fixture, self.contract
            )
            text = paths["auxiliary_sidecar"].read_text(encoding="utf-8")
        self.assertEqual(paths["auxiliary_sidecar"].name, AUXILIARY_SIDECAR_FILENAME)
        self.assertEqual(
            text,
            render_auxiliary_population_sidecar_v1(restored, restored_arrays),
        )
        self.assertIn(f"schema_version={AUXILIARY_SIDECAR_SCHEMA_VERSION}", text)
        self.assertIn(f"validation_contract_hash={self.contract.contract_hash}", text)
        self.assertIn("gp_bin=", text)
        self.assertIn("beta_subgrid_bin=", text)
        self.assertNotIn("matrix_baseline_xB", text)
        self.assertEqual(
            restored["auxiliary_sidecar"]["semantics"],
            AUXILIARY_SIDECAR_SEMANTICS,
        )

    def test_infeasible_target_fails_without_clamping(self) -> None:
        impossible = (
            self.fixture.box_volume_m3
            / self.contract.vm_alpha_m3_mol
            * 2.0
        )
        with self.assertRaises(InfeasibleFixtureConditionedHandoff) as context:
            map_matrix_inventory_preserving_fixture(
                self.fixture, self.contract, impossible
            )
        self.assertGreater(context.exception.minimum_inventory_change_mol, 0.0)


if __name__ == "__main__":
    unittest.main()
