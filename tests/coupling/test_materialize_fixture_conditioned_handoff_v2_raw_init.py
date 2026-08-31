"""Regression coverage for fixture-conditioned v2 raw-init materialization."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from coupling.fixture_conditioned_handoff_v2 import (
    build_fixture_conditioned_handoff_v2,
    h_of_phi,
    load_host_96cube_fixture,
    load_validation_contract,
    reconstruct_matrix_xb_alpha,
    write_fixture_conditioned_handoff_v2,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from materialize_fixture_conditioned_kwn_pf_handoff_v2_raw_init import (  # noqa: E402
    META_FILENAME,
    PHI_FILENAME,
    PROVENANCE_FILENAME,
    XB_FILENAME,
    RawInitMaterializationError,
    materialize_fixture_conditioned_handoff_v2_raw_init,
)


CONTRACT_PATH = ROOT / "contracts" / "pf_kwn_validation_contract_v1.json"
FIXTURE_SPEC = (
    ROOT
    / "data"
    / "qualification"
    / "pf_mass_conserving_library_handoff_v1"
    / "six_particle_96cube_spec.json"
)
HOST_PROFILE_ROOT = (
    ROOT.parent
    / "CUDA_STO_PF"
    / "data"
    / "qualification"
    / "pf_elastic_target_profile_quarter_nm_v2"
    / "profiles"
)


class FixtureConditionedRawInitTests(unittest.TestCase):
    """Materialization uses the real read-only source profiles, not a control."""

    def test_materializes_hash_bound_raw_fields_deterministically(self) -> None:
        if not HOST_PROFILE_ROOT.is_dir():
            self.skipTest("frozen host profile fields are unavailable")
        contract = load_validation_contract(CONTRACT_PATH)
        fixture = load_host_96cube_fixture(
            FIXTURE_SPEC, HOST_PROFILE_ROOT, contract
        )
        metadata, arrays, _ = build_fixture_conditioned_handoff_v2(
            fixture, contract
        )
        with tempfile.TemporaryDirectory(prefix="fixture_v2_raw_init_") as raw:
            root = Path(raw)
            handoff = root / "handoff"
            write_fixture_conditioned_handoff_v2(
                handoff, metadata, arrays, fixture, contract
            )
            written_metadata = json.loads(
                (handoff / "metadata.json").read_text(encoding="utf-8")
            )
            first = root / "first"
            second = root / "second"
            paths = materialize_fixture_conditioned_handoff_v2_raw_init(
                handoff_dir=handoff,
                out=first,
                contract_path=CONTRACT_PATH,
                fixture_spec_path=FIXTURE_SPEC,
                profile_root=HOST_PROFILE_ROOT,
            )
            materialize_fixture_conditioned_handoff_v2_raw_init(
                handoff_dir=handoff,
                out=second,
                contract_path=CONTRACT_PATH,
                fixture_spec_path=FIXTURE_SPEC,
                profile_root=HOST_PROFILE_ROOT,
            )

            self.assertEqual(
                (first / PHI_FILENAME).read_bytes(),
                (second / PHI_FILENAME).read_bytes(),
            )
            self.assertEqual(
                (first / XB_FILENAME).read_bytes(),
                (second / XB_FILENAME).read_bytes(),
            )
            self.assertEqual(
                (first / META_FILENAME).read_bytes(),
                (second / META_FILENAME).read_bytes(),
            )
            self.assertEqual(
                (first / PROVENANCE_FILENAME).read_bytes(),
                (second / PROVENANCE_FILENAME).read_bytes(),
            )

            count = int(np.prod(contract.grid_shape))
            phi = np.fromfile(paths["phi"], dtype="<f8").reshape(
                contract.grid_shape, order="C"
            )
            x_b = np.fromfile(paths["xB"], dtype="<f8").reshape(
                contract.grid_shape, order="C"
            )
            self.assertEqual(paths["phi"].stat().st_size, count * 8)
            self.assertEqual(paths["xB"].stat().st_size, count * 8)
            self.assertTrue(np.array_equal(phi, fixture.phi))
            self.assertTrue(
                np.array_equal(x_b, reconstruct_matrix_xb_alpha(fixture, arrays))
            )
            recovered_delta = fixture.alpha * (
                x_b - float(arrays["matrix_baseline_xB"][0])
            )
            self.assertLessEqual(
                float(np.max(np.abs(recovered_delta - fixture.delta_C_relaxation))),
                5.0e-14,
            )

            init_meta = json.loads((first / META_FILENAME).read_text(encoding="utf-8"))
            for key in (
                "Nx",
                "Ny",
                "Nz",
                "dx_nm",
                "interface_width_nm",
                "dtype",
                "order",
                "phi_path",
                "xB_path",
                "phi_sha256",
                "xB_sha256",
                "validation_contract_hash",
                "source_handoff_hash",
                "package_hash",
                "fixture_hash",
                "auxiliary_sidecar_sha256",
            ):
                self.assertIn(key, init_meta)
            self.assertEqual(init_meta["dtype"], "float64")
            self.assertEqual(init_meta["order"], "C")
            self.assertEqual(init_meta["phi_path"], PHI_FILENAME)
            self.assertEqual(init_meta["xB_path"], XB_FILENAME)
            self.assertEqual(init_meta["validation_contract_hash"], contract.contract_hash)
            self.assertEqual(
                init_meta["source_handoff_hash"],
                written_metadata["source_handoff_hash"],
            )
            self.assertEqual(
                init_meta["package_hash"], written_metadata["package_hash"]
            )
            self.assertEqual(init_meta["fixture_hash"], fixture.fixture_hash)
            expected_mean = float(
                np.mean(
                    (1.0 - h_of_phi(phi)) * x_b + h_of_phi(phi) * contract.v_B,
                    dtype=np.float64,
                )
            )
            self.assertAlmostEqual(init_meta["mean_xBtot"], expected_mean)

            provenance = json.loads(
                (first / PROVENANCE_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(
                provenance["pf_execution"],
                "NOT_RUN_RAW_INITIALIZATION_MATERIALIZATION_ONLY",
            )
            self.assertTrue(
                provenance["preservation"]["phi_fixture_hash_matches_raw"]
            )
            self.assertTrue(
                provenance["inventory"][
                    "auxiliary_inventory_is_not_embedded_in_raw_fields"
                ]
            )

            with self.assertRaises(RawInitMaterializationError):
                materialize_fixture_conditioned_handoff_v2_raw_init(
                    handoff_dir=handoff,
                    out=first,
                    contract_path=CONTRACT_PATH,
                    fixture_spec_path=FIXTURE_SPEC,
                    profile_root=HOST_PROFILE_ROOT,
                )


if __name__ == "__main__":
    unittest.main()
