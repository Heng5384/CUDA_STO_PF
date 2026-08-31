"""Host-profile integration regression for the four-bucket storage control."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.run_four_bucket_storage_controls import (
    DEFAULT_HOST_PROFILE_ROOT,
    ROOT,
    run_controls,
)


CONTRACT_PATH = ROOT / "contracts" / "pf_kwn_validation_contract_v1.json"
FIXTURE_SPEC = (
    ROOT
    / "data"
    / "qualification"
    / "pf_mass_conserving_library_handoff_v1"
    / "six_particle_96cube_spec.json"
)


@unittest.skipUnless(
    DEFAULT_HOST_PROFILE_ROOT.exists(),
    "host-only frozen profile source is unavailable",
)
class FourBucketStorageControlsTests(unittest.TestCase):
    """These controls use the actual read-only host six-particle fields."""

    def test_actual_fixture_closes_without_cuda_or_raw_field_mutation(self) -> None:
        summary, rows = run_controls(
            contract_path=CONTRACT_PATH,
            fixture_spec_path=FIXTURE_SPEC,
            profile_root=DEFAULT_HOST_PROFILE_ROOT,
            gp_fraction=0.01,
            subgrid_fraction=0.005,
            transfer_fraction=0.01,
            run_cpp_v5_test=False,
        )
        self.assertEqual(
            summary["status"], "PASS_FOUR_BUCKET_HOST_STORAGE_CONTROL_NOT_CUDA"
        )
        self.assertEqual(summary["cuda_pf_dynamics"], "NOT_RUN")
        self.assertEqual(
            summary["S1_zero_aux_identity"]["source_field_preservation"][
                "source_phi_max_abs_difference"
            ],
            0.0,
        )
        self.assertEqual(
            summary["S2_nonzero_frozen_aux_storage_only"][
                "source_field_preservation"
            ]["source_xB_alpha_max_abs_difference"],
            0.0,
        )
        self.assertLessEqual(
            summary["S2_nonzero_frozen_aux_storage_only"]["ledger"][
                "relative_residual"
            ],
            1.0e-12,
        )
        self.assertGreater(
            summary["S2_nonzero_frozen_aux_storage_only"]["total_increase_mol"],
            0.0,
        )
        self.assertEqual(
            summary["S2_nonzero_frozen_aux_storage_only"][
                "mapped_matrix_max_xB_difference_from_source"
            ],
            0.0,
        )
        self.assertLessEqual(
            summary["S3_exact_matrix_to_GP_inverse_transfer_and_reverse"][
                "max_xB_roundtrip_error"
            ],
            1.0e-12,
        )
        self.assertEqual(summary["S4_cpp_V5_checkpoint_dependency"]["cuda_pf_dynamics"], "NOT_RUN")
        self.assertEqual(len(rows), 21)


if __name__ == "__main__":
    unittest.main()
