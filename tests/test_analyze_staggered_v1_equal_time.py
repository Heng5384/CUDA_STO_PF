import tempfile
from pathlib import Path
import unittest

import numpy as np

from scripts.analyze_staggered_v1_equal_time import (
    compare_candidate,
    h_phi,
    load_state,
    load_frozen_csv,
    load_split_metrics,
    parse_candidate,
    parse_shape,
    threshold_crossings,
)


class AnalyzeStaggeredV1EqualTimeTests(unittest.TestCase):
    def test_h_phi_endpoints(self):
        values = h_phi(np.asarray([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(values, [0.0, 0.5, 1.0])

    def test_parser_rejects_ambiguous_inputs(self):
        self.assertEqual(parse_shape("8x1x1"), (8, 1, 1))
        self.assertEqual(parse_candidate("fine=/tmp/fine"), ("fine", Path("/tmp/fine")))
        with self.assertRaises(ValueError):
            parse_shape("8,1")
        with self.assertRaises(ValueError):
            parse_candidate("missing_prefix")

    def test_increment_normalization_prevents_background_dilution(self):
        shape = (8, 1, 1)
        initial_phi = np.asarray([0.0, 0.1, 0.4, 0.9, 1.0, 0.9, 0.4, 0.1]).reshape(shape)
        reference_phi = initial_phi.copy()
        reference_phi[2, 0, 0] += 1.0e-3
        candidate_phi = initial_phi.copy()
        candidate_phi[2, 0, 0] += 0.5e-3
        initial = {
            "Ctot": np.full(shape, 0.03),
            "phi": initial_phi,
            "xB_alpha": np.full(shape, 0.01),
        }
        reference = {
            "Ctot": initial["Ctot"].copy(),
            "phi": reference_phi,
            "xB_alpha": initial["xB_alpha"].copy(),
        }
        candidate = {
            "Ctot": initial["Ctot"].copy(),
            "phi": candidate_phi,
            "xB_alpha": initial["xB_alpha"].copy(),
        }
        row = compare_candidate("half_increment", initial, reference, candidate, 1.0, 0.005)
        self.assertAlmostEqual(row["phi_increment_error_linf_rel"], 0.5)
        self.assertFalse(row["optional_skip_qoi_eligible"])

    def test_identical_candidate_is_eligible(self):
        shape = (8, 1, 1)
        phi0 = np.asarray([0.0, 0.1, 0.4, 0.9, 1.0, 0.9, 0.4, 0.1]).reshape(shape)
        phi1 = phi0.copy()
        phi1[2, 0, 0] += 1.0e-3
        initial = {
            "Ctot": np.linspace(0.01, 0.08, 8).reshape(shape),
            "phi": phi0,
            "xB_alpha": np.linspace(0.01, 0.02, 8).reshape(shape),
        }
        reference = {name: values.copy() for name, values in initial.items()}
        reference["Ctot"][1, 0, 0] += 1.0e-4
        reference["Ctot"][7, 0, 0] -= 1.0e-4
        reference["phi"] = phi1
        reference["xB_alpha"][1, 0, 0] += 1.0e-5
        row = compare_candidate("exact", initial, reference, reference, 1.0, 0.005)
        self.assertEqual(row["Ctot_mass_candidate_minus_reference"], 0.0)
        self.assertTrue(row["phase_volume_direction_unchanged"])
        self.assertTrue(row["optional_skip_qoi_eligible"])

    def test_rejected_comparator_cannot_become_skip_eligible(self):
        shape = (2, 1, 1)
        initial = {
            "Ctot": np.asarray([0.02, 0.04]).reshape(shape),
            "phi": np.asarray([0.1, 0.9]).reshape(shape),
            "xB_alpha": np.asarray([0.02, 0.02]).reshape(shape),
        }
        reference = {name: values.copy() for name, values in initial.items()}
        reference["Ctot"][0, 0, 0] += 1.0e-5
        reference["Ctot"][1, 0, 0] -= 1.0e-5
        reference["phi"][0, 0, 0] += 1.0e-5
        row = compare_candidate(
            "rejected", initial, reference, reference, 1.0, 0.005,
            candidate_role="rejected_coupled_comparator",
        )
        self.assertFalse(row["optional_skip_qoi_eligible"])

    def test_raw_loader_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "state"
            for suffix in ("_Ctot.raw", "_phi.raw", "_xB_alpha.raw"):
                np.zeros(7, dtype=np.float64).tofile(str(prefix) + suffix)
            with self.assertRaises(ValueError):
                load_state(prefix, (8, 1, 1))

    def test_frozen_csv_loader_maps_runtime_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.csv"
            path.write_text(
                "idx,C_trial,phi,xalpha\n"
                + "\n".join(
                    f"{index},{0.01 + index},{0.1 * index},{0.02 + index}"
                    for index in range(8)
                )
                + "\n",
                encoding="utf-8",
            )
            state = load_frozen_csv(path, (8, 1, 1))
            self.assertEqual(state["Ctot"].shape, (8, 1, 1))
            self.assertAlmostEqual(state["phi"][7, 0, 0], 0.7)
            self.assertAlmostEqual(state["xB_alpha"][0, 0, 0], 0.02)

    def test_periodic_crossings_are_detected(self):
        profile = np.asarray([0.2, 0.8, 0.8, 0.2])
        crossings = threshold_crossings(profile)
        np.testing.assert_allclose(crossings, [0.5, 2.5])

    def test_split_metrics_are_bound_to_candidate_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root / "ctot_checkpoint_step000002"
            (root / "ctot_split_step_metrics.csv").write_text(
                "physical_step,split_defect_eta,polish_applied,polish_skipped,"
                "transport_residual,phase_KKT,mass_error,accepted\n"
                "1,0.02,0,1,1e-6,2e-11,0,1\n"
                "2,0.01,0,1,2e-6,3e-11,-1e-15,1\n",
                encoding="utf-8",
            )
            metrics = load_split_metrics(prefix)
            self.assertTrue(metrics["runtime_metrics_available"])
            self.assertEqual(metrics["runtime_step_rows"], 2)
            self.assertEqual(metrics["polish_skipped_count"], 2)
            self.assertAlmostEqual(metrics["split_eta_max"], 0.02)
            self.assertTrue(metrics["runtime_all_accepted"])


if __name__ == "__main__":
    unittest.main()
