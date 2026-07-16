import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "fit_interface_onsager_matrix.py"
SPEC = importlib.util.spec_from_file_location("fit_interface_onsager_matrix", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


HEADER = [
    "record_status",
    "temperature_K",
    "phase_driving",
    "diffusion_driving",
    "measured_Vn",
    "measured_JB",
    "uncertainty_Vn",
    "uncertainty_JB",
    "covariance",
    "interface_orientation",
    "coherency_state",
    "elastic_constraint",
    "source_type",
    "source_reference",
    "state_id",
    "replicate_id",
    "notes",
]


def write_rows(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def synthetic_rows(matrix: np.ndarray, temperature=673.15):
    forces = [
        (0.8, 0.0),
        (-0.8, 0.0),
        (0.0, 0.6),
        (0.0, -0.6),
        (0.5, 0.4),
        (-0.3, 0.7),
    ]
    rows = []
    for index, force in enumerate(forces):
        flux = np.linalg.solve(matrix, np.asarray(force, dtype=float))
        for replicate in (1, 2):
            rows.append(
                {
                    "record_status": "DATA",
                    "temperature_K": temperature,
                    "phase_driving": force[0],
                    "diffusion_driving": force[1],
                    "measured_Vn": flux[0],
                    "measured_JB": flux[1],
                    "uncertainty_Vn": 1.0e-3,
                    "uncertainty_JB": 1.0e-3,
                    "covariance": 0.0,
                    "interface_orientation": "[100]",
                    "coherency_state": "coherent",
                    "elastic_constraint": "plane_strain",
                    "source_type": "standalone_planar_rsmd",
                    "source_reference": "synthetic_unit_test_only",
                    "state_id": f"state_{index}",
                    "replicate_id": f"replicate_{replicate}",
                    "notes": "unit test",
                }
            )
    return rows


class InterfaceOnsagerFitTest(unittest.TestCase):
    def test_template_only_is_rejected(self):
        with self.assertRaisesRegex(MODULE.DataContractError, "no DATA records"):
            MODULE.load_observations(ROOT / "examples" / "interface_onsager_data_template.csv")

    def test_forbidden_pf_curved_source_is_rejected(self):
        matrix = np.array([[2.0, 0.25], [0.25, 1.5]])
        rows = synthetic_rows(matrix)
        rows[0]["source_reference"] = "next2 PF_curved velocity matrix"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forbidden.csv"
            write_rows(path, rows)
            with self.assertRaisesRegex(MODULE.DataContractError, "forbidden fit source"):
                MODULE.load_observations(path)

    def test_fixed_a_spd_recovers_synthetic_matrix(self):
        expected = np.array([[2.0, 0.25], [0.25, 1.5]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            write_rows(path, synthetic_rows(expected))
            observations = MODULE.load_observations(path)
        rank, required, _ = MODULE.design_rank(observations, "fixed_A_spd")
        self.assertEqual((rank, required), (2, 2))
        fit = MODULE.fit_temperature_group(observations, "fixed_A_spd", fixed_A=2.0)
        actual = np.asarray(fit["matrix"])
        np.testing.assert_allclose(actual, expected, rtol=1.0e-9, atol=1.0e-10)
        self.assertGreater(fit["minimum_SPD_margin"], 0.0)
        self.assertEqual(fit["cross_coupling_status"], "REQUIRED_AT_95_PERCENT")

    def test_full_cholesky_fit_recovers_synthetic_matrix(self):
        expected = np.array([[1.7, -0.2], [-0.2, 0.9]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            write_rows(path, synthetic_rows(expected))
            observations = MODULE.load_observations(path)
        rank, required, _ = MODULE.design_rank(observations, "full_spd")
        self.assertEqual((rank, required), (3, 3))
        fit = MODULE.fit_temperature_group(observations, "full_spd")
        np.testing.assert_allclose(np.asarray(fit["matrix"]), expected, rtol=1.0e-9, atol=1.0e-10)
        self.assertGreater(fit["determinant"], 0.0)

    def test_non_positive_observation_covariance_is_rejected(self):
        matrix = np.eye(2)
        rows = synthetic_rows(matrix)
        rows[0]["covariance"] = 1.0e-6
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_covariance.csv"
            write_rows(path, rows)
            with self.assertRaisesRegex(MODULE.DataContractError, "not positive definite"):
                MODULE.load_observations(path)


if __name__ == "__main__":
    unittest.main()
