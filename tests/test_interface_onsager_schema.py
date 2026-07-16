import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas" / "interface_onsager_data.schema.json"
TEMPLATE = ROOT / "examples" / "interface_onsager_data_template.csv"


class InterfaceOnsagerSchemaTest(unittest.TestCase):
    def test_schema_has_hard_contract(self):
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        required = set(schema["required"])
        expected = {
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
        }
        self.assertTrue(expected.issubset(required))
        contract = schema["x-project-contract"]
        self.assertEqual(contract["normal_direction"], "beta_to_matrix")
        self.assertEqual(contract["positive_Vn"], "beta_growth")
        self.assertEqual(contract["positive_JB"], "matrix_to_beta")
        self.assertIn("PF_curved_velocity_matrix", contract["forbidden_fit_sources"])

    def test_template_covers_minimum_two_temperature_matrix(self):
        with TEMPLATE.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 24)
        self.assertTrue(all(row["record_status"] == "TEMPLATE" for row in rows))
        for temperature, prefix in (("653.15", "T380"), ("673.15", "T400")):
            subset = [row for row in rows if row["temperature_K"] == temperature]
            self.assertEqual(len(subset), 12)
            states = {row["state_id"] for row in subset}
            expected = {
                f"{prefix}_phase_positive",
                f"{prefix}_phase_negative",
                f"{prefix}_diffusion_positive",
                f"{prefix}_diffusion_negative",
                f"{prefix}_mixed_1",
                f"{prefix}_mixed_2",
            }
            self.assertEqual(states, expected)
            for state in states:
                replicates = {row["replicate_id"] for row in subset if row["state_id"] == state}
                self.assertEqual(replicates, {"replicate_1", "replicate_2"})

    def test_template_cannot_be_mistaken_for_measurements(self):
        with TEMPLATE.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        numeric_measurements = ("measured_Vn", "measured_JB", "uncertainty_Vn", "uncertainty_JB")
        self.assertTrue(all(not row[key] for row in rows for key in numeric_measurements))


if __name__ == "__main__":
    unittest.main()
