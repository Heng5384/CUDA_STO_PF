import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_staggered_v1_step655_replay import (
    PHYSICS_MODEL,
    REFERENCE,
    STAGGERED,
    build_case,
    parse_params,
    validate_frozen_physics,
)


BASE_VALUES = {
    "composition_evolution_mode": "ctot_mimetic_be",
    "D_compound": "0.00000000000000000e+00",
    "D_beta_for_calibration": "0.00000000000000000e+00",
    "v_B": "1.00000000000000000e+00",
    "coarse_interface_mobility_mode": "off",
    "coarse_interface_mobility_a_M": "0.00000000000000000e+00",
    "ctot_finite_interface_antitrapping_enabled": "0",
    "GP_population_mode": "OFF",
    "gp_growth_enabled": "0",
    "enable_legacy_gp_storage_coupling": "0",
    "diagnostic_rsmd_enabled": "0",
}


class PrepareStaggeredStep655ReplayTests(unittest.TestCase):
    def make_base(self, root: Path) -> tuple[Path, dict[str, object]]:
        params = root / "base.params"
        values = dict(BASE_VALUES)
        values.update(
            {
                "PF_RESEARCH_MODEL": "fixed_ctot_ji_chen_coarse4_gp_v1",
                "coarse_model_name": "fixed_ctot_ji_chen_coarse4_gp_v1",
                "coarse_model_version": "1",
                "ctot_outer_acceleration": "ANDERSON_M3_V1",
                "ctot_debug_transport_floor_audit": "1",
                "init_case_tag": "old",
            }
        )
        params.write_text(
            "\n".join(f"{key}={value}" for key, value in values.items()) + "\n"
        )
        meta = {
            "schema": "ctot_checkpoint_v1",
            "coarse_model_name": "fixed_ctot_ji_chen_coarse4_gp_v1",
            "coarse_model_version": "1",
            "step": 54,
        }
        return params, meta

    def test_frozen_physics_validation_is_fail_closed(self):
        validate_frozen_physics(dict(BASE_VALUES))
        changed = dict(BASE_VALUES)
        changed["D_compound"] = "1.0"
        with self.assertRaises(ValueError):
            validate_frozen_physics(changed)

    def test_case_views_change_contract_metadata_not_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            params, meta = self.make_base(root)
            case = build_case(
                root / "cases", params, meta, "staggered", STAGGERED,
                "ALWAYS_ONE_POLISH", "OFF", -1.0, -1.0,
            )
            _, values = parse_params(Path(case["params"]))
            for key, expected in BASE_VALUES.items():
                self.assertEqual(values[key], expected)
            self.assertEqual(values["PF_RESEARCH_MODEL"], PHYSICS_MODEL)
            self.assertEqual(values["coarse_model_name"], PHYSICS_MODEL)
            self.assertEqual(values["coarse_model_version"], "2")
            self.assertEqual(values["ctot_numerics_contract"], STAGGERED)
            self.assertEqual(values["ctot_split_defect_policy"], "ALWAYS_ONE_POLISH")
            self.assertEqual(values["ctot_outer_acceleration"], "OFF")
            self.assertEqual(values["ctot_debug_transport_floor_audit"], "0")
            migrated = json.loads(Path(case["meta"]).read_text())
            self.assertEqual(migrated["PF_RESEARCH_MODEL"], PHYSICS_MODEL)
            self.assertEqual(migrated["ctot_numerics_contract"], STAGGERED)
            self.assertFalse(migrated["contract_view_migration"]["raw_state_modified"])

    def test_reference_contract_retains_m3(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            params, meta = self.make_base(root)
            case = build_case(
                root / "cases", params, meta, "reference", REFERENCE,
                "ALWAYS_ONE_POLISH", "ANDERSON_M3_V1", -1.0, -1.0,
            )
            _, values = parse_params(Path(case["params"]))
            self.assertEqual(values["ctot_numerics_contract"], REFERENCE)
            self.assertEqual(values["ctot_outer_acceleration"], "ANDERSON_M3_V1")

    def test_dt_subdivision_is_explicit_and_does_not_touch_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            params, meta = self.make_base(root)
            case = build_case(
                root / "cases", params, meta, "staggered_dt2", STAGGERED,
                "ALWAYS_ONE_POLISH", "OFF", -1.0, -1.0,
                dt_override=0.0015625, replay_steps=2,
            )
            _, values = parse_params(Path(case["params"]))
            self.assertEqual(values["dt"], "1.56250000000000009e-03")
            self.assertEqual(case["replay_steps"], 2)
            for key, expected in BASE_VALUES.items():
                self.assertEqual(values[key], expected)

    def test_cold_kkt_audit_is_explicit_and_transaction_neutral(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            params, meta = self.make_base(root)
            case = build_case(
                root / "cases", params, meta, "cold_kkt", STAGGERED,
                "ALWAYS_ONE_POLISH", "OFF", -1.0, -1.0,
                dt_override=0.0015625,
                debug_transport_floor_audit=True,
            )
            _, values = parse_params(Path(case["params"]))
            self.assertEqual(values["ctot_debug_transport_floor_audit"], "1")
            self.assertTrue(case["debug_transport_floor_audit"])
            migrated = json.loads(Path(case["meta"]).read_text())
            self.assertFalse(migrated["contract_view_migration"]["raw_state_modified"])

    def test_optional_skip_bracket_is_marked_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            params, meta = self.make_base(root)
            case = build_case(
                root / "cases", params, meta, "optional_bracket", STAGGERED,
                "OPTIONAL_ONE_POLISH", "OFF", 0.3, 0.3,
                dt_override=0.003125, diagnostic_only=True,
            )
            _, values = parse_params(Path(case["params"]))
            self.assertEqual(values["ctot_split_defect_policy"],
                             "OPTIONAL_ONE_POLISH")
            self.assertEqual(values["ctot_split_defect_skip_threshold"],
                             "2.99999999999999989e-01")
            self.assertTrue(case["diagnostic_only"])

    def test_refined_optional_skip_bracket_keeps_equal_physical_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            params, meta = self.make_base(root)
            base_dt = 0.003125
            for divisor in (8, 16, 32, 64, 128):
                case = build_case(
                    root / f"cases_{divisor}", params, meta,
                    f"optional_dt{divisor}", STAGGERED,
                    "OPTIONAL_ONE_POLISH", "OFF", 0.3, 0.3,
                    dt_override=base_dt / divisor,
                    replay_steps=divisor,
                    diagnostic_only=True,
                )
                _, values = parse_params(Path(case["params"]))
                self.assertAlmostEqual(float(values["dt"]) * case["replay_steps"],
                                       base_dt)
                self.assertTrue(case["diagnostic_only"])


if __name__ == "__main__":
    unittest.main()
