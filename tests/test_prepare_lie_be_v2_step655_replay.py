import json
import pathlib
import tempfile
import unittest

from scripts.prepare_lie_be_v2_step655_replay import (
    BASE_DT,
    DIVISORS,
    LIE_BE_V2,
    LIE_POLICY,
    build_case,
    parse_params,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


class PrepareLieBeV2Step655ReplayTests(unittest.TestCase):
    def setUp(self):
        self.base_params = next(
            (ROOT / "tmp/staggered_v1_step655_20260716").glob(
                "staggered_dt2/Results/**/pf_input.params"
            )
        )
        self.base_meta_path = (
            ROOT
            / "tmp/staggered_v1_step655_20260716/frozen_input/"
            "ctot_checkpoint_step000054_meta.json"
        )
        self.base_meta = json.loads(self.base_meta_path.read_text(encoding="utf-8"))

    def test_full_and_half_cases_are_equal_time_and_default_off_from_physics(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            full = build_case(root, self.base_params, self.base_meta, 8, False)
            half = build_case(root, self.base_params, self.base_meta, 8, True)
            _, full_values = parse_params(pathlib.Path(full["params"]))
            _, half_values = parse_params(pathlib.Path(half["params"]))
            self.assertEqual(full_values["ctot_numerics_contract"], LIE_BE_V2)
            self.assertEqual(full_values["ctot_split_defect_policy"], LIE_POLICY)
            self.assertEqual(full_values["ctot_max_coupling_correctors"], "0")
            self.assertEqual(full_values["ctot_outer_acceleration"], "OFF")
            self.assertEqual(full["steps"], 1)
            self.assertEqual(half["steps"], 2)
            self.assertAlmostEqual(full["run_dt"], half["run_dt"] * 2.0)
            self.assertAlmostEqual(
                full["run_dt"] * full["steps"],
                half["run_dt"] * half["steps"],
            )
            self.assertEqual(full_values["diagnostic_rsmd_enabled"], "0")
            self.assertEqual(full_values["gp_growth_enabled"], "0")

    def test_declared_ladder_is_exact_dyadic_range(self):
        self.assertEqual(DIVISORS, (1, 2, 4, 8, 16, 32, 64, 128))
        self.assertEqual(BASE_DT / DIVISORS[-1], 0.003125 / 128)


if __name__ == "__main__":
    unittest.main()
