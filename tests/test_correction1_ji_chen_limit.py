from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.correction1_ji_chen_mapping import (  # noqa: E402
    classify_ratio,
    corrected_limit,
    integrate_zeta0,
    load_inputs,
    phi_equilibrium,
)
from scripts.correction1_planar_sharp_oracle import (  # noqa: E402
    PlanarSharpOracle,
)


INPUTS = (
    ROOT / "reports/pf_ctot_production_candidate/research2_inputs"
    / "T380_physical_inputs.json"
)


def test_lambda_profile_and_threshold_widths() -> None:
    # phi(-w/2)=0.9 and phi(+w/2)=0.1 for w=atanh(0.8)*lambda.
    half = 0.5 * math.atanh(0.8)
    assert abs(float(phi_equilibrium(-half)) - 0.9) < 2.0e-15
    assert abs(float(phi_equilibrium(half)) - 0.1) < 2.0e-15


def test_strict_zeta0_converges_to_one_and_finite_window_is_distinct() -> None:
    finite = integrate_zeta0(0.5, 32001)
    strict = integrate_zeta0(6.0, 32001)
    assert abs(finite - 0.9718924561703741) < 5.0e-10
    assert abs(strict - 1.0) < 2.0e-13
    assert finite < strict


def test_corrected_limits_are_below_historical_finite_window_values() -> None:
    inputs = load_inputs(INPUTS)
    expected = {380: 4.173959248432274, 400: 5.193879264660844}
    for temperature, code in expected.items():
        row = corrected_limit(inputs, temperature)
        assert abs(row["L_phi_diff_code"] - code) < 2.0e-13
        assert row["historical_reference_ratio_to_strict_limit"] > 1.0
        assert classify_ratio(0.99) == "FINITE_POSITIVE_INTERFACE_MOBILITY"
        assert classify_ratio(row["historical_reference_ratio_to_strict_limit"]) == (
            "NONPHYSICAL_UNDER_JI_CHEN_MAPPING"
        )


def test_manifest_contains_only_strictly_below_limit_cases() -> None:
    manifest = json.loads(
        (ROOT / "examples/correction1_Lphi_below_limit_manifest.json").read_text()
    )
    assert manifest["mapping"]["zeta0"] == 1.0
    assert manifest["cases"]
    assert all(0.0 < float(case["ratio_to_L_phi_diff"]) < 1.0
               for case in manifest["cases"])
    assert all(case["finite_interface_mode"] == "off" for case in manifest["cases"])
    assert all(case["GP_S3"] == "off" for case in manifest["cases"])


def test_sharp_oracle_semidiscrete_inventory_derivative_is_roundoff() -> None:
    oracle = PlanarSharpOracle(cells=80)
    state = oracle.initial_state()
    rate = oracle.rhs(0.0, state)
    inventory_rate = rate[-1] + oracle.dy * np.sum(rate[:-1])
    assert abs(float(inventory_rate)) < 2.0e-14


def test_sharp_oracle_growth_equilibrium_and_dissolution_directions() -> None:
    oracle = PlanarSharpOracle(cells=80)
    growth = oracle.run(2.0e-5, 2.0e-6, mode="matched_growth")
    equilibrium = oracle.run(2.0e-5, 2.0e-6, mode="equilibrium")
    dissolution = oracle.run(2.0e-5, 2.0e-6, mode="dissolution")
    assert float(growth["velocity_average_nm_s"]) > 0.0
    assert abs(float(equilibrium["velocity_average_nm_s"])) < 1.0e-12
    assert float(dissolution["velocity_average_nm_s"]) < 0.0
    assert max(float(row["mass_error_rel"])
               for row in (growth, equilibrium, dissolution)) < 1.0e-10


def test_runner_does_not_enable_forbidden_physics() -> None:
    source = (ROOT / "scripts/run_correction1_below_limit_workstation.py").read_text()
    assert '"finite_interface_mode": "off"' in source
    assert '"elasticity": "off"' in source
    assert '"GP_S3": "off"' in source
    assert "0.0 < ratio < 1.0" in source


def test_selected_mode_is_strictly_below_limit_and_keeps_limitation_visible() -> None:
    selected = json.loads(
        (ROOT / "examples/correction1_selected_diffusion_limit_mode.json").read_text()
    )
    assert selected["status"] == "PASS_JI_CHEN_DIFFUSION_LIMIT_LPHI_SELECTED"
    assert selected["ratio_to_L_phi_diff"] == 0.9
    assert selected["ratio_to_L_phi_diff"] < 1.0
    assert selected["finite_interface_mode"] == "off"
    assert selected["GP_S3_enabled"] is False
    assert selected["quantitative_sharp_match"] is False
    assert selected["sharp_velocity_error_rel"] > 0.9


if __name__ == "__main__":
    tests = sorted(
        (name, value) for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"correction1_unit_tests_passed={len(tests)}")
