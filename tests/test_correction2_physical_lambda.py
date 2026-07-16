#!/usr/bin/env python3
"""Formula/contract tests for Correction-2 physical-lambda refinement."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.correction1_ji_chen_mapping import corrected_limit  # noqa: E402
from scripts.prepare_correction2_physical_lambda_matrix import (  # noqa: E402
    parse_physical_inputs,
)
import Unit_Psedobinary as unit  # noqa: E402
from scripts.analyze_correction2_physical_lambda import logged_wall_time_s  # noqa: E402


def test_strict_physical_scaling() -> None:
    inputs = parse_physical_inputs(ROOT / "physical_inputs.example.json")
    rows = []
    for lambda_nm in (0.6, 0.45, 0.3):
        current = replace(
            inputs, temperature_C=400.0, lambda_sm=lambda_nm * 1.0e-9,
            pf_dx=lambda_nm / 12.0 * 1.0e-9,
            dx=lambda_nm / 12.0 * 1.0e-9,
            L_phi_calibration_mode="one_sided_diffusion_controlled",
        )
        rows.append(corrected_limit(current, 400.0))
    for left, right in zip(rows, rows[1:]):
        lambda_ratio = left["lambda_JC_m"] / right["lambda_JC_m"]
        assert math.isclose(
            right["L_phi_diff_physical_m3_J_s"]
            / left["L_phi_diff_physical_m3_J_s"],
            lambda_ratio**2, rel_tol=2.0e-14,
        )
        assert math.isclose(
            right["time_scale_s"] / left["time_scale_s"],
            1.0 / lambda_ratio**2, rel_tol=2.0e-14,
        )
        assert math.isclose(
            right["L_phi_diff_code"] / left["L_phi_diff_code"],
            lambda_ratio, rel_tol=2.0e-14,
        )
        assert right["zeta0"] == 1.0

        # Retained but disabled GP parameter pairs must track the changed
        # common energy/time references without changing their physical values.
        current = replace(
            inputs, temperature_C=400.0,
            lambda_sm=right["lambda_JC_m"],
            pf_dx=right["lambda_JC_m"] / 12.0,
            dx=right["lambda_JC_m"] / 12.0,
            L_phi_calibration_mode="one_sided_diffusion_controlled",
        )
        converted = unit.PFParamConverter().convert(current)
        assert math.isclose(
            converted.gp_W_eta_code,
            converted.gp_W_eta_phys / right["energy_scale_J_m3"],
            rel_tol=2.0e-14,
        )
        assert math.isclose(
            converted.gp_L_eta_code,
            converted.gp_L_eta_phys * right["energy_scale_J_m3"]
            * right["time_scale_s"],
            rel_tol=2.0e-14,
        )


def test_prepared_matrix_contract() -> None:
    path = ROOT / "tmp/correction2_physical_lambda/matrix_manifest.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text())
    assert payload["lambdas_nm"] == [0.6, 0.45, 0.3]
    assert payload["lambda_over_dx"] == 12.0
    assert payload["finite_interface_correction"] == "off"
    assert payload["spectral_transport"] == "off"
    assert payload["GP_S3"] == "off"
    hashes = {case["same_outer_state_sha256"] for case in payload["cases"]}
    assert len(hashes) == 1
    for case in payload["cases"]:
        assert case["L_phi_ratio"] == 0.9
        assert case["zeta0"] == 1.0
        assert case["D_beta_for_calibration"] == 0.0
        assert case["actual_ctot_flux"] == "(1-h)*M_alpha*grad(mu)"
        assert case["finite_interface_mode"] == "off"
        assert case["elasticity"] == "off"
        assert case["GP_S3"] == "off"


def test_logged_wall_time() -> None:
    text = (
        "step=0 t_wall=2026-07-14 00:00:01\n"
        "step=10 t_wall=2026-07-14 00:00:06\n"
    )
    assert logged_wall_time_s(text) == 5.0


if __name__ == "__main__":
    test_strict_physical_scaling()
    test_prepared_matrix_contract()
    test_logged_wall_time()
    print("test_correction2_physical_lambda=PASS")
