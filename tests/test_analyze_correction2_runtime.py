#!/usr/bin/env python3
"""Source-contract checks for Correction-2 matched runtime analysis."""

from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/analyze_correction2_runtime.py").read_text()
    for token in (
        "PF_h_displacement_nm", "sharp_velocity_nm_s",
        "stefan_storage_residual_rel", "mass_error_rel_recomputed",
        '"clip_count=0"', '"projection_mass=0"',
        "oracle.matrix_x = matrix_x",
    ):
        assert token in text, token
    mobility = (
        root / "scripts/analyze_research2_fast_interface_plateau.py"
    ).read_text()
    for token in (
        'params.get("ctot_matrix_support_eps", "1e-10")',
        'mobility_mode == "INTERFACE_BAND_BOOST_V1"',
        "4.0 * hp * (1.0 - hp)",
    ):
        assert token in mobility, token
    print("test_analyze_correction2_runtime=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
