#!/usr/bin/env python3
"""Tests for exact fixed-Ctot mapping of a common pre-aged sharp state."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_correction2_matched_planar_state import prepare_family  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        manifest = prepare_family(
            Path(temporary), [1.0], domain_nm=9.6, dx_nm=0.15,
            lambda_nm=0.6, sharp_cells=100, matrix_x=0.05,
        )
        row = manifest["cases"][0]
        assert bool(row["finite"])
        assert float(row["sharp_mass_error_rel"]) <= 1.0e-10
        assert float(row["h_volume_error_abs_nm"]) <= 2.0e-13
        assert abs(
            float(row["pf_inventory_after_correction_xB_nm"])
            - float(row["target_total_inventory_xB_nm"])
        ) <= 2.0e-13
        assert float(row["outside_profile_Linf"]) <= 1.0e-14
        assert float(row["Ctot_min_minus_h"]) >= -1.0e-14
        assert float(row["one_minus_Ctot_min"]) >= -1.0e-14
        assert float(row["q_alpha_min"]) >= -1.0e-14
        assert float(row["one_minus_h_minus_q_min"]) >= -1.0e-14
    print("test_prepare_correction2_matched_planar_state=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
