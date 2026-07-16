#!/usr/bin/env python3
"""Unit tests for the Correction-2 diagnostic resistance split."""

from __future__ import annotations

import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_correction2_resistance import finite_phase_resistance  # noqa: E402


def test_phase_resistance_ratio_identity() -> None:
    inputs = ROOT / "physical_inputs.example.json"
    for lambda_nm in (0.6, 0.45, 0.3):
        for ratio in (0.9, 0.95, 0.98, 0.99):
            phase, diffusive, _ = finite_phase_resistance(
                lambda_nm, ratio, inputs
            )
            assert phase > 0.0
            assert diffusive > 0.0
            assert math.isclose(
                phase / diffusive, 1.0 / ratio - 1.0,
                rel_tol=5.0e-14, abs_tol=1.0e-15,
            )


if __name__ == "__main__":
    test_phase_resistance_ratio_identity()
    print("test_correction2_resistance=PASS")
