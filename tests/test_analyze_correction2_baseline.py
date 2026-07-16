#!/usr/bin/env python3
"""Operator-isolation tests for the Correction 2 baseline audit."""

from __future__ import annotations

import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_correction2_baseline import (  # noqa: E402
    thermodynamic_drive,
    window_bounds,
)


def test_window_partition() -> None:
    assert window_bounds("early") == (0.0, 0.25)
    assert window_bounds("middle") == (0.25, 0.75)
    assert window_bounds("late") == (0.75, 1.0)
    assert window_bounds("full") == (0.0, 1.0)


def test_drive_vanishes_at_equilibrium() -> None:
    drive = thermodynamic_drive(673.15, 0.005907940326439202,
                                0.005907940326439202, 137790.24)
    assert abs(drive["reaction_affinity_J_mol"]) < 1.0e-10
    assert drive["supersaturation_ratio"] == 0.0


def test_current_Fo_is_early() -> None:
    diffusivity = 9.725359394e-18 * 1.0e18
    elapsed = 2.0821852621180848e-4
    fo = diffusivity * elapsed / 0.6**2
    assert 0.005 < fo < 0.006
    assert fo < 1.0


def main() -> int:
    test_window_partition()
    test_drive_vanishes_at_equilibrium()
    test_current_Fo_is_early()
    print("test_analyze_correction2_baseline=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
