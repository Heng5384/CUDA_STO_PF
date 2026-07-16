#!/usr/bin/env python3
"""Numerical sanity tests for the Correction-2 linearity gate."""

import numpy as np


def main() -> int:
    amplitude = np.array([0.25, 0.5, 1.0])
    velocity = 0.07 * amplitude
    slope, intercept = np.polyfit(amplitude, velocity, 1)
    assert abs(slope - 0.07) < 1.0e-14
    assert abs(intercept) < 1.0e-14
    assert abs(velocity[2] / velocity[1] - 2.0) < 1.0e-14
    assert abs(velocity[1] / velocity[0] - 2.0) < 1.0e-14
    print("test_correction2_linearity=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
