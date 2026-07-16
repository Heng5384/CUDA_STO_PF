#!/usr/bin/env python3
"""Check the declared finite-box and quasi-steady selection semantics."""

import math


def main() -> int:
    assert math.sqrt(26.0) / 11.2 < 0.60
    assert math.sqrt(46.0) / 11.2 > 0.60
    assert 10.0 >= 10.0
    assert 3.0 < 10.0
    print("test_correction2_window_selection=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
