#!/usr/bin/env python3
"""Contract tests for the Correction-2 workstation matrix."""

from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    prep = (root / "scripts/prepare_correction2_small_driving_matrix.py").read_text()
    run = (root / "scripts/run_correction2_matrix_workstation.py").read_text()
    long_time = (root / "scripts/prepare_correction2_long_time_matrix.py").read_text()
    for token in ("(1.0, 0.5, 0.25)", "(0.90, 0.95, 0.98, 0.99)"):
        assert token in prep
    for token in ("workstation-tail", "workstation GPU is occupied",
                  "source hashes do not match", "CTOT_COUPLED_STEP_RETRY"):
        assert token in run
    assert "cluster" not in run.lower()
    assert "(1.0, 3.0, 10.0, 30.0)" in long_time
    assert "[4.0, 6.0, 8.0]" in long_time
    assert "5.0e-4" in long_time
    print("test_correction2_matrix_contract=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
