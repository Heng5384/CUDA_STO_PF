#!/usr/bin/env python3
"""Source-contract checks for the matched runtime case preparer."""

from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/prepare_correction2_runtime_case.py").read_text()
    required = (
        '"composition_evolution_mode": "ctot_mimetic_be"',
        '"ctot_finite_interface_antitrapping_enabled": 0',
        '"elastic_enabled": 0',
        '"D_beta": 0.0',
        'if not (0.0 < args.lphi_ratio < 1.0)',
    )
    for token in required:
        assert token in text, token
    print("test_prepare_correction2_runtime_case=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
