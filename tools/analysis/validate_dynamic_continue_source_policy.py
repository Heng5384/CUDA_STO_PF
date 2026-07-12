#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analysis.dynamic_continue_source_policy import (  # noqa: E402
    NO_SAFE_POSTCRITICAL_RADIUS_AVAILABLE,
    choose_dynamic_continue_source,
)


def main() -> int:
    tests = [
        ("exact_peak_production", [1.50, 1.60, 1.70, 1.80], 1.60, False, "OK", 1.70),
        ("peak_between_points", [1.50, 1.60, 1.70, 1.80], 1.64, False, "OK", 1.80),
        ("no_safe_right_point", [1.50, 1.60], 1.60, False, NO_SAFE_POSTCRITICAL_RADIUS_AVAILABLE, None),
        ("exact_peak_debug", [1.60, 1.70], 1.60, True, "DEBUG_EXACT_PEAK", 1.60),
    ]
    lines = ["# Source Policy Validation Report", ""]
    pass_count = 0
    for name, radii, peak, debug, expected_status, expected_radius in tests:
        choice = choose_dynamic_continue_source(
            radii,
            peak,
            min_margin_nm=0.10,
            fallback_margin_nm=0.05,
            allow_exact_peak_debug=debug,
        )
        ok = choice.status == expected_status and (
            expected_radius is None
            or (choice.radius_nm is not None and abs(choice.radius_nm - expected_radius) < 1.0e-12)
        )
        pass_count += int(ok)
        lines.append(
            f"- {name}: {'PASS' if ok else 'FAIL'} "
            f"(status={choice.status}, radius={choice.radius_nm}, expected={expected_status}/{expected_radius})"
        )
    fail_count = len(tests) - pass_count
    lines += ["", f"validation_pass_count={pass_count}", f"validation_fail_count={fail_count}"]
    out = ROOT / "reports/nucleus_library_workflow/dynamic_continue_source_policy_audit/source_policy_validation_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"validation_pass_count={pass_count}")
    print(f"validation_fail_count={fail_count}")
    print(f"report={out}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
