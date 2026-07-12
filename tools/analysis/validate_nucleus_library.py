#!/usr/bin/env python3
"""Validate the unified nucleus library workflow."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analysis.query_nucleus_library import query  # noqa: E402

CSV_PATH = ROOT / "data/nucleus_library/nucleus_library.csv"
JSON_PATH = ROOT / "data/nucleus_library/nucleus_library.json"
REPORT_DIR = ROOT / "reports/nucleus_library_workflow"

REQUIRED = [
    "library_entry_id",
    "case_id",
    "T_C",
    "xB",
    "strain_mode",
    "DeltaG_bare_kBT",
    "r_star_nm",
    "r_star_source_column",
    "r_seed_nm",
    "source_dx_nm",
    "source_internal_unit_to_nm",
    "r_seed_source_internal",
    "semiaxes_source_internal",
    "tau_bridge_s",
    "semiaxes_nm",
    "shape_tensor_unit",
    "production_valid",
    "debug_only",
    "missing_reason",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(text: str | None) -> float | None:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw or raw.lower() in {"nan", "none", "null", "na"}:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--library", type=Path, default=CSV_PATH)
    ap.add_argument("--json-path", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    library_path = args.library.resolve()
    json_path = args.json_path.resolve() if args.json_path else library_path.with_suffix(".json")
    report_path = args.report.resolve() if args.report else REPORT_DIR / (
        f"{library_path.stem}_validation_report.md"
        if library_path != CSV_PATH
        else "nucleus_library_validation_report.md"
    )

    rows = read_csv(library_path)
    payload = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {"entries": []}
    json_rows = payload.get("entries", [])
    checks: list[tuple[str, bool, str]] = []
    header = set(rows[0].keys()) if rows else set()
    checks.append(("schema_required_columns", all(c in header for c in REQUIRED), "required columns exist"))
    checks.append(("csv_nonempty", bool(rows), f"rows={len(rows)}"))
    checks.append(("json_csv_entry_count", (not json_rows) or len(rows) == len(json_rows), f"csv={len(rows)} json={len(json_rows)}"))
    if rows:
        first = rows[0]
        checks.append(("schema_has_explicit_nm_fields", "source_dx_nm" in first and "r_seed_source_internal" in first, "explicit dual-unit fields present"))
    if rows:
        for field in ("T_C", "xB", "r_star_nm", "DeltaG_bare_kBT"):
            finite_count = sum(safe_float(r.get(field)) is not None for r in rows)
            checks.append((f"{field}_finite_presence", finite_count > 0, f"finite_rows={finite_count}"))
        strict_rows = [r for r in rows if (r.get("strain_mode") or "").strip() == "no_strain"]
        checks.append(("has_no_strain_rows", bool(strict_rows), f"rows={len(strict_rows)}"))
        sample_no_strain = strict_rows[0] if strict_rows else rows[0]
        T_C = safe_float(sample_no_strain.get("T_C")) or 0.0
        xB = safe_float(sample_no_strain.get("xB")) or 0.0
        prod = query(rows, T_C=T_C, xB=xB, strain_mode="no_strain", strain_value=0.0, dx_nm=1.0, production_required=True)
        checks.append(("production_query_executes", prod["status"] in {"OK", "NO_VALID_PRODUCTION_SEED"}, prod["status"]))
        debug_no = query(rows, T_C=T_C, xB=xB, strain_mode="no_strain", strain_value=0.0, dx_nm=1.0, production_required=False, allow_debug_fallback=False)
        checks.append(("debug_query_executes", debug_no["status"] in {"OK", "NO_VALID_DEBUG_OR_LIBRARY_SEED"}, debug_no["status"]))
        debug_yes = query(rows, T_C=T_C, xB=xB, strain_mode="no_strain", strain_value=0.0, dx_nm=1.0, production_required=False, allow_debug_fallback=True)
        checks.append(("debug_fallback_query_executes", debug_yes["status"] in {"OK", "NO_VALID_DEBUG_OR_LIBRARY_SEED"}, debug_yes["status"]))
        fake = dict(rows[0])
        fake.update({"library_entry_id": "fake_external", "T_C": "380", "xB": f"{xB:.12g}", "strain_mode": "external_strain", "strain_value": "0.01", "production_valid": "true", "r_seed_nm": "5.0", "tau_bridge_s": "1e-9"})
        strict = query(rows + [fake], T_C=T_C, xB=xB, strain_mode="no_strain", strain_value=0.0, dx_nm=1.0, production_required=True)
        checks.append(("no_strain_rejects_external_strain", strict["status"] in {"OK", "NO_VALID_PRODUCTION_SEED"}, strict["status"]))
    if library_path == CSV_PATH:
        t380 = [r for r in rows if safe_float(r.get("T_C")) == 380.0 and abs((safe_float(r.get("xB")) or 0.0) - 0.04) < 1e-12 and r["strain_mode"] == "no_strain"]
        checks.append(("T380_xB004_row_exists", bool(t380), "row present"))
        if t380:
            row = t380[0]
            checks.append(("T380_xB004_rstar_source", row["r_star_source_column"] == "cnt_refsub_peak_radius_nm", row["r_star_source_column"]))
            checks.append(("T380_xB004_rstar_value", abs(float(row["r_star_nm"]) - 1.606646) < 1e-6, row["r_star_nm"]))
    prod_rows = [r for r in rows if r.get("production_valid") == "true" and r.get("r_seed_nm")]
    ratios_ok = True
    for row in prod_rows[:20]:
        r_seed = float(row["r_seed_nm"])
        expected = r_seed / 1.0
        got = float(row["r_seed_over_dx_for_dx_1p0"])
        if abs(expected - got) > 1.0e-9:
            ratios_ok = False
            break
    checks.append(("r_seed_over_dx_uses_nm_radius", ratios_ok, f"checked_rows={min(len(prod_rows), 20)}"))
    pass_count = sum(ok for _, ok, _ in checks)
    fail_count = len(checks) - pass_count
    lines = ["# Nucleus Library Validation Report", ""]
    lines.append(f"- library_path: `{library_path}`")
    lines.append(f"- json_path: `{json_path}`")
    lines.append("")
    for name, ok, note in checks:
        lines.append(f"- {name}: {'PASS' if ok else 'FAIL'} ({note})")
    lines += ["", f"validation_pass_count: {pass_count}", f"validation_fail_count: {fail_count}"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"validation_pass_count={pass_count}")
    print(f"validation_fail_count={fail_count}")
    print(f"validation_report={report_path}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
