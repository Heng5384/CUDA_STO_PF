#!/usr/bin/env python3
"""Query the unified nucleus library."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def safe_float(v: Any, default: float | None = None) -> float | None:
    if v is None:
        return default
    text = str(v).strip()
    if not text or text.lower() in {"nan", "none", "null", "na"}:
        return default
    try:
        out = float(text)
    except ValueError:
        return default
    return out if math.isfinite(out) else default


def read_library(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truthy(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def query(
    rows: list[dict[str, str]],
    *,
    T_C: float,
    xB: float,
    strain_mode: str,
    strain_value: float,
    dx_nm: float,
    production_required: bool,
    T_tol_C: float = 5.0,
    xB_tol: float = 0.005,
    allow_debug_fallback: bool = False,
) -> dict[str, Any]:
    candidates: list[tuple[tuple[float, float, float, float], dict[str, str]]] = []
    rejections: list[str] = []
    for row in rows:
        row_T = safe_float(row.get("T_C"))
        row_xB = safe_float(row.get("xB"))
        row_strain = safe_float(row.get("strain_value"), 0.0) or 0.0
        row_mode = (row.get("strain_mode") or "").strip()
        if row_mode != strain_mode:
            rejections.append(f"{row.get('library_entry_id')}:STRAIN_MODE_MISMATCH")
            continue
        if abs(row_strain - strain_value) > 1.0e-12:
            rejections.append(f"{row.get('library_entry_id')}:STRAIN_VALUE_MISMATCH")
            continue
        if row_T is None or abs(row_T - T_C) > T_tol_C:
            rejections.append(f"{row.get('library_entry_id')}:T_MISMATCH")
            continue
        if row_xB is None or abs(row_xB - xB) > xB_tol:
            rejections.append(f"{row.get('library_entry_id')}:XB_MISMATCH")
            continue
        if production_required and not truthy(row.get("production_valid")):
            rejections.append(f"{row.get('library_entry_id')}:NOT_PRODUCTION_VALID:{row.get('missing_reason')}")
            continue
        if production_required:
            r_seed = safe_float(row.get("r_seed_nm"))
            if r_seed is None or r_seed / dx_nm < 4.0:
                rejections.append(f"{row.get('library_entry_id')}:RSEED_OVER_DX_TOO_SMALL")
                continue
        if not production_required and truthy(row.get("debug_only")) and not allow_debug_fallback:
            rejections.append(f"{row.get('library_entry_id')}:DEBUG_FALLBACK_NOT_ALLOWED")
            continue
        priority = safe_float(row.get("source_priority"), 999.0) or 999.0
        candidates.append(((abs(row_T - T_C), abs(row_xB - xB), abs(row_strain - strain_value), priority), row))
    if not candidates:
        reason = "NO_VALID_PRODUCTION_SEED" if production_required else "NO_VALID_DEBUG_OR_LIBRARY_SEED"
        return {"status": reason, "reason": ";".join(rejections[:8]), "entry": None}
    candidates.sort(key=lambda item: item[0])
    return {"status": "OK", "reason": "matched_strict_tolerances", "entry": candidates[0][1]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--T_C", type=float, required=True)
    ap.add_argument("--xB", type=float, required=True)
    ap.add_argument("--strain_mode", default="no_strain")
    ap.add_argument("--strain_value", type=float, default=0.0)
    ap.add_argument("--dx_nm", type=float, default=1.0)
    ap.add_argument("--production_required", type=int, default=1)
    ap.add_argument("--allow_debug_fallback", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    result = query(
        read_library(args.library),
        T_C=args.T_C,
        xB=args.xB,
        strain_mode=args.strain_mode,
        strain_value=args.strain_value,
        dx_nm=args.dx_nm,
        production_required=bool(args.production_required),
        allow_debug_fallback=bool(args.allow_debug_fallback),
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"status={result['status']}")
        print(f"reason={result['reason']}")
        if result["entry"]:
            print(f"library_entry_id={result['entry'].get('library_entry_id')}")
            print(f"case_id={result['entry'].get('case_id')}")
            print(f"r_star_nm={result['entry'].get('r_star_nm')}")
            print(f"r_seed_nm={result['entry'].get('r_seed_nm')}")
            print(f"production_valid={result['entry'].get('production_valid')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
