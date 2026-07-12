#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis.query_nucleus_library import read_library, safe_float, truthy


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pre-generate runtime profile cache entries for production-valid nucleus library seeds."
    )
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--dx_nm", type=float, required=True)
    ap.add_argument("--output-cache", type=Path, required=True)
    ap.add_argument("--iface-width-nm", type=float, default=0.6)
    ap.add_argument("--du-nm", type=float, default=0.05)
    args = ap.parse_args()

    rows = read_library(args.library)
    output_cache = args.output_cache.resolve()
    output_cache.mkdir(parents=True, exist_ok=True)
    builder = REPO_ROOT / "tools" / "analysis" / "build_cuda_profile_from_nucleus_library_seed.py"
    status_rows: list[dict[str, object]] = []

    for row in rows:
        entry_id = (row.get("library_entry_id") or "").strip()
        if not entry_id:
            continue
        r_seed_nm = safe_float(row.get("r_seed_nm"))
        production_valid = truthy(row.get("production_valid"))
        insertable = (r_seed_nm is not None) and (r_seed_nm / args.dx_nm >= 4.0)
        entry_dir = output_cache / entry_id
        source_dir = entry_dir / "source_dyn_dir"
        should_build = production_valid and insertable
        status = "SKIPPED"
        reason = ""
        if should_build:
            cmd = [
                sys.executable,
                str(builder),
                "--library",
                str(args.library.resolve()),
                "--T_C",
                str(row.get("T_C")),
                "--xB",
                str(row.get("xB")),
                "--strain_mode",
                str(row.get("strain_mode") or "no_strain"),
                "--strain_value",
                str(safe_float(row.get("strain_value"), 0.0) or 0.0),
                "--dx_nm",
                str(args.dx_nm),
                "--output-dir",
                str(entry_dir),
                "--source-dir",
                str(source_dir),
                "--iface-width-nm",
                str(args.iface_width_nm),
                "--du-nm",
                str(args.du_nm),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                status = "BUILT"
            else:
                status = "FAILED"
                reason = (result.stderr or result.stdout).strip().splitlines()[-1] if (result.stderr or result.stdout) else "builder_failed"
        else:
            if not production_valid:
                reason = row.get("missing_reason") or "not_production_valid"
            elif not insertable:
                reason = "r_seed_over_dx_too_small"
        status_rows.append(
            {
                "library_entry_id": entry_id,
                "T_C": row.get("T_C"),
                "xB": row.get("xB"),
                "strain_mode": row.get("strain_mode"),
                "dx_nm": args.dx_nm,
                "production_valid": production_valid,
                "r_seed_nm": r_seed_nm,
                "r_seed_over_dx": (r_seed_nm / args.dx_nm) if r_seed_nm else None,
                "cache_dir": str(entry_dir),
                "status": status,
                "reason": reason,
            }
        )

    manifest = output_cache / "cache_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "library_entry_id",
                "T_C",
                "xB",
                "strain_mode",
                "dx_nm",
                "production_valid",
                "r_seed_nm",
                "r_seed_over_dx",
                "cache_dir",
                "status",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(status_rows)

    built = sum(1 for row in status_rows if row["status"] == "BUILT")
    skipped = sum(1 for row in status_rows if row["status"] == "SKIPPED")
    failed = sum(1 for row in status_rows if row["status"] == "FAILED")
    print(f"runtime_profile_cache_built={built}")
    print(f"runtime_profile_cache_skipped={skipped}")
    print(f"runtime_profile_cache_failed={failed}")
    print(f"cache_manifest={manifest}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
