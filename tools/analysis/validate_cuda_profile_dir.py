#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


REQUIRED_COLUMNS = {"family", "region", "u_nm", "phi_mean", "xB_mean"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate minimal CUDA scheduled profile_dir.")
    ap.add_argument("--profile-dir", type=Path, required=True)
    ap.add_argument("--dx_nm", type=float, required=True)
    ap.add_argument("--r-seed-nm", type=float, required=True)
    args = ap.parse_args()

    profile_dir = args.profile_dir.resolve()
    profile_csv = profile_dir / "faceted_family_profiles.csv"
    meta_json = profile_dir / "seed_profile_metadata.json"
    if not profile_csv.exists():
        raise SystemExit(f"[fatal] missing {profile_csv}")
    if not meta_json.exists():
        raise SystemExit(f"[fatal] missing {meta_json}")

    rows = list(csv.DictReader(profile_csv.open(newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit("[fatal] faceted_family_profiles.csv is empty")
    cols = set(rows[0].keys())
    missing = sorted(REQUIRED_COLUMNS - cols)
    if missing:
        raise SystemExit(f"[fatal] missing required columns: {missing}")
    for row in rows:
        for key in ("u_nm", "phi_mean", "xB_mean"):
            try:
                val = float(row[key])
            except Exception as exc:
                raise SystemExit(f"[fatal] non-numeric {key}: {exc}") from exc
            if not math.isfinite(val):
                raise SystemExit(f"[fatal] non-finite {key}")

    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    r_grid = args.r_seed_nm / args.dx_nm
    if abs(float(meta["r_grid"]) - r_grid) > 1.0e-9:
        raise SystemExit("[fatal] metadata r_grid mismatch")
    if r_grid < 4.0:
        raise SystemExit("[fatal] r_grid < 4; not insertable")
    if "tau_bridge_s" not in meta:
        raise SystemExit("[fatal] tau_bridge_s missing in metadata")
    if "mass_seed_B_equiv" not in meta:
        raise SystemExit("[fatal] mass_seed_B_equiv missing in metadata")

    print("profile_dir_valid=true")
    print(f"r_seed_nm={args.r_seed_nm}")
    print(f"dx_nm={args.dx_nm}")
    print(f"r_grid={r_grid}")
    print(f"profile_csv={profile_csv}")
    print(f"metadata_json={meta_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
