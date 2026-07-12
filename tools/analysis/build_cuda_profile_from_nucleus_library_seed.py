#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis.query_nucleus_library import query, read_library, safe_float


FAMILY_LABELS = [f"{sx:+d}{sy:+d}{sz:+d}" for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]


def parse_triplet(text: Any) -> tuple[float, float, float] | None:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    for ch in "[](),;":
        s = s.replace(ch, " ")
    parts = [p for p in s.split() if p]
    if len(parts) < 3:
        return None
    vals = []
    for p in parts[:3]:
        try:
            vals.append(float(p))
        except ValueError:
            return None
    return tuple(vals)  # type: ignore[return-value]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def build_profile_rows(
    *,
    xB_matrix: float,
    semiaxes_nm: tuple[float, float, float],
    iface_width_nm: float,
    du_nm: float,
) -> list[dict[str, object]]:
    a, b, c = semiaxes_nm
    inside_nm = max(a, b, c) + 1.0
    outside_nm = max(4.0, 1.5 * iface_width_nm)
    n = int(round((inside_nm + outside_nm) / du_nm)) + 1
    depletion = min(0.02, 0.6 * xB_matrix)
    rows: list[dict[str, object]] = []
    for family in FAMILY_LABELS:
        for i in range(n):
            u_nm = -inside_nm + i * du_nm
            phi_mean = 0.5 * (1.0 - math.tanh(u_nm / max(iface_width_nm, 1.0e-6)))
            hphi = 6.0 * phi_mean**5 - 15.0 * phi_mean**4 + 10.0 * phi_mean**3
            xB_mean = clamp(xB_matrix - depletion * hphi, 1.0e-8, max(xB_matrix, 1.0e-8))
            rows.append(
                {
                    "family": family,
                    "region": "face",
                    "u_nm": f"{u_nm:.6f}",
                    "phi_mean": f"{phi_mean:.10e}",
                    "phi_std": "0.0",
                    "xB_mean": f"{xB_mean:.10e}",
                    "xB_std": "0.0",
                    "sample_count": "1",
                }
            )
    return rows


def write_synthetic_summary(source_dir: Path, row: dict[str, str], semiaxes_nm: tuple[float, float, float]) -> Path:
    source_dir.mkdir(parents=True, exist_ok=True)
    a, b, c = semiaxes_nm
    center = parse_triplet(row.get("center_of_mass_nm")) or (20.0, 20.0, 20.0)
    bbox = parse_triplet(row.get("bbox_length_xyz_nm")) or (2.0 * a, 2.0 * b, 2.0 * c)
    orientation = parse_triplet(None)
    summary_path = source_dir / "summary.txt"
    long_axis = row.get("orientation_matrix", "")
    long_v = (1.0, 0.0, 0.0)
    mid_v = (0.0, 1.0, 0.0)
    short_v = (0.0, 0.0, 1.0)
    if long_axis:
        try:
            axes = [seg.strip() for seg in long_axis.split(";")]
            if len(axes) >= 3:
                long_v = parse_triplet(axes[0]) or long_v
                mid_v = parse_triplet(axes[1]) or mid_v
                short_v = parse_triplet(axes[2]) or short_v
        except Exception:
            pass
    summary_path.write_text(
        "\n".join(
            [
                "==================== Analysis Summary ====================",
                f"summary_file              : {summary_path}",
                "mode                      : synthetic_library_seed",
                "grid_dimensions           : (400, 400, 400)",
                "spacing_sim_units         : (1.000000, 1.000000, 1.000000)",
                "internal_unit_to_nm       : 1.0000000000e-01",
                "spacing_nm                : (0.100000, 0.100000, 0.100000)",
                f"R_avg_nm                  : {safe_float(row.get('r_seed_nm')) or sum(semiaxes_nm)/3.0:.6e}",
                f"center_of_mass_nm         : [{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]",
                f"bbox_length_xyz_nm        : [{bbox[0]:.6f}, {bbox[1]:.6f}, {bbox[2]:.6f}]",
                f"long_axis                 : [{long_v[0]:.6f}, {long_v[1]:.6f}, {long_v[2]:.6f}]",
                f"mid_axis                  : [{mid_v[0]:.6f}, {mid_v[1]:.6f}, {mid_v[2]:.6f}]",
                f"short_axis                : [{short_v[0]:.6f}, {short_v[1]:.6f}, {short_v[2]:.6f}]",
                f"L1_long_nm                : {2.0*a:.6e}",
                f"L2_mid_nm                 : {2.0*b:.6e}",
                f"L3_short_nm               : {2.0*c:.6e}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a minimal CUDA-ready scheduled profile_dir from one nucleus_library seed.")
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--T_C", type=float, required=True)
    ap.add_argument("--xB", type=float, required=True)
    ap.add_argument("--strain_mode", default="no_strain")
    ap.add_argument("--strain_value", type=float, default=0.0)
    ap.add_argument("--dx_nm", type=float, required=True)
    ap.add_argument("--output-dir", type=Path, required=True, help="profile_dir output path")
    ap.add_argument("--source-dir", type=Path, default=None, help="optional source_dyn_dir output path")
    ap.add_argument("--iface-width-nm", type=float, default=0.6)
    ap.add_argument("--du-nm", type=float, default=0.05)
    args = ap.parse_args()

    rows = read_library(args.library)
    result = query(
        rows,
        T_C=args.T_C,
        xB=args.xB,
        strain_mode=args.strain_mode,
        strain_value=args.strain_value,
        dx_nm=args.dx_nm,
        production_required=True,
    )
    entry = result["entry"]
    if not entry:
        raise SystemExit(f"[fatal] no production-valid library seed found: {result['status']} {result['reason']}")

    output_dir = args.output_dir.resolve()
    source_dir = args.source_dir.resolve() if args.source_dir else (output_dir / "source_dyn_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    semiaxes_nm = parse_triplet(entry.get("semiaxes_nm")) or (
        safe_float(entry.get("r_seed_nm"), 1.0) or 1.0,
        safe_float(entry.get("r_seed_nm"), 1.0) or 1.0,
        safe_float(entry.get("r_seed_nm"), 1.0) or 1.0,
    )
    r_seed_nm = safe_float(entry.get("r_seed_nm"), sum(semiaxes_nm) / 3.0) or (sum(semiaxes_nm) / 3.0)
    xB_matrix = safe_float(entry.get("xB"), args.xB) or args.xB
    profile_rows = build_profile_rows(
        xB_matrix=xB_matrix,
        semiaxes_nm=semiaxes_nm,
        iface_width_nm=args.iface_width_nm,
        du_nm=args.du_nm,
    )

    profile_csv = output_dir / "faceted_family_profiles.csv"
    with profile_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["family", "region", "u_nm", "phi_mean", "phi_std", "xB_mean", "xB_std", "sample_count"],
        )
        writer.writeheader()
        writer.writerows(profile_rows)

    summary_path = write_synthetic_summary(source_dir, entry, semiaxes_nm)

    metadata = {
        "profile_type": "ellipsoid_from_library_seed",
        "library_entry_id": entry.get("library_entry_id"),
        "case_id": entry.get("case_id"),
        "T_C": args.T_C,
        "xB": args.xB,
        "strain_mode": args.strain_mode,
        "strain_value": args.strain_value,
        "r_seed_nm": r_seed_nm,
        "dx_nm": args.dx_nm,
        "r_grid": r_seed_nm / args.dx_nm,
        "semiaxes_nm": list(semiaxes_nm),
        "semiaxes_grid": [v / args.dx_nm for v in semiaxes_nm],
        "orientation_matrix": entry.get("orientation_matrix", ""),
        "shape_tensor": entry.get("shape_tensor", ""),
        "shape_tensor_unit": entry.get("shape_tensor_unit", "nm2"),
        "tau_bridge_s": safe_float(entry.get("tau_bridge_s")),
        "tau_bridge_code_time": safe_float(entry.get("tau_bridge_code_time")),
        "dt_code": safe_float(entry.get("dt_code"), safe_float(entry.get("dynamic_continue_dt_code"))),
        "t_real_unit_s": safe_float(entry.get("t_real_unit_s")),
        "mass_seed_B_equiv": safe_float(entry.get("mass_seed_B_equiv")),
        "profile_dir": str(output_dir),
        "source_dyn_dir": str(source_dir),
        "summary_txt": str(summary_path),
        "source_internal_units_used": False,
        "notes": "Minimal runtime-compatible synthetic faceted profile generated from nucleus_library seed descriptor.",
    }
    metadata_path = output_dir / "seed_profile_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    readme = output_dir / "README_profile_dir.md"
    readme.write_text(
        "\n".join(
            [
                "# CUDA Profile Dir From nucleus_library Seed",
                "",
                f"- library_entry_id: `{entry.get('library_entry_id')}`",
                f"- r_seed_nm: `{r_seed_nm}`",
                f"- dx_nm: `{args.dx_nm}`",
                f"- r_grid: `{r_seed_nm / args.dx_nm}`",
                "- representation: synthetic 8-family face profile with identical ellipsoidal interface curves",
                "- limitation: this is a minimal runtime-compatible profile, not a VTK-reconstructed faceted family profile",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"status=OK")
    print(f"library_entry_id={entry.get('library_entry_id')}")
    print(f"profile_csv={profile_csv}")
    print(f"metadata_json={metadata_path}")
    print(f"source_summary={summary_path}")
    print(f"r_seed_nm={r_seed_nm}")
    print(f"r_grid={r_seed_nm / args.dx_nm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
