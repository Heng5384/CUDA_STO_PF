#!/usr/bin/env python3
"""High-time-resolution 6--8 h morphology audit from available VTK pairs."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

import build_pf_vtk_particle_trajectory_v1 as tracker


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", nargs="+", required=True, help="age_h=phi.vtk=xB.vtk")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=246)
    parser.add_argument("--dx-nm", type=float, default=1.0)
    parser.add_argument("--dt-code", type=float, default=0.02)
    parser.add_argument("--code-time-to-physical-s", type=float, default=49.54630476715921)
    parser.add_argument("--particle-h-threshold", type=float, default=1.0e-4)
    args = parser.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = []
    for item in args.snapshots:
        fields = item.split("=", 2)
        if len(fields) != 3:
            raise SystemExit("snapshot must be age_h=phi.vtk=xB.vtk")
        specs.append((float(fields[0]), Path(fields[1]), Path(fields[2])))
    specs.sort(); rows_out = []; previous = None; next_id = 1
    for age, phi_path, xb_path in specs:
        phi = tracker.read_vtk_ascii(phi_path, args.grid); xb = tracker.read_vtk_ascii(xb_path, args.grid); h = tracker.h_of_phi(phi)
        rows, labels, next_id, bad = tracker.identify(phi, args.dx_nm, previous, next_id, args.particle_h_threshold)
        box = (args.grid * args.dx_nm) ** 3; mask = phi > 0.5
        faces = sum(int(np.count_nonzero(mask != np.roll(mask, -1, axis=axis))) for axis in range(3))
        radii = np.asarray([float(r["equivalent_radius_nm"]) for r in rows], dtype=np.float64)
        rows_out.append({"age_h": age, "step_estimate": (age - 6.0) * 3600.0 / (args.dt_code * args.code_time_to_physical_s), "particle_count": len(rows), "beta_volume_fraction": float(np.mean(h)), "h_volume_fraction_from_components": float(sum(float(r["h_volume_nm3"]) for r in rows) / box), "h_volume_closure_abs": abs(float(np.mean(h)) - sum(float(r["h_volume_nm3"]) for r in rows) / box), "mean_radius_nm": float(np.mean(radii)) if radii.size else float("nan"), "radius_p10_nm": float(np.percentile(radii, 10)) if radii.size else float("nan"), "radius_p90_nm": float(np.percentile(radii, 90)) if radii.size else float("nan"), "Sv_nm^-1": float(4.0 * math.pi * np.sum(radii ** 2) / box), "M6_nm^3": float(np.sum(radii ** 6) / box), "interface_area_density_nm^-1": faces * args.dx_nm ** 2 / box, "matrix_xB_h_lt_0p005": float(np.mean(xb[h < 0.005])), "unexpected_merge": int(bad), "source_phi": str(phi_path), "source_xB": str(xb_path)})
        previous = labels
    write_csv(args.out_dir / "morphology_6_8h_time_series.csv", rows_out)
    status = "PASS_HIGH_TIME_MORPHOLOGY_OBSERVATION_V1" if not any(int(row["unexpected_merge"]) for row in rows_out) else "BLOCKED_HIGH_TIME_MERGE_SPLIT"
    (args.out_dir / "status.txt").write_text(status + "\n", encoding="utf-8")
    (args.out_dir / "morphology_report.md").write_text("# High-time-resolution 6--8 h morphology audit\n\n`%s`\n\nSnapshots are limited to the VTK files supplied by the completed run; no new 48 h integration was performed. The table preserves every available time point and reports h-volume closure, radius percentiles, Sv, M6, interface-area density and matrix xB.\n" % status, encoding="utf-8")
    print(status)


if __name__ == "__main__":
    main()
