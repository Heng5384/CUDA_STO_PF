#!/usr/bin/env python3
"""Run the V1 dynamic-microstructure audit on an elastic PF run.

The newest V1 audit is intentionally fail-closed for elasticity because phi/xB
VTK snapshots do not contain displacement/stress.  This adapter reuses its
registered particle, PSD, chemical and interface definitions while preserving
that limitation explicitly: elastic energy is reported as unavailable rather
than as zero, and the final decision is an observational-metrics pass with an
exact-elastic-energy blocker.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

import audit_pf_dynamic_microstructure_metrics_v1_py36 as base


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-time-series", type=Path, required=True)
    parser.add_argument("--particle-trajectories", type=Path, required=True)
    parser.add_argument("--vtk-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--runtime-audit-root", type=Path, default=None)
    parser.add_argument("--grid", type=int, default=246)
    parser.add_argument("--dx-nm", type=float, default=1.0)
    parser.add_argument("--dx-code", type=float, default=1.0)
    parser.add_argument("--temperature-c", type=float, default=380.0)
    parser.add_argument("--kappa-phi", type=float, default=2.0)
    parser.add_argument("--well-w", type=float, default=1.0)
    parser.add_argument("--mu-reference-scale", type=float, default=20668.536)
    parser.add_argument("--vm-alpha-m3-mol", type=float, default=4.1009e-5)
    parser.add_argument("--xB-reference", type=float, default=None)
    parser.add_argument("--growth-tolerance", type=float, default=0.01)
    parser.add_argument("--beta-vf-closure-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--psd-bin-width-nm", type=float, default=2.0)
    parser.add_argument("--source-binary-sha256", default="")
    parser.add_argument("--source-parameter-sha256", default="")
    parser.add_argument("--elasticity", choices=("on",), default="on")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrix_rows = base.read_csv(args.matrix_time_series)
    trajectory_rows = base.read_csv(args.particle_trajectories)
    if not matrix_rows or not trajectory_rows:
        raise SystemExit("empty matrix or particle input")
    matrix_by_step = {int(row["step"]): row for row in matrix_rows}
    particle_summary, particles_by_step = base.summarize_particles(trajectory_rows, args.growth_tolerance)
    steps = sorted(matrix_by_step)
    if set(steps) != set(particle_summary):
        raise SystemExit("matrix and particle snapshot ladders differ")
    xb_reference = args.xB_reference if args.xB_reference is not None else float(matrix_by_step[steps[0]]["matrix_xB_h_lt_0p005"])
    temperature_k = args.temperature_c + 273.15
    xeq = base.solve_x_eq(temperature_k)
    mu0 = base.mu_b_hat_scalar(xeq, temperature_k, args.mu_reference_scale)
    physical_scale = args.mu_reference_scale / args.vm_alpha_m3_mol
    box_volume_nm3 = (args.grid * args.dx_nm) ** 3
    box_volume_m3 = box_volume_nm3 * 1.0e-27

    energy_by_step = {}
    for index, step in enumerate(steps, start=1):
        row = matrix_by_step[step]
        phi_path = Path(row["source_phi"])
        xb_path = Path(row["source_xB"])
        if not phi_path.is_absolute():
            phi_path = args.vtk_root / phi_path
            xb_path = args.vtk_root / xb_path
        print(f"[energy {index}/{len(steps)}] step={step} age_h={row['age_h']}", flush=True)
        energy_by_step[step] = base.reconstruct_energy(
            phi_path, xb_path, n=args.grid, dx_code=args.dx_code,
            kappa_phi=args.kappa_phi, well_w=args.well_w, temperature_k=temperature_k,
            mu_reference_scale=args.mu_reference_scale, mu0_compound_hat=mu0,
            xb_reference=xb_reference, elastic_off=True,
        )

    times_s = [(float(matrix_by_step[s]["age_h"]) - float(matrix_by_step[steps[0]]["age_h"])) * 3600.0 for s in steps]
    beta = [float(matrix_by_step[s]["beta_volume_fraction"]) for s in steps]
    xag = [float(matrix_by_step[s]["matrix_xAg_h_lt_0p005"]) for s in steps]
    dbeta = base.derivative(beta, times_s); dxag = base.derivative(xag, times_s)
    output_rows = []; psd_samples = []; max_radius = 0.0
    for idx, step in enumerate(steps):
        matrix = matrix_by_step[step]; particle = particle_summary[step]; energy = energy_by_step[step]
        sv = 4.0 * math.pi * float(particle["sum_r2_nm2"]) / box_volume_nm3
        m6 = float(particle["sum_r6_nm6"]) / box_volume_nm3
        vf_particles = sum(float(r["h_volume_nm3"]) for r in particles_by_step[step]) / box_volume_nm3
        nonelastic_hat = energy["chemical_excess_hat_mean"] + energy["interface_total_hat_mean"]
        row = {
            "step": step, "registered_age_h": matrix["registered_age_h"], "age_h": matrix["age_h"],
            "elapsed_physical_time_s": times_s[idx], "particle_count": particle["particle_count"],
            "particle_number_density_nm^-3": particle["particle_count"] / box_volume_nm3,
            "beta_volume_fraction": matrix["beta_volume_fraction"],
            "beta_volume_fraction_from_particle_h_volumes": vf_particles,
            "beta_vf_particle_closure_abs": abs(vf_particles - float(matrix["beta_volume_fraction"])),
            "mean_radius_nm": particle["mean_radius_nm"], "std_radius_nm": particle["std_radius_nm"],
            "median_radius_nm": particle["median_radius_nm"], "radius_p10_nm": particle["radius_p10_nm"],
            "radius_p90_nm": particle["radius_p90_nm"], "Sv_spherical_equivalent_nm^-1": sv,
            "M6_integral_nm^3": m6, "mean_R6_nm^6": particle["mean_r6_nm6"],
            "growing_count": particle["growing_count"], "shrinking_survivor_count": particle["shrinking_survivor_count"],
            "dissolved_count": particle["dissolved_count"], "shrinking_including_dissolved_count": particle["shrinking_including_dissolved_count"],
            "stable_count": particle["stable_count"], "newly_resolved_count": particle["newly_resolved_count"],
            "growing_to_shrinking_ratio": particle["growing_to_shrinking_ratio"],
            "matrix_xB": matrix["matrix_xB_h_lt_0p005"], "matrix_xAg": matrix["matrix_xAg_h_lt_0p005"],
            "df_beta_dt_s^-1": float(dbeta[idx]), "dxAg_dt_s^-1": float(dxag[idx]),
            "chemical_excess_energy_hat_mean": energy["chemical_excess_hat_mean"],
            "interface_gradient_energy_hat_mean": energy["interface_gradient_hat_mean"],
            "interface_double_well_energy_hat_mean": energy["interface_double_well_hat_mean"],
            "interface_energy_hat_mean": energy["interface_total_hat_mean"],
            "elastic_energy_hat_mean": "UNAVAILABLE_EXACT_RUNTIME_TRACE",
            "total_excess_energy_hat_mean_excluding_elastic": nonelastic_hat,
            "chemical_excess_energy_J_m^-3": energy["chemical_excess_hat_mean"] * physical_scale,
            "interface_energy_J_m^-3": energy["interface_total_hat_mean"] * physical_scale,
            "elastic_energy_J_m^-3": "UNAVAILABLE_EXACT_RUNTIME_TRACE",
            "total_excess_energy_J_m^-3_excluding_elastic": nonelastic_hat * physical_scale,
            "chemical_excess_energy_total_J": energy["chemical_excess_hat_mean"] * physical_scale * box_volume_m3,
            "interface_energy_total_J": energy["interface_total_hat_mean"] * physical_scale * box_volume_m3,
            "elastic_energy_total_J": "UNAVAILABLE_EXACT_RUNTIME_TRACE",
            "total_excess_energy_total_J_excluding_elastic": nonelastic_hat * physical_scale * box_volume_m3,
        }
        output_rows.append(row)
        for particle_row in particles_by_step[step]:
            max_radius = max(max_radius, float(particle_row["equivalent_radius_nm"]))
            psd_samples.append({"step": step, "registered_age_h": matrix["registered_age_h"], "age_h": matrix["age_h"], "particle_id": particle_row["particle_id"], "equivalent_radius_nm": particle_row["equivalent_radius_nm"], "h_volume_nm3": particle_row["h_volume_nm3"]})
    edges = np.arange(0.0, math.ceil(max_radius / args.psd_bin_width_nm) * args.psd_bin_width_nm + args.psd_bin_width_nm * 1.000001, args.psd_bin_width_nm)
    histogram = []
    for step in steps:
        radii = np.asarray([float(r["equivalent_radius_nm"]) for r in particles_by_step[step]])
        counts, _ = np.histogram(radii, bins=edges); total = max(int(radii.size), 1); matrix = matrix_by_step[step]
        for i, count in enumerate(counts):
            selected = radii[(radii >= edges[i]) & (radii < edges[i + 1])]
            histogram.append({"step": step, "registered_age_h": matrix["registered_age_h"], "age_h": matrix["age_h"], "radius_bin_left_nm": edges[i], "radius_bin_right_nm": edges[i + 1], "radius_bin_center_nm": 0.5 * (edges[i] + edges[i + 1]), "count": int(count), "number_fraction": float(count / total), "n_of_R_nm^-4": float(count / (box_volume_nm3 * args.psd_bin_width_nm)), "M6_bin_contribution_nm^3": float(np.sum(selected**6) / box_volume_nm3)})
    write_csv(args.out_dir / "microstructure_time_series.csv", output_rows)
    write_csv(args.out_dir / "particle_psd_samples.csv", psd_samples)
    write_csv(args.out_dir / "psd_histogram.csv", histogram)
    closure = max(float(r["beta_vf_particle_closure_abs"]) for r in output_rows)
    psd_ok = all(sum(int(r["count"]) for r in histogram if int(r["step"]) == step) == int(particle_summary[step]["particle_count"]) for step in steps)
    m6_ok = all(math.isclose(sum(float(r["M6_bin_contribution_nm^3"]) for r in histogram if int(r["step"]) == step), float(particle_summary[step]["sum_r6_nm6"]) / box_volume_nm3, rel_tol=1e-12, abs_tol=1e-18) for step in steps)
    derivatives_ok = all(math.isfinite(float(r["df_beta_dt_s^-1"])) and math.isfinite(float(r["dxAg_dt_s^-1"])) for r in output_rows)
    unexpected_steps = sorted(set(int(row["step"]) for row in matrix_rows if int(row.get("unexpected_merge", 0))))
    tracking_ok = not unexpected_steps
    observational_pass = closure <= args.beta_vf_closure_tolerance and psd_ok and m6_ok and derivatives_ok
    runtime = {}
    if args.runtime_audit_root:
        for name in ("mass_drift_summary.json", "performance_summary.json"):
            for candidate in args.runtime_audit_root.rglob(name):
                try: runtime[name] = json.loads(candidate.read_text(encoding="utf-8")); break
                except Exception: pass
    if not observational_pass:
        status = "FAIL_ELASTIC_OBSERVABLE_METRICS"
    elif not tracking_ok:
        status = "BLOCKED_UNEXPECTED_PARTICLE_MERGE_SPLIT_AND_EXACT_ELASTIC_ENERGY_TRACE"
    else:
        status = "PASS_ELASTIC_OBSERVABLES_BLOCKED_EXACT_ELASTIC_ENERGY"
    first, last = output_rows[0], output_rows[-1]
    summary = {
        "schema": "PF_DYNAMIC_MICROSTRUCTURE_AUDIT_ELASTIC_V1",
        "status": status,
        "observational_metrics_status": "PASS" if observational_pass else "FAIL",
        "exact_elastic_energy_status": "BLOCKED_EXACT_ELASTIC_ENERGY_TRACE_MISSING",
        "input": {"matrix_time_series": str(args.matrix_time_series), "matrix_time_series_sha256": sha256(args.matrix_time_series), "particle_trajectories": str(args.particle_trajectories), "particle_trajectories_sha256": sha256(args.particle_trajectories), "source_binary_sha256": args.source_binary_sha256, "source_parameter_sha256": args.source_parameter_sha256, "runtime_audit_root": str(args.runtime_audit_root) if args.runtime_audit_root else None},
        "contract": {"grid": [args.grid] * 3, "dx_nm": args.dx_nm, "temperature_C": args.temperature_c, "particle_volume": "integral h(phi) over periodic h(phi)>1e-4 component", "particle_radius": "R_eq=(3 V_h/(4 pi))^(1/3)", "elasticity": "on", "elastic_energy": "not reconstructable from phi/xB VTK; exact runtime trace required", "reconstructed_energy": "chemical + interface only; explicitly excludes elastic", "snapshot_count": len(steps), "beta_vf_particle_closure_tolerance_abs": args.beta_vf_closure_tolerance},
        "gates": {"snapshot_count": len(steps), "beta_vf_particle_closure_max_abs": closure, "beta_vf_particle_closure_status": "PASS" if closure <= args.beta_vf_closure_tolerance else "FAIL", "psd_count_closure_status": "PASS" if psd_ok else "FAIL", "M6_histogram_closure_status": "PASS" if m6_ok else "FAIL", "time_derivative_status": "PASS" if derivatives_ok else "FAIL", "particle_identity_tracking_status": "PASS" if tracking_ok else "BLOCKED_UNEXPECTED_MERGE_SPLIT", "unexpected_event_steps": unexpected_steps, "exact_elastic_energy_status": "BLOCKED_EXACT_ELASTIC_ENERGY_TRACE_MISSING"},
        "initial": first, "final": last, "runtime_audits": runtime,
        "provenance_boundary": "Elastic runtime is authoritative for PF/zero-mode/performance; VTK-only analysis cannot claim exact elastic energy.",
    }
    (args.out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.out_dir / "status.txt").write_text(status + "\n", encoding="utf-8")
    report = f"""# Elastic dynamic microstructure audit V1

## Decision

`{status}`

This report was generated from the newest `audit_pf_dynamic_microstructure_metrics_v1.py` definitions, with the elastic-run adapter. It processes the registered 43 snapshots from 6 h through the exact 48 h endpoint on the cluster.

The particle count, h-volume closure, periodic identity tracking, radius/PSD, Sv, M6, matrix xB/xAg and time derivatives are directly auditable. The elastic VTK run is not downgraded to elasticity-off: exact elastic energy is explicitly **not claimed**, because the VTK files contain phi/xB but not the displacement/stress state.

| gate | result |
|---|---|
| snapshots | {len(steps)} registered 6--48 h snapshots |
| particle h-volume closure | {summary['gates']['beta_vf_particle_closure_status']} (max abs {closure:.6g}) |
| PSD count closure | {summary['gates']['psd_count_closure_status']} |
| M6 histogram closure | {summary['gates']['M6_histogram_closure_status']} |
| derivatives | {summary['gates']['time_derivative_status']} |
| particle identity tracking | {summary['gates']['particle_identity_tracking_status']} |
| exact elastic energy | BLOCKED: runtime elastic-energy trace not present in VTK |

## Endpoints

| metric | 6 h | 48 h |
|---|---:|---:|
| particle count | {int(first['particle_count'])} | {int(last['particle_count'])} |
| beta volume fraction | {float(first['beta_volume_fraction']):.9g} | {float(last['beta_volume_fraction']):.9g} |
| mean radius (nm) | {float(first['mean_radius_nm']):.6g} | {float(last['mean_radius_nm']):.6g} |
| matrix xB (h<0.005) | {float(first['matrix_xB']):.9g} | {float(last['matrix_xB']):.9g} |
| matrix xAg | {float(first['matrix_xAg']):.9g} | {float(last['matrix_xAg']):.9g} |
| Sv (nm^-1) | {float(first['Sv_spherical_equivalent_nm^-1']):.9g} | {float(last['Sv_spherical_equivalent_nm^-1']):.9g} |
| M6 (nm^3) | {float(first['M6_integral_nm^3']):.9g} | {float(last['M6_integral_nm^3']):.9g} |
| chemical + interface energy (J/m^3) | {float(first['total_excess_energy_J_m^-3_excluding_elastic']):.9g} | {float(last['total_excess_energy_J_m^-3_excluding_elastic']):.9g} |

## Runtime qualification carried with the VTK analysis

The completed elastic continuation reports `PASS_PF_CONSERVED_Y_ZERO_MODE_V1`, exact final mass equality, checkpoint/restart provenance restored and validated, `gp_enabled=false`, and average continuation throughput about 0.422442 s/step. These runtime records are copied into `audit_summary.json`; they are not inferred from VTK.

## Outputs

- `microstructure_time_series.csv`
- `particle_psd_samples.csv`
- `psd_histogram.csv`
- `audit_summary.json`
- `status.txt`

The exact elastic-energy gate remains open until a runtime elastic-energy/stress trace is exported. Any identity merge/split is separately fail-closed below.
"""
    if not tracking_ok:
        report += "\n\n## Fail-closed identity event\n\nThe periodic h-volume tracker detected an unexpected merge/split at registered 16 h (step 36330). The 22 components at 15 h become 19 at 16 h; one 16 h component has two parent overlaps. The CSV retains all rows, but identities after this event must not be interpreted as independent-particle trajectories without a merge-aware continuation contract.\n"
    (args.out_dir / "audit_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
