#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_DYN = (
    "/data/home/luozhiheng/CUDA_STO_PF/Results/workflows/T400_xB0p030/raw/"
    "chel_T400_cuda_400x400x400_dt0.03_steps30000_r2.495nm_xB0.030/continue_dyn_1"
)
DEFAULT_PROFILE = (
    "/data/home/luozhiheng/CUDA_STO_PF/Results/workflows/T400_xB0p030/analysis_tmp/"
    "exx_s0p0025_faceted_profiles_for_sched"
)
DEFAULT_PF_PARAMS = (
    "/data/home/luozhiheng/CUDA_STO_PF/jobs/generated/"
    "pf_params_scheduled_exx_s0p0025_dx1_lambda4.params"
)


def quote(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def dt_tag(dt: float) -> str:
    return f"{dt:.5f}".rstrip("0").rstrip(".").replace(".", "p")


def code_time_to_steps(code_time: float, dt: float) -> int:
    return int(round(code_time / dt))


def event_steps_csv(event_times: list[float], dt: float) -> str:
    return ",".join(str(code_time_to_steps(t, dt)) for t in event_times)


def build_command(
    *,
    exe: str,
    pf_param_effective: str,
    dt: float,
    nsteps: int,
    out_every: int,
    csv_every: int,
    elastic: int,
    external_strain: float,
    init_case_tag: str,
    source_dyn_dir: str,
    profile_dir: str,
    case_kind: str,
    uniform_raw_dir: str | None,
    one_nucleus_raw_dir: str | None,
    event_steps: list[int],
    centers_nm: str,
) -> list[str]:
    cmd = [
        exe,
        "400", "400", "400",
        f"{dt:g}",
        str(nsteps),
        str(out_every),
        str(csv_every),
        str(elastic),
        "--pf-param-file", pf_param_effective,
        "--temperature-C", "400",
        "--ic-23d-xB-out", "0.030",
        "--elastic", str(elastic),
        "--external-strain-exx", f"{external_strain:.10g}",
        "--enable-dynamics-mass-diagnostics",
        "--dynamics-mass-diag-interval", "1",
        "--init-case-tag", init_case_tag,
    ]
    if case_kind == "A_uniform_matrix":
        if not uniform_raw_dir:
            raise ValueError("Case A requires --uniform-raw-init-dir")
        cmd += [
            "--init-mode", "raw_fields",
            "--init-phi-raw", f"{uniform_raw_dir}/phi_init.raw",
            "--init-xB-raw", f"{uniform_raw_dir}/xB_init.raw",
            "--init-meta", f"{uniform_raw_dir}/init_meta.json",
        ]
    elif case_kind == "B_one_embedded_nucleus":
        if not one_nucleus_raw_dir:
            raise ValueError("Case B requires --one-nucleus-raw-init-dir")
        cmd += [
            "--init-mode", "raw_fields",
            "--init-phi-raw", f"{one_nucleus_raw_dir}/phi_init.raw",
            "--init-xB-raw", f"{one_nucleus_raw_dir}/xB_init.raw",
            "--init-meta", f"{one_nucleus_raw_dir}/init_meta.json",
        ]
    elif case_kind in ("C_scheduled_1_event", "D_scheduled_3_events"):
        cmd += [
            "--enable-scheduled-nucleation-test",
            "--scheduled-nuc-source-case-label", "T400_xB0p030_exx_s0p0025",
            "--scheduled-nuc-source-dyn-dir", source_dyn_dir,
            "--scheduled-nuc-profile-dir", profile_dir,
            "--scheduled-nuc-source-step", "1500",
            "--scheduled-nuc-steps", ",".join(str(x) for x in event_steps),
            "--scheduled-nuc-centers-nm", centers_nm,
            "--scheduled-nuc-xB-edge-mode", "sample-current-background-shell",
            "--scheduled-nuc-edge-sample-inner-nm", "0.0",
            "--scheduled-nuc-edge-sample-outer-nm", "3.0",
            "--scheduled-nuc-edge-sample-stat", "mean",
            "--scheduled-nuc-source-dx-nm", "0.1",
            "--scheduled-nuc-source-lambda-nm", "0.6",
            "--scheduled-nuc-target-lambda-nm", "4.0",
            "--scheduled-nuc-scale-geometry", "1.0",
            "--scheduled-nuc-scale-interface-width", "6.6666666667",
            "--scheduled-nuc-scale-xB-profile-width", "6.6666666667",
            "--scheduled-nuc-local-comp-outer-nm", "20.0",
            "--scheduled-nuc-local-comp-taper-nm", "5.0",
            "--scheduled-nucleation-write-event-vtk", "0",
        ]
    else:
        raise ValueError(f"Unsupported case_kind={case_kind}")
    return cmd


def analyze_run_dir(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "mass_drift_summary.json"
    relax_path = run_dir / "relaxation_diagnostics.csv"
    event_path = run_dir / "scheduled_nucleation_events.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    if not relax_path.exists():
        raise FileNotFoundError(f"Missing {relax_path}")

    mass = json.loads(summary_path.read_text(encoding="utf-8"))
    with relax_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {relax_path}")

    def fget(row: dict[str, str], key: str, default: float = math.nan) -> float:
        try:
            return float(row.get(key, ""))
        except ValueError:
            return default

    initial = fget(rows[0], "mean_xBtot")
    final = fget(rows[-1], "mean_xBtot")
    physical_time = fget(rows[-1], "time")
    drift = final - initial
    max_abs_step_drift = abs(float(mass.get("max_abs_step_drift", math.nan)))
    event_mass_error_max = 0.0
    event_mass_error_sum_abs = 0.0
    event_relative_mass_error_max = 0.0
    if event_path.exists():
        with event_path.open("r", encoding="utf-8", newline="") as f:
            event_rows = list(csv.DictReader(f))
        errs = [abs(fget(r, "event_mass_error")) for r in event_rows]
        rels = [abs(fget(r, "relative_event_mass_error")) for r in event_rows]
        if errs:
            event_mass_error_max = max(errs)
            event_mass_error_sum_abs = sum(errs)
        if rels:
            event_relative_mass_error_max = max(rels)
    return {
        "initial_mean_xBtot": initial,
        "final_mean_xBtot": final,
        "absolute_drift": drift,
        "relative_drift": drift / initial if initial else math.nan,
        "drift_per_step": drift / max(len(rows) - 1, 1),
        "drift_per_physical_time": drift / physical_time if physical_time else math.nan,
        "max_abs_step_drift": max_abs_step_drift,
        "cumulative_delta_phi_update": float(mass.get("cumulative_delta_phi_update", math.nan)),
        "cumulative_delta_Y_update": float(mass.get("cumulative_delta_Y_update", math.nan)),
        "cumulative_delta_Y_to_xB": float(mass.get("cumulative_delta_Y_to_xB", math.nan)),
        "cumulative_delta_clipping": float(mass.get("cumulative_delta_clipping", math.nan)),
        "total_clip_count_phi": float(mass.get("total_clip_count_phi", math.nan)),
        "total_clip_count_xB": float(mass.get("total_clip_count_xB", math.nan)),
        "total_clip_count_Y": float(mass.get("total_clip_count_Y", math.nan)),
        "event_mass_error_max": event_mass_error_max,
        "event_mass_error_sum_abs": event_mass_error_sum_abs,
        "event_relative_mass_error_max": event_relative_mass_error_max,
        "suspected_primary_source": mass.get("suspected_primary_source", "mixed_or_unclear"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create/collect pure dynamics xBtot mass-drift benchmark cases.")
    parser.add_argument("--repo", type=Path, default=Path("/data/home/luozhiheng/CUDA_STO_PF"))
    parser.add_argument("--exe", default="./main_cuda")
    parser.add_argument("--pf-param-file", default=DEFAULT_PF_PARAMS)
    parser.add_argument("--source-dyn-dir", default=DEFAULT_SOURCE_DYN)
    parser.add_argument("--profile-dir", default=DEFAULT_PROFILE)
    parser.add_argument("--uniform-raw-init-dir", default=None)
    parser.add_argument("--one-nucleus-raw-init-dir", default=None)
    parser.add_argument("--results-root", default="/data/home/luozhiheng/CUDA_STO_PF/Results/workflows/T400_xB0p030/pure_dynamics_mass_drift_benchmark")
    parser.add_argument("--physical-code-time", type=float, default=25.0)
    parser.add_argument("--dts", default="0.025,0.0125,0.00625")
    parser.add_argument("--cases", default="A,B,C,D")
    parser.add_argument("--out-every-scale", type=float, default=0.5, help="out_every = round(nsteps * scale)")
    parser.add_argument("--csv-every", type=int, default=10)
    parser.add_argument("--elastic", type=int, default=1)
    parser.add_argument("--external-strain-exx", type=float, default=0.0025)
    parser.add_argument("--partition", default="gpu_uvip")
    parser.add_argument("--qos", default="gpu_uvip")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--mass-drift-relative-threshold", type=float, default=1.0e-5)
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    jobs_dir = repo / "jobs/generated"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    results_root = Path(args.results_root)
    dts = [float(x) for x in args.dts.split(",") if x.strip()]
    case_map = {
        "A": ("A_uniform_matrix", []),
        "B": ("B_one_embedded_nucleus", []),
        "C": ("C_scheduled_1_event", [2.5]),
        "D": ("D_scheduled_3_events", [2.5, 7.5, 15.0]),
    }
    selected = [x.strip() for x in args.cases.split(",") if x.strip()]
    manifest_rows: list[dict[str, Any]] = []

    for code in selected:
        case_name, event_times = case_map[code]
        for dt in dts:
            nsteps = code_time_to_steps(args.physical_code_time, dt)
            events = [code_time_to_steps(t, dt) for t in event_times]
            out_every = max(1, int(round(nsteps * args.out_every_scale)))
            label = f"{case_name}_dt{dt_tag(dt)}_steps{nsteps}"
            run_root = results_root / label
            init_case_tag = label
            centers = "100,100,100;180,100,100;100,180,100"
            if case_name == "C_scheduled_1_event":
                centers = "100,100,100"
            try:
                cmd = build_command(
                    exe=args.exe,
                    pf_param_effective="${PF_PARAM_EFFECTIVE}",
                    dt=dt,
                    nsteps=nsteps,
                    out_every=out_every,
                    csv_every=args.csv_every,
                    elastic=args.elastic,
                    external_strain=args.external_strain_exx,
                    init_case_tag=init_case_tag,
                    source_dyn_dir=args.source_dyn_dir,
                    profile_dir=args.profile_dir,
                    case_kind=case_name,
                    uniform_raw_dir=args.uniform_raw_init_dir,
                    one_nucleus_raw_dir=args.one_nucleus_raw_init_dir,
                    event_steps=events,
                    centers_nm=centers,
                )
            except ValueError as exc:
                print(f"[skip] {label}: {exc}")
                continue
            cmd_text = quote(cmd).replace("'${PF_PARAM_EFFECTIVE}'", '"$PF_PARAM_EFFECTIVE"')
            sbatch_text = f"""#!/bin/bash
#SBATCH -J {label}
#SBATCH -p {args.partition}
#SBATCH --qos={args.qos}
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -o {repo}/jobs/logs/{label}_%j.out
#SBATCH -e {repo}/jobs/logs/{label}_%j.err

set -euo pipefail
cd {repo}
module purge >/dev/null 2>&1 || true
module load gcc/13.1.0 >/dev/null 2>&1 || true
module load cuda/cuda-12.8 >/dev/null 2>&1 || true
mkdir -p jobs/logs
PF_PARAM_EFFECTIVE="{repo}/jobs/generated/pf_params_benchmark_dt{dt_tag(dt)}.params"
python3 - <<'PY'
from pathlib import Path
src = Path({args.pf_param_file!r})
dst = Path("{repo}/jobs/generated/pf_params_benchmark_dt{dt_tag(dt)}.params")
dt = {dt!r}
lines = src.read_text(encoding="utf-8").splitlines()
out = []
done = False
for line in lines:
    if line.strip().startswith("dt="):
        out.append(f"dt={{dt:.17e}}")
        done = True
    else:
        out.append(line)
if not done:
    out.append(f"dt={{dt:.17e}}")
dst.write_text("\\n".join(out) + "\\n", encoding="utf-8")
PY
export CUDA_STO_RESULTS_ROOT={shlex.quote(str(run_root))}
mkdir -p "$CUDA_STO_RESULTS_ROOT"
{cmd_text}
"""
            sbatch_path = jobs_dir / f"{label}.sbatch"
            sbatch_path.write_text(sbatch_text, encoding="utf-8")
            manifest_rows.append(
                {
                    "case_name": case_name,
                    "dt": dt,
                    "nsteps": nsteps,
                    "physical_time": args.physical_code_time,
                    "insertion_steps": ",".join(str(x) for x in events),
                    "insertion_times": ",".join(f"{x:g}" for x in event_times),
                    "run_dir": str(run_root),
                    "sbatch_path": str(sbatch_path),
                    "command": cmd_text,
                }
            )
            print(f"[ok] wrote {sbatch_path}")
            if args.submit:
                result = subprocess.run(["sbatch", str(sbatch_path)], cwd=repo, check=True, text=True, capture_output=True)
                print(result.stdout.strip())

    manifest_path = results_root / "pure_dynamics_mass_drift_manifest.csv"
    results_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else [])
        if manifest_rows:
            writer.writeheader()
            writer.writerows(manifest_rows)
    print(f"[ok] wrote {manifest_path}")

    if args.collect:
        rows: list[dict[str, Any]] = []
        for item in manifest_rows:
            run_dir = Path(item["run_dir"])
            try:
                diag = analyze_run_dir(run_dir)
            except Exception as exc:
                print(f"[skip] collect {run_dir}: {exc}")
                continue
            row = dict(item)
            row.update(diag)
            row["pass_fail"] = (
                "PASS"
                if abs(row["event_mass_error_max"]) <= 1.0e-12 and abs(row["relative_drift"]) <= args.mass_drift_relative_threshold
                else "FAIL"
            )
            rows.append(row)
        out_csv = results_root / "pure_dynamics_mass_drift_benchmark.csv"
        if rows:
            with out_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"[ok] wrote {out_csv}")


if __name__ == "__main__":
    main()
