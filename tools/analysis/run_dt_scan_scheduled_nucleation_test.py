#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


DEFAULT_SOURCE_DYN = (
    "/data/home/luozhiheng/CUDA_STO_PF/Results/workflows/T400_xB0p030/raw/"
    "chel_T400_cuda_400x400x400_dt0.03_steps30000_r2.495nm_xB0.030/continue_dyn_1"
)
DEFAULT_PROFILE = (
    "/data/home/luozhiheng/CUDA_STO_PF/Results/workflows/T400_xB0p030/analysis_tmp/"
    "exx_s0p0025_faceted_profiles_for_sched"
)
DEFAULT_PF_PARAMS = "/data/home/luozhiheng/CUDA_STO_PF/jobs/generated/pf_params_scheduled_exx_s0p0025_dx1_lambda4.params"


def quote(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def make_sbatch(
    repo: Path,
    run_label: str,
    dt: float,
    steps: int,
    event_steps: str,
    source_dyn: str,
    profile_dir: str,
    pf_params: str,
    results_root: str,
    partition: str,
    qos: str,
    out_every: int,
    csv_every: int,
) -> str:
    dt_tag = str(dt).replace(".", "p")
    cmd = [
        "./main_cuda",
        "400",
        "400",
        "400",
        f"{dt:g}",
        str(steps),
        str(out_every),
        str(csv_every),
        "1",
        "--pf-param-file",
        "${PF_PARAM_EFFECTIVE}",
        "--enable-scheduled-nucleation-test",
        "--scheduled-nuc-source-case-label",
        "T400_xB0p030_exx_s0p0025",
        "--scheduled-nuc-source-dyn-dir",
        source_dyn,
        "--scheduled-nuc-profile-dir",
        profile_dir,
        "--scheduled-nuc-source-step",
        "1500",
        "--scheduled-nuc-steps",
        event_steps,
        "--scheduled-nuc-centers-nm",
        "100,100,100;180,100,100;100,180,100",
        "--scheduled-nuc-xB-edge-mode",
        "sample-current-background-shell",
        "--scheduled-nuc-edge-sample-inner-nm",
        "0.0",
        "--scheduled-nuc-edge-sample-outer-nm",
        "3.0",
        "--scheduled-nuc-edge-sample-stat",
        "mean",
        "--scheduled-nuc-source-dx-nm",
        "0.1",
        "--scheduled-nuc-source-lambda-nm",
        "0.6",
        "--scheduled-nuc-target-lambda-nm",
        "4.0",
        "--scheduled-nuc-scale-geometry",
        "1.0",
        "--scheduled-nuc-scale-interface-width",
        "6.6666666667",
        "--scheduled-nuc-scale-xB-profile-width",
        "6.6666666667",
        "--scheduled-nuc-local-comp-outer-nm",
        "20.0",
        "--scheduled-nuc-local-comp-taper-nm",
        "5.0",
        "--scheduled-nucleation-write-event-vtk",
        "0",
        "--enable-dynamics-mass-diagnostics",
        "--dynamics-mass-diag-interval",
        "1",
        "--temperature-C",
        "400",
        "--ic-23d-xB-out",
        "0.030",
        "--elastic",
        "1",
        "--external-strain-exx",
        "0.0025",
        "--init-case-tag",
        f"scheduled_nuc_exx_s0p0025_dt{dt_tag}",
    ]
    cmd_text = quote(cmd).replace("'${PF_PARAM_EFFECTIVE}'", '"$PF_PARAM_EFFECTIVE"')
    return f"""#!/bin/bash
#SBATCH -J {run_label}
#SBATCH -p {partition}
#SBATCH --qos={qos}
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -o {repo}/jobs/logs/{run_label}_%j.out
#SBATCH -e {repo}/jobs/logs/{run_label}_%j.err

set -euo pipefail
cd {repo}
module load cuda/cuda-12.8 >/dev/null 2>&1 || true
mkdir -p jobs/logs
PF_PARAM_EFFECTIVE="{repo}/jobs/generated/pf_params_scheduled_exx_s0p0025_dx1_lambda4_dt{dt_tag}.params"
python3 - <<'PY'
from pathlib import Path
src = Path({pf_params!r})
dst = Path("{repo}/jobs/generated/pf_params_scheduled_exx_s0p0025_dx1_lambda4_dt{dt_tag}.params")
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
print(f"[pf-param] wrote {{dst}} with dt={{dt}}")
PY
export CUDA_STO_RESULTS_ROOT={shlex.quote(results_root)}
mkdir -p "$CUDA_STO_RESULTS_ROOT"
echo "[run] {cmd_text}"
{cmd_text}
RUN_DIR=$(find {shlex.quote(results_root)} -type f -name scheduled_nucleation_events.csv -printf '%h\\n' | sort | tail -1)
if [ -n "$RUN_DIR" ]; then
  python3 tools/analysis/analyze_scheduled_nucleation_drift_segments.py --run-dir "$RUN_DIR"
fi
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create/submit scheduled nucleation dt-scan Slurm jobs.")
    parser.add_argument("--repo", type=Path, default=Path("/data/home/luozhiheng/CUDA_STO_PF"))
    parser.add_argument("--source-dyn-dir", default=DEFAULT_SOURCE_DYN)
    parser.add_argument("--profile-dir", default=DEFAULT_PROFILE)
    parser.add_argument("--pf-param-file", default=DEFAULT_PF_PARAMS)
    parser.add_argument("--results-root", default="/data/home/luozhiheng/CUDA_STO_PF/Results/workflows/T400_xB0p030/scheduled_nuc_dt_scan")
    parser.add_argument("--partition", default="gpu_uvip")
    parser.add_argument("--qos", default="gpu_uvip")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--only", choices=("both", "dt0p025", "dt0p0125"), default="both")
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    jobs_dir = repo / "jobs/generated"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    specs = []
    if args.only in ("both", "dt0p025"):
        specs.append(("scheduled_nuc_dt0p025_exx_s0p0025", 0.025, 1000, "100,300,600", 500))
    if args.only in ("both", "dt0p0125"):
        specs.append(("scheduled_nuc_dt0p0125_exx_s0p0025", 0.0125, 2000, "200,600,1200", 1000))

    for label, dt, steps, events, out_every in specs:
        sbatch = make_sbatch(
            repo,
            label,
            dt,
            steps,
            events,
            args.source_dyn_dir,
            args.profile_dir,
            args.pf_param_file,
            f"{args.results_root}/{label}",
            args.partition,
            args.qos,
            out_every,
            10,
        )
        sbatch_path = jobs_dir / f"{label}.sbatch"
        sbatch_path.write_text(sbatch, encoding="utf-8")
        print(f"[ok] wrote {sbatch_path}")
        if args.submit:
            result = subprocess.run(["sbatch", str(sbatch_path)], cwd=repo, check=True, text=True, capture_output=True)
            print(result.stdout.strip())


if __name__ == "__main__":
    main()
