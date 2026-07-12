#!/usr/bin/env python3
import csv
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/home/zhiheng/PF/CUDA_STO_PF"
manifest = ROOT / "params" / "pf_only_x_q_validation" / "case_manifest.csv"
rows = list(csv.DictReader(manifest.open()))
for row in rows:
    tag = row["case"]
    nsteps = row["nsteps"]
    remote_param = f"{REMOTE}/{row['param_file']}"
    cmd = (
        f"cd {REMOTE} && ./main_cuda --Nx 128 --Ny 128 --Nz 128 "
        f"--pf-param-file {remote_param} --nsteps {nsteps} --out-every {nsteps} "
        f"--csv-out-every 10 --elastic 1 --init-case-tag {tag} "
        f"> reports/pf_only_x_q_validation/logs/{tag}.log 2>&1"
    )
    print(f"RUN {tag}", flush=True)
    result = subprocess.run(["ssh", "workstation-direct", cmd])
    if result.returncode != 0:
        raise SystemExit(f"FAILED {tag}: rc={result.returncode}")
    print(f"DONE {tag}", flush=True)
