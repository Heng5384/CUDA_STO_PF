#!/usr/bin/env python3
"""Generate case-level provenance for the PF-only closure audit."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/pf_only_baseline_closure"
OLD_ACCEPTANCE_BINARY = "a3a14280793f274614091921daf81d120cd9e4daba03a45f39b1fc072cbe19a3"


def sha(path: Path | None) -> str:
    if path is None or not path.exists() or not path.is_file():
        return "MISSING"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args: list[str], cwd: Path = ROOT) -> str:
    proc = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    return proc.stdout.strip() if proc.returncode == 0 else f"UNAVAILABLE:{proc.stderr.strip()}"


def read(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def resolve_param(raw: str) -> Path | None:
    name = Path(raw).name
    candidates = [
        ROOT / "params/pf_only_baseline_closure" / name,
        ROOT / "params/pf_only_baseline_closure/remediation" / name,
        ROOT / "params/rsmd_equal_time_domain_low_overshoot" / name,
    ]
    return next((p for p in candidates if p.exists()), None)


def locate_output(case: str, temp: int, dt: float, nsteps: int) -> Path | None:
    base = ROOT / f"Results/chel_T{temp}_cuda_128x128x128_dt{dt}_steps{nsteps}_xB0.008" / case
    if base.exists():
        return base
    found = list((ROOT / "Results").glob(f"chel_T{temp}_cuda_128x128x128_dt*_steps{nsteps}_xB0.008/{case}"))
    return found[0] if found else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    commit = command(["git", "rev-parse", "HEAD"])
    dirty = command(["git", "status", "--short"])
    relevant_dirty = []
    for line in dirty.splitlines():
        path = line[3:] if len(line) > 3 else line
        if path in {"main_cuda.cu", "cuda_kernels.cu", "cuda_kernels.h", "pf_params.h"} or \
                path.startswith("scripts/") or path.startswith("params/pf_only_baseline_closure"):
            relevant_dirty.append(path)
    remote = command([
        "ssh", "workstation-direct",
        "cd /home/zhiheng/PF/CUDA_STO_PF && "
        "printf 'hostname='; hostname; "
        "printf 'binary='; sha256sum main_cuda | awk '{print $1}'; "
        "printf 'nvcc='; /usr/local/cuda-12.9/bin/nvcc --version | tail -1; "
        "printf 'gpu='; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1"
    ])
    remote_meta = {}
    for line in remote.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            remote_meta[key] = value

    library = ROOT / "data/nucleus_library/nucleus_library.with_Zr.staging.csv"
    profiles = {
        400: ROOT / "Results/runtime_profile_cache/dx1p0/nlib_dc_T400_xB003/faceted_family_profiles.csv",
        380: ROOT / "Results/runtime_profile_cache/dx1p0/nlib_00006/faceted_family_profiles.csv",
    }
    rows: list[dict[str, object]] = []

    historical = read(ROOT / "reports/rsmd_equal_time_domain_low_overshoot/rsmd_operator_split_dt_convergence_summary.csv")
    historical_source = ROOT / "reports/rsmd_equal_time_domain_low_overshoot/rsmd_operator_split_dt_convergence_summary.csv"
    for item in historical:
        if item.get("phase") != "pf_only":
            continue
        temp, dt, nsteps = int(item["T_C"]), float(item["dt"]), int(item["nsteps"])
        rows.append({
            "evidence_class": "historical_equal_time_failure", "case": item["case"],
            "operator_case": "P0_full_historical", "T_C": temp, "dt": dt, "nsteps": nsteps,
            "physical_time_s": item.get("post_handoff_physical_time_s", ""),
            "param_file": "params/rsmd_equal_time_domain_low_overshoot/" + item["case"] + ".params",
            "parameter_sha256": sha(resolve_param(item["case"] + ".params")),
            "output_dir": item.get("output_dir", ""), "evidence_file": str(historical_source),
            "evidence_sha256": sha(historical_source), "binary_sha256": "NOT_CAPTURED_IN_ORIGINAL_MANIFEST",
            "seed_library_entry": "nlib_dc_T400_xB003", "seed_profile_sha256": sha(profiles[temp]),
            "output_cadence_steps": "historical_runtime", "projection_cadence": "each_PF_step_after_handoff",
            "history_mode": "lagged_dYdt", "pf_y_update_mode": "lagged_rhs",
            "pf_matrix_storage_floor": "NOT_APPLICABLE", "S3_source_component_frozen": True,
            "run_status": item.get("acceptance_status", "COMPLETE"),
        })

    operator_manifest = read(ROOT / "params/pf_only_baseline_closure/operator_manifest.csv")
    for item in operator_manifest:
        if item["case"].startswith(("P7_shared", "P8_shared")):
            output = None
        else:
            output = locate_output(item["case"], int(item["T_C"]), float(item["dt"]), int(item["nsteps"]))
        evidence = output / "diagnostic_rsmd_seed_growth_time_series.csv" if output else None
        rows.append({
            "evidence_class": "operator_decomposition", "case": item["case"],
            "operator_case": item["operator_case"], "T_C": item["T_C"], "dt": item["dt"],
            "nsteps": item["nsteps"], "physical_time_s": item["post_handoff_physical_time_s"],
            "param_file": item["param_file"], "parameter_sha256": sha(resolve_param(item["param_file"])),
            "output_dir": str(output) if output else item["param_file"],
            "evidence_file": str(evidence) if evidence else item["notes"],
            "evidence_sha256": sha(evidence), "binary_sha256": OLD_ACCEPTANCE_BINARY,
            "seed_library_entry": "nlib_dc_T400_xB003", "seed_profile_sha256": sha(profiles[400]),
            "output_cadence_steps": 10, "projection_cadence": "each_PF_step_after_handoff",
            "history_mode": "gamma_disabled_diagnostic" if item["history_gamma_disabled"] == "1" else "lagged_dYdt",
            "pf_y_update_mode": "lagged_rhs", "pf_matrix_storage_floor": "NOT_APPLICABLE",
            "S3_source_component_frozen": True,
            "run_status": "SHARED_ALIAS" if item["case"].startswith(("P7_shared", "P8_shared")) else ("COMPLETE" if evidence and evidence.exists() else "NOT_RUN"),
        })

    acceptance = read(ROOT / "params/pf_only_baseline_closure/remediation/acceptance_manifest.csv")
    for item in acceptance:
        temp, dt, nsteps = int(item["T_C"]), float(item["dt"]), int(item["nsteps"])
        output = locate_output(item["case"], temp, dt, nsteps)
        evidence = output / "diagnostic_rsmd_seed_growth_time_series.csv" if output else None
        rows.append({
            "evidence_class": "storage_weighted_remediation", "case": item["case"],
            "operator_case": "fix_A_plus_B", "T_C": temp, "dt": dt, "nsteps": nsteps,
            "physical_time_s": item["post_handoff_physical_time_s"], "param_file": item["param_file"],
            "parameter_sha256": sha(resolve_param(item["param_file"])),
            "output_dir": str(output) if output else "NOT_RUN",
            "evidence_file": str(evidence) if evidence else "NOT_RUN", "evidence_sha256": sha(evidence),
            "binary_sha256": remote_meta.get("binary", "UNAVAILABLE"),
            "seed_library_entry": "nlib_dc_T400_xB003" if temp == 400 else "nlib_00006",
            "seed_profile_sha256": sha(profiles[temp]), "output_cadence_steps": round(0.02/dt),
            "projection_cadence": "each_PF_step_after_handoff", "history_mode": "none",
            "pf_y_update_mode": "x_transport_projection_split", "pf_matrix_storage_floor": 0.1,
            "S3_source_component_frozen": True, "run_status": "COMPLETE" if evidence and evidence.exists() else "NOT_RUN",
        })

    common = {
        "git_commit": commit, "worktree_dirty": bool(dirty),
        "worktree_diff_files": ";".join(relevant_dirty),
        "workstation_hostname": remote_meta.get("hostname", "UNAVAILABLE"),
        "cuda_compiler": remote_meta.get("nvcc", "UNAVAILABLE"),
        "gpu_driver": remote_meta.get("gpu", "UNAVAILABLE"),
        "main_cuda_sha256": sha(ROOT / "main_cuda.cu"),
        "cuda_kernels_sha256": sha(ROOT / "cuda_kernels.cu"),
        "cuda_kernels_h_sha256": sha(ROOT / "cuda_kernels.h"),
        "pf_params_sha256": sha(ROOT / "pf_params.h"),
        "seed_library_sha256": sha(library), "domain": "128x128x128", "dx_nm": 1.0,
        "physical_time_per_code_time_s": 0.9254156720524821,
        "RSMD_applied_mass_expected": 0.0,
    }
    rows = [{**row, **common} for row in rows]
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (OUT / "pf_baseline_closure_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(f"pf_baseline_provenance_rows={len(rows)}")


if __name__ == "__main__":
    main()
