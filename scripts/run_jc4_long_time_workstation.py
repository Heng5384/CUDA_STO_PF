#!/usr/bin/env python3
"""Build/sync and run prepared JC4 planar cases on workstation only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "Makefile", "main_cuda.cu", "cuda_kernels.cu", "cuda_kernels.h",
    "pf_params.h", "thermo_utils.h", "phase_functions.h", "cuda_common.h",
    "phase_kkt_utils.h", "phase_pdas_reduction.h",
    "ctot_performance_profiler.h", "ctot_transport_bound_utils.h",
    "Unit_Psedobinary.py",
)


def execute(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, check=False, **kwargs)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_idle(host: str) -> None:
    probe = execute([
        "ssh", host,
        "p=$(pgrep -af main_cuda | grep -v 'pgrep -af' || true); "
        "g=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true); "
        "printf 'PROCESSES\\n%s\\nGPU_PIDS\\n%s\\n' \"$p\" \"$g\"",
    ], capture_output=True)
    if probe.returncode != 0:
        raise RuntimeError(f"workstation preflight failed: {probe.stderr.strip()}")
    sections = probe.stdout.split("GPU_PIDS\n", 1)
    processes = sections[0].replace("PROCESSES\n", "").strip()
    gpu_pids = sections[1].strip() if len(sections) == 2 else "unknown"
    if processes or gpu_pids:
        raise RuntimeError(
            "workstation has an existing main_cuda/GPU task; refusing to displace it\n"
            + probe.stdout
        )


def source_hashes() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


def sync_and_build(host: str, remote_repo: str) -> dict[str, object]:
    remote_idle(host)
    mkdir = execute(["ssh", host, "mkdir -p " + shlex.quote(remote_repo)])
    if mkdir.returncode != 0:
        raise RuntimeError("could not create workstation candidate worktree")
    sync = execute([
        "rsync", "-az", *[str(ROOT / name) for name in SOURCE_FILES],
        f"{host}:{remote_repo}/",
    ])
    if sync.returncode != 0:
        raise RuntimeError("workstation source sync failed")
    command = (
        "cd " + shlex.quote(remote_repo) + " && "
        "make main_cuda NVCC=/usr/local/cuda-12.9/bin/nvcc "
        "CUDA_ROOT=/usr/local/cuda-12.9 -j1"
    )
    started = time.monotonic()
    built = execute(["ssh", host, command], capture_output=True)
    wall = time.monotonic() - started
    report = {
        "host": host,
        "remote_repo": remote_repo,
        "returncode": built.returncode,
        "wall_time_s": wall,
        "stdout_tail": "\n".join(built.stdout.splitlines()[-40:]),
        "stderr_tail": "\n".join(built.stderr.splitlines()[-40:]),
        "source_hashes": source_hashes(),
    }
    if built.returncode != 0:
        raise RuntimeError(json.dumps(report, indent=2))
    remote = execute([
        "ssh", host,
        "cd " + shlex.quote(remote_repo) + " && sha256sum main_cuda "
        + " ".join(SOURCE_FILES),
    ], capture_output=True)
    if remote.returncode != 0:
        raise RuntimeError("could not hash workstation build")
    report["remote_hash_output"] = remote.stdout
    report["binary_sha256"] = remote.stdout.splitlines()[0].split()[0]
    return report


def verify_source(host: str, remote_repo: str) -> tuple[dict[str, str], str]:
    expected = source_hashes()
    probe = execute([
        "ssh", host,
        "cd " + shlex.quote(remote_repo) + " && sha256sum main_cuda "
        + " ".join(SOURCE_FILES),
    ], capture_output=True)
    if probe.returncode != 0:
        raise RuntimeError("workstation source/binary hash probe failed")
    lines = [line.split() for line in probe.stdout.splitlines() if len(line.split()) == 2]
    binary_hash = lines[0][0]
    remote = {parts[1]: parts[0] for parts in lines[1:]}
    if remote != expected:
        raise RuntimeError("workstation source hashes do not match local candidate")
    return expected, binary_hash


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--host", default="workstation-tail")
    parser.add_argument(
        "--remote-repo",
        default="/home/zhiheng/PF/CUDA_STO_PF-pf-ctot-production-candidate",
    )
    parser.add_argument("--sync-build", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    matrix_root = args.matrix_root.resolve()
    if args.sync_build:
        build_report = sync_and_build(args.host, args.remote_repo)
        (matrix_root / "workstation_build_report.json").write_text(
            json.dumps(build_report, indent=2) + "\n", encoding="utf-8"
        )
    remote_idle(args.host)
    hashes, binary_hash = verify_source(args.host, args.remote_repo)
    relative = matrix_root.relative_to(ROOT)
    remote_root = f"{args.remote_repo}/{relative}"
    execute(["ssh", args.host, "mkdir -p " + shlex.quote(remote_root)])
    synced = execute([
        "rsync", "-az", str(matrix_root) + "/", f"{args.host}:{remote_root}/",
    ])
    if synced.returncode != 0:
        raise RuntimeError("JC4 matrix sync failed")
    manifest = json.loads((matrix_root / "manifest.json").read_text())
    selected = set(args.case)
    statuses: list[dict[str, object]] = []
    for case_manifest in manifest["cases"]:
        case_id = str(case_manifest["case_id"])
        if selected and case_id not in selected:
            continue
        case_dir = matrix_root / "cases" / case_id
        status_path = case_dir / "workstation_status.json"
        expected_steps = int(case_manifest["nsteps"])
        if args.resume and status_path.is_file():
            old = json.loads(status_path.read_text())
            if (int(old.get("returncode", 1)) == 0
                    and int(old.get("accepted_steps", -1)) == expected_steps
                    and int(old.get("retry_count", 1)) == 0
                    and int(old.get("reject_count", 1)) == 0):
                statuses.append(old)
                continue
        remote_case = f"{remote_root}/cases/{case_id}"
        nx, ny, nz = map(int, case_manifest["grid"])
        dt = float(case_manifest["dt_code"])
        interval = int(case_manifest["out_every"])
        command = (
            "set -e; repo=" + shlex.quote(args.remote_repo) + "; "
            "c=" + shlex.quote(remote_case) + "; mkdir -p \"$c/run\"; "
            "cd \"$c/run\"; export CUDA_STO_RESULTS_ROOT=\"$c/run/results\"; "
            f"\"$repo/main_cuda\" {nx} {ny} {nz} {dt:.17e} "
            f"{expected_steps} {interval} {interval} 0 "
            "--mode=dynamics --pf-param-file \"$c/runtime.params\" "
            "--temperature-C 400 --init-mode raw_fields "
            "--init-phi-raw \"$c/phi_init.raw\" "
            "--init-xB-raw \"$c/xB_init.raw\" "
            "--init-Ctot-raw \"$c/Ctot_init.raw\" "
            "--init-meta \"$c/init_meta.json\" > run.log 2>&1"
        )
        started = time.monotonic()
        completed = execute(["ssh", args.host, command])
        wall = time.monotonic() - started
        downloaded = execute([
            "rsync", "-az", f"{args.host}:{remote_case}/run/",
            str(case_dir / "run") + "/",
        ])
        log_path = case_dir / "run/run.log"
        log = log_path.read_text(errors="replace") if log_path.is_file() else ""
        status = {
            "case_id": case_id,
            "returncode": completed.returncode,
            "download_returncode": downloaded.returncode,
            "wall_time_s": wall,
            "accepted_steps": log.count("CTOT_MIMETIC_BE_ACCEPT"),
            "retry_count": log.count("CTOT_COUPLED_STEP_RETRY"),
            "reject_count": log.count("CTOT_MIMETIC_BE_REJECT"),
            "expected_steps": expected_steps,
            "source_hashes": hashes,
            "binary_sha256": binary_hash,
            "host": args.host,
            "cluster_used": False,
        }
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        statuses.append(status)
        print(json.dumps(status, sort_keys=True), flush=True)
        if (completed.returncode != 0 or downloaded.returncode != 0
                or status["accepted_steps"] != expected_steps
                or status["retry_count"] or status["reject_count"]):
            break
    output = {"cases": statuses, "cluster_used": False}
    (matrix_root / "workstation_matrix_status.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    passed = bool(statuses) and all(
        int(row["returncode"]) == 0
        and int(row["download_returncode"]) == 0
        and int(row["accepted_steps"]) == int(row["expected_steps"])
        and int(row["retry_count"]) == 0
        and int(row["reject_count"]) == 0
        for row in statuses
    )
    print(f"jc4_workstation_cases_completed={len(statuses)}")
    print(f"jc4_workstation_status={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
