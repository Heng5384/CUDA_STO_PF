#!/usr/bin/env python3
"""Run a prepared low-cost Correction-2 matrix on workstation only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def case_failed(status: dict[str, object]) -> bool:
    return (
        int(status["returncode"]) != 0
        or int(status["download_returncode"]) != 0
        or int(status["accepted_steps"]) != int(status["expected_steps"])
        or int(status["retry_count"]) != 0
        or int(status["reject_count"]) != 0
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, check=False, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--host", default="workstation-tail")
    parser.add_argument(
        "--remote-repo",
        default="/home/zhiheng/PF/CUDA_STO_PF-pf-ctot-production-candidate",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--compact-download", action="store_true",
        help=("keep iteration-level bulk traces on workstation while downloading "
              "all acceptance, outer, energy, checkpoint, and run-log evidence"),
    )
    parser.add_argument(
        "--reuse-completed-remote", action="store_true",
        help="reuse a fail-closed complete remote run after an interrupted download",
    )
    parser.add_argument(
        "--continue-on-failure", action="store_true",
        help=("continue with independent matrix cases after a fail-closed case; "
              "the final matrix status still fails"),
    )
    args = parser.parse_args()
    matrix_root = args.matrix_root.resolve()
    cases_root = matrix_root / "cases"
    local_source_hashes = {
        name: sha256(ROOT / name) for name in (
            "main_cuda.cu", "cuda_kernels.cu", "cuda_kernels.h",
            "pf_params.h", "thermo_utils.h", "Unit_Psedobinary.py",
        )
    }
    remote_check = run([
        "ssh", args.host,
        "cd " + shlex.quote(args.remote_repo) + " && sha256sum "
        + " ".join(local_source_hashes),
    ], capture_output=True)
    if remote_check.returncode != 0:
        raise RuntimeError(remote_check.stderr)
    remote_hashes = {
        line.split()[1]: line.split()[0]
        for line in remote_check.stdout.splitlines() if len(line.split()) == 2
    }
    if remote_hashes != local_source_hashes:
        raise RuntimeError("workstation source hashes do not match local worktree")
    binary_check = run([
        "ssh", args.host,
        "cd " + shlex.quote(args.remote_repo) + " && sha256sum main_cuda",
    ], capture_output=True)
    if binary_check.returncode != 0 or len(binary_check.stdout.split()) != 2:
        raise RuntimeError("could not record workstation main_cuda hash")
    remote_binary_hash = binary_check.stdout.split()[0]
    gpu_check = run([
        "ssh", args.host,
        "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null",
    ], capture_output=True)
    if gpu_check.returncode != 0 or gpu_check.stdout.strip():
        raise RuntimeError("workstation GPU is occupied; refusing to displace work")

    relative = matrix_root.relative_to(ROOT)
    remote_root = f"{args.remote_repo}/{relative}"
    mkdir = run(["ssh", args.host, "mkdir -p " + shlex.quote(remote_root)])
    if mkdir.returncode != 0:
        raise RuntimeError("could not create workstation matrix root")
    sync = run([
        "rsync", "-az", str(matrix_root) + "/", f"{args.host}:{remote_root}/",
    ])
    if sync.returncode != 0:
        raise RuntimeError("matrix input rsync failed")

    statuses: list[dict[str, object]] = []
    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        manifest = json.loads((case_dir / "runtime_manifest.json").read_text())
        expected_steps = int(manifest["nsteps"])
        status_path = case_dir / "workstation_status.json"
        if args.resume and status_path.is_file():
            old = json.loads(status_path.read_text())
            artifact_complete = (
                int(old.get("download_returncode", 1)) == 0
                and int(old.get("accepted_steps", -1)) == expected_steps
                and int(old.get("retry_count", 1)) == 0
                and int(old.get("reject_count", 1)) == 0
                and (case_dir / "run/run.log").is_file()
                and f"steps_completed          : {expected_steps}" in
                    (case_dir / "run/run.log").read_text(errors="replace")
            )
            if artifact_complete:
                if int(old.get("returncode", 1)) != 0:
                    old["ssh_returncode"] = old["returncode"]
                    old["returncode"] = 0
                    old["transport_disconnect_recovered_from_complete_artifacts"] = True
                    status_path.write_text(json.dumps(old, indent=2) + "\n")
                statuses.append(old)
                continue
        remote_case = f"{remote_root}/cases/{case_dir.name}"
        nx, ny, nz = map(int, manifest["grid"])
        dt = float(manifest["dt_code"])
        steps = expected_steps
        command = (
            "set -e; repo=" + shlex.quote(args.remote_repo) + "; "
            "c=" + shlex.quote(remote_case) + "; mkdir -p \"$c/run\"; "
            "cd \"$c/run\"; export CUDA_STO_RESULTS_ROOT=\"$c/run/results\"; "
            f"\"$repo/main_cuda\" {nx} {ny} {nz} {dt:.17e} {steps} {steps} {steps} 0 "
            "--mode=dynamics --pf-param-file \"$c/runtime.params\" "
            "--temperature-C 400 --init-mode raw_fields "
            "--init-phi-raw \"$c/phi_init.raw\" --init-xB-raw \"$c/xB_init.raw\" "
            "--init-Ctot-raw \"$c/Ctot_init.raw\" --init-meta \"$c/init_meta.json\" "
            "> run.log 2>&1"
        )
        reused_remote = False
        if args.resume and args.reuse_completed_remote:
            probe = run([
                "ssh", args.host,
                "f=" + shlex.quote(remote_case + "/run/run.log") + "; "
                "test -f \"$f\" && "
                f"test \"$(grep -c CTOT_MIMETIC_BE_ACCEPT \"$f\")\" -eq {steps} && "
                "test \"$(grep -c CTOT_COUPLED_STEP_RETRY \"$f\")\" -eq 0 && "
                "test \"$(grep -c CTOT_MIMETIC_BE_REJECT \"$f\")\" -eq 0 && "
                f"grep -q 'steps_completed          : {steps}' \"$f\" && echo COMPLETE",
            ], capture_output=True)
            reused_remote = probe.returncode == 0 and probe.stdout.strip() == "COMPLETE"
        if reused_remote:
            completed = subprocess.CompletedProcess([], 0, "", "")
            wall = -1.0
        else:
            started = time.monotonic()
            completed = run(["ssh", args.host, command])
            wall = time.monotonic() - started
        download_command = ["rsync", "-az"]
        if args.compact_download:
            download_command.extend([
                "--exclude=*/ctot_nonlinear_iterations.csv",
                "--exclude=*/ctot_phase_kkt_iterations.csv",
            ])
        download_command.extend([
            f"{args.host}:{remote_case}/run/", str(case_dir / "run") + "/",
        ])
        download = run(download_command)
        log = (case_dir / "run/run.log").read_text(errors="replace") if (
            case_dir / "run/run.log"
        ).is_file() else ""
        artifact_complete = (
            download.returncode == 0
            and log.count("CTOT_MIMETIC_BE_ACCEPT") == steps
            and log.count("CTOT_COUPLED_STEP_RETRY") == 0
            and log.count("CTOT_MIMETIC_BE_REJECT") == 0
            and f"steps_completed          : {steps}" in log
        )
        effective_returncode = (
            0 if completed.returncode == 255 and artifact_complete
            else completed.returncode
        )
        status = {
            "case": case_dir.name,
            "returncode": effective_returncode,
            "ssh_returncode": completed.returncode,
            "transport_disconnect_recovered_from_complete_artifacts": (
                completed.returncode == 255 and artifact_complete
            ),
            "reused_completed_remote_run": reused_remote,
            "compact_download": args.compact_download,
            "download_returncode": download.returncode,
            "wall_time_s": wall,
            "accepted_steps": log.count("CTOT_MIMETIC_BE_ACCEPT"),
            "retry_count": log.count("CTOT_COUPLED_STEP_RETRY"),
            "reject_count": log.count("CTOT_MIMETIC_BE_REJECT"),
            "expected_steps": steps,
            "binary_sha256": remote_binary_hash,
            "source_hashes": local_source_hashes,
            "host": args.host,
        }
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        statuses.append(status)
        print(json.dumps(status, sort_keys=True), flush=True)
        if case_failed(status) and not args.continue_on_failure:
            break
    (matrix_root / "workstation_matrix_status.json").write_text(
        json.dumps({
            "continue_on_failure": args.continue_on_failure,
            "cases": statuses,
        }, indent=2) + "\n"
    )
    failed = any(case_failed(row) for row in statuses) or (
        len(statuses) != len(list(cases_root.iterdir()))
    )
    print(f"workstation_matrix_cases_completed={len(statuses)}")
    print(f"workstation_matrix_status={'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
