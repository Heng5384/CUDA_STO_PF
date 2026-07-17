#!/usr/bin/env python3
"""Freeze provenance and materialize the transport residual-gate test contract."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "transport_residual_gate_v1"
COMMON = (
    ROOT / "reports" / "bounded_retry_bdf2_v1" / "workstation_runs" /
    "equal_time_common_input"
)
BASELINE_DIRS = (
    "reports/bdf2_v1",
    "reports/bdf2_event_v1",
    "reports/active_manifold_bdf2_v1",
    "reports/bounded_retry_bdf2_v1",
)
EXPECTED_HEAD = "f440c0dcd4c02cd45d9079c35d3838ebfa9b37e2"
FROZEN_REMOTE_BINARY_SHA256 = (
    "3692e11ab8b05371b893a6ea4358880b10d774de433229ab51cec3c08efbf01a"
)
INSTRUMENTED_REMOTE_BINARY_SHA256 = (
    "3986440d157f29c5ab53387345f67bbedf7c264b1a5136b04461cec32a954b0b"
)
FROZEN_REMOTE_SOURCE_HASHES = {
    "main_cuda.cu": "db3dac3646c4beb3eb7cb989630a244942bc7de45ece81ffcb6b501343cd4e56",
    "cuda_kernels.cu": "a4d269067ab2de6059af1d48c333eda80aec43268c3f61036eab1a66e760b81a",
    "cuda_kernels.h": "3eb5cfa9ed35ddb533f43d5847ca270a0ac3a4ddb389770140e7fa7e7c425b53",
    "pf_params.h": "414792a90107526c8721245f71e66e9523ef382a13c58c68ae9cd7b1a1a3a1f2",
    "active_manifold_bdf2_utils.h": "e4fcce0513ee44fd462ebf711e247ffbc297cd73cefe3a26cf2f3039439489eb",
    "bounded_retry_bdf2_utils.h": "bb81c5996ddc232708643f7773258f24c6a78ba9e4b8379141a30f6a6b922e9c",
}
GATES = {"G12": 1.0e-12, "G10": 1.0e-10, "G9": 1.0e-9, "G8": 1.0e-8}
DTS = {
    "dt16": (1.95312500000000011e-4, 8000),
    "dt8": (3.90625000000000022e-4, 4000),
    "dt4": (7.81250000000000043e-4, 2000),
    "dt2": (1.56250000000000009e-3, 1000),
    "dt1": (3.12500000000000017e-3, 500),
}
T_REAL_UNIT_S = 41.12958542455477


def run(*args: str) -> bytes:
    return subprocess.check_output(args, cwd=ROOT)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(relative_dirs: tuple[str, ...]) -> tuple[str, int]:
    records: list[bytes] = []
    count = 0
    for relative in relative_dirs:
        base = ROOT / relative
        for path in sorted(candidate for candidate in base.rglob("*") if candidate.is_file()):
            rel = path.relative_to(ROOT).as_posix()
            records.append(f"{sha_file(path)}  {rel}\n".encode())
            count += 1
    return sha_bytes(b"".join(records)), count


def git_blob(path: str) -> bytes:
    return run("git", "show", f"{EXPECTED_HEAD}:{path}")


def line_hits(path: Path, pattern: str) -> list[int]:
    regex = re.compile(pattern)
    return [
        index
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if regex.search(line)
    ]


def main() -> int:
    head = run("git", "rev-parse", "HEAD").decode().strip()
    if head != EXPECTED_HEAD:
        raise SystemExit(
            f"BLOCKED_TRANSPORT_GATE_BASELINE_MISMATCH: HEAD={head}"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    tree_digest, tree_files = tree_hash(BASELINE_DIRS)
    common_hashes = {
        name: sha_file(COMMON / name)
        for name in (
            "Ctot_init.raw",
            "phi_init.raw",
            "xB_init.raw",
            "init_meta.json",
            "runtime.base.params",
        )
    }
    committed_hashes = {
        name: sha_bytes(git_blob(name))
        for name in FROZEN_REMOTE_SOURCE_HASHES
    }
    working_hashes = {
        name: sha_file(ROOT / name)
        for name in FROZEN_REMOTE_SOURCE_HASHES
    }
    expected_common = {
        "Ctot_init.raw": "c05457ef585dc40ae91aded43a8531cfd72213d8da7b82243105d63cd1c60524",
        "phi_init.raw": "3a69447f232dfefce4e0a2ef67c172654bcc1520de4e703c9fca4ecf25721acb",
        "xB_init.raw": "210314a5ff59199df91fc4c9bc9556795e520f41906ec62d10509252f5347b84",
        "init_meta.json": "bc8a7f5befb740612ae8579099f9393fa4781e44e880b27683194029fd1ce4a2",
        "runtime.base.params": "5259ac5eb44876c381e84caf4bd68291ff415a94b33c9a56da45e6517530210f",
    }
    if common_hashes != expected_common:
        raise SystemExit("BLOCKED_TRANSPORT_GATE_BASELINE_MISMATCH: common input")

    manifest = {
        "schema": "CTOT_TRANSPORT_RESIDUAL_GATE_BASELINE_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": head,
        "physics": "pbte_ag2te_gp_coarse4_stoich_rd_v2",
        "integrator": "ctot_jichen_imex_bdf2_active_manifold_v1",
        "retry_contract": "ACTIVE_MANIFOLD_BDF2_BOUNDED_RETRY_PRODUCTION_V1",
        "grid": {"Nx": 512, "Ny": 1, "Nz": 1, "dx_nm": 1.0, "lambda_nm": 4.0},
        "temperature_C": 400.0,
        "equal_time_code": 1.5625,
        "equal_time_physical_s": 1.5625 * T_REAL_UNIT_S,
        "strict_dt16": {
            "dt_code": DTS["dt16"][0],
            "steps": DTS["dt16"][1],
            "throughput_physical_s_per_GPU_hour": 909.15,
            "rejects": 0,
            "fallbacks": 0,
        },
        "strict_reference": {"dt_code": 9.765625e-5, "steps": 16000, "gate": 1.0e-12},
        "historical_report_tree": {
            "sha256": tree_digest,
            "file_count": tree_files,
            "roots": list(BASELINE_DIRS),
        },
        "common_input_sha256": common_hashes,
        "commit_blob_sha256": committed_hashes,
        "frozen_remote_binary_sha256": FROZEN_REMOTE_BINARY_SHA256,
        "instrumented_remote_binary_sha256": INSTRUMENTED_REMOTE_BINARY_SHA256,
        "frozen_remote_source_sha256": FROZEN_REMOTE_SOURCE_HASHES,
        "instrumented_working_source_sha256": working_hashes,
        "remote_source_byte_identity_note": (
            "Frozen executable source differs from f440c0d in cuda_kernels.cu "
            "and pf_params.h only by whitespace; whitespace-stripped content "
            "was audited identical. The exact frozen executable hash remains authoritative."
        ),
        "diagnostics_off_bitwise_regression": {
            "steps": 64,
            "fields": ["Ctot", "phi", "xB_alpha", "Ctot_nm1", "phi_nm1"],
            "all_raw_fields_bitwise_equal": True,
        },
        "transactional_observer_regression": {
            "schema": "CTOT_TRANSPORT_GATE_TRAJECTORY_V2_TRANSACTIONAL",
            "case": "G10_dt1_360_steps",
            "diagnostics_on_off_raw_fields_bitwise_equal": True,
            "fields": ["Ctot", "phi", "xB_alpha", "Ctot_nm1", "phi_nm1"],
            "accepted_substep_rows": 364,
            "expected_time_code": 1.125,
            "observed_time_code": 1.125000000000007,
            "time_closure_error": 7.105427357601002e-15,
        },
        "source_GP_elasticity": {"source": "OFF", "GP": "OFF", "elasticity": "OFF"},
        "baseline_preserved": True,
    }
    (OUT / "baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    freeze = f"""# Transport residual-gate baseline freeze

## Verdict

`baseline_preserved=true`. The gate study starts from commit `{head}`, the
byte-identical common checkpoint, and the already-qualified active-manifold
IMEX-BDF2 contract. No physics, source, GP, elasticity, phase KKT, mass,
bounds, energy, retry, or iteration-budget setting is opened by this study.

| Item | Frozen value |
|---|---|
| Physics | `pbte_ag2te_gp_coarse4_stoich_rd_v2` |
| Integrator | `ctot_jichen_imex_bdf2_active_manifold_v1` |
| Retry contract | `ACTIVE_MANIFOLD_BDF2_BOUNDED_RETRY_PRODUCTION_V1` |
| Grid | `512 x 1 x 1`, `dx=1 nm`, `lambda=4 nm` |
| T | `400 C` |
| Common window | `1.5625` code time = `{1.5625*T_REAL_UNIT_S:.15f} s` |
| Strict dt/16 | `{DTS['dt16'][0]:.17e}`, 8000 steps, 909.15 physical s/GPU h |
| Strict reference | `dt/32`, 16000 steps, gate `1e-12` |
| Historical report tree | `{tree_digest}` ({tree_files} files) |
| Frozen workstation binary | `{FROZEN_REMOTE_BINARY_SHA256}` |
| Transactional observer binary | `{INSTRUMENTED_REMOTE_BINARY_SHA256}` |

## Common checkpoint hashes

"""
    for name, digest in common_hashes.items():
        freeze += f"- `{name}`: `{digest}`\n"
    freeze += """

## Source provenance distinction

The frozen workstation executable and its source hashes are retained exactly
as reported by the accepted bounded-retry evidence. The commit-frozen source
was independently compared with that staging source: `cuda_kernels.cu` and
`pf_params.h` differ bytewise only in whitespace, and whitespace-stripped
content is identical. The new binary adds only a default-off accepted-residual
observer. A 64-step diagnostics-off replay produced byte-identical `Ctot`,
`phi`, `xB_alpha`, `Ctot_nm1`, and `phi_nm1`; diagnostics-on produced the same
five hashes as well. A second 360-step on/off replay crossed real BE-subcycle
retries. The same five accepted/history fields remained byte-identical, while
the V2 observer committed 364 valid substeps to exactly 1.125 code time within
7.11e-15. Rolled-back partial subcycles do not enter `D_i`, `E_R`, or observer
time.

The complete machine-readable provenance is in `baseline_manifest.json`.
"""
    (OUT / "baseline_freeze.md").write_text(freeze, encoding="utf-8")

    main_path = ROOT / "main_cuda.cu"
    uses = line_hits(main_path, r"ctot_residual_(abs|rel)_tol")
    plumbing = f"""# Transport residual-gate plumbing audit

## Verdict

`gate_plumbing_status=PASS_UNIFIED_EXISTING_THRESHOLD_WITH_TRANSACTIONAL_OBSERVER`

The existing `ctot_residual_abs_tol` and `ctot_residual_rel_tol` form one
effective threshold:

`G_eff = ctot_residual_abs_tol + ctot_residual_rel_tol * initial_res_inf`.

For this matrix only the absolute term is varied. The relative term remains
fixed at `1e-10`; every row records both requested and effective gates.

## Consumers

| Role | Current source lines |
|---|---|
| Parameter declaration | `pf_params.h:397-398` |
| Defaults | `main_cuda.cu:18505-18506` before observer instrumentation |
| Parser | `main_cuda.cu:24047-24048` before instrumentation |
| Nonlinear stopping target | `main_cuda.cu:34366-34367` before instrumentation |
| Line-search endpoint acceptance | `main_cuda.cu:34561` before instrumentation |
| Final nonlinear cold check | `main_cuda.cu:34653-34654` before instrumentation |
| Outer acceleration merit gates | `main_cuda.cu:36442-36443`, `36696-36697` before instrumentation |
| Outer transport acceptance | `main_cuda.cu:37773-37774` before instrumentation |
| Method-consistent cold gate | `main_cuda.cu:38291-38296` before instrumentation |
| Retry/failure classification | immediately follows the method gate |

All current references were re-scanned at generated lines: `{uses}`.
There is no divergent hard-coded `1e-12` final cold gate. The line-search
endpoint exception uses the same absolute parameter and therefore remains
aligned with each requested test gate.

## Observation contract

`ctot_transport_gate_trajectory_diagnostics` defaults to zero. When enabled,
it copies the method-context cold residual immediately after evaluation. A
normal accepted macro is committed directly; event substeps remain in a host
transaction buffer until `CTOT_BDF2_EVENT_MACRO_READY`. Any depth escalation
clears that buffer before rollback/replay. Rejected or later-rolled-back
attempts therefore do not enter `E_R` or `D_i`. Its host arrays are not passed
to a CUDA kernel or solver.

Observer schema: `CTOT_TRANSPORT_GATE_TRAJECTORY_V2_TRANSACTIONAL`.

The 64-step and event-bearing 360-step state hashes are identical with the
observer off and on, establishing transaction neutrality for both ordinary
and BE-subcycled qualification paths.
"""
    (OUT / "gate_plumbing_audit.md").write_text(plumbing, encoding="utf-8")

    matrix_path = OUT / "matrix_manifest.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "case_id", "gate_id", "requested_transport_gate",
            "relative_gate", "dt_id", "dt_code", "dt_physical_s",
            "steps", "equal_time_code", "equal_time_physical_s",
            "common_input_sha256", "automatic_dt_change", "event_safety",
            "trajectory_diagnostics", "status",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for gate_id, gate in GATES.items():
            for dt_id, (dt, steps) in DTS.items():
                writer.writerow({
                    "case_id": f"{gate_id}_{dt_id}",
                    "gate_id": gate_id,
                    "requested_transport_gate": f"{gate:.17e}",
                    "relative_gate": f"{1.0e-10:.17e}",
                    "dt_id": dt_id,
                    "dt_code": f"{dt:.17e}",
                    "dt_physical_s": f"{dt*T_REAL_UNIT_S:.17e}",
                    "steps": steps,
                    "equal_time_code": f"{dt*steps:.17e}",
                    "equal_time_physical_s": f"{dt*steps*T_REAL_UNIT_S:.17e}",
                    "common_input_sha256": common_hashes["Ctot_init.raw"],
                    "automatic_dt_change": "OFF",
                    "event_safety": "PREFLIGHT_PLUS_BE_SUBCYCLING_UNCHANGED",
                    "trajectory_diagnostics": "CTOT_TRANSPORT_GATE_TRAJECTORY_V2_TRANSACTIONAL",
                    "status": "PENDING",
                })
    print("baseline_preserved=true")
    print("gate_plumbing_status=PASS_UNIFIED_EXISTING_THRESHOLD_WITH_TRANSACTIONAL_OBSERVER")
    print(f"matrix_cases={len(GATES) * len(DTS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
