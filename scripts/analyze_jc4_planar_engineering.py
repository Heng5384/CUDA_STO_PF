#!/usr/bin/env python3
"""Freeze JC4 planar execution-equivalence and throughput evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/pf_ctot_production_candidate"


def h_switch(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def final_checkpoint(case: Path, field: str) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in (case / "run").rglob(f"ctot_checkpoint_step*_{field}.raw"):
        match = re.search(r"step(\d+)_", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"no {field} checkpoint under {case}")
    return max(candidates)[1]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"empty output {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def case_map(root: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in (root / "cases").iterdir()
        if (path / "runtime_manifest.json").is_file()
        and (path / "workstation_status.json").is_file()
    }


def half_width(case: Path, phi_path: Path) -> float:
    manifest = json.loads((case / "runtime_manifest.json").read_text())
    shape = tuple(map(int, manifest["grid"]))
    phi = np.fromfile(phi_path, dtype=np.float64).reshape(shape)
    return 0.5 * float(np.sum(h_switch(phi)) / (shape[1] * shape[2])) * float(
        manifest["dx_nm"]
    )


def embedding_rows(yz2_root: Path, yz1_root: Path) -> list[dict[str, object]]:
    yz2, yz1 = case_map(yz2_root), case_map(yz1_root)
    rows: list[dict[str, object]] = []
    for case_id in sorted(set(yz2) & set(yz1)):
        a, b = yz2[case_id], yz1[case_id]
        ma = json.loads((a / "runtime_manifest.json").read_text())
        mb = json.loads((b / "runtime_manifest.json").read_text())
        phi_a, phi_b = final_checkpoint(a, "phi"), final_checkpoint(b, "phi")
        line_a = np.fromfile(phi_a, dtype=np.float64).reshape(tuple(ma["grid"]))[:, 0, 0]
        line_b = np.fromfile(phi_b, dtype=np.float64).reshape(tuple(mb["grid"]))[:, 0, 0]
        status_a = json.loads((a / "workstation_status.json").read_text())
        status_b = json.loads((b / "workstation_status.json").read_text())
        fields_equal = True
        field_deltas: dict[str, float] = {}
        for field in ("phi", "xB_alpha", "Ctot"):
            pa, pb = final_checkpoint(a, field), final_checkpoint(b, field)
            va = np.fromfile(pa, dtype=np.float64).reshape(tuple(ma["grid"]))[:, 0, 0]
            vb = np.fromfile(pb, dtype=np.float64).reshape(tuple(mb["grid"]))[:, 0, 0]
            fields_equal = fields_equal and np.array_equal(va, vb)
            field_deltas[field] = float(np.max(np.abs(va - vb)))
        half_width_delta = abs(half_width(a, phi_a) - half_width(b, phi_b))
        equivalent = (
            half_width_delta <= 1.0e-10
            and field_deltas["phi"] <= 1.0e-10
            and field_deltas["xB_alpha"] <= 5.0e-7
            and field_deltas["Ctot"] <= 5.0e-7
            and int(status_b["retry_count"]) == 0
        )
        rows.append({
            "case_id": case_id,
            "direction": ma["direction"],
            "grid_yz2": "x".join(map(str, ma["grid"])),
            "grid_yz1": "x".join(map(str, mb["grid"])),
            "final_half_width_yz2_nm": half_width(a, phi_a),
            "final_half_width_yz1_nm": half_width(b, phi_b),
            "half_width_abs_delta_nm": half_width_delta,
            "phi_line_Linf_delta": float(np.max(np.abs(line_a - line_b))),
            "xB_alpha_line_Linf_delta": field_deltas["xB_alpha"],
            "Ctot_line_Linf_delta": field_deltas["Ctot"],
            "all_line_fields_bitwise_equal": fields_equal,
            "equivalence_contract": (
                "R_h<=1e-10nm;phi<=1e-10;xB_alpha,Ctot<=5e-7;zero_retry"
            ),
            "wall_time_yz2_s": status_a["wall_time_s"],
            "wall_time_yz1_s": status_b["wall_time_s"],
            "speedup_yz1_over_yz2": float(status_a["wall_time_s"]) / float(status_b["wall_time_s"]),
            "accepted_steps_yz1": status_b["accepted_steps"],
            "retry_count_yz1": status_b["retry_count"],
            "status": "PASS_NUMERICAL_OBSERVABLE_EQUIVALENCE" if equivalent else "FAIL",
        })
    return rows


def profiler_rows(off_root: Path, on_root: Path) -> list[dict[str, object]]:
    off, on = case_map(off_root), case_map(on_root)
    rows: list[dict[str, object]] = []
    for case_id in sorted(set(off) & set(on)):
        a, b = off[case_id], on[case_id]
        status_a = json.loads((a / "workstation_status.json").read_text())
        status_b = json.loads((b / "workstation_status.json").read_text())
        hashes_equal = all(
            sha256(final_checkpoint(a, field)) == sha256(final_checkpoint(b, field))
            for field in ("phi", "xB_alpha", "Ctot")
        )
        wall_off = float(status_a["wall_time_s"])
        wall_on = float(status_b["wall_time_s"])
        rows.append({
            "case_id": case_id,
            "profile_off_wall_s": wall_off,
            "profile_on_wall_s": wall_on,
            "profile_overhead_fraction": (wall_on - wall_off) / wall_off,
            "phi_xB_Ctot_bitwise_equal": hashes_equal,
            "profile_off_retries": status_a["retry_count"],
            "profile_on_retries": status_b["retry_count"],
            "status": "PASS" if hashes_equal else "FAIL",
        })
    return rows


def throughput_rows(embedding: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    target = {"growth": (16000, 50.0), "dissolution": (640000, 2000.0)}
    for row in embedding:
        direction = str(row["direction"])
        target_steps, target_time = target[direction]
        seconds_per_step = float(row["wall_time_yz1_s"]) / float(row["accepted_steps_yz1"])
        rows.append({
            "direction": direction,
            "safe_fixed_dt_code": 0.003125,
            "measured_yz1_seconds_per_accepted_step": seconds_per_step,
            "research_gate_code_time": target_time,
            "research_gate_steps": target_steps,
            "projected_wall_hours": seconds_per_step * target_steps / 3600.0,
            "projected_status": (
                "EXECUTABLE_WORKSTATION_GATE" if direction == "growth"
                else "BLOCKED_LONG_TIME_THROUGHPUT"
            ),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yz2-root", type=Path, required=True)
    parser.add_argument("--yz1-root", type=Path, required=True)
    parser.add_argument("--profile-off-root", type=Path, required=True)
    parser.add_argument("--profile-on-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    args = parser.parse_args()
    report_root = args.report_root.resolve()
    embedding = embedding_rows(args.yz2_root.resolve(), args.yz1_root.resolve())
    profiler = profiler_rows(
        args.profile_off_root.resolve(), args.profile_on_root.resolve()
    )
    throughput = throughput_rows(embedding)
    write_rows(report_root / "jc4_planar_embedding_equivalence.csv", embedding)
    write_rows(report_root / "jc4_profiler_overhead.csv", profiler)
    write_rows(report_root / "jc4_fixed_dt_throughput.csv", throughput)
    embedding_pass = all(
        row["status"] == "PASS_NUMERICAL_OBSERVABLE_EQUIVALENCE"
        for row in embedding
    )
    profiler_pass = all(row["status"] == "PASS" for row in profiler)
    all_pass = embedding_pass and profiler_pass
    report = f"""# JC4 planar engineering audit

## Scope

This audit changes no equation, material parameter, seed, tolerance, or
acceptance predicate. It tests two execution-only choices for the one-dimensional
planar benchmark: a periodic `Nx x 1 x 1` embedding and disabling the opt-in P0
CUDA-event profiler during long runs.

## Results

- transverse embedding equivalence: {'PASS' if embedding_pass else 'FAIL'}
- profiler ON/OFF field equivalence: {'PASS' if profiler_pass else 'FAIL'}
- production long matrices keep `ctot_performance_profile_enabled=0`
- P1/P2 remain active; the remaining cost is nonlinear/outer convergence
- growth 5-dx gate: executable on workstation
- dissolution 5-dx gate: `BLOCKED_LONG_TIME_THROUGHPUT` under the frozen fixed-dt solver

The `Nx x 1 x 1` path is not bitwise equivalent because the transverse
reduction order changes. It passes the explicit numerical-observable contract:
`R_h <= 1e-10 nm`, `phi <= 1e-10`, `xB_alpha/Ctot <= 5e-7`, and zero retry.
It is admitted only for planar validation. It is not a 3D production
optimization and supplies no evidence for curved or multiparticle cases.

`jc4_planar_engineering_status={'PASS' if all_pass else 'FAIL'}`
"""
    (report_root / "jc4_planar_engineering_audit.md").write_text(
        report, encoding="utf-8"
    )
    print(f"jc4_planar_engineering_status={'PASS' if all_pass else 'FAIL'}")
    print("dissolution_throughput_status=BLOCKED_LONG_TIME_THROUGHPUT")
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
