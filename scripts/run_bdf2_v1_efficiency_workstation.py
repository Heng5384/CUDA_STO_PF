#!/usr/bin/env python3
"""Measure Lie-BE and BDF2 throughput under one registered error envelope."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

from prepare_bdf2_v1_startup_smoke import rewrite_params


TIME_UNIT_S = 41.12958542455477
LIE_DT = 0.003125 / 128.0
LIE_STEPS = 1000


def only(root: Path, pattern: str) -> Path:
    paths = list(root.rglob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {pattern} below {root}: {paths}")
    return paths[0]


def run_lie(repo: Path, root: Path) -> dict[str, object]:
    case = root / "lie_be_dt_div128_1000"
    case.mkdir(parents=True)
    base = repo / "runs/preparation_v2/cases/T400_step655_lie_be_v2_dt_div1_full"
    params = case / "runtime.params"
    rewrite_params(
        base / "runtime.params", params, "lie_be_dt_div128_1000", LIE_DT,
        extra_replacements={
            "ctot_numerics_contract": "ctot_jichen_lie_be_v2",
            "ctot_split_defect_policy": "LIE_NO_POST_PHASE_POLISH",
            "ctot_step_max_retries": "0",
        },
    )
    frozen = repo / "runs/frozen_input"
    command = [
        str(repo / "main_cuda"), "512", "1", "1", f"{LIE_DT:.17e}",
        str(LIE_STEPS), str(LIE_STEPS), "1", "0", "--mode", "dynamics",
        "--pf-param-file", str(params), "--init-mode", "raw_fields",
        "--init-phi-raw", str(frozen / "ctot_checkpoint_step000054_phi.raw"),
        "--init-xB-raw", str(frozen / "ctot_checkpoint_step000054_xB_alpha.raw"),
        "--init-Ctot-raw", str(frozen / "ctot_checkpoint_step000054_Ctot.raw"),
        "--init-meta", str(base / "init_meta.json"),
        "--init-case-tag", "lie_be_dt_div128_1000",
    ]
    (case / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    run = case / "run"
    run.mkdir()
    started = time.perf_counter()
    with (run / "run.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=run, stdout=log, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - started
    text = (run / "run.log").read_text(encoding="utf-8", errors="replace")
    physical_time = LIE_DT * LIE_STEPS * TIME_UNIT_S
    peak = re.search(r"Peak estimate \(resident\+transient\) ~ same .*?\(([0-9.]+) MB\)", text)
    return {
        "method": "ctot_jichen_lie_be_v2",
        "dt_code": LIE_DT,
        "dt_physical_s": LIE_DT * TIME_UNIT_S,
        "steps": LIE_STEPS,
        "accepted_steps": text.count("CTOT_MIMETIC_BE_ACCEPT"),
        "reject_count": text.count("_REJECT step="),
        "wall_seconds": wall,
        "accepted_physical_time_s": physical_time,
        "throughput_physical_s_per_GPU_hour": physical_time * 3600.0 / wall,
        "peak_memory_MB": float(peak.group(1)) if peak else math.nan,
        "transport_solves": sum(
            int(value) for value in re.findall(r"transport_solves=(\d+)", text)
        ),
        "phase_solves": sum(
            int(value) for value in re.findall(r"phase_solves=(\d+)", text)
        ),
        "mechanics_solves": sum(
            int(value) for value in re.findall(r"mechanics_solves=(\d+)", text)
        ),
        "max_mass_error": max(
            (abs(float(value)) for value in re.findall(r"mass_error=([^ ]+)", text)),
            default=math.inf,
        ),
        "status": "PASS" if (
            completed.returncode == 0
            and text.count("CTOT_MIMETIC_BE_ACCEPT") == LIE_STEPS
            and text.count("_REJECT step=") == 0
        ) else "FAIL",
        "log": str(run / "run.log"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = repo / "runs/bdf2_v1/lie_equal_error_comparison"
    if root.exists():
        if not args.overwrite:
            raise RuntimeError(f"refusing to overwrite {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    lie = run_lie(repo, root)
    qualification = json.loads(
        (repo / "runs/bdf2_v1/fixed_dt_qualification/summary.json").read_text()
    )
    bdf2 = next(
        row for row in qualification["candidates"]
        if row["original_dt_factor"] == qualification["selected_original_dt_factor"]
    )
    equal_time = json.loads(
        (repo / "runs/bdf2_v1/fixed_dt_equal_time_analysis/summary.json").read_text()
    )
    lie_ladder = repo / "reports/lie_be_v2/step655_dt_ladder.csv"
    lie_error = math.nan
    for line in lie_ladder.read_text(encoding="utf-8").splitlines()[1:]:
        cells = line.split(",")
        if int(cells[0]) == 128:
            lie_error = float(cells[6])
            break
    bdf2_throughput = float(bdf2["accepted_physical_time_per_GPU_hour_s"])
    lie_throughput = float(lie["throughput_physical_s_per_GPU_hour"])
    summary = {
        "registered_increment_error_envelope": 0.005,
        "Lie_BE": lie,
        "Lie_BE_Ctot_increment_error": lie_error,
        "BDF2_selected_original_dt_factor": bdf2["original_dt_factor"],
        "BDF2_dt_code": bdf2["dt_code"],
        "BDF2_dt_physical_s": bdf2["dt_physical_s"],
        "BDF2_steps": bdf2["accepted_steps"],
        "BDF2_wall_seconds": bdf2["wall_seconds"],
        "BDF2_throughput_physical_s_per_GPU_hour": bdf2_throughput,
        "BDF2_equal_time_Ctot_increment_error": equal_time[
            "Ctot_increment_relative_L2_error"
        ],
        "BDF2_speedup_at_common_error_envelope": bdf2_throughput / lie_throughput,
        "same_hardware": True,
        "same_physics": True,
        "both_inside_error_envelope": (
            lie_error <= 0.005
            and equal_time["Ctot_increment_relative_L2_error"] <= 0.005
        ),
    }
    summary["status"] = "PASS_BDF2_EFFICIENCY" if (
        lie["status"] == "PASS"
        and summary["both_inside_error_envelope"]
        and bdf2_throughput > lie_throughput
    ) else "FAIL_BDF2_EFFICIENCY"
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS_BDF2_EFFICIENCY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
