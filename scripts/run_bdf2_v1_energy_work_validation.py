#!/usr/bin/env python3
"""Run low-cost workstation validation of the BDF2 discrete-work contract."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from prepare_bdf2_v1_startup_smoke import rewrite_meta, rewrite_params


DT = 0.000390625
STEPS = 4
ENERGY_RE = re.compile(r"CTOT_IMEX_BDF2_ENERGY_WORK (?P<body>.*)")


def h(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def tokens(body: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^ ]+)", body))


def write_meta(source: Path, destination: Path, shape: tuple[int, int, int],
               reference_type: str) -> None:
    rewrite_meta(source, destination)
    data = json.loads(destination.read_text(encoding="utf-8"))
    data.update(
        {
            "Nx": shape[0],
            "Ny": shape[1],
            "Nz": shape[2],
            "step": 0,
            "time_code": 0.0,
            "reference_type": reference_type,
        }
    )
    destination.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_smooth_state(case: Path) -> tuple[tuple[int, int, int], dict[str, Path]]:
    shape = (32, 32, 32)
    i, j, k = np.indices(shape, dtype=np.float64)
    phi = 0.18 + 0.04 * (
        np.sin(2.0 * math.pi * i / shape[0])
        * np.cos(2.0 * math.pi * j / shape[1])
        * np.cos(2.0 * math.pi * k / shape[2])
    )
    xb = 0.025 + 0.002 * (
        np.cos(2.0 * math.pi * i / shape[0])
        * np.sin(2.0 * math.pi * j / shape[1])
    )
    ctot = h(phi) + (1.0 - h(phi)) * xb
    paths = {}
    for name, field in (("phi", phi), ("xB", xb), ("Ctot", ctot)):
        path = case / f"{name}_init.raw"
        np.asarray(field, dtype=np.float64).tofile(path)
        paths[name] = path
    return shape, paths


def write_weak_elastic_state(case: Path) -> tuple[tuple[int, int, int], dict[str, Path]]:
    shape = (16, 16, 16)
    i, j, k = np.indices(shape, dtype=np.float64)
    mode = (
        np.sin(2.0 * math.pi * i / shape[0])
        * np.cos(2.0 * math.pi * j / shape[1])
        * np.cos(2.0 * math.pi * k / shape[2])
    )
    phi = 0.02 + 0.001 * mode
    xb = 0.03 + 1.0e-8 * mode
    ctot = h(phi) + (1.0 - h(phi)) * xb
    paths = {}
    for name, field in (("phi", phi), ("xB", xb), ("Ctot", ctot)):
        path = case / f"{name}_init.raw"
        np.asarray(field, dtype=np.float64).tofile(path)
        paths[name] = path
    return shape, paths


def write_dissolution_state(repo: Path, case: Path) -> tuple[tuple[int, int, int], dict[str, Path]]:
    frozen = repo / "runs/frozen_input"
    phi = np.fromfile(frozen / "ctot_checkpoint_step000054_phi.raw", np.float64)
    xb = np.full_like(phi, 0.005)
    ctot = h(phi) + (1.0 - h(phi)) * xb
    paths = {}
    for name, field in (("phi", phi), ("xB", xb), ("Ctot", ctot)):
        path = case / f"{name}_init.raw"
        field.tofile(path)
        paths[name] = path
    return (phi.size, 1, 1), paths


def frozen_state(repo: Path) -> tuple[tuple[int, int, int], dict[str, Path]]:
    root = repo / "runs/frozen_input"
    return (512, 1, 1), {
        "phi": root / "ctot_checkpoint_step000054_phi.raw",
        "xB": root / "ctot_checkpoint_step000054_xB_alpha.raw",
        "Ctot": root / "ctot_checkpoint_step000054_Ctot.raw",
    }


def stationary_state(repo: Path, case: Path) -> tuple[tuple[int, int, int], dict[str, Path]]:
    generated = case / "stationary_profile"
    command = [
        "python3", str(repo / "scripts/prepare_stationary_planar_slab_profile.py"),
        "--base-params",
        str(repo / "runs/preparation_v2/cases/T400_step655_lie_be_v2_dt_div1_full/runtime.params"),
        "--out-dir", str(generated), "--size", "32",
    ]
    subprocess.run(command, cwd=repo, check=True)
    return (32, 32, 32), {
        "phi": generated / "phi_init.raw",
        "xB": generated / "xB_init.raw",
        "Ctot": generated / "Ctot_init.raw",
    }


def run_case(repo: Path, root: Path, scenario: str, state: str,
             elasticity: bool) -> dict[str, object]:
    case = root / scenario
    case.mkdir(parents=True)
    if state == "smooth":
        shape, fields = write_smooth_state(case)
    elif state == "weak_elastic_smooth":
        shape, fields = write_weak_elastic_state(case)
    elif state == "stationary":
        shape, fields = stationary_state(repo, case)
    elif state == "growth":
        shape, fields = frozen_state(repo)
    elif state == "dissolution":
        shape, fields = write_dissolution_state(repo, case)
    else:
        raise ValueError(state)

    base = repo / "runs/preparation_v2/cases/T400_step655_lie_be_v2_dt_div1_full"
    params = case / "runtime.params"
    meta = case / "init_meta.json"
    rewrite_params(
        base / "runtime.params", params, scenario, DT,
        extra_replacements={
            "elastic_enabled": "1" if elasticity else "0",
            "ctot_elastic_validation_enabled": "1" if elasticity else "0",
        },
    )
    write_meta(base / "init_meta.json", meta, shape, f"bdf2_energy_{state}")
    command = [
        str(repo / "main_cuda"), *(str(value) for value in shape), f"{DT:.17e}",
        str(STEPS), str(STEPS), "1", "0", "--mode", "dynamics",
        "--pf-param-file", str(params), "--init-mode", "raw_fields",
        "--init-phi-raw", str(fields["phi"]),
        "--init-xB-raw", str(fields["xB"]),
        "--init-Ctot-raw", str(fields["Ctot"]),
        "--init-meta", str(meta), "--init-case-tag", scenario,
    ]
    (case / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    run = case / "run"
    run.mkdir()
    started = time.perf_counter()
    with (run / "run.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=run, stdout=log, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - started
    text = (run / "run.log").read_text(encoding="utf-8", errors="replace")
    energy_rows = [tokens(match.group("body")) for match in ENERGY_RE.finditer(text)]
    balance = [abs(float(row["balance_rel"])) for row in energy_rows]
    mechanics = [abs(float(row["mechanics_work"])) for row in energy_rows]
    result = {
        "scenario": scenario,
        "state_class": state,
        "grid": "x".join(str(value) for value in shape),
        "elasticity_enabled": elasticity,
        "dt_code": DT,
        "steps": STEPS,
        "exit_code": completed.returncode,
        "accepted_steps": text.count("CTOT_MIMETIC_BE_ACCEPT"),
        "rejected_steps": text.count("_REJECT step="),
        "bdf2_energy_rows": len(energy_rows),
        "bdf2_energy_pass_rows": sum(row.get("pass") == "1" for row in energy_rows),
        "max_balance_rel": max(balance, default=math.inf),
        "max_abs_mechanics_work": max(mechanics, default=math.nan),
        "max_mass_error": max(
            (abs(float(value)) for value in re.findall(r"mass_error=([^ ]+)", text)),
            default=math.inf,
        ),
        "clipping_count": sum(int(value) for value in re.findall(r"clip_count=(\d+)", text)),
        "projection_count": sum(
            int(value) for value in re.findall(r"projection_mass=(\d+)", text)
        ),
        # Startup provenance intentionally prints unavailable history values as
        # nan.  Runtime state failure has explicit nonfinite/fatal markers.
        "nan_or_inf": bool(
            re.search(r"(?:nonfinite|nan_inf|NaN/Inf)\s*[=:]\s*(?:1|true)", text,
                      re.IGNORECASE)
            or "nonfinite_state" in text
        ),
        "wall_seconds": wall,
        "log": str(run / "run.log"),
    }
    result["status"] = "PASS" if (
        result["exit_code"] == 0
        and result["accepted_steps"] == STEPS
        and result["rejected_steps"] == 0
        and result["bdf2_energy_rows"] == STEPS - 1
        and result["bdf2_energy_pass_rows"] == STEPS - 1
        and result["max_mass_error"] <= 1.0e-10
        and result["clipping_count"] == 0
        and result["projection_count"] == 0
        and not result["nan_or_inf"]
    ) else "FAIL"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = repo / "runs/bdf2_v1/energy_work_validation"
    if root.exists():
        if not args.overwrite:
            raise RuntimeError(f"refusing to overwrite {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    rows = [
        run_case(repo, root, "manufactured_smooth_elastic_off", "smooth", False),
        run_case(
            repo, root, "manufactured_weak_smooth_elastic_on",
            "weak_elastic_smooth", True,
        ),
        run_case(repo, root, "stationary_interface_elastic_off", "stationary", False),
        run_case(repo, root, "planar_growth_elastic_off", "growth", False),
        run_case(repo, root, "planar_dissolution_elastic_off", "dissolution", False),
    ]
    with (root / "energy_work_validation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "rows": rows,
        "all_pass": all(row["status"] == "PASS" for row in rows),
        "status": "PASS_BDF2_ENERGY_WORK_SCENARIOS"
        if all(row["status"] == "PASS" for row in rows)
        else "FAIL_BDF2_ENERGY_WORK_SCENARIOS",
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
