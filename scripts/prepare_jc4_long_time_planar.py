#!/usr/bin/env python3
"""Prepare equal-inventory JC4 long-displacement planar oracle/runtime cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.jc4_ji_chen_1d_oracle import (  # noqa: E402
    FixedCtotJiChen1DOracle,
    JiChen1DOracle,
    OracleConfig,
)
from scripts.prepare_jc4_research_model import (  # noqa: E402
    canonical_hash,
    model_contract,
    parameter_row,
    runtime_params,
    sha256,
    write_params,
)


CASE_SPECS = {
    "growth": {
        "cells": 512,
        "half_width_nm": 10.0,
        "matrix_xB": 0.05,
        "full_time_code": 100.0,
        "full_dt_code": 0.10,
    },
    "dissolution": {
        "cells": 3072,
        "half_width_nm": 10.0,
        "matrix_xB": 1.0e-4,
        "full_time_code": 5000.0,
        "full_dt_code": 5.0,
    },
}


def provenance_hashes() -> tuple[str, str]:
    contract_hash = canonical_hash(model_contract())
    paths = [
        ROOT / "scripts/correction1_ji_chen_mapping.py",
        ROOT / "scripts/jc4_ji_chen_1d_oracle.py",
        ROOT / "Unit_Psedobinary.py",
        ROOT / "thermo_utils.h",
        ROOT / "phase_functions.h",
        Path("/Users/heng/Documents/STO_SM_with_elastic_notes.pdf"),
    ]
    reference_hash = canonical_hash({path.name: sha256(path) for path in paths})
    return contract_hash, reference_hash


def write_state(
    case: Path,
    oracle: FixedCtotJiChen1DOracle,
    transverse_cells: int = 2,
) -> dict[str, object]:
    if transverse_cells < 1:
        raise ValueError("transverse_cells must be positive")
    cfg = oracle.config
    state = oracle.initial_state()
    values = oracle.unpack(state)
    shape = (cfg.cells, transverse_cells, transverse_cells)
    for name, line in (
        ("phi", values["phi"]),
        ("xB", values["x"]),
        ("Ctot", values["C"]),
    ):
        np.broadcast_to(line[:, None, None], shape).copy().astype(np.float64).tofile(
            case / f"{name}_init.raw"
        )
    metadata = {
        "schema": "jc4_long_time_planar_initial_state_v1",
        "Nx": shape[0], "Ny": shape[1], "Nz": shape[2],
        "dx_nm": cfg.dx_nm,
        "interface_width_nm": cfg.lambda_nm,
        "dtype": "float64", "order": "C",
        "authoritative_state": "Ctot",
        "geometry": "periodic_planar_beta_slab_two_interfaces",
        "initial_beta_half_width_nm": cfg.initial_beta_half_width_nm,
        "initial_matrix_xB": cfg.initial_matrix_xB,
        "mean_xBtot": float(np.mean(values["C"])),
        "total_C_line_integral": float(np.sum(values["C"]) * cfg.dx_nm),
        "oracle_initial_source_identity_Linf": float(
            oracle.metrics(state, 0.0)["source_identity_Linf"]
        ),
    }
    (case / "init_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def write_oracle(case: Path, oracle: FixedCtotJiChen1DOracle) -> list[dict[str, float]]:
    records = oracle.run()["records"]
    path = case / "oracle_reference.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def build(
    root: Path,
    mode: str,
    run_host_reference: bool = True,
    performance_profile_enabled: bool = False,
    transverse_cells: int = 2,
    matrix_support_eps: float = 1.0e-10,
) -> dict[str, object]:
    if transverse_cells < 1:
        raise ValueError("transverse_cells must be positive")
    if not 0.0 < matrix_support_eps < 1.0:
        raise ValueError("matrix_support_eps must lie strictly between zero and one")
    root.mkdir(parents=True, exist_ok=True)
    contract_hash, reference_hash = provenance_hashes()
    cases: list[dict[str, object]] = []
    for direction, spec in CASE_SPECS.items():
        if mode == "pilot":
            dt_values = (0.10, 0.50, 1.0, 5.0, 10.0)
            final_time = 10.0 if direction == "growth" else 100.0
        elif mode == "stability":
            dt_values = (0.00625,)
            final_time = 0.625
        elif mode == "stability_refined":
            dt_values = (0.003125,)
            final_time = 0.625
        elif mode == "research_gate":
            dt_values = (0.003125,)
            # The independent sharp oracle reaches enough displacement here
            # that the measured coarse4 transfer error still leaves at least
            # five production cells of PF motion.
            final_time = 50.0 if direction == "growth" else 2000.0
        elif mode == "crossing_probe":
            dt_values = (0.003125,)
            final_time = 3.125
        else:
            dt_values = (float(spec["full_dt_code"]),)
            final_time = float(spec["full_time_code"])
        base_config = OracleConfig(
            temperature_c=400.0,
            cells=int(spec["cells"]),
            dx_nm=1.0,
            lambda_nm=4.0,
            initial_beta_half_width_nm=float(spec["half_width_nm"]),
            initial_matrix_xB=float(spec["matrix_xB"]),
            lphi_ratio=0.90,
            final_time_code=final_time,
            output_points=21,
            max_step_code=(0.2 if direction == "growth" else 10.0),
            rtol=2.0e-11,
            atol=2.0e-13,
        )
        initial_oracle = JiChen1DOracle(base_config)
        reference_oracle = FixedCtotJiChen1DOracle(base_config)
        oracle_records: list[dict[str, float]] | None = None
        for dt_code in dt_values:
            nsteps = max(1, int(math.ceil(final_time / dt_code)))
            dt_exact = final_time / nsteps
            case_id = f"T400_{direction}_{mode}_dt{str(dt_exact).replace('.', 'p')}"
            case = root / "cases" / case_id
            case.mkdir(parents=True, exist_ok=True)
            state_meta = write_state(case, initial_oracle, transverse_cells)
            if run_host_reference and oracle_records is None:
                oracle_records = write_oracle(case, reference_oracle)
            elif run_host_reference:
                source = root / "cases" / cases[-1]["case_id"] / "oracle_reference.csv"
                (case / "oracle_reference.csv").write_bytes(source.read_bytes())
            else:
                stale_reference = case / "oracle_reference.csv"
                if stale_reference.exists():
                    stale_reference.unlink()
            params = runtime_params(400.0, contract_hash, reference_hash)
            params.update({
                "dt": dt_exact,
                # Numerical convergence budget; does not alter the accepted operator.
                "ctot_nonlinear_max_iter": 500,
                "ctot_phase_linear_max_iter": 500,
                "ctot_outer_max_iter": 30,
                "ctot_step_max_retries": 8,
                "ctot_retry_shrink_factor": 0.5,
                "ctot_dt_min_ratio": 1.0 / 1024.0,
                "ctot_automatic_dt_growth": 0,
                "ctot_performance_profile_enabled": int(performance_profile_enabled),
                "ctot_matrix_support_eps": matrix_support_eps,
                "init_case_tag": case_id,
            })
            write_params(case / "runtime.params", params)
            initial_mean = float(state_meta["mean_xBtot"])
            equilibrium_h_fraction = (
                (initial_mean - initial_oracle.x_eq) / (1.0 - initial_oracle.x_eq)
            )
            equilibrium_half_width = 0.5 * base_config.length_nm * max(
                0.0, min(1.0, equilibrium_h_fraction)
            )
            equilibrium_displacement = (
                equilibrium_half_width - base_config.initial_beta_half_width_nm
            )
            final_oracle = oracle_records[-1] if oracle_records else None
            manifest = {
                "schema": "jc4_long_time_planar_runtime_case_v1",
                "case_id": case_id,
                "direction": direction,
                "study_mode": mode,
                "grid": [int(spec["cells"]), transverse_cells, transverse_cells],
                "grid_role": "one_dimensional_thin_slab_not_3d_production",
                "dx_nm": 1.0,
                "lambda_nm": 4.0,
                "lambda_over_dx": 4.0,
                "temperature_C": 400.0,
                "matrix_xB": float(spec["matrix_xB"]),
                "initial_beta_half_width_nm": float(spec["half_width_nm"]),
                "dt_code": dt_exact,
                "nsteps": nsteps,
                "final_time_code": final_time,
                "elapsed_s": final_time * float(
                    parameter_row(400.0)[0]["time_scale_s"]
                ),
                "out_every": max(1, nsteps // 20),
                "host_long_time_oracle_status": (
                    "COMPLETED" if oracle_records
                    else "NOT_RUN_ACTIVE_SET_BACKEND_PERFORMANCE_BLOCKED"
                ),
                "oracle_final_displacement_nm": (
                    float(final_oracle["beta_half_width_nm"]
                          - oracle_records[0]["beta_half_width_nm"])
                    if oracle_records else None
                ),
                "finite_box_equilibrium_half_width_nm": equilibrium_half_width,
                "finite_box_equilibrium_displacement_nm": equilibrium_displacement,
                "model_contract_hash": contract_hash,
                "reference_evidence_hash": reference_hash,
                "same_finite_box_and_total_inventory": True,
                "elasticity": "off",
                "GP": "off",
                "finite_interface_mode": "off",
                "performance_profile_enabled": performance_profile_enabled,
                "matrix_support_eps": matrix_support_eps,
                "a_M": 0.0,
                "initial_state": state_meta,
            }
            (case / "runtime_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            cases.append(manifest)
    top = {
        "schema": "jc4_long_time_planar_matrix_v1",
        "mode": mode,
        "cases": cases,
        "formal_3d_production": False,
        "cluster_used": False,
        "host_long_time_reference_requested": run_host_reference,
        "performance_profile_enabled": performance_profile_enabled,
        "transverse_cells": transverse_cells,
        "matrix_support_eps": matrix_support_eps,
    }
    (root / "manifest.json").write_text(
        json.dumps(top, indent=2) + "\n", encoding="utf-8"
    )
    return top


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "pilot", "stability", "stability_refined", "crossing_probe",
            "research_gate", "full"
        ),
        default="pilot",
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--skip-host-reference", action="store_true",
        help="prepare CUDA cases while fail-closed marking the long host backend unrun",
    )
    parser.add_argument(
        "--enable-performance-profile", action="store_true",
        help=(
            "enable the event-heavy P0 profiler for short engineering probes; "
            "long production matrices keep it disabled"
        ),
    )
    parser.add_argument(
        "--transverse-cells", type=int, default=2,
        help="periodic transverse cell count for the 1D planar embedding",
    )
    parser.add_argument("--matrix-support-eps", type=float, default=1.0e-10)
    args = parser.parse_args()
    root = args.root or ROOT / f"tmp/jc4_long_time_planar_{args.mode}"
    manifest = build(
        root.resolve(), args.mode,
        run_host_reference=not args.skip_host_reference,
        performance_profile_enabled=args.enable_performance_profile,
        transverse_cells=args.transverse_cells,
        matrix_support_eps=args.matrix_support_eps,
    )
    print(f"jc4_long_time_planar_mode={args.mode}")
    print(f"cases={len(manifest['cases'])}")
    for case in manifest["cases"]:
        print(
            f"{case['case_id']} grid={case['grid']} steps={case['nsteps']} "
            f"oracle_status={case['host_long_time_oracle_status']} "
            f"equilibrium_displacement_nm={case['finite_box_equilibrium_displacement_nm']:.9g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
