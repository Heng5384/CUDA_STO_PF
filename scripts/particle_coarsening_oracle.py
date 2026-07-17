#!/usr/bin/env python3
"""Minimal mass-conserving particle coarsening trend oracle.

The model is deliberately uncalibrated and cannot be used as a quantitative
PbTe/Ag2Te KWN model. It exists only to test a PF handoff ledger and qualitative
growth/dissolution/extinction ordering over a short overlap window.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class OracleParams:
    c_eq_inf: float
    capillary_length_nm: float
    kinetic_nm2_per_s: float
    dt_max_s: float
    relative_radius_step_limit: float = 0.005
    extinction_radius_nm: float = 0.05


def sphere_volume_nm3(radius_nm: np.ndarray) -> np.ndarray:
    return (4.0 * math.pi / 3.0) * radius_nm**3


def matrix_composition(
    total_inventory: float,
    gp_inventory: float,
    radii_nm: np.ndarray,
    box_cells: float,
    cell_volume_nm3: float,
    v_B: float,
) -> tuple[float, float, float]:
    beta_cells = float(np.sum(sphere_volume_nm3(radii_nm)) / cell_volume_nm3)
    beta_inventory = beta_cells * v_B
    matrix_cells = box_cells - beta_cells
    if matrix_cells <= 0.0:
        raise ValueError("particle volume exhausts matrix volume")
    matrix_inventory = total_inventory - gp_inventory - beta_inventory
    if matrix_inventory < -1.0e-12:
        raise ValueError("particle inventory exceeds total ledger")
    return matrix_inventory / matrix_cells, matrix_inventory, beta_inventory


def run_oracle(
    handoff: dict,
    params: OracleParams,
    end_time_s: float,
) -> tuple[list[dict], dict]:
    if handoff.get("schema") != "PF_PARTICLE_HANDOFF_LEDGER_V1":
        raise ValueError("unsupported handoff schema")
    radii = np.asarray(
        [p["equivalent_radius_nm"] for p in handoff["particles"]],
        dtype=np.float64,
    )
    particle_ids = np.asarray(
        [p["particle_id"] for p in handoff["particles"]], dtype=np.int64
    )
    dx_nm = float(handoff["dx_nm"])
    cell_volume = float(handoff["cell_volume_nm3"])
    box_cells = float(math.prod(handoff["grid_shape"]))
    v_B = float(handoff["v_B"])
    ledger = handoff["ledger"]
    total = float(ledger["total_inventory_with_GP"])
    gp_inventory = float(ledger["GP_inventory"])
    if not (params.c_eq_inf > 0.0 and params.kinetic_nm2_per_s > 0.0 and
            params.dt_max_s > 0.0 and end_time_s >= 0.0):
        raise ValueError("invalid oracle parameters")

    initial_radii = radii.copy()
    initial_count = radii.size
    trajectory: list[dict] = []
    t = 0.0
    step = 0

    def append_rows(event: str) -> None:
        c_matrix, matrix_inventory, beta_inventory = matrix_composition(
            total, gp_inventory, radii, box_cells, cell_volume, v_B
        )
        reconstructed = matrix_inventory + beta_inventory + gp_inventory
        rel = abs(reconstructed - total) / max(abs(total), np.finfo(float).tiny)
        for pid, radius in zip(particle_ids, radii):
            trajectory.append({
                "step": step,
                "time_s": t,
                "particle_id": int(pid),
                "radius_nm": float(radius),
                "matrix_xB": c_matrix,
                "matrix_inventory": matrix_inventory,
                "beta_inventory": beta_inventory,
                "GP_inventory": gp_inventory,
                "total_inventory": reconstructed,
                "mass_error_rel": rel,
                "event": event,
            })

    append_rows("initial")
    max_mass_error = 0.0
    extinction_order: list[int] = []
    while t < end_time_s and radii.size:
        c_matrix, _, _ = matrix_composition(
            total, gp_inventory, radii, box_cells, cell_volume, v_B
        )
        c_eq = params.c_eq_inf * np.exp(params.capillary_length_nm / radii)
        rates = params.kinetic_nm2_per_s * (c_matrix - c_eq) / radii
        nonzero = np.abs(rates) > 0.0
        dt = min(params.dt_max_s, end_time_s - t)
        if np.any(nonzero):
            dt = min(
                dt,
                float(np.min(
                    params.relative_radius_step_limit * radii[nonzero] /
                    np.abs(rates[nonzero])
                )),
            )
        if not (dt > 0.0 and math.isfinite(dt)):
            raise ValueError("oracle cannot form a finite positive step")
        trial = radii + dt * rates
        extinct = trial <= params.extinction_radius_nm
        event = "advance"
        if np.any(extinct):
            extinct_ids = particle_ids[extinct].tolist()
            extinction_order.extend(int(x) for x in extinct_ids)
            keep = ~extinct
            radii = trial[keep]
            particle_ids = particle_ids[keep]
            event = "extinction:" + ";".join(str(x) for x in extinct_ids)
        else:
            radii = trial
        t += dt
        step += 1
        append_rows(event)
        max_mass_error = max(max_mass_error, trajectory[-1]["mass_error_rel"])

    final_c, final_matrix, final_beta = matrix_composition(
        total, gp_inventory, radii, box_cells, cell_volume, v_B
    )
    summary = {
        "schema": "PARTICLE_COARSENING_TREND_ORACLE_V1",
        "status": "UNCALIBRATED_TREND_ORACLE_ONLY",
        "initial_particle_count": initial_count,
        "final_particle_count": int(radii.size),
        "initial_radii_nm": initial_radii.tolist(),
        "final_particle_ids": particle_ids.tolist(),
        "final_radii_nm": radii.tolist(),
        "extinction_order": extinction_order,
        "final_matrix_xB": final_c,
        "final_matrix_inventory": final_matrix,
        "final_beta_inventory": final_beta,
        "GP_inventory": gp_inventory,
        "max_mass_error_rel": max_mass_error,
        "end_time_s": t,
        "parameters": params.__dict__,
        "physical_claim_allowed": False,
    }
    return trajectory, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--end-time-s", type=float, required=True)
    parser.add_argument("--c-eq-inf", type=float, required=True)
    parser.add_argument("--capillary-length-nm", type=float, required=True)
    parser.add_argument("--kinetic-nm2-per-s", type=float, required=True)
    parser.add_argument("--dt-max-s", type=float, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    handoff = json.loads(args.handoff.read_text(encoding="utf-8"))
    trajectory, summary = run_oracle(
        handoff,
        OracleParams(args.c_eq_inf, args.capillary_length_nm,
                     args.kinetic_nm2_per_s, args.dt_max_s),
        args.end_time_s,
    )
    args.trajectory.parent.mkdir(parents=True, exist_ok=True)
    with args.trajectory.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trajectory[0]))
        writer.writeheader()
        writer.writerows(trajectory)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
