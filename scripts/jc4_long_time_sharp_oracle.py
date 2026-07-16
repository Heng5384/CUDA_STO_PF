#!/usr/bin/env python3
"""Finite-box one-sided Stefan reference for JC4 long planar cases."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import Unit_Psedobinary as unit  # noqa: E402


@dataclass(frozen=True)
class SharpConfig:
    temperature_c: float = 400.0
    length_nm: float = 512.0
    initial_beta_half_width_nm: float = 10.0
    initial_matrix_xB: float = 0.05
    matrix_cells: int = 800
    final_time_s: float = 100.0
    output_points: int = 21
    rtol: float = 2.0e-10
    atol: float = 2.0e-12


class LongTimePlanarSharpOracle:
    """Conservative ALE finite-volume solver for one half-period."""

    def __init__(self, config: SharpConfig):
        if config.matrix_cells < 32:
            raise ValueError("matrix_cells must be at least 32")
        if not 0.0 < config.initial_beta_half_width_nm < 0.5 * config.length_nm:
            raise ValueError("invalid initial beta half width")
        self.config = config
        self.outer_nm = 0.5 * config.length_nm
        self.dy = 1.0 / config.matrix_cells
        self.x_eq = unit.xAg2Te_eq_from_T(config.temperature_c + 273.15)
        self.diffusivity_nm2_s = unit.D_Ag_in_PbTe_m2_per_s(
            config.temperature_c + 273.15
        ) * 1.0e18
        n = config.matrix_cells
        # q_i couples only to nearest q neighbors and the radius; the radius
        # equation depends on q_0 and radius.
        main = np.ones(n + 1, dtype=bool)
        lower = np.ones(n, dtype=bool)
        upper = np.ones(n, dtype=bool)
        pattern = diags((lower, main, upper), (-1, 0, 1), shape=(n + 1, n + 1),
                        dtype=bool).tolil()
        pattern[:n, n] = True
        pattern[n, 0] = True
        self.jac_sparsity = pattern.tocsr()

    def initial_state(self) -> np.ndarray:
        radius = self.config.initial_beta_half_width_nm
        length = self.outer_nm - radius
        q = np.full(
            self.config.matrix_cells,
            length * self.config.initial_matrix_xB,
            dtype=np.float64,
        )
        return np.concatenate((q, np.array([radius], dtype=np.float64)))

    def concentration(self, state: np.ndarray) -> np.ndarray:
        matrix_length = self.outer_nm - float(state[-1])
        return np.asarray(state[:-1]) / matrix_length

    def inventory(self, state: np.ndarray) -> float:
        return float(state[-1] + self.dy * np.sum(state[:-1]))

    def velocity(self, state: np.ndarray) -> float:
        matrix_length = self.outer_nm - float(state[-1])
        x = self.concentration(state)
        gradient = (x[0] - self.x_eq) / (0.5 * self.dy * matrix_length)
        return self.diffusivity_nm2_s * gradient / (1.0 - self.x_eq)

    def rhs(self, _time: float, state: np.ndarray) -> np.ndarray:
        radius = float(state[-1])
        matrix_length = self.outer_nm - radius
        if not 0.0 < radius < self.outer_nm:
            raise RuntimeError("sharp interface left the finite box")
        x = self.concentration(state)
        velocity = self.velocity(state)
        flux = np.zeros(self.config.matrix_cells + 1, dtype=np.float64)
        flux[0] = -velocity
        for face in range(1, self.config.matrix_cells):
            diffusive = -self.diffusivity_nm2_s * (
                x[face] - x[face - 1]
            ) / (self.dy * matrix_length)
            grid_velocity = (1.0 - face * self.dy) * velocity
            advected = x[face] if grid_velocity >= 0.0 else x[face - 1]
            flux[face] = diffusive - grid_velocity * advected
        flux[-1] = 0.0
        q_rate = -(flux[1:] - flux[:-1]) / self.dy
        return np.concatenate((q_rate, np.array([velocity])))

    def metrics(self, state: np.ndarray, time_s: float) -> dict[str, float]:
        x = self.concentration(state)
        return {
            "time_s": time_s,
            "radius_nm": float(state[-1]),
            "displacement_nm": float(
                state[-1] - self.config.initial_beta_half_width_nm
            ),
            "matrix_mean_xB": float(np.mean(x)),
            "matrix_min_xB": float(np.min(x)),
            "matrix_max_xB": float(np.max(x)),
            "interface_cell_xB": float(x[0]),
            "outer_cell_xB": float(x[-1]),
            "velocity_nm_s": self.velocity(state),
            "inventory_half_box": self.inventory(state),
        }

    def run(self) -> dict[str, object]:
        cfg = self.config
        initial = self.initial_state()
        times = np.linspace(0.0, cfg.final_time_s, cfg.output_points)
        solution = solve_ivp(
            self.rhs,
            (0.0, cfg.final_time_s),
            initial,
            method="BDF",
            t_eval=times,
            rtol=cfg.rtol,
            atol=cfg.atol,
            max_step=max(cfg.final_time_s / 500.0, 1.0e-12),
            jac_sparsity=self.jac_sparsity,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        records = [
            self.metrics(solution.y[:, i], float(time))
            for i, time in enumerate(solution.t)
        ]
        mass0 = self.inventory(initial)
        max_mass = max(
            abs(float(row["inventory_half_box"]) - mass0)
            / max(abs(mass0), 1.0)
            for row in records
        )
        return {"solution": solution, "records": records,
                "max_mass_error_rel": max_mass}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_matrix(matrix_root: Path) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for case in sorted((matrix_root / "cases").iterdir()):
        manifest_path = case / "runtime_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = SharpConfig(
            temperature_c=float(manifest["temperature_C"]),
            length_nm=float(manifest["grid"][0]) * float(manifest["dx_nm"]),
            initial_beta_half_width_nm=float(manifest["initial_beta_half_width_nm"]),
            initial_matrix_xB=float(manifest["matrix_xB"]),
            final_time_s=float(manifest["elapsed_s"]),
        )
        oracle = LongTimePlanarSharpOracle(config)
        result = oracle.run()
        records = result["records"]
        write_csv(case / "sharp_reference.csv", records)
        last = records[-1]
        summaries.append({
            "case_id": manifest["case_id"],
            "direction": manifest["direction"],
            "length_nm": config.length_nm,
            "initial_radius_nm": config.initial_beta_half_width_nm,
            "initial_matrix_xB": config.initial_matrix_xB,
            "xB_eq": oracle.x_eq,
            "elapsed_s": config.final_time_s,
            "sharp_final_radius_nm": last["radius_nm"],
            "sharp_displacement_nm": last["displacement_nm"],
            "sharp_final_matrix_mean_xB": last["matrix_mean_xB"],
            "sharp_mass_error_rel": result["max_mass_error_rel"],
            "direction_pass": (
                float(last["displacement_nm"]) > 0.0
                if manifest["direction"] == "growth"
                else float(last["displacement_nm"]) < 0.0
            ),
        })
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.matrix_root.resolve()
    summaries = run_matrix(root)
    if not summaries:
        raise RuntimeError("no prepared JC4 cases")
    write_csv(root / "sharp_reference_summary.csv", summaries)
    print(f"jc4_sharp_reference_cases={len(summaries)}")
    print(f"jc4_sharp_max_mass_error_rel={max(float(r['sharp_mass_error_rel']) for r in summaries):.17e}")
    for row in summaries:
        print(f"{row['case_id']}_sharp_displacement_nm={float(row['sharp_displacement_nm']):.12g}")
    return 0 if all(bool(row["direction_pass"]) for row in summaries) else 2


if __name__ == "__main__":
    raise SystemExit(main())
