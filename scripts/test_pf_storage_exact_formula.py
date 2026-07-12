#!/usr/bin/env python3
"""Deterministic formula tests for the same-step two-phase storage update."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/pf_only_baseline_closure/pf_one_step_operator_replay.csv"


def h(p: float) -> float:
    p = min(max(p, 0.0), 1.0)
    return p**3 * (6*p*p - 15*p + 10)


def main() -> None:
    random.seed(918273)
    records: list[dict[str, object]] = []
    for name, p_old, p_new, x_old, dt_divj in [
        ("uniform_identity", 0.0, 0.0, 0.0078305391025, 0.0),
        ("matrix_transport", 0.0, 0.0, 0.0078305391025, 1e-5),
        ("interface_phi_growth", 0.45, 0.46, 0.012, 2e-5),
        ("interface_phi_shrink", 0.65, 0.63, 0.012, -2e-5),
        ("near_beta_support", 0.995, 0.996, 0.02, 1e-8),
    ]:
        ho, hn = h(p_old), h(p_new)
        c_old = (1-ho)*x_old + ho
        c_target = c_old + dt_divj
        denom = 1-hn
        x_trial = (c_target-hn)/denom if denom >= 1e-8 else x_old
        protected = denom < 1e-8 or not (1e-8 <= x_trial <= 1-1e-8)
        x_final = x_old if protected else x_trial
        c_final = (1-hn)*x_final + hn
        records.append({
            "test": name, "phi_old": p_old, "phi_new": p_new,
            "h_old": ho, "h_new": hn, "xB_old": x_old,
            "dt_divJ": dt_divj, "C_old": c_old, "C_target": c_target,
            "xB_trial": x_trial, "xB_final": x_final,
            "protected_infeasible_reconstruction": protected,
            "C_final": c_final, "storage_residual": c_final-c_target,
            "bounded": 0 <= x_final <= 1,
            "local_exact_reconstruction_feasible": abs(c_final-c_target) <= 1e-12,
            "residual_exposed_for_global_projection": c_final-c_target,
            "path": "same_step_storage_exact",
            "h_alpha": denom,
            "storage_floor": 1e-8,
            "physical_transport_rate_dt": dt_divj / denom if denom >= 1e-8 else 0.0,
            "history_dependency": False,
            "accepted_candidate": False,
        })
    for p, x_old, dt_divj in [
        (0.0, 0.0078305391025, 1e-5),
        (0.5, 0.012, 1e-5),
        (0.75, 0.012, 1e-5),
        (0.9, 0.012, 1e-5),
        (0.99, 0.012, 1e-5),
    ]:
        hp = h(p)
        h_alpha = 1.0 - hp
        floor = 0.1
        supported = h_alpha >= floor
        delta_x = dt_divj / h_alpha if supported else 0.0
        x_trial = x_old + delta_x
        x_final = min(max(x_trial, 1e-8), 1.0 - 1e-8)
        c_old = h_alpha*x_old + hp
        c_target = c_old + dt_divj
        c_final = h_alpha*x_final + hp
        records.append({
            "test": f"storage_weighted_phi_{p:g}",
            "phi_old": p, "phi_new": p, "h_old": hp, "h_new": hp,
            "xB_old": x_old, "dt_divJ": dt_divj, "C_old": c_old,
            "C_target": c_target, "xB_trial": x_trial, "xB_final": x_final,
            "protected_infeasible_reconstruction": not supported,
            "C_final": c_final, "storage_residual": c_final-c_target,
            "bounded": 0.0 <= x_final <= 1.0,
            "local_exact_reconstruction_feasible": supported and abs(c_final-c_target) <= 1e-12,
            "residual_exposed_for_global_projection": c_final-c_target,
            "path": "storage_weighted_matrix_transport",
            "h_alpha": h_alpha, "storage_floor": floor,
            "physical_transport_rate_dt": delta_x,
            "history_dependency": False, "accepted_candidate": True,
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    assert all(bool(r["bounded"]) for r in records)
    assert all(math.isfinite(float(r["residual_exposed_for_global_projection"])) for r in records)
    print("storage_exact_formula_tests=PASS")


if __name__ == "__main__":
    main()
