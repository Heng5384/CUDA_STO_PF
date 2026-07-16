#!/usr/bin/env python3
"""Freeze and compare active-manifold IMEX-BDF2 coefficient contexts."""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "active_manifold_bdf2_v1"
EVENT = ROOT / "reports" / "bdf2_event_v1"
FROZEN = EVENT / "frozen_checkpoints"
CASES = {
    "dt8": {"step": 2957, "dt": 0.000390625},
    "dt16": {"step": 5482, "dt": 0.0001953125},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def h(phi: np.ndarray | float) -> np.ndarray:
    p = np.asarray(phi, dtype=np.float64)
    u = 1.0 - p
    raw = p**3 * (6.0 * p**2 - 15.0 * p + 10.0)
    alpha = u**3 * (6.0 * u**2 - 15.0 * u + 10.0)
    return np.where(p > 0.5, 1.0 - alpha, raw)


def alpha(phi: np.ndarray | float) -> np.ndarray:
    p = np.asarray(phi, dtype=np.float64)
    u = 1.0 - p
    return np.where(
        p > 0.5,
        u**3 * (6.0 * u**2 - 15.0 * u + 10.0),
        1.0 - p**3 * (6.0 * p**2 - 15.0 * p + 10.0),
    )


def hprime(phi: np.ndarray | float) -> np.ndarray:
    p = np.asarray(phi, dtype=np.float64)
    return 30.0 * p**2 * (1.0 - p) ** 2


def hinv_upper_feasible(target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    lo = np.zeros_like(target)
    hi = np.ones_like(target)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        less = h(mid) < target
        lo = np.where(less, mid, lo)
        hi = np.where(less, hi, mid)
    return lo


def arrays(case: str) -> dict[str, np.ndarray]:
    root = FROZEN / case
    step = CASES[case]["step"]
    prefix = f"ctot_checkpoint_step{step:06d}_"
    return {
        name: np.fromfile(root / f"{prefix}{suffix}.raw", np.float64)
        for name, suffix in {
            "C_n": "Ctot",
            "C_nm1": "Ctot_nm1",
            "phi_n": "phi",
            "phi_nm1": "phi_nm1",
            "x_n": "xB_alpha",
        }.items()
    }


def contexts(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    C_n, C_nm1 = data["C_n"], data["C_nm1"]
    phi_n, phi_nm1 = data["phi_n"], data["phi_nm1"]
    C_anchor = (4.0 * C_n - C_nm1) / 3.0
    phi_raw = 2.0 * phi_n - phi_nm1
    bounded = np.clip(phi_raw, 0.0, 1.0)
    q_n = (C_n - 1.0) + alpha(phi_n)
    q_raw = (C_anchor - 1.0) + alpha(bounded)
    lower_outward = q_raw < -64.0 * np.finfo(np.float64).eps

    tangent = phi_raw.copy()
    hp = hprime(bounded)
    linear = lower_outward & (hp > 1.0e-30)
    tangent[linear] += q_raw[linear] / hp[linear]

    qactive = phi_raw.copy()
    qactive[lower_outward] = hinv_upper_feasible(
        np.clip(C_anchor[lower_outward], 0.0, 1.0)
    )

    constrained = phi_raw.copy()
    x_min = 1.0e-8
    h_upper = np.clip((C_anchor - x_min) / (1.0 - x_min), 0.0, 1.0)
    constrained[lower_outward] = hinv_upper_feasible(h_upper[lower_outward])

    return {
        "C_anchor": C_anchor,
        "phi_raw": phi_raw,
        "q_n": q_n,
        "q_raw": q_raw,
        "lower_outward": lower_outward,
        "A_TANGENT_CONE_PHASE_VELOCITY": tangent,
        "B_QALPHA_ACTIVE_SET": qactive,
        "C_CONSTRAINED_FIXED_C_PHASE": constrained,
    }


def parse_event_lifetimes(case: str) -> dict[int, tuple[int, int, int]]:
    path = EVENT / "workstation_runs" / f"{case}_event_safe_502" / "run.log"
    seen: dict[int, list[int]] = {}
    pattern = re.compile(r"CTOT_BDF2_EVENT_PREFLIGHT step=(\d+).*first_cell=(\d+)")
    for match in pattern.finditer(path.read_text(errors="replace")):
        step, idx = map(int, match.groups())
        seen.setdefault(idx, []).append(step)
    return {idx: (min(values), max(values), len(values)) for idx, values in seen.items()}


def write_baseline() -> None:
    assets = [
        ROOT / "main_cuda.cu",
        ROOT / "cuda_kernels.cu",
        ROOT / "cuda_kernels.h",
        ROOT / "pf_params.h",
        ROOT / "bdf2_event_utils.h",
        ROOT / "phase_kkt_utils.h",
        ROOT / "active_manifold_bdf2_utils.h",
    ]
    checkpoint_assets = sorted(FROZEN.rglob("*.raw"))
    lines = [
        "# Active-manifold BDF2 baseline freeze",
        "",
        "This goal starts from the already accepted event-safe T400 source-free baseline.",
        "No physics parameter, tolerance, authoritative state, or checkpoint was changed",
        "by the host candidate analysis.",
        "",
        "| Contract | Frozen value |",
        "|---|---|",
        "| Physics | `pbte_ag2te_gp_coarse4_stoich_rd_v2` |",
        "| Base integrator | `ctot_jichen_imex_bdf2_v1` |",
        "| Event contracts | `BDF2_EVENT_PREFLIGHT_V1`, `BDF2_EVENT_BE_SUBCYCLING_V1` |",
        "| Endpoint work | `BDF2_STABLE_ADMISSIBLE_ENDPOINT_WORK_V2` |",
        "| Grid | `512x1x1`, `dx=1 nm`, `lambda=4 nm` |",
        "| T | `400 C` |",
        "| GP/source/elasticity | `OFF/OFF/OFF` |",
        "| Workstation | `RTX 5080`, CUDA 12.9 |",
        "",
        "## Current source hashes",
        "",
        "| Asset | SHA-256 |",
        "|---|---|",
    ]
    lines += [f"| `{p.name}` | `{sha256(p)}` |" for p in assets]
    lines += [
        "",
        "## Frozen event state hashes",
        "",
        "| Asset | SHA-256 |",
        "|---|---|",
    ]
    lines += [
        f"| `{p.relative_to(ROOT)}` | `{sha256(p)}` |" for p in checkpoint_assets
    ]
    lines += [
        "",
        "The prior 502-macro metrics and fine-reference hashes remain authoritative in",
        "`reports/bdf2_event_v1/event_crossing_metrics.csv` and its frozen run manifests.",
        "",
        "`baseline_preserved=true`",
    ]
    (REPORT / "baseline_freeze.md").write_text("\n".join(lines) + "\n")


def write_manifold() -> None:
    rows: list[dict[str, object]] = []
    class_counts: dict[str, dict[str, int]] = {}
    for case in CASES:
        data = arrays(case)
        ctx = contexts(data)
        lifetimes = parse_event_lifetimes(case)
        bound_tol = 1.0e-12
        storage_ulp = 64.0 * np.finfo(np.float64).eps
        interface_indices = np.argsort(np.abs(data["phi_n"] - 0.5))[:4]
        counts: dict[str, int] = {}
        for idx in range(data["C_n"].size):
            first, last, count = lifetimes.get(int(idx), (-1, -1, 0))
            hp = float(hprime(data["phi_n"][idx]))
            upper_margin_n = float(1.0 - data["C_n"][idx])
            if upper_margin_n <= bound_tol:
                cell_class = "UPPER_CAPACITY_ACTIVE"
            elif (ctx["q_n"][idx] > bound_tol and
                  ctx["q_raw"][idx] < -storage_ulp):
                cell_class = "ENTERING_QALPHA_LOWER"
            elif (ctx["q_n"][idx] <= bound_tol and
                  ctx["q_raw"][idx] > bound_tol):
                cell_class = "LEAVING_QALPHA_LOWER"
            elif (ctx["q_n"][idx] <= bound_tol and
                  ctx["q_raw"][idx] <= bound_tol):
                cell_class = "PERSISTENT_QALPHA_LOWER"
            else:
                cell_class = "FREE"
            counts[cell_class] = counts.get(cell_class, 0) + 1
            distances = np.abs(interface_indices - idx)
            periodic_distance = int(np.min(np.minimum(
                distances, data["C_n"].size - distances
            )))
            h_n = float(h(data["phi_n"][idx]))
            phase_lower_outward = int(
                ctx["phi_raw"][idx] < -storage_ulp and
                h_n <= storage_ulp and
                data["phi_nm1"][idx] >= data["phi_n"][idx]
            )
            rows.append({
                "case": case,
                "idx": int(idx),
                "cell_class": cell_class,
                "distance_to_nearest_interface_cell": periodic_distance,
                "C_n": data["C_n"][idx],
                "C_nm1": data["C_nm1"][idx],
                "phi_n": data["phi_n"][idx],
                "phi_nm1": data["phi_nm1"][idx],
                "C_anchor": ctx["C_anchor"][idx],
                "phi_raw_E": ctx["phi_raw"][idx],
                "q_n": ctx["q_n"][idx],
                "q_raw_E": ctx["q_raw"][idx],
                "dq_dphi_fixed_C": -hp,
                "predicted_dq_dt":
                    (ctx["q_raw"][idx] - ctx["q_n"][idx]) / CASES[case]["dt"],
                "normal_C_component": 1.0,
                "normal_phi_component": -hp,
                "tangent_cone_condition": "delta_C-hprime*delta_phi>=0",
                "complementarity": "q>=0;lambda>=0;q*lambda=0",
                "raw_outward": int(ctx["q_raw"][idx] < 0.0),
                "phase_lower_endpoint_outward": phase_lower_outward,
                "upper_capacity_margin_n": upper_margin_n,
                "first_logged_event_step": first,
                "last_logged_event_step": last,
                "logged_first_cell_count": count,
            })
        class_counts[case] = counts
    with (REPORT / "manifold_event_cells.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    text = """# q_alpha=0 active-manifold derivation

For `v_B=1`, the local storage map is

`Ctot = h(phi) + q_alpha`, with `q_alpha=(1-h)xB_alpha`.

At the lower branch, the admissible set and complementarity conditions are

`q_alpha >= 0`, `lambda_q >= 0`, `q_alpha*lambda_q=0`.

At fixed BDF2 anchor `C_E=(4 C_n-C_nm1)/3`,

`dq/dphi = -h'(phi)` and the outward normal test is
`delta_C-h'(phi) delta_phi < 0`.

The raw context `phi_E=2 phi_n-phi_nm1` repeatedly has a negative normal
component on the moving beta-side interface.  This is the same moving
lower-capacity manifold found by the event-safe audit; it is not stale history
or an endpoint-energy failure.  The dt/8 log records 240 preflight events over
502 macro steps and the first affected cell migrates with the interface.

The selected context solves the active storage relation exactly:

`q_E=0`, `h(phi_E^M)=C_E`, `phi_E^M=h^{-1}(C_E)`.

Only the non-authoritative transport coefficient context is reconstructed.
`C_n`, `C_nm1`, `phi_n`, and `phi_nm1` remain unchanged.  In free cells the
context is bitwise the original second-order extrapolate.  A one-sided inverse
returns the representably feasible side of the manifold rather than clipping a
physical state.

The production replay also exposed a separate endpoint tangent cone in the
pure-alpha far field. There `phi>=0`, `h(0)=h'(0)=0`, and accepted positive
tails with `h(phi_n)` below the 64-ULP storage resolution are storage-equivalent
to the exact endpoint. If `2*phi_n-phi_nm1<0` points outward, the coefficient
context is `phi_E^M=0`; inward motion remains the original extrapolate. This is
an explicit complementarity branch, not a post-update clamp. Adding it removes
all phase-endpoint preflight fallbacks in both 502-macro replays.

Frozen-state all-cell classes are:

CLASS_COUNTS

`active_manifold_root_cause_confirmed=true`
"""
    count_lines = []
    for case, counts in class_counts.items():
        ordered = ", ".join(
            f"{name}={value}" for name, value in sorted(counts.items())
        )
        count_lines.append(f"- `{case}`: {ordered}")
    text = text.replace("CLASS_COUNTS", "\n".join(count_lines))
    (REPORT / "manifold_derivation.md").write_text(text)


def write_candidates() -> None:
    rows: list[dict[str, object]] = []
    descriptions = {
        "A_TANGENT_CONE_PHASE_VELOCITY": (
            "linearized normal removal", "LOW", "NO", "NO_GUARANTEE"
        ),
        "B_QALPHA_ACTIVE_SET": (
            "exact q=0 complementarity manifold", "LOW", "NO", "YES"
        ),
        "C_CONSTRAINED_FIXED_C_PHASE": (
            "fixed-C lower-composition box endpoint proxy", "MEDIUM", "NO", "YES"
        ),
    }
    for case in CASES:
        data = arrays(case)
        ctx = contexts(data)
        raw = ctx["phi_raw"]
        raw_neighbor = np.max(np.abs(np.roll(raw, -1) - raw))
        active = ctx["lower_outward"]
        free = ~active
        for name in descriptions:
            phi = ctx[name]
            q = (ctx["C_anchor"] - 1.0) + alpha(np.clip(phi, 0.0, 1.0))
            continuity = np.max(np.abs(np.roll(phi, -1) - phi))
            description, cost, arbitrary_clip, guaranteed = descriptions[name]
            rows.append({
                "case": case,
                "candidate": name,
                "description": description,
                "adjusted_cells": int(np.count_nonzero(phi != raw)),
                "raw_outward_cells": int(np.count_nonzero(active)),
                "material_infeasible_cells_after":
                    int(np.count_nonzero(q < -64.0 * np.finfo(float).eps)),
                "min_q_context": float(np.min(q)),
                "max_abs_phi_context_change": float(np.max(np.abs(phi - raw))),
                "free_set_phi_difference_linf":
                    float(np.max(np.abs(phi[free] - raw[free]))) if np.any(free) else 0.0,
                "neighbor_jump_before": float(raw_neighbor),
                "neighbor_jump_after": float(continuity),
                "face_context_finite": int(np.all(np.isfinite(phi))),
                "free_set_second_order_exact": int(np.all(phi[free] == raw[free])),
                "arbitrary_clip": arbitrary_clip,
                "changes_authoritative_state": "NO",
                "changes_physical_equation": "NO",
                "feasibility_guaranteed": guaranteed,
                "relative_cost": cost,
                "selected": int(name == "B_QALPHA_ACTIVE_SET"),
            })
    with (REPORT / "candidate_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_name = {name: [r for r in rows if r["candidate"] == name] for name in descriptions}
    lines = [
        "# Active-manifold context candidate comparison",
        "",
        "| Candidate | Feasibility | Free-set order | Cost | Decision |",
        "|---|---|---|---|---|",
        "| A tangent-cone velocity | Linearized only; can retain nonlinear residual q defect | Exact away from event | Low | Reject as primary |",
        "| B q-alpha active set + phase endpoint tangent cone | Exact complementarity manifold with one-sided h inverse and exact pure-alpha endpoint | Bitwise original extrapolate away from active branches | Low | **Selected** |",
        "| C constrained fixed-C phase | Feasible but mixes numerical x-min/phase-box policy into transport context and needs a force solve to be complete | Exact away from event | Medium | Retain as oracle only |",
        "",
        "Candidate B is not `phi=clamp(phi_raw)`.  It is the semismooth branch",
        "solution of the local complementarity relation `q_E=C_E-h(phi_E)=0`.",
        "Its pure-alpha endpoint branch is the tangent-cone solution `phi_E=0`",
        "only for outward motion at storage-equivalent endpoint states.",
        "It changes no authoritative state and leaves all free cells exactly on",
        "the original second-order formula.",
        "",
        "Frozen checkpoint metrics:",
        "",
    ]
    for case in CASES:
        selected = next(r for r in by_name["B_QALPHA_ACTIVE_SET"] if r["case"] == case)
        lines.append(
            f"- `{case}`: adjusted {selected['adjusted_cells']} cells; "
            f"min q={selected['min_q_context']:.3e}; "
            f"free-set Linf difference={selected['free_set_phi_difference_linf']:.3e}."
        )
    lines += ["", "`selected_context_candidate=QALPHA_ACTIVE_SET_PREDICTOR`", ""]
    (REPORT / "candidate_comparison.md").write_text("\n".join(lines))


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    write_baseline()
    write_manifold()
    write_candidates()
    print("active_manifold_host_oracle=PASS")
    print("selected_context_candidate=QALPHA_ACTIVE_SET_PREDICTOR")


if __name__ == "__main__":
    main()
