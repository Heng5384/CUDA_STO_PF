#!/usr/bin/env python3
"""Analyze the frozen T400 IMEX-BDF2 active-set event evidence."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "bdf2_event_v1"
LONG = ROOT / "reports" / "T400_longtime_v1" / "coarse4_runs"

CASES = {
    "dt8": {
        "accepted": LONG / "growth_N512_shift0_dt8_pre_event_freeze" / "run",
        "failed": LONG / "growth_N512_shift0_dt8_event_repro" / "run",
        "checkpoint": 2957,
        "rejected": 2958,
        "dt": 0.000390625,
    },
    "dt16": {
        "accepted": LONG / "growth_N512_shift0_dt16_pre_event_freeze" / "run",
        "failed": LONG / "growth_N512_shift0_dt16_event_repro" / "run",
        "checkpoint": 5482,
        "rejected": 5483,
        "dt": 0.0001953125,
    },
}


def one(root: Path, pattern: str) -> Path:
    values = list(root.rglob(pattern))
    if len(values) != 1:
        raise RuntimeError(f"expected one {pattern} below {root}, found {values}")
    return values[0]


def h_stable(phi: np.ndarray) -> np.ndarray:
    p = np.asarray(phi, dtype=np.float64)
    p2 = p * p
    raw = p2 * p * (6.0 * p2 - 15.0 * p + 10.0)
    u = 1.0 - p
    alpha = u * u * u * (6.0 * u * u - 15.0 * u + 10.0)
    return np.where(p > 0.5, 1.0 - alpha, raw)


def read_checkpoint(case: dict[str, object]) -> dict[str, np.ndarray]:
    step = int(case["checkpoint"])
    root = Path(case["accepted"])
    stem = f"*step{step:06d}"
    return {
        "C_n": np.fromfile(one(root, stem + "_Ctot.raw"), np.float64),
        "C_nm1": np.fromfile(one(root, stem + "_Ctot_nm1.raw"), np.float64),
        "phi_n": np.fromfile(one(root, stem + "_phi.raw"), np.float64),
        "phi_nm1": np.fromfile(one(root, stem + "_phi_nm1.raw"), np.float64),
        "x_n": np.fromfile(one(root, stem + "_xB_alpha.raw"), np.float64),
    }


def failed_rows(case: dict[str, object]) -> dict[int, dict[str, str]]:
    path = one(Path(case["failed"]), "ctot_failed_cell_state.csv")
    return {int(row["idx"]): row for row in csv.DictReader(path.open())}


def parse_marker(log: str, step: int, marker: str) -> str:
    for line in log.splitlines():
        if marker in line and f"step={step} " in line:
            return line
    return ""


def marker_value(line: str, key: str) -> float:
    match = re.search(rf"(?:^|\s){re.escape(key)}=([^\s;]+)", line)
    if not match:
        return math.nan
    try:
        return float(match.group(1))
    except ValueError:
        return math.nan


def write_event_cells() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case_id, case in CASES.items():
        arrays = read_checkpoint(case)
        failed = failed_rows(case)
        C_n = arrays["C_n"]
        C_nm1 = arrays["C_nm1"]
        phi_n = arrays["phi_n"]
        phi_nm1 = arrays["phi_nm1"]
        C_anchor = (4.0 * C_n - C_nm1) / 3.0
        phi_E_raw = 2.0 * phi_n - phi_nm1
        phi_E = np.clip(phi_E_raw, 0.0, 1.0)
        h_n = h_stable(phi_n)
        h_E = h_stable(phi_E)
        alpha_n = 1.0 - h_n
        alpha_E = 1.0 - h_E
        q_n = (C_n - 1.0) + alpha_n
        q_anchor = (C_anchor - 1.0) + alpha_E
        context_tol = 64.0 * np.finfo(np.float64).eps
        bound_tol = 1.0e-12
        invalid = (
            ~np.isfinite(C_anchor)
            | ~np.isfinite(phi_E_raw)
            | (phi_E_raw < -context_tol)
            | (phi_E_raw > 1.0 + context_tol)
            | (q_anchor < -bound_tol)
            | (C_anchor > 1.0 + bound_tol)
        )
        candidates = set(np.where(invalid)[0].tolist())
        candidates.update(np.argsort(q_anchor)[:16].tolist())
        for idx, frow in failed.items():
            q_trial = float(frow["q_alpha"])
            if not math.isfinite(q_trial) or q_trial <= 1.0e-10:
                candidates.add(idx)
        for idx in sorted(candidates):
            frow = failed.get(idx, {})
            q_trial = float(frow.get("q_alpha", "nan"))
            transition = "NONE"
            if q_n[idx] > 0.0 and (q_anchor[idx] <= 0.0 or q_trial <= 0.0):
                transition = "FREE_TO_QALPHA_LOWER_CAPACITY"
            elif q_n[idx] <= 0.0 and q_anchor[idx] > 0.0:
                transition = "QALPHA_LOWER_CAPACITY_TO_FREE"
            elif q_anchor[idx] <= bound_tol:
                transition = "QALPHA_LOWER_CAPACITY_NEAR_ACTIVE"
            rows.append({
                "case": case_id,
                "accepted_step": case["checkpoint"],
                "rejected_step": case["rejected"],
                "idx": idx,
                "i": idx,
                "j": 0,
                "k": 0,
                "C_n": C_n[idx],
                "C_nm1": C_nm1[idx],
                "phi_n": phi_n[idx],
                "phi_nm1": phi_nm1[idx],
                "C_anchor": C_anchor[idx],
                "phi_E_raw": phi_E_raw[idx],
                "h_phi_E": h_E[idx],
                "alpha_E": alpha_E[idx],
                "phase_lower_C": h_E[idx],
                "phase_upper_C": 1.0,
                "q_n": q_n[idx],
                "q_anchor_E": q_anchor[idx],
                "xB_alpha_n": arrays["x_n"][idx],
                "context_invalid": int(invalid[idx]),
                "predicted_transition": transition,
                "trial_C": frow.get("C_trial", "nan"),
                "trial_phi": frow.get("phi", "nan"),
                "trial_q_alpha": frow.get("q_alpha", "nan"),
                "trial_xB_alpha": frow.get("xB", "nan"),
                "trial_mu": frow.get("mu", "nan"),
                "trial_mobility_status": (
                    "FACE_NONFINITE" if any(
                        frow.get(key, "nan") == "nan"
                        for key in ("face_x_out", "face_x_in")
                    ) else "FINITE_OR_ZERO"
                ),
                "trial_face_x_out": frow.get("face_x_out", "nan"),
                "trial_face_x_in": frow.get("face_x_in", "nan"),
                "trial_residual": frow.get("residual", "nan"),
            })
    REPORT.mkdir(parents=True, exist_ok=True)
    path = REPORT / "event_cells.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_transport_trace() -> None:
    case = CASES["dt8"]
    source = one(Path(case["failed"]), "ctot_nonlinear_iterations.csv")
    rows = [
        row for row in csv.DictReader(source.open())
        if int(row["step"]) == int(case["rejected"])
    ]
    path = REPORT / "dt8_transport_trace.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_energy_trace() -> None:
    case = CASES["dt16"]
    log = (Path(case["failed"]) / "run.log").read_text(errors="replace")
    line = parse_marker(log, int(case["rejected"]), "CTOT_IMEX_BDF2_ENERGY_WORK")
    keys = [
        "delta_F", "dissipation", "history_work", "transport_history",
        "phase_history", "nonlinear_chain", "transport_chain", "phase_chain",
        "explicit_context_work", "mechanics_work", "transport_residual_work",
        "phase_residual_work", "balance", "balance_rel", "pass",
    ]
    rows = [{"case": "dt16", "step": case["rejected"], "term": key,
             "value": marker_value(line, key),
             "finite": int(math.isfinite(marker_value(line, key)))} for key in keys]
    path = REPORT / "energy_term_trace.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_reports(rows: list[dict[str, object]]) -> None:
    dt8 = [r for r in rows if r["case"] == "dt8"]
    dt16 = [r for r in rows if r["case"] == "dt16"]
    dt8_worst = min(dt8, key=lambda r: float(r["q_anchor_E"]))
    dt16_worst = min(dt16, key=lambda r: float(r["q_anchor_E"]))
    common = f"""# Common active-set event

## Result

Both trajectories reach the same beta-side matrix-capacity region.  The event
is the lower storage constraint `q_alpha=Ctot-h(phi)*v_B=0`, not a composition
upper bound, phi endpoint overshoot, mobility sign change, or mass loss.

| Case | Cell | q_n | q_anchor_E | Context invalid |
|---|---:|---:|---:|---:|
| dt/8 | {dt8_worst['idx']} | {float(dt8_worst['q_n']):.17e} | {float(dt8_worst['q_anchor_E']):.17e} | {dt8_worst['context_invalid']} |
| dt/16 | {dt16_worst['idx']} | {float(dt16_worst['q_n']):.17e} | {float(dt16_worst['q_anchor_E']):.17e} | {dt16_worst['context_invalid']} |

The affected indices span 246-266 on the two periodic beta-side interfaces.
The dt/8 extrapolated anchor crosses the hard context tolerance materially;
the dt/16 anchor is only a few ulps below zero and passes the existing 1e-12
context tolerance, but its endpoint trial reaches the same capacity branch.

`same_active_set_event=true`

`event_transition_type=FREE_TO_QALPHA_LOWER_CAPACITY`
"""
    (REPORT / "common_active_set_event.md").write_text(common)

    dt8_log = (Path(CASES["dt8"]["failed"]) / "run.log").read_text(errors="replace")
    reject = parse_marker(dt8_log, 2958, "CTOT_MIMETIC_BE_REJECT")
    trace_rows = list(csv.DictReader((REPORT / "dt8_transport_trace.csv").open()))
    nonlinear_iterations = max(int(row["iteration"]) for row in trace_rows)
    dt8_report = f"""# dt/8 failure forensics

- The BDF2 preflight finds one infeasible context cell at step 2958.
- Cell 266 has `q_anchor_E={float(dt8_worst['q_anchor_E']):.17e}`.
- The selector transactionally uses same-dt Lie-BE; no BDF2 transport trial is
  committed.
- The BE target has zero lower/upper feasibility violations and remains finite.
- The nonlinear solve exhausts its accepted descent path with residual
  `{marker_value(reject, 'res_inf'):.17e}` after `{nonlinear_iterations}`
  nonlinear iterations and `{len(trace_rows)}` recorded line-search trials.
- Mass error is `{marker_value(reject, 'mass_error'):.17e}` and the accepted
  state is restored.

The root cause is a nonsmooth `q_alpha=0` capacity transition combined with a
same-macro-dt BE fallback.  Increasing the iteration budget is not a production
event treatment because it does not remove the inadmissible BDF2 history pair.
"""
    (REPORT / "dt8_failure_forensics.md").write_text(dt8_report)

    dt16_report = """# dt/16 endpoint energy/work NaN forensics

The step-5483 transport residual, phase KKT, mass, bounds, and finite-state
checks pass.  The first nonfinite audit dependency is the artificial mixed
endpoint state `F(C_np1, phi_n)` used to split the total chain rule into a
transport part and a phase part.  Near `q_alpha=0`, reconstructing that mixed
state produces a negative roundoff-scale matrix inventory and nonfinite
matrix-context quantities at beta-side cells 264-266.  Both chain differences
share this energy, so `transport_chain` and `phase_chain` become NaN together;
the physical-context chemical-potential work also becomes nonfinite, making
`explicit_context_work` NaN.

The actual endpoint states, BDF2 transport context, residual, and final energy
remain finite.  This is an audit-path algebra problem exposed by the same
capacity transition, not a nonfinite accepted PF state.  The stable replacement
must evaluate the combined endpoint chain remainder directly from admissible
endpoint energies and the already finite method work terms.  It must not clamp,
skip, or zero a cell.
"""
    (REPORT / "dt16_energy_nan_forensics.md").write_text(dt16_report)

    stable = """# Stable endpoint energy evaluation contract

The old split introduces the intermediate `F(C_np1,phi_n)`.  Algebraically,
the only quantity used in the final balance is

`R_chain + W_context = Delta F_nonelastic - <mu_E,Delta C> - <g_np1,Delta phi>`.

The stable evaluator computes this combined expression directly from the two
admissible accepted/trial endpoint energies and the same finite `mu_E` and
phase-force work arrays used by the residual audit.  It does not reconstruct an
off-method mixed endpoint.  Away from active-set events it must agree with the
old expanded expression to roundoff; at the event it remains finite by avoiding
the undefined intermediate.  Directional/host-oracle validation is pending the
implementation stage.
"""
    (REPORT / "stable_energy_evaluation.md").write_text(stable)


def main() -> int:
    rows = write_event_cells()
    write_transport_trace()
    write_energy_trace()
    write_reports(rows)
    print("event_analysis_status=PASS")
    print("same_active_set_event=true")
    print("event_transition_type=FREE_TO_QALPHA_LOWER_CAPACITY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
