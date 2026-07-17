#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


DT_RE = re.compile(r"CTOT_VARIABLE_BDF2_CONTROLLER_COMMIT.*accepted_dt=([^ ]+)")
ERR_RE = re.compile(
    r"CTOT_VARIABLE_BDF2_ERROR.*combined=([^ ]+).*accept=([01])")
SUMMARY_RE = re.compile(
    r"macro_steps=(\d+).*internal_trial_rejects=(\d+).*fallback_macros=(\d+).*"
    r"reject_trial_wall_fraction=([^ ]+)")


def h(phi):
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def read_raw(path):
    return np.fromfile(path, dtype=np.float64)


def locate(case_dir, stem):
    matches = sorted(case_dir.glob(f"**/ctot_checkpoint*_{stem}.raw"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {stem} raw file under {case_dir}, got {matches}")
    return matches[0]


def locate_meta(case_dir):
    matches = sorted(case_dir.glob("**/ctot_checkpoint*_meta.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one checkpoint meta under {case_dir}")
    return matches[0]


def read_meta(path):
    text = path.read_text(encoding="utf-8")
    return json.loads(text.replace(": nan", ": null"))


def crossings(phi, level=0.5):
    out = []
    n = len(phi)
    for i in range(n):
        j = (i + 1) % n
        a = phi[i] - level
        b = phi[j] - level
        if a == 0.0:
            out.append(float(i))
        elif a * b < 0.0:
            frac = -a / (b - a)
            out.append((i + frac) % n)
    return sorted(out)


def periodic_distance(a, b, n):
    d = abs(a - b)
    return min(d, n - d)


def crossing_error(candidate, reference):
    if len(candidate) != len(reference) or not reference:
        return math.inf
    remaining = list(candidate)
    errors = []
    for ref in reference:
        idx = min(range(len(remaining)),
                  key=lambda k: periodic_distance(remaining[k], ref, 512))
        errors.append(periodic_distance(remaining.pop(idx), ref, 512))
    return max(errors)


def norm_error(candidate, reference, initial):
    delta_ref = reference - initial
    diff = candidate - reference
    l2 = np.linalg.norm(diff) / max(np.linalg.norm(delta_ref), 1e-300)
    linf = np.max(np.abs(diff)) / max(np.max(np.abs(delta_ref)), 1e-300)
    return float(l2), float(linf)


def relative(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def parse_run(log_path):
    dts = []
    accepted_errors = []
    rejected_errors = []
    summary = None
    for line in log_path.read_text(errors="replace").splitlines():
        m = DT_RE.search(line)
        if m:
            dts.append(float(m.group(1)))
        m = ERR_RE.search(line)
        if m:
            target = accepted_errors if m.group(2) == "1" else rejected_errors
            target.append(float(m.group(1)))
        m = SUMMARY_RE.search(line)
        if m:
            summary = {
                "macro_steps": int(m.group(1)),
                "internal_rejects": int(m.group(2)),
                "fallback_macros": int(m.group(3)),
                "retry_wall_fraction": float(m.group(4)),
            }
    return dts, accepted_errors, rejected_errors, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--initial-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()

    cases = {
        "variable_G9": args.run_root / "variable",
        "fixed_G9_dt4": args.run_root / "fixed_G9_dt4",
        "strict_G12_dt32": args.run_root / "strict_G12_dt32",
    }
    fields = {}
    metas = {}
    for name, directory in cases.items():
        fields[name] = {
            "C": read_raw(locate(directory, "Ctot")),
            "phi": read_raw(locate(directory, "phi")),
            "x": read_raw(locate(directory, "xB_alpha")),
        }
        metas[name] = read_meta(locate_meta(directory))

    initial = {
        "C": read_raw(args.initial_root / "ctot_checkpoint_step005482_Ctot.raw"),
        "phi": read_raw(args.initial_root / "ctot_checkpoint_step005482_phi.raw"),
        "x": read_raw(args.initial_root / "ctot_checkpoint_step005482_xB_alpha.raw"),
    }
    initial_meta = read_meta(
        args.initial_root / "ctot_checkpoint_step005482_meta.json")

    ref = fields["strict_G12_dt32"]
    ref_h = h(ref["phi"])
    initial_h = h(initial["phi"])
    ref_q = ref["C"] - ref_h
    initial_q = initial["C"] - initial_h
    ref_cross = crossings(ref["phi"])
    ref_transfer = float(np.sum(ref_h) - np.sum(initial_h))
    ref_far_mask = ref_h < 1.0e-3
    ref_far = float(np.mean(ref["x"][ref_far_mask]))

    rows = []
    for name in ("variable_G9", "fixed_G9_dt4", "strict_G12_dt32"):
        case = fields[name]
        C_l2, C_linf = norm_error(case["C"], ref["C"], initial["C"])
        phi_l2, phi_linf = norm_error(case["phi"], ref["phi"], initial["phi"])
        case_h = h(case["phi"])
        h_increment = float(np.sum(case_h) - np.sum(initial_h))
        h_error = relative(h_increment, ref_transfer)
        # xB_alpha is not an observable matrix profile where 1-h approaches zero.
        # Compare its conserved, capacity-weighted representation q_alpha instead.
        case_q = case["C"] - case_h
        capacity_profile_error = float(
            np.linalg.norm(case_q - ref_q) /
            max(np.linalg.norm(ref_q), 1e-300))
        capacity_increment_error = float(
            np.linalg.norm(case_q - ref_q) /
            max(np.linalg.norm(ref_q - initial_q), 1e-300))
        raw_x_profile_error = float(
            np.linalg.norm(case["x"] - ref["x"]) /
            max(np.linalg.norm(ref["x"]), 1e-300))
        interface_error = crossing_error(crossings(case["phi"]), ref_cross)
        far = float(np.mean(case["x"][ref_far_mask]))
        far_error = relative(far, ref_far)
        end_time = float(metas[name]["bdf2_time_code"])
        delta_time = end_time - float(initial_meta["bdf2_time_code"])
        rows.append({
            "case": name,
            "delta_time_code": delta_time,
            "time_mismatch_vs_variable_rel": 0.0,
            "Ctot_increment_L2_error": C_l2,
            "Ctot_increment_Linf_error": C_linf,
            "phi_increment_L2_error": phi_l2,
            "phi_increment_Linf_error": phi_linf,
            "h_volume_increment_error": h_error,
            "matrix_profile_L2_error": capacity_profile_error,
            "matrix_profile_increment_L2_error": capacity_increment_error,
            "raw_xB_alpha_L2_diagnostic_only": raw_x_profile_error,
            "interface_position_error_dx": interface_error,
            "transfer_error": relative(h_increment, ref_transfer),
            "far_field_error": far_error,
            "h_volume_increment": h_increment,
            "far_field_xB": far,
        })
    variable_time = rows[0]["delta_time_code"]
    for row in rows:
        row["time_mismatch_vs_variable_rel"] = relative(
            row["delta_time_code"], variable_time)

    args.report_root.mkdir(parents=True, exist_ok=True)
    metrics_path = args.report_root / "variable_step_accuracy_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    dts, accepted_errors, rejected_errors, summary = parse_run(
        cases["variable_G9"] / "run.log")
    if len(dts) != 200 or summary is None:
        raise RuntimeError("variable run diagnostics incomplete")
    dt_rows = []
    for i, dt in enumerate(dts, 1):
        dt_rows.append({
            "step": i,
            "accepted_dt_code": dt,
            "accepted_dt_physical_s": dt * 41.1295854245547687,
        })
    with (args.report_root / "variable_step_dt_distribution.csv").open(
            "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dt_rows[0]))
        writer.writeheader()
        writer.writerows(dt_rows)

    var = rows[0]
    gates = {
        "Ctot_increment": var["Ctot_increment_L2_error"] <= 0.02,
        "phi_increment": var["phi_increment_L2_error"] <= 0.02,
        "h_volume": var["h_volume_increment_error"] <= 0.02,
        "matrix_profile": var["matrix_profile_L2_error"] <= 0.03,
        "interface": var["interface_position_error_dx"] <= 0.25,
        "transfer": var["transfer_error"] <= 0.02,
        "far_field": var["far_field_error"] <= 0.01,
        "retry": summary["internal_rejects"] / summary["macro_steps"] <= 0.01,
        "fallback": summary["fallback_macros"] / summary["macro_steps"] <= 0.01,
        "retry_wall": summary["retry_wall_fraction"] <= 0.05,
    }
    status = "PASS_VARIABLE_STEP_SHORT_WINDOW_ACCURACY" if all(gates.values()) \
        else "FAIL_VARIABLE_STEP_SHORT_WINDOW_ACCURACY"
    dt_array = np.array(dts)
    report = f"""# Variable-step accuracy qualification

## Evidence

All trajectories start from the same accepted step-5482 `Ctot`, `phi`, and BDF2 history.
The variable candidate uses G9 with `dt16 <= dt <= dt4`; the reference uses G12/dt32.
The variable interval is `{variable_time:.17e}` code time. The strict endpoint mismatch is
`{rows[2]['time_mismatch_vs_variable_rel']:.6%}` and the fixed G9/dt4 mismatch is
`{rows[1]['time_mismatch_vs_variable_rel']:.6%}`.

| Metric | Variable error | Gate | Pass |
|---|---:|---:|---|
| Ctot increment L2 | {var['Ctot_increment_L2_error']:.6e} | 2% | {gates['Ctot_increment']} |
| phi increment L2 | {var['phi_increment_L2_error']:.6e} | 2% | {gates['phi_increment']} |
| h-volume increment | {var['h_volume_increment_error']:.6e} | 2% | {gates['h_volume']} |
| capacity-weighted matrix profile L2 (`q_alpha=Ctot-h`) | {var['matrix_profile_L2_error']:.6e} | 3% | {gates['matrix_profile']} |
| interface position | {var['interface_position_error_dx']:.6e} dx | 0.25 dx | {gates['interface']} |
| transfer | {var['transfer_error']:.6e} | 2% | {gates['transfer']} |
| far field | {var['far_field_error']:.6e} | 1% | {gates['far_field']} |

The 200-step run accepted all `{summary['macro_steps']}` macros with
`{summary['internal_rejects']}` internal rejects and `{summary['fallback_macros']}` fallback
macros. Mean accepted dt is `{np.mean(dt_array):.17e}` code units; p50/p90/p99 are
`{np.quantile(dt_array, 0.5):.17e}`, `{np.quantile(dt_array, 0.9):.17e}`, and
`{np.quantile(dt_array, 0.99):.17e}`. Maximum accepted normalized embedded error is
`{max(accepted_errors) if accepted_errors else math.nan:.17e}`. Maximum rejected-trial
error is `{max(rejected_errors) if rejected_errors else math.nan:.17e}`; rejected trials
were rolled back and are not part of the accepted trajectory.

The raw all-domain `xB_alpha` difference is
`{var['raw_xB_alpha_L2_diagnostic_only']:.6e}` and is diagnostic only. In beta support,
`1-h` approaches zero, so raw `xB_alpha` is not the conserved matrix profile and must not
be used as a production accuracy gate. The gated profile is `q_alpha=(1-h)xB_alpha`.

Status: `{status}`.

This is a short-window one-dimensional accuracy qualification. It does not replace the
required 400-cube P/M throughput benchmark or establish a long-time GP model.
"""
    (args.report_root / "variable_step_accuracy.md").write_text(report, encoding="utf-8")
    print(status)


if __name__ == "__main__":
    main()
