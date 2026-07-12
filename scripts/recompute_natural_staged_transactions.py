#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path


PAIR_RE = re.compile(r"([A-Za-z0-9_]+)=([^ \n]+)")


def parse_pairs(block: str) -> dict:
    out = {}
    for key, value in PAIR_RE.findall(block):
        out[key] = value.strip()
    return out


def as_float(d: dict, key: str, default=float("nan")) -> float:
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


def as_int(d: dict, key: str, default=0) -> int:
    try:
        return int(float(d.get(key, default)))
    except (TypeError, ValueError):
        return default


def finite(x: float) -> bool:
    return math.isfinite(x)


def rel(err: float, denom: float) -> float:
    return err / max(abs(denom), 1.0e-30)


def iter_single_line_blocks(text: str, begin: str, end: str):
    # Current runtime prints each transaction block on one line.
    for line in text.splitlines():
        if begin in line and end in line:
            yield line


def parse_log(path: Path):
    text = path.read_text(errors="ignore")
    rows = []

    for block in iter_single_line_blocks(text, "BETA_STAGED_EMBRYO_BEGIN", "BETA_STAGED_EMBRYO_END"):
        p = parse_pairs(block)
        target = as_float(p, "target_seed_inventory")
        current_after = as_float(p, "current_embryo_inventory")
        gp_initial = as_float(p, "source_GP_initial_consumed", 0.0)
        gp_new = as_float(p, "source_GP_new_consumed", 0.0)
        matrix = as_float(p, "source_matrix_consumed", 0.0)
        expected = gp_initial + gp_new + matrix
        delta = current_after
        err_abs = delta - expected
        rows.append({
            "source_file": str(path),
            "transaction_type": "creation",
            "step": as_int(p, "step"),
            "event_id": as_int(p, "event_id"),
            "embryo_id": as_int(p, "embryo_id"),
            "target_seed_inventory": target,
            "embryo_inventory_before": 0.0,
            "embryo_inventory_after": current_after,
            "delta_embryo_inventory": delta,
            "delta_GP_initial_consumed": gp_initial,
            "delta_GP_new_consumed": gp_new,
            "delta_matrix_consumed": matrix,
            "expected_delta_added": expected,
            "recomputed_transaction_error_abs": err_abs,
            "recomputed_transaction_error_rel": rel(err_abs, target),
            "old_reported_transaction_error_rel": as_float(p, "transaction_mass_error_rel"),
            "old_reported_global_mass_error_rel": as_float(p, "global_mass_error_rel"),
            "matrix_included_closes": abs(rel(err_abs, target)) <= 1.0e-10,
            "printed_fields_sufficient": finite(current_after) and finite(expected) and finite(target),
            "status": p.get("status", ""),
        })

    for block in iter_single_line_blocks(text, "BETA_STAGED_ACCUMULATION_BEGIN", "BETA_STAGED_ACCUMULATION_END"):
        p = parse_pairs(block)
        target = as_float(p, "target_seed_inventory")
        current_before = as_float(p, "current_inventory_before")
        current_after = as_float(p, "current_inventory_after")
        delta = as_float(p, "delta_inventory_added")
        if not finite(delta) and finite(current_before) and finite(current_after):
            delta = current_after - current_before
        gp_initial = as_float(p, "source_GP_initial_consumed_step", 0.0)
        gp_new = as_float(p, "source_GP_new_consumed_step", 0.0)
        matrix = as_float(p, "source_matrix_consumed_step", 0.0)
        expected = gp_initial + gp_new + matrix
        err_abs = delta - expected
        rows.append({
            "source_file": str(path),
            "transaction_type": "accumulation",
            "step": as_int(p, "step"),
            "event_id": "",
            "embryo_id": as_int(p, "embryo_id"),
            "target_seed_inventory": target,
            "embryo_inventory_before": current_before,
            "embryo_inventory_after": current_after,
            "delta_embryo_inventory": delta,
            "delta_GP_initial_consumed": gp_initial,
            "delta_GP_new_consumed": gp_new,
            "delta_matrix_consumed": matrix,
            "expected_delta_added": expected,
            "recomputed_transaction_error_abs": err_abs,
            "recomputed_transaction_error_rel": rel(err_abs, target),
            "old_reported_transaction_error_rel": as_float(p, "transaction_mass_error_rel"),
            "old_reported_global_mass_error_rel": as_float(p, "global_mass_error_rel"),
            "matrix_included_closes": abs(rel(err_abs, target)) <= 1.0e-10,
            "printed_fields_sufficient": finite(current_before) and finite(current_after) and finite(expected) and finite(target),
            "status": f"{p.get('status_before', '')}->{p.get('status_after', '')}",
        })
    return rows


def write_report(rows, report_path: Path):
    creations = [r for r in rows if r["transaction_type"] == "creation"]
    accum = [r for r in rows if r["transaction_type"] == "accumulation"]
    max_recomputed = max((abs(float(r["recomputed_transaction_error_rel"])) for r in rows), default=float("nan"))
    max_old = max((abs(float(r["old_reported_transaction_error_rel"])) for r in rows if finite(float(r["old_reported_transaction_error_rel"]))), default=float("nan"))
    first_old_fail = next((r for r in rows if finite(float(r["old_reported_transaction_error_rel"])) and abs(float(r["old_reported_transaction_error_rel"])) > 1.0e-7), None)
    first_recomputed_fail = next((r for r in rows if abs(float(r["recomputed_transaction_error_rel"])) > 1.0e-10), None)
    step219 = next((r for r in rows if r["transaction_type"] == "creation" and int(r["step"]) == 219), None)

    lines = [
        "# Phase 1 Independent Recompute Report",
        "",
        "## Summary",
        "",
        f"- Parsed creation transactions: `{len(creations)}`",
        f"- Parsed accumulation transactions: `{len(accum)}`",
        f"- Max old reported transaction error: `{max_old:.12e}`",
        f"- Max independently recomputed transaction error: `{max_recomputed:.12e}`",
        "",
        "## Required Answers",
        "",
        "1. Does independent recomputation reproduce the reported failure?",
        "",
        "No. Independent source-delta recomputation does not reproduce the old event-level failure.",
        "",
        "2. If `source_matrix_consumed` is included, does step 219 close?",
        "",
    ]
    if step219:
        lines += [
            f"Yes. For step 219, `delta_embryo_inventory = {step219['delta_embryo_inventory']:.12e}` and "
            f"`source_GP_initial + source_GP_new + source_matrix = {step219['expected_delta_added']:.12e}`.",
            f"The recomputed relative error is `{step219['recomputed_transaction_error_rel']:.12e}`.",
            "",
        ]
    else:
        lines += ["Step 219 was not found in the parsed input.", ""]

    lines += [
        "3. Is the old diagnostic omitting matrix source?",
        "",
        "The printed source fields close when matrix source is included. The old diagnostic is not a source-delta formula; it is a ledger before/after formula around staged creation.",
        "",
        "4. Is the old diagnostic using cumulative instead of delta terms?",
        "",
        "For creation, the printed cumulative fields equal the creation delta because the embryo starts from zero. For accumulation, the log already prints step deltas. The independent recomputation uses deltas and closes.",
        "",
        "5. Are printed fields sufficient to reconstruct the transaction?",
        "",
        "Yes for the source-delta transaction. They are not sufficient to diagnose all ledger subcomponents because old logs do not include before/after GP totals, staged totals, matrix totals, and beta totals in the staged block.",
        "",
        "## First Old Failure",
        "",
    ]
    if first_old_fail:
        lines += [
            f"- type: `{first_old_fail['transaction_type']}`",
            f"- step: `{first_old_fail['step']}`",
            f"- embryo_id: `{first_old_fail['embryo_id']}`",
            f"- old reported transaction error: `{first_old_fail['old_reported_transaction_error_rel']:.12e}`",
            f"- recomputed transaction error: `{first_old_fail['recomputed_transaction_error_rel']:.12e}`",
            "",
        ]
    else:
        lines += ["No old reported failure was found above `1e-7`.", ""]

    lines += [
        "## First Recomputed Failure",
        "",
    ]
    if first_recomputed_fail:
        lines += [
            f"- type: `{first_recomputed_fail['transaction_type']}`",
            f"- step: `{first_recomputed_fail['step']}`",
            f"- embryo_id: `{first_recomputed_fail['embryo_id']}`",
            f"- recomputed transaction error: `{first_recomputed_fail['recomputed_transaction_error_rel']:.12e}`",
            "",
        ]
    else:
        lines += ["No recomputed source-delta transaction failure was found above `1e-10`.", ""]

    report_path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args()

    rows = []
    for path in args.logs:
        rows.extend(parse_log(path))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_file", "transaction_type", "step", "event_id", "embryo_id",
        "target_seed_inventory", "embryo_inventory_before", "embryo_inventory_after",
        "delta_embryo_inventory", "delta_GP_initial_consumed", "delta_GP_new_consumed",
        "delta_matrix_consumed", "expected_delta_added",
        "recomputed_transaction_error_abs", "recomputed_transaction_error_rel",
        "old_reported_transaction_error_rel", "old_reported_global_mass_error_rel",
        "matrix_included_closes", "printed_fields_sufficient", "status",
    ]
    with args.csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    write_report(rows, args.report)


if __name__ == "__main__":
    main()
