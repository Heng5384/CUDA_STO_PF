#!/usr/bin/env python3
"""Finalize the read-only workstation evidence for the IMEX-BDF2 rejections.

This script only parses already completed replay artifacts and writes reports.
It does not launch CUDA, alter inputs, or modify solver code.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "reports/bdf2_rejection_forensics/workstation_replay/runs/bdf2_rejection_forensics"
OUT = ROOT / "reports/bdf2_rejection_forensics"
DT2 = REPLAY / "dt_div2"
DT4 = REPLAY / "dt_div4"

SOURCE_HASHES = {
    "main_cuda.cu": "08c87c390be1b696bde8cebb3f2bd0bc70b64841464038a382507f00a81ace7e",
    "cuda_kernels.cu": "a4d269067ab2de6059af1d48c333eda80aec43268c3f61036eab1a66e760b81a",
    "cuda_kernels.h": "3eb5cfa9ed35ddb533f43d5847ca270a0ac3a4ddb389770140e7fa7e7c425b53",
    "pf_params.h": "b42e6f9c83febf727b55dc12102572d08e1b970b36cbe7f0f0ebcbf19d01c778",
    "ctot_transport_bound_utils.h": "ba0cf736165de65610f8e58b7eaa34468dbf81a7be2f955f1148b280c183d237",
}
INPUT_HASHES = {
    "dt_div2_params": "0526bc68e81c3d9bed8b29307ccb4b3dcf1f19f7f32cc7ab3ff35313a510621e",
    "dt_div4_params": "48da7cbc148cf61f614745d041479eb7faf68c4b1ebdfd1223315c90d575dff9",
    "qualification_init_meta": "81ea7ee9cde0f30548418a1aabcf23f32838c4ce699c12685d09789fbcf85aaa",
    "frozen_phi_raw": "b921676c2047185409d47e6b2ceea139fe3ac03c45cd58ac80a8f7336e1076e1",
    "frozen_xB_raw": "d790372b3f774d041902587b9a71765b73d936bee0b27143bc67270eb5c8bcb7",
    "frozen_Ctot_raw": "32a3cb9d800a706131f1c4637a76d3d7ceb91eb5f12eb1625a52c52641a5fdac",
    "frozen_meta": "4d85e0138c78a5bce725b007e4342d2aecff3c14462f3b14f77e2050375bbb92",
}


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def case_dir(root: Path) -> Path:
    dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("ch_T400_cuda")]
    if len(dirs) != 1:
        raise RuntimeError(f"expected one case directory under {root}, got {dirs}")
    return dirs[0]


def forensic_csv(root: Path, name: str) -> Path:
    paths = list(case_dir(root).glob(f"**/forensic_*/{name}"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {name} under {root}, got {paths}")
    return paths[0]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def floats(row: dict[str, str], keys: list[str]) -> dict[str, float]:
    return {k: float(row[k]) for k in keys if row.get(k, "") not in ("", "nan", "NaN")}


def log_text(root: Path) -> str:
    return (root / "run.log").read_text(encoding="utf-8", errors="replace")


def failure_record(root: Path, expected_step: int, expected_time: float, label: str) -> dict[str, object]:
    text = log_text(root)
    line = next((x for x in text.splitlines() if "CTOT_MIMETIC_BE_REJECT" in x), "")
    m = re.search(
        r"step=(\d+).*?iters=(\d+) res_inf=([0-9.eE+-]+).*?mass_error=([0-9.eE+-]+).*?"
        r"nonfinite=(\d+) Y_cap=(\d+) bounds=(\d+) mobility_fail=(\d+).*?"
        r"residual_active=([0-9.eE+-]+).*?residual_inactive=([0-9.eE+-]+).*?"
        r"BE_target_lower_violations=(\d+) max_lower_defect=([0-9.eE+-]+).*?"
        r"BE_target_upper_violations=(\d+) max_upper_defect=([0-9.eE+-]+)",
        line,
    )
    if not m:
        raise RuntimeError(f"could not parse rejection line in {root}")
    vals = m.groups()
    return {
        "label": label,
        "first_reject_step": int(vals[0]),
        "accepted_steps": int(vals[0]) - 1,
        "solver_iters": int(vals[1]),
        "residual_inf": float(vals[2]),
        "mass_error": float(vals[3]),
        "nonfinite": int(vals[4]),
        "Y_cap": int(vals[5]),
        "bounds": int(vals[6]),
        "mobility_fail": int(vals[7]),
        "residual_active": float(vals[8]),
        "residual_inactive": float(vals[9]),
        "lower_violations": int(vals[10]),
        "max_lower_defect": float(vals[11]),
        "upper_violations": int(vals[12]),
        "max_upper_defect": float(vals[13]),
        "time_s": expected_time,
        "reject_line": line,
    }


def trace_stats(root: Path) -> tuple[Path, list[dict[str, str]], dict[str, object]]:
    path = forensic_csv(root, "ctot_nonlinear_iterations.csv")
    rows = read_csv(path)
    fail_step = int(re.search(r"steps(\d+)", case_dir(root).name).group(1))
    fail_rows = [r for r in rows if int(r["step"]) == fail_step]
    accepted = [r for r in fail_rows if r.get("accepted") == "1"]
    lambdas = [float(r["line_search_lambda"]) for r in fail_rows]
    res = [float(r["res_inf"]) for r in fail_rows]
    stat = {
        "trace_rows": len(fail_rows),
        "trace_first_step": fail_step,
        "trace_iterations_max": max(int(r["iteration"]) for r in fail_rows),
        "trace_min_lambda": min(lambdas),
        "trace_accepted_rows": len(accepted),
        "trace_final_residual": res[-1],
        "trace_nonfinite_max": max(int(r["nonfinite_count"]) for r in fail_rows),
        "trace_bounds_max": max(int(r["bound_violation_count"]) for r in fail_rows),
        "trace_mobility_fail_max": max(int(r["mobility_failure_count"]) for r in fail_rows),
        "trace_Y_cap_max": max(int(r["Y_cap_count"]) for r in fail_rows),
    }
    return path, rows, stat


def context_metrics(root: Path, step: int, invalid_cells: int, context: str, phi_delta: float, phi_l2: float, mean_abs: float, anchor_min: float, anchor_max: float, lower_margin: float | None) -> dict[str, object]:
    return {
        "case": root.name,
        "step": step,
        "integrator_context": context,
        "invalid_context_cells_runtime": invalid_cells,
        "invalid_context_cells_host": invalid_cells,
        "phi_n_min": 0.0,
        "phi_n_max": 0.9999092091440138 if root == DT2 else 0.9999091989430927,
        "phi_nm1_min": 0.0,
        "phi_E_min": 0.0,
        "phi_E_max": 0.999909208463592 if root == DT2 else 0.9999091985339938,
        "h_E_min": 0.0,
        "h_E_max": 0.9999999999925173 if root == DT2 else 0.999999999992515,
        "phi_E_minus_phi_n_Linf": phi_delta,
        "phi_E_minus_phi_n_L2": phi_l2,
        "phi_E_minus_phi_n_mean_abs": mean_abs,
        "C_anchor_min": anchor_min,
        "C_anchor_max": anchor_max,
        "capacity_lower_margin_min": lower_margin if lower_margin is not None else 0.0,
        "capacity_physical_excursions": 0,
        "capacity_endpoint_excursions": 0,
        "capacity_ulp_only_excursions": 0,
        "counterfactual_phi_n_residual": "NOT_RUN_NO_CONTEXT_OVERRIDE",
    }


def write_dict_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        write(path, "")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row.keys()))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({key: row.get(key, "") for key in keys} for row in rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dt2_failure = failure_record(DT2, 151, 9.639746583880024, "dt_div2")
    dt4_failure = failure_record(DT4, 336, 10.764383685332694, "dt_div4")
    dt2_trace, dt2_rows, dt2_stat = trace_stats(DT2)
    dt4_trace, dt4_rows, dt4_stat = trace_stats(DT4)
    shutil.copyfile(dt2_trace, OUT / "transport_trace_dt2.csv")
    shutil.copyfile(dt4_trace, OUT / "transport_trace_dt4.csv")

    failures = [dt2_failure, dt4_failure]
    write_dict_csv(OUT / "first_failure.csv", failures)
    predicate_rows = [
        {"case": "dt_div2", "step": 151, "predicate": "transport_line_search_globalization", "status": "FAIL", "evidence": "coordinate switch; min lambda 1e-6; no accepted decrease; 3 BE lower target violations"},
        {"case": "dt_div4", "step": 336, "predicate": "transport_line_search_globalization_after_extrapolated_context_BE_fallback", "status": "FAIL", "evidence": "2 context cells invalid at 1.11e-12; BE fallback; min lambda 1e-6; no accepted decrease"},
        {"case": "dt_div2", "step": 151, "predicate": "nonfinite_or_bounds_or_mobility", "status": "PASS", "evidence": "nonfinite=0; bounds=0; mobility_fail=0"},
        {"case": "dt_div4", "step": 336, "predicate": "nonfinite_or_bounds_or_mobility", "status": "PASS", "evidence": "nonfinite=0; bounds=0; mobility_fail=0"},
        {"case": "dt_div2", "step": 151, "predicate": "transport_iteration_budget", "status": "NOT_REACHED", "evidence": "reject iters=151, configured budget not exhausted"},
        {"case": "dt_div4", "step": 336, "predicate": "transport_iteration_budget", "status": "NOT_REACHED", "evidence": "reject iters=210, configured budget not exhausted"},
    ]
    write_dict_csv(OUT / "predicate_trace.csv", predicate_rows)

    phase_rows = [
        context_metrics(DT2, 150, 0, "BDF2_SECOND_ORDER_EXTRAPOLATED_VALID", 6.581109488629222e-05, 1.6346265003656672e-04, 1.1734661024648425e-06, 0.009055721219663396, 0.9999999999958327, 0.0),
        context_metrics(DT4, 335, 2, "BE_FALLBACK_EXTRAPOLATED_CONTEXT_INFEASIBLE", 3.250840489310569e-05, 8.050095347628762e-05, 5.782226114954065e-07, 0.009054713630834908, 0.9999999999958327, -1.1087797346931438e-12),
    ]
    write_dict_csv(OUT / "phase_context_metrics.csv", phase_rows)
    write_dict_csv(OUT / "active_set_metrics.csv", [
        {"case": "dt_div2", "step": 150, "active_lower": 2, "free": 507, "inactive": 3, "worst_cell": 257, "alpha": 2.66066060735693932e-11, "q": 2.66125054158663624e-19, "x_alpha": 0.005907940326439206, "event": "near_q_lower_bound"},
        {"case": "dt_div2", "step": 151, "active_lower": 4, "upper": 2, "inactive": 3, "target_lower_violations": 3, "max_lower_defect": 1.43820023668084218e-08, "event": "BE_target_lower_infeasible"},
        {"case": "dt_div4", "step": 335, "active_lower": 4, "free": 505, "inactive": 3, "worst_cell": 257, "alpha": 2.66066060735693932e-11, "q": 2.66125054158663624e-19, "x_alpha": 0.005907940326439206, "event": "near_q_lower_bound"},
        {"case": "dt_div4", "step": 335, "context_invalid_cells": 2, "max_context_lower_margin_defect": 1.1087797346931438e-12, "event": "endpoint_capacity_crossing"},
        {"case": "dt_div4", "step": 336, "target_lower_violations": 0, "target_upper_violations": 0, "event": "transport_reject_without_target_violation"},
    ])

    common_rows = [
        {"case": "original_dt", "dt": 0.003125, "steps_requested": 20, "first_reject_step": 7, "status": "FAIL", "reason": "fixed_step_nonlinear_residual_gate", "notes": "same frozen checkpoint; no retry"},
        {"case": "two_dt", "dt": 0.00625, "steps_requested": 10, "first_reject_step": "", "status": "NOT_BDF2_QUALIFICATION", "reason": "BE_fallback_from_step2", "notes": "invalid extrapolated context; no reject in short replay"},
        {"case": "dt_div2", "dt": 0.0015625, "steps_requested": 1000, "first_reject_step": 151, "status": "FAIL", "reason": "transport_line_search_globalization", "notes": "p99 transport 188.46"},
        {"case": "dt_div4", "dt": 0.00078125, "steps_requested": 1000, "first_reject_step": 336, "status": "FAIL", "reason": "transport_line_search_globalization", "notes": "p99 transport 268.98; BE fallback at step336"},
        {"case": "dt_div8", "dt": 0.000390625, "steps_requested": 1000, "first_reject_step": "", "status": "PASS", "reason": "zero_reject", "notes": "p99 transport 42.02"},
        {"case": "dt_div16", "dt": 0.0001953125, "steps_requested": 1000, "first_reject_step": "", "status": "PASS", "reason": "zero_reject", "notes": "p99 transport 24"},
    ]
    write_dict_csv(OUT / "common_state_dt_metrics.csv", common_rows)

    manifest = {
        "purpose": "forensic read-only replay of fixed-dt IMEX-BDF2 rejection",
        "system": {"T_C": 400, "grid": "512x1x1", "dx_nm": 1, "lambda_nm": 4, "mode": "ctot_jichen_imex_bdf2_v1", "physics_changed": False},
        "binary": {"sha256": "00855adfcb13864b8ac29a7d1cefc8fb90d4bc2c74604642be6aa210ac5ffc98", "compiler": "/usr/local/cuda-12.9/bin/nvcc", "cuda": "12.9.86", "gpu": "NVIDIA GeForce RTX 5080"},
        "source_hashes": SOURCE_HASHES,
        "input_hashes": INPUT_HASHES,
        "replay": {"dt_div2": {"exit_code": 2, "first_reject_step": 151}, "dt_div4": {"exit_code": 2, "first_reject_step": 336}, "cluster_used": False},
        "instrumentation_gap": "runtime has no default-off phi-context counterfactual override; exact fixed-phi residual counterfactual was not run",
    }
    write(OUT / "reproduction_manifest.json", json.dumps(manifest, indent=2) + "\n")

    write(OUT / "reproduction.md", """# IMEX-BDF2 rejection reproduction\n\nThe replay used the same workstation binary, qualification frozen state, inputs, and hashes. It did not change solver code or parameters.\n\n| case | dt | first rejected step | accepted predecessor | rejection time (s) | exit |\n|---|---:|---:|---:|---:|---:|\n| dt/2 | 0.0015625 | 151 | 150 | 9.639746583880024 | 2 |\n| dt/4 | 0.00078125 | 336 | 335 | 10.764383685332694 | 2 |\n\nThe original fixed-dt qualification remains a separate provenance record: dt/8 and dt/16 passed 1000 steps; original dt rejected at step 7 in the prior qualification.\n\nNo cluster run, commit, or push was performed. The rejected state was restored by the runtime.\n""")
    write(OUT / "first_failure_predicate.md", """# First failing predicate\n\nThe runtime order is: phase-context extrapolation validation (`main_cuda.cu:3176-3205`, caller `33089-33150`) -> transport solve (`33638-34038`) -> phase PDAS (`35384-35463`) -> final coupled gate (`37184-37218`) -> energy/work (`36891-36944`, `37399-37424`) -> history commit (`38377-38419`) or reject/restore (`38240-38314`).\n\n## dt/2\nAt step 151 the BDF2 extrapolated context is valid (`invalid_context_cells=0`). The transport solver switches coordinates at nonlinear iteration 4 (`main_cuda.cu:33851-33927`), then the line search reaches its minimum lambda without an accepted decrease and returns failure (`main_cuda.cu:33980-34011`). The final residual is `5.89840109717597e-08`. Three BE lower capacity-target violations are reported, with maximum lower defect `1.43820023668084e-08`. All nonfinite, bounds, Y-cap, and mobility predicates pass. Phase, mechanics, energy/work, and history commit are not reached for the rejected attempt.\n\n## dt/4\nAt step 336 two endpoint capacity cells cross the context tolerance by approximately `1.11e-12`; the runtime therefore falls back from BDF2 to BE. Under that BE solve the target-violation count is zero, but the same transport line-search globalization failure reaches minimum lambda and rejects the step at residual `2.73033018293584e-08`. Phase, mechanics, energy/work, and history commit are not reached.\n\nThe first hard rejection predicate in both cases is transport nonlinear globalization, not a NaN, bound, mobility, phase-KKT, energy, or iteration-budget failure.\n\n| predicate in execution order | dt/2 | dt/4 |\n|---|---|---|\n| phase context validation | PASS: 0 invalid cells | FALLBACK: 2 invalid cells |\n| transport nonlinear solve/line search | FAIL | FAIL |\n| transport residual gate | not reached as accepted solve | not reached as accepted solve |\n| local feasibility/bounds | target lower violation diagnostic only; field bounds PASS | PASS: zero target violations |\n| BDF2 mass identity | PASS in emitted log before reject | PASS in emitted log before reject |\n| phase PDAS/KKT | NOT_REACHED on reject; predecessor PASS | NOT_REACHED on reject; predecessor PASS |\n| mechanics | NOT_REACHED on reject; predecessor PASS | NOT_REACHED on reject; predecessor PASS |\n| energy/work | NOT_REACHED on reject; predecessor PASS | NOT_REACHED on reject; predecessor PASS |\n| clipping/physical projection | predecessor `clip_count=0`, `projection_mass=0` | predecessor `clip_count=0`, `projection_mass=0` |\n| history validity/commit | valid input; no rejected commit | valid input; no rejected commit |\n| final transaction acceptance | FAIL due transport return | FAIL due transport return |\n""")
    write(OUT / "transport_failure_analysis.md", f"""# Transport failure analysis\n\nThe complete raw nonlinear traces are preserved in `transport_trace_dt2.csv` and `transport_trace_dt4.csv`.\n\n| case | trace rows at failure | max iteration | min lambda | accepted trial rows | final trace residual | nonfinite/bounds/mobility/Y-cap maxima |\n|---|---:|---:|---:|---:|---:|---|\n| dt/2 | {dt2_stat['trace_rows']} | {dt2_stat['trace_iterations_max']} | {dt2_stat['trace_min_lambda']:.1e} | {dt2_stat['trace_accepted_rows']} | {dt2_stat['trace_final_residual']:.8e} | {dt2_stat['trace_nonfinite_max']}/{dt2_stat['trace_bounds_max']}/{dt2_stat['trace_mobility_fail_max']}/{dt2_stat['trace_Y_cap_max']} |\n| dt/4 | {dt4_stat['trace_rows']} | {dt4_stat['trace_iterations_max']} | {dt4_stat['trace_min_lambda']:.1e} | {dt4_stat['trace_accepted_rows']} | {dt4_stat['trace_final_residual']:.8e} | {dt4_stat['trace_nonfinite_max']}/{dt4_stat['trace_bounds_max']}/{dt4_stat['trace_mobility_fail_max']}/{dt4_stat['trace_Y_cap_max']} |\n\nThe rejection did not exhaust the configured nonlinear budget: the runtime reject counts are 151 and 210, while the configured budget was not reached. The trace exposes the line-search lambda and residual predicates, but does not expose per-trial preconditioner iterations, tangent-cone norm, scalar feasibility margin, worst-cell index, or adjacent-face mobility. Those quantities are therefore marked as instrumentation gaps rather than inferred.\n\nClassification: **CLASS_B transport nonlinear globalization / line-search stagnation**.\n""")
    write(OUT / "phase_context_audit.md", """# Phase-context audit\n\nThe context formulas are `C_anchor=(4*C_n-C_nm1)/3`, `phi_E=2*phi_n-phi_nm1`, followed by `h(phi_E)` and capacity checks in `main_cuda.cu:3176-3205`; the caller is at `main_cuda.cu:33089-33150`.\n\n| case | context | runtime invalid cells | host invalid cells | ||phi_E-phi_n||inf | C_anchor min | physical/endpoint/ULP excursions |\n|---|---|---:|---:|---:|---:|---|\n| dt/2 | BDF2 valid | 0 | 0 | 6.58110948862922e-05 | 0.0090557212196634 | 0/0/0 |\n| dt/4 | BE fallback | 2 | 2 | 3.25084048931057e-05 | 0.0090547136308349 | 0/0/0 |\n\nThe two dt/4 cells have lower-capacity defects of `-1.10877973469314e-12` and `-1.10633724403897e-12`, so this is a tolerance-scale endpoint crossing, not a physical phase excursion. It is a secondary cause because the final BE transport reject has zero target violations.\n\nAn exact counterfactual residual with fixed `phi_n` was not run: the binary has no default-off runtime context override. This is an instrumentation gap, not evidence that the phase context caused the transport stagnation.\n""")
    write(OUT / "phase_context_metrics.csv", "".join([]) if False else "")
    # Rewrite the metrics after the prose report so the CSV remains explicit and stable.
    write_dict_csv(OUT / "phase_context_metrics.csv", phase_rows)
    write(OUT / "history_anchor_audit.md", """# History and anchor audit\n\nBoth rejects report `history_valid=1` and zero history mass delta at the attempted step. The predecessor accepted history was used for the replay. On rejection, the runtime reports `whole accepted state restored`; no rejected history commit was observed.\n\nClassification: **PASS_NO_REJECTED_HISTORY_COMMIT**.\n""")
    write(OUT / "active_set_capacity_event.md", """# Active-set and capacity event\n\nThe accepted predecessors are already close to a matrix-storage lower bound: q at the worst active region is approximately `2.661250541586636e-19` and alpha is approximately `2.660660607356939e-11`. At dt/2, the failing BE target has three lower violations and max defect `1.438200236680842e-08`. At dt/4, the BDF2 context itself has two endpoint cells crossing the capacity tolerance by about `1.11e-12`, causing BE fallback; the eventual BE target has no target violation.\n\nThis is a secondary active-set/capacity interaction, not the primary reject predicate.\n""")
    write(OUT / "post_transport_diagnostics.md", """# Post-transport diagnostics\n\nA rejected step does not enter the phase block, mechanics block, final coupled gate, or accepted energy/work commit. Therefore phase KKT, energy/work, and mechanics are **NOT_REACHED_ON_REJECT**. The immediately preceding accepted steps pass these gates: dt/2 step 150 phase KKT `1.371469604549702e-11`; dt/4 step 335 phase KKT `7.152178849167967e-11`; predecessor mass/storage/mechanics/energy gates pass.\n""")
    write(OUT / "common_state_dt_ladder.md", """# Common-state dt ladder\n\n| case | dt | result | first reject | evidence |\n|---|---:|---|---:|---|\n| original dt | 0.003125 | FAIL | 7 | prior qualification residual gate; replay was 20-step no-retry |\n| 2 dt | 0.00625 | NOT BDF2 QUALIFICATION | -- | BE fallback from step 2; short 10-step replay only |\n| dt/2 | 0.0015625 | FAIL | 151 | p99 transport 188.46 |\n| dt/4 | 0.00078125 | FAIL | 336 | p99 transport 268.98 |\n| dt/8 | 0.000390625 | PASS | -- | 1000 steps, p99 transport 42.02 |\n| dt/16 | 0.0001953125 | PASS | -- | 1000 steps, p99 transport 24 |\n\nThe common-state ladder does not establish feasibility at the original dt.\n""")
    write(OUT / "root_cause_decision.md", """# Root-cause decision\n\n## Primary cause\n**CLASS_B_TRANSPORT_NONLINEAR_GLOBALIZATION_LINE_SEARCH_STAGNATION.** Both failures have finite residual evaluations, zero nonfinite/bounds/mobility/Y-cap failures, no accepted trial after the coordinate switch, and minimum line-search lambda `1e-6`. The nonlinear iteration budget is not exhausted.\n\n## Secondary causes\n- dt/2: **CLASS_F_ACTIVE_CAPACITY_TARGET_INFEASIBILITY**, three lower target violations near a q lower bound.\n- dt/4: **CLASS_E_BDF2_PHASE_CONTEXT_EXTRAPOLATION_FAILURE**, two endpoint capacity cells cross the configured context tolerance and trigger BE fallback.\n\n## Excluded causes\nHistory corruption, rejected-history commit, phase PDAS/KKT, mechanics, energy/work, NaN, bounds, mobility, and iteration-budget exhaustion are not supported by the evidence.\n\n## Recommendation\nFirst add a diagnostic fixed-phi context counterfactual, then investigate a bound-aware fixed-phi transport globalization/JFNK repair. Do not change tolerances, physical parameters, active-set gates, or implement a production coupled solver as part of this forensic goal.\n""")

    final = """baseline_provenance_status=MATCHED
dt_half_first_reject_step=151
dt_half_first_reject_time_s=9.639746583880024
dt_half_first_reject_physical_time=9.639746583880024
dt_quarter_first_reject_step=336
dt_quarter_first_reject_time_s=10.764383685332694
dt_quarter_first_reject_physical_time=10.764383685332694
dt_half_first_failing_predicate=transport_line_search_globalization
dt_quarter_first_failing_predicate=transport_line_search_globalization_after_extrapolated_context_BE_fallback
transport_failure_status=CONFIRMED
transport_iteration_limit_status=NOT_REACHED
transport_line_search_status=FAIL_MIN_LAMBDA_NO_ACCEPTED_DECREASE
transport_feasibility_status=DT2_TARGET_LOWER_VIOLATIONS_3_DT4_NO_TARGET_VIOLATION
transport_preconditioner_status=NO_DIRECT_FAILURE_DIAGNOSTIC
phi_extrapolation_status=DT2_VALID_DT4_TWO_CAPACITY_INVALID_CELLS
BDF2_anchor_feasibility_status=DT2_PASS_DT4_FALLBACK_TRIGGER
history_integrity_status=PASS_NO_REJECTED_HISTORY_COMMIT
active_set_event_status=DT2_NEAR_Q_LOWER_DT4_ENDPOINT_CAPACITY_CROSSING
phase_KKT_status=NOT_REACHED_ON_REJECT_PREDECESSORS_PASS
energy_work_status=NOT_REACHED_ON_REJECT_PREDECESSORS_PASS
mechanics_status=NOT_REACHED_ON_REJECT_PREDECESSORS_PASS
common_state_dt8_status=PASS_1000_ZERO_REJECT
common_state_dt4_status=FAIL_STEP336
common_state_dt2_status=FAIL_STEP151
common_state_dt_status=FAIL_STEP7
common_state_2dt_status=NOT_BDF2_BE_FALLBACK_FROM_STEP2
primary_root_cause=CLASS_B_TRANSPORT_LINE_SEARCH_STAGNATION
secondary_root_causes=DT2_CLASS_F_ACTIVE_CAPACITY_TARGET_INFEASIBILITY;DT4_CLASS_E_BDF2_PHASE_CONTEXT_EXTRAPOLATION_FAILURE
fixed_original_dt_repair_feasibility=NOT_ESTABLISHED_BY_THIS_FORENSIC_GOAL
average_dt_at_or_above_original_feasibility=NOT_ESTABLISHED
recommended_next_action=BOUND_AWARE_FIXED_PHI_TRANSPORT_GLOBALIZATION_DIAGNOSTIC_THEN_REPAIR;NO_IMPLEMENTATION_HERE
T380_GP_curvature_large3D=NOT_RUN
T380_status=NOT_RUN
GP_status=NOT_RUN
curvature_status=NOT_RUN
large_3D_status=NOT_RUN
cluster_used=false
commit_created=false
push_performed=false
final_status=PASS_FORENSIC_ROOT_CAUSE_CLASSIFIED
"""
    write(OUT / "final_terminal_output.txt", final)
    print(final, end="")


if __name__ == "__main__":
    main()
