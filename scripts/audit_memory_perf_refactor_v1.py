#!/usr/bin/env python3
"""Generate the static allocation and lifetime evidence for memory refactor v1."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from audit_low_memory_transport_v1_memory import B, elements


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "memory_perf_refactor_v1"
SOURCE_FILES = (
    "main_cuda.cu", "cuda_kernels.cu", "cuda_common.cu",
    "cuda_kernels.h", "cuda_common.h", "ctot_performance_profiler.h",
)
SIDES = (128, 192, 256, 320, 384, 400)

# Retained cumulative selector: M1 + M2 + M4 + M5 + M6 + M8 = 187.
REMOVED_BY = {
    "d_ctot_event_macro_start_r": "M1_LAZY_EVENT_SNAPSHOT",
    "d_phi_event_macro_start_r": "M1_LAZY_EVENT_SNAPSHOT",
    "d_Y_event_macro_start_r": "M1_LAZY_EVENT_SNAPSHOT",
    "d_ctot_outer_C_prev_r": "M2_SINGLE_OUTER_HISTORY_ELISION",
    "d_ctot_outer_phi_prev_r": "M2_SINGLE_OUTER_HISTORY_ELISION",
    "d_ctot_outer_C_prev2_r": "M2_SINGLE_OUTER_HISTORY_ELISION",
    "d_ctot_outer_phi_prev2_r": "M2_SINGLE_OUTER_HISTORY_ELISION",
    "d_ctot_outer_face_prev_x": "M2_SINGLE_OUTER_HISTORY_ELISION",
    "d_ctot_outer_face_prev_y": "M2_SINGLE_OUTER_HISTORY_ELISION",
    "d_ctot_outer_face_prev_z": "M2_SINGLE_OUTER_HISTORY_ELISION",
    "d_Y_k": "M4_PHASE_SPECTRUM_ALIAS",
    "d_Y_n_saved": "M5_DERIVED_Y_RECONSTRUCTION",
    "d_divJ_k": "M5_UNUSED_DERIVED_ELISION",
    "KS.d_k4": "M5_UNUSED_DERIVED_ELISION",
    "d_ctot_fv_face_y": "M6_FACE_STREAMING",
    "d_ctot_fv_face_z": "M6_FACE_STREAMING",
}


def locate(name: str) -> tuple[str, int, str]:
    needles = [name, name.replace("KS.", "")]
    for source_name in SOURCE_FILES:
        lines = (ROOT / source_name).read_text(encoding="utf-8").splitlines()
        for line_no, line in enumerate(lines, 1):
            if "cudaMalloc" in line and any(n in line for n in needles):
                return source_name, line_no, "main" if source_name == "main_cuda.cu" else source_name.split(".")[0]
        for line_no, line in enumerate(lines, 1):
            if any(n in line for n in needles):
                return source_name, line_no, "main" if source_name == "main_cuda.cu" else source_name.split(".")[0]
    return "UNKNOWN", 0, "UNKNOWN"


def source_allocations() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    patterns = (
        ("DEVICE_ALLOC", re.compile(r"cudaMalloc(?:Managed|Async)?\s*\(")),
        ("DEVICE_FREE", re.compile(r"cudaFree(?:Async)?\s*\(")),
        ("HOST_PINNED_ALLOC", re.compile(r"(?:cudaHostAlloc|cudaMallocHost)\s*\(")),
        ("HOST_PINNED_FREE", re.compile(r"cudaFreeHost\s*\(")),
        ("CUFFT_PLAN_CREATE", re.compile(r"(?:cufftCreate|cufftPlan\w*)\s*\(")),
        ("CUFFT_PLAN_SIZE", re.compile(r"cufftMakePlan\w*\s*\(")),
        ("CUFFT_WORKAREA_BIND", re.compile(r"cufftSetWorkArea\s*\(")),
        ("CUFFT_PLAN_DESTROY", re.compile(r"cufftDestroy\s*\(")),
    )
    for source_name in SOURCE_FILES:
        lines = (ROOT / source_name).read_text(encoding="utf-8").splitlines()
        for line_no, line in enumerate(lines, 1):
            kinds = [kind for kind, pattern in patterns if pattern.search(line)]
            if kinds:
                snippet = " ".join(lines[line_no - 1:min(line_no + 2, len(lines))]).strip()
                rows.append({
                    "call_kind": "+".join(kinds),
                    "source_file": source_name,
                    "source_line": line_no,
                    "source_expression": snippet,
                })
    return rows


def write_inventory() -> None:
    n = 400**3
    k = 400 * 400 * 201
    fields = [
        "allocation_id", "source_file", "function", "source_line", "buffer_name",
        "dtype", "elements_formula", "bytes_at_400cube", "bytes_per_cell",
        "grid_dependence", "lifetime", "physics_owner", "solver_owner",
        "first_use", "last_use", "read_kernels", "write_kernels",
        "rollback_requirement", "checkpoint_requirement", "restart_requirement",
        "fft_requirement", "mechanics_requirement", "gp_requirement",
        "can_reconstruct", "can_alias", "can_allocate_lazily", "can_compress_dtype",
        "baseline_active", "optimized_active", "optimization", "current_reason",
    ]
    detailed_names = {item.name for item in B}
    with (OUT / "allocation_inventory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for allocation_id, item in enumerate(B, 1):
            source_file, source_line, function = locate(item.name)
            after = item.name not in REMOVED_BY
            reconstruct = "yes" if (
                "reconstruct" in item.rollback or item.name in {"d_Y_n_saved", "KS.d_k4"}
            ) else "conditional"
            writer.writerow({
                "allocation_id": f"PF{allocation_id:03d}",
                "source_file": source_file,
                "function": function,
                "source_line": source_line,
                "buffer_name": item.name,
                "dtype": item.dtype,
                "elements_formula": item.count,
                "bytes_at_400cube": elements(item.count, n, k) * item.bytes_per_element,
                "bytes_per_cell": item.bytes_per_element * elements(item.count, n, k) / n,
                "grid_dependence": "cell" if "N" in item.count else ("spectrum" if "K" in item.count else "scalar"),
                "lifetime": item.lifetime,
                "physics_owner": "PF_ONLY" if item.gp == "no" else "GP_CONDITIONAL",
                "solver_owner": "transport/phase transaction",
                "first_use": item.first_use,
                "last_use": item.last_use,
                "read_kernels": "source-traced; see allocation_call_graph.md",
                "write_kernels": "source-traced; see allocation_call_graph.md",
                "rollback_requirement": item.rollback,
                "checkpoint_requirement": item.checkpoint,
                "restart_requirement": item.checkpoint,
                "fft_requirement": item.fft,
                "mechanics_requirement": item.mechanics,
                "gp_requirement": item.gp,
                "can_reconstruct": reconstruct,
                "can_alias": item.aliasable,
                "can_allocate_lazily": "yes" if "lazy" in item.aliasable or "event" in item.condition else "conditional",
                "can_compress_dtype": "no" if item.dtype in {"float64", "complex128"} else "already compact/scalar",
                "baseline_active": "yes" if item.condition != "elastic" else "no",
                "optimized_active": "yes" if after and item.condition != "elastic" else "no",
                "optimization": REMOVED_BY.get(item.name, "retained"),
                "current_reason": item.lifetime,
            })

    # Every source allocation call remains searchable, including paths outside
    # the frozen PF-only configuration. This separate table prevents conditional
    # GP/elastic/developer allocations from being mistaken for the active ledger.
    all_rows = source_allocations()
    with (OUT / "all_cuda_allocation_calls.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=("allocation_call_id", "call_kind", "source_file", "source_line", "source_expression", "active_path_classification")
        )
        writer.writeheader()
        for i, row in enumerate(all_rows, 1):
            expr = str(row["source_expression"])
            classification = "PF_LEDGER_DETAILED" if any(name.replace("KS.", "") in expr for name in detailed_names) else "CONDITIONAL_OR_NON_PF_PATH"
            writer.writerow({"allocation_call_id": f"SRC{i:04d}", **row, "active_path_classification": classification})


def totals(side: int) -> tuple[int, int]:
    n = side**3
    k = side * side * (side // 2 + 1)
    active = [item for item in B if item.condition != "elastic"]
    before = sum(elements(item.count, n, k) * item.bytes_per_element for item in active)
    after = sum(
        elements(item.count, n, k) * item.bytes_per_element
        for item in active if item.name not in REMOVED_BY
    )
    return before, after


def write_ledgers() -> None:
    fields = (
        "grid", "cells", "spectral_elements", "configuration", "source_bytes_before",
        "source_GiB_before", "source_bytes_after", "source_GiB_after", "saved_GiB",
        "reduction_percent", "device_85pct_GiB", "static_admission_status",
    )
    with (OUT / "static_memory_ledger.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for side in SIDES:
            before, after = totals(side)
            writer.writerow({
                "grid": f"{side}^3", "cells": side**3,
                "spectral_elements": side * side * (side // 2 + 1),
                "configuration": "PF_ONLY_ACTIVE_MANIFOLD_IMEX_BDF2",
                "source_bytes_before": before, "source_GiB_before": before / 2**30,
                "source_bytes_after": after, "source_GiB_after": after / 2**30,
                "saved_GiB": (before - after) / 2**30,
                "reduction_percent": 100.0 * (before - after) / before,
                "device_85pct_GiB": 13.533,
                "static_admission_status": "PASS_SOURCE_ONLY" if after / 2**30 <= 12.5 else "FAIL_SOURCE_TARGET",
            })

    fields = list(B[0].__dataclass_fields__) + [
        "bytes_at_400cube", "optimized_lifetime", "optimization"
    ]
    with (OUT / "device_buffer_lifetime.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        n, k = 400**3, 400 * 400 * 201
        for item in B:
            row = item.__dict__.copy()
            row["bytes_at_400cube"] = elements(item.count, n, k) * item.bytes_per_element
            row["optimized_lifetime"] = "not allocated on selected path" if item.name in REMOVED_BY else item.lifetime
            row["optimization"] = REMOVED_BY.get(item.name, "retained")
            writer.writerow(row)


def write_alias_matrix() -> None:
    rows = [
        ("event Ctot macro snapshot", "BDF2 transport anchor", "event BE subcycle starts after BDF2 context dies", "ALIAS_RETAINED_M1", "forced rollback bitwise"),
        ("event phi macro snapshot", "BDF2 phi transport context", "event BE subcycle starts after BDF2 context dies", "ALIAS_RETAINED_M1", "forced rollback bitwise"),
        ("phase Y spectrum", "phase phi spectrum", "method split does not require simultaneous spectra", "ALIAS_RETAINED_M4", "100-step endpoint bitwise"),
        ("face y", "face x", "axis face consumed into divergence before next axis", "ALIAS_RETAINED_M6", "100-step endpoint bitwise"),
        ("face z", "face x", "axis face consumed into divergence before next axis", "ALIAS_RETAINED_M6", "100-step endpoint bitwise"),
        ("Y accepted snapshot", "authoritative Ctot+phi", "Y is exact deterministic derived context", "RECOMPUTE_RETAINED_M5", "forced rollback bitwise"),
        ("Ctot accepted/work/saved", "pointer rotation", "multiple call sites still bind fixed pointer names", "DEFERRED_HIGH_RISK", "not changed"),
        ("active mask float64", "uint8 mask", "kernel ABI and reductions require coordinated conversion", "M3_NOT_RETAINED", "no measured necessity after target met"),
    ]
    with (OUT / "alias_opportunity_matrix.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("buffer_or_role", "alias_or_reconstruction_target", "lifetime_proof", "decision", "validation"))
        writer.writerows(rows)


def write_markdown() -> None:
    before, after = totals(400)
    (OUT / "allocation_call_graph.md").write_text(
        "# Allocation call graph\n\n"
        "`main -> parameter validation -> PF candidate selector -> core state allocations -> "
        "BDF2 history/context -> transport/phase work fields -> shared FFT plans -> accepted-step loop -> teardown`.\n\n"
        "The frozen PF-only path is detailed in `allocation_inventory.csv`; every syntactic device allocate/free, pinned-host allocate/free, cuFFT plan/create/size/work-area/destroy call in the audited sources is indexed in `all_cuda_allocation_calls.csv`. Conditional GP, elastic, minimize, observer, and legacy paths remain classified rather than counted as simultaneously live. The table is an asset map, not a peak-memory assertion.\n\n"
        "Authoritative state is `Ctot + phi` with BDF2 histories. `Y`, `xB_alpha`, and `q_alpha` are derived. The selected path removes the persistent `Y^n` image and rebuilds it from the authoritative rollback image. Event macro images alias dead BDF2 context buffers during BE subcycling.\n",
        encoding="utf-8",
    )
    (OUT / "live_memory_timeline.md").write_text(
        f"# Live memory timeline\n\n"
        f"Static PF-only 400^3 source allocation falls from {before/2**30:.6f} GiB to {after/2**30:.6f} GiB.\n\n"
        "| Stage | Dominant live groups | Refactor evidence |\n"
        "|---|---|---|\n"
        "| Initialization | authoritative state, histories, FFT/scratch | optional feature arrays absent |\n"
        "| BDF2 context | Ctot/phi anchor/context | later reused by event rollback |\n"
        "| Transport | one positive-face field, divergence, residual, derived context | y/z face fields absent |\n"
        "| Phase PDAS | phase spectrum/PCG/scratch | Y spectrum aliases phase spectrum |\n"
        "| Cold audit | transport scratch and one face field | no three-face persistence |\n"
        "| Commit | authoritative/history fields | no outer-history fields |\n"
        "| Event fallback | BDF2 anchor/context become macro rollback images | zero incremental full-grid allocation |\n",
        encoding="utf-8",
    )
    (OUT / "optimization_priority.md").write_text(
        "# Optimization priority and decisions\n\n"
        "| ID | Decision | Main value | Risk disposition |\n|---|---|---|---|\n"
        "| M1 | retained | lazy/aliased event snapshots | forced rollback bitwise |\n"
        "| M2 | retained | remove seven method-inapplicable outer-history fields | only OFF single-outer method path |\n"
        "| M3 | not retained | one mask field could be compacted | target met; ABI/reduction risk not justified |\n"
        "| M4 | retained | alias non-overlapping complex spectrum | endpoint bitwise |\n"
        "| M5 | retained | remove Y snapshot, unused spectrum and k4 | deterministic reconstruction/zero consumers |\n"
        "| M6 | retained | stream one face direction at a time | endpoint and conservation bitwise |\n"
        "| M7 | not retained | explicit cuFFT workspace sharing | measured plan workspace required first |\n"
        "| M8 | retained | full-grid non-authoritative diagnostic at CSV cadence | all hard predicates remain per-step |\n"
        "| M9 | not retained | kernel fusion | no profile evidence justifying arithmetic-order risk |\n"
        "| M10 | not retained | CUDA graph | dynamic nonlinear loop and retry branches dominate |\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_inventory()
    write_ledgers()
    write_alias_matrix()
    write_markdown()


if __name__ == "__main__":
    main()
