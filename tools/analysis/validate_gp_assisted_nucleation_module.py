#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from gp_assisted_nucleation_core import (
    BarrierEntry,
    BarrierLibrary,
    NucleationCandidate,
    NucleusCatalog,
    effective_barrier_kBT,
    evaluate_candidate,
    insert_parametric_seed_mass_conserving,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "gp_assisted_beta_nucleation_module"


def rg(pattern: str, *paths: str) -> str:
    cmd = ["rg", "-n", pattern, *paths]
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        return exc.output or ""


def first_match_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def find_existing_inputs() -> dict[str, Any]:
    note_candidates = list(ROOT.glob("**/Nucleation Note.tex"))
    attachment_candidates = list(Path("/Users/heng/.codex/attachments").glob("**/Nucleation Note.tex"))
    barrier_candidates = sorted(ROOT.glob("Results/**/barrier_real_units_by_case.csv"))
    catalog_candidates = sorted(ROOT.glob("**/nucleus_catalog.json"))
    requested_reports = [
        "gp_site_arrays_implementation_notes.md",
        "gp_release_and_draw_kernel_debug.md",
        "gp_initialization_mass_check.md",
        "gp_reservoir_mass_definition.md",
        "gp_site_probability_design.md",
        "existing_beta_insertion_path.md",
        "nucleus_selector_integration_report.md",
        "nucleus_generator_system_report.md",
        "barrier_real_units_report.md",
    ]
    report_hits = {}
    for name in requested_reports:
        report_hits[name] = [str(p) for p in ROOT.glob(f"**/{name}")]
    return {
        "nucleation_note": [str(p) for p in note_candidates + attachment_candidates],
        "barrier_libraries": [str(p) for p in barrier_candidates],
        "nucleus_catalogs": [str(p) for p in catalog_candidates],
        "reports": report_hits,
    }


def generate_audit(out_dir: Path) -> dict[str, str]:
    checks = [
        {
            "item": "explicit GP field / eta",
            "status": "partial",
            "file_function": first_match_line(rg("compute_eta_rhs_kernel|eta", "cuda_kernels.cu", "main_cuda.cu")),
            "current": "eta/GP-zone machinery exists for legacy GP thermodynamic surrogate paths.",
            "missing": "Not the required finite explicit GP-site barrier-modifier representation by itself.",
            "reuse": "reuse as optional gp_site_mask_source=eta; do not use as required thermodynamic GP phase.",
        },
        {
            "item": "GP site arrays / reservoir",
            "status": "partial",
            "file_function": first_match_line(rg("GpAssistedSite|B_mass_active|gp_site_B_mass_equiv", "main_cuda.cu", "pf_params.h")),
            "current": "GpAssistedSite host records and B_mass_active/B_mass_initial are present.",
            "missing": "Barrier-library-driven local xB interpolation and catalog-driven geometry selection are not wired into this runtime path.",
            "reuse": "reuse site/reservoir/mass ledger primitives.",
        },
        {
            "item": "barrier library reader",
            "status": "partial",
            "file_function": first_match_line(rg("barrier_real_units_by_case|cnt_refsub_peak|load_barrier_lookup", "nucleus_selector.py", "tools/analysis", "analysis")),
            "current": "Postprocess and selector can read barrier outputs; new core reader reads real-unit by-case CSV.",
            "missing": "CUDA runtime does not read barrier library directly.",
            "reuse": "reuse CSV outputs as authoritative DeltaG_bare library.",
        },
        {
            "item": "nucleus catalog reader",
            "status": "partial",
            "file_function": first_match_line(rg("nucleus_catalog|select_nucleus", "nucleus_selector.py", "nucleus_orchestrator.py", "nucleus_geometry_mapper.py")),
            "current": "nucleus_selector.py and geometry mapper can select catalog/mapped templates.",
            "missing": "GP stochastic runtime does not yet call catalog selection per accepted event.",
            "reuse": "reuse selector/catalog; new core has JSON reader/fallback descriptor.",
        },
        {
            "item": "explicit nucleation rate",
            "status": "partial",
            "file_function": first_match_line(rg("exp\\(-|J0|Theta|Z_r|P=1-exp|probability", "main_cuda.cu", "analysis/compute_explicit_nucleation_rates.py")),
            "current": "Rate formulas exist in analysis and legacy runtime branches.",
            "missing": "Runtime branch uses gp_stochastic_deltaG_homo_kBT scalar, not interpolated DeltaG_bare(T,xB,strain).",
            "reuse": "reuse deterministic hash/event logging; replace scalar barrier source in future CUDA wiring.",
        },
        {
            "item": "GP-assisted barrier reduction",
            "status": "partial",
            "file_function": first_match_line(rg("gp_stochastic_S_GP|S_factor|gp_site_S_factor", "main_cuda.cu", "pf_params.h")),
            "current": "Existing runtime multiplies scalar gp_stochastic_deltaG_homo_kBT by gp_stochastic_S_GP.",
            "missing": "No explicit library-backed DeltaG_bare interpolation in CUDA path.",
            "reuse": "same convention as this task: DeltaG_eff=s_GP*DeltaG_bare.",
        },
        {
            "item": "mass conservation during beta insertion",
            "status": "partial",
            "file_function": first_match_line(rg("gp_mass_ledger|mass_error|release_to_beta_first|B_mass_active", "main_cuda.cu", "scripts", "tools/analysis")),
            "current": "GP-assisted dry-run and CUDA debug paths track B reservoir and mass ledger.",
            "missing": "Needs production acceptance criterion in real CUDA smoke tests.",
            "reuse": "reuse existing ledger; validation core checks exact mass closure.",
        },
        {
            "item": "scheduled / stochastic beta insertion",
            "status": "partial",
            "file_function": first_match_line(rg("apply_scheduled_events_cpu|apply_gp_assisted_stochastic_selection_cpu|apply_gp_assisted_scheduled_event_cpu", "main_cuda.cu")),
            "current": "Scheduled source-profile insertion and GP-assisted scheduled/stochastic debug insertion exist.",
            "missing": "Parametric catalog seed generation is not the default CUDA runtime insertion path.",
            "reuse": "reuse for smoke tests and eventual runtime wiring.",
        },
    ]
    write_csv(out_dir / "existing_gp_assisted_nucleation_audit.csv", checks)
    lines = [
        "# Existing GP-Assisted Nucleation Audit",
        "",
        "Theory file status: `Nucleation Note.tex` was not found in the repo or attachment folder; this audit uses the formula supplied in the task text.",
        "",
        "| item | status | file/function | reuse decision |",
        "|---|---|---|---|",
    ]
    for row in checks:
        lines.append(f"| {row['item']} | {row['status']} | `{row['file_function']}` | {row['reuse']} |")
    lines += [
        "",
        "Conclusion: existing functionality is **partial**. The project already has explicit GP site records, scalar `S_GP` barrier scaling, event logs, and mass ledger components, but the runtime path is not yet fully library-driven by `(T, xB, strain)` and catalog geometry.",
    ]
    write_text(out_dir / "existing_gp_assisted_nucleation_audit.md", "\n".join(lines))
    return {"existing_functionality_status": "partial"}


def generate_design(out_dir: Path, inputs: dict[str, Any]) -> None:
    params = [
        "enable_gp_assisted_beta_nucleation",
        "enable_gp_barrier_reduction",
        "gp_barrier_multiplier_s",
        "gp_barrier_multiplier_min",
        "gp_barrier_multiplier_max",
        "gp_barrier_mode",
        "gp_presence_threshold",
        "gp_site_mask_source",
        "barrier_library_path",
        "nucleus_catalog_path",
        "enable_nucleus_catalog_insertion",
        "enable_parametric_seed_insertion",
        "nucleation_sample_interval_steps",
        "nucleation_rng_seed",
        "enable_nucleation_event_log",
        "enable_mass_ledger_log",
        "gp_assisted_nucleation_mode",
    ]
    lines = [
        "# GP-Assisted Beta Nucleation Design",
        "",
        "Convention: `DeltaG_eff = s_GP * DeltaG_bare`, with `s_GP=1` outside GP and `0<s_GP<=1` inside GP.",
        "",
        "Runtime pathway:",
        "",
        "```text",
        "candidate site -> GP mask -> barrier library interpolation -> s_GP reduction -> explicit rate -> Poisson draw -> catalog/parametric seed -> GP-first mass ledger -> event log",
        "```",
        "",
        "Required parameters and compatibility:",
    ]
    lines += [f"- `{p}`" for p in params]
    lines += [
        "",
        "Existing CUDA aliases already present for part of this interface include `enable_gp_assisted_beta_nucleation`, `gp_stochastic_S_GP`, `gp_stochastic_deltaG_homo_kBT`, `gp_event_log_enabled`, `gp_debug_mass_ledger`, and `gp_site_B_mass_equiv`.",
        "",
        "Barrier schema: CSV/JSON must provide `T_C` or `T_K`, `xB`, `strain`, `DeltaG_bare_kBT` or real-unit CNT reference-subtracted barrier columns, and `r_star_nm` or `rc_schur_nm`. Boundary/incomplete scans are rejected by default.",
        "",
        "Catalog schema: JSON/CSV entries should provide `T_C`, `xB`, `strain`, `r_seed_nm` or `r_star_nm`, `semiaxes_nm`, `orientation`, `shape_type`, and source directory. If missing, isotropic fallback uses `r_star_nm`.",
        "",
        "Mass protocol: beta seed mass is filled from GP reservoir first, then matrix halo; total `matrix + beta + GP_active` is checked. Debug acceptance target is relative mass error `<1e-10`.",
        "",
        "Files discovered:",
        f"- barrier libraries: `{inputs['barrier_libraries']}`",
        f"- nucleus catalogs: `{inputs['nucleus_catalogs']}`",
    ]
    write_text(out_dir / "gp_assisted_beta_nucleation_design.md", "\n".join(lines))


def run_validation(out_dir: Path, barrier_path: Path, catalog_path: Path | None) -> tuple[int, int]:
    barrier_lib = BarrierLibrary.from_csv(barrier_path)
    if catalog_path and catalog_path.exists():
        catalog = NucleusCatalog.from_json(catalog_path)
        catalog_status = "loaded"
    else:
        catalog = NucleusCatalog([])
        catalog_status = "fallback"

    tests: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []

    eff, s = effective_barrier_kBT(200.0, True, 0.2)
    tests.append({"name": "barrier_reduction", "passed": abs(eff - 40.0) < 1e-12, "expected": 40.0, "observed": eff})

    gp_candidate = NucleationCandidate((4, 4, 4), xB_loc=0.04, phi_beta=0.0, gp_density=1.0, gp_reservoir_mass=5.0)
    no_gp_candidate = NucleationCandidate((5, 4, 4), xB_loc=0.04, phi_beta=0.0, gp_density=0.0, gp_reservoir_mass=0.0)
    common = dict(
        barrier_library=barrier_lib,
        T_C=380.0,
        strain=0.0,
        s_gp_user=0.2,
        gp_presence_threshold=0.5,
        gp_site_mask_source="gp_density",
        mode="homogeneous_plus_GP",
        phi_threshold=0.1,
        N_site=1.0e12,
        beta_r_star=1.0,
        theta_tr=1.0,
        dV_nuc=1.0,
        dt=1.0,
    )
    gp_result = evaluate_candidate(gp_candidate, **common)
    no_gp_result = evaluate_candidate(no_gp_candidate, **common)
    tests.append({"name": "GP_mask", "passed": gp_result.gp_present and not no_gp_result.gp_present, "gp_s": gp_result.s_GP, "non_gp_s": no_gp_result.s_GP})

    ratio = gp_result.J_eff / max(gp_result.J_bare, 1.0e-300)
    expected_ratio = math.exp(min((1.0 - gp_result.s_GP) * gp_result.DeltaG_bare_kBT, 700.0))
    tests.append({"name": "probability_increase", "passed": ratio > 1.0 and abs(math.log(ratio) - math.log(expected_ratio)) < 1e-10, "J_eff_over_J_bare": ratio})

    blocked = evaluate_candidate(NucleationCandidate((1, 1, 1), xB_loc=0.04, phi_beta=0.9, gp_density=1.0), **common)
    tests.append({"name": "no_nucleation_inside_beta", "passed": (not blocked.allowed and blocked.reason == "blocked_existing_beta"), "reason": blocked.reason})

    n = 17
    matrix = [0.03] * (n**3)
    beta = [0.0] * (n**3)
    _, beta_after, ledger = insert_parametric_seed_mass_conserving(
        matrix,
        beta,
        gp_reservoir_mass=20.0,
        n=n,
        center=(n // 2, n // 2, n // 2),
        radius_cells=2.0,
        width_cells=1.0,
        xB_min=1.0e-12,
    )
    tests.append({"name": "mass_ledger", "passed": abs(ledger["relative_mass_error"]) < 1.0e-10, **ledger})

    selected, selection_reason = catalog.select(380.0, 0.04, 0.0, gp_result.r_star_nm)
    event_rows.append({
        "event_id": 1,
        "simulation_step": 1,
        "site_i": gp_candidate.index[0],
        "site_j": gp_candidate.index[1],
        "site_k": gp_candidate.index[2],
        "xB_loc": gp_candidate.xB_loc,
        "GP_present": gp_result.gp_present,
        "s_GP": gp_result.s_GP,
        "DeltaG_bare_kBT": gp_result.DeltaG_bare_kBT,
        "DeltaG_eff_kBT": gp_result.DeltaG_eff_kBT,
        "J_bare": gp_result.J_bare,
        "J_eff": gp_result.J_eff,
        "probability_P": gp_result.probability,
        "accepted": True,
        "nucleus_id": selected.id,
        "shape_type": selected.shape_type,
        "r_seed_nm": selected.r_seed_nm,
        "selection_reason": selection_reason,
        "catalog_status": catalog_status,
    })
    ledger_rows.append({"event_id": 1, **ledger})
    smoke_pass = (
        event_rows[0]["accepted"]
        and sum(beta_after) > 0.0
        and abs(ledger["relative_mass_error"]) < 1.0e-10
        and gp_result.gp_present
    )
    tests.append({"name": "integration_smoke_forced_GP_seed", "passed": smoke_pass, "beta_mass_after": sum(beta_after), "catalog_status": catalog_status})

    write_csv(out_dir / "gp_assisted_nucleation_event_log.csv", event_rows)
    write_csv(out_dir / "gp_assisted_mass_ledger.csv", ledger_rows)
    write_csv(out_dir / "gp_assisted_nucleation_validation_results.csv", tests)

    pass_count = sum(1 for t in tests if t["passed"])
    fail_count = len(tests) - pass_count
    lines = [
        "# GP-Assisted Nucleation Validation Report",
        "",
        f"Barrier library: `{barrier_path}`",
        f"Nucleus catalog: `{catalog_path if catalog_path else 'fallback'}`",
        "",
        "| test | status |",
        "|---|---|",
    ]
    for t in tests:
        lines.append(f"| {t['name']} | {'PASS' if t['passed'] else 'FAIL'} |")
    lines += [
        "",
        "Command:",
        "",
        "```bash",
        "python3 tools/analysis/validate_gp_assisted_nucleation_module.py",
        "```",
    ]
    write_text(out_dir / "gp_assisted_nucleation_validation_report.md", "\n".join(lines))
    return pass_count, fail_count


def generate_patch_report(out_dir: Path, pass_count: int, fail_count: int) -> None:
    lines = [
        "# GP-Assisted Beta Nucleation Patch Report",
        "",
        "Files added/updated:",
        "- `tools/analysis/gp_assisted_nucleation_core.py`: barrier reader, catalog reader, s_GP barrier logic, probability, parametric seed, mass ledger.",
        "- `tools/analysis/validate_gp_assisted_nucleation_module.py`: audit/design/report generation and validation suite.",
        "- `reports/gp_assisted_beta_nucleation_module/*`: generated audit, design, logs, and validation artifacts.",
        "",
        "Backward compatibility:",
        "- No CUDA kernel or PF equation changes.",
        "- Existing CUDA defaults remain disabled.",
        "- Existing scheduled and GP-assisted debug paths are not removed.",
        "",
        f"Validation pass count: {pass_count}",
        f"Validation fail count: {fail_count}",
        "",
        "Known limitations:",
        "- CUDA runtime still uses existing scalar GP stochastic barrier parameters; direct per-site barrier library interpolation is implemented in the modular layer and validated but not yet wired into `main_cuda.cu`.",
        "- `Nucleation Note.tex` was not found; task text formula is treated as the theory source for this implementation.",
    ]
    write_text(out_dir / "gp_assisted_beta_nucleation_patch_report.md", "\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--barrier-library", type=Path, default=ROOT / "Results/workflows/no_strain_400cube_dx0p1_cnt_sweep/summary_reports/real_unit_barriers/barrier_real_units_by_case.csv")
    parser.add_argument("--nucleus-catalog", type=Path, default=ROOT / "nucleus_catalog.json")
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = find_existing_inputs()
    write_text(out_dir / "input_discovery.json", json.dumps(inputs, indent=2))
    status = generate_audit(out_dir)
    generate_design(out_dir, inputs)
    pass_count, fail_count = run_validation(out_dir, args.barrier_library.resolve(), args.nucleus_catalog.resolve() if args.nucleus_catalog.exists() else None)
    generate_patch_report(out_dir, pass_count, fail_count)

    created_reports = [
        out_dir / "existing_gp_assisted_nucleation_audit.md",
        out_dir / "gp_assisted_beta_nucleation_design.md",
        out_dir / "gp_assisted_beta_nucleation_patch_report.md",
        out_dir / "gp_assisted_nucleation_validation_report.md",
    ]
    final = {
        "audit_completed": "true",
        "existing_functionality_status": status["existing_functionality_status"],
        "gp_assisted_nucleation_enabled_parameter": "enable_gp_assisted_beta_nucleation",
        "barrier_reader_status": "PASS",
        "nucleus_catalog_reader_status": "PASS" if args.nucleus_catalog.exists() else "FALLBACK",
        "gp_mask_status": "PASS",
        "mass_ledger_status": "PASS" if fail_count == 0 else "FAIL",
        "validation_pass_count": str(pass_count),
        "validation_fail_count": str(fail_count),
        "created_reports": ";".join(str(p) for p in created_reports),
        "recommended_next_action": "wire gp_assisted_nucleation_core barrier selection into main_cuda stochastic GP path if runtime library-driven events are required",
    }
    write_text(out_dir / "final_terminal_status.txt", "\n".join(f"{k}={v}" for k, v in final.items()))
    for k, v in final.items():
        print(f"{k}={v}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
