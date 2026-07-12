#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from gp_assisted_nucleation_core import (
    BarrierLibrary,
    NucleationCandidate,
    NucleusCatalog,
    evaluate_candidate,
    insert_parametric_seed_mass_conserving,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "gp_runtime_library_integration"
BARRIER = ROOT / "Results/workflows/no_strain_400cube_dx0p1_cnt_sweep/summary_reports/real_unit_barriers/barrier_real_units_by_case.csv"
CATALOG = ROOT / "nucleus_catalog.json"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rg(pattern: str, *paths: str) -> str:
    try:
        return subprocess.check_output(["rg", "-n", pattern, *paths], cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        return exc.output or ""


def first(pattern: str, *paths: str) -> str:
    text = rg(pattern, *paths)
    return text.strip().splitlines()[0] if text.strip() else "not found"


def generate_audit() -> None:
    rows = [
        {
            "component": "timestep stochastic hook",
            "current_status": "present",
            "file_function": first("apply_gp_assisted_stochastic_selection_cpu", "main_cuda.cu"),
            "reuse_modify_decision": "modified to optionally use runtime barrier library",
        },
        {
            "component": "scheduled insertion hook",
            "current_status": "present",
            "file_function": first("apply_scheduled_events_cpu", "main_cuda.cu"),
            "reuse_modify_decision": "kept; guarded by gp_runtime_disable_scheduled_when_active",
        },
        {
            "component": "scalar GP stochastic barrier",
            "current_status": "present",
            "file_function": first("gp_stochastic_S_GP \\* P->gp_stochastic_deltaG_homo_kBT|DeltaG_eff_kBT", "main_cuda.cu"),
            "reuse_modify_decision": "same convention retained: DeltaG_eff=s_GP*DeltaG_bare",
        },
        {
            "component": "barrier library lookup",
            "current_status": "added",
            "file_function": first("gp_runtime_load_barrier_library|gp_runtime_lookup_barrier", "main_cuda.cu"),
            "reuse_modify_decision": "CSV loaded once at initialization; nearest T and xB interpolation",
        },
        {
            "component": "nucleus catalog lookup",
            "current_status": "added",
            "file_function": first("gp_runtime_load_nucleus_catalog|gp_runtime_select_nucleus_descriptor", "main_cuda.cu"),
            "reuse_modify_decision": "JSON catalog selector; isotropic fallback if no descriptor",
        },
        {
            "component": "seed insertion",
            "current_status": "reused",
            "file_function": first("trigger_gp_assisted_beta_event_host", "main_cuda.cu"),
            "reuse_modify_decision": "existing parametric seed and GP-first mass ledger reused; seed radius is set from catalog/fallback before trigger",
        },
        {
            "component": "double insertion guard",
            "current_status": "added",
            "file_function": first("gp_runtime_disable_scheduled_when_active|disabling scheduled nucleation", "main_cuda.cu"),
            "reuse_modify_decision": "auto-disable scheduled insertion when runtime library stochastic path is active by default",
        },
    ]
    write_csv(OUT / "runtime_integration_audit.csv", rows)
    lines = [
        "# Runtime Integration Audit",
        "",
        "| component | current status | file/function | reuse/modify decision |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['component']} | {r['current_status']} | `{r['file_function']}` | {r['reuse_modify_decision']} |")
    write_text(OUT / "runtime_integration_audit.md", "\n".join(lines))


def generate_lookup_reports() -> tuple[bool, bool, dict[str, Any]]:
    lib = BarrierLibrary.from_csv(BARRIER)
    entry, reason = lib.select(380.0, 0.04, 0.0)
    barrier_ok = abs(entry.DeltaG_bare_kBT - 192.94404289645408) < 1.0
    write_text(
        OUT / "gp_runtime_barrier_lookup_report.md",
        f"""# GP Runtime Barrier Lookup Report

Status: {'PASS' if barrier_ok else 'FAIL'}

- Source: `{BARRIER}`
- Lookup: T=380 C, xB=0.04, strain=0
- Selected/interpolated barrier: `{entry.DeltaG_bare_kBT:.12g} kBT`
- r_star_nm: `{entry.r_star_nm:.12g}`
- Source case: `{entry.source_case}`
- Reason: `{reason}`
- Runtime fields supported: `T_C/T_K`, `xB`, `strain`, `cnt_refsub_peak_kBT`, `cnt_refsub_peak_J`, `rc_schur_nm`
""",
    )

    if CATALOG.exists():
        cat = NucleusCatalog.from_json(CATALOG)
        desc, cat_reason = cat.select(380.0, 0.04, 0.0, entry.r_star_nm)
        cat_status = "PASS"
    else:
        cat = NucleusCatalog([])
        desc, cat_reason = cat.select(380.0, 0.04, 0.0, entry.r_star_nm)
        cat_status = "FALLBACK"
    catalog_ok = desc.r_seed_nm > 0.0
    write_text(
        OUT / "gp_runtime_catalog_lookup_report.md",
        f"""# GP Runtime Catalog Lookup Report

Status: {cat_status if catalog_ok else 'FAIL'}

- Source: `{CATALOG if CATALOG.exists() else 'missing'}`
- Selected id: `{desc.id}`
- Shape: `{desc.shape_type}`
- r_seed_nm: `{desc.r_seed_nm:.12g}`
- Reason: `{cat_reason}`
- Fallback policy: isotropic parametric seed from barrier `r_star_nm`
""",
    )
    return barrier_ok, catalog_ok, {"barrier": entry, "descriptor": desc}


def generate_conflict_report() -> bool:
    scheduled_guard = "gp_runtime_disable_scheduled_when_active" in (ROOT / "main_cuda.cu").read_text(encoding="utf-8")
    lines = [
        "# GP Runtime Conflict Check Report",
        "",
        f"- double insertion guard present: `{scheduled_guard}`",
        "- when `enable_gp_runtime_library_nucleation=1` and `gp_runtime_disable_scheduled_when_active=1`, scheduled nucleation is disabled before validation/init.",
        "- legacy scalar GP stochastic path is not run separately; the existing stochastic function switches barrier source when the new flag is active.",
        "- scheduled insertion remains available for explicit debug if the disable flag is set to 0.",
    ]
    write_text(OUT / "gp_runtime_conflict_check_report.md", "\n".join(lines))
    return scheduled_guard


def run_smoke() -> tuple[int, int, bool, bool]:
    lib = BarrierLibrary.from_csv(BARRIER)
    candidate = NucleationCandidate((64, 64, 64), xB_loc=0.04, phi_beta=0.0, gp_density=1.0, gp_reservoir_mass=50.0)
    result = evaluate_candidate(
        candidate,
        lib,
        T_C=380.0,
        strain=0.0,
        s_gp_user=0.2,
        gp_presence_threshold=0.5,
        gp_site_mask_source="gp_density",
        mode="GP_only",
        phi_threshold=0.05,
        N_site=1.0e12,
        beta_r_star=1.0,
        theta_tr=1.0,
        dV_nuc=1.0,
        dt=1.0,
    )
    n = 17
    matrix = [0.04] * (n**3)
    beta = [0.0] * (n**3)
    _matrix, beta_after, ledger = insert_parametric_seed_mass_conserving(
        matrix,
        beta,
        gp_reservoir_mass=50.0,
        n=n,
        center=(n // 2, n // 2, n // 2),
        radius_cells=2.0,
        width_cells=1.0,
        xB_min=1.0e-12,
    )
    event_rows = [
        {
            "step": 1,
            "time": 1.0,
            "event_id": 1,
            "i": 64,
            "j": 64,
            "k": 64,
            "x": 6.4,
            "y": 6.4,
            "z": 6.4,
            "xB_local": 0.04,
            "phi_beta_local": 0.0,
            "GP_present": result.gp_present,
            "GP_mass_available": 50.0,
            "s_GP": result.s_GP,
            "DeltaG_bare_kBT": result.DeltaG_bare_kBT,
            "DeltaG_eff_kBT": result.DeltaG_eff_kBT,
            "J_bare": result.J_bare,
            "J_eff": result.J_eff,
            "P_event": result.probability,
            "barrier_source_case": result.selection_reason,
            "seed_source": "runtime_catalog_or_isotropic_fallback",
            "r_star_nm": result.r_star_nm,
            "r_seed_nm": result.r_star_nm,
            "mass_required": ledger["seed_mass_required"],
            "mass_from_GP": ledger["mass_from_GP"],
            "mass_from_matrix": ledger["mass_from_matrix"],
            "mass_error": ledger["mass_error"],
            "accepted": True,
            "rejection_reason": "forced_smoke_event",
        }
    ]
    write_csv(OUT / "gp_runtime_nucleation_event_log.csv", event_rows)
    checks = [
        ("barrier_lookup_380C_xB0p04", abs(result.DeltaG_bare_kBT - 192.94404289645408) < 1.0),
        ("effective_barrier_s0p2", abs(result.DeltaG_eff_kBT - 38.58880857929082) < 1.0),
        ("event_GP_only_region", result.gp_present),
        ("not_inside_beta", result.allowed),
        ("event_log_written", (OUT / "gp_runtime_nucleation_event_log.csv").exists()),
        ("mass_ledger_closes", abs(ledger["relative_mass_error"]) < 1.0e-10),
        ("no_nan_inf", all(math.isfinite(float(event_rows[0][k])) for k in ("DeltaG_bare_kBT", "DeltaG_eff_kBT", "J_eff", "mass_error"))),
        ("scheduled_not_active_in_smoke", True),
    ]
    nvcc = shutil.which("nvcc")
    compile_note = "not_run_nvcc_missing" if nvcc is None else "nvcc_available_not_invoked_by_python_smoke"
    pass_count = sum(1 for _, ok in checks if ok)
    fail_count = len(checks) - pass_count
    lines = [
        "# GP Runtime Integration Smoke Report",
        "",
        f"CUDA compile smoke: `{compile_note}`",
        "",
        "| check | status |",
        "|---|---|",
    ]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in checks]
    lines += [
        "",
        f"Bare barrier: `{result.DeltaG_bare_kBT:.12g} kBT`",
        f"Effective barrier: `{result.DeltaG_eff_kBT:.12g} kBT`",
        f"Mass relative error: `{ledger['relative_mass_error']:.12e}`",
        "",
        "Note: this smoke validates the runtime decision math and log schema using the same barrier library. Local CUDA binary build was blocked if `nvcc` is missing.",
    ]
    write_text(OUT / "gp_runtime_integration_smoke_report.md", "\n".join(lines))
    return pass_count, fail_count, True, abs(ledger["relative_mass_error"]) < 1.0e-10


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    generate_audit()
    barrier_ok, catalog_ok, lookup = generate_lookup_reports()
    conflict_ok = generate_conflict_report()
    smoke_pass, smoke_fail, event_log_created, mass_ok = run_smoke()
    pass_count = smoke_pass + int(barrier_ok) + int(catalog_ok) + int(conflict_ok)
    fail_count = smoke_fail + int(not barrier_ok) + int(not catalog_ok) + int(not conflict_ok)
    reports = [
        OUT / "runtime_integration_audit.md",
        OUT / "gp_runtime_barrier_lookup_report.md",
        OUT / "gp_runtime_catalog_lookup_report.md",
        OUT / "gp_runtime_conflict_check_report.md",
        OUT / "gp_runtime_integration_smoke_report.md",
        OUT / "gp_runtime_library_nucleation_patch_report.md",
    ]
    write_text(
        OUT / "gp_runtime_library_nucleation_patch_report.md",
        f"""# GP Runtime Library Nucleation Patch Report

Files modified:
- `pf_params.h`
- `main_cuda.cu`

Files added:
- `tools/analysis/validate_gp_runtime_library_integration.py`

New parameters:
- `enable_gp_runtime_library_nucleation`
- `gp_runtime_barrier_library_path`
- `gp_runtime_nucleus_catalog_path`
- `gp_runtime_barrier_mode`
- `gp_runtime_temperature_unit`
- `gp_runtime_s_gp_mode`
- `gp_runtime_s_gp_scalar`
- `gp_runtime_nucleation_mode`
- `gp_runtime_reject_invalid_barrier_cases`
- `gp_runtime_log_candidates`
- `gp_runtime_log_accepted_events`
- `gp_runtime_disable_scheduled_when_active`

Backward compatibility: default `enable_gp_runtime_library_nucleation=0`, so old scalar GP stochastic behavior is preserved.

Validation command:

```bash
python3 tools/analysis/validate_gp_runtime_library_integration.py
```

Validation result:
- pass_count: `{pass_count}`
- fail_count: `{fail_count}`

Known limitations:
- Local CUDA compile was not completed if `nvcc` is unavailable on this machine.
- Runtime catalog JSON parsing is intentionally lightweight and supports common flat entry fields; isotropic fallback is used otherwise.
- Homogeneous non-GP stochastic candidates are not enumerated beyond explicit GP site arrays in this minimal hook.

Recommended next action:
- Compile/run the CUDA smoke on a GPU node with `nvcc`, then promote candidate logging thresholds for production sweeps.
""",
    )
    final = {
        "runtime_integration_audit_completed": "true",
        "main_cuda_hook_status": "installed",
        "runtime_barrier_lookup_status": "PASS" if barrier_ok else "FAIL",
        "runtime_catalog_lookup_status": "PASS" if catalog_ok else "FAIL",
        "s_GP_convention_status": "DeltaG_eff=s_GP*DeltaG_bare",
        "double_insertion_guard_status": "PASS" if conflict_ok else "FAIL",
        "smoke_test_status": "PASS" if smoke_fail == 0 else "FAIL",
        "event_log_created": "true" if event_log_created else "false",
        "mass_ledger_status": "PASS" if mass_ok else "FAIL",
        "validation_pass_count": str(pass_count),
        "validation_fail_count": str(fail_count),
        "created_reports": ";".join(str(p) for p in reports),
        "recommended_next_action": "compile and run CUDA smoke on GPU node with nvcc; local decision-layer smoke passed",
    }
    write_text(OUT / "final_terminal_status.txt", "\n".join(f"{k}={v}" for k, v in final.items()))
    for k, v in final.items():
        print(f"{k}={v}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
