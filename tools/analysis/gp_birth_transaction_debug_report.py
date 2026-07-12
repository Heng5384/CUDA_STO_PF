#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "gp_birth_transaction_debug"
LONG_DIR = ROOT / "reports" / "gp_after_quench_long_test"
BUILD_STDOUT = ROOT / "reports" / "gp_birth_transaction_debug" / "workstation_build_stdout.log"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def write(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text)


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def build_failure_summary() -> None:
    acceptance = read_text(LONG_DIR / "after_quench_long_test_acceptance_report.md")
    stage_a = read_text(LONG_DIR / "stageA_1000step_report.md")
    ledger = read_text(LONG_DIR / "ledger_diagnostics_report.md")
    jgp = read_text(LONG_DIR / "JGP_source_tracking_report.md")
    summary = "\n".join(
        [
            "# Failure Reproduction Summary",
            "",
            "Pre-fix evidence from the workstation long test:",
            "",
            "- `T450`: no accepted new GP birth, 1000/1000 completed, zero reconstructed drift.",
            "- `T380`: exactly one accepted new GP birth, run later drifted badly.",
            "- `T400`: exactly one accepted new GP birth, run later drifted badly.",
            "- `JGP` source mode stayed `from_current_mean_xB_alpha` throughout.",
            "- Legacy GP storage stayed disabled and GP growth stayed disabled.",
            "",
            "Direct report excerpts:",
            "",
            "## Acceptance",
            acceptance.strip(),
            "",
            "## Stage A",
            stage_a.strip(),
            "",
            "## Ledger Diagnostics",
            ledger.strip(),
            "",
            "## JGP Source Tracking",
            jgp.strip(),
            "",
            "Conclusion: the failure is conditioned on accepted new GP birth events; the after-quench APT initial state itself is stable when no new birth occurs.",
        ]
    )
    write(REPORT_DIR / "failure_reproduction_summary.md", summary)


def build_code_audit() -> None:
    text = "\n".join(
        [
            "# GP Birth Transaction Code Path Audit",
            "",
            "## Current path",
            "",
            "1. New GP birth acceptance entry:",
            f"- file: `{ROOT / 'main_cuda.cu'}`",
            "- function: `apply_gp_literature_births_cpu`",
            "- lines: `7635-7724`",
            "- role: resolves `J_GP`, samples or forces births, chooses site coordinates, calls the append transaction.",
            "",
            "2. Site creation and requested inventory resolution:",
            f"- file: `{ROOT / 'main_cuda.cu'}`",
            "- function: `gp_literature_requested_birth_mass`",
            "- lines: `7402-7422`",
            "- role: maps runtime birth policy to requested GP inventory.",
            "",
            "3. New birth transaction implementation:",
            f"- file: `{ROOT / 'main_cuda.cu'}`",
            "- function: `gp_literature_append_birth_site_host`",
            "- lines: `7424-7561`",
            "- role: snapshots pre-transaction ledger, computes requested mass, performs smooth depletion, splits initial/new GP ledgers, writes transaction diagnostics, and enforces per-transaction mass conservation.",
            "",
            "4. Matrix depletion implementation:",
            f"- file: `{ROOT / 'main_cuda.cu'}`",
            "- function: `gp_apply_smooth_local_depletion_for_site_host`",
            "- lines: `6524-6621`",
            "- role: computes local removable matrix mass, smooth depletion kernel, actual removed mass, and shortage diagnostics.",
            "",
            "5. Initial AQ inventory construction:",
            f"- file: `{ROOT / 'main_cuda.cu'}`",
            "- function: `build_gp_assisted_sites_from_params`",
            "- lines: `6149-6205` plus the inventory assignment immediately below in the same function",
            "- role: constructs after-quench initial GP sites and stores only excess inventory above the matrix far-field baseline.",
            "",
            "## Audit answers",
            "",
            "- Where new GP birth is accepted: `apply_gp_literature_births_cpu -> gp_literature_append_birth_site_host`.",
            "- Where new GP radius is assigned: `init_gp_assisted_site_at` sets `site.radius_nm = gp_marker_core_radius_nm`; the created site inherits that before depletion and ledger insertion.",
            "- Where requested GP inventory is computed now: `gp_literature_requested_birth_mass`.",
            "- Where matrix depletion is applied: `gp_apply_smooth_local_depletion_for_site_host`.",
            "- Where actual removed matrix mass is measured: the same depletion function returns `removed_total` and now also exports `local_available_matrix_mass` and `target_mass` through `GpSmoothDepletionDiag`.",
            "- Where actual/new inventory separation is enforced: `gp_assisted_split_active_mass_by_origin` together with `rt->initial_gp_site_count` inside `gp_literature_append_birth_site_host`.",
            "",
            "## Pre-fix classification",
            "",
            "`UNKNOWN_NEEDS_INSTRUMENTATION` moved to a concrete root cause: the runtime literature birth path reused the legacy `gp_site_B_mass_equiv` request mass (`~442`) for new GP births, instead of a GP-marker excess inventory consistent with the after-quench `xB_far -> xB_GP` ledger. The transaction itself already wrote back `removed_mass`, but the requested birth target was physically mismatched to the AQ GP marker problem.",
        ]
    )
    write(REPORT_DIR / "gp_birth_transaction_code_path_audit.md", text)


def build_policy_report() -> None:
    text = "\n".join(
        [
            "# GP Birth Inventory Policy Report",
            "",
            "Implemented runtime policies:",
            "",
            "- `actual_removed_mass`:",
            "  new literature GP births request an excess-style inventory target and add only the matrix mass actually removed by the smooth depletion transaction.",
            "",
            "- `excess_over_local_background`:",
            "  new literature GP births request excess inventory above local background and reject shortage cases instead of silently reducing them.",
            "",
            "- `legacy_requested_mass`:",
            "  preserves the older behavior of using `gp_site_B_mass_equiv` as the request hint.",
            "",
            "Current production recommendation for after-quench APT runs: `gp_birth_inventory_policy=actual_removed_mass`.",
            "",
            "Inventory policy before this change: implicit legacy request mass path.",
            "Inventory policy after this change: explicit, logged, and validated by parameter value.",
        ]
    )
    write(REPORT_DIR / "gp_birth_inventory_policy_report.md", text)


def build_debug_mode_report() -> None:
    text = "\n".join(
        [
            "# Debug Mode Implementation Report",
            "",
            "Added deterministic single-birth controls:",
            "",
            "- `gp_birth_debug_force_single_event`",
            "- `gp_birth_debug_force_step`",
            "- `gp_birth_debug_force_position_mode`",
            "- `gp_birth_debug_disable_poisson_randomness`",
            "- `gp_birth_debug_max_events_total`",
            "",
            "Implemented in:",
            f"- file: `{ROOT / 'main_cuda.cu'}`",
            "- function: `apply_gp_literature_births_cpu`",
            "- lines: `7643-7705`",
            "",
            "Generated debug params file:",
            f"- `{ROOT / 'params/gp_birth_transaction_debug/gp_AQ_Yu_APT_force_one_GP_birth_T400.params'}`",
            "",
            "This mode forces exactly one new GP birth at a known step and position without changing AQ initialization or JGP source mode.",
        ]
    )
    write(REPORT_DIR / "debug_mode_implementation_report.md", text)


def build_transaction_diag_report() -> None:
    text = "\n".join(
        [
            "# Transaction Diagnostics Instrumentation Report",
            "",
            "Added transaction-level runtime print blocks:",
            "",
            "- `GP_BIRTH_TRANSACTION_BEGIN ... GP_BIRTH_TRANSACTION_END`",
            "",
            "These blocks now print:",
            "- step / site / position",
            "- radius / volume",
            "- local background `xB` and target `xB_GP`",
            "- requested mass, locally available mass, actual matrix removal, actual GP inventory addition",
            "- separate `M_GP_initial_existing` and `M_GP_new_existing` before/after",
            "- full matrix / beta / total ledger before and after",
            "- transaction absolute and relative mass error",
            "- pre/post `xB_alpha` extrema",
            "",
            "Implementation location:",
            f"- file: `{ROOT / 'main_cuda.cu'}`",
            "- function: `gp_literature_append_birth_site_host`",
            "- lines: `7480-7556`",
        ]
    )
    write(REPORT_DIR / "transaction_diagnostics_instrumentation_report.md", text)


def build_build_report() -> None:
    text = "\n".join(
        [
            "# Build Report",
            "",
            "- target: workstation build only",
            "- build command: `make main_cuda NVCC=/usr/local/cuda-12.9/bin/nvcc CUDA_ROOT=/usr/local/cuda-12.9 -j1`",
            "- result: `PASS`",
            "- note: compile completed after adding a forward declaration for `gp_assisted_flat_index`.",
            "- remaining runtime validation: pending because the workstation SSH session began closing connections during the subsequent smoke-run step.",
        ]
    )
    write(REPORT_DIR / "build_report.md", text)


def build_pending_runtime_reports() -> None:
    forced = "\n".join(
        [
            "# Forced Single Birth Debug Report",
            "",
            "- status: `PENDING_WORKSTATION_RECONNECT`",
            "- reason: workstation build completed, but the SSH server started closing new sessions before the forced-single-birth smoke log could be retrieved.",
            f"- prepared params: `{ROOT / 'params/gp_birth_transaction_debug/gp_AQ_Yu_APT_force_one_GP_birth_T400.params'}`",
            "- expected acceptance targets remain:",
            "  - transaction mass error <= `1e-12`",
            "  - global mass drift <= `1e-10`",
            "  - no NaN/Inf",
            "  - `M_GP_initial_existing` unchanged",
            "  - `M_GP_new` grows only by accepted new-birth inventory",
        ]
    )
    stagea = "\n".join(
        [
            "# Stage A After-Fix Report",
            "",
            "- status: `PENDING_WORKSTATION_RECONNECT`",
            "- planned cases: `T380`, `T400`, `T450`",
            "- policy to be used: `gp_birth_inventory_policy=actual_removed_mass`",
            "- blocker: workstation SSH currently closes new sessions before the post-build smoke and rerun stage can be launched/retrieved.",
        ]
    )
    acceptance = "\n".join(
        [
            "# GP Birth Transaction Debug Acceptance Report",
            "",
            "- status: `PASS_ROOT_CAUSE_IDENTIFIED_FIX_PENDING`",
            "",
            "## Answers",
            "",
            "1. Root cause identified:",
            "   the literature new-GP birth path was still using the legacy `gp_site_B_mass_equiv` request mass as the birth target, instead of an after-quench GP marker inventory derived from the `xB_far -> xB_GP` excess ledger.",
            "",
            "2. Was requested GP inventory added instead of actual removed matrix mass?",
            "   not in the final site ledger writeback; the transaction already wrote back `removed_mass`. The real mismatch was the requested target itself.",
            "",
            "3. Was full `xB_GP*V_GP` used where excess inventory was required?",
            "   no; the pre-fix path was worse in a different way: it reused a legacy fixed request mass (`gp_site_B_mass_equiv ~ 442`) unrelated to the AQ excess marker inventory.",
            "",
            "4. Does new GP birth now preserve the per-transaction `xB_tot` ledger contract in code?",
            "   yes at the transaction layer: requested policy is explicit, actual removal is measured, and separate before/after ledgers are printed for each birth.",
            "",
            "5. Are initial AQ GP inventory and new GP birth inventory separated?",
            "   yes, through `rt->initial_gp_site_count` and split mass accounting.",
            "",
            "6. Forced single birth conserve-mass run:",
            "   pending workstation reconnect.",
            "",
            "7. Stage A after fix:",
            "   pending workstation reconnect.",
            "",
            "8. JGP source mode remains correct?",
            "   yes; no change was made to `from_current_mean_xB_alpha` resolution.",
            "",
            "9. GP growth and legacy GP storage still disabled?",
            "   yes; no change was made to those controls.",
            "",
            "10. Ready to retry Stage B?",
            "   not yet; forced single-birth and Stage A rerun still need workstation runtime acceptance.",
        ]
    )
    write(REPORT_DIR / "forced_single_birth_debug_report.md", forced)
    write(REPORT_DIR / "stageA_after_fix_report.md", stagea)
    write(REPORT_DIR / "gp_birth_transaction_debug_acceptance_report.md", acceptance)


def main() -> None:
    ensure_dir(REPORT_DIR)
    build_failure_summary()
    build_code_audit()
    build_policy_report()
    build_debug_mode_report()
    build_transaction_diag_report()
    build_build_report()
    build_pending_runtime_reports()


if __name__ == "__main__":
    main()
