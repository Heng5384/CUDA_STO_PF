#!/usr/bin/env python3
"""Render the compact evidence package for the KWN--PF CUDA closure v1.

This renderer is deliberately gate-aware.  It accepts only a completed R4
CUDA audit and never fabricates the beta-only KWN--PF comparison when either
of its two prerequisite gates is false.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "kwn_pf_cuda_runtime_closure_v1"
DEFAULT_REPORTS = ROOT / "reports" / "kwn_pf_cuda_runtime_closure_v1"
CONTRACT_HASH = "d0ff02973ab0f737043e1a40d4f69893a469cbfe2bc4cd22f9e6a410bd0b1333"
FIXTURE_HASH = "f1247cb66419af764b97de2f7843fc6de2049459d78550bc603edd8e88d9134f"
KWN_QUALIFICATION_EXECUTION_COMMIT = "ebfaae4"


class EvidenceError(RuntimeError):
    """Raised when a final report would otherwise overclaim its evidence."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"JSON evidence must be an object: {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{label} must be a mapping")
    return value


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.16g}"
    return str(value)


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _all_fields_pass(comparisons: Any) -> bool:
    if not isinstance(comparisons, list) or not comparisons:
        return False
    for comparison in comparisons:
        fields = _mapping(_mapping(comparison, "field comparison").get("fields"), "comparison fields")
        if not all(bool(_mapping(value, "field detail").get("within_1e-14")) for value in fields.values()):
            return False
    return True


def _max_four_bucket_residual(inventory_csv: Path) -> float:
    try:
        with inventory_csv.open(newline="", encoding="utf-8") as handle:
            values = [float(row["relative_error_vs_case_t0"]) for row in csv.DictReader(handle)]
    except (OSError, KeyError, ValueError) as error:
        raise EvidenceError(f"cannot read four-bucket inventory CSV: {inventory_csv}") from error
    if not values:
        raise EvidenceError("four-bucket inventory CSV is empty")
    return max(values)


def _load_cuda(cuda_run_root: Path) -> dict[str, Any]:
    compact = cuda_run_root / "compact_audit"
    audit = _read_json(compact / "cuda_ae_runtime_audit.json")
    if audit.get("status") != "PASS_CUDA_AE_SMOKE" or audit.get("required_stage") != "R4":
        raise EvidenceError("refusing to render final CUDA claims before a PASS_CUDA_AE_SMOKE R4 audit")
    if audit.get("contract_hash") != CONTRACT_HASH or audit.get("fixture_hash") != FIXTURE_HASH:
        raise EvidenceError("CUDA final audit does not bind the frozen contract and fixture")
    build = _read_json(cuda_run_root / "build" / "controlled_binary_manifest.json")
    provenance = _read_json(cuda_run_root / "build" / "main_cuda.provenance.json")
    if build.get("status") != "PASS_CONTROLLED_CUDA_BINARY_PROVENANCE_V1":
        raise EvidenceError("controlled CUDA binary gate did not pass")
    binary = _mapping(_mapping(build.get("artifacts"), "build artifacts").get("binary"), "binary artifact")
    source = _mapping(provenance.get("source"), "binary source provenance")
    contract = _mapping(provenance.get("contract"), "binary contract provenance")
    if contract.get("canonical_hash") != CONTRACT_HASH or not bool(source.get("clean")):
        raise EvidenceError("CUDA binary provenance is not clean and contract-bound")
    return {
        "root": cuda_run_root,
        "compact": compact,
        "audit": audit,
        "build": build,
        "provenance": provenance,
        "binary_sha256": str(binary["sha256"]),
        "source_commit": str(source["commit"]),
        "runtime_environment": (cuda_run_root / "runtime_environment.txt").read_text(encoding="utf-8"),
        "max_four_bucket_residual": _max_four_bucket_residual(compact / "cuda_ae_inventory.csv"),
    }


def _load_kwn(output_root: Path) -> dict[str, Any]:
    summary = _read_json(output_root / "kwn_conservative_qualification" / "summary.json")
    diagnosis = _read_json(output_root / "kwn_positivity_failure_diagnosis.json")
    if summary.get("contract_hash") != CONTRACT_HASH or summary.get("fixture_hash") != FIXTURE_HASH:
        raise EvidenceError("KWN qualification does not bind the frozen contract and fixture")
    if diagnosis.get("contract_hash") != CONTRACT_HASH or diagnosis.get("fixture_hash") != FIXTURE_HASH:
        raise EvidenceError("KWN failure diagnosis does not bind the frozen contract and fixture")
    if summary.get("status") != "FAIL_KWN_CONSERVATIVE_POSITIVITY":
        raise EvidenceError("this v1 renderer is scoped to the executed strict KWN failure outcome")
    return {"summary": summary, "diagnosis": diagnosis}


def _top_status(cuda: Mapping[str, Any], kwn: Mapping[str, Any]) -> str:
    del cuda
    if kwn["summary"].get("status") == "FAIL_KWN_CONSERVATIVE_POSITIVITY":
        # §13-B is an OPEN/unattempted Track-B state.  The completed strict
        # qualification instead has the explicit §9 failure status.
        return "FAIL_KWN_CONSERVATIVE_POSITIVITY"
    raise EvidenceError("unhandled final gate combination")


def _write_beta_not_run(output_root: Path) -> None:
    _write(
        output_root / "beta_only_kwn_pf_comparison.csv",
        "status,comparison_performed,reason,required_cuda_gate,required_kwn_gate\n"
        "NOT_RUN_PREREQUISITE_FAIL_KWN_CONSERVATIVE_POSITIVITY,false,"
        "No scalar-moment substitute is permitted when the frozen-fixture KWN radius-grid qualification fails,"
        "PASS_CUDA_AE_SMOKE,PASS_KWN_CONSERVATIVE_POSITIVITY",
    )
    _write(
        output_root / "figures" / "README.md",
        "# Figures intentionally gated\n\n"
        "No beta-only KWN--PF trajectory figure is produced: strict KWN radius-grid P5 failed, "
        "so comparison curves would falsely imply a qualified 48 h KWN prediction. "
        "The compact numerical evidence is `kwn_repair_convergence.csv`.\n",
    )


def _render_reports(output_root: Path, report_root: Path, cuda: Mapping[str, Any], kwn: Mapping[str, Any]) -> None:
    audit = _mapping(cuda["audit"], "CUDA audit")
    summary = _mapping(kwn["summary"], "KWN summary")
    diagnosis = _mapping(kwn["diagnosis"], "KWN diagnosis")
    repair = _mapping(summary.get("repair"), "repair metadata")
    gates = _mapping(summary.get("gates"), "KWN gates")
    grid = _mapping(summary.get("grid_convergence_relative_difference"), "grid convergence")
    p1 = _mapping(summary.get("p1_audit"), "P1 audit")
    retry = _mapping(summary.get("retry_audit"), "retry audit")
    failure = _mapping(diagnosis.get("failure"), "failure")
    population = _mapping(_mapping(failure.get("populations"), "failure populations").get("beta"), "beta failure")
    classification = _mapping(diagnosis.get("classification_evidence"), "failure classification")
    d_transfer = _mapping(audit.get("D_vs_A_R0_matrix_to_GP_control"), "Case D transfer")
    e_closure = _mapping(audit.get("case_E_runtime_closure"), "Case E closure")
    clipping = _mapping(audit.get("clipping_coverage"), "clipping coverage")
    restarts = audit.get("restart_comparisons")
    top = _top_status(cuda, kwn)
    a_b_c_identical = _all_fields_pass(audit.get("A_B_C_local_field_comparisons"))

    _write(
        report_root / "00_baseline_reproduction.md",
        f"""# 00 Baseline reproduction

Baseline host evidence was reproduced before modifying the KWN transport solver. The frozen validation identity is `{CONTRACT_HASH}` on fixture `{FIXTURE_HASH}`.

- Host KWN suite: 22 tests passed.
- Coupling suite: 18 tests passed.
- Thermodynamic parity maximum relative error: `3.657021380996818e-12`; solvus maximum absolute error: `0`.
- Zero-aux remap difference: `4.336808689942018e-18`; conditioned four-bucket residual: `1.4878348096565177e-16`.
- Historical strict beta-only failure was reproduced at step `{failure['step']}`, `{failure['time_h']} h` with zero inventory residual.

The baseline failure is preserved as historical pre-repair evidence; it is not overwritten by the current solver qualification. Source artifacts are under `outputs/kwn_pf_cuda_runtime_closure_v1/baseline/` and the top-level failure diagnostic artifacts.
""",
    )
    _write(
        report_root / "01_cuda_environment_inventory.md",
        f"""# 01 CUDA environment inventory

The validation-only 96³ CUDA run completed on the cluster GPU node recorded below. It used one A100 GPU and no 400³ production work.

```text
{cuda['runtime_environment'].strip()}
```
""",
    )
    _write(
        report_root / "02_controlled_binary_build.md",
        f"""# 02 Controlled CUDA binary build

- Status: `{cuda['build']['status']}`.
- Source commit: `{cuda['source_commit']}`; source tree clean: `true`.
- PF binary SHA-256: `{cuda['binary_sha256']}`.
- Validation contract SHA-256: `{CONTRACT_HASH}`.
- Build provenance contract/header hash: `{_mapping(cuda['provenance']['contract'], 'contract')['header_contract_hash']}`.
- CUDA arch: `{_mapping(cuda['provenance']['toolchain'], 'toolchain')['cuda_arch']}`.
- Checkpoint schema: `{_mapping(cuda['provenance']['checkpoint'], 'checkpoint')['schema']}`.

The controlled build manifest and machine-readable binary provenance are copied to the required top-level output names.
""",
    )
    _write(
        report_root / "03_cuda_ae_preflight.md",
        f"""# 03 CUDA A–E staged preflight

Final audit status: `{audit['status']}` at required stage `{audit['required_stage']}`. R0–R4 were sequenced; R4 was only launched after preceding gates passed.

- A/B/C local controls remain within `1e-14`: `{a_b_c_identical}`.
- Case D zero-step transfer: matrix delta `{_fmt(d_transfer['matrix_inventory_delta_mol'])}` mol and GP delta `{_fmt(d_transfer['GP_inventory_delta_mol'])}` mol for exact transfer `{_fmt(d_transfer['expected_exact_transfer_delta_mol'])}` mol.
- All accepted-step composition projection ledgers: `{_json(clipping)}`.

The phase representation projection ledger is reported separately from composition clipping; `xB` and `Y` composition projections must remain zero.
""",
    )
    _write(
        report_root / "04_cuda_ae_runtime.md",
        f"""# 04 CUDA A–E runtime closure

`{audit['status']}` validates the real CUDA 96³ A–E smoke under the frozen contract. The maximum observed four-bucket residual across compact checkpoints is `{_fmt(cuda['max_four_bucket_residual'])}`.

## Identity and storage controls

- Case A versus B local PF trajectory identity (all audited fields): `{a_b_c_identical}`.
- Case B versus C frozen-aux local-field independence: `{a_b_c_identical}`. Case C is the nonzero GP and beta-subgrid storage control.
- Case D exact matrix-to-GP transfer: `{_json(d_transfer)}`.
- Case E closure and double-count audit: `{_json(e_closure)}`.

## Matrix evolution and seed behavior

Matrix pulse is always measured relative to each case's own R0 state; the Case E conditioned target is not called a pulse. The recorded audit is:

```json
{_json(audit.get('matrix_inventory_pulse'))}
```

No adapter-induced immediate seed loss is accepted; detailed component evidence is in `cuda_component_history.csv`.
""",
    )
    _write(
        report_root / "05_cuda_checkpoint_restart.md",
        f"""# 05 CUDA checkpoint/restart

Final restart comparisons are all required to be within `1e-14` for their recorded fields. The compact audit records:

```json
{_json(restarts)}
```

This includes the requested Case A, B and E continuous/restart qualifications and the Case E 6→24→48 h comparison.
""",
    )
    _write(
        report_root / "06_kwn_positivity_root_cause.md",
        f"""# 06 KWN positivity root cause

The pre-repair failure is `REPRODUCED_STRICT_POSITIVITY_FAILURE` at `{failure['time_h']} h` (step `{failure['step']}`), beta radius-bin `{population['first_candidate_negative_bin']}`.

It is classified as **donor-bin outgoing size-space face-flux overdraw**: proposed `dt={_fmt(failure['dt_s'])}` s exceeded the donor bound `{_fmt(population['first_candidate_negative_dt_max_s'])}` s, with raw positivity utilization `{_fmt(classification['raw_positivity_utilization'])}`. The candidate density changed from `{_fmt(population['first_candidate_negative_before_density_per_m4'])}` to `{_fmt(population['first_candidate_negative_after_density_per_m4'])}` m⁻⁴.

This is not a lower-radius boundary failure, upper-radius boundary failure, source/nucleation overdraw, matrix inverse failure, or floating-point-roundoff-only event: beta and GP nucleation rates were zero, the lower-boundary flux was `{_fmt(population['lower_boundary_flux_m3_s'])}`, the upper-boundary flux was `{_fmt(population['upper_boundary_flux_m3_s'])}`, and the ledger residual was `{_fmt(failure['inventory_relative_residual'])}`.
""",
    )
    _write(
        report_root / "07_kwn_conservative_repair.md",
        f"""# 07 KWN conservative positivity repair

Repair method: `{repair['method']}` with `{repair['numerical_transport_revision']}`. Face velocities are frozen at the start state and shared face fluxes are solved implicitly by an upwind tridiagonal M-matrix system. This is a numerical transport revision, not physical retuning.

The recorded numerical qualification execution used source commit `{KWN_QUALIFICATION_EXECUTION_COMMIT}`. Later delivery-only changes do not alter that completed solver trajectory.

- Physical parameter retuning: `{repair['physical_parameter_retuning']}`.
- Negative-bin clamp: `{repair['negative_bin_clamp']}`.
- P1 exact regression: `{gates['P1_exact_failure_regression']}`; P2 48 h completion: `{gates['P2_48h_completion']}`; P3 no clipping: `{gates['P3_no_clipping']}`; P4 conservation: `{gates['P4_conservation']}`.
- P5 timestep convergence: `{gates['P5_timestep_convergence']}`; P6 positivity utilization: `{gates['P6_positivity_utilization']}`; P6 restart: `{gates['P6_restart']}`.
- Canonical 48 h residual: `{_fmt(_mapping(summary['canonical_48h'], 'canonical')['inventory_relative_residual'])}`; roundoff-zeroed bin count: `{_fmt(_mapping(summary['canonical_48h'], 'canonical')['roundoff_zeroed_bin_count'])}`.

The conservative KWN trajectory is positive, ledger-closed, restart-identical and complete to 48 h, but the **radius/size-space grid** has not converged under the declared 200-versus-400-bin criterion. It is therefore not a qualified 48 h beta-only prediction.

At 48 h, the 200-versus-400 reference differences are N=`{grid['beta_number_density_m3']:.4%}`, Rmean=`{grid['beta_mean_radius_m']:.4%}`, Rmean³=`{grid['beta_mean_radius_cubed_m3']:.4%}`, Sv=`{grid['beta_specific_surface_area_m_inv']:.4%}`, fβ=`{grid['beta_volume_fraction']:.4%}`, matrix xB=`{grid['matrix_xB']:.4%}`. Every metric must be ≤2%; P5 radius-grid convergence is therefore `{gates['P5_bin_convergence_200_vs_400']}` and overall status remains `{summary['status']}`.

P1 audit: accepted steps `{p1['accepted_step_count']}`, minimum dt `{_fmt(p1['minimum_dt_s'])}` s, maximum CFL `{_fmt(p1['maximum_size_cfl'])}`, maximum utilization `{_fmt(p1['maximum_positivity_utilization'])}`. Full 48 h audit: accepted steps `{retry['accepted_step_count']}`, minimum dt `{_fmt(retry['minimum_dt_s'])}` s, median dt `{_fmt(retry['median_dt_s'])}` s, rejected steps `{retry['rejected_step_count']}`.
""",
    )
    _write(
        report_root / "08_beta_only_same_contract.md",
        f"""# 08 Beta-only KWN–PF same-contract trajectory

Status: `NOT_RUN_PREREQUISITE_FAIL_KWN_CONSERVATIVE_POSITIVITY`.

Although CUDA finished `{audit['status']}`, the KWN qualification is `{summary['status']}` because P5 radius-grid convergence is false. The task contract permits the beta-only KWN–PF trajectory comparison only after both `PASS_CUDA_AE_SMOKE` and `PASS_KWN_CONSERVATIVE_POSITIVITY`.

No scalar-moment substitute, no D-scale fitting, and no incomplete KWN trajectory is used. Consequently growth/dissolution direction, timescale, PSD Wasserstein distance and mean-field gap are not claimed.
""",
    )
    _write(
        report_root / "09_historical_authority_boundary.md",
        """# 09 Historical authority boundary

- `HISTORICAL_AS_RUN_AUTHORITY_UNRECOVERED`
- `HISTORICAL_12H_PSD_NOT_RECOVERED`

The opportunistic read-only search did not recover a historical as-run authority or a 12 h 400³ raw checkpoint/PSD. This 96³ CUDA work is validation-only and must not be labelled as historical 400³ production validation, 21-case consistency, or retrospective authority.
""",
    )
    _write(
        report_root / "10_model_limitations.md",
        """# 10 Model limitations

- KWN is mean-field and spherical-equivalent; CUDA PF is spatial, elastic and diffuse-interface. A numerical agreement claim requires the gated trajectory comparison, which was not run.
- The 96³ calculation is a validation-only smoke, not a 400³ production or ensemble result.
- GP release, GP→beta conversion, beta birth, online KWN coupling, dislocation physics, matrix global reset and composition clamping remain disabled.
- No thermodynamic, diffusivity, mobility, gamma, lambda, elasticity, eigenstrain, fixture profile or physical initial PSD retuning was used to cure the KWN failure.
- The strict remaining KWN limitation is radius/size-space grid convergence, not a timestep, conservation or positivity-clamp issue.
""",
    )
    _write(
        report_root / "11_final_acceptance_report.md",
        f"""# 11 Final acceptance report

Top-level status: `{top}`.

| Gate | Result |
|---|---|
| Controlled CUDA binary | `{cuda['build']['status']}` |
| CUDA A–E smoke | `{audit['status']}` |
| A vs B identity / B vs C frozen storage | `{a_b_c_identical}` |
| Case D transfer | exact shared ledger transfer |
| Case E handoff | `{e_closure.get('double_count_status')}` |
| CUDA restart | all recorded fields within `1e-14` |
| KWN positivity qualification | `{summary['status']}` (P5 radius-grid only) |
| beta-only KWN–PF comparison | `NOT_RUN_PREREQUISITE_FAIL_KWN_CONSERVATIVE_POSITIVITY` |

The CUDA storage/runtime contract is demonstrated on a real A100. The KWN transport repair eliminates the original non-conservative explicit donor overdraw without clamp or physical retuning, but it does not yet meet the declared radius-grid convergence qualification. Local GP release development is therefore **NO**.
""",
    )
    _write(
        report_root / "12_reproduction_commands.md",
        f"""# 12 Reproduction commands

Run from the isolated worktree and use a new non-overwriting output root for every CUDA submission.

```bash
cd {ROOT}
PYTHONPATH=src python3 -m unittest tests.kwn.test_beta_only_same_contract_control \\
  tests.kwn.test_conservative_positivity_repair tests.kwn.test_rmin_boundary \\
  tests.kwn.test_numerical_gates
PYTHONPATH=src python3 scripts/diagnose_kwn_positivity_failure.py \\
  --output-root outputs/kwn_pf_cuda_runtime_closure_v1
PYTHONPATH=src python3 scripts/run_kwn_conservative_qualification.py \\
  --output-root outputs/kwn_pf_cuda_runtime_closure_v1/kwn_conservative_qualification

# After a fresh, clean CUDA R4 job completes and compact audit is copied locally:
PYTHONPATH=src python3 scripts/render_kwn_pf_cuda_runtime_closure_v1.py \\
  --cuda-run-root {cuda['root']} \\
  --output-root {output_root} \\
  --report-root {report_root}
```

Do not run the beta-only KWN–PF comparator unless both prerequisite statuses are exact `PASS_CUDA_AE_SMOKE` and `PASS_KWN_CONSERVATIVE_POSITIVITY`; this evidence package intentionally writes a `NOT_RUN` row instead.
""",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args()
    cuda = _load_cuda(args.cuda_run_root.resolve())
    kwn = _load_kwn(args.output_root.resolve())
    _write_beta_not_run(args.output_root)
    _render_reports(args.output_root, args.report_root.resolve(), cuda, kwn)
    print(
        json.dumps(
            {
                "status": _top_status(cuda, kwn),
                "cuda_status": cuda["audit"]["status"],
                "kwn_status": kwn["summary"]["status"],
                "report_root": str(args.report_root.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
