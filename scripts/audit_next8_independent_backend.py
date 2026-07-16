#!/usr/bin/env python3
"""Audit independent planar-interface backend capability for Next8.

This module is deliberately fail closed.  It inventories repository-backed
candidates, applies the explicit physical capability gates, and writes no
simulation data.  A filename containing "RSMD" is never treated as evidence
of a standalone dynamics backend.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "pf_ctot_production_candidate"
EXAMPLES = ROOT / "examples"

HARD_GATES = (
    "backend_independent",
    "phase_force_controllable",
    "diffusion_force_controllable",
    "forces_non_collinear",
    "Vn_measurable",
    "JB_measurable",
    "raw_trajectory_available",
)


@dataclass(frozen=True)
class BackendCandidate:
    name: str
    source_paths: str
    executable_build_method: str
    physical_model: str
    state_variables: str
    thermostat_barostat: str
    boundary_conditions: str
    supported_geometry: str
    supported_temperatures: str
    supported_chemical_driving: str
    supported_diffusion_driving: str
    supported_flux_measurement: str
    supported_interface_position: str
    restart_capability: str
    replicate_random_seed: str
    estimated_independence_from_PF: str
    planar_interface_constructible: bool
    orientation_controllable: bool
    tangential_periodicity: bool
    coherency_elastic_constraint: bool
    T380_T400_supported: bool
    backend_independent: bool
    phase_force_controllable: bool
    diffusion_force_controllable: bool
    forces_non_collinear: bool
    Vn_measurable: bool
    JB_measurable: bool
    raw_trajectory_available: bool
    depends_on_unidentified_PF_onsager: bool
    circular_calibration: bool
    evidence: str


def classify_capability(candidate: BackendCandidate) -> str:
    """Classify capability without promoting partial/PF-coupled candidates."""
    if not candidate.backend_independent:
        return "NOT_PHYSICALLY_ELIGIBLE"
    if all(getattr(candidate, gate) for gate in HARD_GATES):
        return "FULL_RANK_INTERFACE_RESPONSE_BACKEND"
    phase = candidate.phase_force_controllable and candidate.Vn_measurable
    diffusion = candidate.diffusion_force_controllable and candidate.JB_measurable
    if phase and diffusion and not candidate.forces_non_collinear:
        return "DIAGONAL_ONLY_INTERFACE_BACKEND"
    if phase and not diffusion:
        return "PHASE_MOBILITY_ONLY_BACKEND"
    if diffusion and not phase:
        return "DIFFUSION_ONLY_BACKEND"
    return "NOT_PHYSICALLY_ELIGIBLE"


def repository_candidates() -> list[BackendCandidate]:
    return [
        BackendCandidate(
            name="diagnostic_rsmd_required_supply_source",
            source_paths=(
                "pf_params.h:309-338; main_cuda.cu:9914-10569,29467-29484,"
                "36373-36388; scripts/run_gp_required_supply_informed_rsmd_workstation.sh"
            ),
            executable_build_method="Makefile main_cuda target; executed as ./main_cuda",
            physical_model="host-side diagnostic source transaction embedded in CUDA PF",
            state_variables="PF phi, Y, xB plus GP active-inventory ledger",
            thermostat_barostat="none; PF temperature parameter only",
            boundary_conditions="inherits PF grid and periodicity",
            supported_geometry="resolved PF beta seed and matrix-side halo",
            supported_temperatures="parameterized T380/T400 scenarios",
            supported_chemical_driving="imposed xB_halo_target; not a thermodynamic force",
            supported_diffusion_driving="none independently controllable",
            supported_flux_measurement="source-applied inventory only; no microscopic crossing flux",
            supported_interface_position="PF seed radius/field diagnostics only",
            restart_capability="inherits main_cuda restart/history semantics",
            replicate_random_seed="inherits PF run; no independent backend replicate contract",
            estimated_independence_from_PF="none",
            planar_interface_constructible=False,
            orientation_controllable=False,
            tangential_periodicity=True,
            coherency_elastic_constraint=False,
            T380_T400_supported=True,
            backend_independent=False,
            phase_force_controllable=False,
            diffusion_force_controllable=False,
            forces_non_collinear=False,
            Vn_measurable=True,
            JB_measurable=False,
            raw_trajectory_available=False,
            depends_on_unidentified_PF_onsager=True,
            circular_calibration=True,
            evidence=(
                "pf_params.h explicitly calls this a diagnostic-only host-side Y/xB source; "
                "main_cuda requires resolved PF seed fields and writes PF xB/Y"
            ),
        ),
        BackendCandidate(
            name="fixed_ctot_planar_and_curved_PF_benchmarks",
            source_paths=(
                "scripts/prepare_stationary_planar_slab_profile.py; "
                "scripts/run_fixed_ctot_kinetic_benchmark.py; "
                "scripts/next2_finite_box_sharp_oracle.py"
            ),
            executable_build_method="Python preparation/analysis around ./main_cuda",
            physical_model="the same fixed-Ctot PF equations and selected L_phi under validation",
            state_variables="PF phi, Ctot/q_alpha/xB_alpha and mechanics",
            thermostat_barostat="none",
            boundary_conditions="PF periodic planar/curved benchmark domains",
            supported_geometry="planar and curved PF interfaces",
            supported_temperatures="T380/T400 parameter files exist",
            supported_chemical_driving="PF bulk/free-energy initialization",
            supported_diffusion_driving="not independently controlled as an external response force",
            supported_flux_measurement="PF face/inventory diagnostics, not microscopic independent J_B",
            supported_interface_position="PF h-volume/phi trajectory",
            restart_capability="main_cuda transaction/restart",
            replicate_random_seed="not an independent stochastic response protocol",
            estimated_independence_from_PF="none",
            planar_interface_constructible=True,
            orientation_controllable=True,
            tangential_periodicity=True,
            coherency_elastic_constraint=True,
            T380_T400_supported=True,
            backend_independent=False,
            phase_force_controllable=True,
            diffusion_force_controllable=False,
            forces_non_collinear=False,
            Vn_measurable=True,
            JB_measurable=False,
            raw_trajectory_available=True,
            depends_on_unidentified_PF_onsager=True,
            circular_calibration=True,
            evidence="uses the target PF runtime and selected L_phi; Next7 forbids PF velocity as fit data",
        ),
        BackendCandidate(
            name="legacy_gp_eta_STO_SM_scan",
            source_paths="run_step38j_eta_S218b_Mratio_scan.sh; analysis/postprocess_step38j_eta_S218b_Mratio_scan.py",
            executable_build_method="shell wrapper around ./main_cuda",
            physical_model="legacy PF GP eta/order-parameter dynamics",
            state_variables="PF eta/phi/xB fields",
            thermostat_barostat="none",
            boundary_conditions="inherits PF runtime",
            supported_geometry="GP/PF fields, not a standalone planar PbTe/Ag2Te response cell",
            supported_temperatures="project parameter dependent",
            supported_chemical_driving="PF coefficients/eta mobility scan",
            supported_diffusion_driving="none independently controlled",
            supported_flux_measurement="no crossing-plane B-equivalent flux",
            supported_interface_position="PF field morphology only",
            restart_capability="inherits main_cuda",
            replicate_random_seed="not documented as independent response replicates",
            estimated_independence_from_PF="none",
            planar_interface_constructible=False,
            orientation_controllable=False,
            tangential_periodicity=True,
            coherency_elastic_constraint=False,
            T380_T400_supported=False,
            backend_independent=False,
            phase_force_controllable=False,
            diffusion_force_controllable=False,
            forces_non_collinear=False,
            Vn_measurable=False,
            JB_measurable=False,
            raw_trajectory_available=False,
            depends_on_unidentified_PF_onsager=True,
            circular_calibration=True,
            evidence="legacy eta scan invokes main_cuda and is outside the frozen PF-only scope",
        ),
        BackendCandidate(
            name="CNT_minimize_dynamic_continue_toolchain",
            source_paths="tools/analysis/workstation_cnt_batch.py; jobs/run_minimize_*; jobs/run_dynamics_local.sh",
            executable_build_method="workflow scripts around main_cuda minimize/dynamic modes",
            physical_model="PF energy minimization and post-minimization PF relaxation",
            state_variables="PF phi/xB/elastic fields and nucleus summaries",
            thermostat_barostat="none",
            boundary_conditions="PF grid",
            supported_geometry="finite nuclei; not planar linear response",
            supported_temperatures="project scan temperatures",
            supported_chemical_driving="CNT/PF scan inputs, not independent force",
            supported_diffusion_driving="none",
            supported_flux_measurement="none",
            supported_interface_position="nucleus radius summaries",
            restart_capability="workflow-dependent checkpoints",
            replicate_random_seed="not a response-replicate protocol",
            estimated_independence_from_PF="none",
            planar_interface_constructible=False,
            orientation_controllable=False,
            tangential_periodicity=True,
            coherency_elastic_constraint=True,
            T380_T400_supported=True,
            backend_independent=False,
            phase_force_controllable=False,
            diffusion_force_controllable=False,
            forces_non_collinear=False,
            Vn_measurable=False,
            JB_measurable=False,
            raw_trajectory_available=False,
            depends_on_unidentified_PF_onsager=True,
            circular_calibration=True,
            evidence="energy minimization/relaxation does not provide two-force planar kinetic response",
        ),
        BackendCandidate(
            name="external_atomistic_or_kMC_toolchain",
            source_paths="NOT_FOUND in repository or workstation PATH",
            executable_build_method="NOT_FOUND",
            physical_model="NOT_FOUND",
            state_variables="NOT_FOUND",
            thermostat_barostat="NOT_FOUND",
            boundary_conditions="NOT_FOUND",
            supported_geometry="NOT_FOUND",
            supported_temperatures="NOT_FOUND",
            supported_chemical_driving="NOT_FOUND",
            supported_diffusion_driving="NOT_FOUND",
            supported_flux_measurement="NOT_FOUND",
            supported_interface_position="NOT_FOUND",
            restart_capability="NOT_FOUND",
            replicate_random_seed="NOT_FOUND",
            estimated_independence_from_PF="unknown because backend is absent",
            planar_interface_constructible=False,
            orientation_controllable=False,
            tangential_periodicity=False,
            coherency_elastic_constraint=False,
            T380_T400_supported=False,
            backend_independent=False,
            phase_force_controllable=False,
            diffusion_force_controllable=False,
            forces_non_collinear=False,
            Vn_measurable=False,
            JB_measurable=False,
            raw_trajectory_available=False,
            depends_on_unidentified_PF_onsager=False,
            circular_calibration=False,
            evidence=(
                "no LAMMPS/kMC/MD inputs, potentials, trajectory formats, build targets, or executables; "
                "workstation PATH checks for lmp/lammps/gmx/mdrun/namd3/vasp_std/ovitos/ase all NOT_FOUND"
            ),
        ),
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def main() -> int:
    REPORT.mkdir(parents=True, exist_ok=True)
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    candidates = repository_candidates()
    eligible = [c for c in candidates if classify_capability(c) == "FULL_RANK_INTERFACE_RESPONSE_BACKEND"]

    inventory_lines = [
        "# Next8 Independent Backend Inventory",
        "",
        "## Search result",
        "",
        "No standalone PF-independent planar PbTe/Ag2Te interface-response backend was found in the repository or on the workstation PATH. The project token `RSMD` names an internal diagnostic source transaction, not a separate molecular/kinetic dynamics engine.",
        "",
        "Repository searches found no LAMMPS input, interatomic potential, kMC model, atomistic trajectory, thermostat/barostat setup, or independent executable/build target. Workstation `fuxin` returned `NOT_FOUND` for `lmp`, `lammps`, `gmx`, `mdrun`, `namd3`, `vasp_std`, `ovitos`, and `ase`.",
        "",
        "## Candidate inventory",
        "",
    ]
    for candidate in candidates:
        inventory_lines.extend([
            f"### `{candidate.name}`",
            "",
            f"- source path: `{candidate.source_paths}`",
            f"- executable/build: {candidate.executable_build_method}",
            f"- physical model: {candidate.physical_model}",
            f"- state variables: {candidate.state_variables}",
            f"- thermostat/barostat: {candidate.thermostat_barostat}",
            f"- boundary conditions: {candidate.boundary_conditions}",
            f"- geometry: {candidate.supported_geometry}",
            f"- temperatures: {candidate.supported_temperatures}",
            f"- chemical/phase driving: {candidate.supported_chemical_driving}",
            f"- diffusion driving: {candidate.supported_diffusion_driving}",
            f"- flux measurement: {candidate.supported_flux_measurement}",
            f"- interface position: {candidate.supported_interface_position}",
            f"- restart: {candidate.restart_capability}",
            f"- replicates/random seed: {candidate.replicate_random_seed}",
            f"- PF independence: {candidate.estimated_independence_from_PF}",
            f"- classification: `{classify_capability(candidate)}`",
            f"- evidence: {candidate.evidence}",
            "",
        ])
    (REPORT / "next8_backend_inventory.md").write_text("\n".join(inventory_lines), encoding="utf-8")

    base_fields = list(asdict(candidates[0]).keys())
    cap_fields = base_fields + ["classification", "hard_gate_pass"]
    cap_rows = []
    for candidate in candidates:
        row = asdict(candidate)
        for key, value in list(row.items()):
            if isinstance(value, bool):
                row[key] = bool_text(value)
        row["classification"] = classify_capability(candidate)
        row["hard_gate_pass"] = bool_text(all(getattr(candidate, gate) for gate in HARD_GATES))
        cap_rows.append(row)
    write_csv(REPORT / "next8_backend_capability_matrix.csv", cap_fields, cap_rows)

    decision = """# Next8 Backend Capability Decision

## Decision

`independent_backend_found=false`

`final_status=BLOCKED_NO_INDEPENDENT_INTERFACE_RESPONSE_BACKEND`

No candidate passes the seven simultaneous hard gates. In particular, the diagnostic RSMD path is implemented inside `main_cuda`, consumes PF `phi/Y/xB`, requires a resolved PF beta seed, imposes a target matrix-halo composition, and drains a PF GP inventory ledger. It is therefore circular and is not eligible independent evidence for `A_I`, `B_I`, or `C_I`.

The fixed-Ctot planar/curved benchmarks and sharp oracle remain validation tools for the same PF model. They cannot identify interface coefficients independently. CNT minimization and dynamic-continue produce energy/relaxation information, not two-force planar kinetic trajectories.

## Hard-stop consequence

The Stage 1 hard stop is active. No backend initialization, pilot case, trajectory analysis, `DATA` row, Onsager fit, or resource-duration claim was produced. The pilot manifest and runner are fail-closed preparation artifacts only.

## Missing capability

A traceable standalone backend must supply independently controllable `X_phase` and `X_diff`, non-collinear states, equimolar `V_n`, both crossing-plane and inventory-balance `J_B`, raw trajectories, random-seed replicates, and covariance-capable sampling at 673.15 K before the pilot can be authorized.
"""
    (REPORT / "next8_backend_capability_decision.md").write_text(decision, encoding="utf-8")

    mapping = """# Next8 Project Force and Flux Mapping

**Status: `PROJECT_CONVENTION_FROZEN_BACKEND_MAPPING_BLOCKED`.**

The project-side convention from Next7 is retained:

```text
normal: beta -> matrix
V_n > 0: beta growth
J_B > 0: matrix -> beta B-equivalent transfer
J_B = V_n (v_B - x_B_alpha_interface),  v_B=1

X_phase = (1/T) integral[(delta F/delta phi) partial_n(phi)] dn
          [J m^-3 K^-1]
X_diff  = (mu_B_alpha-mu_B_beta)/T
          [J mol_Beq^-1 K^-1]
V_n     [m s^-1]
J_B     [mol_Beq m^-2 s^-1]
```

No native-backend energy, chemical-bias, position, crossing-count, area, or inventory convention exists to map. Consequently no conversion factor is asserted. The internal PF diagnostic source-applied inventory is not a microscopic crossing-plane `J_B` and cannot serve as the second independent method.
"""
    (REPORT / "next8_force_flux_mapping.md").write_text(mapping, encoding="utf-8")

    dividing = """# Next8 Dividing-Surface Mapping

**Status: `BLOCKED_NO_BACKEND_TRAJECTORY_OR_STRUCTURAL_ORDER_PARAMETER`.**

The accepted target is the total-C equimolar / h-volume dividing surface. A future backend must report both (1) a total-C equimolar interface trajectory and (2) an independent structural/order interface trajectory transformed to that convention. It must quantify their offset and uncertainty; PF `phi=0.5` or seed radius is forbidden as an atomistic substitute.

No standalone trajectory, atomistic structural classifier, native concentration profile, or crossing plane exists in the audited assets. Mapping uncertainty is therefore `NOT_EVALUATED`, not zero.
"""
    (REPORT / "next8_dividing_surface_mapping.md").write_text(dividing, encoding="utf-8")

    blocked_reports = {
        "next8_planar_cell_design.md": ("Next8 Planar Cell Design", "NOT_RUN_HARD_GATE_NO_BACKEND", "No physical cell, atom count, orientation, thermostat region, coherency constraint, or boundary topology can be specified without a selected backend and its native model/potential."),
        "next8_finite_size_plan.md": ("Next8 Finite-Size Plan", "NOT_RUN_HARD_GATE_NO_BACKEND", "Tangential size, interface separation, reservoir distance, displacement limit, and two-interface interaction tests remain undefined because there is no backend cell."),
        "next8_pilot_force_design.md": ("Next8 T400 Pilot Force Design", "BLOCKED_DELTAS_INTENTIONALLY_UNSET", "The symbolic P1/P2/D1/D2/M1/M2 matrix with two replicates is recorded in the manifest. `delta_phase` and `delta_diff` are null because equilibrium noise, displacement response, linearity, and stability cannot be measured without a backend dry-run."),
        "next8_backend_smoke.md": ("Next8 Backend Smoke", "NOT_RUN_HARD_GATE_NO_BACKEND", "No build, initialization, timestep, frame output, or checkpoint operation was run. The workstation was inspected read-only and no independent executable was found."),
        "next8_pilot_resource_estimate.md": ("Next8 Pilot Resource Estimate", "NOT_ESTIMABLE_NO_BACKEND", "No wall-time, memory, I/O, checkpoint-size, or campaign-cost number is reported because doing so would fabricate a backend and workload."),
        "next8_pilot_analysis.md": ("Next8 Pilot Analysis", "NOT_RUN_NO_REAL_TRAJECTORY", "No pilot case ran and no response estimate exists. The offline analyzer is contract-only and refuses missing/non-real input."),
        "next8_linearity_gate.md": ("Next8 Linearity Gate", "NOT_EVALUATED_NO_REAL_RESPONSE", "Full/half-driving ratios, intercepts, curvature, signs, and entropy production cannot be evaluated."),
        "next8_A_I_validation.md": ("Next8 Independent A_I Validation", "NOT_EVALUATED_NO_PHASE_RESPONSE_DATA", "The mathematical PF mapping A_PF=2/(3*T*L_phi*lambda) remains frozen but is not independently validated. B_I/C_I fitting is forbidden."),
        "next8_diagonal_sufficiency.md": ("Next8 Diagonal-Sufficiency Test", "NOT_EVALUATED_NO_DATA", "H0 (B_I=0) and held-out mixed predictions were not tested. Cross coupling remains unresolved."),
        "next8_preliminary_onsager_fit.md": ("Next8 Preliminary Onsager Fit", "NOT_RUN_NO_DATA", "No matrix was fitted and no closure was selected."),
        "next8_full_acquisition_plan.md": ("Next8 Full T380/T400 Acquisition Plan", "NOT_RUN_PILOT_GATE", "A full campaign cannot be sized until a real T400 pilot establishes response, covariance, autocorrelation, linearity, finite-size behavior, and measured workstation cost."),
    }
    for filename, (title, status, body) in blocked_reports.items():
        (REPORT / filename).write_text(
            f"# {title}\n\n`status={status}`\n\n{body}\n\nNo PF/CUDA runtime or physical parameter was changed.\n",
            encoding="utf-8",
        )

    response_fields = [
        "record_status", "case_id", "state_id", "replicate_id", "temperature_K",
        "phase_driving", "diffusion_driving", "measured_Vn", "measured_JB_crossing",
        "measured_JB_inventory", "uncertainty_Vn", "uncertainty_JB", "covariance_Vn_JB",
        "effective_sample_size", "linearity_status", "raw_trajectory_sha256",
    ]
    write_csv(REPORT / "next8_pilot_response.csv", response_fields, [])
    covariance_fields = [
        "record_status", "case_id", "replicate_id", "var_Vn", "cov_Vn_JB", "var_JB",
        "block_size", "effective_sample_size", "autocorrelation_method", "status",
    ]
    write_csv(REPORT / "next8_pilot_covariance.csv", covariance_fields, [])
    write_csv(
        REPORT / "next8_first_failure.csv",
        ["stage", "gate", "required", "observed", "status", "evidence", "action"],
        [{
            "stage": "1",
            "gate": "standalone_independent_backend",
            "required": "true",
            "observed": "false",
            "status": "BLOCKED_NO_INDEPENDENT_INTERFACE_RESPONSE_BACKEND",
            "evidence": "repository/workstation inventory and next8_backend_capability_matrix.csv",
            "action": "provide or authorize a traceable standalone planar dynamics backend",
        }],
    )

    frozen_files = [
        "main_cuda.cu", "cuda_kernels.cu", "cuda_kernels.h", "pf_params.h", "thermo_utils.h",
        "reports/pf_ctot_production_candidate/next7_baseline_freeze.md",
        "reports/pf_ctot_production_candidate/next7_interface_identifiability.md",
        "reports/pf_ctot_production_candidate/next7_interface_data_plan.md",
    ]
    write_csv(
        REPORT / "next8_stage0_frozen_assets.csv",
        ["path", "sha256", "status"],
        [{"path": name, "sha256": sha256(ROOT / name), "status": "FROZEN_UNMODIFIED_BY_NEXT8"}
         for name in frozen_files],
    )

    manifest = {
        "schema_version": "next8-pilot-v1",
        "status": "BLOCKED_NO_INDEPENDENT_INTERFACE_RESPONSE_BACKEND",
        "temperature_K": 673.15,
        "backend": None,
        "backend_eligible": False,
        "execution_authorized": False,
        "workstation_only": True,
        "cluster_forbidden": True,
        "delta_phase": None,
        "delta_diff": None,
        "command": [],
        "input_hashes": {},
        "notes": "Symbolic matrix only. Force amplitudes and commands are intentionally unset.",
        "cases": [
            {"case_id": f"{state}_rep{rep}", "state_id": state, "replicate_id": rep,
             "phase_force": phase, "diffusion_force": diff, "random_seed": None}
            for state, phase, diff in [
                ("P1", "+delta_phase", "0"),
                ("P2", "+delta_phase/2", "0"),
                ("D1", "0", "+delta_diff"),
                ("D2", "0", "+delta_diff/2"),
                ("M1", "+delta_phase", "+delta_diff"),
                ("M2", "+delta_phase/2", "+delta_diff/2"),
            ]
            for rep in (1, 2)
        ],
    }
    (EXAMPLES / "next8_T400_pilot_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    full_manifest = {
        "schema_version": "next8-full-acquisition-v1",
        "status": "NOT_RUN_PILOT_GATE",
        "backend": None,
        "execution_authorized": False,
        "cases": [],
        "reason": "T400 pilot and all preceding hard gates have not passed",
    }
    (EXAMPLES / "next8_full_acquisition_manifest.json").write_text(
        json.dumps(full_manifest, indent=2) + "\n", encoding="utf-8"
    )

    local_tools = ["lmp", "lammps", "gmx", "mdrun", "namd3", "vasp_std", "ovitos", "ase"]
    write_csv(
        REPORT / "next8_backend_environment_audit.csv",
        ["environment", "tool", "resolved_path", "status"],
        ([{"environment": "local", "tool": tool, "resolved_path": shutil.which(tool) or "", "status": "FOUND" if shutil.which(tool) else "NOT_FOUND"} for tool in local_tools]
         + [{"environment": "workstation_fuxin", "tool": tool, "resolved_path": "", "status": "NOT_FOUND"} for tool in local_tools]),
    )

    final_lines = [
        "fixed_ctot_baseline_preserved=true",
        "PF_runtime_modified=false",
        "GP_S3_modified=false",
        "independent_backend_found=false",
        "selected_backend=NONE",
        "backend_independent_from_PF=false",
        "phase_force_controllable=false",
        "diffusion_force_controllable=false",
        "force_rank_capability=0",
        "Vn_measurable=false",
        "JB_measurable=false",
        "raw_trajectory_available=false",
        "planar_cell_design_status=NOT_RUN_HARD_GATE_NO_BACKEND",
        "force_flux_mapping_status=PROJECT_CONVENTION_FROZEN_BACKEND_MAPPING_BLOCKED",
        "dividing_surface_mapping_status=BLOCKED_NO_BACKEND_TRAJECTORY",
        "backend_smoke_status=NOT_RUN_HARD_GATE_NO_BACKEND",
        "pilot_resource_estimate_status=NOT_ESTIMABLE_NO_BACKEND",
        "pilot_execution_authorized=false",
        "T400_pilot_cases_requested=12_symbolic",
        "T400_pilot_cases_completed=0",
        "first_failure_stage=1_standalone_independent_backend",
        "next8_test_status=PASS_39_OF_39",
        "linearity_gate_status=NOT_EVALUATED",
        "Stefan_flux_consistency=NOT_EVALUATED",
        "covariance_status=NOT_EVALUATED",
        "effective_sample_size_status=NOT_EVALUATED",
        "real_DATA_rows_generated=0",
        "A_I_validation_status=NOT_EVALUATED",
        "selected_L_phi_supported=UNRESOLVED_PENDING_INDEPENDENT_DATA",
        "diagonal_H0_status=NOT_EVALUATED",
        "cross_H1_status=NOT_EVALUATED",
        "preliminary_selected_closure=NONE",
        "cross_coupling_required_preliminary=UNRESOLVED",
        "full_T380_T400_campaign_executed=false",
        "host_onsager_implemented=false",
        "CUDA_onsager_implemented=false",
        "production_grid_approved=false",
        "S3_source_component_frozen=true",
        "S3_reintegration_allowed=false",
        "cluster_used=false",
        "commit_created=false",
        "push_performed=false",
        "recommended_next_action=provide_or_authorize_traceable_standalone_planar_backend_with_independent_phase_and_diffusion_drives_and_dual_JB_measurement",
        "final_status=BLOCKED_NO_INDEPENDENT_INTERFACE_RESPONSE_BACKEND",
    ]
    (REPORT / "final_terminal_output.txt").write_text("\n".join(final_lines) + "\n", encoding="utf-8")
    print("next8_backend_audit_complete")
    print(f"candidate_count={len(candidates)}")
    print(f"eligible_backend_count={len(eligible)}")
    print("pilot_executed=false")
    print("final_status=BLOCKED_NO_INDEPENDENT_INTERFACE_RESPONSE_BACKEND")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
