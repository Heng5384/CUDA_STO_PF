#!/usr/bin/env python3
"""Run the validation-only beta KWN direction control on the frozen 96-cube.

This is deliberately a *host-side code control*, not a replacement PF run.
It projects the already-resolved six-particle 96-cube fixture onto a compact
spherical KWN beta PSD, while preserving the fixture's total matrix and
resolved-beta B inventory.  The only PF comparator is the exact
contract-defined curvature threshold at the initial size classes.  CUDA PF
trajectory values are left blank unless a separately qualified PF run exists;
this script never fabricates or extrapolates one.

The existing ``beta_only_pf_consistency.yaml`` supplies numerical controls
(time-step/CFL/radius grid) only.  Its historical thermodynamic fields are not
used: all beta thermodynamic and kinetic quantities are loaded from the
hash-bound validation contract at runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coupling.fixture_conditioned_handoff_v2 import (  # noqa: E402
    FixtureConditionedHandoffError,
    FixtureState,
    ValidationContract,
    load_host_96cube_fixture,
    load_validation_contract,
)
from kwn_mvp.config import ConfigurationError  # noqa: E402
from kwn_mvp.ledger import InventoryError  # noqa: E402
from kwn_mvp.radius_grid import RadiusGrid  # noqa: E402
from kwn_mvp.solver import (  # noqa: E402
    KWNSolver,
    RadiusGridOverflowError,
    SolverConfig,
    SolverStateError,
)
from kwn_mvp.thermo_adapter import ThermodynamicDomainError  # noqa: E402


OUTPUT_ROOT = ROOT / "outputs" / "kwn_pf_state_closure_v1"
DEFAULT_CONTRACT = ROOT / "contracts" / "pf_kwn_validation_contract_v1.json"
DEFAULT_FIXTURE_SPEC = (
    ROOT
    / "data"
    / "qualification"
    / "pf_mass_conserving_library_handoff_v1"
    / "six_particle_96cube_spec.json"
)
DEFAULT_PROFILE_ROOT = (
    ROOT.parent
    / "CUDA_STO_PF"
    / "data"
    / "qualification"
    / "pf_elastic_target_profile_quarter_nm_v2"
    / "profiles"
)
NUMERICAL_TEMPLATE = ROOT / "configs" / "kwn" / "beta_only_pf_consistency.yaml"

PF_CUDA_NOT_RUN = "NOT_RUN_NO_CUDA_OR_PF_BINARY"
HISTORICAL_AUTHORITY_STATUS = "HISTORICAL_AS_RUN_AUTHORITY_UNRECOVERED"
HISTORICAL_PSD_STATUS = "HISTORICAL_12H_PSD_NOT_RECOVERED"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON mapping {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _inside_output_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(OUTPUT_ROOT.resolve())
        return True
    except ValueError:
        return False


def _sphere_volume(radius_m: float) -> float:
    return 4.0 * math.pi * float(radius_m) ** 3 / 3.0


def _contract_valid_grid_minimum_m(
    *, contract: ValidationContract, requested_minimum_m: float, upper_probe_m: float
) -> float:
    """Return the least radius whose declared curvature equilibrium is valid.

    ``ValidationContractEquilibriumAdapter`` deliberately evaluates every grid
    centre and rejects a Gibbs--Thomson value outside ``(0, 1)``.  The legacy
    numerical template begins at 0.25 nm, below the exact-contract domain.
    This helper changes no material parameter: it computes the smallest valid
    finite-volume edge from the contract itself and records that numerical
    domain gate in the output provenance.
    """

    if requested_minimum_m <= 0.0 or upper_probe_m <= requested_minimum_m:
        raise RuntimeError("invalid numerical radius-grid bounds")

    def valid(radius_m: float) -> bool:
        try:
            value = contract.canonical.curvature_equilibrium_xb(contract.temperature_K, radius_m)
        except ValueError:
            return False
        return math.isfinite(value) and 0.0 < value < 1.0

    if valid(requested_minimum_m):
        return requested_minimum_m
    lower = requested_minimum_m
    upper = upper_probe_m
    while not valid(upper):
        upper *= 2.0
        if upper >= 1.0e-3:
            raise RuntimeError("could not bracket a contract-valid curvature radius")
    for _ in range(160):
        middle = 0.5 * (lower + upper)
        if valid(middle):
            upper = middle
        else:
            lower = middle
    # Protect against a platform-level last-bit roundoff back onto xeq=1.
    return math.nextafter(upper, math.inf)


def _sign_label(value: float) -> str:
    if value > 0.0:
        return "GROWTH"
    if value < 0.0:
        return "DISSOLUTION"
    return "NEUTRAL"


def _project_fixture_resolved_psd(
    fixture: FixtureState,
    contract: ValidationContract,
    grid: RadiusGrid,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Project fixed PF resolved volume onto KWN grid centres without tuning.

    The frozen fixture supplies six registered radii but its diffuse ``h(phi)``
    storage volume is not exactly a sum of six sharp spheres.  A common
    number-scale factor preserves that *actual resolved field inventory* when
    the radii are represented on the KWN finite-volume grid.  This is an
    inventory projection, not an adjusted radius, diffusivity, or interface
    parameter.
    """

    source_radii = np.asarray(fixture.resolved_equivalent_radii_m, dtype=np.float64)
    if source_radii.ndim != 1 or source_radii.size == 0 or np.any(source_radii <= 0.0):
        raise RuntimeError("fixture has no positive resolved-beta equivalent radii")
    grouped: dict[int, int] = defaultdict(int)
    for radius in source_radii:
        grouped[grid.bin_index(float(radius))] += 1
    target_volume_fraction = (
        fixture.source_resolved_inventory_mol
        * contract.vm_beta_m3_mol
        / (contract.v_B * fixture.box_volume_m3)
    )
    if not 0.0 < target_volume_fraction < 1.0:
        raise RuntimeError("fixture resolved-beta inventory does not define a physical volume fraction")
    unscaled_box_volume = sum(
        count * _sphere_volume(float(grid.centres_m[index])) for index, count in grouped.items()
    )
    if unscaled_box_volume <= 0.0:
        raise RuntimeError("resolved-beta PSD projection has zero equivalent-sphere volume")
    target_box_volume = target_volume_fraction * fixture.box_volume_m3
    number_scale = target_box_volume / unscaled_box_volume
    entries: list[dict[str, float]] = []
    for index, source_count in sorted(grouped.items()):
        centre = float(grid.centres_m[index])
        entries.append(
            {
                "radius_m": centre,
                "number_density_m3": number_scale * source_count / fixture.box_volume_m3,
                "source_particle_count": float(source_count),
                "grid_bin_index": float(index),
            }
        )
    projected = sum(
        item["number_density_m3"] * _sphere_volume(item["radius_m"]) for item in entries
    )
    if not math.isclose(projected, target_volume_fraction, rel_tol=1.0e-14, abs_tol=1.0e-18):
        raise RuntimeError("resolved-beta PSD projection did not preserve its target volume fraction")
    return entries, {
        "source_resolved_radius_count": int(source_radii.size),
        "source_radii_m": [float(value) for value in source_radii],
        "source_unique_radius_count": int(len(grouped)),
        "target_resolved_volume_fraction": target_volume_fraction,
        "projected_resolved_volume_fraction": projected,
        "resolved_equivalent_number_scale": number_scale,
        "projection": "COMMON_NUMBER_SCALE_TO_PRESERVE_FIXED_PF_H_VOLUME_ON_KWN_GRID_CENTRES",
    }


def build_same_contract_config(
    *,
    fixture: FixtureState,
    contract: ValidationContract,
    numerical_template_path: Path = NUMERICAL_TEMPLATE,
) -> tuple[SolverConfig, dict[str, Any]]:
    """Build a beta-only solver configuration with every beta field contract-owned."""

    template = _read_json_mapping(numerical_template_path)
    grid_data = template.get("radius_grid")
    simulation = template.get("simulation")
    matrix_template = template.get("matrix")
    if not isinstance(grid_data, Mapping) or not isinstance(simulation, Mapping) or not isinstance(matrix_template, Mapping):
        raise RuntimeError("numerical template lacks radius_grid, simulation, or matrix mappings")
    template_minimum_m = float(grid_data["minimum_m"])
    numerical_minimum_m = _contract_valid_grid_minimum_m(
        contract=contract,
        requested_minimum_m=template_minimum_m,
        upper_probe_m=float(np.min(fixture.resolved_equivalent_radii_m)),
    )
    grid_data_for_control = dict(grid_data)
    grid_data_for_control["minimum_m"] = numerical_minimum_m
    grid = RadiusGrid.logarithmic(
        numerical_minimum_m, float(grid_data["maximum_m"]), int(grid_data["bins"])
    )
    entries, projection = _project_fixture_resolved_psd(fixture, contract, grid)
    beta_fraction = projection["projected_resolved_volume_fraction"]
    matrix_fraction = 1.0 - float(beta_fraction)
    if matrix_fraction <= 0.0:
        raise RuntimeError("resolved-beta projection leaves no KWN matrix volume")
    matrix_xb = (
        fixture.source_matrix_inventory_mol
        * contract.vm_alpha_m3_mol
        / (matrix_fraction * fixture.box_volume_m3)
    )
    if not 0.0 < matrix_xb < 1.0:
        raise RuntimeError("fixture-derived mean matrix composition lies outside (0, 1)")
    total_b_mol_m3 = fixture.source_total_inventory_mol / fixture.box_volume_m3
    planar_solvus = contract.canonical.planar_solvus_xb(contract.temperature_K)
    beta_parameters = {
        "xB": contract.v_B,
        "molar_volume_m3_mol": contract.vm_beta_m3_mol,
        "diffusivity_m2_s": contract.canonical.matrix_diffusivity_m2_s(contract.temperature_K),
        "gamma_j_m2": contract.canonical.gamma_j_m2,
        "xeq_infinity": planar_solvus,
        "shape_factor": 1.0,
        "elastic_penalty_j_m3": contract.canonical.kwn_elastic_penalty_j_m3,
        "nucleation": {"mode": "off"},
    }
    # KWNSolver has a mandatory ``g`` carrier.  It is exactly empty and never
    # evaluated; contract-owned beta values avoid creating a second GP model.
    g_parameters = {
        **beta_parameters,
        "initial": {"kind": "empty"},
    }
    beta_parameters = {
        **beta_parameters,
        "initial": {"kind": "discrete", "entries": entries},
    }
    config_mapping: dict[str, Any] = {
        "simulation": dict(simulation),
        "matrix": {
            "molar_volume_m3_mol": contract.vm_alpha_m3_mol,
            "initial_xB": matrix_xb,
            "total_b_mol_m3": total_b_mol_m3,
            "inventory_tolerance_relative": float(matrix_template["inventory_tolerance_relative"]),
        },
        "radius_grid": grid_data_for_control,
        "thermodynamics": {
            "mode": "pf_contract",
            "contract_path": str(contract.path),
            "contract_hash": contract.contract_hash,
            "planar_reference_xB": planar_solvus,
        },
        "populations": {
            "g": g_parameters,
            "beta": beta_parameters,
        },
    }
    config = SolverConfig.from_mapping(config_mapping)
    construction = {
        "numerical_template_path": str(numerical_template_path),
        "numerical_template_sha256": _sha256_file(numerical_template_path),
        "contract_hash": contract.contract_hash,
        "contract_path": str(contract.path),
        "thermodynamics_mode": "pf_contract",
        "gp_population_state": "EMPTY_COMPATIBILITY_CARRIER_NO_GP_DYNAMICS",
        "gp_nucleation_mode": "off",
        "beta_nucleation_mode": "off",
        "matrix_xB_from_fixed_fixture_inventory": matrix_xb,
        "matrix_inventory_mol": fixture.source_matrix_inventory_mol,
        "resolved_inventory_mol": fixture.source_resolved_inventory_mol,
        "total_inventory_mol": fixture.source_total_inventory_mol,
        "total_inventory_mol_m3": total_b_mol_m3,
        "planar_solvus_xB": planar_solvus,
        "beta_parameters_from_contract": {
            "D_alpha_m2_s": beta_parameters["diffusivity_m2_s"],
            "gamma_J_m2": beta_parameters["gamma_j_m2"],
            "Vm_beta_m3_mol": beta_parameters["molar_volume_m3_mol"],
            "xB_beta": beta_parameters["xB"],
            "xeq_infinity": beta_parameters["xeq_infinity"],
            "elastic_penalty_J_m3": beta_parameters["elastic_penalty_j_m3"],
        },
        "resolved_psd_projection": projection,
        "numerical_domain_gate": {
            "template_minimum_m": template_minimum_m,
            "contract_valid_minimum_m": numerical_minimum_m,
            "minimum_initial_fixture_radius_m": float(np.min(fixture.resolved_equivalent_radii_m)),
            "initial_fixture_radii_changed": False,
            "physical_parameters_changed": False,
            "reason": "unused legacy bins below this radius yield a nonphysical exact-contract curvature equilibrium and are rejected before KWN evaluation",
        },
        "config_mapping": config_mapping,
    }
    return config, construction


def _snapshot_metrics(solver: KWNSolver) -> dict[str, float]:
    beta = solver.population("beta")
    ledger = solver.ledger.snapshot(matrix_xb=solver.matrix_xb, populations=solver.population_list(), beta_resolved_fraction=1.0)
    return {
        "kwn_time_h": solver.time_s / 3600.0,
        "kwn_step": float(solver.step),
        "kwn_matrix_xB": solver.matrix_xb,
        "kwn_beta_number_density_m3": beta.number_density_m3(),
        "kwn_beta_mean_radius_m": beta.mean_radius_m(),
        "kwn_beta_volume_fraction": beta.volume_fraction(),
        "kwn_beta_surface_area_m_inv": beta.specific_surface_area_m_inv(),
        "kwn_total_B_mol_m3": ledger.total_mol_m3,
        "kwn_matrix_B_mol_m3": ledger.matrix_mol_m3,
        "kwn_gp_B_mol_m3": ledger.gp_mol_m3,
        "kwn_beta_resolved_B_mol_m3": ledger.beta_resolved_mol_m3,
        "kwn_inventory_relative_residual": ledger.relative_residual,
    }


def _direction_rows(
    *,
    solver: KWNSolver,
    contract: ValidationContract,
    entries: Iterable[Mapping[str, float]],
) -> tuple[list[dict[str, Any]], bool]:
    """Evaluate only t=0 curvature-sign parity at initially occupied classes."""

    rates = solver.growth_rates()["beta"]
    rows: list[dict[str, Any]] = []
    matched = True
    for item in entries:
        index = int(item["grid_bin_index"])
        radius = float(item["radius_m"])
        equilibrium = contract.canonical.curvature_equilibrium_xb(contract.temperature_K, radius)
        threshold_direction = _sign_label(solver.matrix_xb - equilibrium)
        kwn_direction = _sign_label(float(rates[index]))
        direction_match = threshold_direction == kwn_direction
        matched = matched and direction_match
        rows.append(
            {
                "record_type": "t0_size_class_direction",
                "target_time_h": 0.0,
                "kwn_time_h": 0.0,
                "size_class_radius_m": radius,
                "source_particle_count": int(item["source_particle_count"]),
                "size_class_number_density_m3": float(item["number_density_m3"]),
                "matrix_xB": solver.matrix_xb,
                "pf_contract_curvature_xB": equilibrium,
                "pf_contract_direction": threshold_direction,
                "kwn_growth_rate_m_s": float(rates[index]),
                "kwn_direction": kwn_direction,
                "direction_match": str(direction_match).lower(),
                "kwn_run_status": "EVALUATED_T0",
                "pf_cuda_trajectory_status": PF_CUDA_NOT_RUN,
                "pf_cuda_beta_number_density_m3": "",
                "pf_cuda_beta_mean_radius_m": "",
                "pf_cuda_matrix_xB": "",
                "message": "host exact-contract curvature comparator; not a CUDA PF trajectory",
            }
        )
    return rows, matched


def _summary_row(target_time_h: float, metrics: Mapping[str, float], *, kwn_status: str, message: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {
        "record_type": "kwn_time_snapshot",
        "target_time_h": target_time_h,
        "kwn_time_h": metrics.get("kwn_time_h", ""),
        "size_class_radius_m": "",
        "source_particle_count": "",
        "size_class_number_density_m3": "",
        "matrix_xB": metrics.get("kwn_matrix_xB", ""),
        "pf_contract_curvature_xB": "",
        "pf_contract_direction": "",
        "kwn_growth_rate_m_s": "",
        "kwn_direction": "",
        "direction_match": "",
        "kwn_run_status": kwn_status,
        "pf_cuda_trajectory_status": PF_CUDA_NOT_RUN,
        "pf_cuda_beta_number_density_m3": "",
        "pf_cuda_beta_mean_radius_m": "",
        "pf_cuda_matrix_xB": "",
        "message": message,
    }
    row.update(metrics)
    return row


CSV_FIELDS = [
    "record_type",
    "target_time_h",
    "kwn_time_h",
    "size_class_radius_m",
    "source_particle_count",
    "size_class_number_density_m3",
    "matrix_xB",
    "pf_contract_curvature_xB",
    "pf_contract_direction",
    "kwn_growth_rate_m_s",
    "kwn_direction",
    "direction_match",
    "kwn_run_status",
    "pf_cuda_trajectory_status",
    "pf_cuda_beta_number_density_m3",
    "pf_cuda_beta_mean_radius_m",
    "pf_cuda_matrix_xB",
    "kwn_step",
    "kwn_beta_number_density_m3",
    "kwn_beta_mean_radius_m",
    "kwn_beta_volume_fraction",
    "kwn_beta_surface_area_m_inv",
    "kwn_total_B_mol_m3",
    "kwn_matrix_B_mol_m3",
    "kwn_gp_B_mol_m3",
    "kwn_beta_resolved_B_mol_m3",
    "kwn_inventory_relative_residual",
    "message",
]


def _pf_runtime_status() -> dict[str, Any]:
    executable_candidates = [ROOT / "main_cuda", ROOT / "build" / "main_cuda", ROOT / "bin" / "main_cuda"]
    executables = [str(path) for path in executable_candidates if path.is_file() and os.access(path, os.X_OK)]
    nvcc = shutil.which("nvcc")
    return {
        "status": PF_CUDA_NOT_RUN,
        "nvcc_path": nvcc,
        "pf_binary_candidates_found": executables,
        "reason": "this host control does not invoke CUDA PF; no local nvcc or qualified PF binary is available",
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_historical_not_run(path: Path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "comparison_id",
                "status",
                "historical_authority_status",
                "historical_psd_status",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "comparison_id": "historical_12h_broad_narrow_beta_only",
                "status": "NOT_RUN_HISTORICAL_AUTHORITY_OR_PSD_UNAVAILABLE",
                "historical_authority_status": HISTORICAL_AUTHORITY_STATUS,
                "historical_psd_status": HISTORICAL_PSD_STATUS,
                "reason": "no recoverable historical as-run broad/narrow 12 h PSD is available locally",
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--fixture-spec", type=Path, default=DEFAULT_FIXTURE_SPEC)
    parser.add_argument("--profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
    parser.add_argument("--numerical-template", type=Path, default=NUMERICAL_TEMPLATE)
    parser.add_argument("--csv", type=Path, default=OUTPUT_ROOT / "beta_only_code_control.csv")
    parser.add_argument("--summary", type=Path, default=OUTPUT_ROOT / "beta_only_code_control_summary.json")
    parser.add_argument("--historical-csv", type=Path, default=OUTPUT_ROOT / "beta_only_historical.csv")
    parser.add_argument("--checkpoint", type=Path, default=OUTPUT_ROOT / "beta_only_code_control_6h_checkpoint.npz")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for output in (args.csv, args.summary, args.historical_csv, args.checkpoint):
        if not _inside_output_root(output):
            raise SystemExit(f"refusing output outside validation result root: {output}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    contract = load_validation_contract(args.contract)
    try:
        fixture = load_host_96cube_fixture(args.fixture_spec, args.profile_root, contract)
    except FixtureConditionedHandoffError as error:
        raise SystemExit(f"frozen 96-cube fixture is unavailable or invalid: {error}") from error
    try:
        config, construction = build_same_contract_config(
            fixture=fixture,
            contract=contract,
            numerical_template_path=args.numerical_template,
        )
        solver = KWNSolver(config)
    except (ConfigurationError, InventoryError, RuntimeError, ValueError) as error:
        raise SystemExit(f"cannot construct same-contract beta-only control: {error}") from error

    rows, direction_pass = _direction_rows(
        solver=solver,
        contract=contract,
        entries=construction["config_mapping"]["populations"]["beta"]["initial"]["entries"],
    )
    rows.append(_summary_row(0.0, _snapshot_metrics(solver), kwn_status="EVALUATED_T0"))
    timepoints_reached = [0.0]
    checkpoint_provenance: dict[str, Any] = {
        "status": "NOT_WRITTEN",
        "path": str(args.checkpoint),
    }
    runtime_error: str | None = None
    runtime_failure_metrics: dict[str, float] | None = None
    try:
        solver.run_to_time(6.0 * 3600.0)
        rows.append(_summary_row(6.0, _snapshot_metrics(solver), kwn_status="RUN_OK"))
        timepoints_reached.append(6.0)
        solver.save_checkpoint(args.checkpoint)
        restarted = KWNSolver.load_checkpoint(config=config, path=args.checkpoint)
        before = solver.state_arrays()
        after = restarted.state_arrays()
        state_equal = all(np.array_equal(before[name], after[name]) for name in before)
        with np.load(args.checkpoint, allow_pickle=False) as archive:
            checkpoint_metadata = json.loads(str(archive["metadata_json"].item()))
        checkpoint_provenance = {
            "status": "PASS_KWN_HASH_BOUND_CHECKPOINT_RELOAD" if state_equal else "FAIL_KWN_CHECKPOINT_STATE_MISMATCH",
            "path": str(args.checkpoint),
            "sha256": _sha256_file(args.checkpoint),
            "metadata": checkpoint_metadata,
            "state_arrays_bitwise_equal_after_reload": state_equal,
        }
        if not state_equal:
            raise RuntimeError("KWN checkpoint/reload state arrays differ")
        solver = restarted
        solver.run_to_time(48.0 * 3600.0)
        rows.append(_summary_row(48.0, _snapshot_metrics(solver), kwn_status="RUN_OK"))
        timepoints_reached.append(48.0)
    except (
        RadiusGridOverflowError,
        SolverStateError,
        InventoryError,
        ThermodynamicDomainError,
        RuntimeError,
        ValueError,
    ) as error:
        runtime_error = f"{type(error).__name__}: {error}"
        runtime_failure_metrics = _snapshot_metrics(solver)
        next_time = 6.0 if 6.0 not in timepoints_reached else 48.0
        rows.append(
            _summary_row(
                next_time,
                runtime_failure_metrics,
                kwn_status="FAILED_KWN_RUNTIME",
                message=runtime_error,
            )
        )

    _write_csv(args.csv, rows)
    _write_historical_not_run(args.historical_csv)
    all_timepoints_reached = timepoints_reached == [0.0, 6.0, 48.0]
    if direction_pass and all_timepoints_reached:
        status = "PASS_BETA_ONLY_CODE_DIRECTION"
    elif direction_pass:
        status = "PARTIAL_BETA_ONLY_DIRECTION_T0_ONLY_KWN_RUNTIME_INCOMPLETE"
    else:
        status = "FAIL_BETA_ONLY_CODE_DIRECTION"
    summary = {
        "schema_version": "BETA_ONLY_SAME_CONTRACT_CODE_CONTROL_V1",
        "status": status,
        "scope": {
            "validation_only": True,
            "historical_as_run_claim": False,
            "comparison_semantics": "HOST_KWN_T0_CURVATURE_DIRECTION_ONLY_NOT_PF_TRAJECTORY_OR_TIMESCALE_COMPARISON",
            "gp_dynamics": "off",
            "beta_nucleation": "off",
        },
        "contract": {
            "hash": contract.contract_hash,
            "path": str(contract.path),
            "schema_version": contract.document["schema_version"],
        },
        "fixture": {
            "id": fixture.fixture_id,
            "hash": fixture.fixture_hash,
            "source_kind": fixture.source_kind,
            "source_details": dict(fixture.source_details),
            "source_total_inventory_mol": fixture.source_total_inventory_mol,
            "source_matrix_inventory_mol": fixture.source_matrix_inventory_mol,
            "source_resolved_inventory_mol": fixture.source_resolved_inventory_mol,
        },
        "construction": construction,
        "kwn": {
            "solver_version": KWNSolver.solver_version,
            "source_config_hash": config.source_config_hash,
            "validation_contract_hash": solver.contract_hash,
            "timepoints_requested_h": [0.0, 6.0, 48.0],
            "timepoints_reached_h": timepoints_reached,
            "runtime_error": runtime_error,
            "runtime_failure_metrics": runtime_failure_metrics,
            "runtime_failure_time_h": (
                None if runtime_failure_metrics is None else runtime_failure_metrics["kwn_time_h"]
            ),
            "checkpoint_provenance": checkpoint_provenance,
        },
        "t0_direction_gate": {
            "status": "PASS_BETA_ONLY_CODE_DIRECTION" if direction_pass else "FAIL_BETA_ONLY_CODE_DIRECTION",
            "definition": "each initially occupied KWN size class has the same growth/dissolution sign as matrix_xB minus the exact contract curvature equilibrium",
            "class_count": len(construction["config_mapping"]["populations"]["beta"]["initial"]["entries"]),
        },
        "pf_cuda_trajectory": _pf_runtime_status(),
        "historical_comparison": {
            "status": "NOT_RUN_HISTORICAL_AUTHORITY_OR_PSD_UNAVAILABLE",
            "authority_status": HISTORICAL_AUTHORITY_STATUS,
            "psd_status": HISTORICAL_PSD_STATUS,
            "csv": str(args.historical_csv),
        },
        "outputs": {
            "beta_only_code_control_csv": str(args.csv),
            "beta_only_historical_csv": str(args.historical_csv),
            "summary_json": str(args.summary),
        },
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "contract_hash": contract.contract_hash, "timepoints_reached_h": timepoints_reached, "direction_pass": direction_pass, "csv": str(args.csv), "summary": str(args.summary)}, sort_keys=True))
    return 0 if status == "PASS_BETA_ONLY_CODE_DIRECTION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
