#!/usr/bin/env python3
"""Capture the unmodified beta-only KWN positivity failure at bin level.

This program deliberately observes the solver through its existing public and
finite-volume helper methods.  It never changes a population or retries a
step: the rejected candidate is reconstructed from the exact same face-flux
formula immediately before ``advance_one`` is called.  It therefore provides
the required baseline evidence before a conservative positivity repair is
introduced.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from coupling.fixture_conditioned_handoff_v2 import (  # noqa: E402
    FixtureConditionedHandoffError,
    load_host_96cube_fixture,
    load_validation_contract,
)
from kwn_mvp.radius_grid import RadiusGrid  # noqa: E402
from kwn_mvp.solver import (  # noqa: E402
    KWNSolver,
    RadiusGridOverflowError,
    SolverConfig,
    SolverStateError,
    StepDiagnostics,
)
from kwn_mvp.nucleation import nucleation_rate  # noqa: E402

from scripts import run_beta_only_same_contract_control as beta_control  # noqa: E402


DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "kwn_pf_cuda_runtime_closure_v1"
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
LEGACY_SOLVER_COMMIT = "404b0578071120577b72a1e6f253a79ae1ca393c"
LEGACY_CFL_NUMBER_SUPPORT_RELATIVE_THRESHOLD = 1.0e-30


class LegacyUnlimitedFluxDiagnosticSolver(KWNSolver):
    """Exact pre-repair update used only to preserve the failure evidence.

    The production solver is deliberately not toggled back to its unsafe
    update.  This diagnostic subclass pins the two numerical choices from the
    frozen pre-repair commit: number-weighted CFL support and unmodified
    first-order upwind face fluxes.  It is therefore an observer/reproducer of
    the historical strict failure, not a selectable production mode.
    """

    solver_version = "internal_kwn_finite_volume_v1_legacy_unlimited_diagnosis"

    def _choose_dt(
        self, velocities: Mapping[str, np.ndarray], maximum_s: float | None
    ) -> tuple[float, float]:
        cfl_rates: list[float] = []
        for name, velocity in velocities.items():
            population = self.populations[name]
            cell_number = population.number_density_per_m4 * population.grid.widths_m
            total_number = float(np.sum(cell_number))
            if total_number == 0.0:
                continue
            active = cell_number > max(
                total_number * LEGACY_CFL_NUMBER_SUPPORT_RELATIVE_THRESHOLD,
                1.0e-300,
            )
            active[:-1] |= active[1:]
            active[1:] |= active[:-1]
            if np.any(active):
                cfl_rates.append(
                    float(
                        np.max(
                            np.abs(velocity[active])
                            / population.grid.widths_m[active]
                        )
                    )
                )
        max_rate = max(cfl_rates, default=0.0)
        cap = (
            self.config.max_dt_s
            if maximum_s is None
            else min(self.config.max_dt_s, maximum_s)
        )
        dt_s = cap if max_rate == 0.0 else min(cap, self.config.size_cfl / max_rate)
        if dt_s < self.config.min_dt_s:
            raise SolverStateError(
                f"CFL-limited step {dt_s:.3e} s is below configured "
                f"min_dt_s={self.config.min_dt_s:.3e} s"
            )
        return dt_s, max_rate * dt_s

    def _advect_population(
        self, population: Any, velocity_m_s: np.ndarray, dt_s: float
    ) -> tuple[float, float, float, int]:
        density = population.number_density_per_m4
        widths = population.grid.widths_m
        faces = _face_fluxes(density, velocity_m_s)
        lower_dissolution_flux = max(-faces[0], 0.0)
        rmax_outflow_flux = max(faces[-1], 0.0)
        existing_number = max(population.number_density_m3(), 1.0e-300)
        relative_outflow = rmax_outflow_flux * dt_s / existing_number
        if relative_outflow > self.config.rmax_outflow_relative_tolerance:
            raise RadiusGridOverflowError(
                f"{population.parameters.name} would lose {relative_outflow:.3e} "
                "of its number density through Rmax. Expand the configured "
                "radius grid; material was not discarded."
            )
        updated = density - dt_s * (faces[1:] - faces[:-1]) / widths
        roundoff_floor = -1.0e-280
        if np.any(updated < roundoff_floor) or not np.all(np.isfinite(updated)):
            raise SolverStateError(
                f"{population.parameters.name} finite-volume update violated "
                "positivity or finiteness"
            )
        roundoff_mask = (updated < 0.0) & (updated >= roundoff_floor)
        roundoff_zeroed = int(np.count_nonzero(roundoff_mask))
        updated[roundoff_mask] = 0.0
        population.number_density_per_m4[:] = updated
        outgoing = np.maximum(faces[1:], 0.0) + np.maximum(-faces[:-1], 0.0)
        cell_number = density * widths
        utilization = np.divide(
            dt_s * outgoing,
            cell_number,
            out=np.zeros_like(cell_number),
            where=cell_number > 0.0,
        )
        return (
            lower_dissolution_flux,
            rmax_outflow_flux,
            float(np.max(utilization)),
            roundoff_zeroed,
        )

    def advance_one(self, maximum_dt_s: float | None = None) -> StepDiagnostics:
        velocities = self.growth_rates()
        dt_s, cfl = self._choose_dt(velocities, maximum_dt_s)
        g_source = nucleation_rate(
            population=self.populations["g"].parameters,
            matrix_xb=self.matrix_xb,
            temperature_k=self.config.temperature_k,
            time_s=self.time_s,
        )
        beta_source = nucleation_rate(
            population=self.populations["beta"].parameters,
            matrix_xb=self.matrix_xb,
            temperature_k=self.config.temperature_k,
            time_s=self.time_s,
        )
        lower_flux = upper_flux = positivity_utilization = 0.0
        roundoff_zeroed_bin_count = 0
        for name in ("g", "beta"):
            lower, upper, utilization, zeroed = self._advect_population(
                self.populations[name], velocities[name], dt_s
            )
            lower_flux += lower
            upper_flux += upper
            positivity_utilization = max(positivity_utilization, utilization)
            roundoff_zeroed_bin_count += zeroed
        self.roundoff_zeroed_bin_count += roundoff_zeroed_bin_count
        self._inject_nucleation(self.populations["g"], g_source, dt_s)
        self._inject_nucleation(self.populations["beta"], beta_source, dt_s)
        if (
            any(np.any(value != 0.0) for value in velocities.values())
            or g_source.rate_m3_s != 0.0
            or beta_source.rate_m3_s != 0.0
        ):
            self.matrix_xb = self.ledger.recover_matrix_xb(self.population_list())
        self.time_s += dt_s
        self.step += 1
        inventory = self.ledger.snapshot(
            matrix_xb=self.matrix_xb, populations=self.population_list()
        )
        diagnostic = StepDiagnostics(
            step=self.step,
            time_s=self.time_s,
            dt_s=dt_s,
            size_cfl=cfl,
            positivity_utilization=positivity_utilization,
            roundoff_zeroed_bin_count=roundoff_zeroed_bin_count,
            matrix_xb=self.matrix_xb,
            inventory=inventory,
            gp_nucleation_rate_m3_s=g_source.rate_m3_s,
            beta_nucleation_rate_m3_s=beta_source.rate_m3_s,
            rmin_dissolution_flux_m3_s=lower_flux,
            rmax_outflow_flux_m3_s=upper_flux,
        )
        self.history.append(diagnostic)
        return diagnostic


def _face_fluxes(density: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Return the solver's first-order upwind radius-space face fluxes."""

    faces = np.empty(density.size + 1, dtype=np.float64)
    faces[0] = velocity[0] * density[0] if velocity[0] < 0.0 else 0.0
    internal_velocity = 0.5 * (velocity[:-1] + velocity[1:])
    faces[1:-1] = np.where(
        internal_velocity >= 0.0,
        internal_velocity * density[:-1],
        internal_velocity * density[1:],
    )
    faces[-1] = velocity[-1] * density[-1] if velocity[-1] > 0.0 else 0.0
    return faces


def _critical_radius_m(solver: KWNSolver, population_name: str) -> float | None:
    """Locate the contract-growth sign transition without changing state."""

    # This beta-only validation contract deliberately does not introduce a GP
    # thermodynamic model. Its empty GP compatibility carrier has no such
    # critical radius to report.
    if population_name != "beta":
        return None
    population = solver.population(population_name)
    lower = float(population.grid.edges_m[0])
    upper = float(population.grid.edges_m[-1])

    def residual(radius_m: float) -> float:
        equilibrium = solver.equilibrium_adapter.equilibrium_xb(
            np.asarray([radius_m], dtype=np.float64), population.parameters
        )
        return solver.matrix_xb - float(equilibrium[0])

    lower_value = residual(lower)
    upper_value = residual(upper)
    if lower_value == 0.0:
        return lower
    if upper_value == 0.0:
        return upper
    if lower_value * upper_value > 0.0:
        return None
    for _ in range(100):
        middle = 0.5 * (lower + upper)
        middle_value = residual(middle)
        if middle_value == 0.0:
            return middle
        if lower_value * middle_value < 0.0:
            upper = middle
            upper_value = middle_value
        else:
            lower = middle
            lower_value = middle_value
    return 0.5 * (lower + upper)


def _source_record(
    solver: KWNSolver, population_name: str, dt_s: float
) -> dict[str, float | None]:
    population = solver.population(population_name)
    source = nucleation_rate(
        population=population.parameters,
        matrix_xb=solver.matrix_xb,
        temperature_k=solver.config.temperature_k,
        time_s=solver.time_s,
    )
    return {
        "rate_m3_s": source.rate_m3_s,
        "injected_number_density_m3": source.rate_m3_s * dt_s,
        "configured_critical_radius_m": source.critical_radius_m,
        "growth_sign_critical_radius_m": _critical_radius_m(solver, population_name),
        "source_radius_m": source.radius_m,
        "barrier_j": source.barrier_j,
        "driving_energy_j_m3": source.driving_energy_j_m3,
    }


def _candidate_state(
    solver: KWNSolver,
    *,
    maximum_dt_s: float | None,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Calculate, without mutating, exactly the next finite-volume proposal."""

    velocities = solver.growth_rates()
    dt_s, cfl = solver._choose_dt(velocities, maximum_dt_s)
    details: dict[str, dict[str, np.ndarray]] = {}
    per_population: dict[str, dict[str, float | int | None]] = {}
    for name in ("g", "beta"):
        population = solver.population(name)
        density = population.number_density_per_m4.copy()
        widths = population.grid.widths_m
        velocity = velocities[name].copy()
        raw_faces = _face_fluxes(density, velocity)
        # This must remain the literal pre-repair proposal.  The production
        # limiter is intentionally absent from the baseline evidence path.
        faces = raw_faces
        candidate = density - dt_s * (raw_faces[1:] - raw_faces[:-1]) / widths
        cell_number = density * widths
        candidate_cell_number = candidate * widths
        outgoing_rate = np.maximum(raw_faces[1:], 0.0) + np.maximum(
            -raw_faces[:-1], 0.0
        )
        dt_max = np.divide(
            cell_number,
            outgoing_rate,
            out=np.full_like(cell_number, np.inf),
            where=outgoing_rate > 0.0,
        )
        first_negative = np.flatnonzero(candidate < 0.0)
        first_index = int(first_negative[0]) if first_negative.size else None
        per_population[name] = {
            "minimum_candidate_density_per_m4": float(np.min(candidate)),
            "minimum_candidate_cell_number_m3": float(np.min(candidate_cell_number)),
            "first_candidate_negative_bin": first_index,
            "first_candidate_negative_before_density_per_m4": (
                None if first_index is None else float(density[first_index])
            ),
            "first_candidate_negative_after_density_per_m4": (
                None if first_index is None else float(candidate[first_index])
            ),
            "first_candidate_negative_dt_max_s": (
                None if first_index is None else float(dt_max[first_index])
            ),
            "minimum_donor_dt_max_s": float(np.min(dt_max)),
            "lower_boundary_flux_m3_s": float(max(-raw_faces[0], 0.0)),
            "upper_boundary_flux_m3_s": float(max(raw_faces[-1], 0.0)),
            "raw_positivity_utilization": float(
                np.max(
                    np.divide(
                        dt_s * outgoing_rate,
                        cell_number,
                        out=np.zeros_like(cell_number),
                        where=cell_number > 0.0,
                    )
                )
            ),
        }
        details[name] = {
            "density_before_m4": density,
            "cell_number_before_m3": cell_number,
            "velocity_m_s": velocity,
            "face_flux_m3_s": raw_faces,
            "raw_face_flux_m3_s": raw_faces,
            "candidate_density_m4": candidate,
            "candidate_cell_number_m3": candidate_cell_number,
            "outgoing_rate_m3_s": outgoing_rate,
            "dt_max_s": dt_max,
        }
    before = solver.ledger.snapshot(
        matrix_xb=solver.matrix_xb, populations=solver.population_list(), beta_resolved_fraction=1.0
    )
    sources = {
        name: _source_record(solver, name, dt_s) for name in ("g", "beta")
    }
    summary: dict[str, Any] = {
        "time_s": solver.time_s,
        "time_h": solver.time_s / 3600.0,
        "step": solver.step,
        "dt_s": dt_s,
        "size_cfl": cfl,
        "raw_positivity_utilization": max(
            item["raw_positivity_utilization"] for item in per_population.values()
        ),
        "matrix_xB": solver.matrix_xb,
        "matrix_inventory_mol_m3": before.matrix_mol_m3,
        "particle_inventory_mol_m3": before.beta_resolved_mol_m3 + before.gp_mol_m3,
        "total_inventory_mol_m3": before.total_mol_m3,
        "inventory_relative_residual": before.relative_residual,
        "sources": sources,
        "populations": per_population,
    }
    return summary, details


def _accepted_record(diagnostic: Any) -> dict[str, Any]:
    return {
        "record_kind": "accepted",
        "step": diagnostic.step,
        "time_s": diagnostic.time_s,
        "time_h": diagnostic.time_s / 3600.0,
        "dt_s": diagnostic.dt_s,
        "size_cfl": diagnostic.size_cfl,
        "positivity_utilization": diagnostic.positivity_utilization,
        "roundoff_zeroed_bin_count": diagnostic.roundoff_zeroed_bin_count,
        "matrix_xB": diagnostic.matrix_xb,
        "matrix_inventory_mol_m3": diagnostic.inventory.matrix_mol_m3,
        "particle_inventory_mol_m3": (
            diagnostic.inventory.gp_mol_m3
            + diagnostic.inventory.beta_subgrid_mol_m3
            + diagnostic.inventory.beta_resolved_mol_m3
        ),
        "total_inventory_mol_m3": diagnostic.inventory.total_mol_m3,
        "inventory_relative_residual": diagnostic.inventory.relative_residual,
        "rmin_dissolution_flux_m3_s": diagnostic.rmin_dissolution_flux_m3_s,
        "rmax_outflow_flux_m3_s": diagnostic.rmax_outflow_flux_m3_s,
        "gp_nucleation_rate_m3_s": diagnostic.gp_nucleation_rate_m3_s,
        "beta_nucleation_rate_m3_s": diagnostic.beta_nucleation_rate_m3_s,
        "beta_growth_sign_critical_radius_m": "",
        "candidate_negative_population": "",
        "candidate_negative_bin": "",
        "candidate_before_density_per_m4": "",
        "candidate_after_density_per_m4": "",
        "candidate_dt_max_s": "",
        "message": "",
    }


def _rejected_record(candidate: Mapping[str, Any], error: BaseException) -> dict[str, Any]:
    negative_population = ""
    values: Mapping[str, Any] | None = None
    for name in ("g", "beta"):
        item = candidate["populations"][name]
        if item["first_candidate_negative_bin"] is not None:
            negative_population = name
            values = item
            break
    return {
        "record_kind": "rejected",
        "step": candidate["step"] + 1,
        "time_s": candidate["time_s"] + candidate["dt_s"],
        "time_h": (candidate["time_s"] + candidate["dt_s"]) / 3600.0,
        "dt_s": candidate["dt_s"],
        "size_cfl": candidate["size_cfl"],
        "positivity_utilization": candidate["raw_positivity_utilization"],
        "roundoff_zeroed_bin_count": "",
        "matrix_xB": candidate["matrix_xB"],
        "matrix_inventory_mol_m3": candidate["matrix_inventory_mol_m3"],
        "particle_inventory_mol_m3": candidate["particle_inventory_mol_m3"],
        "total_inventory_mol_m3": candidate["total_inventory_mol_m3"],
        "inventory_relative_residual": candidate["inventory_relative_residual"],
        "rmin_dissolution_flux_m3_s": "",
        "rmax_outflow_flux_m3_s": "",
        "gp_nucleation_rate_m3_s": candidate["sources"]["g"]["rate_m3_s"],
        "beta_nucleation_rate_m3_s": candidate["sources"]["beta"]["rate_m3_s"],
        "beta_growth_sign_critical_radius_m": candidate["sources"]["beta"][
            "growth_sign_critical_radius_m"
        ],
        "candidate_negative_population": negative_population,
        "candidate_negative_bin": "" if values is None else values["first_candidate_negative_bin"],
        "candidate_before_density_per_m4": (
            "" if values is None else values["first_candidate_negative_before_density_per_m4"]
        ),
        "candidate_after_density_per_m4": (
            "" if values is None else values["first_candidate_negative_after_density_per_m4"]
        ),
        "candidate_dt_max_s": "" if values is None else values["first_candidate_negative_dt_max_s"],
        "message": f"{type(error).__name__}: {error}",
    }


def _build_solver(args: argparse.Namespace) -> tuple[KWNSolver, dict[str, Any], Any, Any]:
    contract = load_validation_contract(args.contract)
    fixture = load_host_96cube_fixture(args.fixture_spec, args.profile_root, contract)
    base_config, construction = beta_control.build_same_contract_config(
        fixture=fixture,
        contract=contract,
        numerical_template_path=args.numerical_template,
    )
    mapping = deepcopy(construction["config_mapping"])
    if args.bins != base_config.grid.bins:
        minimum = float(mapping["radius_grid"]["minimum_m"])
        maximum = float(mapping["radius_grid"]["maximum_m"])
        grid = RadiusGrid.logarithmic(minimum, maximum, args.bins)
        entries, projection = beta_control._project_fixture_resolved_psd(fixture, contract, grid)
        mapping["radius_grid"]["bins"] = int(args.bins)
        mapping["populations"]["beta"]["initial"]["entries"] = entries
        construction["resolved_psd_projection"] = projection
    mapping["simulation"]["max_dt_s"] = (
        float(mapping["simulation"]["max_dt_s"]) * args.max_dt_factor
    )
    mapping["simulation"].pop("positivity_safety", None)
    mapping["simulation"].pop("cfl_active_inventory_relative_threshold", None)
    construction["legacy_reproduction"] = {
        "source_commit": LEGACY_SOLVER_COMMIT,
        "cfl_support": "number_weighted_relative_threshold",
        "cfl_active_population_relative_threshold": (
            LEGACY_CFL_NUMBER_SUPPORT_RELATIVE_THRESHOLD
        ),
        "face_flux_limiter": "OFF_PRE_REPAIR_BASELINE_ONLY",
        "negative_bin_handling": "STRICT_FAILURE_BELOW_SUBNORMAL_ROUNDOFF",
    }
    return (
        LegacyUnlimitedFluxDiagnosticSolver(SolverConfig.from_mapping(mapping)),
        construction,
        contract,
        fixture,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--fixture-spec", type=Path, default=DEFAULT_FIXTURE_SPEC)
    parser.add_argument("--profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
    parser.add_argument("--numerical-template", type=Path, default=beta_control.NUMERICAL_TEMPLATE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--max-dt-factor", type=float, default=1.0)
    parser.add_argument(
        "--end-time-h",
        type=float,
        default=48.0,
        help="physical-time ceiling for a diagnostic that does not fail first",
    )
    parser.add_argument(
        "--max-accepted-steps",
        type=int,
        default=50000,
        help="explicit work ceiling for a no-failure diagnostic cell",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.bins < 2
        or not math.isfinite(args.max_dt_factor)
        or args.max_dt_factor <= 0.0
        or not math.isfinite(args.end_time_h)
        or args.end_time_h <= 0.0
        or args.max_accepted_steps < 1
    ):
        raise SystemExit("invalid bins, dt factor, end time, or accepted-step ceiling")
    output_root = args.output_root.resolve()
    allowed = DEFAULT_OUTPUT_ROOT.resolve()
    try:
        output_root.relative_to(allowed)
    except ValueError as error:
        raise SystemExit("refusing output outside this task's output root") from error
    output_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_root / "kwn_positivity_failure_snapshot.npz"
    trace_path = output_root / "kwn_positivity_failure_trace.csv"
    summary_path = output_root / "kwn_positivity_failure_diagnosis.json"
    if any(path.exists() for path in (snapshot_path, trace_path, summary_path)):
        raise SystemExit("refusing to overwrite existing positivity-diagnosis artifacts")
    try:
        solver, construction, contract, fixture = _build_solver(args)
    except (FixtureConditionedHandoffError, ValueError, RuntimeError) as error:
        raise SystemExit(f"cannot construct frozen beta-only baseline: {error}") from error

    records: deque[dict[str, Any]] = deque(maxlen=20)
    candidate: dict[str, Any] | None = None
    details: dict[str, dict[str, np.ndarray]] | None = None
    error: BaseException | None = None
    end_time_s = args.end_time_h * 3600.0
    while solver.time_s < end_time_s and solver.step < args.max_accepted_steps:
        candidate, details = _candidate_state(
            solver, maximum_dt_s=end_time_s - solver.time_s
        )
        try:
            diagnostic = solver.advance_one(maximum_dt_s=end_time_s - solver.time_s)
        except (SolverStateError, RadiusGridOverflowError, RuntimeError, ValueError) as caught:
            error = caught
            break
        records.append(_accepted_record(diagnostic))
    assert candidate is not None and details is not None
    rows = list(records)
    if error is not None:
        rejected = _rejected_record(candidate, error)
        rows.append(rejected)
    else:
        rejected = None
    beta_failure = candidate["populations"]["beta"]
    first_negative = beta_failure["first_candidate_negative_bin"]
    if error is not None and first_negative is None:
        raise SystemExit("solver failed without a negative beta candidate; do not classify as positivity")
    with trace_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    completed_without_failure = error is None
    final_inventory = solver.ledger.snapshot(
        matrix_xb=solver.matrix_xb, populations=solver.population_list(), beta_resolved_fraction=1.0
    )
    metadata = {
        "schema_version": "KWN_POSITIVITY_FAILURE_DIAGNOSIS_V1",
        "status": (
            "NO_STRICT_POSITIVITY_FAILURE_WITHIN_DIAGNOSTIC_WINDOW"
            if completed_without_failure
            else "REPRODUCED_STRICT_POSITIVITY_FAILURE"
        ),
        "error": None if error is None else f"{type(error).__name__}: {error}",
        "contract_hash": contract.contract_hash,
        "fixture_id": fixture.fixture_id,
        "fixture_hash": fixture.fixture_hash,
        "kwn_solver_version": solver.solver_version,
        "radius_bins": args.bins,
        "max_dt_factor": args.max_dt_factor,
        "construction": construction,
        "failure": None if completed_without_failure else candidate,
        "completion": {
            "time_s": solver.time_s,
            "time_h": solver.time_s / 3600.0,
            "accepted_steps": solver.step,
            "reached_time_ceiling": bool(solver.time_s >= end_time_s),
            "reached_step_ceiling": bool(solver.step >= args.max_accepted_steps),
            "inventory_relative_residual": final_inventory.relative_residual,
        },
        "classification_evidence": {
            "candidate_negative_beta_bin": first_negative if error is not None else None,
            "candidate_dt_s": candidate["dt_s"],
            "donor_dt_max_s": (
                beta_failure["first_candidate_negative_dt_max_s"] if error is not None else None
            ),
            "strictly_exceeds_donor_bound": bool(error is not None and (
                candidate["dt_s"] > beta_failure["first_candidate_negative_dt_max_s"]
            )),
            "lower_boundary_flux_m3_s": beta_failure["lower_boundary_flux_m3_s"],
            "upper_boundary_flux_m3_s": beta_failure["upper_boundary_flux_m3_s"],
            "raw_positivity_utilization": beta_failure["raw_positivity_utilization"],
            "beta_growth_sign_critical_radius_m": candidate["sources"]["beta"][
                "growth_sign_critical_radius_m"
            ],
            "gp_nucleation_rate_m3_s": candidate["sources"]["g"]["rate_m3_s"],
            "beta_nucleation_rate_m3_s": candidate["sources"]["beta"]["rate_m3_s"],
            "source_overdraw": False,
        },
    }
    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "radius_bin_edges_m": solver.config.grid.edges_m,
        "radius_bin_centres_m": solver.config.grid.centres_m,
    }
    for population_name, values in details.items():
        for field, value in values.items():
            arrays[f"{population_name}_{field}"] = value
    np.savez_compressed(snapshot_path, **arrays)
    summary_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "failure_time_h": None if completed_without_failure else candidate["time_h"],
                "rejected_end_time_h": None if rejected is None else rejected["time_h"],
                "beta_bin": None if completed_without_failure else first_negative,
                "dt_s": candidate["dt_s"],
                "donor_dt_max_s": (
                    None if completed_without_failure else beta_failure["first_candidate_negative_dt_max_s"]
                ),
                "trace": str(trace_path),
                "snapshot": str(snapshot_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
