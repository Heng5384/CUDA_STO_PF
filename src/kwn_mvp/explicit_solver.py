"""Independent conservative explicit SSPRK2 KWN reference solver.

This module deliberately does not alter the production implicit solver.  It
uses the same finite-volume face flux and physical-Rmin boundary kernels, but
advances a private two-stage SSPRK2 state with an exact per-cell donor bound.
The resulting solver is intended as a small, deterministic numerical
reference and fails closed whenever the requested explicit step cannot be
positive under the configured minimum timestep.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
from numpy.typing import NDArray

from .growth import growth_rate_m_s
from .ledger import InventoryError, InventorySnapshot
from .lower_boundary import (
    boundary_growth_velocity,
    boundary_inventory_diagnostic,
    boundary_number_flux_diagnostic,
    boundary_radius,
    particle_inventory_at_radius,
)
from .nucleation import NucleationResult, nucleation_rate
from .populations import Population
from .solver import KWNSolver, RadiusGridOverflowError, SolverConfig, SolverStateError


@dataclass(frozen=True)
class DonorBoundDiagnostics:
    """Exact forward-Euler donor bound for one full two-population state."""

    bound_s: float
    population_name: str
    cell_index: int
    donor_number_m3: float
    outgoing_flux_m3_s: float


@dataclass(frozen=True)
class ExplicitStepDiagnostics:
    """One accepted SSPRK2 macro-step and its conservative stage telemetry."""

    step: int
    time_s: float
    dt_s: float
    donor_safety: float
    stage0_donor_bound_s: float
    stage1_donor_bound_s: float
    stage0_donor_bound_population: str
    stage0_donor_bound_cell_index: int
    stage1_donor_bound_population: str
    stage1_donor_bound_cell_index: int
    size_cfl: float
    positivity_utilization: float
    roundoff_zeroed_bin_count: int
    matrix_xb: float
    inventory: InventorySnapshot
    gp_nucleation_rate_m3_s: float
    beta_nucleation_rate_m3_s: float
    rmin_dissolution_flux_m3_s: float
    rmax_outflow_flux_m3_s: float
    beta_rmin_number_flux_m3_s: float
    beta_rmin_volume_flux_s: float
    beta_rmin_mol_b_flux_mol_m3_s: float
    beta_boundary_radius_m: float
    beta_boundary_growth_velocity_m_s: float
    rejected_step_count: int
    rejected_step_count_delta: int
    stage0_donor_courant: float
    stage1_donor_courant: float


@dataclass(frozen=True)
class _StageOperator:
    """Frozen explicit operator data for one SSPRK2 stage."""

    rhs_per_m4_s: Dict[str, NDArray[np.float64]]
    face_velocity_m_s: Dict[str, NDArray[np.float64]]
    face_flux_m3_s: Dict[str, NDArray[np.float64]]
    centre_velocity_m_s: Dict[str, NDArray[np.float64]]
    source_per_m4_s: Dict[str, NDArray[np.float64]]
    nucleation: Dict[str, NucleationResult]


class _StepRejected(RuntimeError):
    """Private signal used to retry an untouched SSPRK2 macro-step."""

    def __init__(self, reason: str, *, allowed_dt_s: float | None = None) -> None:
        super().__init__(reason)
        self.allowed_dt_s = allowed_dt_s


class ExplicitSSPRK2Solver(KWNSolver):
    """Conservative upwind KWN solver advanced by explicit SSPRK2.

    The class inherits initial-state construction, the equilibrium adapter,
    and the inventory ledger from :class:`KWNSolver`.  It intentionally never
    invokes its inherited implicit update path.  Every trial step operates on
    copied density arrays, so a positivity, stage-CFL, or Rmax rejection
    leaves the accepted state bit-for-bit unchanged.
    """

    solver_version = "explicit_kwn_ssprk2_donor_bound_v1"
    _MAX_REJECTIONS_PER_STEP = 64

    def __init__(self, config: SolverConfig, donor_safety: float = 1.0) -> None:
        if not isinstance(donor_safety, (int, float)) or not math.isfinite(float(donor_safety)):
            raise ValueError("donor_safety must be finite")
        if not 0.0 < float(donor_safety) <= 1.0:
            raise ValueError("donor_safety must lie in (0, 1]")
        super().__init__(config)
        self.donor_safety = float(donor_safety)
        self.rejected_step_count = 0
        self.history: list[ExplicitStepDiagnostics] = []

    def _state_from_populations(self) -> Dict[str, NDArray[np.float64]]:
        return {
            name: self.populations[name].number_density_per_m4.copy()
            for name in ("g", "beta")
        }

    def _state_populations(
        self, state: Mapping[str, NDArray[np.float64]]
    ) -> list[Population]:
        populations: list[Population] = []
        for name in ("g", "beta"):
            density = np.asarray(state[name], dtype=np.float64)
            population = self.populations[name]
            if density.shape != population.number_density_per_m4.shape:
                raise SolverStateError(f"{name} stage density shape differs from the configured grid")
            if not np.all(np.isfinite(density)) or np.any(density < 0.0):
                raise SolverStateError(f"{name} stage density is not finite and non-negative")
            populations.append(
                Population(
                    population.parameters,
                    population.grid,
                    density.copy(),
                    production_quadrature=population.production_quadrature,
                )
            )
        return populations

    def _recover_stage_matrix_xb(self, state: Mapping[str, NDArray[np.float64]]) -> float:
        """Close the matrix composition algebraically for a trial state."""

        try:
            matrix_xb = self.ledger.recover_matrix_xb(self._state_populations(state))
        except InventoryError as error:
            raise _StepRejected(f"stage inventory closure failed: {error}") from error
        if not math.isfinite(matrix_xb) or not 0.0 <= matrix_xb <= 1.0:
            raise _StepRejected("stage inventory closure returned an invalid matrix composition")
        return float(matrix_xb)

    def _stage_velocity_and_faces(
        self,
        name: str,
        density_per_m4: NDArray[np.float64],
        matrix_xb: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return stage-local centre and shared-physical-face velocities.

        This hook is deliberately small so synthetic transport tests can
        prescribe a velocity field without changing the production growth or
        lower-boundary contracts.  The default path evaluates the lower face
        directly at the physical grid edge through ``boundary_growth_velocity``.
        """

        population = self.populations[name]
        density = np.asarray(density_per_m4, dtype=np.float64)
        if density.shape != population.number_density_per_m4.shape:
            raise SolverStateError(f"{name} stage density shape differs from the configured grid")
        if not np.all(np.isfinite(density)) or np.any(density < 0.0):
            raise SolverStateError(f"{name} stage density is not finite and non-negative")
        if (
            float(np.sum(density * population.grid.widths_m)) == 0.0
            and population.parameters.nucleation.get("mode", "off") == "off"
        ):
            velocity = np.zeros(population.grid.bins, dtype=np.float64)
            return velocity, self._face_velocities(velocity, lower_boundary_velocity_m_s=0.0)
        velocity = growth_rate_m_s(
            radii_m=population.grid.centres_m,
            matrix_xb=matrix_xb,
            equilibrium_xb=self.equilibrium_adapter.equilibrium_xb(
                population.grid.centres_m, population.parameters
            ),
            parameters=population.parameters,
        )
        lower = boundary_growth_velocity(
            radius_m=boundary_radius(population.grid),
            matrix_xb=matrix_xb,
            parameters=population.parameters,
            equilibrium_adapter=self.equilibrium_adapter,
        )
        faces = self._face_velocities(velocity, lower_boundary_velocity_m_s=lower)
        return np.asarray(velocity, dtype=np.float64), faces

    def _stage_sources(
        self, matrix_xb: float, time_s: float
    ) -> tuple[Dict[str, NDArray[np.float64]], Dict[str, NucleationResult]]:
        """Assemble non-negative source rates into radius bins without mutation."""

        sources: Dict[str, NDArray[np.float64]] = {}
        results: Dict[str, NucleationResult] = {}
        for name in ("g", "beta"):
            population = self.populations[name]
            result = nucleation_rate(
                population=population.parameters,
                matrix_xb=matrix_xb,
                temperature_k=self.config.temperature_k,
                time_s=time_s,
            )
            if (
                not math.isfinite(result.rate_m3_s)
                or result.rate_m3_s < 0.0
                or (result.rate_m3_s > 0.0 and (not math.isfinite(result.radius_m) or result.radius_m <= 0.0))
            ):
                raise SolverStateError(f"{name} nucleation source is not finite and non-negative")
            source = np.zeros(population.grid.bins, dtype=np.float64)
            if result.rate_m3_s > 0.0:
                try:
                    index = population.grid.bin_index(result.radius_m)
                except ValueError as error:
                    raise SolverStateError(
                        f"{name} nucleation radius lies outside the configured radius grid"
                    ) from error
                source[index] = result.rate_m3_s / population.grid.widths_m[index]
            sources[name] = source
            results[name] = result
        return sources, results

    def _stage_operator(
        self,
        state: Mapping[str, NDArray[np.float64]],
        matrix_xb: float,
        time_s: float,
    ) -> _StageOperator:
        """Build the literal shared-face conservative operator at one stage."""

        sources, nucleation = self._stage_sources(matrix_xb, time_s)
        rhs: Dict[str, NDArray[np.float64]] = {}
        face_velocities: Dict[str, NDArray[np.float64]] = {}
        face_fluxes: Dict[str, NDArray[np.float64]] = {}
        centre_velocities: Dict[str, NDArray[np.float64]] = {}
        for name in ("g", "beta"):
            population = self.populations[name]
            density = np.asarray(state[name], dtype=np.float64)
            velocity, faces = self._stage_velocity_and_faces(name, density, matrix_xb)
            if velocity.shape != density.shape or faces.shape != (density.size + 1,):
                raise SolverStateError(f"{name} stage velocity shape is inconsistent with its grid")
            if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(faces)):
                raise SolverStateError(f"{name} stage velocity is not finite")
            flux = KWNSolver._upwind_face_fluxes(
                density, velocity, face_velocity_m_s=faces
            )
            rhs[name] = -(flux[1:] - flux[:-1]) / population.grid.widths_m + sources[name]
            face_velocities[name] = faces
            face_fluxes[name] = flux
            centre_velocities[name] = velocity
        return _StageOperator(
            rhs_per_m4_s=rhs,
            face_velocity_m_s=face_velocities,
            face_flux_m3_s=face_fluxes,
            centre_velocity_m_s=centre_velocities,
            source_per_m4_s=sources,
            nucleation=nucleation,
        )

    def _donor_bound_for_state(
        self,
        state: Mapping[str, NDArray[np.float64]],
        matrix_xb: float,
        time_s: float,
        *,
        operator: _StageOperator | None = None,
    ) -> DonorBoundDiagnostics:
        """Return the exact full-domain bound from each cell's two outflows.

        For a cell ``i`` the forward-Euler donor coefficient is the sum of
        its right-going and left-going *face velocities*, divided by its
        width.  The inverse is state-independent for the frozen upwind
        operator and therefore remains a positivity guarantee even for an
        initially empty cell.  Current donor number and flux are retained as
        diagnostics only.  Summing both faces is essential near a velocity
        reversal; a max-face CFL is not a positivity bound there.
        """

        assembled = self._stage_operator(state, matrix_xb, time_s) if operator is None else operator
        best = DonorBoundDiagnostics(
            bound_s=math.inf,
            population_name="",
            cell_index=-1,
            donor_number_m3=0.0,
            outgoing_flux_m3_s=0.0,
        )
        for name in ("g", "beta"):
            density = np.asarray(state[name], dtype=np.float64)
            widths = self.populations[name].grid.widths_m
            faces = assembled.face_velocity_m_s[name]
            outgoing_speed = np.maximum(faces[1:], 0.0) + np.maximum(-faces[:-1], 0.0)
            numbers = density * widths
            valid = outgoing_speed > 0.0
            if not np.any(valid):
                continue
            candidate = np.full(widths.shape, math.inf, dtype=np.float64)
            candidate[valid] = widths[valid] / outgoing_speed[valid]
            index = int(np.argmin(candidate))
            value = float(candidate[index])
            if not math.isfinite(value) or value < 0.0:
                raise SolverStateError(f"{name} donor bound is not finite and non-negative")
            if value < best.bound_s:
                best = DonorBoundDiagnostics(
                    bound_s=value,
                    population_name=name,
                    cell_index=index,
                    donor_number_m3=float(numbers[index]),
                    outgoing_flux_m3_s=float(outgoing_speed[index] * density[index]),
                )
        return best

    def donor_bound_diagnostics(self) -> DonorBoundDiagnostics:
        """Return the current full-domain donor limiter and its controlling cell."""

        state = self._state_from_populations()
        return self._donor_bound_for_state(state, self.matrix_xb, self.time_s)

    def exact_donor_bound_s(self) -> float:
        """Return the current un-safetied full-domain exact donor bound in s."""

        return self.donor_bound_diagnostics().bound_s

    def _assert_rmax_closed(self, operator: _StageOperator, state: Mapping[str, NDArray[np.float64]], dt_s: float) -> None:
        """Fail before any trial state can discard material through ``Rmax``."""

        for name in ("g", "beta"):
            flux = float(max(operator.face_flux_m3_s[name][-1], 0.0))
            existing = max(
                float(np.sum(np.asarray(state[name], dtype=np.float64) * self.populations[name].grid.widths_m)),
                1.0e-300,
            )
            relative_outflow = flux * dt_s / existing
            if relative_outflow > self.config.rmax_outflow_relative_tolerance:
                raise RadiusGridOverflowError(
                    f"{name} would lose {relative_outflow:.3e} of its number density through Rmax. "
                    "Expand the configured radius grid; material was not discarded."
                )

    def _forward_stage(
        self,
        state: Mapping[str, NDArray[np.float64]],
        operator: _StageOperator,
        dt_s: float,
        previous_matrix_xb: float,
    ) -> tuple[Dict[str, NDArray[np.float64]], float]:
        """Take an un-clamped forward stage and algebraically close its matrix."""

        updated: Dict[str, NDArray[np.float64]] = {}
        for name in ("g", "beta"):
            value = np.asarray(state[name], dtype=np.float64) + dt_s * operator.rhs_per_m4_s[name]
            if not np.all(np.isfinite(value)):
                raise _StepRejected(f"{name} forward stage produced a non-finite density")
            if np.any(value < 0.0):
                raise _StepRejected(f"{name} forward stage violated positivity")
            updated[name] = value
        changed = any(not np.array_equal(updated[name], state[name]) for name in ("g", "beta"))
        matrix_xb = self._recover_stage_matrix_xb(updated) if changed else float(previous_matrix_xb)
        return updated, matrix_xb

    @staticmethod
    def _same_state(
        left: Mapping[str, NDArray[np.float64]], right: Mapping[str, NDArray[np.float64]]
    ) -> bool:
        return all(np.array_equal(left[name], right[name]) for name in ("g", "beta"))

    def _cap_dt(self, maximum_dt_s: float | None) -> float:
        if maximum_dt_s is None:
            return float(self.config.max_dt_s)
        maximum = float(maximum_dt_s)
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum_dt_s must be finite and positive when supplied")
        return min(float(self.config.max_dt_s), maximum)

    def _require_usable_dt(self, dt_s: float, *, reason: str) -> None:
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise SolverStateError(f"explicit donor-bound step is not positive ({reason})")
        if dt_s < self.config.min_dt_s:
            raise SolverStateError(
                f"explicit donor-bound step {dt_s:.17e} s is below configured "
                f"min_dt_s={self.config.min_dt_s:.17e} s ({reason})"
            )

    def _raw_face_rate(self, operator: _StageOperator) -> tuple[float, int, int, float, float]:
        candidates = []
        for name in ("g", "beta"):
            candidates.append(
                (
                    name,
                    self._raw_face_operator_rate(
                        operator.centre_velocity_m_s[name],
                        self.populations[name].grid.widths_m,
                        face_velocity_m_s=operator.face_velocity_m_s[name],
                    ),
                )
            )
        _, result = max(candidates, key=lambda item: item[1][0])
        return result

    def advance_one(self, maximum_dt_s: float | None = None) -> ExplicitStepDiagnostics:
        """Advance one accepted conservative SSPRK2 step, with full rollback.

        No member of ``self.populations`` is touched until both stages, both
        Rmax checks, positivity, and algebraic matrix closure have succeeded.
        A stage-one donor restriction or failed stage therefore retries from
        the original accepted state instead of partially committing an Euler
        stage or clamping a negative bin.
        """

        cap = self._cap_dt(maximum_dt_s)
        state0 = self._state_from_populations()
        matrix0 = float(self.matrix_xb)
        time0 = float(self.time_s)
        initial_operator = self._stage_operator(state0, matrix0, time0)
        initial_bound = self._donor_bound_for_state(
            state0, matrix0, time0, operator=initial_operator
        )
        initial_allowed = self.donor_safety * initial_bound.bound_s
        dt_s = cap if math.isinf(initial_allowed) else min(cap, initial_allowed)
        self._require_usable_dt(dt_s, reason="stage-0 exact donor bound")
        rejected_before = self.rejected_step_count

        for _ in range(self._MAX_REJECTIONS_PER_STEP):
            try:
                # Rebuild stage 0 after a rejected trial so every retry starts
                # from the same accepted state and frozen physical time.
                stage0 = self._stage_operator(state0, matrix0, time0)
                donor0 = self._donor_bound_for_state(
                    state0, matrix0, time0, operator=stage0
                )
                allowed0 = self.donor_safety * donor0.bound_s
                if not math.isinf(allowed0) and dt_s > allowed0:
                    raise _StepRejected("stage-0 exact donor bound tightened", allowed_dt_s=allowed0)
                self._assert_rmax_closed(stage0, state0, dt_s)
                state1, matrix1 = self._forward_stage(state0, stage0, dt_s, matrix0)

                stage1 = self._stage_operator(state1, matrix1, time0 + dt_s)
                donor1 = self._donor_bound_for_state(
                    state1, matrix1, time0 + dt_s, operator=stage1
                )
                allowed1 = self.donor_safety * donor1.bound_s
                if not math.isinf(allowed1) and dt_s > allowed1:
                    raise _StepRejected("stage-1 exact donor bound tightened", allowed_dt_s=allowed1)
                self._assert_rmax_closed(stage1, state1, dt_s)
                state2, _ = self._forward_stage(state1, stage1, dt_s, matrix1)

                state_new = {
                    name: 0.5 * (state0[name] + state2[name]) for name in ("g", "beta")
                }
                for name, value in state_new.items():
                    if not np.all(np.isfinite(value)):
                        raise _StepRejected(f"{name} SSPRK2 combination produced a non-finite density")
                    if np.any(value < 0.0):
                        raise _StepRejected(f"{name} SSPRK2 combination violated positivity")
                matrix_new = (
                    matrix0
                    if self._same_state(state_new, state0)
                    else self._recover_stage_matrix_xb(state_new)
                )
                try:
                    inventory_new = self.ledger.snapshot(
                        matrix_xb=matrix_new,
                        populations=self._state_populations(state_new),
                    )
                except InventoryError as error:
                    raise _StepRejected(f"final inventory closure failed: {error}") from error
                break
            except _StepRejected as error:
                self.rejected_step_count += 1
                retry_limit = 0.5 * dt_s
                if error.allowed_dt_s is not None:
                    retry_limit = min(retry_limit, error.allowed_dt_s)
                self._require_usable_dt(retry_limit, reason=str(error))
                dt_s = retry_limit
        else:
            raise SolverStateError(
                f"explicit SSPRK2 step exceeded {self._MAX_REJECTIONS_PER_STEP} full-step rejections"
            )

        # All state changes occur in one final commit after the accepted trial.
        for name in ("g", "beta"):
            self.populations[name].number_density_per_m4[:] = state_new[name]
        self.matrix_xb = matrix_new
        self.time_s = time0 + dt_s
        self.step += 1
        inventory = inventory_new

        lower0 = {
            name: boundary_number_flux_diagnostic(
                stage0.face_velocity_m_s[name][0], state0[name][0]
            )
            for name in ("g", "beta")
        }
        lower1 = {
            name: boundary_number_flux_diagnostic(
                stage1.face_velocity_m_s[name][0], state1[name][0]
            )
            for name in ("g", "beta")
        }
        upper0 = {name: max(stage0.face_flux_m3_s[name][-1], 0.0) for name in ("g", "beta")}
        upper1 = {name: max(stage1.face_flux_m3_s[name][-1], 0.0) for name in ("g", "beta")}
        beta_lower_flux = 0.5 * (lower0["beta"] + lower1["beta"])
        beta = self.populations["beta"]
        beta_radius = boundary_radius(beta.grid)
        beta_flux = boundary_inventory_diagnostic(
            beta_lower_flux,
            particle_inventory_at_radius(
                beta_radius,
                x_b=beta.parameters.x_b,
                molar_volume_m3_mol=beta.parameters.molar_volume_m3_mol,
            ),
        )
        raw_rate, _, _, _, _ = self._raw_face_rate(stage0)
        donor0_courant = 0.0 if math.isinf(donor0.bound_s) else dt_s / donor0.bound_s
        donor1_courant = 0.0 if math.isinf(donor1.bound_s) else dt_s / donor1.bound_s
        diagnostic = ExplicitStepDiagnostics(
            step=self.step,
            time_s=self.time_s,
            dt_s=dt_s,
            donor_safety=self.donor_safety,
            stage0_donor_bound_s=donor0.bound_s,
            stage1_donor_bound_s=donor1.bound_s,
            stage0_donor_bound_population=donor0.population_name,
            stage0_donor_bound_cell_index=donor0.cell_index,
            stage1_donor_bound_population=donor1.population_name,
            stage1_donor_bound_cell_index=donor1.cell_index,
            size_cfl=raw_rate * dt_s,
            positivity_utilization=max(donor0_courant, donor1_courant),
            roundoff_zeroed_bin_count=0,
            matrix_xb=self.matrix_xb,
            inventory=inventory,
            gp_nucleation_rate_m3_s=0.5
            * (stage0.nucleation["g"].rate_m3_s + stage1.nucleation["g"].rate_m3_s),
            beta_nucleation_rate_m3_s=0.5
            * (stage0.nucleation["beta"].rate_m3_s + stage1.nucleation["beta"].rate_m3_s),
            rmin_dissolution_flux_m3_s=0.5
            * (lower0["g"] + lower0["beta"] + lower1["g"] + lower1["beta"]),
            rmax_outflow_flux_m3_s=0.5
            * (upper0["g"] + upper0["beta"] + upper1["g"] + upper1["beta"]),
            beta_rmin_number_flux_m3_s=beta_lower_flux,
            beta_rmin_volume_flux_s=beta_flux.beta_volume_flux_out_s,
            beta_rmin_mol_b_flux_mol_m3_s=beta_flux.b_mol_flux_out_mol_m3_s,
            beta_boundary_radius_m=beta_radius,
            beta_boundary_growth_velocity_m_s=0.5
            * (stage0.face_velocity_m_s["beta"][0] + stage1.face_velocity_m_s["beta"][0]),
            rejected_step_count=self.rejected_step_count,
            rejected_step_count_delta=self.rejected_step_count - rejected_before,
            stage0_donor_courant=donor0_courant,
            stage1_donor_courant=donor1_courant,
        )
        self.history.append(diagnostic)
        return diagnostic

    def run_steps(self, steps: int) -> list[ExplicitStepDiagnostics]:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        return [self.advance_one() for _ in range(int(steps))]

    def run_to_time(self, end_time_s: float) -> list[ExplicitStepDiagnostics]:
        if end_time_s < self.time_s:
            raise ValueError("end_time_s precedes the current solver time")
        accepted: list[ExplicitStepDiagnostics] = []
        while self.time_s < end_time_s:
            accepted.append(self.advance_one(maximum_dt_s=end_time_s - self.time_s))
        return accepted

    def state_arrays(self) -> Dict[str, NDArray[np.generic]]:
        arrays: Dict[str, NDArray[np.generic]] = super().state_arrays()
        arrays["rejected_step_count"] = np.asarray([self.rejected_step_count], dtype=np.int64)
        arrays["donor_safety"] = np.asarray([self.donor_safety], dtype=np.float64)
        return arrays

    def save_checkpoint(self, path: str | Path) -> None:
        """Write a deterministic explicit-SSPRK2 checkpoint with provenance."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "solver_version": self.solver_version,
            "source_config_hash": self.config.source_config_hash,
            "temperature_K": self.config.temperature_k,
            "total_b_mol_m3": self.ledger.total_b_mol_m3,
            "validation_contract_hash": self.contract_hash,
            "donor_safety": self.donor_safety,
        }
        arrays = self.state_arrays()
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
        np.savez(output, **arrays)

    @classmethod
    def load_checkpoint(
        cls,
        *,
        config: SolverConfig,
        path: str | Path,
        donor_safety: float | None = None,
    ) -> "ExplicitSSPRK2Solver":
        """Restore an explicit state after strict solver and ledger validation."""

        checkpoint_path = Path(path)
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            try:
                metadata = json.loads(str(archive["metadata_json"].item()))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise SolverStateError("explicit checkpoint metadata is invalid") from error
            if metadata.get("solver_version") != cls.solver_version:
                raise SolverStateError("Checkpoint solver version differs from explicit SSPRK2")
            if metadata.get("source_config_hash") != config.source_config_hash:
                raise SolverStateError("Checkpoint config hash differs from the requested restart config")
            stored_safety = metadata.get("donor_safety")
            if not isinstance(stored_safety, (int, float)):
                raise SolverStateError("Checkpoint donor safety is invalid")
            requested_safety = float(stored_safety) if donor_safety is None else float(donor_safety)
            if not math.isclose(requested_safety, float(stored_safety), rel_tol=0.0, abs_tol=0.0):
                raise SolverStateError("Checkpoint donor safety differs from the requested restart setting")
            solver = cls(config, donor_safety=requested_safety)
            if metadata.get("validation_contract_hash") != solver.contract_hash:
                raise SolverStateError(
                    "Checkpoint validation contract hash differs from the requested restart config"
                )
            if not math.isclose(
                float(metadata.get("total_b_mol_m3", math.nan)),
                solver.ledger.total_b_mol_m3,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise SolverStateError("Checkpoint total inventory differs from configuration")
            restored: Dict[str, NDArray[np.float64]] = {}
            for name in ("g", "beta"):
                key = f"{name}_number_density_per_m4"
                try:
                    values = np.asarray(archive[key], dtype=np.float64)
                except KeyError as error:
                    raise SolverStateError(f"Checkpoint is missing {key}") from error
                if values.shape != solver.populations[name].number_density_per_m4.shape:
                    raise SolverStateError(f"Checkpoint {key} shape differs from the requested grid")
                if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                    raise SolverStateError(f"Checkpoint {key} is not finite and non-negative")
                restored[name] = values.copy()
            try:
                matrix_xb = float(archive["matrix_xb"][0])
                time_s = float(archive["time_s"][0])
                step = int(archive["step"][0])
                rejected = int(archive["rejected_step_count"][0])
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise SolverStateError("Checkpoint explicit state fields are invalid") from error
            if not math.isfinite(matrix_xb) or not 0.0 <= matrix_xb <= 1.0:
                raise SolverStateError("Checkpoint matrix composition is invalid")
            if not math.isfinite(time_s) or time_s < 0.0 or step < 0 or rejected < 0:
                raise SolverStateError("Checkpoint time, step, or rejected count is invalid")
            for name in ("g", "beta"):
                solver.populations[name].number_density_per_m4[:] = restored[name]
            solver.matrix_xb = matrix_xb
            solver.time_s = time_s
            solver.step = step
            solver.rejected_step_count = rejected
            solver.roundoff_zeroed_bin_count = 0
            solver.history = []
            try:
                solver.ledger.snapshot(matrix_xb=solver.matrix_xb, populations=solver.population_list())
            except InventoryError as error:
                raise SolverStateError(f"Checkpoint fails algebraic inventory closure: {error}") from error
            return solver
