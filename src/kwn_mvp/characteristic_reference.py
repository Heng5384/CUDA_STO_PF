"""Self-consistent conservative characteristic reference for beta-only KWN.

This is CR1: a cell-integrated, piecewise-constant conservative remap through
a deterministic autonomous-radius Gauss--Legendre backward trace.  It is deliberately independent from the
implicit Eulerian face solve and from the donor-bound SSPRK2 reference, whose
canonical full-domain timestep is computationally blocked.  The solver is
scoped to smooth, post-nucleation beta populations; it rejects GP or any
nucleation source rather than silently applying a production GP path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .conservative_remap import (
    CharacteristicTrace,
    ConservativeRemapError,
    ConservativeRemapResult,
    conservative_remap_piecewise_constant,
    trace_departure_faces_rk2,
)
from .growth import growth_rate_m_s
from .ledger import InventoryError, InventorySnapshot
from .lower_boundary import (
    boundary_growth_velocity,
    boundary_inventory_diagnostic,
    boundary_radius,
    particle_inventory_at_radius,
)
from .populations import Population
from .population_metrics import cell_moments_from_piecewise_constant_cells
from .solver import KWNSolver, RadiusGridOverflowError, SolverConfig, SolverStateError


REMAP_ORDER = "CR1_piecewise_constant"
TRACE_INTEGRATOR = "AUTONOMOUS_RADIUS_GAUSS_LEGENDRE_2_BACKWARD_V1"
FIXED_POINT_CLOSURE = "PICARD_WITH_EXACT_PERIOD_2_OR_4_CYCLE_BRACKETED_ROOT_V2"
FIXED_POINT_PICARD_MAX_ITERATIONS_DEFAULT = 128
FIXED_POINT_SCALAR_ROOT_MAX_ITERATIONS = 64
FIXED_POINT_PERIODIC_CYCLE_PERIODS = (2, 4)


class CharacteristicReferenceError(SolverStateError):
    """Raised when the independent characteristic reference cannot be closed."""


@dataclass(frozen=True)
class CharacteristicStepDiagnostics:
    """Telemetry for one accepted self-consistent CR1 characteristic step."""

    step: int
    time_s: float
    dt_s: float
    matrix_xb: float
    midpoint_matrix_xb: float
    fixed_point_iterations: int
    fixed_point_picard_iterations: int
    fixed_point_xb_residual: float
    fixed_point_population_residual: float
    fixed_point_cell_measure_residual: float
    fixed_point_convergence_rate: float
    fixed_point_convergence_mode: str
    fixed_point_periodic_cycle_period: int
    fixed_point_bracketed_root_iterations: int
    fixed_point_bracket_initial_width: float
    fixed_point_bracket_final_width: float
    fixed_point_bracket_left_xb: float
    fixed_point_bracket_right_xb: float
    fixed_point_bracket_left_signed_residual: float
    fixed_point_bracket_right_signed_residual: float
    fixed_point_root_trial_xb_residual: float
    fixed_point_root_verification_population_residual: float
    inventory: InventorySnapshot
    rmin_number_loss_m3: float
    rmin_number_flux_m3_s: float
    rmin_beta_volume_loss: float
    rmin_beta_volume_flux_s: float
    rmin_mol_b_loss_mol_m3: float
    rmin_mol_b_flux_mol_m3_s: float
    rmax_number_loss_m3: float
    remap_number_conservation_residual_m3: float
    lower_no_inflow_face_count: int
    upper_no_inflow_face_count: int
    beta_boundary_radius_m: float
    beta_boundary_growth_velocity_m_s: float


@dataclass(frozen=True)
class _FixedPointResult:
    """Private fully closed trial state, retained until final step commit."""

    cell_number_m3: NDArray[np.float64]
    matrix_xb: float
    midpoint_matrix_xb: float
    iterations: int
    picard_iterations: int
    xb_residual: float
    population_residual: float
    cell_measure_residual: float
    convergence_rate: float
    convergence_mode: str
    periodic_cycle_period: int
    bracketed_root_iterations: int
    bracket_initial_width: float
    bracket_final_width: float
    bracket_left_xb: float
    bracket_right_xb: float
    bracket_left_signed_residual: float
    bracket_right_signed_residual: float
    root_trial_xb_residual: float
    root_verification_population_residual: float
    inventory: InventorySnapshot
    trace: CharacteristicTrace
    remap: ConservativeRemapResult


@dataclass(frozen=True)
class _ClosureTrial:
    """One uncommitted evaluation of the unchanged scalar CR1 closure map."""

    x_guess: float
    midpoint_matrix_xb: float
    cell_number_m3: NDArray[np.float64]
    matrix_xb: float
    signed_xb_residual: float
    xb_tolerance: float
    inventory: InventorySnapshot
    trace: CharacteristicTrace
    remap: ConservativeRemapResult


class CharacteristicReferenceSolver(KWNSolver):
    """Conservative semi-Lagrangian CR1 reference for a smooth beta measure.

    A step traces the fixed Eulerian faces backward with a deterministic
    autonomous-radius Gauss--Legendre time-of-flight map, maps
    departure intervals through the old piecewise-constant cumulative
    population, and closes the matrix composition by the existing algebraic
    inventory ledger.  The fixed-point loop repeats that coupled operation
    without mutating accepted state.  There is no independent lower-boundary
    inventory source and no density clamp.
    """

    solver_version = "kwn_conservative_characteristic_remap_cr1_gl2_bracket_v4"

    def __init__(
        self,
        config: SolverConfig,
        *,
        fixed_point_rtol: float = 1.0e-11,
        fixed_point_atol: float = 5.0e-12,
        fixed_point_max_iterations: int = FIXED_POINT_PICARD_MAX_ITERATIONS_DEFAULT,
        under_relaxation: float = 1.0,
    ) -> None:
        super().__init__(config)
        if not math.isfinite(float(fixed_point_rtol)) or float(fixed_point_rtol) <= 0.0:
            raise ValueError("fixed_point_rtol must be finite and positive")
        if not math.isfinite(float(fixed_point_atol)) or float(fixed_point_atol) < 0.0:
            raise ValueError("fixed_point_atol must be finite and non-negative")
        if isinstance(fixed_point_max_iterations, bool) or int(fixed_point_max_iterations) < 2:
            raise ValueError("fixed_point_max_iterations must be an integer of at least two")
        if not math.isfinite(float(under_relaxation)) or not 0.0 < float(under_relaxation) <= 1.0:
            raise ValueError("under_relaxation must lie in (0, 1]")
        self.fixed_point_rtol = float(fixed_point_rtol)
        self.fixed_point_atol = float(fixed_point_atol)
        self.fixed_point_max_iterations = int(fixed_point_max_iterations)
        self.under_relaxation = float(under_relaxation)
        # The autonomous trace resolves a physical stationary radius through
        # binary64 root and local quadrature operations.  At the canonical
        # 3200-face grid, reducing xB below a few e-12 merely moves a CDF face
        # by sub-ulp-scale amounts and produces a round-off-sized cell-L1
        # oscillation.  This floor remains orders of magnitude below every
        # registered 0.25% reference observable and prevents falsely calling
        # a physically closed fixed point unconverged.
        self._population_convergence_rtol = max(1.0e-9, 16.0 * self.fixed_point_rtol)
        self.cumulative_number_dissolution_m3 = 0.0
        self.cumulative_beta_volume_dissolution = 0.0
        self.cumulative_mol_b_returned_mol_m3 = 0.0
        self.history: list[CharacteristicStepDiagnostics] = []
        self._assert_beta_only_non_nucleating_scope()

    def _assert_beta_only_non_nucleating_scope(self) -> None:
        gp = self.population("g")
        beta = self.population("beta")
        if gp.number_density_m3() != 0.0:
            raise CharacteristicReferenceError("characteristic reference is beta-only; GP population must be empty")
        if gp.parameters.nucleation.get("mode", "off") != "off":
            raise CharacteristicReferenceError("characteristic reference does not implement GP nucleation")
        if beta.parameters.nucleation.get("mode", "off") != "off":
            raise CharacteristicReferenceError("characteristic reference does not implement beta nucleation")

    def _beta_cell_numbers(self) -> NDArray[np.float64]:
        beta = self.population("beta")
        return np.asarray(beta.number_density_per_m4 * beta.grid.widths_m, dtype=np.float64).copy()

    def _beta_population_from_numbers(self, cell_number_m3: NDArray[np.float64]) -> Population:
        beta = self.population("beta")
        number = np.asarray(cell_number_m3, dtype=np.float64)
        if number.shape != beta.grid.widths_m.shape:
            raise CharacteristicReferenceError("trial beta cell number shape differs from the frozen grid")
        if not np.all(np.isfinite(number)) or np.any(number < 0.0):
            raise CharacteristicReferenceError("trial beta cell number is not finite and non-negative")
        return Population(
            beta.parameters,
            beta.grid,
            number / beta.grid.widths_m,
            production_quadrature=beta.production_quadrature,
        )

    def _trial_populations(self, beta_cell_number_m3: NDArray[np.float64]) -> list[Population]:
        return [self.population("g").copy(), self._beta_population_from_numbers(beta_cell_number_m3)]

    def _recover_matrix_xb(self, beta_cell_number_m3: NDArray[np.float64]) -> float:
        try:
            matrix_xb = float(self.ledger.recover_matrix_xb(self._trial_populations(beta_cell_number_m3)))
        except InventoryError as error:
            raise CharacteristicReferenceError(f"characteristic trial inventory closure failed: {error}") from error
        if not math.isfinite(matrix_xb) or not 0.0 <= matrix_xb <= 1.0:
            raise CharacteristicReferenceError("characteristic trial recovered an invalid matrix composition")
        return matrix_xb

    def _inventory_snapshot(
        self, beta_cell_number_m3: NDArray[np.float64], matrix_xb: float
    ) -> InventorySnapshot:
        try:
            return self.ledger.snapshot(
                matrix_xb=matrix_xb,
                populations=self._trial_populations(beta_cell_number_m3),
                beta_resolved_fraction=1.0,
            )
        except InventoryError as error:
            raise CharacteristicReferenceError(f"characteristic inventory audit failed: {error}") from error

    def _velocity_at_radii(
        self, radii_m: NDArray[np.float64], matrix_xb: float
    ) -> NDArray[np.float64]:
        """Evaluate the shared beta growth law at arbitrary trace radii."""

        beta = self.population("beta")
        radii = np.asarray(radii_m, dtype=np.float64)
        if radii.ndim != 1 or not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
            raise CharacteristicReferenceError("characteristic trace radii must be finite and positive")
        equilibrium = self.equilibrium_adapter.equilibrium_xb(radii, beta.parameters)
        velocity = growth_rate_m_s(
            radii_m=radii,
            matrix_xb=float(matrix_xb),
            equilibrium_xb=equilibrium,
            parameters=beta.parameters,
        )
        result = np.asarray(velocity, dtype=np.float64)
        if result.shape != radii.shape or not np.all(np.isfinite(result)):
            raise CharacteristicReferenceError("shared beta growth kernel returned an invalid characteristic velocity")
        # The frozen physical lower edge is a separately audited binary64
        # operator.  Preserve that exact helper value even though its formula
        # is algebraically the same arbitrary-radius growth kernel.
        rmin = boundary_radius(beta.grid)
        lower_mask = radii == rmin
        if np.any(lower_mask):
            result = result.copy()
            result[lower_mask] = boundary_growth_velocity(
                radius_m=rmin,
                matrix_xb=float(matrix_xb),
                parameters=beta.parameters,
                equilibrium_adapter=self.equilibrium_adapter,
            )
        return result

    def _trace_faces(self, *, dt_s: float, midpoint_matrix_xb: float) -> CharacteristicTrace:
        beta = self.population("beta")
        try:
            return trace_departure_faces_rk2(
                beta.grid.edges_m,
                dt_s=dt_s,
                velocity_m_s=lambda radii: self._velocity_at_radii(radii, midpoint_matrix_xb),
                lower_radius_m=boundary_radius(beta.grid),
                upper_radius_m=float(beta.grid.edges_m[-1]),
            )
        except ConservativeRemapError as error:
            raise CharacteristicReferenceError(f"characteristic trace failed: {error}") from error

    def _remap_once(
        self,
        old_cell_number_m3: NDArray[np.float64],
        *,
        dt_s: float,
        midpoint_matrix_xb: float,
    ) -> tuple[CharacteristicTrace, ConservativeRemapResult]:
        beta = self.population("beta")
        trace = self._trace_faces(dt_s=dt_s, midpoint_matrix_xb=midpoint_matrix_xb)
        try:
            remap = conservative_remap_piecewise_constant(
                beta.grid.edges_m, old_cell_number_m3, trace.departure_faces_m
            )
        except ConservativeRemapError as error:
            raise CharacteristicReferenceError(f"conservative characteristic remap failed: {error}") from error
        relative_upper_loss = remap.upper_number_loss_m3 / max(remap.old_number_m3, 1.0e-300)
        if relative_upper_loss > self.config.rmax_outflow_relative_tolerance:
            raise RadiusGridOverflowError(
                f"beta would lose {relative_upper_loss:.3e} of its number density through Rmax; "
                "expand the frozen grid rather than discard material"
            )
        return trace, remap

    @staticmethod
    def _cell_measure_relative_residual(
        candidate: NDArray[np.float64], previous: NDArray[np.float64] | None
    ) -> float:
        if previous is None:
            return math.inf
        denominator = max(
            float(np.sum(np.abs(candidate), dtype=np.float64)),
            float(np.sum(np.abs(previous), dtype=np.float64)),
            1.0e-300,
        )
        return float(np.sum(np.abs(candidate - previous), dtype=np.float64) / denominator)

    def _population_observable_residual(
        self, candidate: NDArray[np.float64], previous: NDArray[np.float64] | None
    ) -> float:
        """Compare the physical M0--M3 state, not a CDF-cell round-off norm."""

        if previous is None:
            return math.inf
        edges = self.population("beta").grid.edges_m
        candidate_moments = np.sum(
            cell_moments_from_piecewise_constant_cells(edges, candidate), axis=1, dtype=np.float64
        )
        previous_moments = np.sum(
            cell_moments_from_piecewise_constant_cells(edges, previous), axis=1, dtype=np.float64
        )
        denominator = np.maximum(
            np.maximum(np.abs(candidate_moments), np.abs(previous_moments)), 1.0e-300
        )
        return float(np.max(np.abs(candidate_moments - previous_moments) / denominator))

    def _evaluate_closure_trial(
        self,
        *,
        old_cell_number_m3: NDArray[np.float64],
        dt_s: float,
        x_start: float,
        x_guess: float,
    ) -> _ClosureTrial:
        """Evaluate the documented CR1 scalar closure without state mutation."""

        midpoint = 0.5 * (x_start + x_guess)
        trace, remap = self._remap_once(
            old_cell_number_m3, dt_s=dt_s, midpoint_matrix_xb=midpoint
        )
        candidate = remap.cell_number_m3
        matrix_candidate = self._recover_matrix_xb(candidate)
        inventory = self._inventory_snapshot(candidate, matrix_candidate)
        xb_tolerance = self.fixed_point_atol + self.fixed_point_rtol * max(
            abs(matrix_candidate), abs(x_guess)
        )
        return _ClosureTrial(
            x_guess=float(x_guess),
            midpoint_matrix_xb=midpoint,
            cell_number_m3=candidate,
            matrix_xb=matrix_candidate,
            signed_xb_residual=matrix_candidate - x_guess,
            xb_tolerance=xb_tolerance,
            inventory=inventory,
            trace=trace,
            remap=remap,
        )

    def _fixed_point_result(
        self,
        trial: _ClosureTrial,
        *,
        previous_population: NDArray[np.float64] | None,
        iterations: int,
        picard_iterations: int,
        convergence_rate: float,
        convergence_mode: str,
        periodic_cycle_period: int = 0,
        bracketed_root_iterations: int = 0,
        bracket_initial_width: float = 0.0,
        bracket_final_width: float = 0.0,
        bracket_left_xb: float = 0.0,
        bracket_right_xb: float = 0.0,
        bracket_left_signed_residual: float = 0.0,
        bracket_right_signed_residual: float = 0.0,
        root_trial_xb_residual: float = 0.0,
        root_verification_population_residual: float = 0.0,
    ) -> _FixedPointResult:
        """Package one already-audited trial without altering its remap state."""

        return _FixedPointResult(
            cell_number_m3=trial.cell_number_m3,
            matrix_xb=trial.matrix_xb,
            midpoint_matrix_xb=trial.midpoint_matrix_xb,
            iterations=iterations,
            picard_iterations=picard_iterations,
            xb_residual=abs(trial.signed_xb_residual),
            population_residual=self._population_observable_residual(
                trial.cell_number_m3, previous_population
            ),
            cell_measure_residual=self._cell_measure_relative_residual(
                trial.cell_number_m3, previous_population
            ),
            convergence_rate=convergence_rate,
            convergence_mode=convergence_mode,
            periodic_cycle_period=periodic_cycle_period,
            bracketed_root_iterations=bracketed_root_iterations,
            bracket_initial_width=bracket_initial_width,
            bracket_final_width=bracket_final_width,
            bracket_left_xb=bracket_left_xb,
            bracket_right_xb=bracket_right_xb,
            bracket_left_signed_residual=bracket_left_signed_residual,
            bracket_right_signed_residual=bracket_right_signed_residual,
            root_trial_xb_residual=root_trial_xb_residual,
            root_verification_population_residual=root_verification_population_residual,
            inventory=trial.inventory,
            trace=trial.trace,
            remap=trial.remap,
        )

    def _solve_exact_periodic_cycle_bracket(
        self,
        *,
        old_cell_number_m3: NDArray[np.float64],
        dt_s: float,
        x_start: float,
        first_trial: _ClosureTrial,
        second_trial: _ClosureTrial,
        picard_iterations: int,
        cycle_period: int,
    ) -> _FixedPointResult:
        """Close an exact raw-Picard periodic-cycle bracket on ``T(x)-x``.

        An exact recurrence of a supported raw Picard period is not itself a
        converged state and does not certify a scalar root.  It merely supplies
        a narrowly scoped bracket for the original CR1 equation ``T(x)-x=0``.
        This routine accepts only a full remap/inventory trial that satisfies
        the ordinary direct residual and physical-population criteria; bracket
        width, the cycle phase, and a cycle average are never acceptance
        conditions.
        """

        if cycle_period not in FIXED_POINT_PERIODIC_CYCLE_PERIODS:
            raise CharacteristicReferenceError(
                f"unsupported exact raw-Picard cycle period {cycle_period}"
            )
        if first_trial.signed_xb_residual * second_trial.signed_xb_residual >= 0.0:
            raise CharacteristicReferenceError(
                f"exact fixed-point period-{cycle_period} cycle did not provide an oppositely signed scalar bracket"
            )
        initial_left, initial_right = sorted(
            (first_trial, second_trial), key=lambda item: item.x_guess
        )
        left, right = initial_left, initial_right
        initial_width = right.x_guess - left.x_guess
        if not initial_width > 0.0:
            raise CharacteristicReferenceError(
                f"exact fixed-point period-{cycle_period} cycle has a degenerate scalar bracket"
            )

        evaluations = picard_iterations
        attempted_root_iterations = 0
        stop_reason = "scalar-root iteration cap reached"
        last_error = "no bracketed scalar-root trial was attempted"
        for root_iteration in range(1, FIXED_POINT_SCALAR_ROOT_MAX_ITERATIONS + 1):
            attempted_root_iterations = root_iteration
            midpoint = 0.5 * (left.x_guess + right.x_guess)
            if midpoint == left.x_guess or midpoint == right.x_guess:
                stop_reason = "binary64 midpoint coalesced with a bracket endpoint"
                break
            trial = self._evaluate_closure_trial(
                old_cell_number_m3=old_cell_number_m3,
                dt_s=dt_s,
                x_start=x_start,
                x_guess=midpoint,
            )
            evaluations += 1
            final_width = right.x_guess - left.x_guess
            if abs(trial.signed_xb_residual) <= trial.xb_tolerance:
                # Verify at T(x), as a normal direct iteration would.  This
                # makes population convergence part of the same acceptance
                # contract and avoids accepting a merely narrow discontinuity.
                verification = self._evaluate_closure_trial(
                    old_cell_number_m3=old_cell_number_m3,
                    dt_s=dt_s,
                    x_start=x_start,
                    x_guess=trial.matrix_xb,
                )
                evaluations += 1
                population_residual = self._population_observable_residual(
                    verification.cell_number_m3, trial.cell_number_m3
                )
                if (
                    abs(verification.signed_xb_residual) <= verification.xb_tolerance
                    and population_residual <= self._population_convergence_rtol
                ):
                    convergence_rate = (
                        math.nan
                        if trial.signed_xb_residual == 0.0
                        else abs(verification.signed_xb_residual / trial.signed_xb_residual)
                    )
                    return self._fixed_point_result(
                        verification,
                        previous_population=trial.cell_number_m3,
                        iterations=evaluations,
                        picard_iterations=picard_iterations,
                        convergence_rate=convergence_rate,
                        convergence_mode="BRACKETED_SCALAR_ROOT",
                        periodic_cycle_period=cycle_period,
                        bracketed_root_iterations=root_iteration,
                        bracket_initial_width=initial_width,
                        bracket_final_width=final_width,
                        bracket_left_xb=initial_left.x_guess,
                        bracket_right_xb=initial_right.x_guess,
                        bracket_left_signed_residual=initial_left.signed_xb_residual,
                        bracket_right_signed_residual=initial_right.signed_xb_residual,
                        root_trial_xb_residual=abs(trial.signed_xb_residual),
                        root_verification_population_residual=population_residual,
                    )
                last_error = (
                    f"root_iteration={root_iteration}, scalar_residual={abs(trial.signed_xb_residual):.3e}, "
                    f"verification_residual={abs(verification.signed_xb_residual):.3e}, "
                    f"verification_tolerance={verification.xb_tolerance:.3e}, "
                    f"population_residual={population_residual:.3e}"
                )
            else:
                last_error = (
                    f"root_iteration={root_iteration}, scalar_residual={abs(trial.signed_xb_residual):.3e}, "
                    f"scalar_tolerance={trial.xb_tolerance:.3e}"
                )

            if trial.signed_xb_residual == 0.0:
                stop_reason = "exact scalar residual failed map-verification population closure"
                break
            if trial.signed_xb_residual * left.signed_xb_residual > 0.0:
                left = trial
            else:
                right = trial

        raise CharacteristicReferenceError(
            f"exact fixed-point period-{cycle_period} cycle bracket did not close the original scalar equation "
            f"after {attempted_root_iterations} bisection iterations ({stop_reason}; "
            f"initial_width={initial_width:.3e}, final_width={right.x_guess - left.x_guess:.3e}, {last_error})"
        )

    @staticmethod
    def _exact_raw_periodic_cycle(
        recent_trials: list[_ClosureTrial], *, period: int
    ) -> tuple[_ClosureTrial, ...] | None:
        """Return a full bitwise raw-Picard cycle only for a supported period.

        Both adjacent period-length blocks must repeat exactly in scalar guess,
        closed matrix value, and cell-integrated population.  This deliberately
        does not identify approximate oscillations or manufacture a bracket
        from unrelated historical iterates.
        """

        if period not in FIXED_POINT_PERIODIC_CYCLE_PERIODS:
            raise ValueError(f"unsupported exact raw-Picard cycle period {period}")
        if len(recent_trials) < 2 * period:
            return None
        prior_cycle = recent_trials[-2 * period : -period]
        current_cycle = recent_trials[-period:]
        if all(
            current.x_guess == prior.x_guess
            and current.matrix_xb == prior.matrix_xb
            and np.array_equal(current.cell_number_m3, prior.cell_number_m3)
            for current, prior in zip(current_cycle, prior_cycle)
        ):
            return tuple(current_cycle)
        return None

    @staticmethod
    def _cycle_adjacent_scalar_bracket(
        cycle: tuple[_ClosureTrial, ...]
    ) -> tuple[_ClosureTrial, _ClosureTrial]:
        """Select a deterministic adjacent opposite-sign pair from one cycle."""

        period = len(cycle)
        if period not in FIXED_POINT_PERIODIC_CYCLE_PERIODS:
            raise ValueError(f"unsupported exact raw-Picard cycle period {period}")
        for first, second in zip(cycle, cycle[1:] + cycle[:1]):
            if (
                0.0 <= first.x_guess <= 1.0
                and 0.0 <= second.x_guess <= 1.0
                and first.signed_xb_residual * second.signed_xb_residual < 0.0
            ):
                return first, second
        raise CharacteristicReferenceError(
            f"exact fixed-point period-{period} cycle did not provide an adjacent oppositely signed scalar bracket"
        )

    def _solve_fixed_point(self, *, old_cell_number_m3: NDArray[np.float64], dt_s: float) -> _FixedPointResult:
        x_start = float(self.matrix_xb)
        x_guess = x_start
        previous_trial: _ClosureTrial | None = None
        two_back_trial: _ClosureTrial | None = None
        previous_xb_residual: float | None = None
        recent_trials: list[_ClosureTrial] = []
        last_error = "no fixed-point iteration was attempted"
        for iteration in range(1, self.fixed_point_max_iterations + 1):
            trial = self._evaluate_closure_trial(
                old_cell_number_m3=old_cell_number_m3,
                dt_s=dt_s,
                x_start=x_start,
                x_guess=x_guess,
            )
            previous_population = None if previous_trial is None else previous_trial.cell_number_m3
            cell_measure_residual = self._cell_measure_relative_residual(
                trial.cell_number_m3, previous_population
            )
            population_residual = self._population_observable_residual(
                trial.cell_number_m3, previous_population
            )
            convergence_rate = (
                math.nan
                if previous_xb_residual is None or previous_xb_residual == 0.0
                else abs(trial.signed_xb_residual) / previous_xb_residual
            )
            unchanged = (
                np.array_equal(trial.cell_number_m3, old_cell_number_m3)
                and trial.matrix_xb == x_start
            )
            direct_converged = (
                abs(trial.signed_xb_residual) <= trial.xb_tolerance
                and (
                    unchanged
                    or (iteration >= 2 and population_residual <= self._population_convergence_rtol)
                )
            )
            exact_two_cycle = (
                self.under_relaxation == 1.0
                and iteration >= 3
                and previous_trial is not None
                and two_back_trial is not None
                and trial.x_guess == two_back_trial.x_guess
                and np.array_equal(trial.cell_number_m3, two_back_trial.cell_number_m3)
                and trial.matrix_xb == two_back_trial.matrix_xb
                and trial.matrix_xb != previous_trial.matrix_xb
                and 0.0 <= previous_trial.x_guess <= 1.0
                and 0.0 <= trial.x_guess <= 1.0
                and previous_trial.signed_xb_residual * trial.signed_xb_residual < 0.0
            )
            recent_trials.append(trial)
            if len(recent_trials) > 2 * max(FIXED_POINT_PERIODIC_CYCLE_PERIODS):
                del recent_trials[0]
            exact_four_cycle = (
                self._exact_raw_periodic_cycle(recent_trials, period=4)
                if self.under_relaxation == 1.0
                else None
            )
            if direct_converged:
                result = self._fixed_point_result(
                    trial,
                    previous_population=previous_population,
                    iterations=iteration,
                    picard_iterations=iteration,
                    convergence_rate=convergence_rate,
                    convergence_mode="DIRECT",
                )
                if unchanged:
                    return replace(
                        result,
                        population_residual=0.0,
                        cell_measure_residual=0.0,
                    )
                return result
            if exact_two_cycle:
                return self._solve_exact_periodic_cycle_bracket(
                    old_cell_number_m3=old_cell_number_m3,
                    dt_s=dt_s,
                    x_start=x_start,
                    first_trial=previous_trial,
                    second_trial=trial,
                    picard_iterations=iteration,
                    cycle_period=2,
                )
            if exact_four_cycle is not None:
                first_trial, second_trial = self._cycle_adjacent_scalar_bracket(exact_four_cycle)
                return self._solve_exact_periodic_cycle_bracket(
                    old_cell_number_m3=old_cell_number_m3,
                    dt_s=dt_s,
                    x_start=x_start,
                    first_trial=first_trial,
                    second_trial=second_trial,
                    picard_iterations=iteration,
                    cycle_period=4,
                )
            last_error = (
                f"iteration={iteration}, xb_residual={abs(trial.signed_xb_residual):.3e}, "
                f"xb_tolerance={trial.xb_tolerance:.3e}, population_residual={population_residual:.3e}, "
                f"cell_measure_residual={cell_measure_residual:.3e}, "
                f"exact_two_cycle_trigger={exact_two_cycle}, "
                f"exact_four_cycle_trigger={exact_four_cycle is not None}"
            )
            two_back_trial = previous_trial
            previous_trial = trial
            previous_xb_residual = abs(trial.signed_xb_residual)
            x_guess = self.under_relaxation * trial.matrix_xb + (1.0 - self.under_relaxation) * x_guess
        raise CharacteristicReferenceError(
            f"characteristic matrix/population fixed point did not converge after "
            f"{self.fixed_point_max_iterations} iterations ({last_error})"
        )

    def _choose_step_dt(self, maximum_dt_s: float | None) -> float:
        maximum = float(self.config.max_dt_s if maximum_dt_s is None else maximum_dt_s)
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum_dt_s must be finite and positive when supplied")
        return min(float(self.config.max_dt_s), maximum)

    def advance_one(self, maximum_dt_s: float | None = None) -> CharacteristicStepDiagnostics:
        """Advance exactly one fully coupled CR1 step without any CFL limiter."""

        self._assert_beta_only_non_nucleating_scope()
        dt_s = self._choose_step_dt(maximum_dt_s)
        old_cell_number = self._beta_cell_numbers()
        beta = self.population("beta")
        identity_trace = self._trace_faces(dt_s=dt_s, midpoint_matrix_xb=self.matrix_xb)
        if np.array_equal(identity_trace.departure_faces_m, beta.grid.edges_m):
            # Preserve a true zero-mobility state bit-for-bit.  Reconstructing
            # n -> N -> n would be mathematically identical yet can perturb a
            # binary64 density through division/multiplication round-off.
            inventory = self.ledger.snapshot(
                matrix_xb=self.matrix_xb,
                populations=self.population_list(),
                beta_resolved_fraction=1.0,
            )
            boundary = particle_inventory_at_radius(
                boundary_radius(beta.grid),
                x_b=beta.parameters.x_b,
                molar_volume_m3_mol=beta.parameters.molar_volume_m3_mol,
            )
            boundary_velocity = boundary_growth_velocity(
                radius_m=boundary.radius_m,
                matrix_xb=self.matrix_xb,
                parameters=beta.parameters,
                equilibrium_adapter=self.equilibrium_adapter,
            )
            self.time_s += dt_s
            self.step += 1
            diagnostic = CharacteristicStepDiagnostics(
                step=self.step,
                time_s=self.time_s,
                dt_s=dt_s,
                matrix_xb=self.matrix_xb,
                midpoint_matrix_xb=self.matrix_xb,
                fixed_point_iterations=1,
                fixed_point_picard_iterations=1,
                fixed_point_xb_residual=0.0,
                fixed_point_population_residual=0.0,
                fixed_point_cell_measure_residual=0.0,
                fixed_point_convergence_rate=0.0,
                fixed_point_convergence_mode="IDENTITY",
                fixed_point_periodic_cycle_period=0,
                fixed_point_bracketed_root_iterations=0,
                fixed_point_bracket_initial_width=0.0,
                fixed_point_bracket_final_width=0.0,
                fixed_point_bracket_left_xb=0.0,
                fixed_point_bracket_right_xb=0.0,
                fixed_point_bracket_left_signed_residual=0.0,
                fixed_point_bracket_right_signed_residual=0.0,
                fixed_point_root_trial_xb_residual=0.0,
                fixed_point_root_verification_population_residual=0.0,
                inventory=inventory,
                rmin_number_loss_m3=0.0,
                rmin_number_flux_m3_s=0.0,
                rmin_beta_volume_loss=0.0,
                rmin_beta_volume_flux_s=0.0,
                rmin_mol_b_loss_mol_m3=0.0,
                rmin_mol_b_flux_mol_m3_s=0.0,
                rmax_number_loss_m3=0.0,
                remap_number_conservation_residual_m3=0.0,
                lower_no_inflow_face_count=identity_trace.lower_no_inflow_face_count,
                upper_no_inflow_face_count=identity_trace.upper_no_inflow_face_count,
                beta_boundary_radius_m=boundary.radius_m,
                beta_boundary_growth_velocity_m_s=boundary_velocity,
            )
            self.history.append(diagnostic)
            return diagnostic
        result = self._solve_fixed_point(old_cell_number_m3=old_cell_number, dt_s=dt_s)

        beta.number_density_per_m4[:] = result.cell_number_m3 / beta.grid.widths_m
        self.matrix_xb = result.matrix_xb
        self.time_s += dt_s
        self.step += 1

        lower_number_loss = result.remap.lower_number_loss_m3
        boundary = particle_inventory_at_radius(
            boundary_radius(beta.grid),
            x_b=beta.parameters.x_b,
            molar_volume_m3_mol=beta.parameters.molar_volume_m3_mol,
        )
        boundary_flux = boundary_inventory_diagnostic(lower_number_loss / dt_s, boundary)
        beta_volume_loss = lower_number_loss * boundary.volume_m3
        mol_b_loss = lower_number_loss * boundary.b_moles_mol
        self.cumulative_number_dissolution_m3 += lower_number_loss
        self.cumulative_beta_volume_dissolution += beta_volume_loss
        self.cumulative_mol_b_returned_mol_m3 += mol_b_loss
        boundary_velocity = boundary_growth_velocity(
            radius_m=boundary_radius(beta.grid),
            matrix_xb=result.midpoint_matrix_xb,
            parameters=beta.parameters,
            equilibrium_adapter=self.equilibrium_adapter,
        )
        diagnostic = CharacteristicStepDiagnostics(
            step=self.step,
            time_s=self.time_s,
            dt_s=dt_s,
            matrix_xb=self.matrix_xb,
            midpoint_matrix_xb=result.midpoint_matrix_xb,
            fixed_point_iterations=result.iterations,
            fixed_point_picard_iterations=result.picard_iterations,
            fixed_point_xb_residual=result.xb_residual,
            fixed_point_population_residual=result.population_residual,
            fixed_point_cell_measure_residual=result.cell_measure_residual,
            fixed_point_convergence_rate=result.convergence_rate,
            fixed_point_convergence_mode=result.convergence_mode,
            fixed_point_periodic_cycle_period=result.periodic_cycle_period,
            fixed_point_bracketed_root_iterations=result.bracketed_root_iterations,
            fixed_point_bracket_initial_width=result.bracket_initial_width,
            fixed_point_bracket_final_width=result.bracket_final_width,
            fixed_point_bracket_left_xb=result.bracket_left_xb,
            fixed_point_bracket_right_xb=result.bracket_right_xb,
            fixed_point_bracket_left_signed_residual=result.bracket_left_signed_residual,
            fixed_point_bracket_right_signed_residual=result.bracket_right_signed_residual,
            fixed_point_root_trial_xb_residual=result.root_trial_xb_residual,
            fixed_point_root_verification_population_residual=result.root_verification_population_residual,
            inventory=result.inventory,
            rmin_number_loss_m3=lower_number_loss,
            rmin_number_flux_m3_s=boundary_flux.number_flux_out_m3_s,
            rmin_beta_volume_loss=beta_volume_loss,
            rmin_beta_volume_flux_s=boundary_flux.beta_volume_flux_out_s,
            rmin_mol_b_loss_mol_m3=mol_b_loss,
            rmin_mol_b_flux_mol_m3_s=boundary_flux.b_mol_flux_out_mol_m3_s,
            rmax_number_loss_m3=result.remap.upper_number_loss_m3,
            remap_number_conservation_residual_m3=result.remap.conservation_residual_m3,
            lower_no_inflow_face_count=result.trace.lower_no_inflow_face_count,
            upper_no_inflow_face_count=result.trace.upper_no_inflow_face_count,
            beta_boundary_radius_m=boundary.radius_m,
            beta_boundary_growth_velocity_m_s=boundary_velocity,
        )
        self.history.append(diagnostic)
        return diagnostic

    def run_to_time(
        self, end_time_s: float, *, maximum_step_s: float | None = None
    ) -> list[CharacteristicStepDiagnostics]:
        """Advance to an exact physical output time using caller-selected CR1 dt."""

        end = float(end_time_s)
        if not math.isfinite(end) or end < self.time_s:
            raise ValueError("end_time_s must be finite and not precede the current state")
        limit = None if maximum_step_s is None else float(maximum_step_s)
        if limit is not None and (not math.isfinite(limit) or limit <= 0.0):
            raise ValueError("maximum_step_s must be finite and positive when supplied")
        accepted: list[CharacteristicStepDiagnostics] = []
        while self.time_s < end:
            remaining = end - self.time_s
            maximum = remaining if limit is None else min(remaining, limit)
            accepted.append(self.advance_one(maximum_dt_s=maximum))
        return accepted

    def state_arrays(self) -> dict[str, NDArray[np.generic]]:
        """Return deterministic state plus all algebraic boundary diagnostics."""

        arrays = super().state_arrays()
        arrays.update(
            {
                "cumulative_number_dissolution_m3": np.asarray(
                    [self.cumulative_number_dissolution_m3], dtype=np.float64
                ),
                "cumulative_beta_volume_dissolution": np.asarray(
                    [self.cumulative_beta_volume_dissolution], dtype=np.float64
                ),
                "cumulative_mol_b_returned_mol_m3": np.asarray(
                    [self.cumulative_mol_b_returned_mol_m3], dtype=np.float64
                ),
            }
        )
        return arrays

    def save_checkpoint(self, path: str | Path) -> None:
        """Write a restartable CR1 state with configuration and method provenance."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "solver_version": self.solver_version,
            "source_config_hash": self.config.source_config_hash,
            "temperature_K": self.config.temperature_k,
            "total_b_mol_m3": self.ledger.total_b_mol_m3,
            "validation_contract_hash": self.contract_hash,
            "fixed_point_rtol": self.fixed_point_rtol,
            "fixed_point_atol": self.fixed_point_atol,
            "fixed_point_max_iterations": self.fixed_point_max_iterations,
            "under_relaxation": self.under_relaxation,
            "fixed_point_closure": FIXED_POINT_CLOSURE,
            "fixed_point_scalar_root_max_iterations": FIXED_POINT_SCALAR_ROOT_MAX_ITERATIONS,
            "fixed_point_scalar_root_trigger": "EXACT_BITWISE_RAW_PICARD_PERIOD_2_OR_4_CYCLE_ONLY",
            "fixed_point_scalar_root_cycle_periods": list(FIXED_POINT_PERIODIC_CYCLE_PERIODS),
            "fixed_point_scalar_root_requires_original_xb_tolerance": True,
            "fixed_point_scalar_root_requires_map_verification_population_check": True,
            "remap_order": REMAP_ORDER,
            "trace_integrator": TRACE_INTEGRATOR,
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
        fixed_point_rtol: float | None = None,
        fixed_point_atol: float | None = None,
        fixed_point_max_iterations: int | None = None,
        under_relaxation: float | None = None,
    ) -> "CharacteristicReferenceSolver":
        """Restore a CR1 state after exact configuration/method verification."""

        checkpoint = Path(path)
        with np.load(checkpoint, allow_pickle=False) as archive:
            try:
                metadata = json.loads(str(archive["metadata_json"].item()))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise CharacteristicReferenceError("characteristic checkpoint metadata is invalid") from error
            if metadata.get("solver_version") != cls.solver_version:
                raise CharacteristicReferenceError("checkpoint solver version differs from CR1")
            if metadata.get("remap_order") != REMAP_ORDER:
                raise CharacteristicReferenceError("checkpoint remap order differs from CR1")
            if metadata.get("trace_integrator") != TRACE_INTEGRATOR:
                raise CharacteristicReferenceError("checkpoint trace integrator differs from CR1")
            if metadata.get("fixed_point_closure") != FIXED_POINT_CLOSURE:
                raise CharacteristicReferenceError("checkpoint fixed-point closure differs from CR1")
            if int(metadata.get("fixed_point_scalar_root_max_iterations", -1)) != FIXED_POINT_SCALAR_ROOT_MAX_ITERATIONS:
                raise CharacteristicReferenceError("checkpoint fixed-point scalar-root iteration cap differs from CR1")
            if metadata.get("fixed_point_scalar_root_trigger") != "EXACT_BITWISE_RAW_PICARD_PERIOD_2_OR_4_CYCLE_ONLY":
                raise CharacteristicReferenceError("checkpoint fixed-point scalar-root trigger differs from CR1")
            if metadata.get("fixed_point_scalar_root_cycle_periods") != list(FIXED_POINT_PERIODIC_CYCLE_PERIODS):
                raise CharacteristicReferenceError("checkpoint fixed-point scalar-root cycle periods differ from CR1")
            if metadata.get("fixed_point_scalar_root_requires_original_xb_tolerance") is not True:
                raise CharacteristicReferenceError("checkpoint scalar-root tolerance contract differs from CR1")
            if metadata.get("fixed_point_scalar_root_requires_map_verification_population_check") is not True:
                raise CharacteristicReferenceError("checkpoint scalar-root population contract differs from CR1")
            if metadata.get("source_config_hash") != config.source_config_hash:
                raise CharacteristicReferenceError("checkpoint config hash differs from the requested config")
            options: dict[str, Any] = {
                "fixed_point_rtol": metadata.get("fixed_point_rtol"),
                "fixed_point_atol": metadata.get("fixed_point_atol"),
                "fixed_point_max_iterations": metadata.get("fixed_point_max_iterations"),
                "under_relaxation": metadata.get("under_relaxation"),
            }
            requested = {
                "fixed_point_rtol": fixed_point_rtol,
                "fixed_point_atol": fixed_point_atol,
                "fixed_point_max_iterations": fixed_point_max_iterations,
                "under_relaxation": under_relaxation,
            }
            for key, value in requested.items():
                if value is not None:
                    if key == "fixed_point_max_iterations":
                        if int(value) != int(options[key]):
                            raise CharacteristicReferenceError(f"checkpoint {key} differs from requested value")
                    elif float(value) != float(options[key]):
                        raise CharacteristicReferenceError(f"checkpoint {key} differs from requested value")
                    options[key] = value
            solver = cls(config, **options)
            if metadata.get("validation_contract_hash") != solver.contract_hash:
                raise CharacteristicReferenceError("checkpoint validation contract hash differs from config")
            if not math.isclose(
                float(metadata.get("total_b_mol_m3", math.nan)),
                solver.ledger.total_b_mol_m3,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise CharacteristicReferenceError("checkpoint total inventory differs from config")
            try:
                g_density = np.asarray(archive["g_number_density_per_m4"], dtype=np.float64)
                beta_density = np.asarray(archive["beta_number_density_per_m4"], dtype=np.float64)
                matrix_xb = float(archive["matrix_xb"][0])
                time_s = float(archive["time_s"][0])
                step = int(archive["step"][0])
                cumulative_number = float(archive["cumulative_number_dissolution_m3"][0])
                cumulative_volume = float(archive["cumulative_beta_volume_dissolution"][0])
                cumulative_mol_b = float(archive["cumulative_mol_b_returned_mol_m3"][0])
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise CharacteristicReferenceError("characteristic checkpoint is missing state arrays") from error
            if g_density.shape != solver.population("g").number_density_per_m4.shape or beta_density.shape != solver.population("beta").number_density_per_m4.shape:
                raise CharacteristicReferenceError("checkpoint density shape differs from frozen grid")
            if not np.all(np.isfinite(g_density)) or not np.all(np.isfinite(beta_density)) or np.any(g_density < 0.0) or np.any(beta_density < 0.0):
                raise CharacteristicReferenceError("checkpoint density state is invalid")
            if not math.isfinite(matrix_xb) or not 0.0 <= matrix_xb <= 1.0:
                raise CharacteristicReferenceError("checkpoint matrix composition is invalid")
            if not math.isfinite(time_s) or time_s < 0.0 or step < 0:
                raise CharacteristicReferenceError("checkpoint time/step is invalid")
            if any(not math.isfinite(value) or value < 0.0 for value in (cumulative_number, cumulative_volume, cumulative_mol_b)):
                raise CharacteristicReferenceError("checkpoint cumulative lower-boundary diagnostics are invalid")
            solver.population("g").number_density_per_m4[:] = g_density
            solver.population("beta").number_density_per_m4[:] = beta_density
            solver.matrix_xb = matrix_xb
            solver.time_s = time_s
            solver.step = step
            solver.cumulative_number_dissolution_m3 = cumulative_number
            solver.cumulative_beta_volume_dissolution = cumulative_volume
            solver.cumulative_mol_b_returned_mol_m3 = cumulative_mol_b
            solver._assert_beta_only_non_nucleating_scope()
            solver.ledger.snapshot(matrix_xb=solver.matrix_xb, populations=solver.population_list())
            return solver
