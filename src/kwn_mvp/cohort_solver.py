"""Event-aware discrete-cohort beta-only KWN comparator.

This module deliberately represents finite populations as material cohorts,
not as a radius-grid density.  It reuses the validation-contract equilibrium
adapter and the production diffusion-controlled growth law while obtaining the
matrix composition algebraically from the total B inventory after every ODE
evaluation.  Consequently there is no radius-bin projection, numerical
diffusion, separately integrated matrix source, or clamp-based inventory fix.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from .growth import growth_rate_m_s
from .populations import PopulationParameters
from .solver import KWNSolver
from .thermo_adapter import DiluteEquilibriumAdapter, ValidationContractEquilibriumAdapter


class CohortSolverError(RuntimeError):
    """Raised when a cohort state cannot satisfy the frozen physical contract."""


EquilibriumAdapter = Union[DiluteEquilibriumAdapter, ValidationContractEquilibriumAdapter]


def sphere_volume_m3(radius_m: float) -> float:
    """Return the volume of one spherical-equivalent particle in SI units."""

    radius = float(radius_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise CohortSolverError(f"radius must be finite and positive; got {radius!r}")
    return 4.0 * math.pi * radius**3 / 3.0


def _sign_label(rate_m_s: float, *, tolerance: float = 0.0) -> str:
    if rate_m_s > tolerance:
        return "GROWTH"
    if rate_m_s < -tolerance:
        return "DISSOLUTION"
    return "NEUTRAL"


@dataclass
class Cohort:
    """One unsmoothed finite cohort carried by the characteristic comparator."""

    initial_id: str
    radius_m: float
    weight_m3: float
    active: bool = True
    dissolution_time_s: float | None = None
    radius_before_event_m: float | None = None
    returned_inventory_mol_m3: float = 0.0
    initial_radius_m: float | None = None
    initial_growth_sign: str = "UNASSESSED"

    def __post_init__(self) -> None:
        if not self.initial_id:
            raise CohortSolverError("each cohort requires a non-empty initial_id")
        if not math.isfinite(float(self.radius_m)) or float(self.radius_m) <= 0.0:
            raise CohortSolverError(f"{self.initial_id}: radius must be finite and positive")
        if not math.isfinite(float(self.weight_m3)) or float(self.weight_m3) <= 0.0:
            raise CohortSolverError(f"{self.initial_id}: weight_m3 must be finite and positive")
        if self.initial_radius_m is None:
            self.initial_radius_m = float(self.radius_m)
        if not math.isfinite(float(self.initial_radius_m)) or float(self.initial_radius_m) <= 0.0:
            raise CohortSolverError(f"{self.initial_id}: initial radius must be finite and positive")


@dataclass(frozen=True)
class CohortSnapshot:
    """Closed aggregate state at a requested output time."""

    time_s: float
    N_m0_m3: float
    M0_m3: float
    M1_m2: float
    M2_m: float
    M3_dimensionless: float
    Rmean_m: float
    Rmean3_m3: float
    Sv_m_inv: float
    f_beta: float
    matrix_xB: float
    beta_inventory_mol_m3: float
    matrix_inventory_mol_m3: float
    total_inventory_mol_m3: float
    inventory_residual_mol_m3: float
    inventory_relative_residual: float
    active_cohort_count: int
    cumulative_dissolution_inventory_mol_m3: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


class CohortSolver:
    """No-bin beta-only characteristic solver with exact lower-bound events.

    ``matrix_xB`` is never integrated.  It is recovered algebraically from
    the fixed total inventory and the instantaneous active-cohort volume.
    Cohorts are sorted by initial ID internally so equivalent input
    permutations have a deterministic reduction order and event identity.
    """

    solver_version = "kwn_discrete_cohort_characteristic_v1"

    def __init__(
        self,
        *,
        cohorts: Sequence[Cohort],
        beta_parameters: PopulationParameters,
        matrix_molar_volume_m3_mol: float,
        total_b_mol_m3: float,
        equilibrium_adapter: EquilibriumAdapter,
        temperature_k: float,
        r_diss_m: float,
        inventory_tolerance_relative: float = 1.0e-10,
        rtol: float = 1.0e-10,
        atol_m: float = 1.0e-18,
        method: str = "DOP853",
        contract_hash: str | None = None,
        source_config_hash: str | None = None,
    ) -> None:
        if beta_parameters.name != "beta":
            raise CohortSolverError("cohort comparator accepts only the beta population")
        if beta_parameters.nucleation.get("mode", "off") != "off":
            raise CohortSolverError("cohort comparator is beta-only and requires nucleation=off")
        if matrix_molar_volume_m3_mol <= 0.0:
            raise CohortSolverError("matrix molar volume must be positive")
        if total_b_mol_m3 <= 0.0:
            raise CohortSolverError("total B inventory must be positive")
        if temperature_k <= 0.0:
            raise CohortSolverError("temperature must be positive")
        if r_diss_m <= 0.0:
            raise CohortSolverError("r_diss_m must be positive")
        if not 0.0 < inventory_tolerance_relative <= 1.0:
            raise CohortSolverError("inventory tolerance must lie in (0, 1]")
        if not 0.0 < rtol < 1.0:
            raise CohortSolverError("rtol must lie in (0, 1)")
        if not 0.0 < atol_m:
            raise CohortSolverError("atol_m must be positive")
        if not cohorts:
            raise CohortSolverError("at least one cohort is required")

        copied = [copy.deepcopy(item) for item in cohorts]
        ids = [item.initial_id for item in copied]
        if len(ids) != len(set(ids)):
            raise CohortSolverError("cohort initial_id values must be unique")
        self.cohorts = sorted(copied, key=lambda item: item.initial_id)
        self.beta_parameters = beta_parameters
        self.matrix_molar_volume_m3_mol = float(matrix_molar_volume_m3_mol)
        self.total_b_mol_m3 = float(total_b_mol_m3)
        self.equilibrium_adapter = equilibrium_adapter
        self.temperature_k = float(temperature_k)
        self.r_diss_m = float(r_diss_m)
        self.inventory_tolerance_relative = float(inventory_tolerance_relative)
        self.rtol = float(rtol)
        self.atol_m = float(atol_m)
        self.method = str(method)
        self.contract_hash = contract_hash
        self.source_config_hash = source_config_hash
        self.time_s = 0.0
        self.accepted_segment_count = 0
        self._event_radius_tolerance_m = max(8.0 * self.atol_m, 1.0e-12 * self.r_diss_m)
        # The validation contract is only defined on the one-sided Rmin
        # domain.  End the numerical characteristic at the next representable
        # value above Rmin, then commit the physical event at Rmin itself.
        # This bounds location error by one binary64 ulp without evaluating an
        # invalid Gibbs--Thomson state below the lower-bound convention.
        self._event_coordinate_m = math.nextafter(self.r_diss_m, math.inf)

        for cohort in self.cohorts:
            if cohort.active and cohort.radius_m <= self.r_diss_m:
                raise CohortSolverError(
                    f"{cohort.initial_id}: initial radius is at/below frozen R_diss; "
                    "do not silently clamp or delete it"
                )
        initial_rates = self.growth_rates()
        for cohort, rate in zip(self._active_cohorts(), initial_rates):
            cohort.initial_growth_sign = _sign_label(float(rate))
        self._assert_inventory_closed()

    @classmethod
    def from_kwn_solver(
        cls,
        *,
        kwn_solver: KWNSolver,
        cohorts: Sequence[Cohort],
        rtol: float = 1.0e-10,
        atol_m: float = 1.0e-18,
        method: str = "DOP853",
    ) -> "CohortSolver":
        """Build from the exact beta-only KWN configuration without retuning."""

        beta = kwn_solver.population("beta")
        gp = kwn_solver.population("g")
        if gp.number_density_m3() != 0.0:
            raise CohortSolverError("cohort comparator is scoped to an empty GP population")
        return cls(
            cohorts=cohorts,
            beta_parameters=beta.parameters,
            matrix_molar_volume_m3_mol=kwn_solver.config.matrix_molar_volume_m3_mol,
            total_b_mol_m3=kwn_solver.ledger.total_b_mol_m3,
            equilibrium_adapter=kwn_solver.equilibrium_adapter,
            temperature_k=kwn_solver.config.temperature_k,
            r_diss_m=float(kwn_solver.config.grid.edges_m[0]),
            inventory_tolerance_relative=kwn_solver.config.inventory_tolerance_relative,
            rtol=rtol,
            atol_m=atol_m,
            method=method,
            contract_hash=kwn_solver.contract_hash,
            source_config_hash=kwn_solver.config.source_config_hash,
        )

    def clone(self) -> "CohortSolver":
        """Return an independent solver retaining the current accepted state."""

        result = CohortSolver(
            cohorts=copy.deepcopy(self.cohorts),
            beta_parameters=self.beta_parameters,
            matrix_molar_volume_m3_mol=self.matrix_molar_volume_m3_mol,
            total_b_mol_m3=self.total_b_mol_m3,
            equilibrium_adapter=self.equilibrium_adapter,
            temperature_k=self.temperature_k,
            r_diss_m=self.r_diss_m,
            inventory_tolerance_relative=self.inventory_tolerance_relative,
            rtol=self.rtol,
            atol_m=self.atol_m,
            method=self.method,
            contract_hash=self.contract_hash,
            source_config_hash=self.source_config_hash,
        )
        # Construction assesses the live growth sign; preserve the original
        # event identity and initial-sign provenance of the accepted state.
        result.cohorts = copy.deepcopy(self.cohorts)
        result.time_s = self.time_s
        result.accepted_segment_count = self.accepted_segment_count
        return result

    def _active_cohorts(self) -> list[Cohort]:
        return [item for item in self.cohorts if item.active]

    def _weights_for_active(self) -> NDArray[np.float64]:
        return np.asarray([item.weight_m3 for item in self._active_cohorts()], dtype=np.float64)

    def _radii_for_active(self) -> NDArray[np.float64]:
        return np.asarray([item.radius_m for item in self._active_cohorts()], dtype=np.float64)

    def _matrix_xb_from_active_radii(self, radii_m: NDArray[np.float64]) -> float:
        radii = np.asarray(radii_m, dtype=np.float64)
        if radii.ndim != 1 or np.any(~np.isfinite(radii)) or np.any(radii <= 0.0):
            raise CohortSolverError("active cohort radii must remain finite and positive")
        weights = self._weights_for_active()
        if radii.shape != weights.shape:
            raise CohortSolverError("active radius vector does not match active cohort state")
        beta_fraction = math.fsum(
            float(weight) * sphere_volume_m3(float(radius))
            for weight, radius in zip(weights, radii)
        )
        matrix_fraction = 1.0 - beta_fraction
        if matrix_fraction <= 0.0:
            raise CohortSolverError("active cohorts leave no matrix volume")
        beta_inventory = beta_fraction * self.beta_parameters.x_b / self.beta_parameters.molar_volume_m3_mol
        matrix_xb = (
            self.matrix_molar_volume_m3_mol
            * (self.total_b_mol_m3 - beta_inventory)
            / matrix_fraction
        )
        if not math.isfinite(matrix_xb) or not 0.0 <= matrix_xb <= 1.0:
            raise CohortSolverError(
                "inventory closure would require an unphysical matrix composition; no clamp was applied"
            )
        return float(matrix_xb)

    @property
    def matrix_xb(self) -> float:
        """Exact algebraic matrix composition at the current accepted state."""

        return self._matrix_xb_from_active_radii(self._radii_for_active())

    def growth_rates(self) -> NDArray[np.float64]:
        """Return current beta growth rates for active cohorts in canonical ID order."""

        radii = self._radii_for_active()
        if radii.size == 0:
            return np.zeros(0, dtype=np.float64)
        equilibrium = self.equilibrium_adapter.equilibrium_xb(radii, self.beta_parameters)
        return growth_rate_m_s(
            radii_m=radii,
            matrix_xb=self._matrix_xb_from_active_radii(radii),
            equilibrium_xb=equilibrium,
            parameters=self.beta_parameters,
        )

    def _rhs(self, _time_s: float, radii_m: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix_xb = self._matrix_xb_from_active_radii(radii_m)
        equilibrium = self.equilibrium_adapter.equilibrium_xb(radii_m, self.beta_parameters)
        return growth_rate_m_s(
            radii_m=radii_m,
            matrix_xb=matrix_xb,
            equilibrium_xb=equilibrium,
            parameters=self.beta_parameters,
        )

    def _inventory_components(self) -> tuple[float, float, float, float]:
        radii = self._radii_for_active()
        weights = self._weights_for_active()
        beta_fraction = math.fsum(
            float(weight) * sphere_volume_m3(float(radius))
            for weight, radius in zip(weights, radii)
        )
        matrix_fraction = 1.0 - beta_fraction
        matrix_xb = self._matrix_xb_from_active_radii(radii)
        beta_inventory = beta_fraction * self.beta_parameters.x_b / self.beta_parameters.molar_volume_m3_mol
        matrix_inventory = matrix_fraction * matrix_xb / self.matrix_molar_volume_m3_mol
        residual = matrix_inventory + beta_inventory - self.total_b_mol_m3
        return beta_fraction, beta_inventory, matrix_inventory, residual

    def _assert_inventory_closed(self) -> None:
        _, _, _, residual = self._inventory_components()
        relative = abs(residual) / max(abs(self.total_b_mol_m3), 1.0e-300)
        if relative > self.inventory_tolerance_relative:
            raise CohortSolverError(
                f"cohort inventory residual {relative:.3e} exceeds {self.inventory_tolerance_relative:.3e}"
            )

    def _retire_boundary_cohorts(self, radii_m: NDArray[np.float64]) -> None:
        """Commit one accepted R_diss event and return its inventory to matrix."""

        active = self._active_cohorts()
        candidates = [
            index
            for index, radius in enumerate(radii_m)
            if abs(float(radius) - self.r_diss_m) <= self._event_radius_tolerance_m
        ]
        if not candidates:
            nearest = int(np.argmin(radii_m))
            if float(radii_m[nearest]) <= self.r_diss_m + self._event_radius_tolerance_m:
                candidates = [nearest]
            else:
                raise CohortSolverError("terminal event did not identify a cohort at R_diss")
        for index in candidates:
            cohort = active[index]
            if not cohort.active:
                continue
            # Preserve the one-sided accepted characteristic coordinate for
            # the event diagnostic before committing the physical R_diss
            # state.  It is deliberately not a clamp: the coordinate is the
            # next binary64 number above the frozen lower-radius edge.
            cohort.radius_before_event_m = float(radii_m[index])
            cohort.radius_m = self.r_diss_m
            returned = (
                cohort.weight_m3
                * sphere_volume_m3(self.r_diss_m)
                * self.beta_parameters.x_b
                / self.beta_parameters.molar_volume_m3_mol
            )
            cohort.returned_inventory_mol_m3 += returned
            cohort.active = False
            cohort.dissolution_time_s = self.time_s
        self._assert_inventory_closed()

    def advance_to(self, target_time_s: float) -> None:
        """Advance through exact lower-radius events to an accepted target time."""

        target = float(target_time_s)
        if not math.isfinite(target) or target < self.time_s:
            raise CohortSolverError("target time must be finite and not precede the accepted state")
        if target == self.time_s:
            return
        try:
            from scipy.integrate import solve_ivp
        except ImportError as error:  # pragma: no cover - installation is part of runtime qualification
            raise CohortSolverError("scipy is required for the adaptive cohort integrator") from error

        while self.time_s < target:
            active = self._active_cohorts()
            if not active:
                self.time_s = target
                self._assert_inventory_closed()
                return
            radii = self._radii_for_active()
            rates = self.growth_rates()
            smallest_index = int(np.argmin(radii))
            if float(rates[smallest_index]) < 0.0:
                self._advance_via_dissolving_characteristic(
                    solve_ivp=solve_ivp,
                    target_time_s=target,
                    reference_index=smallest_index,
                )
            else:
                self._advance_without_lower_event(solve_ivp=solve_ivp, target_time_s=target)

    def _advance_without_lower_event(self, *, solve_ivp: Any, target_time_s: float) -> None:
        """Advance until target time or the smallest cohort turns dissolving."""

        active = self._active_cohorts()
        initial = self._radii_for_active()
        smallest_index = int(np.argmin(initial))
        # Stop a comfortably finite distance from the sign transition.  This
        # is a numerical event locator only; the subsequent characteristic
        # segment uses the unchanged physical growth law and R_diss contract.
        dissolution_guard_m_s = 1.0e-10

        def dissolution_onset_event(time_s: float, radii_m: NDArray[np.float64]) -> float:
            return float(self._rhs(time_s, radii_m)[smallest_index] + dissolution_guard_m_s)

        dissolution_onset_event.terminal = True  # type: ignore[attr-defined]
        dissolution_onset_event.direction = -1.0  # type: ignore[attr-defined]
        result = solve_ivp(
            self._rhs,
            (self.time_s, target_time_s),
            initial,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol_m,
            events=dissolution_onset_event,
        )
        if not result.success:
            raise CohortSolverError(f"cohort integrator failed: {result.message}")
        final_radii = np.asarray(result.y[:, -1], dtype=np.float64)
        if np.any(~np.isfinite(final_radii)) or np.any(
            final_radii <= self.r_diss_m + self._event_radius_tolerance_m
        ):
            raise CohortSolverError("growth-state integrator approached R_diss unexpectedly")
        for cohort, radius in zip(active, final_radii):
            cohort.radius_m = float(radius)
        self.time_s = float(result.t[-1])
        self.accepted_segment_count += 1
        if result.t_events[0].size:
            self._assert_inventory_closed()
            return
        if not math.isclose(self.time_s, target_time_s, rel_tol=0.0, abs_tol=1.0e-8):
            raise CohortSolverError("growth-state integrator stopped before target without a sign event")
        self.time_s = target_time_s
        self._assert_inventory_closed()

    def _advance_via_dissolving_characteristic(
        self, *, solve_ivp: Any, target_time_s: float, reference_index: int
    ) -> None:
        """Advance in the smallest dissolving radius rather than physical time.

        The exact validation contract becomes extremely stiff just above
        ``R_diss``.  A conventional time integrator can probe below the valid
        thermodynamic domain while looking for a terminal event.  Here the
        independent coordinate is the smallest physical radius itself, whose
        integration interval terminates *at* ``R_diss``.  Explicit DOP853
        stages remain one-sided on that interval, and the physical event is
        therefore an accepted endpoint rather than a negative-radius clamp or
        a rejected state promoted to physics.
        """

        active = self._active_cohorts()
        if not 0 <= reference_index < len(active):
            raise CohortSolverError("invalid dissolving reference cohort index")
        reference = active[reference_index]
        initial_radii = self._radii_for_active()
        reference_radius = float(initial_radii[reference_index])
        if reference_radius <= self.r_diss_m:
            raise CohortSolverError("dissolving reference has already crossed R_diss")
        # Exact equal-radius/equal-weight classes are independent physical
        # cohorts but share the same characteristic by symmetry.  Keep their
        # radii analytically tied to the reference coordinate so an adaptive
        # trial cannot separate two mathematically identical particles just
        # enough to probe below Rmin for one of them.
        tied_indices = [
            index
            for index, item in enumerate(active)
            if index != reference_index
            and item.weight_m3 == reference.weight_m3
            and item.radius_m == reference.radius_m
        ]
        other_indices = [
            index
            for index in range(len(active))
            if index != reference_index and index not in tied_indices
        ]
        initial_state = np.concatenate(
            (np.asarray([self.time_s], dtype=np.float64), initial_radii[other_indices])
        )

        def reconstruct(reference_radius_m: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
            radii = np.empty(len(active), dtype=np.float64)
            radii[reference_index] = float(reference_radius_m)
            radii[tied_indices] = float(reference_radius_m)
            radii[other_indices] = np.asarray(state[1:], dtype=np.float64)
            if np.any(radii < self.r_diss_m):
                raise CohortSolverError(
                    "a characteristic trial reached below R_diss before its event; "
                    f"the accepted state was rolled back (min={float(np.min(radii)):.17e} m)"
                )
            return radii

        def characteristic_rhs(
            reference_radius_m: float, state: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            radii = reconstruct(reference_radius_m, state)
            rates = self._rhs(float(state[0]), radii)
            reference_rate = float(rates[reference_index])
            if not math.isfinite(reference_rate):
                raise CohortSolverError("dissolving characteristic produced a non-finite growth rate")
            # RK trial stages can straddle the zero-growth turning point.
            # Keep the *trial-only* continuation one-sided so root finding can
            # locate the true ``reference_turn_event`` below; no such stage is
            # ever committed to a cohort state.
            if reference_rate >= 0.0:
                reference_rate = -max(reference_rate, 1.0e-30)
            return np.concatenate(
                (
                    np.asarray([1.0 / reference_rate], dtype=np.float64),
                    np.asarray([rates[index] / reference_rate for index in other_indices], dtype=np.float64),
                )
            )

        def target_time_event(_radius_m: float, state: NDArray[np.float64]) -> float:
            return float(state[0] - target_time_s)

        target_time_event.terminal = True  # type: ignore[attr-defined]
        target_time_event.direction = 1.0  # type: ignore[attr-defined]

        def reference_turn_event(
            reference_radius_m: float, state: NDArray[np.float64]
        ) -> float:
            radii = reconstruct(reference_radius_m, state)
            return float(self._rhs(float(state[0]), radii)[reference_index])

        # A finite cohort can restore its own matrix supersaturation as it
        # shrinks.  Locate that physical turning point rather than assuming a
        # momentarily dissolving cohort must reach Rmin.
        reference_turn_event.terminal = True  # type: ignore[attr-defined]
        reference_turn_event.direction = 1.0  # type: ignore[attr-defined]
        coordinate_span = reference_radius - self._event_coordinate_m
        max_coordinate_step = coordinate_span / 64.0
        result = solve_ivp(
            characteristic_rhs,
            (reference_radius, self._event_coordinate_m),
            initial_state,
            method=self.method,
            rtol=self.rtol,
            atol=np.concatenate((np.asarray([1.0e-8]), np.full(len(other_indices), self.atol_m))),
            events=(target_time_event, reference_turn_event),
            first_step=coordinate_span / 512.0,
            max_step=max_coordinate_step,
        )
        if not result.success:
            raise CohortSolverError(f"dissolving-characteristic integrator failed: {result.message}")
        final_reference_radius = float(result.t[-1])
        final_state = np.asarray(result.y[:, -1], dtype=np.float64)
        final_radii = reconstruct(final_reference_radius, final_state)
        if np.any(~np.isfinite(final_radii)):
            raise CohortSolverError("dissolving-characteristic integrator produced a non-finite radius")
        for cohort, radius in zip(active, final_radii):
            cohort.radius_m = float(radius)
        self.time_s = float(final_state[0])
        self.accepted_segment_count += 1
        if result.t_events[0].size:
            event_time_tolerance_s = max(1.0e-2, 1.0e-6 * target_time_s)
            if abs(self.time_s - target_time_s) > event_time_tolerance_s:
                raise CohortSolverError(
                    "target-time event does not close to the requested accepted time: "
                    f"observed={self.time_s:.17e} requested={target_time_s:.17e}"
                )
            if self.time_s != target_time_s:
                self._advance_short_time_correction(
                    solve_ivp=solve_ivp, target_time_s=target_time_s
                )
            self.time_s = target_time_s
            self._assert_inventory_closed()
            return
        if result.t_events[1].size:
            self._assert_inventory_closed()
            return
        if not math.isclose(
            final_reference_radius,
            self._event_coordinate_m,
            rel_tol=0.0,
            abs_tol=self._event_radius_tolerance_m,
        ):
            raise CohortSolverError("dissolving characteristic stopped before R_diss without a time event")
        self._retire_boundary_cohorts(final_radii)

    def _advance_short_time_correction(self, *, solve_ivp: Any, target_time_s: float) -> None:
        """Finish a dense-output event bracket at the exact requested time.

        ``solve_ivp`` locates events in an interpolant of the characteristic
        state.  When its event state falls microscopically before a requested
        output time, integrate that final, already-safe physical-time sliver
        instead of relabelling an earlier state as the output time.
        """

        remaining = target_time_s - self.time_s
        if remaining == 0.0:
            return
        active = self._active_cohorts()
        radii = self._radii_for_active()
        result = solve_ivp(
            self._rhs,
            (self.time_s, target_time_s),
            radii,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol_m,
            max_step=abs(remaining),
        )
        if not result.success:
            raise CohortSolverError(f"target-time correction failed: {result.message}")
        final_radii = np.asarray(result.y[:, -1], dtype=np.float64)
        if np.any(final_radii <= self.r_diss_m + self._event_radius_tolerance_m):
            raise CohortSolverError("target-time correction approached R_diss; event must be resolved first")
        for cohort, radius in zip(active, final_radii):
            cohort.radius_m = float(radius)
        self.time_s = target_time_s
        self.accepted_segment_count += 1

    def snapshot(self) -> CohortSnapshot:
        """Return moment, matrix, and strict-inventory diagnostics at this state."""

        radii = self._radii_for_active()
        weights = self._weights_for_active()
        moments = [
            math.fsum(float(weight) * float(radius) ** order for weight, radius in zip(weights, radii))
            for order in range(4)
        ]
        m0, m1, m2, m3 = (float(value) for value in moments)
        beta_fraction, beta_inventory, matrix_inventory, residual = self._inventory_components()
        relative = abs(residual) / max(abs(self.total_b_mol_m3), 1.0e-300)
        return CohortSnapshot(
            time_s=self.time_s,
            N_m0_m3=m0,
            M0_m3=m0,
            M1_m2=m1,
            M2_m=m2,
            M3_dimensionless=m3,
            Rmean_m=0.0 if m0 == 0.0 else m1 / m0,
            Rmean3_m3=0.0 if m0 == 0.0 else m3 / m0,
            Sv_m_inv=4.0 * math.pi * m2,
            f_beta=4.0 * math.pi * m3 / 3.0,
            matrix_xB=self.matrix_xb,
            beta_inventory_mol_m3=beta_inventory,
            matrix_inventory_mol_m3=matrix_inventory,
            total_inventory_mol_m3=self.total_b_mol_m3,
            inventory_residual_mol_m3=residual,
            inventory_relative_residual=relative,
            active_cohort_count=len(self._active_cohorts()),
            cumulative_dissolution_inventory_mol_m3=math.fsum(
                item.returned_inventory_mol_m3 for item in self.cohorts
            ),
        )

    def cohort_rows(self) -> list[dict[str, Any]]:
        """Return per-initial-ID state and event diagnostics without bin mapping."""

        rows: list[dict[str, Any]] = []
        rates = {
            item.initial_id: float(rate)
            for item, rate in zip(self._active_cohorts(), self.growth_rates())
        }
        for item in self.cohorts:
            rows.append(
                {
                    "initial_id": item.initial_id,
                    "initial_radius_m": float(item.initial_radius_m),
                    "weight_m3": item.weight_m3,
                    "initial_inventory_mol_m3": (
                        item.weight_m3
                        * sphere_volume_m3(float(item.initial_radius_m))
                        * self.beta_parameters.x_b
                        / self.beta_parameters.molar_volume_m3_mol
                    ),
                    "active": item.active,
                    "radius_m": item.radius_m,
                    "initial_growth_sign": item.initial_growth_sign,
                    "current_growth_dissolution_sign": (
                        _sign_label(rates[item.initial_id]) if item.active else "DISSOLVED"
                    ),
                    "dissolution_time_s": item.dissolution_time_s,
                    "radius_before_event_m": item.radius_before_event_m,
                    "returned_inventory_mol_m3": item.returned_inventory_mol_m3,
                    "current_inventory_mol_m3": (
                        item.weight_m3
                        * sphere_volume_m3(item.radius_m)
                        * self.beta_parameters.x_b
                        / self.beta_parameters.molar_volume_m3_mol
                        if item.active
                        else 0.0
                    ),
                }
            )
        return rows

    def checkpoint(self) -> dict[str, Any]:
        """Return a JSON-serialisable accepted-state checkpoint for restart tests."""

        return {
            "schema_version": "KWN_DISCRETE_COHORT_CHECKPOINT_V1",
            "solver_version": self.solver_version,
            "time_s": self.time_s,
            "r_diss_m": self.r_diss_m,
            "total_b_mol_m3": self.total_b_mol_m3,
            "contract_hash": self.contract_hash,
            "source_config_hash": self.source_config_hash,
            "cohorts": [
                {
                    "initial_id": item.initial_id,
                    "radius_m": item.radius_m,
                    "weight_m3": item.weight_m3,
                    "active": item.active,
                    "initial_radius_m": item.initial_radius_m,
                    "initial_growth_sign": item.initial_growth_sign,
                    "dissolution_time_s": item.dissolution_time_s,
                    "radius_before_event_m": item.radius_before_event_m,
                    "returned_inventory_mol_m3": item.returned_inventory_mol_m3,
                }
                for item in self.cohorts
            ],
        }

    def restore_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        """Restore a compatible accepted state; incompatible contracts fail closed."""

        if checkpoint.get("schema_version") != "KWN_DISCRETE_COHORT_CHECKPOINT_V1":
            raise CohortSolverError("unsupported cohort checkpoint schema")
        if checkpoint.get("contract_hash") != self.contract_hash:
            raise CohortSolverError("cohort checkpoint validation-contract hash mismatch")
        if checkpoint.get("source_config_hash") != self.source_config_hash:
            raise CohortSolverError("cohort checkpoint source-config hash mismatch")
        if not math.isclose(float(checkpoint.get("r_diss_m", float("nan"))), self.r_diss_m, rel_tol=0.0, abs_tol=0.0):
            raise CohortSolverError("cohort checkpoint lower-radius convention mismatch")
        if not math.isclose(float(checkpoint.get("total_b_mol_m3", float("nan"))), self.total_b_mol_m3, rel_tol=0.0, abs_tol=0.0):
            raise CohortSolverError("cohort checkpoint total inventory mismatch")
        saved = checkpoint.get("cohorts")
        if not isinstance(saved, list) or len(saved) != len(self.cohorts):
            raise CohortSolverError("cohort checkpoint cohort list is incompatible")
        by_id = {str(item.get("initial_id")): item for item in saved if isinstance(item, Mapping)}
        if len(by_id) != len(self.cohorts):
            raise CohortSolverError("cohort checkpoint IDs are invalid")
        for cohort in self.cohorts:
            item = by_id.get(cohort.initial_id)
            if item is None or not math.isclose(float(item["weight_m3"]), cohort.weight_m3, rel_tol=0.0, abs_tol=0.0):
                raise CohortSolverError("cohort checkpoint weights are incompatible")
            cohort.radius_m = float(item["radius_m"])
            cohort.active = bool(item["active"])
            cohort.initial_growth_sign = str(item["initial_growth_sign"])
            cohort.dissolution_time_s = (
                None if item.get("dissolution_time_s") is None else float(item["dissolution_time_s"])
            )
            cohort.radius_before_event_m = (
                None if item.get("radius_before_event_m") is None else float(item["radius_before_event_m"])
            )
            cohort.returned_inventory_mol_m3 = float(item["returned_inventory_mol_m3"])
        self.time_s = float(checkpoint["time_s"])
        self._assert_inventory_closed()
