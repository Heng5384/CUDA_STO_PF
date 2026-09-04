"""Read-only semigroup diagnostics for the conservative CR1 operator.

This module is deliberately diagnostic-only.  It exposes an in-memory
one-step operator built from the unchanged CR1 production candidate step and
never writes a checkpoint, accepts an externally selected root, or mutates
the caller's accepted trajectory.  It also contains the two causal-isolation
variants required by the CR1 semigroup audit:

* a frozen-matrix transport map; and
* a prescribed-matrix bridge whose values come from an external trajectory.

Neither variant is a production solver or a proposed replacement integrator.
They exist only to separate transport/remap time refinement from the nonlinear
population--matrix feedback used by the ordinary CR1 map.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .characteristic_dt_continuation import SolverStateSnapshot, accepted_state_hash
from .characteristic_pathology import (
    ClosureMapEvaluation,
    capture_prestate,
    evaluate_closure_map,
    evaluation_from_recorded_trial,
    solver_state_hash,
    state_distance,
)
from .characteristic_reference import (
    CharacteristicReferenceError,
    CharacteristicReferenceSolver,
    CharacteristicStepDiagnostics,
    _ClosureTrial,
)
from .ledger import InventorySnapshot
from .population_metrics import (
    cell_moments_from_piecewise_constant_cells,
    metrics_from_piecewise_constant_cells,
)
from .solver import RadiusGridOverflowError


class SemigroupAuditError(RuntimeError):
    """A diagnostic-contract error; never a request to repair CR1."""


_PHYSICAL_SCALAR_FIELDS = (
    "xB",
    "M0_m3",
    "M1_m2",
    "M2_m",
    "M3_dimensionless",
    "Rmean_m",
    "Rmean3_m3",
    "Sv_m_inv",
    "f_beta",
    "Q_beta_mol_m3",
    "Q_matrix_mol_m3",
    "Q_total_mol_m3",
)


def _cdf(cells: NDArray[np.float64]) -> NDArray[np.float64]:
    total = float(np.sum(cells, dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        return np.zeros(cells.shape, dtype=np.float64)
    return np.cumsum(cells, dtype=np.float64) / total


class RecordingCharacteristicReferenceSolver(CharacteristicReferenceSolver):
    """Production CR1 plus immutable trial telemetry for diagnostics only."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._recorded_trials: list[Any] = []
        super().__init__(*args, **kwargs)

    def _evaluate_closure_trial(self, **kwargs: Any) -> Any:
        trial = super()._evaluate_closure_trial(**kwargs)
        self._recorded_trials.append(trial)
        return trial

    def trial_cursor(self) -> int:
        return len(self._recorded_trials)

    def trials_since(self, cursor: int) -> tuple[Any, ...]:
        return tuple(self._recorded_trials[int(cursor) :])


class FrozenMatrixCharacteristicReferenceSolver(RecordingCharacteristicReferenceSolver):
    """CR1 transport with a fixed matrix composition, for causal isolation."""

    def __init__(self, *args: Any, frozen_matrix_xb: float, **kwargs: Any) -> None:
        value = float(frozen_matrix_xb)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("frozen_matrix_xb must lie in [0, 1]")
        self.frozen_matrix_xb = value
        super().__init__(*args, **kwargs)

    def _recover_matrix_xb(self, beta_cell_number_m3: NDArray[np.float64]) -> float:
        # Preserve the normal trial inventory calculation in the caller, but
        # cut only its candidate-population -> xB feedback edge.  With the
        # accepted state itself also at this xB, ordinary Picard closes at the
        # first unchanged matrix iterate and leaves all transport mechanics
        # untouched.
        del beta_cell_number_m3
        return self.frozen_matrix_xb

    def _inventory_snapshot(
        self, beta_cell_number_m3: NDArray[np.float64], matrix_xb: float
    ) -> InventorySnapshot:
        return _open_matrix_inventory_snapshot(self, beta_cell_number_m3, matrix_xb)


def _open_matrix_inventory_snapshot(
    solver: CharacteristicReferenceSolver,
    beta_cell_number_m3: NDArray[np.float64],
    matrix_xb: float,
) -> InventorySnapshot:
    """Report, but do not reject, inventory drift for an external-xB probe.

    Frozen and prescribed matrix diagnostics intentionally break the algebraic
    population -> matrix feedback edge.  Their inventory residual is therefore
    causal evidence, not a closure error.  This reproduces the ledger's
    quantities without changing its strict production ``snapshot`` method.
    """

    populations = solver._trial_populations(np.asarray(beta_cell_number_m3, dtype=np.float64))
    gp, beta = populations
    matrix_fraction = 1.0 - gp.volume_fraction() - beta.volume_fraction()
    if not math.isfinite(matrix_fraction) or matrix_fraction < 0.0:
        raise SemigroupAuditError("external-matrix diagnostic has an invalid matrix fraction")
    value = float(matrix_xb)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise SemigroupAuditError("external-matrix diagnostic xB is not physical")
    gp_inventory = float(gp.b_inventory_mol_m3())
    beta_inventory = float(beta.b_inventory_mol_m3())
    matrix_inventory = matrix_fraction * value / solver.ledger.matrix_molar_volume_m3_mol
    reconstructed = matrix_inventory + gp_inventory + beta_inventory
    residual = reconstructed - solver.ledger.total_b_mol_m3
    denominator = max(abs(solver.ledger.total_b_mol_m3), 1.0e-300)
    if not all(math.isfinite(item) for item in (gp_inventory, beta_inventory, matrix_inventory, residual)):
        raise SemigroupAuditError("external-matrix diagnostic inventory is non-finite")
    return InventorySnapshot(
        total_mol_m3=float(solver.ledger.total_b_mol_m3),
        matrix_mol_m3=matrix_inventory,
        gp_mol_m3=gp_inventory,
        beta_subgrid_mol_m3=0.0,
        beta_resolved_mol_m3=beta_inventory,
        residual_mol_m3=residual,
        relative_residual=abs(residual) / denominator,
        matrix_fraction=matrix_fraction,
    )


@dataclass(frozen=True)
class PiecewiseLinearMatrixTrajectory:
    """A finite, immutable prescribed xB(t) bridge trajectory."""

    times_s: NDArray[np.float64]
    xb: NDArray[np.float64]

    def __post_init__(self) -> None:
        times = np.asarray(self.times_s, dtype=np.float64)
        values = np.asarray(self.xb, dtype=np.float64)
        if (
            times.ndim != 1
            or values.ndim != 1
            or times.size < 2
            or times.shape != values.shape
            or not np.all(np.isfinite(times))
            or not np.all(np.isfinite(values))
            or np.any(np.diff(times) <= 0.0)
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise ValueError("prescribed xB trajectory must be finite, ordered, and physical")
        object.__setattr__(self, "times_s", times.copy())
        object.__setattr__(self, "xb", values.copy())

    def value(self, time_s: float) -> float:
        time = float(time_s)
        lower, upper = float(self.times_s[0]), float(self.times_s[-1])
        # The audit uses exact binary64 time subdivisions.  Permit the
        # unavoidable one-ulp endpoint representation only, never extrapolate.
        tolerance = 8.0 * np.finfo(np.float64).eps * max(abs(lower), abs(upper), 1.0)
        if time < lower - tolerance or time > upper + tolerance:
            raise SemigroupAuditError("prescribed xB trajectory does not cover diagnostic time")
        return float(np.interp(min(max(time, lower), upper), self.times_s, self.xb))


class PrescribedMatrixCharacteristicReferenceSolver(RecordingCharacteristicReferenceSolver):
    """CR1 tracing under a shared external xB(t), without population feedback."""

    def __init__(
        self,
        *args: Any,
        prescribed_matrix_trajectory: PiecewiseLinearMatrixTrajectory,
        **kwargs: Any,
    ) -> None:
        self.prescribed_matrix_trajectory = prescribed_matrix_trajectory
        super().__init__(*args, **kwargs)

    def _evaluate_closure_trial(
        self,
        *,
        old_cell_number_m3: NDArray[np.float64],
        dt_s: float,
        x_start: float,
        x_guess: float,
    ) -> _ClosureTrial:
        """Use prescribed endpoint values with the production midpoint rule.

        The ordinary CR1 loop still evaluates its own scalar residual, but the
        candidate matrix value is fixed by xB(t + h).  The first Picard
        iterate moves the guess to that endpoint; the second confirms the
        normal midpoint ``0.5 * (x_n + x_{n+1})``.  No root is selected.
        """

        start = self.prescribed_matrix_trajectory.value(self.time_s)
        endpoint = self.prescribed_matrix_trajectory.value(self.time_s + float(dt_s))
        if float(x_start) != start:
            raise SemigroupAuditError("prescribed-x diagnostic state is not aligned to xB(t)")
        midpoint = 0.5 * (start + endpoint)
        trace, remap = self._remap_once(
            np.asarray(old_cell_number_m3, dtype=np.float64),
            dt_s=float(dt_s),
            midpoint_matrix_xb=midpoint,
        )
        candidate = np.asarray(remap.cell_number_m3, dtype=np.float64)
        inventory = self._inventory_snapshot(candidate, endpoint)
        tolerance = self.fixed_point_atol + self.fixed_point_rtol * max(abs(endpoint), abs(x_guess))
        trial = _ClosureTrial(
            x_guess=float(x_guess),
            midpoint_matrix_xb=midpoint,
            cell_number_m3=candidate,
            matrix_xb=endpoint,
            signed_xb_residual=endpoint - float(x_guess),
            xb_tolerance=tolerance,
            inventory=inventory,
            trace=trace,
            remap=remap,
        )
        self._recorded_trials.append(trial)
        return trial

    def _inventory_snapshot(
        self, beta_cell_number_m3: NDArray[np.float64], matrix_xb: float
    ) -> InventorySnapshot:
        return _open_matrix_inventory_snapshot(self, beta_cell_number_m3, matrix_xb)


@dataclass(frozen=True)
class PhiResult:
    """One disposable evaluation of the CR1 one-step operator."""

    status: str
    mode: str
    requested_dt_s: float
    completed_substeps: int
    error_type: str | None
    error_message: str | None
    state: Mapping[str, Any] | None
    diagnostic: CharacteristicStepDiagnostics | None
    accepted_evaluation: ClosureMapEvaluation | None
    solver: CharacteristicReferenceSolver | None
    topology_path: tuple[Mapping[str, Any], ...]


def _constructor_kwargs(source: CharacteristicReferenceSolver) -> dict[str, Any]:
    return {
        "fixed_point_rtol": float(source.fixed_point_rtol),
        "fixed_point_atol": float(source.fixed_point_atol),
        "fixed_point_max_iterations": int(source.fixed_point_max_iterations),
        "under_relaxation": float(source.under_relaxation),
    }


def _clone_for_mode(
    source: CharacteristicReferenceSolver,
    *,
    mode: str,
    frozen_matrix_xb: float | None,
    prescribed_matrix_trajectory: PiecewiseLinearMatrixTrajectory | None,
) -> RecordingCharacteristicReferenceSolver:
    kwargs = _constructor_kwargs(source)
    if mode == "DYNAMIC":
        clone: RecordingCharacteristicReferenceSolver = RecordingCharacteristicReferenceSolver(
            source.config, **kwargs
        )
    elif mode == "FROZEN_MATRIX":
        if frozen_matrix_xb is None:
            raise SemigroupAuditError("frozen-matrix operator requires its initial xB")
        clone = FrozenMatrixCharacteristicReferenceSolver(
            source.config, frozen_matrix_xb=float(frozen_matrix_xb), **kwargs
        )
    elif mode == "PRESCRIBED_X":
        if prescribed_matrix_trajectory is None:
            raise SemigroupAuditError("prescribed-x operator requires a trajectory")
        clone = PrescribedMatrixCharacteristicReferenceSolver(
            source.config,
            prescribed_matrix_trajectory=prescribed_matrix_trajectory,
            **kwargs,
        )
    else:
        raise ValueError(f"unsupported semigroup diagnostic mode {mode}")
    SolverStateSnapshot.capture(source).restore(clone)
    if mode == "PRESCRIBED_X":
        required = prescribed_matrix_trajectory.value(clone.time_s)
        if float(clone.matrix_xb) != required:
            raise SemigroupAuditError("prescribed-x initial state differs from the bridge trajectory")
    return clone


def _accepted_trial_evaluation(
    solver: RecordingCharacteristicReferenceSolver,
    *,
    cursor: int,
    old_cells: NDArray[np.float64],
    diagnostic: CharacteristicStepDiagnostics,
) -> ClosureMapEvaluation:
    """Recover the exact trial that produced an accepted disposable step."""

    widths = np.asarray(solver.population("beta").grid.widths_m, dtype=np.float64)
    committed = np.asarray(solver._beta_cell_numbers(), dtype=np.float64)
    matches = [
        trial
        for trial in solver.trials_since(cursor)
        if trial.matrix_xb == float(solver.matrix_xb)
        and trial.midpoint_matrix_xb == float(diagnostic.midpoint_matrix_xb)
        and np.array_equal(
            (np.asarray(trial.cell_number_m3, dtype=np.float64) / widths) * widths,
            committed,
        )
    ]
    if not matches:
        raise SemigroupAuditError("accepted disposable CR1 step has no matching trial telemetry")
    return evaluation_from_recorded_trial(solver, trial=matches[-1], old_cell_number_m3=old_cells)


def state_observation(
    solver: CharacteristicReferenceSolver,
    *,
    accepted_evaluation: ClosureMapEvaluation | None,
) -> dict[str, Any]:
    """Capture physical state and exact last-step topology without a write."""

    cells = np.asarray(solver._beta_cell_numbers(), dtype=np.float64).copy()
    edges = np.asarray(solver.population("beta").grid.edges_m, dtype=np.float64)
    metrics = metrics_from_piecewise_constant_cells(edges, cells)
    inventory = solver._inventory_snapshot(cells, float(solver.matrix_xb))
    return {
        "population_array": cells,
        "cdf": _cdf(cells),
        "xB": float(solver.matrix_xb),
        "M0_m3": float(metrics.M0_m3),
        "M1_m2": float(metrics.M1_m2),
        "M2_m": float(metrics.M2_m),
        "M3_dimensionless": float(metrics.M3_dimensionless),
        "Rmean_m": float(metrics.Rmean_number_m),
        "Rmean3_m3": float(metrics.Rmean_cubed_m3),
        "Sv_m_inv": float(metrics.Sv_m_inv),
        "f_beta": float(metrics.f_beta),
        "Q_beta_mol_m3": float(inventory.beta_resolved_mol_m3),
        "Q_matrix_mol_m3": float(inventory.matrix_mol_m3),
        "Q_total_mol_m3": float(inventory.total_mol_m3),
        "inventory_relative_residual": float(inventory.relative_residual),
        "time_s": float(solver.time_s),
        "step": int(solver.step),
        "accepted_state_hash": accepted_state_hash(solver),
        "diagnostic_state_hash": solver_state_hash(solver),
        "departure_faces_m": (
            np.empty(0, dtype=np.float64)
            if accepted_evaluation is None
            else np.asarray(accepted_evaluation.departure_faces_m, dtype=np.float64).copy()
        ),
        "source_cell_indices": (
            np.empty(0, dtype=np.int64)
            if accepted_evaluation is None
            else np.asarray(accepted_evaluation.source_cell_indices, dtype=np.int64).copy()
        ),
        "remap_topology_signature": (
            "UNAVAILABLE"
            if accepted_evaluation is None
            else str(accepted_evaluation.remap_topology_signature)
        ),
        "trace_topology_signature": (
            "UNAVAILABLE"
            if accepted_evaluation is None
            else str(accepted_evaluation.trace_topology_signature)
        ),
        "departure_signature": (
            "UNAVAILABLE"
            if accepted_evaluation is None
            else str(accepted_evaluation.departure_signature)
        ),
    }


def phi_step(
    source: CharacteristicReferenceSolver,
    *,
    dt_s: float,
    mode: str = "DYNAMIC",
    frozen_matrix_xb: float | None = None,
    prescribed_matrix_trajectory: PiecewiseLinearMatrixTrajectory | None = None,
) -> PhiResult:
    """Evaluate the production CR1 candidate step on a disposable clone.

    The normal ordinary/P2/P4 closure policy is deliberately retained.  A
    non-closing raw CR1 map returns ``NONCLOSING`` and no state; this function
    never calls the legacy final raw-Picard safeguard or any generic scalar
    root mechanism.
    """

    requested = float(dt_s)
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError("Phi_h requires a finite positive h")
    before = SolverStateSnapshot.capture(source)
    clone = _clone_for_mode(
        source,
        mode=mode,
        frozen_matrix_xb=frozen_matrix_xb,
        prescribed_matrix_trajectory=prescribed_matrix_trajectory,
    )
    old_cells = np.asarray(clone._beta_cell_numbers(), dtype=np.float64).copy()
    cursor = clone.trial_cursor()
    try:
        diagnostic = clone.advance_one_ordinary_or_qualified_cycle(maximum_dt_s=requested)
        if clone.trials_since(cursor):
            accepted = _accepted_trial_evaluation(
                clone, cursor=cursor, old_cells=old_cells, diagnostic=diagnostic
            )
        elif str(diagnostic.fixed_point_convergence_mode) == "IDENTITY":
            # The production identity branch intentionally skips a closure
            # trial to retain its accepted density bit-for-bit.  Reconstruct
            # its topology only after that accepted clone step, through the
            # existing read-only map evaluator; this cannot alter the clone's
            # committed identity state or its caller.
            telemetry_clone = _clone_for_mode(
                source,
                mode=mode,
                frozen_matrix_xb=frozen_matrix_xb,
                prescribed_matrix_trajectory=prescribed_matrix_trajectory,
            )
            identity_prestate = capture_prestate(telemetry_clone)
            accepted = evaluate_closure_map(
                telemetry_clone,
                prestate=identity_prestate,
                dt_s=requested,
                x_trial=identity_prestate.x_start,
            )
        else:
            raise SemigroupAuditError("accepted Phi_h step has no closure-trial telemetry")
        state = state_observation(clone, accepted_evaluation=accepted)
    except (CharacteristicReferenceError, RadiusGridOverflowError, ValueError, FloatingPointError) as error:
        if not before.matches(source):
            raise SemigroupAuditError("a nonclosing Phi_h evaluation mutated its input state") from error
        return PhiResult(
            status="NONCLOSING",
            mode=mode,
            requested_dt_s=requested,
            completed_substeps=0,
            error_type=type(error).__name__,
            error_message=str(error),
            state=None,
            diagnostic=None,
            accepted_evaluation=None,
            solver=None,
            topology_path=(),
        )
    if not before.matches(source):
        raise SemigroupAuditError("a successful Phi_h evaluation mutated its input state")
    return PhiResult(
        status="SUCCESS",
        mode=mode,
        requested_dt_s=requested,
        completed_substeps=1,
        error_type=None,
        error_message=None,
        state=state,
        diagnostic=diagnostic,
        accepted_evaluation=accepted,
        solver=clone,
        topology_path=(state,),
    )


def phi_compose(
    source: CharacteristicReferenceSolver,
    *,
    dt_s: float,
    count: int,
    mode: str = "DYNAMIC",
    frozen_matrix_xb: float | None = None,
    prescribed_matrix_trajectory: PiecewiseLinearMatrixTrajectory | None = None,
) -> PhiResult:
    """Compose exactly ``count`` disposable Phi evaluations from one state."""

    if isinstance(count, bool) or int(count) < 1:
        raise ValueError("Phi composition count must be a positive integer")
    current: CharacteristicReferenceSolver = source
    final: PhiResult | None = None
    topology_path: list[Mapping[str, Any]] = []
    for index in range(1, int(count) + 1):
        result = phi_step(
            current,
            dt_s=dt_s,
            mode=mode,
            frozen_matrix_xb=frozen_matrix_xb,
            prescribed_matrix_trajectory=prescribed_matrix_trajectory,
        )
        if result.status != "SUCCESS":
            return PhiResult(
                status="NONCLOSING",
                mode=mode,
                requested_dt_s=float(dt_s),
                completed_substeps=index - 1,
                error_type=result.error_type,
                error_message=result.error_message,
                state=None,
                diagnostic=None,
                accepted_evaluation=None,
                solver=None,
                topology_path=tuple(topology_path),
            )
        if result.solver is None:
            raise SemigroupAuditError("successful Phi_h did not retain its disposable state")
        current = result.solver
        final = result
        topology_path.extend(result.topology_path)
    if final is None:
        raise AssertionError("unreachable empty Phi composition")
    return PhiResult(
        status="SUCCESS",
        mode=mode,
        requested_dt_s=float(dt_s),
        completed_substeps=int(count),
        error_type=None,
        error_message=None,
        state=final.state,
        diagnostic=final.diagnostic,
        accepted_evaluation=final.accepted_evaluation,
        solver=final.solver,
        topology_path=tuple(topology_path),
    )


def physical_state_distance(
    left: Mapping[str, Any], right: Mapping[str, Any], *, edges_m: NDArray[np.float64]
) -> dict[str, Any]:
    """Return every requested physical error plus one dimensionless E_h."""

    values: dict[str, Any] = state_distance(left, right, edges_m=np.asarray(edges_m, dtype=np.float64))
    dimensionless = [
        float(values["population_L1"]),
        float(values["population_Linf"]),
        float(values["CDF_difference"]),
        float(values["PSD_Wasserstein_m"])
        / max(abs(float(right["Rmean_m"])), abs(float(left["Rmean_m"])), 1.0e-300),
    ]
    values["PSD_Wasserstein_normalized"] = dimensionless[-1]
    for field in _PHYSICAL_SCALAR_FIELDS:
        absolute = abs(float(left[field]) - float(right[field]))
        relative = absolute / max(abs(float(left[field])), abs(float(right[field])), 1.0e-300)
        values[f"{field}_error"] = absolute
        values[f"{field}_relative_error"] = relative
        dimensionless.append(relative)
    values["E_physical_max_relative"] = max(dimensionless)
    return values


def topology_comparison(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Compare array-valued CR1 partitions rather than topology hashes alone."""

    left_source = np.asarray(left["source_cell_indices"], dtype=np.int64)
    right_source = np.asarray(right["source_cell_indices"], dtype=np.int64)
    comparable = left_source.shape == right_source.shape and left_source.size > 0
    changed = (
        np.flatnonzero(left_source != right_source).astype(np.int64)
        if comparable
        else np.empty(0, dtype=np.int64)
    )
    signature_equal = str(left["remap_topology_signature"]) == str(right["remap_topology_signature"])
    return {
        "topology_equal": bool(signature_equal and comparable),
        "topology_signature_equal": signature_equal,
        "source_partition_comparable": comparable,
        "changed_face_count": int(changed.size),
        "changed_face_indices": changed,
        "left_topology_signature": str(left["remap_topology_signature"]),
        "right_topology_signature": str(right["remap_topology_signature"]),
        "left_trace_topology_signature": str(left["trace_topology_signature"]),
        "right_trace_topology_signature": str(right["trace_topology_signature"]),
    }


def topology_path_summary(path: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe a composed map's full leaf topology, without collapsing it."""

    signatures = [str(state["remap_topology_signature"]) for state in path]
    departure_signatures = [str(state["departure_signature"]) for state in path]
    source_partitions = [
        np.asarray(state["source_cell_indices"], dtype=np.int64).copy()
        for state in path
    ]
    changed_faces: set[int] = set()
    transitions: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(path, path[1:]), start=1):
        comparison = topology_comparison(left, right)
        faces = [int(face) for face in comparison["changed_face_indices"]]
        changed_faces.update(faces)
        transitions.append({
            "from_leaf": index,
            "to_leaf": index + 1,
            "changed_face_count": len(faces),
            "changed_face_indices": np.asarray(faces, dtype=np.int64),
            "topology_signature_equal": bool(comparison["topology_signature_equal"]),
        })
    return {
        "leaf_count": len(path),
        "leaf_topology_signatures": signatures,
        "leaf_departure_signatures": departure_signatures,
        "leaf_source_cell_indices": source_partitions,
        "unique_topology_signatures": sorted(set(signatures)),
        "internal_transition_count": len(transitions),
        "internal_changed_face_union": np.asarray(sorted(changed_faces), dtype=np.int64),
        "internal_transitions": transitions,
    }


def topology_path_comparison(
    left_path: Sequence[Mapping[str, Any]], right_path: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compare complete leaf histories, never just unequal final leaf sizes.

    Paths with different leaf counts cannot be used as an equality gate.  The
    result therefore records their full signature histories and whether their
    *unique* combinatorial regimes differ; the physical semigroup verdict is
    kept entirely separate.
    """

    left = topology_path_summary(left_path)
    right = topology_path_summary(right_path)
    return {
        "left": left,
        "right": right,
        "same_leaf_count": left["leaf_count"] == right["leaf_count"],
        "leaf_signature_sequence_equal": left["leaf_topology_signatures"] == right["leaf_topology_signatures"],
        "unique_signature_sets_equal": (
            left["unique_topology_signatures"] == right["unique_topology_signatures"]
        ),
        "different_discrete_topology_regimes": (
            left["unique_topology_signatures"] != right["unique_topology_signatures"]
        ),
        "comparison_is_not_a_physical_semigroup_gate": True,
    }


def semigroup_triplet(
    source: CharacteristicReferenceSolver,
    *,
    h_s: float,
    mode: str,
    frozen_matrix_xb: float | None = None,
    prescribed_matrix_trajectory: PiecewiseLinearMatrixTrajectory | None = None,
) -> dict[str, Any]:
    """Compare Phi_h, Phi_(h/2)^2, and Phi_(h/4)^4 from exactly one U."""

    h = float(h_s)
    one = phi_compose(
        source, dt_s=h, count=1, mode=mode, frozen_matrix_xb=frozen_matrix_xb,
        prescribed_matrix_trajectory=prescribed_matrix_trajectory,
    )
    two = phi_compose(
        source, dt_s=0.5 * h, count=2, mode=mode, frozen_matrix_xb=frozen_matrix_xb,
        prescribed_matrix_trajectory=prescribed_matrix_trajectory,
    )
    four = phi_compose(
        source, dt_s=0.25 * h, count=4, mode=mode, frozen_matrix_xb=frozen_matrix_xb,
        prescribed_matrix_trajectory=prescribed_matrix_trajectory,
    )
    payload: dict[str, Any] = {
        "mode": mode,
        "h_s": h,
        "Phi_h_status": one.status,
        "Phi_h2x2_status": two.status,
        "Phi_h4x4_status": four.status,
        "Phi_h_completed_substeps": one.completed_substeps,
        "Phi_h2x2_completed_substeps": two.completed_substeps,
        "Phi_h4x4_completed_substeps": four.completed_substeps,
        "Phi_h_error": one.error_message,
        "Phi_h2x2_error": two.error_message,
        "Phi_h4x4_error": four.error_message,
    }
    if any(result.status != "SUCCESS" for result in (one, two, four)):
        payload["status"] = "INSUFFICIENT_SUCCESSFUL_PATHS"
        return payload
    if one.state is None or two.state is None or four.state is None:
        raise SemigroupAuditError("successful semigroup path lacks its physical state")
    edges = np.asarray(source.population("beta").grid.edges_m, dtype=np.float64)
    h_vs_two = physical_state_distance(one.state, two.state, edges_m=edges)
    two_vs_four = physical_state_distance(two.state, four.state, edges_m=edges)
    h_vs_four = physical_state_distance(one.state, four.state, edges_m=edges)
    payload.update({
        "status": "SUCCESS",
        "E_h": float(h_vs_two["E_physical_max_relative"]),
        "E_h2_internal": float(two_vs_four["E_physical_max_relative"]),
        "E_h_vs_h4": float(h_vs_four["E_physical_max_relative"]),
        "h_vs_two": h_vs_two,
        "two_vs_four": two_vs_four,
        "h_vs_four": h_vs_four,
        "topology_h_vs_two": topology_comparison(one.state, two.state),
        "topology_two_vs_four": topology_comparison(two.state, four.state),
        "topology_h_vs_four": topology_comparison(one.state, four.state),
        "topology_path_h": topology_path_summary(one.topology_path),
        "topology_path_h2x2": topology_path_summary(two.topology_path),
        "topology_path_h4x4": topology_path_summary(four.topology_path),
        "topology_path_h_vs_h2x2": topology_path_comparison(one.topology_path, two.topology_path),
        "topology_path_h2x2_vs_h4x4": topology_path_comparison(two.topology_path, four.topology_path),
        "Phi_h_state": one.state,
        "Phi_h2x2_state": two.state,
        "Phi_h4x4_state": four.state,
    })
    return payload


def classify_semigroup_sequence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Classify observed E_h reduction without introducing an error threshold."""

    successful = [row for row in rows if row.get("status") == "SUCCESS"]
    values = [float(row["E_h"]) for row in successful]
    if len(values) < 3:
        return {
            "classification": "INSUFFICIENT_SUCCESSFUL_LEVELS",
            "E_values": values,
            "observed_ratios": [],
        }
    ratios = [
        (right / left if left != 0.0 else (0.0 if right == 0.0 else math.inf))
        for left, right in zip(values, values[1:])
    ]
    if all(right < left for left, right in zip(values, values[1:])) or all(value == 0.0 for value in values):
        classification = "PHYSICAL_SEMIGROUP_CONVERGENCE"
    elif values[-1] >= values[0]:
        classification = "PHYSICAL_SEMIGROUP_DIVERGENCE"
    else:
        classification = "PHYSICAL_SEMIGROUP_STAGNATION"
    return {
        "classification": classification,
        "E_values": values,
        "observed_ratios": ratios,
    }


def topology_event_proximity_rows(
    states: Mapping[str, Mapping[str, Any]], *, edges_m: NDArray[np.float64]
) -> list[dict[str, Any]]:
    """Locate differing CR1 faces relative to their nearest source boundary."""

    labels = sorted(states)
    changed_faces: set[int] = set()
    for left_label, right_label in combinations(labels, 2):
        comparison = topology_comparison(states[left_label], states[right_label])
        changed_faces.update(int(value) for value in comparison["changed_face_indices"])
    edges = np.asarray(edges_m, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for label in labels:
        state = states[label]
        departure = np.asarray(state["departure_faces_m"], dtype=np.float64)
        source = np.asarray(state["source_cell_indices"], dtype=np.int64)
        if departure.shape != source.shape:
            raise SemigroupAuditError("topology-event state has inconsistent departure/source arrays")
        for face in sorted(changed_faces):
            if face < 0 or face >= departure.size:
                continue
            source_cell = int(source[face])
            lower = float(edges[source_cell])
            upper = float(edges[source_cell + 1])
            radius = float(departure[face])
            lower_distance = radius - lower
            upper_distance = radius - upper
            signed = lower_distance if abs(lower_distance) <= abs(upper_distance) else upper_distance
            width = upper - lower
            rows.append({
                "route": label,
                "face_index": face,
                "departure_radius_m": radius,
                "source_cell": source_cell,
                "source_cell_lower_m": lower,
                "source_cell_upper_m": upper,
                "delta_R_event_signed_m": signed,
                "delta_R_event_m": abs(signed),
                "local_delta_R_cell_m": width,
                "eta_signed": signed / width,
                "eta_abs": abs(signed) / width,
                "topology_signature": str(state["remap_topology_signature"]),
            })
    return rows


def event_error_localization(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    edges_m: NDArray[np.float64],
) -> dict[str, Any]:
    """Measure what fraction of a pairwise physical difference is event-local."""

    topology = topology_comparison(left, right)
    left_cells = np.asarray(left["population_array"], dtype=np.float64)
    right_cells = np.asarray(right["population_array"], dtype=np.float64)
    if left_cells.shape != right_cells.shape:
        raise SemigroupAuditError("event localization requires matching population grids")
    count = left_cells.size
    mask = np.zeros(count, dtype=bool)
    left_source = np.asarray(left["source_cell_indices"], dtype=np.int64)
    right_source = np.asarray(right["source_cell_indices"], dtype=np.int64)
    for face in np.asarray(topology["changed_face_indices"], dtype=np.int64):
        candidates = [int(face) - 1, int(face)]
        if face < left_source.size:
            candidates.append(int(left_source[face]))
        if face < right_source.size:
            candidates.append(int(right_source[face]))
        for cell in candidates:
            for adjacent in (cell - 1, cell, cell + 1):
                if 0 <= adjacent < count:
                    mask[adjacent] = True
    absolute_difference = np.abs(left_cells - right_cells)
    total_l1 = float(np.sum(absolute_difference, dtype=np.float64))
    local_l1 = float(np.sum(absolute_difference[mask], dtype=np.float64))
    # Shape is (moment, cell); the public helper exactly matches the CR1
    # cell-integrated measure.
    moments = np.asarray(
        cell_moments_from_piecewise_constant_cells(
            np.asarray(edges_m, dtype=np.float64), absolute_difference
        ),
        dtype=np.float64,
    )
    total_moments = np.sum(moments, axis=1, dtype=np.float64)
    local_moments = np.sum(moments[:, mask], axis=1, dtype=np.float64)
    return {
        "changed_face_count": int(topology["changed_face_count"]),
        "event_neighborhood_cell_count": int(np.count_nonzero(mask)),
        "event_neighborhood_population_L1_fraction": (
            local_l1 / total_l1 if total_l1 > 0.0 else 0.0
        ),
        "event_neighborhood_M0_fraction": float(local_moments[0] / total_moments[0]) if total_moments[0] > 0.0 else 0.0,
        "event_neighborhood_M1_fraction": float(local_moments[1] / total_moments[1]) if total_moments[1] > 0.0 else 0.0,
        "event_neighborhood_M2_fraction": float(local_moments[2] / total_moments[2]) if total_moments[2] > 0.0 else 0.0,
        "event_neighborhood_M3_fraction": float(local_moments[3] / total_moments[3]) if total_moments[3] > 0.0 else 0.0,
        "event_neighborhood_cell_indices": np.flatnonzero(mask).astype(np.int64),
    }
