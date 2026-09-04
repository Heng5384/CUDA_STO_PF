"""Read-only CR1 nonlinear-closure pathology diagnostics.

This module deliberately contains no acceptance rule, root selection, timestep
controller, or state commit.  It exposes the already-frozen CR1 map
``F(x) = T(x) - x`` from an immutable accepted state so a diagnostic runner
can distinguish a numerical closure problem from a trajectory problem without
quietly repairing either one.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .characteristic_reference import CharacteristicReferenceSolver
from .diagnostics import discrete_wasserstein_distance
from .population_metrics import (
    cell_moments_from_piecewise_constant_cells,
    metrics_from_piecewise_constant_cells,
    positive_cell_quadrature,
)


class ClosurePathologyError(RuntimeError):
    """A diagnostic contract error; never an invitation to accept a trial."""


@dataclass(frozen=True)
class DiagnosticPrestate:
    """A copy of one accepted CR1 state used only as immutable map input."""

    old_cell_number_m3: NDArray[np.float64]
    x_start: float
    state_hash: str
    step: int
    time_s: float


@dataclass(frozen=True)
class ClosureMapEvaluation:
    """One non-mutating evaluation of the frozen CR1 scalar map."""

    x_trial: float
    x_closure: float
    signed_f: float
    x_tolerance: float
    midpoint_xb: float
    cell_number_m3: NDArray[np.float64]
    cdf: NDArray[np.float64]
    cdf_signature: str
    population_hash: str
    departure_signature: str
    remap_topology_signature: str
    trace_topology_signature: str
    trace_topology_mode: str
    departure_faces_m: NDArray[np.float64]
    source_cell_indices: NDArray[np.int64]
    M0_m3: float
    M1_m2: float
    M2_m: float
    M3_dimensionless: float
    Rmean_m: float
    Rmean3_m3: float
    Sv_m_inv: float
    f_beta: float
    Q_beta_mol_m3: float
    Q_matrix_mol_m3: float
    Q_total_mol_m3: float
    inventory_relative_residual: float
    cell_measure: float
    critical_radius_m: float | None
    lower_tail_M0_m3: float
    lower_tail_M3_dimensionless: float


def _hash_parts(parts: Iterable[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, value in parts:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(repr(tuple(array.shape)).encode("ascii"))
            digest.update(array.tobytes())
        elif isinstance(value, (float, np.floating)):
            digest.update(float(value).hex().encode("ascii"))
        elif isinstance(value, (int, np.integer, bool, np.bool_)):
            digest.update(repr(int(value)).encode("ascii"))
        elif value is None:
            digest.update(b"NONE")
        else:
            digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def population_hash(cell_number_m3: NDArray[np.float64] | np.ndarray) -> str:
    """Return a dtype- and shape-bound deterministic population digest."""

    cells = np.asarray(cell_number_m3, dtype=np.float64)
    return _hash_parts((("cell_number_m3", cells),))


def solver_state_hash(solver: CharacteristicReferenceSolver) -> str:
    """Hash physical accepted arrays, independent from diagnostic history."""

    arrays = solver.state_arrays()
    return _hash_parts((key, np.asarray(arrays[key])) for key in sorted(arrays))


def _immutable_snapshot(
    solver: CharacteristicReferenceSolver,
) -> tuple[dict[str, NDArray[np.generic]], tuple[Any, ...], str]:
    arrays = {key: np.asarray(value).copy() for key, value in solver.state_arrays().items()}
    return arrays, tuple(solver.history), solver_state_hash(solver)


def _assert_immutable(
    solver: CharacteristicReferenceSolver,
    before_arrays: Mapping[str, NDArray[np.generic]],
    before_history: tuple[Any, ...],
    before_hash: str,
) -> None:
    current = solver.state_arrays()
    arrays_match = set(current) == set(before_arrays) and all(
        np.array_equal(np.asarray(current[name]), np.asarray(before_arrays[name]))
        for name in before_arrays
    )
    if not arrays_match or tuple(solver.history) != before_history or solver_state_hash(solver) != before_hash:
        raise ClosurePathologyError(
            "read-only closure-map evaluation mutated an accepted CR1 state"
        )


def capture_prestate(solver: CharacteristicReferenceSolver) -> DiagnosticPrestate:
    """Freeze an accepted state before a candidate closure attempt."""

    cells = np.asarray(solver._beta_cell_numbers(), dtype=np.float64).copy()
    if not np.all(np.isfinite(cells)) or np.any(cells < 0.0):
        raise ClosurePathologyError("cannot diagnose a non-finite or negative accepted population")
    return DiagnosticPrestate(
        old_cell_number_m3=cells,
        x_start=float(solver.matrix_xb),
        state_hash=solver_state_hash(solver),
        step=int(solver.step),
        time_s=float(solver.time_s),
    )


def _critical_radius_m(
    solver: CharacteristicReferenceSolver, *, matrix_xb: float
) -> float | None:
    """Locate the frozen growth-sign transition without changing the state."""

    beta = solver.population("beta")
    lower = float(beta.grid.edges_m[0])
    upper = float(beta.grid.edges_m[-1])

    def residual(radius: float) -> float:
        equilibrium = solver.equilibrium_adapter.equilibrium_xb(
            np.asarray([radius], dtype=np.float64), beta.parameters
        )
        return float(matrix_xb) - float(equilibrium[0])

    low_value, high_value = residual(lower), residual(upper)
    if low_value == 0.0:
        return lower
    if high_value == 0.0:
        return upper
    if low_value * high_value > 0.0:
        return None
    for _ in range(128):
        middle = 0.5 * (lower + upper)
        value = residual(middle)
        if value == 0.0:
            return middle
        if low_value * value < 0.0:
            upper, high_value = middle, value
        else:
            lower, low_value = middle, value
    return 0.5 * (lower + upper)


def _cdf(cells: NDArray[np.float64]) -> NDArray[np.float64]:
    total = float(np.sum(cells, dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        return np.zeros(cells.shape, dtype=np.float64)
    return np.cumsum(cells, dtype=np.float64) / total


def _lower_tail_moments(
    solver: CharacteristicReferenceSolver,
    cells: NDArray[np.float64],
    *,
    critical_radius_m: float | None,
) -> tuple[float, float]:
    """Report resolved cells at or below the instantaneous critical radius.

    This is an explicitly geometric diagnostic, not an identity or a source
    term.  If no sign transition exists, the lower-tail contribution is zero.
    """

    if critical_radius_m is None:
        return (0.0, 0.0)
    beta = solver.population("beta")
    mask = np.asarray(beta.grid.centres_m <= critical_radius_m, dtype=bool)
    moments = cell_moments_from_piecewise_constant_cells(beta.grid.edges_m, cells)
    return (
        float(np.sum(moments[0, mask], dtype=np.float64)),
        float(np.sum(moments[3, mask], dtype=np.float64)),
    )


def _topology_signatures(
    solver: CharacteristicReferenceSolver, trial: Any
) -> tuple[str, str, str, NDArray[np.int64]]:
    partition = solver._cdf_source_partition(trial)
    topology = solver._trace_topology(trial)
    if partition is None:
        source_indices = np.empty(0, dtype=np.int64)
        partition_signature = "UNAVAILABLE"
    else:
        source_indices = np.asarray(partition.source_cell_indices, dtype=np.int64).copy()
        partition_signature = _hash_parts(
            (
                ("source_cell_indices", source_indices),
                ("lower_endpoint_mask", np.asarray(partition.lower_endpoint_mask)),
                ("upper_endpoint_mask", np.asarray(partition.upper_endpoint_mask)),
                ("identity_departure_map", partition.identity_departure_map),
                ("lower_no_inflow", partition.lower_no_inflow_face_count),
                ("upper_no_inflow", partition.upper_no_inflow_face_count),
            )
        )
    if topology is None:
        return partition_signature, "UNAVAILABLE", "UNAVAILABLE", source_indices
    trace_signature = _hash_parts(
        (
            ("mode", topology.mode),
            ("node_sign", np.asarray(topology.node_sign)),
            ("gauss_left_sign", np.asarray(topology.gauss_left_sign)),
            ("gauss_right_sign", np.asarray(topology.gauss_right_sign)),
            ("valid_interval", np.asarray(topology.valid_interval)),
            ("arrival_face_run_id", np.asarray(topology.arrival_face_run_id)),
            ("lower_no_inflow", topology.lower_no_inflow_face_count),
            ("upper_no_inflow", topology.upper_no_inflow_face_count),
            ("identity_departure_map", topology.identity_departure_map),
        )
    )
    combined = _hash_parts(
        (("partition", partition_signature), ("trace", trace_signature))
    )
    return combined, trace_signature, str(topology.mode), source_indices


def evaluation_from_recorded_trial(
    solver: CharacteristicReferenceSolver,
    *,
    trial: Any,
    old_cell_number_m3: NDArray[np.float64] | np.ndarray,
) -> ClosureMapEvaluation:
    """Flatten an already-run, uncommitted CR1 trial into diagnostic telemetry.

    This is used by a recording subclass immediately around a normal accepted
    baseline step.  It does not re-evaluate a map or alter the current solver.
    The grid/parameters are immutable across a CR1 step, so they remain the
    correct metadata source even after the accepted state has committed.
    """

    old_cells = np.asarray(old_cell_number_m3, dtype=np.float64)
    combined_signature, trace_signature, topology_mode, source_indices = _topology_signatures(
        solver, trial
    )
    candidate = np.asarray(trial.cell_number_m3, dtype=np.float64).copy()
    metrics = metrics_from_piecewise_constant_cells(
        solver.population("beta").grid.edges_m, candidate
    )
    cdf = _cdf(candidate)
    critical = _critical_radius_m(solver, matrix_xb=float(trial.matrix_xb))
    lower_m0, lower_m3 = _lower_tail_moments(
        solver, candidate, critical_radius_m=critical
    )
    departure = np.asarray(trial.trace.departure_faces_m, dtype=np.float64).copy()
    return ClosureMapEvaluation(
        x_trial=float(trial.x_guess),
        x_closure=float(trial.matrix_xb),
        signed_f=float(trial.signed_xb_residual),
        x_tolerance=float(trial.xb_tolerance),
        midpoint_xb=float(trial.midpoint_matrix_xb),
        cell_number_m3=candidate,
        cdf=cdf,
        cdf_signature=_hash_parts((("cdf", cdf),)),
        population_hash=population_hash(candidate),
        departure_signature=_hash_parts((("departure_faces_m", departure),)),
        remap_topology_signature=combined_signature,
        trace_topology_signature=trace_signature,
        trace_topology_mode=topology_mode,
        departure_faces_m=departure,
        source_cell_indices=source_indices,
        M0_m3=float(metrics.M0_m3),
        M1_m2=float(metrics.M1_m2),
        M2_m=float(metrics.M2_m),
        M3_dimensionless=float(metrics.M3_dimensionless),
        Rmean_m=float(metrics.Rmean_number_m),
        Rmean3_m3=float(metrics.Rmean_cubed_m3),
        Sv_m_inv=float(metrics.Sv_m_inv),
        f_beta=float(metrics.f_beta),
        Q_beta_mol_m3=float(trial.inventory.beta_resolved_mol_m3),
        Q_matrix_mol_m3=float(trial.inventory.matrix_mol_m3),
        Q_total_mol_m3=float(trial.inventory.total_mol_m3),
        inventory_relative_residual=float(trial.inventory.relative_residual),
        cell_measure=float(
            CharacteristicReferenceSolver._cell_measure_relative_residual(candidate, old_cells)
        ),
        critical_radius_m=critical,
        lower_tail_M0_m3=lower_m0,
        lower_tail_M3_dimensionless=lower_m3,
    )


def evaluate_closure_map(
    solver: CharacteristicReferenceSolver,
    *,
    prestate: DiagnosticPrestate,
    dt_s: float,
    x_trial: float,
) -> ClosureMapEvaluation:
    """Evaluate the frozen CR1 map from an immutable accepted pre-state.

    The function is intentionally not an acceptance mechanism.  It checks the
    accepted state before and after the private CR1 trial and raises if the
    supposed read-only evaluation has a side effect.
    """

    if not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
        raise ValueError("diagnostic dt_s must be finite and positive")
    if not math.isfinite(float(x_trial)) or not 0.0 <= float(x_trial) <= 1.0:
        raise ValueError("diagnostic x_trial must lie in the full physical [0, 1] interval")
    if solver_state_hash(solver) != prestate.state_hash:
        raise ClosurePathologyError("solver no longer matches the frozen diagnostic pre-state")
    if float(solver.matrix_xb) != float(prestate.x_start):
        raise ClosurePathologyError("diagnostic pre-state matrix composition differs")
    current_cells = np.asarray(solver._beta_cell_numbers(), dtype=np.float64)
    if not np.array_equal(current_cells, prestate.old_cell_number_m3):
        raise ClosurePathologyError("diagnostic pre-state population differs")

    before_arrays, before_history, before_hash = _immutable_snapshot(solver)
    try:
        trial = solver._evaluate_closure_trial(
            old_cell_number_m3=np.asarray(prestate.old_cell_number_m3, dtype=np.float64),
            dt_s=float(dt_s),
            x_start=float(prestate.x_start),
            x_guess=float(x_trial),
        )
        return evaluation_from_recorded_trial(
            solver, trial=trial, old_cell_number_m3=prestate.old_cell_number_m3
        )
    finally:
        _assert_immutable(solver, before_arrays, before_history, before_hash)


def state_summary(
    solver: CharacteristicReferenceSolver,
    *,
    fraction: Fraction,
    m: int,
    substep: int,
    evaluation: ClosureMapEvaluation | None = None,
) -> dict[str, Any]:
    """Return compact accepted-state diagnostics suitable for CSV/NPZ indexes."""

    cells = np.asarray(solver._beta_cell_numbers(), dtype=np.float64).copy()
    metrics = metrics_from_piecewise_constant_cells(solver.population("beta").grid.edges_m, cells)
    inventory = solver.ledger.snapshot(
        matrix_xb=solver.matrix_xb,
        populations=solver.population_list(),
        beta_resolved_fraction=1.0,
    )
    critical = _critical_radius_m(solver, matrix_xb=float(solver.matrix_xb))
    lower_m0, lower_m3 = _lower_tail_moments(solver, cells, critical_radius_m=critical)
    return {
        "m": int(m),
        "substep": int(substep),
        "fraction_numerator": int(fraction.numerator),
        "fraction_denominator": int(fraction.denominator),
        "fraction": float(fraction),
        "physical_time_s": float(solver.time_s),
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
        "inventory_residual_mol_m3": float(inventory.residual_mol_m3),
        "inventory_relative_residual": float(inventory.relative_residual),
        "state_hash": solver_state_hash(solver),
        "population_array_hash": population_hash(cells),
        "cdf_signature": _hash_parts((("cdf", _cdf(cells)),)),
        "departure_signature": "NOT_RECONSTRUCTED" if evaluation is None else evaluation.departure_signature,
        "remap_topology_signature": "NOT_RECONSTRUCTED" if evaluation is None else evaluation.remap_topology_signature,
        "trace_topology_signature": "NOT_RECONSTRUCTED" if evaluation is None else evaluation.trace_topology_signature,
        "trace_topology_mode": "NOT_RECONSTRUCTED" if evaluation is None else evaluation.trace_topology_mode,
        "critical_radius_m": critical,
        "lower_tail_M0_m3": lower_m0,
        "lower_tail_M3_dimensionless": lower_m3,
        "population_array": cells,
        "cdf": _cdf(cells),
    }


def state_distance(
    left: Mapping[str, Any], right: Mapping[str, Any], *, edges_m: NDArray[np.float64]
) -> dict[str, float]:
    """Compute continuous and full-population distances between accepted states."""

    left_cells = np.asarray(left["population_array"], dtype=np.float64)
    right_cells = np.asarray(right["population_array"], dtype=np.float64)
    difference = left_cells - right_cells
    denominator = max(float(np.sum(np.abs(right_cells), dtype=np.float64)), 1.0e-300)
    l1 = float(np.sum(np.abs(difference), dtype=np.float64) / denominator)
    linf = float(
        np.max(np.abs(difference)) / max(float(np.max(np.abs(right_cells))), 1.0e-300)
    )
    left_radii, left_weights = positive_cell_quadrature(edges_m, left_cells, 2)
    right_radii, right_weights = positive_cell_quadrature(edges_m, right_cells, 2)
    return {
        "population_L1": l1,
        "population_Linf": linf,
        "PSD_Wasserstein_m": float(
            discrete_wasserstein_distance(left_radii, left_weights, right_radii, right_weights)
        ),
        "CDF_difference": float(
            np.max(np.abs(np.asarray(left["cdf"]) - np.asarray(right["cdf"])))
        ),
    }


def common_time_comparisons(
    states_by_m: Mapping[int, Sequence[Mapping[str, Any]]], *, edges_m: NDArray[np.float64]
) -> list[dict[str, Any]]:
    """Compare every pair of accepted states at exactly shared rational times."""

    indexed: dict[int, dict[Fraction, Mapping[str, Any]]] = {}
    for m, states in states_by_m.items():
        indexed[int(m)] = {
            Fraction(int(item["fraction_numerator"]), int(item["fraction_denominator"])): item
            for item in states
        }
    rows: list[dict[str, Any]] = []
    multiplicities = sorted(indexed)
    for offset, left_m in enumerate(multiplicities):
        for right_m in multiplicities[offset + 1 :]:
            for fraction in sorted(set(indexed[left_m]).intersection(indexed[right_m])):
                left, right = indexed[left_m][fraction], indexed[right_m][fraction]
                distances = state_distance(left, right, edges_m=edges_m)
                metric_errors = {
                    metric: abs(float(left[metric]) - float(right[metric]))
                    / max(abs(float(right[metric])), 1.0e-300)
                    for metric in (
                        "xB", "M0_m3", "M1_m2", "M2_m", "M3_dimensionless", "Rmean_m",
                        "Rmean3_m3", "Sv_m_inv", "f_beta", "Q_beta_mol_m3", "Q_matrix_mol_m3",
                        "Q_total_mol_m3", "lower_tail_M0_m3", "lower_tail_M3_dimensionless",
                    )
                }
                rows.append({
                    "left_m": left_m,
                    "right_m": right_m,
                    "fraction_numerator": fraction.numerator,
                    "fraction_denominator": fraction.denominator,
                    "fraction": float(fraction),
                    "left_physical_time_s": left["physical_time_s"],
                    "right_physical_time_s": right["physical_time_s"],
                    "topology_equal": (
                        left["remap_topology_signature"] == right["remap_topology_signature"]
                    ),
                    "departure_signature_equal": (
                        left["departure_signature"] == right["departure_signature"]
                    ),
                    **metric_errors,
                    **distances,
                })
    return rows


def evaluation_row(
    evaluation: ClosureMapEvaluation,
    *,
    iteration: int | None = None,
    previous: ClosureMapEvaluation | None = None,
    label: str = "",
) -> dict[str, Any]:
    """Flatten a map evaluation while retaining exact binary64 evidence."""

    if previous is None:
        population_l1 = math.nan
        population_linf = math.nan
        cell_measure = math.nan
    else:
        delta = evaluation.cell_number_m3 - previous.cell_number_m3
        denominator = max(float(np.sum(np.abs(previous.cell_number_m3), dtype=np.float64)), 1.0e-300)
        population_l1 = float(np.sum(np.abs(delta), dtype=np.float64) / denominator)
        population_linf = float(
            np.max(np.abs(delta))
            / max(float(np.max(np.abs(previous.cell_number_m3))), 1.0e-300)
        )
        cell_measure = CharacteristicReferenceSolver._cell_measure_relative_residual(
            evaluation.cell_number_m3, previous.cell_number_m3
        )
    return {
        "label": label,
        "iteration": "" if iteration is None else int(iteration),
        "x_trial": evaluation.x_trial,
        "x_trial_hex": evaluation.x_trial.hex(),
        "x_closure": evaluation.x_closure,
        "x_closure_hex": evaluation.x_closure.hex(),
        "signed_F": evaluation.signed_f,
        "signed_F_hex": evaluation.signed_f.hex(),
        "absF": abs(evaluation.signed_f),
        "x_tolerance": evaluation.x_tolerance,
        "M0_m3": evaluation.M0_m3,
        "M1_m2": evaluation.M1_m2,
        "M2_m": evaluation.M2_m,
        "M3_dimensionless": evaluation.M3_dimensionless,
        "Rmean_m": evaluation.Rmean_m,
        "Rmean3_m3": evaluation.Rmean3_m3,
        "Sv_m_inv": evaluation.Sv_m_inv,
        "f_beta": evaluation.f_beta,
        "Q_beta_mol_m3": evaluation.Q_beta_mol_m3,
        "Q_matrix_mol_m3": evaluation.Q_matrix_mol_m3,
        "Q_total_mol_m3": evaluation.Q_total_mol_m3,
        "inventory_relative_residual": evaluation.inventory_relative_residual,
        "population_L1_from_previous": population_l1,
        "population_Linf_from_previous": population_linf,
        "cell_measure_from_previous": cell_measure,
        "CDF_signature": evaluation.cdf_signature,
        "departure_signature": evaluation.departure_signature,
        "topology_signature": evaluation.remap_topology_signature,
        "trace_topology_signature": evaluation.trace_topology_signature,
        "trace_topology_mode": evaluation.trace_topology_mode,
        "critical_radius_m": evaluation.critical_radius_m,
        "lower_tail_M0_m3": evaluation.lower_tail_M0_m3,
        "lower_tail_M3_dimensionless": evaluation.lower_tail_M3_dimensionless,
        "population_hash": evaluation.population_hash,
    }


def _exact_period(keys: Sequence[tuple[str, ...]], *, maximum_period: int = 32) -> int | None:
    for period in range(1, min(maximum_period, len(keys) // 2) + 1):
        if list(keys[-period:]) == list(keys[-2 * period : -period]):
            return period
    return None


def _trace_classification(rows: Sequence[Mapping[str, Any]], keys: Sequence[tuple[str, ...]]) -> dict[str, Any]:
    period = _exact_period(keys)
    observed_periods = [
        value for value in range(1, min(32, len(keys) // 2) + 1)
        if list(keys[-value:]) == list(keys[-2 * value : -value])
    ]
    residuals = [abs(float(row["signed_F"])) for row in rows]
    signatures = {str(row["topology_signature"]) for row in rows}
    signs = [math.copysign(1.0, float(row["signed_F"])) for row in rows if float(row["signed_F"]) != 0.0]
    sign_changes = sum(left != right for left, right in zip(signs, signs[1:]))
    monotone = all(right <= left for left, right in zip(residuals, residuals[1:]))
    if period is not None and period >= 5:
        classification = "P5_PLUS_PERIOD"
    elif period in (2, 3, 4):
        classification = f"P{period}"
    elif period == 1:
        classification = "MONOTONE_CONTRACTION"
    elif len(signatures) > 1:
        classification = "TOPOLOGY_SWITCHING"
    elif monotone and len(residuals) >= 2 and residuals[-1] < residuals[0]:
        classification = "MONOTONE_CONTRACTION"
    elif len(residuals) >= 2 and residuals[-1] < residuals[0]:
        classification = "SLOW_CONTRACTION"
    elif len(set(keys[-min(8, len(keys)) :])) < min(8, len(keys)):
        classification = "FLOATING_STAGNATION"
    elif sign_changes:
        classification = "OSCILLATORY_APERIODIC"
    else:
        classification = "NONCONTRACTIVE"
    return {
        "classification": classification,
        "exact_period": period,
        "observed_exact_periods": observed_periods,
        "topology_signature_count": len(signatures),
        "sign_change_count": sign_changes,
        "initial_absF": None if not residuals else residuals[0],
        "final_absF": None if not residuals else residuals[-1],
        "monotone_absF": monotone,
    }


def raw_picard_trace(
    solver: CharacteristicReferenceSolver,
    *,
    prestate: DiagnosticPrestate,
    dt_s: float,
    maximum_iterations: int,
    stop_after_first_exact_period: bool = False,
    label: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any], list[ClosureMapEvaluation]]:
    """Record raw Picard iterations without invoking a closure or root path."""

    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    rows: list[dict[str, Any]] = []
    evaluations: list[ClosureMapEvaluation] = []
    keys: list[tuple[str, ...]] = []
    x_guess = float(prestate.x_start)
    previous: ClosureMapEvaluation | None = None
    for iteration in range(1, int(maximum_iterations) + 1):
        evaluation = evaluate_closure_map(
            solver, prestate=prestate, dt_s=dt_s, x_trial=x_guess
        )
        rows.append(evaluation_row(evaluation, iteration=iteration, previous=previous, label=label))
        evaluations.append(evaluation)
        keys.append((
            evaluation.x_trial.hex(), evaluation.x_closure.hex(), evaluation.population_hash,
            evaluation.cdf_signature, evaluation.remap_topology_signature,
        ))
        period = _exact_period(keys)
        if stop_after_first_exact_period and period is not None and period >= 2:
            break
        previous = evaluation
        x_guess = float(evaluation.x_closure)
    summary = _trace_classification(rows, keys)
    summary.update({"iterations_recorded": len(rows), "terminated_on_exact_period": bool(
        stop_after_first_exact_period and summary["exact_period"] is not None
    )})
    return rows, summary, evaluations


def full_physical_x_interval() -> tuple[float, float]:
    """The full matrix-composition/thermodynamic candidate interval.

    ``x_trial`` is a candidate matrix composition, so its only universal
    admissibility bounds in the frozen pseudo-binary contract are [0, 1].
    The map evaluator independently rejects a trace or inventory that cannot
    exist for a particular candidate; scan errors remain observations.
    """

    return (0.0, 1.0)


def _scan_row_from_error(*, x_trial: float, error: Exception) -> dict[str, Any]:
    return {
        "x_trial": float(x_trial),
        "x_trial_hex": float(x_trial).hex(),
        "status": "EVALUATION_ERROR",
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _intervals_intersect(
    left: float, right: float, intervals: Sequence[tuple[float, float]]
) -> bool:
    low, high = sorted((float(left), float(right)))
    return any(max(low, min(a, b)) <= min(high, max(a, b)) for a, b in intervals)


def _local_topology_transition_audit(
    solver: CharacteristicReferenceSolver,
    *,
    prestate: DiagnosticPrestate,
    dt_s: float,
    left: ClosureMapEvaluation,
    right: ClosureMapEvaluation,
    raw_update_intervals: Sequence[tuple[float, float]],
    changed_face_count: int,
    refinement_levels: int = 8,
) -> dict[str, Any]:
    """Narrow one observed topology switch without selecting a scalar root.

    The fixed bisection here follows the *topology label*, not F's sign; it
    never feeds a bracket to a solver.  A smooth finite-slope map has a
    one-sided F difference that contracts with the x interval.  This audit
    records whether an observed difference instead persists over eight fixed
    refinements.  It is evidence, not a mathematical discontinuity proof.
    """

    initial_left, initial_right = left, right
    current_left, current_right = left, right
    left_signature = left.remap_topology_signature
    right_signature = right.remap_topology_signature
    history: list[dict[str, Any]] = []
    status = "COMPLETED"
    levels_completed = 0
    for level in range(int(refinement_levels) + 1):
        width = float(current_right.x_trial - current_left.x_trial)
        delta_f = float(current_right.signed_f - current_left.signed_f)
        history.append({
            "level": level,
            "x_left": current_left.x_trial,
            "x_right": current_right.x_trial,
            "width": width,
            "F_left": current_left.signed_f,
            "F_right": current_right.signed_f,
            "delta_F": delta_f,
            "left_topology_signature": current_left.remap_topology_signature,
            "right_topology_signature": current_right.remap_topology_signature,
        })
        if level == int(refinement_levels):
            levels_completed = level
            break
        midpoint = 0.5 * (current_left.x_trial + current_right.x_trial)
        if midpoint == current_left.x_trial or midpoint == current_right.x_trial:
            status = "BINARY64_MIDPOINT_COALESCED"
            levels_completed = level
            break
        try:
            middle = evaluate_closure_map(
                solver, prestate=prestate, dt_s=dt_s, x_trial=midpoint
            )
        except Exception as error:  # preserve the diagnostic observation
            status = f"LOCAL_EVALUATION_ERROR:{type(error).__name__}"
            levels_completed = level
            break
        if middle.remap_topology_signature == left_signature:
            current_left = middle
        elif middle.remap_topology_signature == right_signature:
            current_right = middle
        else:
            status = "INTERMEDIATE_TOPOLOGY_OBSERVED"
            levels_completed = level + 1
            history.append({
                "level": level + 1,
                "x_left": middle.x_trial,
                "x_right": middle.x_trial,
                "width": 0.0,
                "F_left": middle.signed_f,
                "F_right": middle.signed_f,
                "delta_F": 0.0,
                "left_topology_signature": middle.remap_topology_signature,
                "right_topology_signature": middle.remap_topology_signature,
            })
            break
    initial_width = float(initial_right.x_trial - initial_left.x_trial)
    final_width = float(current_right.x_trial - current_left.x_trial)
    initial_delta = abs(float(initial_right.signed_f - initial_left.signed_f))
    final_delta = abs(float(current_right.signed_f - current_left.signed_f))
    raw_intersects = _intervals_intersect(
        initial_left.x_trial, initial_right.x_trial, raw_update_intervals
    )
    same_trace_topology = (
        initial_left.trace_topology_signature == initial_right.trace_topology_signature
    )
    endpoint_repeatable: bool | None = None
    if raw_intersects and changed_face_count == 1 and same_trace_topology:
        _repeat_rows, endpoint_repeatable = repeatability_audit(
            solver,
            prestate=prestate,
            dt_s=dt_s,
            x_trials=(current_left.x_trial, current_right.x_trial),
            repeats=10,
        )
    retained_fraction = final_delta / max(initial_delta, np.finfo(np.float64).tiny)
    binary64_noise = 32.0 * np.finfo(np.float64).eps
    persistent = bool(
        status == "COMPLETED"
        and levels_completed >= 6
        and initial_width > 0.0
        and final_width <= initial_width / 64.0
        and final_delta > binary64_noise
        and retained_fraction >= 0.25
    )
    return {
        "raw_update_intersects_transition_interval": raw_intersects,
        "same_trace_topology": same_trace_topology,
        "local_refinement_status": status,
        "local_refinement_levels_completed": levels_completed,
        "local_initial_width": initial_width,
        "local_final_width": final_width,
        "local_initial_delta_F": initial_delta,
        "local_final_delta_F": final_delta,
        "local_delta_retained_fraction": retained_fraction,
        "local_delta_above_binary64_noise": final_delta > binary64_noise,
        "local_one_sided_F_delta_persistent": persistent,
        "local_endpoint_bitwise_repeatable": endpoint_repeatable,
        "local_refinement_history": history,
    }


def scalar_map_scan(
    solver: CharacteristicReferenceSolver,
    *,
    prestate: DiagnosticPrestate,
    dt_s: float,
    coarse_points: int = 256,
    anchor_x: Sequence[float] = (),
    raw_update_intervals: Sequence[tuple[float, float]] = (),
    label: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Coarsely and adaptively sample a map without selecting a root.

    Refinement samples only quantify observed sign/topology transitions.  No
    bracketing result is fed back into Picard or into a physical step.
    """

    if coarse_points < 2:
        raise ValueError("coarse scalar scan needs at least two points")
    low, high = full_physical_x_interval()
    samples = {float(value) for value in np.linspace(low, high, int(coarse_points), dtype=np.float64)}
    for value in anchor_x:
        if math.isfinite(float(value)) and low <= float(value) <= high:
            samples.add(float(value))
    rows_by_x: dict[float, dict[str, Any]] = {}
    evaluations_by_x: dict[float, ClosureMapEvaluation] = {}

    def evaluate_samples(values: Iterable[float]) -> None:
        for x_value in values:
            if x_value in rows_by_x:
                continue
            try:
                evaluation = evaluate_closure_map(
                    solver, prestate=prestate, dt_s=dt_s, x_trial=x_value
                )
                row = evaluation_row(evaluation, label=label)
                row["status"] = "OK"
                evaluations_by_x[x_value] = evaluation
            except Exception as error:  # data row, not a recovery path
                row = _scan_row_from_error(x_trial=x_value, error=error)
                row["label"] = label
            rows_by_x[x_value] = row

    evaluate_samples(samples)
    ordered = [rows_by_x[key] for key in sorted(rows_by_x)]
    adaptive: set[float] = set()
    for left, right in zip(ordered, ordered[1:]):
        if left.get("status") != "OK" or right.get("status") != "OK":
            continue
        left_f, right_f = float(left["signed_F"]), float(right["signed_F"])
        topology_change = str(left["topology_signature"]) != str(right["topology_signature"])
        sign_change = left_f == 0.0 or right_f == 0.0 or left_f * right_f < 0.0
        if topology_change or sign_change:
            left_x, right_x = float(left["x_trial"]), float(right["x_trial"])
            adaptive.update((
                left_x + 0.25 * (right_x - left_x),
                0.5 * (left_x + right_x),
                left_x + 0.75 * (right_x - left_x),
            ))
    for value in anchor_x:
        if math.isfinite(float(value)) and low <= float(value) <= high:
            adaptive.add(float(np.nextafter(float(value), low)))
            adaptive.add(float(np.nextafter(float(value), high)))
    evaluate_samples(value for value in adaptive if low <= value <= high)
    rows = [rows_by_x[key] for key in sorted(rows_by_x)]

    transition_rows: list[dict[str, Any]] = []
    valid_deltas = [
        abs(float(right["signed_F"]) - float(left["signed_F"]))
        for left, right in zip(rows, rows[1:])
        if left.get("status") == "OK" and right.get("status") == "OK"
        and left.get("topology_signature") == right.get("topology_signature")
    ]
    baseline_delta = float(np.median(valid_deltas)) if valid_deltas else 0.0
    partition = 0
    previous_signature: str | None = None
    crossings_by_partition: dict[int, int] = {}
    observed_crossings = 0
    for row in rows:
        if row.get("status") != "OK":
            row["topology_partition"] = "UNAVAILABLE"
            previous_signature = None
            continue
        signature = str(row["topology_signature"])
        if previous_signature is None or signature != previous_signature:
            partition += 1
        row["topology_partition"] = partition
        previous_signature = signature
    for left, right in zip(rows, rows[1:]):
        if left.get("status") != "OK" or right.get("status") != "OK":
            continue
        left_f, right_f = float(left["signed_F"]), float(right["signed_F"])
        sign_change = left_f == 0.0 or right_f == 0.0 or left_f * right_f < 0.0
        topology_change = str(left["topology_signature"]) != str(right["topology_signature"])
        if sign_change:
            observed_crossings += 1
            crossings_by_partition[int(left["topology_partition"])] = (
                crossings_by_partition.get(int(left["topology_partition"]), 0) + 1
            )
        if topology_change:
            delta_f = right_f - left_f
            threshold = max(8.0 * baseline_delta, 32.0 * np.finfo(np.float64).eps)
            left_eval = evaluations_by_x[float(left["x_trial"])]
            right_eval = evaluations_by_x[float(right["x_trial"])]
            left_indices = np.asarray(left_eval.source_cell_indices, dtype=np.int64)
            right_indices = np.asarray(right_eval.source_cell_indices, dtype=np.int64)
            changed_faces = np.flatnonzero(left_indices != right_indices)
            changed_face_count = int(changed_faces.size)
            local_evidence = _local_topology_transition_audit(
                solver,
                prestate=prestate,
                dt_s=dt_s,
                left=left_eval,
                right=right_eval,
                raw_update_intervals=raw_update_intervals,
                changed_face_count=changed_face_count,
            )
            if changed_faces.size == 0:
                changed_faces = np.asarray([-1], dtype=np.int64)
            edges = np.asarray(solver.population("beta").grid.edges_m, dtype=np.float64)
            local_fields = {
                key: value for key, value in local_evidence.items()
                if key != "local_refinement_history"
            }
            for face_offset, face in enumerate(changed_faces):
                if int(face) < 0:
                    left_source = right_source = "UNAVAILABLE"
                    left_departure = right_departure = math.nan
                    crossed_lower = crossed_upper = math.nan
                    endpoint_change = False
                else:
                    left_source = int(left_indices[int(face)])
                    right_source = int(right_indices[int(face)])
                    left_departure = float(left_eval.departure_faces_m[int(face)])
                    right_departure = float(right_eval.departure_faces_m[int(face)])
                    lower_index, upper_index = sorted((left_source, right_source))
                    crossed_lower = float(edges[lower_index])
                    crossed_upper = float(edges[min(upper_index + 1, edges.size - 1)])
                    endpoint_change = bool(
                        (left_departure == edges[0]) != (right_departure == edges[0])
                        or (left_departure == edges[-1]) != (right_departure == edges[-1])
                    )
                transition_rows.append({
                    "label": label,
                    "x_left": left["x_trial"],
                    "x_right": right["x_trial"],
                    "F_left": left_f,
                    "F_right": right_f,
                    "delta_F": delta_f,
                    "map_jump_observed": abs(delta_f) > threshold,
                    # A remap transition is closure-relevant only when its
                    # two locally sampled sides also straddle F=0.  Retain
                    # ordinary topology jumps as observations, but never let
                    # an unrelated global-[0,1] transition determine the
                    # diagnostic root-cause classification.
                    "sign_change_observed": sign_change,
                    "same_trace_topology": left["trace_topology_signature"] == right["trace_topology_signature"],
                    "source_partition_changed": True,
                    "changed_face_count": changed_face_count,
                    "single_departure_cell_crossing": changed_face_count == 1,
                    "left_topology_partition": left["topology_partition"],
                    "right_topology_partition": right["topology_partition"],
                    "face_index": int(face),
                    "departure_left_m": left_departure,
                    "departure_right_m": right_departure,
                    "source_cell_left": left_source,
                    "source_cell_right": right_source,
                    "crossed_edge_lower_m": crossed_lower,
                    "crossed_edge_upper_m": crossed_upper,
                    "endpoint_mask_changed": endpoint_change,
                    "departure_cell_crossing": int(face) >= 0,
                    **local_fields,
                    # A topology transition can change many departure faces.
                    # Keep its event-level narrowing history once, while every
                    # face still receives the same traceable event summary.
                    "local_refinement_history": (
                        local_evidence["local_refinement_history"]
                        if face_offset == 0 else "SEE_FIRST_FACE_OF_SAME_TRANSITION"
                    ),
                })
    summary = {
        "physical_x_interval": [low, high],
        "coarse_point_count": int(coarse_points),
        "total_point_count": len(rows),
        "NUMBER_OF_OBSERVED_SIGN_CROSSINGS": observed_crossings,
        "NUMBER_OF_TOPOLOGY_PARTITIONS": partition,
        "ROOTS_PER_TOPOLOGY_PARTITION": {
            str(key): value for key, value in sorted(crossings_by_partition.items())
        },
        "MAP_JUMP_LOCATIONS": [
            {"x_left": row["x_left"], "x_right": row["x_right"]}
            for row in transition_rows if bool(row["map_jump_observed"])
        ],
        "observed_crossings_are_not_accepted_roots": True,
    }
    return rows, transition_rows, summary


def pairwise_sum(values: NDArray[np.float64] | np.ndarray) -> float:
    """Deterministic binary-tree sum used only as a reduction shadow."""

    work = np.asarray(values, dtype=np.float64).ravel().copy()
    if work.size == 0:
        return 0.0
    while work.size > 1:
        even = work.size - work.size % 2
        paired = work[:even:2] + work[1:even:2]
        work = paired if even == work.size else np.concatenate((paired, work[-1:]))
    return float(work[0])


def neumaier_sum(values: NDArray[np.float64] | np.ndarray) -> float:
    """Compensated scalar sum used only for diagnosis."""

    total = 0.0
    compensation = 0.0
    for value in np.asarray(values, dtype=np.float64).ravel():
        candidate = total + float(value)
        if abs(total) >= abs(float(value)):
            compensation += (total - candidate) + float(value)
        else:
            compensation += (float(value) - candidate) + total
        total = candidate
    return total + compensation


def precision_reduction_shadow(
    solver: CharacteristicReferenceSolver,
    evaluation: ClosureMapEvaluation,
    *,
    previous_cells: NDArray[np.float64] | None = None,
) -> dict[str, Any]:
    """Compare reduction/inventory algebra without replacing CR1 arithmetic."""

    beta = solver.population("beta")
    cells = np.asarray(evaluation.cell_number_m3, dtype=np.float64)
    per_cell = cell_moments_from_piecewise_constant_cells(beta.grid.edges_m, cells)
    result: dict[str, Any] = {
        "x_trial": evaluation.x_trial,
        "x_trial_hex": evaluation.x_trial.hex(),
        "signed_F_float64": evaluation.signed_f,
        "signed_F_float64_hex": evaluation.signed_f.hex(),
        "longdouble_epsilon": float(np.finfo(np.longdouble).eps),
    }
    ordinary_moments: list[float] = []
    for order in range(4):
        values = np.asarray(per_cell[order], dtype=np.float64)
        ordinary = float(np.sum(values, dtype=np.float64))
        pairwise = pairwise_sum(values)
        compensated = neumaier_sum(values)
        ordinary_moments.append(ordinary)
        result.update({
            f"M{order}_ordinary": ordinary,
            f"M{order}_pairwise": pairwise,
            f"M{order}_neumaier": compensated,
            f"M{order}_max_reduction_delta": max(abs(ordinary - pairwise), abs(ordinary - compensated)),
            f"M{order}_max_reduction_relative_delta": max(
                abs(ordinary - pairwise), abs(ordinary - compensated)
            ) / max(abs(ordinary), 1.0e-300),
        })
    beta_parameters = beta.parameters
    q_beta_ordinary = (
        (4.0 * math.pi / 3.0) * ordinary_moments[3]
        * float(beta_parameters.x_b) / float(beta_parameters.molar_volume_m3_mol)
    )
    q_beta_pairwise = (
        (4.0 * math.pi / 3.0) * result["M3_pairwise"]
        * float(beta_parameters.x_b) / float(beta_parameters.molar_volume_m3_mol)
    )
    q_beta_compensated = (
        (4.0 * math.pi / 3.0) * result["M3_neumaier"]
        * float(beta_parameters.x_b) / float(beta_parameters.molar_volume_m3_mol)
    )
    result.update({
        "Q_beta_ordinary": q_beta_ordinary,
        "Q_beta_pairwise": q_beta_pairwise,
        "Q_beta_neumaier": q_beta_compensated,
        "Q_beta_max_reduction_delta": max(
            abs(q_beta_ordinary - q_beta_pairwise), abs(q_beta_ordinary - q_beta_compensated)
        ),
        "Q_beta_max_reduction_relative_delta": max(
            abs(q_beta_ordinary - q_beta_pairwise), abs(q_beta_ordinary - q_beta_compensated)
        ) / max(abs(q_beta_ordinary), 1.0e-300),
    })
    if previous_cells is not None:
        previous = np.asarray(previous_cells, dtype=np.float64)
        numerator = np.abs(cells - previous)
        denominator_values = np.concatenate((np.abs(cells), np.abs(previous)))
        ordinary_cell_measure = float(np.sum(numerator, dtype=np.float64)) / max(
            float(np.sum(denominator_values, dtype=np.float64)), 1.0e-300
        )
        pairwise_cell_measure = pairwise_sum(numerator) / max(pairwise_sum(denominator_values), 1.0e-300)
        compensated_cell_measure = neumaier_sum(numerator) / max(neumaier_sum(denominator_values), 1.0e-300)
        result.update({
            "cell_measure_ordinary": ordinary_cell_measure,
            "cell_measure_pairwise": pairwise_cell_measure,
            "cell_measure_neumaier": compensated_cell_measure,
            "cell_measure_max_reduction_delta": max(
                abs(ordinary_cell_measure - pairwise_cell_measure),
                abs(ordinary_cell_measure - compensated_cell_measure),
            ),
            "cell_measure_max_reduction_relative_delta": max(
                abs(ordinary_cell_measure - pairwise_cell_measure),
                abs(ordinary_cell_measure - compensated_cell_measure),
            ) / max(abs(ordinary_cell_measure), 1.0e-300),
        })

    edges = np.asarray(beta.grid.edges_m, dtype=np.longdouble)
    numbers = np.asarray(cells, dtype=np.longdouble)
    widths = edges[1:] - edges[:-1]
    m3_cells = numbers / widths * (edges[1:] ** 4 - edges[:-1] ** 4) / np.longdouble(4.0)
    m3_shadow = np.sum(m3_cells, dtype=np.longdouble)
    pi_shadow = np.longdouble(str(math.pi))
    beta_fraction_shadow = np.longdouble(4.0) * pi_shadow * m3_shadow / np.longdouble(3.0)
    q_beta_shadow = beta_fraction_shadow * np.longdouble(beta_parameters.x_b) / np.longdouble(
        beta_parameters.molar_volume_m3_mol
    )
    matrix_fraction = np.longdouble(1.0) - beta_fraction_shadow
    x_closure_shadow = np.longdouble(solver.ledger.matrix_molar_volume_m3_mol) * (
        np.longdouble(solver.ledger.total_b_mol_m3) - q_beta_shadow
    ) / matrix_fraction
    f_shadow = x_closure_shadow - np.longdouble(evaluation.x_trial)
    result.update({
        "M3_longdouble": str(m3_shadow),
        "Q_beta_longdouble": str(q_beta_shadow),
        "x_closure_longdouble": str(x_closure_shadow),
        "signed_F_longdouble": str(f_shadow),
        "signed_F_longdouble_as_float64": float(f_shadow),
        "signed_F_shadow_delta": abs(float(f_shadow) - evaluation.signed_f),
    })
    return result


def repeatability_audit(
    solver: CharacteristicReferenceSolver,
    *,
    prestate: DiagnosticPrestate,
    dt_s: float,
    x_trials: Sequence[float],
    repeats: int = 10,
) -> tuple[list[dict[str, Any]], bool]:
    """Repeat immutable evaluations and demand exact binary64 signatures."""

    if repeats < 2:
        raise ValueError("repeatability audit needs at least two repeats")
    rows: list[dict[str, Any]] = []
    overall = True
    for x_trial in x_trials:
        reference: tuple[str, ...] | None = None
        for repetition in range(1, int(repeats) + 1):
            evaluation = evaluate_closure_map(
                solver, prestate=prestate, dt_s=dt_s, x_trial=float(x_trial)
            )
            signature = (
                evaluation.x_closure.hex(), evaluation.signed_f.hex(), evaluation.population_hash,
                evaluation.cdf_signature, evaluation.departure_signature,
                evaluation.remap_topology_signature,
            )
            equal = reference is None or signature == reference
            overall = overall and equal
            if reference is None:
                reference = signature
            rows.append({
                "x_trial": float(x_trial),
                "x_trial_hex": float(x_trial).hex(),
                "repeat": repetition,
                "bitwise_equal_to_first": equal,
                "x_closure_hex": evaluation.x_closure.hex(),
                "signed_F_hex": evaluation.signed_f.hex(),
                "population_hash": evaluation.population_hash,
                "topology_signature": evaluation.remap_topology_signature,
            })
    return rows, bool(overall)
