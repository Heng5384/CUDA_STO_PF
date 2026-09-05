"""Read-only probes for the production autonomous time-of-flight tracer.

This module is deliberately an audit companion, not an alternative KWN
transport operator.  Its default table is required to reproduce the public
``trace_departure_faces_rk2`` map bit-for-bit before it may be used for a
component ablation.  Configurable table and inversion modes are then useful
only for assigning a numerical cause in a frozen autonomous experiment.

In particular, this code owns no population remap, matrix closure, dynamic
composition update, or accepted solver state.  A qualified autonomous wrapper
below explicitly rejects a dynamic-xB request; qualifying a frozen trace does
not create a dynamic time reference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray

from .conservative_remap import (
    CharacteristicTrace,
    ConservativeRemapError,
    _GAUSS_ABSCISSA,
    _checked_velocity,
    _gauss_time_of_flight,
    _gauss_time_of_flight_intervals,
    _invert_time_of_flight,
    _log_subdivided_faces,
    trace_departure_faces_rk2,
)
from .frozen_exact_flow_reference import FrozenAutonomousExactFlow


class ProductionTraceAuditError(RuntimeError):
    """Raised when an audit-only production-trace probe is not admissible."""


VelocityFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]
TableKind = Literal["PRODUCTION_GL2", "EXACT_TAU"]
InversionMode = Literal["PRODUCTION_KERNEL", "BRACKETED_TABLE"]


@dataclass(frozen=True)
class ProductionTraceAuditConfig:
    """A frozen auxiliary-table variant, never a population-grid change."""

    subcells_per_cell: int = 16
    table_kind: TableKind = "PRODUCTION_GL2"
    inversion_mode: InversionMode = "PRODUCTION_KERNEL"

    def __post_init__(self) -> None:
        if self.subcells_per_cell <= 0 or self.subcells_per_cell != int(self.subcells_per_cell):
            raise ProductionTraceAuditError("trace-table subcells per cell must be a positive integer")
        if self.table_kind not in {"PRODUCTION_GL2", "EXACT_TAU"}:
            raise ProductionTraceAuditError("unrecognized trace-table representation")
        if self.inversion_mode not in {"PRODUCTION_KERNEL", "BRACKETED_TABLE"}:
            raise ProductionTraceAuditError("unrecognized trace inversion mode")


@dataclass(frozen=True)
class ProductionTraceRun:
    """One same-sign, anchor-normalized ephemeral production TOF table."""

    run_id: int
    sign: float
    coordinates_m: NDArray[np.float64]
    cumulative_time_s: NDArray[np.float64]


@dataclass(frozen=True)
class ProductionTraceQuery:
    """A read-only departure query and its actual numerical path label."""

    radius_m: NDArray[np.float64]
    status: NDArray[np.str_]
    source_run_id: NDArray[np.int64]
    inversion_iteration_count: NDArray[np.int64]
    inversion_bracket_width_m: NDArray[np.float64]
    inversion_residual_s: NDArray[np.float64]


@dataclass(frozen=True)
class ProductionInversionResult:
    """One local table inversion with transparent stopping diagnostics."""

    radius_m: float
    iteration_count: int
    bracket_width_m: float
    residual_s: float


def _as_edges(edges_m: NDArray[np.float64] | np.ndarray) -> NDArray[np.float64]:
    edges = np.asarray(edges_m, dtype=np.float64)
    if (
        edges.ndim != 1
        or edges.size < 3
        or not np.all(np.isfinite(edges))
        or np.any(edges <= 0.0)
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ProductionTraceAuditError("frozen trace edges must be finite, positive, and increasing")
    return edges.copy()


def _as_points(points_m: NDArray[np.float64] | np.ndarray, *, edges_m: NDArray[np.float64]) -> NDArray[np.float64]:
    points = np.asarray(points_m, dtype=np.float64)
    if (
        points.ndim != 1
        or not np.all(np.isfinite(points))
        or np.any(points < edges_m[0])
        or np.any(points > edges_m[-1])
    ):
        raise ProductionTraceAuditError("trace query points must remain in the frozen physical domain")
    return points.copy()


def _require_duration(duration_s: float) -> float:
    duration = float(duration_s)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ProductionTraceAuditError("trace duration must be finite and positive")
    return duration


class ProductionAutonomousTOFTable:
    """A parity-checked query view of production's transient TOF tables.

    The table has the same log submesh and same-sign predicates as the public
    trace.  It does not construct a stationary tail: a query that would need
    such a new path is intentionally rejected rather than silently acquiring
    a second implementation.  The formal frozen ladder is separately checked
    to be interior before this restriction is relied on.
    """

    def __init__(
        self,
        *,
        edges_m: NDArray[np.float64] | np.ndarray,
        velocity_m_s: VelocityFunction,
        config: ProductionTraceAuditConfig = ProductionTraceAuditConfig(),
        exact_flow: FrozenAutonomousExactFlow | None = None,
    ) -> None:
        self.edges_m = _as_edges(edges_m)
        self.velocity_m_s = velocity_m_s
        self.config = config
        self.exact_flow = exact_flow
        if config.table_kind == "EXACT_TAU" and exact_flow is None:
            raise ProductionTraceAuditError("exact-Tau table audit requires the independent frozen flow")
        self._runs = self._build_runs()

    @property
    def runs(self) -> tuple[ProductionTraceRun, ...]:
        return self._runs

    def _exact_anchor_time(self, run: ProductionTraceRun, radius_m: float) -> float:
        if self.exact_flow is None:
            raise ProductionTraceAuditError("exact-Tau table lacks an independent reference flow")
        anchor = float(run.coordinates_m[0])
        return abs(float(self.exact_flow.tau(float(radius_m))) - float(self.exact_flow.tau(anchor)))

    def _build_runs(self) -> tuple[ProductionTraceRun, ...]:
        subdivision = int(self.config.subcells_per_cell)
        nodes = _log_subdivided_faces(self.edges_m, subcells_per_cell=subdivision)
        node_velocity = _checked_velocity(self.velocity_m_s, nodes, label="audit trace mesh")
        node_sign = np.sign(node_velocity)
        left = nodes[:-1]
        right = nodes[1:]
        midpoint = 0.5 * (left + right)
        half_width = 0.5 * (right - left)
        gauss_left = midpoint - _GAUSS_ABSCISSA * half_width
        gauss_right = midpoint + _GAUSS_ABSCISSA * half_width
        gauss_left_sign = np.sign(_checked_velocity(self.velocity_m_s, gauss_left, label="audit left Gauss mesh"))
        gauss_right_sign = np.sign(_checked_velocity(self.velocity_m_s, gauss_right, label="audit right Gauss mesh"))
        valid_interval = (
            (node_sign[:-1] != 0.0)
            & (node_sign[1:] == node_sign[:-1])
            & (gauss_left_sign == node_sign[:-1])
            & (gauss_right_sign == node_sign[:-1])
        )
        run_id = np.cumsum(np.concatenate((np.asarray([True]), ~valid_interval)), dtype=np.int64) - 1
        runs: list[ProductionTraceRun] = []
        for identifier in np.unique(run_id):
            node_indices = np.flatnonzero(run_id == identifier)
            if node_indices.size < 2:
                continue
            start, stop = int(node_indices[0]), int(node_indices[-1])
            sign = float(node_sign[start])
            if sign == 0.0:
                continue
            if np.any(node_sign[node_indices] != sign):
                raise ProductionTraceAuditError("audit trace table has an inconsistent same-sign run")
            coordinates = np.asarray(nodes[start : stop + 1], dtype=np.float64)
            if self.config.table_kind == "PRODUCTION_GL2":
                try:
                    increments = _gauss_time_of_flight(
                        coordinates, expected_sign=sign, velocity_m_s=self.velocity_m_s
                    )
                except ConservativeRemapError as error:
                    raise ProductionTraceAuditError("production GL2 TOF table construction failed") from error
                cumulative = np.concatenate((np.asarray([0.0]), np.cumsum(increments, dtype=np.float64)))
            else:
                provisional = ProductionTraceRun(
                    run_id=int(identifier),
                    sign=sign,
                    coordinates_m=coordinates.copy(),
                    cumulative_time_s=np.empty(coordinates.shape, dtype=np.float64),
                )
                cumulative = np.asarray(
                    [self._exact_anchor_time(provisional, float(radius)) for radius in coordinates],
                    dtype=np.float64,
                )
            if not np.all(np.isfinite(cumulative)) or np.any(np.diff(cumulative) <= 0.0):
                raise ProductionTraceAuditError("audit trace time coordinate is not strictly increasing")
            runs.append(
                ProductionTraceRun(
                    run_id=int(identifier),
                    sign=sign,
                    coordinates_m=coordinates.copy(),
                    cumulative_time_s=np.asarray(cumulative, dtype=np.float64).copy(),
                )
            )
        if not runs:
            raise ProductionTraceAuditError("audit trace has no queryable same-sign table")
        return tuple(runs)

    def _run_for_point(self, point_m: float, sign: float) -> ProductionTraceRun:
        matches = [
            run
            for run in self.runs
            if sign == run.sign and float(run.coordinates_m[0]) <= point_m <= float(run.coordinates_m[-1])
        ]
        if len(matches) != 1:
            raise ProductionTraceAuditError("trace query does not lie in one unambiguous same-sign table run")
        return matches[0]

    def _partial_time(self, run: ProductionTraceRun, left_m: float, right_m: float) -> float:
        left = float(left_m)
        right = float(right_m)
        if right < left:
            raise ProductionTraceAuditError("trace local flight-time interval is reversed")
        if right == left:
            return 0.0
        if self.config.table_kind == "PRODUCTION_GL2":
            try:
                return float(
                    _gauss_time_of_flight_intervals(
                        np.asarray([left], dtype=np.float64),
                        np.asarray([right], dtype=np.float64),
                        expected_sign=run.sign,
                        velocity_m_s=self.velocity_m_s,
                    )[0]
                )
            except ConservativeRemapError as error:
                raise ProductionTraceAuditError("production local GL2 flight-time evaluation failed") from error
        return self._exact_anchor_time(run, right) - self._exact_anchor_time(run, left)

    def _source_time(self, run: ProductionTraceRun, point_m: float) -> float:
        point = float(point_m)
        coordinates = run.coordinates_m
        position = int(np.searchsorted(coordinates, point, side="left"))
        if position < coordinates.size and float(coordinates[position]) == point:
            return float(run.cumulative_time_s[position])
        slot = int(np.clip(position - 1, 0, coordinates.size - 2))
        return float(run.cumulative_time_s[slot] + self._partial_time(run, float(coordinates[slot]), point))

    def _production_kernel_inverse(self, run: ProductionTraceRun, target_s: float) -> ProductionInversionResult:
        target = float(target_s)
        try:
            radius = float(
                _invert_time_of_flight(
                    np.asarray([target], dtype=np.float64),
                    run.cumulative_time_s,
                    run.coordinates_m,
                    expected_sign=run.sign,
                    velocity_m_s=self.velocity_m_s,
                )[0]
            )
        except ConservativeRemapError as error:
            raise ProductionTraceAuditError("production trace-kernel inversion failed") from error
        slot = int(np.clip(np.searchsorted(run.cumulative_time_s, target, side="right") - 1, 0, run.coordinates_m.size - 2))
        local_target = target - float(run.cumulative_time_s[slot])
        residual = self._partial_time(run, float(run.coordinates_m[slot]), radius) - local_target
        return ProductionInversionResult(
            radius_m=radius,
            # The public kernel deliberately exposes a radius only.  Do not
            # invent an iteration count or a final bracket width in this
            # parity companion; -1/NaN explicitly mean "not exposed".
            iteration_count=-1,
            bracket_width_m=math.nan,
            residual_s=float(residual),
        )

    def _bracketed_table_inverse(self, run: ProductionTraceRun, target_s: float) -> ProductionInversionResult:
        target = float(target_s)
        times = run.cumulative_time_s
        coordinates = run.coordinates_m
        if target < float(times[0]) or target > float(times[-1]):
            raise ProductionTraceAuditError("bracketed table inversion target leaves the physical table")
        position = int(np.searchsorted(times, target, side="left"))
        if position < times.size and target == float(times[position]):
            return ProductionInversionResult(
                radius_m=float(coordinates[position]),
                iteration_count=0,
                bracket_width_m=0.0,
                residual_s=0.0,
            )
        slot = int(np.clip(position - 1, 0, coordinates.size - 2))
        interval_left = float(coordinates[slot])
        low = interval_left
        high = float(coordinates[slot + 1])
        cumulative_left = float(times[slot])
        local_target = target - cumulative_left
        scale_time = max(abs(local_target), abs(float(times[slot + 1] - times[slot])), 1.0e-300)
        tolerance_s = 64.0 * np.finfo(np.float64).eps * scale_time
        value = 0.5 * (low + high)
        residual = math.inf
        for iteration in range(1, 65):
            partial = self._partial_time(run, interval_left, value)
            residual = partial - local_target
            if abs(residual) <= tolerance_s:
                return ProductionInversionResult(value, iteration, high - low, residual)
            if residual > 0.0:
                high = value
            else:
                low = value
            velocity = float(_checked_velocity(self.velocity_m_s, np.asarray([value], dtype=np.float64), label="audit bracketed inversion")[0])
            proposal = value - residual * abs(velocity)
            if not (low < proposal < high) or not math.isfinite(proposal):
                proposal = 0.5 * (low + high)
            value = float(proposal)
            width_floor = 32.0 * np.finfo(np.float64).eps * max(abs(low), abs(high), 1.0e-300)
            if high - low <= width_floor:
                midpoint = 0.5 * (low + high)
                final_residual = self._partial_time(run, float(coordinates[slot]), midpoint) - local_target
                return ProductionInversionResult(midpoint, iteration, high - low, final_residual)
        raise ProductionTraceAuditError("bracketed table inversion did not reach binary64 resolution")

    def invert(self, run: ProductionTraceRun, target_s: float) -> ProductionInversionResult:
        if self.config.inversion_mode == "PRODUCTION_KERNEL":
            return self._production_kernel_inverse(run, target_s)
        return self._bracketed_table_inverse(run, target_s)

    def query_departure(self, points_m: NDArray[np.float64] | np.ndarray, duration_s: float) -> ProductionTraceQuery:
        points = _as_points(points_m, edges_m=self.edges_m)
        duration = _require_duration(duration_s)
        velocity = _checked_velocity(self.velocity_m_s, points, label="audit trace query")
        signs = np.sign(velocity)
        result = np.empty_like(points)
        status = np.empty(points.shape, dtype="<U24")
        run_ids = np.full(points.shape, -1, dtype=np.int64)
        iteration_counts = np.zeros(points.shape, dtype=np.int64)
        widths = np.zeros(points.shape, dtype=np.float64)
        residuals = np.zeros(points.shape, dtype=np.float64)
        for index, (point, sign) in enumerate(zip(points, signs)):
            if sign == 0.0:
                result[index] = point
                status[index] = "STATIONARY_CRITICAL"
                continue
            run = self._run_for_point(float(point), float(sign))
            run_ids[index] = run.run_id
            source_time = self._source_time(run, float(point))
            target = source_time - duration if sign > 0.0 else source_time + duration
            if target < 0.0:
                if float(run.coordinates_m[0]) == float(self.edges_m[0]):
                    result[index] = self.edges_m[0]
                    status[index] = "NO_INFLOW_LOWER"
                    continue
                raise ProductionTraceAuditError("trace query requires an unimplemented stationary lower tail")
            if target > float(run.cumulative_time_s[-1]):
                if float(run.coordinates_m[-1]) == float(self.edges_m[-1]):
                    result[index] = self.edges_m[-1]
                    status[index] = "NO_INFLOW_UPPER"
                    continue
                raise ProductionTraceAuditError("trace query requires an unimplemented stationary upper tail")
            inverted = self.invert(run, target)
            result[index] = inverted.radius_m
            status[index] = "INTERIOR"
            iteration_counts[index] = inverted.iteration_count
            widths[index] = inverted.bracket_width_m
            residuals[index] = inverted.residual_s
        if not np.all(np.isfinite(result)):
            raise ProductionTraceAuditError("trace table query yielded a non-finite departure radius")
        return ProductionTraceQuery(
            radius_m=result,
            status=status,
            source_run_id=run_ids,
            inversion_iteration_count=iteration_counts,
            inversion_bracket_width_m=widths,
            inversion_residual_s=residuals,
        )

    def require_default_public_parity(self, duration_s: float) -> CharacteristicTrace:
        """Fail closed unless the default audit table is the actual public map."""

        if self.config != ProductionTraceAuditConfig():
            raise ProductionTraceAuditError("public parity is defined only for the unmodified production table")
        duration = _require_duration(duration_s)
        try:
            public = trace_departure_faces_rk2(
                self.edges_m,
                dt_s=duration,
                velocity_m_s=self.velocity_m_s,
                lower_radius_m=float(self.edges_m[0]),
                upper_radius_m=float(self.edges_m[-1]),
            )
        except ConservativeRemapError as error:
            raise ProductionTraceAuditError("public production trace failed during audit parity") from error
        query = self.query_departure(self.edges_m, duration)
        if not np.array_equal(query.radius_m, public.departure_faces_m):
            raise ProductionTraceAuditError("audit default TOF table does not bitwise reproduce public production trace")
        return public

    def tau_rows(self, *, include_midpoints: bool = True) -> list[dict[str, float | int | str]]:
        """Return anchor-normalized production and independent Tau samples."""

        if self.exact_flow is None:
            raise ProductionTraceAuditError("Tau comparison requires the independent frozen flow")
        rows: list[dict[str, float | int | str]] = []
        for run in self.runs:
            coordinates = run.coordinates_m
            sample_points: list[tuple[str, float, float]] = [
                ("NODE", float(radius), float(time))
                for radius, time in zip(coordinates, run.cumulative_time_s)
            ]
            if include_midpoints:
                for left, right, base in zip(coordinates[:-1], coordinates[1:], run.cumulative_time_s[:-1]):
                    middle = 0.5 * (float(left) + float(right))
                    sample_points.append(("MIDPOINT", middle, float(base + self._partial_time(run, float(left), middle))))
            anchor = float(coordinates[0])
            anchor_exact = float(self.exact_flow.tau(anchor))
            for kind, radius, production_tau in sample_points:
                exact_tau = abs(float(self.exact_flow.tau(radius)) - anchor_exact)
                difference = production_tau - exact_tau
                rows.append(
                    {
                        "run_id": int(run.run_id),
                        "branch_sign": int(np.sign(run.sign)),
                        "sample_kind": kind,
                        "radius_m": radius,
                        "tau_production_anchor_normalized_s": production_tau,
                        "tau_exact_anchor_normalized_s": exact_tau,
                        "absolute_error_s": abs(difference),
                        "relative_error": abs(difference) / max(abs(exact_tau), 1.0e-300),
                    }
                )
        return rows


def normalized_production_statuses(
    *,
    arrival_m: NDArray[np.float64] | np.ndarray,
    trace: CharacteristicTrace,
    velocity_m_s: VelocityFunction,
    lower_radius_m: float,
    upper_radius_m: float,
) -> NDArray[np.str_]:
    """Normalize the backward-map statuses for a public production trace."""

    arrival = np.asarray(arrival_m, dtype=np.float64)
    departure = np.asarray(trace.departure_faces_m, dtype=np.float64)
    velocity = _checked_velocity(velocity_m_s, arrival, label="production status normalization")
    if arrival.shape != departure.shape:
        raise ProductionTraceAuditError("production status normalization shape mismatch")
    statuses = np.full(arrival.shape, "INTERIOR", dtype="<U24")
    zero = velocity == 0.0
    statuses[zero] = "STATIONARY_CRITICAL"
    statuses[(velocity > 0.0) & (departure == float(lower_radius_m))] = "NO_INFLOW_LOWER"
    statuses[(velocity < 0.0) & (departure == float(upper_radius_m))] = "NO_INFLOW_UPPER"
    return statuses


def public_production_trace_probe(
    *,
    arrival_faces_m: NDArray[np.float64] | np.ndarray,
    duration_s: float,
    velocity_m_s: VelocityFunction,
    lower_radius_m: float,
    upper_radius_m: float,
) -> tuple[CharacteristicTrace, NDArray[np.str_]]:
    """Read the actual public production trace without changing any state."""

    faces = _as_edges(arrival_faces_m)
    try:
        trace = trace_departure_faces_rk2(
            faces,
            dt_s=_require_duration(duration_s),
            velocity_m_s=velocity_m_s,
            lower_radius_m=float(lower_radius_m),
            upper_radius_m=float(upper_radius_m),
        )
    except ConservativeRemapError as error:
        raise ProductionTraceAuditError("public production trace probe failed") from error
    return trace, normalized_production_statuses(
        arrival_m=faces,
        trace=trace,
        velocity_m_s=velocity_m_s,
        lower_radius_m=float(lower_radius_m),
        upper_radius_m=float(upper_radius_m),
    )


@dataclass(frozen=True)
class QualifiedAutonomousTOFTraceKernelV1:
    """A scope-tagged frozen-autonomous trace interface.

    It intentionally does not accept a time-varying matrix composition.  The
    underlying public trace remains a local autonomous numerical operation;
    this wrapper records only what has been qualified here, not a dynamic
    nonlinear solver capability.
    """

    mode: str = "AUTONOMOUS_TOF"

    def trace(
        self,
        *,
        arrival_faces_m: NDArray[np.float64] | np.ndarray,
        duration_s: float,
        velocity_m_s: VelocityFunction,
        lower_radius_m: float,
        upper_radius_m: float,
        matrix_xb_constant_over_interval: bool,
    ) -> CharacteristicTrace:
        if self.mode != "AUTONOMOUS_TOF" or not matrix_xb_constant_over_interval:
            raise ProductionTraceAuditError("QUALIFIED_TOF_TRACE_KERNEL_V1 rejects dynamic xB use")
        trace, _statuses = public_production_trace_probe(
            arrival_faces_m=arrival_faces_m,
            duration_s=duration_s,
            velocity_m_s=velocity_m_s,
            lower_radius_m=lower_radius_m,
            upper_radius_m=upper_radius_m,
        )
        return trace
