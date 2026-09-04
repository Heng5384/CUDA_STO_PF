"""Read-only frozen CR1 semigroup-error decomposition diagnostics.

This module deliberately does *not* alter the CR1 production solver.  It
reuses its public characteristic tracer and conservative piecewise-constant
remap to distinguish three maps of one immutable, autonomous population:

``A``
    one full characteristic trace followed by one CR1 remap;
``B``
    two half characteristic traces composed in physical-radius space,
    followed by one CR1 remap; and
``C``
    two ordinary half trace/remap operations.

Consequently ``C - A = (B - A) + (C - B)`` separates a flow-map composition
defect from the effect of the intermediate CR1 projection.  The code below is
an analysis harness only: it owns no closure loop, root selection, timestep
controller, checkpoint, or accepted solver state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .conservative_remap import (
    CharacteristicRemapPartition,
    CharacteristicTrace,
    ConservativeRemapError,
    ConservativeRemapResult,
    _GAUSS_ABSCISSA,
    _TRACE_SUBCELLS_PER_CELL,
    _checked_velocity,
    _gauss_time_of_flight,
    _gauss_time_of_flight_intervals,
    _invert_time_of_flight,
    _log_subdivided_faces,
    characteristic_remap_partition,
    conservative_remap_piecewise_constant,
    trace_departure_faces_rk2,
)
from .diagnostics import discrete_wasserstein_distance
from .population_metrics import positive_cell_quadrature


class FrozenSemigroupDecompositionError(RuntimeError):
    """A read-only decomposition-contract failure."""


VelocityFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class FrozenCR1Path:
    """One immutable CR1 endpoint together with the map that produced it."""

    label: str
    cells: NDArray[np.float64]
    trace: CharacteristicTrace
    partition: CharacteristicRemapPartition
    remap: ConservativeRemapResult
    intermediate_cells: NDArray[np.float64] | None = None
    first_half_trace: CharacteristicTrace | None = None
    second_half_trace: CharacteristicTrace | None = None


@dataclass(frozen=True)
class _CanonicalTraceRun:
    """One existing same-sign time-of-flight table from the public tracer."""

    sign: float
    coordinates_m: NDArray[np.float64]
    cumulative_time_s: NDArray[np.float64]


@dataclass(frozen=True)
class _CanonicalHalfFlowTable:
    """Read-only query view of the production canonical trace table.

    This is deliberately narrower than a new characteristic integrator.  It
    retains the canonical production submesh and its existing Gauss/inversion
    semantics, and permits only queries and targets that remain strictly
    inside one already-tabulated same-sign run.  There is no boundary or
    stationary-tail fallback in this diagnostic interface.
    """

    edges_m: NDArray[np.float64]
    runs: tuple[_CanonicalTraceRun, ...]


@dataclass(frozen=True)
class SparseCR1Map:
    """A small dependency-free CSR representation of a CR1 linear map.

    The state is the cell-integrated number measure.  Rows index fixed arrival
    cells and columns index source cells.  It is intentionally diagnostic-only
    and does not replace the production CDF/remap evaluation.
    """

    indptr: NDArray[np.int64]
    indices: NDArray[np.int64]
    data: NDArray[np.float64]
    size: int

    def __post_init__(self) -> None:
        if (
            self.indptr.shape != (self.size + 1,)
            or self.indices.ndim != 1
            or self.data.ndim != 1
            or self.indices.shape != self.data.shape
            or int(self.indptr[0]) != 0
            or int(self.indptr[-1]) != self.indices.size
            or np.any(np.diff(self.indptr) < 0)
            or np.any(self.indices < 0)
            or np.any(self.indices >= self.size)
            or not np.all(np.isfinite(self.data))
        ):
            raise FrozenSemigroupDecompositionError("invalid sparse CR1 map")

    @property
    def nnz(self) -> int:
        return int(self.data.size)

    def row(self, index: int) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        row = int(index)
        if row < 0 or row >= self.size:
            raise IndexError("sparse CR1 row is out of range")
        start, stop = int(self.indptr[row]), int(self.indptr[row + 1])
        return self.indices[start:stop], self.data[start:stop]

    def apply(self, values: NDArray[np.float64] | np.ndarray) -> NDArray[np.float64]:
        source = np.asarray(values, dtype=np.float64)
        if source.shape != (self.size,) or not np.all(np.isfinite(source)):
            raise FrozenSemigroupDecompositionError("sparse CR1 input is invalid")
        result = np.zeros(self.size, dtype=np.float64)
        for row in range(self.size):
            indices, data = self.row(row)
            if indices.size:
                result[row] = float(np.sum(data * source[indices], dtype=np.float64))
        return result

    def one_norm(self) -> float:
        columns = np.zeros(self.size, dtype=np.float64)
        np.add.at(columns, self.indices, np.abs(self.data))
        return float(np.max(columns)) if columns.size else 0.0

    def infinity_norm(self) -> float:
        rows = np.zeros(self.size, dtype=np.float64)
        for row in range(self.size):
            _indices, data = self.row(row)
            rows[row] = float(np.sum(np.abs(data), dtype=np.float64))
        return float(np.max(rows)) if rows.size else 0.0

    def compose(self, right: "SparseCR1Map") -> "SparseCR1Map":
        """Return ``self @ right`` using deterministic row-local reductions."""

        if self.size != right.size:
            raise FrozenSemigroupDecompositionError("sparse CR1 maps have incompatible sizes")
        indptr = [0]
        columns: list[int] = []
        weights: list[float] = []
        for row in range(self.size):
            accumulator: dict[int, float] = {}
            intermediate, outer_weights = self.row(row)
            for middle, outer in zip(intermediate, outer_weights):
                source, inner_weights = right.row(int(middle))
                for column, inner in zip(source, inner_weights):
                    key = int(column)
                    accumulator[key] = accumulator.get(key, 0.0) + float(outer * inner)
            for column in sorted(accumulator):
                value = float(accumulator[column])
                if value != 0.0:
                    columns.append(column)
                    weights.append(value)
            indptr.append(len(columns))
        return SparseCR1Map(
            indptr=np.asarray(indptr, dtype=np.int64),
            indices=np.asarray(columns, dtype=np.int64),
            data=np.asarray(weights, dtype=np.float64),
            size=self.size,
        )

    def difference(self, right: "SparseCR1Map") -> "SparseCR1Map":
        """Return ``self - right`` without materialising a dense matrix."""

        if self.size != right.size:
            raise FrozenSemigroupDecompositionError("sparse CR1 maps have incompatible sizes")
        indptr = [0]
        columns: list[int] = []
        weights: list[float] = []
        for row in range(self.size):
            accumulator: dict[int, float] = {}
            for column, value in zip(*self.row(row)):
                accumulator[int(column)] = accumulator.get(int(column), 0.0) + float(value)
            for column, value in zip(*right.row(row)):
                accumulator[int(column)] = accumulator.get(int(column), 0.0) - float(value)
            for column in sorted(accumulator):
                value = float(accumulator[column])
                if value != 0.0:
                    columns.append(column)
                    weights.append(value)
            indptr.append(len(columns))
        return SparseCR1Map(
            indptr=np.asarray(indptr, dtype=np.int64),
            indices=np.asarray(columns, dtype=np.int64),
            data=np.asarray(weights, dtype=np.float64),
            size=self.size,
        )


@dataclass(frozen=True)
class FrozenSemigroupDecomposition:
    """The A/B/C endpoints, additive defects, and optional sparse maps."""

    h_s: float
    direct: FrozenCR1Path
    composed_flow: FrozenCR1Path
    sequential: FrozenCR1Path
    d_total: NDArray[np.float64]
    d_trace: NDArray[np.float64]
    d_remap: NDArray[np.float64]
    additive_residual: NDArray[np.float64]
    P_h: SparseCR1Map
    P_half_1: SparseCR1Map
    P_half_2: SparseCR1Map
    P_compflow: SparseCR1Map
    P_sequential: SparseCR1Map
    composed_query_audit: Mapping[str, object]


def _validate_problem(
    edges_m: NDArray[np.float64] | np.ndarray,
    initial_cells: NDArray[np.float64] | np.ndarray,
    dt_s: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    edges = np.asarray(edges_m, dtype=np.float64)
    cells = np.asarray(initial_cells, dtype=np.float64)
    dt = float(dt_s)
    if (
        edges.ndim != 1
        or edges.size < 3
        or not np.all(np.isfinite(edges))
        or np.any(edges <= 0.0)
        or np.any(np.diff(edges) <= 0.0)
        or cells.shape != (edges.size - 1,)
        or not np.all(np.isfinite(cells))
        or np.any(cells < 0.0)
        or not math.isfinite(dt)
        or dt <= 0.0
    ):
        raise FrozenSemigroupDecompositionError("frozen CR1 decomposition input is invalid")
    return edges.copy(), cells.copy(), dt


def _trace_on_domain_edges(
    edges: NDArray[np.float64],
    *,
    dt_s: float,
    velocity_m_s: VelocityFunction,
) -> CharacteristicTrace:
    try:
        return trace_departure_faces_rk2(
            edges,
            dt_s=float(dt_s),
            velocity_m_s=velocity_m_s,
            lower_radius_m=float(edges[0]),
            upper_radius_m=float(edges[-1]),
        )
    except ConservativeRemapError as error:
        raise FrozenSemigroupDecompositionError(f"diagnostic characteristic trace failed: {error}") from error


def _build_canonical_half_flow_table(
    edges_m: NDArray[np.float64], *, velocity_m_s: VelocityFunction
) -> _CanonicalHalfFlowTable:
    """Build only the already-existing canonical tracer tables.

    This mirrors the public tracer's fixed 16-subcell log mesh, sign predicates,
    and Gauss time-of-flight tables.  It intentionally owns neither a new
    integration rule nor a stationary-tail construction.
    """

    edges = np.asarray(edges_m, dtype=np.float64)
    nodes = _log_subdivided_faces(edges, subcells_per_cell=_TRACE_SUBCELLS_PER_CELL)
    node_velocity = _checked_velocity(velocity_m_s, nodes, label="fixed-canonical trace mesh")
    node_sign = np.sign(node_velocity)
    left = nodes[:-1]
    right = nodes[1:]
    midpoint = 0.5 * (left + right)
    half_width = 0.5 * (right - left)
    gauss_left = midpoint - _GAUSS_ABSCISSA * half_width
    gauss_right = midpoint + _GAUSS_ABSCISSA * half_width
    gauss_left_sign = np.sign(
        _checked_velocity(velocity_m_s, gauss_left, label="fixed-canonical left Gauss")
    )
    gauss_right_sign = np.sign(
        _checked_velocity(velocity_m_s, gauss_right, label="fixed-canonical right Gauss")
    )
    valid_interval = (
        (node_sign[:-1] != 0.0)
        & (node_sign[1:] == node_sign[:-1])
        & (gauss_left_sign == node_sign[:-1])
        & (gauss_right_sign == node_sign[:-1])
    )
    run_id = np.cumsum(
        np.concatenate((np.asarray([True]), ~valid_interval)), dtype=np.int64
    ) - 1
    runs: list[_CanonicalTraceRun] = []
    for identifier in np.unique(run_id):
        node_indices = np.flatnonzero(run_id == identifier)
        if node_indices.size < 2:
            continue
        start, stop = int(node_indices[0]), int(node_indices[-1])
        sign = float(node_sign[start])
        if sign == 0.0:
            continue
        if np.any(node_sign[node_indices] != sign):
            raise FrozenSemigroupDecompositionError(
                "fixed-canonical trace table has an inconsistent same-sign run"
            )
        coordinates = np.asarray(nodes[start : stop + 1], dtype=np.float64)
        try:
            interval_time = _gauss_time_of_flight(
                coordinates, expected_sign=sign, velocity_m_s=velocity_m_s
            )
        except ConservativeRemapError as error:
            raise FrozenSemigroupDecompositionError(
                f"fixed-canonical trace table construction failed: {error}"
            ) from error
        cumulative = np.concatenate(
            (np.asarray([0.0]), np.cumsum(interval_time, dtype=np.float64))
        )
        if np.any(np.diff(cumulative) <= 0.0) or not np.all(np.isfinite(cumulative)):
            raise FrozenSemigroupDecompositionError("fixed-canonical trace time table is invalid")
        runs.append(
            _CanonicalTraceRun(
                sign=sign,
                coordinates_m=coordinates.copy(),
                cumulative_time_s=np.asarray(cumulative, dtype=np.float64).copy(),
            )
        )
    if not runs:
        raise FrozenSemigroupDecompositionError("fixed-canonical trace table has no queryable run")
    return _CanonicalHalfFlowTable(edges_m=edges.copy(), runs=tuple(runs))


def _query_canonical_half_flow_table(
    table: _CanonicalHalfFlowTable,
    points_m: NDArray[np.float64],
    *,
    half_dt_s: float,
    velocity_m_s: VelocityFunction,
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Evaluate physical half-flow points only inside an existing table run.

    Exact physical no-inflow at an existing physical table boundary is
    retained exactly as in the public tracer.  Every non-boundary query and
    target must remain strictly inside one pre-existing same-sign table; this
    deliberately fails closed rather than creating a new mesh, stationary
    tail, or interpolation rule.
    """

    points = np.asarray(points_m, dtype=np.float64)
    edges = np.asarray(table.edges_m, dtype=np.float64)
    half = float(half_dt_s)
    if (
        points.ndim != 1
        or not np.all(np.isfinite(points))
        or not math.isfinite(half)
        or half <= 0.0
        or np.any(points < edges[0])
        or np.any(points > edges[-1])
    ):
        raise FrozenSemigroupDecompositionError("canonical-table half-flow query is invalid")
    velocity = _checked_velocity(velocity_m_s, points, label="canonical-table query")
    signs = np.sign(velocity)
    if np.any(signs == 0.0):
        raise FrozenSemigroupDecompositionError(
            "B_QUERY_ON_STATIONARY_RADIUS_IS_NOT_ADMISSIBLE"
        )
    result = np.empty_like(points)
    lower_endpoint = (points == edges[0]) & (signs > 0.0)
    upper_endpoint = (points == edges[-1]) & (signs < 0.0)
    result[lower_endpoint] = edges[0]
    result[upper_endpoint] = edges[-1]
    endpoint_assigned = lower_endpoint | upper_endpoint
    run_assigned = np.zeros(points.shape, dtype=bool)
    run_query_counts: list[int] = []
    table_boundary_no_inflow_count = 0
    for run in table.runs:
        coordinates = np.asarray(run.coordinates_m, dtype=np.float64)
        cumulative = np.asarray(run.cumulative_time_s, dtype=np.float64)
        candidate = (
            ~endpoint_assigned
            &
            (signs == float(run.sign))
            & (points >= coordinates[0])
            & (points <= coordinates[-1])
        )
        if np.any(candidate & run_assigned):
            raise FrozenSemigroupDecompositionError("canonical-table query belongs to multiple runs")
        mask = candidate
        if not np.any(mask):
            run_query_counts.append(0)
            continue
        query = points[mask]
        position = np.searchsorted(coordinates, query, side="left")
        is_node = (position < coordinates.size) & (coordinates[np.minimum(position, coordinates.size - 1)] == query)
        slot = np.clip(position - 1, 0, coordinates.size - 2)
        source_time = np.empty_like(query)
        source_time[is_node] = cumulative[position[is_node]]
        if np.any(~is_node):
            try:
                source_time[~is_node] = cumulative[slot[~is_node]] + _gauss_time_of_flight_intervals(
                    coordinates[slot[~is_node]],
                    query[~is_node],
                    expected_sign=float(run.sign),
                    velocity_m_s=velocity_m_s,
                )
            except ConservativeRemapError as error:
                raise FrozenSemigroupDecompositionError(
                    f"canonical-table local source-time query failed: {error}"
                ) from error
        target_time = source_time - half if run.sign > 0.0 else source_time + half
        reaches_physical_lower = (
            (run.sign > 0.0)
            & (target_time <= 0.0)
            & (coordinates[0] == edges[0])
        )
        reaches_physical_upper = (
            (run.sign < 0.0)
            & (target_time >= cumulative[-1])
            & (coordinates[-1] == edges[-1])
        )
        interior = ~(reaches_physical_lower | reaches_physical_upper)
        if np.any((target_time <= 0.0) & ~reaches_physical_lower) or np.any(
            (target_time >= cumulative[-1]) & ~reaches_physical_upper
        ):
            raise FrozenSemigroupDecompositionError(
                "B_QUERY_TARGET_OUTSIDE_CANONICAL_INTERIOR"
            )
        local_result = np.empty_like(query)
        local_result[reaches_physical_lower] = edges[0]
        local_result[reaches_physical_upper] = edges[-1]
        if np.any(interior):
            try:
                local_result[interior] = _invert_time_of_flight(
                    target_time[interior],
                    cumulative,
                    coordinates,
                    expected_sign=float(run.sign),
                    velocity_m_s=velocity_m_s,
                )
            except ConservativeRemapError as error:
                raise FrozenSemigroupDecompositionError(
                    f"canonical-table local inverse query failed: {error}"
                ) from error
        result[mask] = local_result
        table_boundary_no_inflow_count += int(
            np.count_nonzero(reaches_physical_lower | reaches_physical_upper)
        )
        run_assigned[mask] = True
        run_query_counts.append(int(np.count_nonzero(mask)))
    assigned = endpoint_assigned | run_assigned
    if not np.all(assigned):
        raise FrozenSemigroupDecompositionError(
            "B_QUERY_OUTSIDE_CANONICAL_SAME_SIGN_RUN"
        )
    if not np.all(np.isfinite(result)):
        raise FrozenSemigroupDecompositionError("canonical-table half-flow query is non-finite")
    return np.asarray(result, dtype=np.float64), {
        "canonical_run_count": len(table.runs),
        "canonical_run_query_counts": run_query_counts,
        "physical_endpoint_no_inflow_query_count": int(
            np.count_nonzero(lower_endpoint | upper_endpoint)
        ),
        "physical_table_no_inflow_query_count": table_boundary_no_inflow_count,
        "all_nonboundary_targets_strictly_interior": True,
    }


def canonical_table_half_flow_query(
    *,
    canonical_edges_m: NDArray[np.float64],
    query_points_m: NDArray[np.float64],
    half_dt_s: float,
    velocity_m_s: VelocityFunction,
    canonical_public_trace: CharacteristicTrace,
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Query B's second physical half-flow from a parity-checked CR1 table.

    The public tracer remains the authority for the canonical half flow.  This
    function first independently reproduces that map at every canonical face
    using exactly its existing time-of-flight tables.  Only after bitwise
    parity does it query the first-half physical departure radii.  It is a
    frozen diagnostic interface, not a general characteristic solver.
    """

    edges = np.asarray(canonical_edges_m, dtype=np.float64)
    expected_arrival = np.asarray(canonical_public_trace.arrival_faces_m, dtype=np.float64)
    expected_departure = np.asarray(canonical_public_trace.departure_faces_m, dtype=np.float64)
    if (
        expected_arrival.shape != edges.shape
        or expected_departure.shape != edges.shape
        or not np.array_equal(expected_arrival, edges)
    ):
        raise FrozenSemigroupDecompositionError(
            "canonical public half trace does not bind the frozen grid"
        )
    query = np.asarray(query_points_m, dtype=np.float64)
    if query.ndim != 1 or not np.all(np.isfinite(query)) or np.any(query < edges[0]) or np.any(query > edges[-1]):
        raise FrozenSemigroupDecompositionError("canonical-table query points are invalid")
    # This is the public tracer's explicit bitwise identity branch.  It is
    # not a table interpolation or a special B fallback: every physical
    # radius is unchanged when the canonical flow itself is literally zero.
    if np.array_equal(expected_departure, edges):
        return query.copy(), {
            "mode": "PUBLIC_ZERO_MOBILITY_IDENTITY",
            "canonical_face_count": int(edges.size),
            "canonical_edge_bitwise_parity": True,
            "canonical_parity_audit": {"identity": True},
            "query_audit": {"identity": True},
            "no_augmented_trace_mesh": True,
            "no_new_boundary_or_stationary_tail_path": True,
        }
    table = _build_canonical_half_flow_table(edges, velocity_m_s=velocity_m_s)
    reproduced, parity_audit = _query_canonical_half_flow_table(
        table, edges, half_dt_s=half_dt_s, velocity_m_s=velocity_m_s
    )
    if not np.array_equal(reproduced, expected_departure):
        raise FrozenSemigroupDecompositionError(
            "canonical-table query does not reproduce the public canonical half trace bitwise"
        )
    queried, query_audit = _query_canonical_half_flow_table(
        table,
        query,
        half_dt_s=half_dt_s,
        velocity_m_s=velocity_m_s,
    )
    return queried, {
        "mode": "FIXED_CANONICAL_PRODUCTION_TIME_OF_FLIGHT_TABLE",
        "canonical_face_count": int(edges.size),
        "canonical_edge_bitwise_parity": True,
        "canonical_parity_audit": parity_audit,
        "query_audit": query_audit,
        "no_augmented_trace_mesh": True,
        "no_new_boundary_or_stationary_tail_path": True,
    }


def _composed_half_trace(
    edges: NDArray[np.float64],
    *,
    dt_s: float,
    velocity_m_s: VelocityFunction,
    first_half_trace: CharacteristicTrace,
) -> tuple[CharacteristicTrace, dict[str, object]]:
    """Compose physical half-flow maps through a parity-checked fixed table."""

    half = 0.5 * float(dt_s)
    second_departure, query_audit = canonical_table_half_flow_query(
        canonical_edges_m=edges,
        query_points_m=np.asarray(first_half_trace.departure_faces_m, dtype=np.float64),
        half_dt_s=half,
        velocity_m_s=velocity_m_s,
        canonical_public_trace=first_half_trace,
    )
    # The midpoint array is the actual physical radius reached after the first
    # backward half trace, not an addition of two departure offsets.
    composed = CharacteristicTrace(
        arrival_faces_m=edges.copy(),
        departure_faces_m=second_departure,
        midpoint_faces_m=np.asarray(first_half_trace.departure_faces_m, dtype=np.float64).copy(),
        lower_no_inflow_face_count=int(np.count_nonzero(second_departure == edges[0])),
        upper_no_inflow_face_count=int(np.count_nonzero(second_departure == edges[-1])),
    )
    return composed, query_audit


def _remap_path(
    label: str,
    *,
    edges: NDArray[np.float64],
    old_cells: NDArray[np.float64],
    trace: CharacteristicTrace,
    intermediate_cells: NDArray[np.float64] | None = None,
    first_half_trace: CharacteristicTrace | None = None,
    second_half_trace: CharacteristicTrace | None = None,
) -> FrozenCR1Path:
    try:
        remap = conservative_remap_piecewise_constant(edges, old_cells, trace.departure_faces_m)
        partition = characteristic_remap_partition(edges, trace=trace)
    except ConservativeRemapError as error:
        raise FrozenSemigroupDecompositionError(f"diagnostic CR1 remap failed: {error}") from error
    return FrozenCR1Path(
        label=label,
        cells=np.asarray(remap.cell_number_m3, dtype=np.float64).copy(),
        trace=trace,
        partition=partition,
        remap=remap,
        intermediate_cells=None if intermediate_cells is None else np.asarray(intermediate_cells, dtype=np.float64).copy(),
        first_half_trace=first_half_trace,
        second_half_trace=second_half_trace,
    )


def sparse_cr1_map(
    edges_m: NDArray[np.float64] | np.ndarray,
    departure_faces_m: NDArray[np.float64] | np.ndarray,
) -> SparseCR1Map:
    """Construct the exact geometric CR1 overlap operator for one face map."""

    edges = np.asarray(edges_m, dtype=np.float64)
    departure = np.asarray(departure_faces_m, dtype=np.float64)
    size = int(edges.size - 1)
    scale = max(float(np.max(np.abs(edges))) if edges.size else 0.0, 1.0e-300)
    monotonic_tolerance = 64.0 * np.finfo(np.float64).eps * scale
    if (
        edges.ndim != 1
        or size < 2
        or departure.shape != edges.shape
        or not np.all(np.isfinite(edges))
        or not np.all(np.isfinite(departure))
        or np.any(np.diff(edges) <= 0.0)
        # Match the production CR1 remap's monotonicity contract exactly so a
        # trace it accepts cannot be rejected only by this diagnostic matrix.
        or np.any(np.diff(departure) < -monotonic_tolerance)
        or np.any(departure < edges[0])
        or np.any(departure > edges[-1])
    ):
        raise FrozenSemigroupDecompositionError("cannot build sparse CR1 map from invalid faces")
    widths = np.diff(edges)
    indptr = [0]
    columns: list[int] = []
    weights: list[float] = []
    for destination in range(size):
        lower = float(departure[destination])
        upper = float(departure[destination + 1])
        if upper > lower:
            source = int(np.searchsorted(edges, lower, side="right") - 1)
            source = min(max(source, 0), size - 1)
            while source < size and float(edges[source]) < upper:
                overlap = min(upper, float(edges[source + 1])) - max(lower, float(edges[source]))
                if overlap > 0.0:
                    columns.append(source)
                    weights.append(float(overlap / widths[source]))
                source += 1
        indptr.append(len(columns))
    return SparseCR1Map(
        indptr=np.asarray(indptr, dtype=np.int64),
        indices=np.asarray(columns, dtype=np.int64),
        data=np.asarray(weights, dtype=np.float64),
        size=size,
    )


def decompose_frozen_cr1(
    *,
    edges_m: NDArray[np.float64] | np.ndarray,
    initial_cells: NDArray[np.float64] | np.ndarray,
    h_s: float,
    velocity_m_s: VelocityFunction,
) -> FrozenSemigroupDecomposition:
    """Run the A/B/C frozen autonomous CR1 decomposition in memory only."""

    edges, cells, h = _validate_problem(edges_m, initial_cells, h_s)
    direct_trace = _trace_on_domain_edges(edges, dt_s=h, velocity_m_s=velocity_m_s)
    direct = _remap_path("DIRECT", edges=edges, old_cells=cells, trace=direct_trace)
    half = 0.5 * h
    first_half_trace = _trace_on_domain_edges(edges, dt_s=half, velocity_m_s=velocity_m_s)
    first_half = _remap_path("FIRST_HALF", edges=edges, old_cells=cells, trace=first_half_trace)
    # Compute the production-equivalent canonical half grid before asking the
    # new physical-query path to construct B.  Thus B's inability to form a
    # valid map cannot erase the already completed C evidence.
    second_half_trace = _trace_on_domain_edges(edges, dt_s=half, velocity_m_s=velocity_m_s)
    sequential = _remap_path(
        "SEQUENTIAL_HALF_STEPS",
        edges=edges,
        old_cells=first_half.cells,
        trace=second_half_trace,
        intermediate_cells=first_half.cells,
        first_half_trace=first_half_trace,
        second_half_trace=second_half_trace,
    )
    composed_trace, composed_query_audit = _composed_half_trace(
        edges,
        dt_s=h,
        velocity_m_s=velocity_m_s,
        first_half_trace=first_half_trace,
    )
    composed = _remap_path("COMPOSED_FLOW_SINGLE_REMAP", edges=edges, old_cells=cells, trace=composed_trace)
    d_total = np.asarray(sequential.cells - direct.cells, dtype=np.float64)
    d_trace = np.asarray(composed.cells - direct.cells, dtype=np.float64)
    d_remap = np.asarray(sequential.cells - composed.cells, dtype=np.float64)
    residual = np.asarray(d_total - d_trace - d_remap, dtype=np.float64)
    P_h = sparse_cr1_map(edges, direct.trace.departure_faces_m)
    P_half_1 = sparse_cr1_map(edges, first_half_trace.departure_faces_m)
    P_half_2 = sparse_cr1_map(edges, second_half_trace.departure_faces_m)
    P_compflow = sparse_cr1_map(edges, composed.trace.departure_faces_m)
    P_sequential = P_half_2.compose(P_half_1)
    return FrozenSemigroupDecomposition(
        h_s=h,
        direct=direct,
        composed_flow=composed,
        sequential=sequential,
        d_total=d_total,
        d_trace=d_trace,
        d_remap=d_remap,
        additive_residual=residual,
        P_h=P_h,
        P_half_1=P_half_1,
        P_half_2=P_half_2,
        P_compflow=P_compflow,
        P_sequential=P_sequential,
        composed_query_audit=composed_query_audit,
    )


def _cell_moment_weights(edges_m: NDArray[np.float64]) -> NDArray[np.float64]:
    edges = np.asarray(edges_m, dtype=np.float64)
    widths = np.diff(edges)
    result = np.empty((4, widths.size), dtype=np.float64)
    for order in range(4):
        result[order] = (edges[1:] ** (order + 1) - edges[:-1] ** (order + 1)) / ((order + 1) * widths)
    return result


def additive_residual_metrics(
    residual: NDArray[np.float64] | np.ndarray,
    *,
    reference_cells: NDArray[np.float64] | np.ndarray,
) -> dict[str, float]:
    """Report raw and scale-relative cellwise additive closure residuals."""

    value = np.asarray(residual, dtype=np.float64)
    reference = np.asarray(reference_cells, dtype=np.float64)
    if value.shape != reference.shape or not np.all(np.isfinite(value)):
        raise FrozenSemigroupDecompositionError("additive residual is invalid")
    l1 = float(np.sum(np.abs(value), dtype=np.float64))
    linf = float(np.max(np.abs(value))) if value.size else 0.0
    scale_l1 = max(float(np.sum(np.abs(reference), dtype=np.float64)), 1.0e-300)
    scale_linf = max(float(np.max(np.abs(reference))), 1.0e-300)
    return {
        "L1_abs": l1,
        "Linf_abs": linf,
        "L1_relative": l1 / scale_l1,
        "Linf_relative": linf / scale_linf,
    }


def defect_metrics(
    left_cells: NDArray[np.float64] | np.ndarray,
    right_cells: NDArray[np.float64] | np.ndarray,
    *,
    edges_m: NDArray[np.float64] | np.ndarray,
) -> dict[str, float | list[float]]:
    """Physical population metrics plus signed and absolute M0--M3 defects.

    ``left - right`` is the signed defect convention used by the three-path
    decomposition.  The returned absolute weighted moments are deliberately
    computed before summation so cancellation cannot conceal an error.
    """

    left = np.asarray(left_cells, dtype=np.float64)
    right = np.asarray(right_cells, dtype=np.float64)
    edges = np.asarray(edges_m, dtype=np.float64)
    if (
        left.shape != right.shape
        or left.shape != (edges.size - 1,)
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
        or np.any(left < 0.0)
        or np.any(right < 0.0)
    ):
        raise FrozenSemigroupDecompositionError("defect metric populations are invalid")
    delta = np.asarray(left - right, dtype=np.float64)
    absolute = np.abs(delta)
    left_total = float(np.sum(left, dtype=np.float64))
    right_total = float(np.sum(right, dtype=np.float64))
    l1_abs = float(np.sum(absolute, dtype=np.float64))
    linf_abs = float(np.max(absolute)) if absolute.size else 0.0
    denominator = max(right_total, 1.0e-300)
    weights = _cell_moment_weights(edges)
    signed = np.sum(weights * delta[np.newaxis, :], axis=1, dtype=np.float64)
    weighted_absolute = np.sum(weights * absolute[np.newaxis, :], axis=1, dtype=np.float64)
    left_cdf = np.zeros_like(left) if left_total <= 0.0 else np.cumsum(left, dtype=np.float64) / left_total
    right_cdf = np.zeros_like(right) if right_total <= 0.0 else np.cumsum(right, dtype=np.float64) / right_total
    left_radii, left_weights = positive_cell_quadrature(edges, left, 2)
    right_radii, right_weights = positive_cell_quadrature(edges, right, 2)
    wasserstein = float(discrete_wasserstein_distance(left_radii, left_weights, right_radii, right_weights))
    result: dict[str, float | list[float]] = {
        "population_L1_abs": l1_abs,
        "population_Linf_abs": linf_abs,
        "population_relative_L1": l1_abs / denominator,
        "population_relative_Linf": linf_abs / max(float(np.max(np.abs(right))), 1.0e-300),
        "CDF_max_error": float(np.max(np.abs(left_cdf - right_cdf))) if left.size else 0.0,
        "Wasserstein_m": wasserstein,
        "signed_moments_M0_to_M3": [float(value) for value in signed],
        "absolute_weighted_M0_to_M3": [float(value) for value in weighted_absolute],
    }
    for order in range(4):
        result[f"signed_M{order}"] = float(signed[order])
        result[f"absolute_weighted_M{order}"] = float(weighted_absolute[order])
    return result


def defect_cosine(
    left: NDArray[np.float64] | np.ndarray, right: NDArray[np.float64] | np.ndarray
) -> float | None:
    """Return the Euclidean cosine or ``None`` if either defect is zero."""

    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm == 0.0 or second_norm == 0.0:
        return None
    return float(np.dot(first, second) / (first_norm * second_norm))


def ratio(numerator: float, denominator: float) -> float | None:
    """Return a transparent ratio without forcing cancellation to percentages."""

    if denominator == 0.0:
        return None if numerator == 0.0 else math.inf
    return float(numerator / denominator)


def face_flow_rows(
    decomposition: FrozenSemigroupDecomposition,
    *,
    edges_m: NDArray[np.float64] | np.ndarray,
    near_boundary_fraction: float = 1.0e-3,
) -> list[dict[str, object]]:
    """Compare direct and composed physical departure maps face by face."""

    edges = np.asarray(edges_m, dtype=np.float64)
    if not math.isfinite(float(near_boundary_fraction)) or not 0.0 < float(near_boundary_fraction) < 1.0:
        raise ValueError("near-boundary fraction must lie in (0, 1)")
    direct = np.asarray(decomposition.direct.trace.departure_faces_m, dtype=np.float64)
    composed = np.asarray(decomposition.composed_flow.trace.departure_faces_m, dtype=np.float64)
    direct_source = np.asarray(decomposition.direct.partition.source_cell_indices, dtype=np.int64)
    composed_source = np.asarray(decomposition.composed_flow.partition.source_cell_indices, dtype=np.int64)
    if direct.shape != edges.shape or composed.shape != edges.shape:
        raise FrozenSemigroupDecompositionError("face-flow arrays do not match frozen grid")
    widths = np.diff(edges)
    rows: list[dict[str, object]] = []
    for face in range(edges.size):
        source_a = int(direct_source[face])
        source_b = int(composed_source[face])
        width_a = float(widths[source_a])
        width_b = float(widths[source_b])
        direct_distance = min(
            abs(float(direct[face]) - float(edges[source_a])),
            abs(float(edges[source_a + 1]) - float(direct[face])),
        )
        composed_distance = min(
            abs(float(composed[face]) - float(edges[source_b])),
            abs(float(edges[source_b + 1]) - float(composed[face])),
        )
        # These values are radii in metres.  A unit-scale floor would make a
        # round-off diagnostic into an artificial ~1e-14 m neighbourhood on
        # this nanometre grid, so the scale stays in physical-radius units.
        machine_distance = 128.0 * np.finfo(np.float64).eps * max(
            abs(float(edges[0])),
            abs(float(edges[-1])),
            abs(float(direct[face])),
            abs(float(composed[face])),
            float(np.finfo(np.float64).tiny),
        )
        geometry_distance = float(near_boundary_fraction) * min(width_a, width_b)
        threshold = max(machine_distance, geometry_distance)
        delta = float(composed[face] - direct[face])
        rows.append({
            "h_s": float(decomposition.h_s),
            "face_index": face,
            "arrival_face_radius_m": float(edges[face]),
            "direct_departure_radius_m": float(direct[face]),
            "composed_departure_radius_m": float(composed[face]),
            "delta_departure_radius_m": delta,
            "direct_source_cell_index": source_a,
            "composed_source_cell_index": source_b,
            "direct_source_cell_width_m": width_a,
            "composed_source_cell_width_m": width_b,
            "normalization_cell_width_m": min(width_a, width_b),
            "delta_departure_over_local_cell_width": delta / min(width_a, width_b),
            "source_cell_changed": bool(source_a != source_b),
            "direct_nearest_source_boundary_distance_m": direct_distance,
            "composed_nearest_source_boundary_distance_m": composed_distance,
            "near_boundary_machine_distance_m": machine_distance,
            "near_boundary_geometry_distance_m": geometry_distance,
            "near_boundary_threshold_m": threshold,
            "S_near_boundary": bool(direct_distance <= threshold or composed_distance <= threshold),
        })
    return rows


def changed_face_indices(decomposition: FrozenSemigroupDecomposition) -> NDArray[np.int64]:
    """Return exactly the direct/composed source-cell topology changes."""

    return np.flatnonzero(
        np.asarray(decomposition.direct.partition.source_cell_indices, dtype=np.int64)
        != np.asarray(decomposition.composed_flow.partition.source_cell_indices, dtype=np.int64)
    ).astype(np.int64)


def support_from_faces(
    faces: Iterable[int], *, cell_count: int, halo: int = 0
) -> NDArray[np.bool_]:
    """Destination-cell support for changed faces, expanded by a fixed halo."""

    if cell_count < 1 or halo < 0:
        raise ValueError("support dimensions are invalid")
    mask = np.zeros(cell_count, dtype=bool)
    for face_value in faces:
        face = int(face_value)
        for base in (face - 1, face):
            for cell in range(base - int(halo), base + int(halo) + 1):
                if 0 <= cell < cell_count:
                    mask[cell] = True
    return mask


def _row_weight_map(operator: SparseCR1Map, row: int) -> dict[int, float]:
    indices, values = operator.row(row)
    return {int(index): float(value) for index, value in zip(indices, values)}


def _sequential_intermediate_cells(
    half_2: SparseCR1Map, half_1: SparseCR1Map, *, final_cell: int, source_cell: int
) -> list[int]:
    intermediate, outer = half_2.row(final_cell)
    result: list[int] = []
    for middle, outer_weight in zip(intermediate, outer):
        source, inner = half_1.row(int(middle))
        for column, inner_weight in zip(source, inner):
            if int(column) == int(source_cell) and float(outer_weight) * float(inner_weight) != 0.0:
                result.append(int(middle))
                break
    return sorted(set(result))


def projection_connectivity_rows(
    decomposition: FrozenSemigroupDecomposition,
    *,
    coefficient_rtol: float = 64.0 * np.finfo(np.float64).eps,
) -> tuple[list[dict[str, object]], NDArray[np.int64]]:
    """Compare old→intermediate→final mixing with one-remap composed flow.

    A final cell enters ``S_projection_changed_cells`` when its old-source
    support differs, or when the same support has a coefficient difference
    above a declared floating-point round-off scale.  No population value is
    consulted for this topology/operator set.
    """

    if coefficient_rtol <= 0.0 or not math.isfinite(coefficient_rtol):
        raise ValueError("coefficient round-off tolerance must be finite and positive")
    sequential = decomposition.P_sequential
    composed = decomposition.P_compflow
    rows: list[dict[str, object]] = []
    changed_cells: list[int] = []
    for final_cell in range(composed.size):
        direct_weights = _row_weight_map(composed, final_cell)
        sequential_weights = _row_weight_map(sequential, final_cell)
        sources = sorted(set(direct_weights).union(sequential_weights))
        direct_support = set(direct_weights)
        sequential_support = set(sequential_weights)
        l1_delta = float(sum(abs(sequential_weights.get(source, 0.0) - direct_weights.get(source, 0.0)) for source in sources))
        scale = max(
            float(sum(abs(value) for value in direct_weights.values())),
            float(sum(abs(value) for value in sequential_weights.values())),
            1.0,
        )
        support_changed = direct_support != sequential_support
        coefficient_changed = l1_delta > coefficient_rtol * scale
        projection_changed = bool(support_changed or coefficient_changed)
        if projection_changed:
            changed_cells.append(final_cell)
        for source in sources:
            direct_weight = float(direct_weights.get(source, 0.0))
            sequential_weight = float(sequential_weights.get(source, 0.0))
            intermediate = _sequential_intermediate_cells(
                decomposition.P_half_2, decomposition.P_half_1,
                final_cell=final_cell, source_cell=source,
            )
            rows.append({
                "h_s": float(decomposition.h_s),
                "final_destination_cell": final_cell,
                "old_source_cell": source,
                "composed_flow_direct_weight": direct_weight,
                "sequential_weight": sequential_weight,
                "weight_delta_sequential_minus_composed": sequential_weight - direct_weight,
                "composed_flow_link": bool(source in direct_support),
                "sequential_link": bool(source in sequential_support),
                "intermediate_cells_json": "[" + ",".join(str(value) for value in intermediate) + "]",
                "intermediate_path_count": len(intermediate),
                "source_support_changed": support_changed,
                "row_weight_L1_difference": l1_delta,
                "coefficient_changed_above_roundoff": coefficient_changed,
                "S_projection_changed_cell": projection_changed,
            })
        if not sources:
            rows.append({
                "h_s": float(decomposition.h_s),
                "final_destination_cell": final_cell,
                "old_source_cell": "",
                "composed_flow_direct_weight": 0.0,
                "sequential_weight": 0.0,
                "weight_delta_sequential_minus_composed": 0.0,
                "composed_flow_link": False,
                "sequential_link": False,
                "intermediate_cells_json": "[]",
                "intermediate_path_count": 0,
                "source_support_changed": support_changed,
                "row_weight_L1_difference": l1_delta,
                "coefficient_changed_above_roundoff": coefficient_changed,
                "S_projection_changed_cell": projection_changed,
            })
    return rows, np.asarray(changed_cells, dtype=np.int64)


def support_pairing_metrics(
    defect: NDArray[np.float64] | np.ndarray,
    *,
    edges_m: NDArray[np.float64] | np.ndarray,
    support: NDArray[np.bool_] | np.ndarray,
) -> dict[str, object]:
    """Pair a signed physical error with one topology-defined cell support."""

    delta = np.asarray(defect, dtype=np.float64)
    edges = np.asarray(edges_m, dtype=np.float64)
    mask = np.asarray(support, dtype=bool)
    if delta.shape != mask.shape or delta.shape != (edges.size - 1,) or not np.all(np.isfinite(delta)):
        raise FrozenSemigroupDecompositionError("support pairing inputs are invalid")
    absolute = np.abs(delta)
    total_l1 = float(np.sum(absolute, dtype=np.float64))
    support_l1 = float(np.sum(absolute[mask], dtype=np.float64))
    complement = ~mask
    weights = _cell_moment_weights(edges)
    absolute_moments = weights * absolute[np.newaxis, :]
    total_moments = np.sum(absolute_moments, axis=1, dtype=np.float64)
    support_moments = np.sum(absolute_moments[:, mask], axis=1, dtype=np.float64)
    result: dict[str, object] = {
        "support_cell_count": int(np.count_nonzero(mask)),
        "complement_cell_count": int(np.count_nonzero(complement)),
        "total_population_L1_abs": total_l1,
        "support_population_L1_abs": support_l1,
        "complement_population_L1_abs": float(np.sum(absolute[complement], dtype=np.float64)),
        "support_population_L1_fraction": support_l1 / total_l1 if total_l1 > 0.0 else 0.0,
        "support_population_Linf_abs": float(np.max(absolute[mask])) if np.any(mask) else 0.0,
        "complement_population_Linf_abs": float(np.max(absolute[complement])) if np.any(complement) else 0.0,
    }
    for order in range(4):
        total = float(total_moments[order])
        local = float(support_moments[order])
        result[f"total_absolute_weighted_M{order}"] = total
        result[f"support_absolute_weighted_M{order}"] = local
        result[f"complement_absolute_weighted_M{order}"] = total - local
        result[f"support_absolute_weighted_M{order}_fraction"] = local / total if total > 0.0 else 0.0
    return result


def support_expansion_rows(
    defect: NDArray[np.float64] | np.ndarray,
    *,
    edges_m: NDArray[np.float64] | np.ndarray,
    seed_cells: Iterable[int],
    kind: str,
    h_s: float,
) -> list[dict[str, object]]:
    """Return exact support plus the required ±1 and ±2 cell halos."""

    values = np.asarray(defect, dtype=np.float64)
    seeds = sorted({int(value) for value in seed_cells if 0 <= int(value) < values.size})
    rows: list[dict[str, object]] = []
    for halo in (0, 1, 2):
        mask = np.zeros(values.size, dtype=bool)
        for seed in seeds:
            mask[max(0, seed - halo) : min(values.size, seed + halo + 1)] = True
        metrics = support_pairing_metrics(values, edges_m=edges_m, support=mask)
        rows.append({
            "h_s": float(h_s),
            "defect": str(kind),
            "halo_cells": halo,
            "seed_cells_json": "[" + ",".join(str(value) for value in seeds) + "]",
            **metrics,
        })
    return rows


def operator_audit(
    decomposition: FrozenSemigroupDecomposition,
    *,
    initial_cells: NDArray[np.float64] | np.ndarray,
) -> dict[str, object]:
    """Audit state/operator decomposition agreement and exact sparse norms."""

    source = np.asarray(initial_cells, dtype=np.float64)
    trace_operator = decomposition.P_compflow.difference(decomposition.P_h)
    remap_operator = decomposition.P_sequential.difference(decomposition.P_compflow)
    state_direct = decomposition.direct.cells
    state_composed = decomposition.composed_flow.cells
    state_sequential = decomposition.sequential.cells
    operator_direct = decomposition.P_h.apply(source)
    operator_composed = decomposition.P_compflow.apply(source)
    operator_sequential = decomposition.P_sequential.apply(source)
    trace_state = state_composed - state_direct
    remap_state = state_sequential - state_composed
    trace_operator_state = trace_operator.apply(source)
    remap_operator_state = remap_operator.apply(source)

    def residual(values: NDArray[np.float64]) -> dict[str, float]:
        return {"L1_abs": float(np.sum(np.abs(values), dtype=np.float64)), "Linf_abs": float(np.max(np.abs(values)))}

    return {
        "P_h_nnz": decomposition.P_h.nnz,
        "P_half_1_nnz": decomposition.P_half_1.nnz,
        "P_half_2_nnz": decomposition.P_half_2.nnz,
        "P_compflow_nnz": decomposition.P_compflow.nnz,
        "P_sequential_nnz": decomposition.P_sequential.nnz,
        "operator_trace_1_norm": trace_operator.one_norm(),
        "operator_trace_infinity_norm": trace_operator.infinity_norm(),
        "operator_remap_1_norm": remap_operator.one_norm(),
        "operator_remap_infinity_norm": remap_operator.infinity_norm(),
        "P_h_state_residual": residual(operator_direct - state_direct),
        "P_compflow_state_residual": residual(operator_composed - state_composed),
        "P_sequential_state_residual": residual(operator_sequential - state_sequential),
        "trace_operator_state_residual": residual(trace_operator_state - trace_state),
        "remap_operator_state_residual": residual(remap_operator_state - remap_state),
    }


def search_zero_event_control(
    *,
    edges_m: NDArray[np.float64] | np.ndarray,
    initial_cells: NDArray[np.float64] | np.ndarray,
    velocity_m_s: VelocityFunction,
    starting_h_s: float,
    maximum_halvings: int = 64,
) -> tuple[FrozenSemigroupDecomposition | None, list[dict[str, object]]]:
    """Find a dyadic frozen h with no direct/composed changed source face."""

    if maximum_halvings < 0:
        raise ValueError("maximum halvings must be non-negative")
    candidate = float(starting_h_s)
    rows: list[dict[str, object]] = []
    for halving in range(int(maximum_halvings) + 1):
        try:
            decomposition = decompose_frozen_cr1(
                edges_m=edges_m, initial_cells=initial_cells, h_s=candidate, velocity_m_s=velocity_m_s
            )
        except FrozenSemigroupDecompositionError as error:
            rows.append({"h_s": candidate, "halving": halving, "status": "TRACE_FAILURE", "error": str(error), "changed_face_count": ""})
            candidate *= 0.5
            continue
        count = int(changed_face_indices(decomposition).size)
        rows.append({"h_s": candidate, "halving": halving, "status": "EVALUATED", "error": "", "changed_face_count": count})
        if count == 0:
            return decomposition, rows
        candidate *= 0.5
    return None, rows


def synthetic_single_event_control(
    *,
    bins: int = 16,
    maximum_scan_points: int = 257,
) -> tuple[FrozenSemigroupDecomposition | None, dict[str, object]]:
    """Algorithmically find a smooth autonomous toy case with one/few events.

    It uses the identical CR1 trace/remap and a smooth positive velocity on a
    logarithmic radius grid.  It is intentionally not a PbTe state and cannot
    be used to infer material behaviour.
    """

    if bins < 6 or maximum_scan_points < 8:
        raise ValueError("synthetic control grid/scan are too small")
    lower, upper = 5.0e-9, 5.0e-8
    edges = np.geomspace(lower, upper, int(bins) + 1, dtype=np.float64)
    centres = np.sqrt(edges[:-1] * edges[1:])
    log_centres = np.log(centres)
    smooth = np.exp(-0.5 * ((log_centres - float(np.mean(log_centres))) / 0.45) ** 2)
    cells = np.asarray(smooth / np.sum(smooth, dtype=np.float64) * 1.0e18, dtype=np.float64)

    def velocity(radii: NDArray[np.float64]) -> NDArray[np.float64]:
        normalized = (np.asarray(radii, dtype=np.float64) - lower) / (upper - lower)
        return 3.0e-10 * (0.15 + normalized * normalized)

    # A generic geometric scan almost never lands inside the several-ulp-wide
    # interval where two high-order flow maps select opposite sides of a cell
    # boundary.  We therefore locate a *geometric* crossing first, using a
    # predeclared arrival face and source boundary, then inspect a fixed ULP
    # neighbourhood.  This is an algorithmic diagnostic construction—not a
    # population-error search and not a production scalar-root procedure.
    scale = float(np.min(np.diff(edges)) / 3.0e-10)
    coarse_h = np.geomspace(
        1.0e-3 * scale,
        512.0 * scale,
        int(maximum_scan_points),
        dtype=np.float64,
    )
    target_face = max(2, min(int(bins) - 1, int(round(0.75 * int(bins)))))
    target_boundary = max(1, min(int(bins) - 1, int(round(0.15 * int(bins)))))
    target_radius = float(edges[target_boundary])

    coarse_values: list[tuple[float, float]] = []
    for h_s in coarse_h:
        try:
            trace = _trace_on_domain_edges(edges, dt_s=float(h_s), velocity_m_s=velocity)
        except FrozenSemigroupDecompositionError:
            continue
        coarse_values.append((float(h_s), float(trace.departure_faces_m[target_face] - target_radius)))

    bracket: tuple[float, float] | None = None
    for (left_h, left_value), (right_h, right_value) in zip(coarse_values, coarse_values[1:]):
        if left_value == 0.0:
            bracket = (left_h, left_h)
            break
        if right_value == 0.0:
            bracket = (right_h, right_h)
            break
        if (left_value < 0.0) != (right_value < 0.0):
            bracket = (left_h, right_h)
            break

    candidates: list[float] = []
    if bracket is not None:
        lower_h, upper_h = bracket
        if lower_h != upper_h:
            lower_value = next(value for h_s, value in coarse_values if h_s == lower_h)
            # Trace-only bisection merely localizes the prescribed geometric
            # crossing; no solver state, closure, or numerical policy is
            # altered by this synthetic control.
            for _ in range(80):
                midpoint = 0.5 * (lower_h + upper_h)
                trace = _trace_on_domain_edges(edges, dt_s=midpoint, velocity_m_s=velocity)
                midpoint_value = float(trace.departure_faces_m[target_face] - target_radius)
                if midpoint_value == 0.0:
                    lower_h = upper_h = midpoint
                    break
                if (midpoint_value < 0.0) == (lower_value < 0.0):
                    lower_h, lower_value = midpoint, midpoint_value
                else:
                    upper_h = midpoint
        crossing_h = 0.5 * (lower_h + upper_h)
        candidates.append(float(crossing_h))
        below = float(crossing_h)
        above = float(crossing_h)
        # The ULP window is fixed before inspecting any population defect.
        for _ in range(512):
            below = float(np.nextafter(below, -np.inf))
            above = float(np.nextafter(above, np.inf))
            candidates.extend((below, above))

    # Keep the broad scan as a fail-closed fallback for a platform where the
    # target crossing is not bracketed, but never rank candidates by error.
    candidates.extend(float(value) for value in coarse_h)
    seen: set[float] = set()
    best: tuple[int, float, FrozenSemigroupDecomposition] | None = None
    scanned = 0
    for h_s in candidates:
        if h_s in seen:
            continue
        seen.add(h_s)
        try:
            decomposition = decompose_frozen_cr1(
                edges_m=edges,
                initial_cells=cells,
                h_s=h_s,
                velocity_m_s=velocity,
            )
        except FrozenSemigroupDecompositionError:
            continue
        scanned += 1
        count = int(changed_face_indices(decomposition).size)
        if 1 <= count <= 4:
            candidate = (count, float(h_s), decomposition)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
    if best is None:
        return None, {
            "status": "NO_SYNTHETIC_TOPOLOGY_EVENT_FOUND",
            "bins": int(bins),
            "scan_points": int(maximum_scan_points),
            "successful_candidates": scanned,
            "target_face_index": target_face,
            "target_source_boundary_index": target_boundary,
            "physical_material_state_used": False,
        }
    count, h_s, decomposition = best
    return decomposition, {
        "status": "SYNTHETIC_SINGLE_OR_FEW_EVENT_FOUND",
        "bins": int(bins),
        "scan_points": int(maximum_scan_points),
        "successful_candidates": scanned,
        "selected_h_s": h_s,
        "changed_face_count": count,
        "target_face_index": target_face,
        "target_source_boundary_index": target_boundary,
        "physical_material_state_used": False,
    }
