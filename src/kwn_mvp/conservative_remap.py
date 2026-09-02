"""Conservative piecewise-constant remapping for characteristic KWN steps.

The routines in this module operate on *cell-integrated* particle number,
not point samples.  They are intentionally narrow: CR1 is a deterministic,
piecewise-constant cumulative-distribution remap.  Higher-order
reconstruction belongs in a separately qualified CR2 implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from numpy.typing import NDArray


class ConservativeRemapError(RuntimeError):
    """Raised when a proposed characteristic map is not a valid CR1 remap."""


@dataclass(frozen=True)
class CharacteristicTrace:
    """A bounded higher-order backward map from arrival to departure faces."""

    arrival_faces_m: NDArray[np.float64]
    departure_faces_m: NDArray[np.float64]
    midpoint_faces_m: NDArray[np.float64]
    lower_no_inflow_face_count: int
    upper_no_inflow_face_count: int


@dataclass(frozen=True)
class ConservativeRemapResult:
    """One CR1 remap and its only admissible domain-boundary losses."""

    cell_number_m3: NDArray[np.float64]
    lower_number_loss_m3: float
    upper_number_loss_m3: float
    old_number_m3: float
    new_number_m3: float
    conservation_residual_m3: float


def _validate_edges_and_numbers(
    edges_m: NDArray[np.float64] | np.ndarray,
    cell_number_m3: NDArray[np.float64] | np.ndarray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    edges = np.asarray(edges_m, dtype=np.float64)
    number = np.asarray(cell_number_m3, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 3 or not np.all(np.isfinite(edges)):
        raise ConservativeRemapError("cell edges must be a finite one-dimensional array")
    if np.any(edges <= 0.0) or np.any(np.diff(edges) <= 0.0):
        raise ConservativeRemapError("cell edges must be strictly increasing and positive")
    if number.ndim != 1 or number.shape != (edges.size - 1,):
        raise ConservativeRemapError("cell-integrated number must match the edge count")
    if not np.all(np.isfinite(number)) or np.any(number < 0.0):
        raise ConservativeRemapError("cell-integrated number must be finite and non-negative")
    return edges, number


def trace_departure_faces_rk2(
    arrival_faces_m: NDArray[np.float64] | np.ndarray,
    *,
    dt_s: float,
    velocity_m_s: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    lower_radius_m: float,
    upper_radius_m: float,
) -> CharacteristicTrace:
    """Trace all faces with a deterministic higher-order characteristic map.

    For a frozen midpoint matrix state the growth law is autonomous.  Rather
    than taking an unstable explicit stage at the singular physical ``Rmin``
    edge, this routine writes the same backward ODE in the radius coordinate,

    ``d tau / d R = -1 / G(R)``,

    and integrates its time of flight with composite two-node
    Gauss--Legendre quadrature.  That is a fourth-order deterministic ODE
    trace (and therefore exceeds the required RK2 order) followed by a
    monotone inverse map.  It is specifically important that an intermediate
    characteristic stage is *not* clipped to ``Rmax``: doing so can turn the
    frozen, strongly dissolving ``Rmin`` velocity into spurious zero loss.

    The physical domain has no external inflow.  If the inverse trace reaches
    a physical endpoint before the requested time, its departure location is
    that endpoint.  This is a domain-coverage condition, not a density clamp:
    the subsequent cumulative remap supplies no mass from outside the
    resolved interval.

    The public function name is retained for the original CR1 call site.  The
    implementation is a higher-order replacement for the initial midpoint
    prototype, not an explicit donor-CFL method.
    """

    arrival = np.asarray(arrival_faces_m, dtype=np.float64)
    dt = float(dt_s)
    lower = float(lower_radius_m)
    upper = float(upper_radius_m)
    if arrival.ndim != 1 or arrival.size < 3 or not np.all(np.isfinite(arrival)):
        raise ConservativeRemapError("arrival faces must be finite and one-dimensional")
    if np.any(np.diff(arrival) <= 0.0):
        raise ConservativeRemapError("arrival faces must be strictly increasing")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ConservativeRemapError("characteristic timestep must be finite and positive")
    if not math.isfinite(lower) or not math.isfinite(upper) or not 0.0 < lower < upper:
        raise ConservativeRemapError("characteristic domain bounds are invalid")
    if not np.array_equal(arrival, np.clip(arrival, lower, upper)):
        raise ConservativeRemapError("arrival faces must equal the frozen physical domain edges")

    first = np.asarray(velocity_m_s(arrival), dtype=np.float64)
    if first.shape != arrival.shape or not np.all(np.isfinite(first)):
        raise ConservativeRemapError("arrival-face characteristic velocity is invalid")

    # Preserve the exact zero-mobility contract before constructing a refined
    # trace mesh.  In particular, CR1 must not alter an accepted binary64
    # cell measure through an otherwise harmless CDF/reduction round trip.
    if np.array_equal(first, np.zeros_like(first)):
        return CharacteristicTrace(
            arrival_faces_m=arrival.copy(),
            departure_faces_m=arrival.copy(),
            midpoint_faces_m=arrival.copy(),
            lower_no_inflow_face_count=0,
            upper_no_inflow_face_count=0,
        )

    # Constant translation is both an exact analytic characteristic and a
    # useful CR1 qualification path.  Keep it exact rather than introducing
    # avoidable quadrature/interpolation round-off in that benchmark.
    if np.array_equal(first, np.full_like(first, first[0])):
        midpoint_raw = arrival - 0.5 * dt * first
        departure_raw = arrival - dt * first
        midpoint = np.clip(midpoint_raw, lower, upper)
        departure = np.clip(departure_raw, lower, upper)
        lower_no_inflow = int(np.count_nonzero(departure_raw < lower))
        upper_no_inflow = int(np.count_nonzero(departure_raw > upper))
    else:
        departure, lower_no_inflow, upper_no_inflow = _trace_autonomous_time_of_flight(
            arrival,
            duration_s=dt,
            velocity_m_s=velocity_m_s,
            lower_radius_m=lower,
            upper_radius_m=upper,
        )
        midpoint, _, _ = _trace_autonomous_time_of_flight(
            arrival,
            duration_s=0.5 * dt,
            velocity_m_s=velocity_m_s,
            lower_radius_m=lower,
            upper_radius_m=upper,
        )

    # A non-monotone map would double-cover or omit a physical interval.  Do
    # not repair it by reordering faces: reject it before any state mutation.
    scale = max(float(np.max(np.abs(arrival))), 1.0e-300)
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    if np.any(np.diff(departure) < -tolerance):
        raise ConservativeRemapError("backward characteristic face map is non-monotone")
    return CharacteristicTrace(
        arrival_faces_m=arrival.copy(),
        departure_faces_m=departure,
        midpoint_faces_m=midpoint,
        lower_no_inflow_face_count=lower_no_inflow,
        upper_no_inflow_face_count=upper_no_inflow,
    )


_TRACE_SUBCELLS_PER_CELL = 16
_ROOT_TAIL_SEGMENTS = 16
_GAUSS_ABSCISSA = 1.0 / math.sqrt(3.0)


def _log_subdivided_faces(
    edges_m: NDArray[np.float64], *, subcells_per_cell: int
) -> NDArray[np.float64]:
    """Return a deterministic log-radius trace mesh containing every face."""

    fractions = np.arange(subcells_per_cell, dtype=np.float64) / float(subcells_per_cell)
    logarithmic = np.log(edges_m)
    segments = np.exp(
        logarithmic[:-1, np.newaxis]
        + (logarithmic[1:] - logarithmic[:-1])[:, np.newaxis] * fractions[np.newaxis, :]
    )
    refined = np.concatenate((segments.ravel(), np.asarray([edges_m[-1]], dtype=np.float64)))
    # ``exp(log(edge))`` is not reliably bitwise equal to the frozen edge.
    # Restore every canonical face exactly, both for the zero-velocity
    # contract and for unambiguous CDF interval locations.
    refined[::subcells_per_cell] = edges_m
    return refined


def _checked_velocity(
    velocity_m_s: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    radii_m: NDArray[np.float64],
    *,
    label: str,
) -> NDArray[np.float64]:
    values = np.asarray(velocity_m_s(radii_m), dtype=np.float64)
    if values.shape != radii_m.shape or not np.all(np.isfinite(values)):
        raise ConservativeRemapError(f"{label} characteristic velocity is invalid")
    return values


def _gauss_time_of_flight(
    coordinates_m: NDArray[np.float64],
    *,
    expected_sign: float,
    velocity_m_s: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Integrate ``|d tau / d R|`` on same-sign characteristic intervals."""

    if coordinates_m.size < 2 or np.any(np.diff(coordinates_m) <= 0.0):
        raise ConservativeRemapError("characteristic quadrature coordinates are not strictly increasing")
    return _gauss_time_of_flight_intervals(
        coordinates_m[:-1],
        coordinates_m[1:],
        expected_sign=expected_sign,
        velocity_m_s=velocity_m_s,
    )


def _gauss_time_of_flight_intervals(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    *,
    expected_sign: float,
    velocity_m_s: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Two-node Gauss time-of-flight integrals on independent intervals."""

    if left.shape != right.shape or left.ndim != 1 or np.any(right < left):
        raise ConservativeRemapError("characteristic quadrature intervals are invalid")
    midpoint = 0.5 * (left + right)
    half_width = 0.5 * (right - left)
    first_points = midpoint - _GAUSS_ABSCISSA * half_width
    second_points = midpoint + _GAUSS_ABSCISSA * half_width
    first_velocity = _checked_velocity(velocity_m_s, first_points, label="first Gauss stage")
    second_velocity = _checked_velocity(velocity_m_s, second_points, label="second Gauss stage")
    if np.any(np.sign(first_velocity) != expected_sign) or np.any(np.sign(second_velocity) != expected_sign):
        raise ConservativeRemapError(
            "characteristic quadrature crossed a stationary radius before its physical branch"
        )
    duration = half_width * (1.0 / np.abs(first_velocity) + 1.0 / np.abs(second_velocity))
    if not np.all(np.isfinite(duration)) or np.any(duration <= 0.0):
        # A zero-length inversion interval represents an exactly tabulated
        # cumulative time and is harmless.  All other intervals must carry a
        # finite positive physical flight time.
        if not np.all(np.isfinite(duration)) or np.any((duration <= 0.0) & (right > left)):
            raise ConservativeRemapError("characteristic time-of-flight quadrature is invalid")
    return duration


def _invert_time_of_flight(
    targets_s: NDArray[np.float64],
    cumulative_time_s: NDArray[np.float64],
    coordinates_m: NDArray[np.float64],
    *,
    expected_sign: float,
    velocity_m_s: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Invert a tabulated flight-time map with local Gauss/Newton refinement.

    Linear interpolation is sufficient to locate the enclosing trace
    interval, but it is only second-order as an inverse.  The safeguarded
    local solve below re-integrates that one interval with the same
    Gauss--Legendre rule and restores the intended high-order characteristic
    accuracy without globally refining every face.
    """

    targets = np.asarray(targets_s, dtype=np.float64)
    if targets.ndim != 1 or not np.all(np.isfinite(targets)):
        raise ConservativeRemapError("characteristic inversion targets are invalid")
    if cumulative_time_s.ndim != 1 or coordinates_m.ndim != 1 or cumulative_time_s.shape != coordinates_m.shape:
        raise ConservativeRemapError("characteristic cumulative trace table is invalid")
    if np.any(np.diff(cumulative_time_s) <= 0.0):
        raise ConservativeRemapError("characteristic cumulative trace time is not strictly increasing")
    if np.any(targets < cumulative_time_s[0]) or np.any(targets > cumulative_time_s[-1]):
        raise ConservativeRemapError("characteristic inversion target left its physical trace table")
    interval = np.searchsorted(cumulative_time_s, targets, side="right") - 1
    interval = np.clip(interval, 0, coordinates_m.size - 2)
    left = coordinates_m[interval]
    right = coordinates_m[interval + 1]
    cumulative_left = cumulative_time_s[interval]
    interval_time = cumulative_time_s[interval + 1] - cumulative_left
    fraction = (targets - cumulative_left) / interval_time
    estimate = left + fraction * (right - left)
    low = left.copy()
    high = right.copy()
    for _ in range(6):
        partial = _gauss_time_of_flight_intervals(
            left,
            estimate,
            expected_sign=expected_sign,
            velocity_m_s=velocity_m_s,
        )
        residual = cumulative_left + partial - targets
        high = np.where(residual > 0.0, estimate, high)
        low = np.where(residual <= 0.0, estimate, low)
        local_velocity = _checked_velocity(velocity_m_s, estimate, label="trace inversion")
        proposal = estimate - residual * np.abs(local_velocity)
        safe_proposal = (proposal > low) & (proposal < high) & np.isfinite(proposal)
        estimate = np.where(safe_proposal, proposal, 0.5 * (low + high))
    return estimate


def _stationary_root(
    *,
    lower_m: float,
    upper_m: float,
    lower_velocity: float,
    upper_velocity: float,
    velocity_m_s: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> float:
    """Locate one physical same-domain stationary radius without clipping it."""

    if lower_velocity == 0.0:
        return lower_m
    if upper_velocity == 0.0:
        return upper_m
    if math.copysign(1.0, lower_velocity) == math.copysign(1.0, upper_velocity):
        raise ConservativeRemapError("cannot bracket a stationary characteristic radius")
    left = float(lower_m)
    right = float(upper_m)
    left_velocity = float(lower_velocity)
    right_velocity = float(upper_velocity)
    for _ in range(80):
        midpoint = 0.5 * (left + right)
        if midpoint == left or midpoint == right:
            break
        midpoint_velocity = float(
            _checked_velocity(
                velocity_m_s, np.asarray([midpoint], dtype=np.float64), label="stationary-root"
            )[0]
        )
        if midpoint_velocity == 0.0:
            return midpoint
        if math.copysign(1.0, midpoint_velocity) == math.copysign(1.0, left_velocity):
            left = midpoint
            left_velocity = midpoint_velocity
        else:
            right = midpoint
            right_velocity = midpoint_velocity
    return 0.5 * (left + right)


def _stationary_tail_from_endpoint(
    *,
    stationary_radius_m: float,
    branch_endpoint_m: float,
    branch_sign: float,
    required_duration_s: float,
    velocity_m_s: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Trace only as far toward a stationary radius as the step requires.

    A simple stationary radius takes infinite physical time to cross.  The
    geometric tail supplies a resolved approach to that limit without ever
    evaluating the growth law on the other side.  Its time coordinate starts
    at the actual branch endpoint, rather than at an arbitrary near-root
    cutoff; that avoids a large, numerically noisy constant being subtracted
    from a one-second physical trace.
    """

    root = float(stationary_radius_m)
    endpoint = float(branch_endpoint_m)
    required = float(required_duration_s)
    if not math.isfinite(required) or required < 0.0:
        raise ConservativeRemapError("stationary-characteristic tail duration is invalid")
    directional: list[float] = [endpoint]
    if branch_sign < 0.0:
        if not endpoint < root:
            raise ConservativeRemapError("negative-velocity branch does not approach its upper stationary radius")
        candidates = root - (root - endpoint) * np.exp2(
            -np.arange(1, _ROOT_TAIL_SEGMENTS + 1, dtype=np.float64)
        )
    else:
        if not root < endpoint:
            raise ConservativeRemapError("positive-velocity branch does not approach its lower stationary radius")
        candidates = root + (endpoint - root) * np.exp2(
            -np.arange(_ROOT_TAIL_SEGMENTS, 0, -1, dtype=np.float64)
        )
        # This branch moves from the endpoint down toward the root.
        candidates = candidates[::-1]
    elapsed: list[float] = [0.0]
    for candidate in candidates:
        value = float(candidate)
        previous = directional[-1]
        if value == previous or not min(root, endpoint) < value < max(root, endpoint):
            continue
        interval = np.asarray(sorted((previous, value)), dtype=np.float64)
        flight = float(
            _gauss_time_of_flight(interval, expected_sign=branch_sign, velocity_m_s=velocity_m_s)[0]
        )
        directional.append(value)
        elapsed.append(elapsed[-1] + flight)
        if elapsed[-1] >= required:
            return np.asarray(directional, dtype=np.float64), np.asarray(elapsed, dtype=np.float64)
    raise ConservativeRemapError(
        "stationary-characteristic tail is insufficient for the requested physical timestep"
    )


def _invert_descending_time_of_flight(
    targets_s: NDArray[np.float64],
    cumulative_time_s: NDArray[np.float64],
    decreasing_coordinates_m: NDArray[np.float64],
    *,
    expected_sign: float,
    velocity_m_s: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Invert a trace table whose physical radius decreases with time."""

    coordinates = np.asarray(decreasing_coordinates_m, dtype=np.float64)
    if coordinates.ndim != 1 or np.any(np.diff(coordinates) >= 0.0):
        raise ConservativeRemapError("descending characteristic tail is not strictly ordered")
    reflected = _invert_time_of_flight(
        targets_s,
        cumulative_time_s,
        -coordinates,
        expected_sign=expected_sign,
        velocity_m_s=lambda reflected_radius: velocity_m_s(-reflected_radius),
    )
    return -reflected


def _trace_autonomous_time_of_flight(
    arrival_faces_m: NDArray[np.float64],
    *,
    duration_s: float,
    velocity_m_s: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    lower_radius_m: float,
    upper_radius_m: float,
) -> tuple[NDArray[np.float64], int, int]:
    """Map faces backward through a monotone autonomous time-of-flight trace."""

    if duration_s <= 0.0 or not math.isfinite(duration_s):
        raise ConservativeRemapError("characteristic trace duration is invalid")
    subdivision = _TRACE_SUBCELLS_PER_CELL
    nodes = _log_subdivided_faces(arrival_faces_m, subcells_per_cell=subdivision)
    node_velocity = _checked_velocity(velocity_m_s, nodes, label="trace-mesh")
    node_sign = np.sign(node_velocity)
    original_nodes = np.arange(arrival_faces_m.size, dtype=np.int64) * subdivision
    if not np.array_equal(nodes[original_nodes], arrival_faces_m):
        raise ConservativeRemapError("refined characteristic mesh lost a frozen arrival face")

    departure = np.empty_like(arrival_faces_m)
    lower_no_inflow = np.zeros(arrival_faces_m.shape, dtype=bool)
    upper_no_inflow = np.zeros(arrival_faces_m.shape, dtype=bool)
    zero_faces = node_sign[original_nodes] == 0.0
    departure[zero_faces] = arrival_faces_m[zero_faces]

    left = nodes[:-1]
    right = nodes[1:]
    midpoint = 0.5 * (left + right)
    half_width = 0.5 * (right - left)
    gauss_left = midpoint - _GAUSS_ABSCISSA * half_width
    gauss_right = midpoint + _GAUSS_ABSCISSA * half_width
    gauss_left_sign = np.sign(_checked_velocity(velocity_m_s, gauss_left, label="left Gauss mesh"))
    gauss_right_sign = np.sign(_checked_velocity(velocity_m_s, gauss_right, label="right Gauss mesh"))
    interval_sign = node_sign[:-1]
    valid_interval = (
        (interval_sign != 0.0)
        & (node_sign[1:] == interval_sign)
        & (gauss_left_sign == interval_sign)
        & (gauss_right_sign == interval_sign)
    )
    # A run consists only of intervals known to remain on one side of a
    # stationary radius.  It avoids treating a trial stage across G=0 as a
    # valid physical characteristic.
    run_id = np.cumsum(np.concatenate((np.asarray([True]), ~valid_interval)), dtype=np.int64) - 1
    face_runs = run_id[original_nodes]
    active_faces = ~zero_faces

    for current_run in np.unique(face_runs[active_faces]):
        face_indices = np.flatnonzero(active_faces & (face_runs == current_run))
        node_indices = original_nodes[face_indices]
        run_nodes = np.flatnonzero(run_id == current_run)
        run_start = int(run_nodes[0])
        run_end = int(run_nodes[-1])
        sign = float(node_sign[run_start])
        if sign == 0.0 or np.any(node_sign[run_nodes] != sign):
            raise ConservativeRemapError("characteristic trace run has inconsistent velocity sign")
        base_coordinates = nodes[run_start : run_end + 1]
        physical_lower = run_start == 0
        physical_upper = run_end == nodes.size - 1

        if sign < 0.0:
            # Backward motion is toward larger R.  A sign change at the upper
            # end is a stationary radius, not an upper-domain no-inflow edge.
            base_interval_time = _gauss_time_of_flight(
                base_coordinates, expected_sign=sign, velocity_m_s=velocity_m_s
            )
            base_time = np.concatenate((np.asarray([0.0]), np.cumsum(base_interval_time, dtype=np.float64)))
            source_time = base_time[node_indices - run_start]
            target_time = source_time + duration_s
            interior = target_time <= base_time[-1]
            departure[face_indices[interior]] = _invert_time_of_flight(
                target_time[interior],
                base_time,
                base_coordinates,
                expected_sign=sign,
                velocity_m_s=velocity_m_s,
            )
            beyond = ~interior
            if np.any(beyond) and physical_upper:
                boundary_faces = face_indices[~interior]
                departure[boundary_faces] = upper_radius_m
                upper_no_inflow[boundary_faces] = True
            elif np.any(beyond):
                next_index = run_end + 1
                root = _stationary_root(
                    lower_m=float(nodes[run_end]),
                    upper_m=float(nodes[next_index]),
                    lower_velocity=float(node_velocity[run_end]),
                    upper_velocity=float(node_velocity[next_index]),
                    velocity_m_s=velocity_m_s,
                )
                remaining_time = target_time[beyond] - base_time[-1]
                tail_coordinates, tail_time = _stationary_tail_from_endpoint(
                    stationary_radius_m=root,
                    branch_endpoint_m=float(base_coordinates[-1]),
                    branch_sign=sign,
                    required_duration_s=float(np.max(remaining_time)),
                    velocity_m_s=velocity_m_s,
                )
                departure[face_indices[beyond]] = _invert_time_of_flight(
                    remaining_time,
                    tail_time,
                    tail_coordinates,
                    expected_sign=sign,
                    velocity_m_s=velocity_m_s,
                )
        else:
            # Backward motion is toward smaller R.  Keep the base table's
            # origin at its actual lower endpoint.  When a face needs the
            # one-sided stationary tail, trace only the remaining physical
            # time from that endpoint; never subtract an arbitrary, enormous
            # near-root cumulative time.
            base_interval_time = _gauss_time_of_flight(
                base_coordinates, expected_sign=sign, velocity_m_s=velocity_m_s
            )
            base_time = np.concatenate((np.asarray([0.0]), np.cumsum(base_interval_time, dtype=np.float64)))
            source_time = base_time[node_indices - run_start]
            target_time = source_time - duration_s
            interior = target_time >= 0.0
            departure[face_indices[interior]] = _invert_time_of_flight(
                target_time[interior],
                base_time,
                base_coordinates,
                expected_sign=sign,
                velocity_m_s=velocity_m_s,
            )
            beyond = ~interior
            if np.any(beyond) and physical_lower:
                boundary_faces = face_indices[~interior]
                departure[boundary_faces] = lower_radius_m
                lower_no_inflow[boundary_faces] = True
            elif np.any(beyond):
                previous_index = run_start - 1
                root = _stationary_root(
                    lower_m=float(nodes[previous_index]),
                    upper_m=float(nodes[run_start]),
                    lower_velocity=float(node_velocity[previous_index]),
                    upper_velocity=float(node_velocity[run_start]),
                    velocity_m_s=velocity_m_s,
                )
                remaining_time = -target_time[beyond]
                tail_coordinates, tail_time = _stationary_tail_from_endpoint(
                    stationary_radius_m=root,
                    branch_endpoint_m=float(base_coordinates[0]),
                    branch_sign=sign,
                    required_duration_s=float(np.max(remaining_time)),
                    velocity_m_s=velocity_m_s,
                )
                departure[face_indices[beyond]] = _invert_descending_time_of_flight(
                    remaining_time,
                    tail_time,
                    tail_coordinates,
                    expected_sign=sign,
                    velocity_m_s=velocity_m_s,
                )

    if not np.all(np.isfinite(departure)):
        raise ConservativeRemapError("characteristic trace did not assign every arrival face")
    return departure, int(np.count_nonzero(lower_no_inflow)), int(np.count_nonzero(upper_no_inflow))


def piecewise_constant_cdf(
    edges_m: NDArray[np.float64] | np.ndarray,
    cell_number_m3: NDArray[np.float64] | np.ndarray,
    locations_m: NDArray[np.float64] | np.ndarray,
) -> NDArray[np.float64]:
    """Evaluate the exact CR1 cumulative population at in-domain locations."""

    edges, number = _validate_edges_and_numbers(edges_m, cell_number_m3)
    locations = np.asarray(locations_m, dtype=np.float64)
    if locations.ndim != 1 or not np.all(np.isfinite(locations)):
        raise ConservativeRemapError("CDF locations must be finite and one-dimensional")
    lower = float(edges[0])
    upper = float(edges[-1])
    if np.any(locations < lower) or np.any(locations > upper):
        raise ConservativeRemapError("CDF locations must remain within the physical radius domain")
    cumulative = np.empty(number.size + 1, dtype=np.float64)
    cumulative[0] = 0.0
    cumulative[1:] = np.cumsum(number, dtype=np.float64)
    indices = np.searchsorted(edges, locations, side="right") - 1
    indices = np.clip(indices, 0, number.size - 1)
    widths = np.diff(edges)
    local_fraction = (locations - edges[indices]) / widths[indices]
    values = cumulative[indices] + number[indices] * local_fraction
    values = np.asarray(values, dtype=np.float64)
    values[locations == lower] = 0.0
    values[locations == upper] = cumulative[-1]
    return values


def conservative_remap_piecewise_constant(
    edges_m: NDArray[np.float64] | np.ndarray,
    cell_number_m3: NDArray[np.float64] | np.ndarray,
    departure_faces_m: NDArray[np.float64] | np.ndarray,
) -> ConservativeRemapResult:
    """Remap a CR1 cell measure through an ordered backward face map.

    The new number in each fixed arrival cell is exactly the old cumulative
    population over its departure interval.  The only allowed loss is the
    old lower/upper interval omitted by the physical no-inflow map.
    """

    edges, number = _validate_edges_and_numbers(edges_m, cell_number_m3)
    departure = np.asarray(departure_faces_m, dtype=np.float64)
    if departure.shape != edges.shape or not np.all(np.isfinite(departure)):
        raise ConservativeRemapError("departure faces must match the frozen grid edges")
    if np.any(departure < edges[0]) or np.any(departure > edges[-1]):
        raise ConservativeRemapError("departure faces must remain within the physical domain")
    scale = max(float(np.max(np.abs(edges))), 1.0e-300)
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    if np.any(np.diff(departure) < -tolerance):
        raise ConservativeRemapError("departure faces are non-monotone")
    # The exact zero-velocity invariant is part of CR1 qualification.  The
    # ordinary CDF/difference path is mathematically identical but can alter
    # binary64 round-off in a non-zero cell, so preserve the literal accepted
    # measure when the characteristic map is the identity.
    if np.array_equal(departure, edges):
        return ConservativeRemapResult(
            cell_number_m3=number.copy(),
            lower_number_loss_m3=0.0,
            upper_number_loss_m3=0.0,
            old_number_m3=float(np.sum(number, dtype=np.float64)),
            new_number_m3=float(np.sum(number, dtype=np.float64)),
            conservation_residual_m3=0.0,
        )
    cdf = piecewise_constant_cdf(edges, number, departure)
    remapped = np.diff(cdf)
    if not np.all(np.isfinite(remapped)) or np.any(remapped < 0.0):
        raise ConservativeRemapError("conservative remap created an invalid negative cell number")
    # Use the identical cumulative reduction that produced ``cdf``.  Mixing
    # np.sum's pairwise reduction with a sequential cumulative reduction can
    # manufacture a tiny negative Rmax loss solely from round-off order.
    old_total = float(piecewise_constant_cdf(edges, number, np.asarray([edges[-1]]))[0])
    # The actual accepted state is ``remapped``, so its total must be measured
    # with the same reduction used by downstream inventory closure.  A
    # telescoping CDF identity can differ from this finite-array reduction by
    # one binary64 rounding unit; record that transparently rather than
    # reporting an exact identity for an array that was not stored.
    new_total = float(np.sum(remapped, dtype=np.float64))
    lower_loss = float(cdf[0])
    upper_loss = float(old_total - cdf[-1])
    if lower_loss < 0.0 or upper_loss < 0.0:
        raise ConservativeRemapError("cumulative remap reported a negative boundary loss")
    residual = old_total - lower_loss - upper_loss - new_total
    return ConservativeRemapResult(
        cell_number_m3=remapped,
        lower_number_loss_m3=lower_loss,
        upper_number_loss_m3=upper_loss,
        old_number_m3=old_total,
        new_number_m3=new_total,
        conservation_residual_m3=float(residual),
    )
