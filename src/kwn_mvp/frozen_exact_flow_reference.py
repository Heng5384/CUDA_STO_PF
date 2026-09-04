"""Independent frozen-autonomous characteristic reference machinery.

This module is intentionally diagnostic-only.  It shares the qualified beta
growth *physics* with KWN, but it does not import the production
characteristic trace, its time-of-flight helpers, or the CR1 remap.  The
reference flow is constructed directly from a frozen autonomous law using an
independent SciPy Gauss--Kronrod time coordinate and bracketed inversions.

The two measure classes below are likewise deliberately separate from CR1:
``PiecewiseConstantCumulativeMeasure`` is the authoritative initial-measure
interpretation, while ``PiecewiseLinearCumulativeMeasure`` exists only for a
bounded representation-sensitivity diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import IntegrationWarning, quad

from .growth import growth_rate_m_s
from .lower_boundary import boundary_growth_velocity, boundary_radius
from .population_metrics import PopulationMetrics, metrics_from_piecewise_constant_cells
from .populations import PopulationParameters
from .units import gas_constant_j_mol_k


class FrozenExactFlowReferenceError(RuntimeError):
    """Raised when the independent frozen reference cannot remain admissible."""


_FLOAT_EPS = float(np.finfo(np.float64).eps)


def _as_edges(edges_m: NDArray[np.float64] | np.ndarray) -> NDArray[np.float64]:
    edges = np.asarray(edges_m, dtype=np.float64)
    if (
        edges.ndim != 1
        or edges.size < 3
        or not np.all(np.isfinite(edges))
        or np.any(edges <= 0.0)
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise FrozenExactFlowReferenceError("reference radius edges must be finite, positive, and increasing")
    return edges.copy()


def _as_cells(cells: NDArray[np.float64] | np.ndarray, *, expected_size: int) -> NDArray[np.float64]:
    value = np.asarray(cells, dtype=np.float64)
    if value.ndim != 1 or value.size != expected_size or not np.all(np.isfinite(value)) or np.any(value < 0.0):
        raise FrozenExactFlowReferenceError("cell-integrated number measure is invalid")
    return value.copy()


def _float_close(left: float, right: float, *, multiplier: float = 512.0) -> bool:
    scale = max(abs(float(left)), abs(float(right)), 1.0e-300)
    return abs(float(left) - float(right)) <= multiplier * _FLOAT_EPS * scale


@dataclass(frozen=True)
class FrozenAutonomousGrowthLaw:
    """Frozen beta growth law using the shared qualified physical kernel.

    ``critical_radius_m`` is derived analytically from the validation
    Gibbs--Thomson contract.  It is never found by a generic physical scalar
    root search: the only scalar roots in this module are inverse-flow roots
    in the independently defined time coordinate.
    """

    parameters: PopulationParameters
    equilibrium_adapter: Any
    matrix_xb: float
    lower_radius_m: float
    upper_radius_m: float
    temperature_k: float | None = None
    planar_xb: float = field(init=False)
    gas_constant_j_mol_k: float = field(init=False)
    capillary_length_m: float = field(init=False)
    elastic_exponent: float = field(init=False)
    critical_radius_m: float | None = field(init=False)

    def __post_init__(self) -> None:
        lower = boundary_radius(self.lower_radius_m)
        upper = float(self.upper_radius_m)
        xb = float(self.matrix_xb)
        if not math.isfinite(upper) or not lower < upper:
            raise FrozenExactFlowReferenceError("frozen reference radius domain is invalid")
        if not math.isfinite(xb) or not 0.0 < xb < 1.0:
            raise FrozenExactFlowReferenceError("frozen matrix composition must lie in (0, 1)")
        if self.parameters.name != "beta":
            raise FrozenExactFlowReferenceError("frozen exact-flow reference is beta-only")
        adapter_temperature = getattr(self.equilibrium_adapter, "temperature_k", None)
        temperature = float(adapter_temperature if self.temperature_k is None else self.temperature_k)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise FrozenExactFlowReferenceError("frozen reference temperature is invalid")
        if adapter_temperature is not None and not _float_close(float(adapter_temperature), temperature):
            raise FrozenExactFlowReferenceError("growth-law and equilibrium-adapter temperatures differ")
        contract = getattr(self.equilibrium_adapter, "contract", None)
        gas = float(contract.gas_constant_j_mol_k) if contract is not None else float(gas_constant_j_mol_k())
        planar = float(self.parameters.xeq_infinity)
        if not math.isfinite(gas) or gas <= 0.0 or not math.isfinite(planar) or not 0.0 < planar < 1.0:
            raise FrozenExactFlowReferenceError("frozen Gibbs--Thomson constants are invalid")
        capillary = 2.0 * float(self.parameters.gamma_j_m2) * float(self.parameters.molar_volume_m3_mol) / (gas * temperature)
        elastic = float(self.parameters.elastic_penalty_j_m3) * float(self.parameters.molar_volume_m3_mol) / (gas * temperature)
        if not math.isfinite(capillary) or capillary < 0.0 or not math.isfinite(elastic):
            raise FrozenExactFlowReferenceError("frozen Gibbs--Thomson exponent is invalid")
        denominator = math.log(xb / planar) - elastic
        critical: float | None
        if capillary == 0.0 or denominator <= 0.0:
            critical = None
        else:
            candidate = capillary / denominator
            critical = float(candidate) if math.isfinite(candidate) and candidate > 0.0 else None
        object.__setattr__(self, "lower_radius_m", lower)
        object.__setattr__(self, "upper_radius_m", upper)
        object.__setattr__(self, "matrix_xb", xb)
        object.__setattr__(self, "temperature_k", temperature)
        object.__setattr__(self, "planar_xb", planar)
        object.__setattr__(self, "gas_constant_j_mol_k", gas)
        object.__setattr__(self, "capillary_length_m", capillary)
        object.__setattr__(self, "elastic_exponent", elastic)
        object.__setattr__(self, "critical_radius_m", critical)
        self._validate_equilibrium_contract()

    @property
    def has_domain_critical_radius(self) -> bool:
        critical = self.critical_radius_m
        return critical is not None and self.lower_radius_m < critical < self.upper_radius_m

    def _validation_radii(self) -> NDArray[np.float64]:
        values = [self.lower_radius_m, self.upper_radius_m, math.sqrt(self.lower_radius_m * self.upper_radius_m)]
        if self.has_domain_critical_radius and self.critical_radius_m is not None:
            critical = self.critical_radius_m
            values.extend([
                0.5 * (self.lower_radius_m + critical),
                0.5 * (critical + self.upper_radius_m),
            ])
        return np.asarray(sorted(set(float(item) for item in values)), dtype=np.float64)

    def _validate_equilibrium_contract(self) -> None:
        """Confirm that the frozen adapter is the explicit exponential law.

        This is a contract check, not a new thermo model.  It prevents a
        reference calculation from quietly treating an arbitrary adapter as
        one that possesses the analytic critical-radius branch structure.
        """

        radii = self._validation_radii()
        try:
            observed = np.asarray(self.equilibrium_adapter.equilibrium_xb(radii, self.parameters), dtype=np.float64)
        except Exception as error:  # pragma: no cover - exercised by integration fixtures
            raise FrozenExactFlowReferenceError("could not evaluate frozen equilibrium adapter") from error
        expected = self.planar_xb * np.exp(self.capillary_length_m / radii + self.elastic_exponent)
        if observed.shape != radii.shape or not np.all(np.isfinite(observed)):
            raise FrozenExactFlowReferenceError("frozen equilibrium adapter returned invalid values")
        if not np.allclose(observed, expected, rtol=2.0e-13, atol=2.0e-15):
            raise FrozenExactFlowReferenceError(
                "frozen exact-flow reference requires the validation exponential Gibbs--Thomson contract"
            )

    def equilibrium_xb(self, radii_m: NDArray[np.float64] | np.ndarray) -> NDArray[np.float64]:
        radii = np.asarray(radii_m, dtype=np.float64)
        if radii.ndim != 1 or not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
            raise FrozenExactFlowReferenceError("growth-law radii must be finite and positive")
        values = np.asarray(self.equilibrium_adapter.equilibrium_xb(radii, self.parameters), dtype=np.float64)
        if values.shape != radii.shape or not np.all(np.isfinite(values)):
            raise FrozenExactFlowReferenceError("equilibrium adapter returned invalid frozen values")
        return values

    def velocity(self, radii_m: NDArray[np.float64] | np.ndarray) -> NDArray[np.float64]:
        """Evaluate only the shared beta growth kernel, preserving Rmin exactly."""

        radii = np.asarray(radii_m, dtype=np.float64)
        if radii.ndim != 1 or not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
            raise FrozenExactFlowReferenceError("growth-law radii must be finite and positive")
        values = growth_rate_m_s(
            radii_m=radii,
            matrix_xb=self.matrix_xb,
            equilibrium_xb=self.equilibrium_xb(radii),
            parameters=self.parameters,
        )
        result = np.asarray(values, dtype=np.float64)
        if result.shape != radii.shape or not np.all(np.isfinite(result)):
            raise FrozenExactFlowReferenceError("shared frozen growth kernel returned invalid velocity")
        lower_mask = radii == self.lower_radius_m
        if np.any(lower_mask):
            result = result.copy()
            result[lower_mask] = boundary_growth_velocity(
                radius_m=self.lower_radius_m,
                matrix_xb=self.matrix_xb,
                parameters=self.parameters,
                equilibrium_adapter=self.equilibrium_adapter,
            )
        return result

    def velocity_scalar(self, radius_m: float) -> float:
        return float(self.velocity(np.asarray([float(radius_m)], dtype=np.float64))[0])

    def branch_name(self, radius_m: float) -> str:
        radius = float(radius_m)
        if not math.isfinite(radius) or radius < self.lower_radius_m or radius > self.upper_radius_m:
            raise FrozenExactFlowReferenceError("radius lies outside the frozen reference domain")
        if self.has_domain_critical_radius and self.critical_radius_m is not None:
            if radius == self.critical_radius_m:
                return "stationary_critical"
            return "shrinking" if radius < self.critical_radius_m else "growing"
        velocity = self.velocity_scalar(radius)
        if velocity == 0.0:
            raise FrozenExactFlowReferenceError("zero frozen velocity lacks an admissible analytic critical branch")
        return "shrinking" if velocity < 0.0 else "growing"

    def derivative_diagnostic(self, radius_m: float) -> float:
        """Return a bounded finite-difference ``dG/dR`` diagnostic.

        The derivative is a reporting diagnostic only.  It does not choose a
        branch or alter the flow map.
        """

        radius = float(radius_m)
        branch = self.branch_name(radius)
        if branch == "stationary_critical":
            radius = float(np.nextafter(radius, self.lower_radius_m))
            branch = "shrinking"
        distance_to_lower = radius - self.lower_radius_m
        distance_to_upper = self.upper_radius_m - radius
        distance_to_critical = math.inf
        if self.has_domain_critical_radius and self.critical_radius_m is not None:
            distance_to_critical = abs(radius - self.critical_radius_m)
        step = min(max(abs(radius) * 1.0e-6, abs(radius) * 64.0 * _FLOAT_EPS), 0.25 * min(
            distance_to_lower if distance_to_lower > 0.0 else math.inf,
            distance_to_upper if distance_to_upper > 0.0 else math.inf,
            distance_to_critical,
        ))
        if not math.isfinite(step) or step <= 0.0:
            step = max(abs(radius) * 1.0e-7, 64.0 * _FLOAT_EPS * abs(radius))
        left = radius - step
        right = radius + step
        if left >= self.lower_radius_m and right <= self.upper_radius_m and self.branch_name(left) == branch and self.branch_name(right) == branch:
            return (self.velocity_scalar(right) - self.velocity_scalar(left)) / (right - left)
        if right <= self.upper_radius_m and self.branch_name(right) == branch:
            return (self.velocity_scalar(right) - self.velocity_scalar(radius)) / (right - radius)
        if left >= self.lower_radius_m and self.branch_name(left) == branch:
            return (self.velocity_scalar(radius) - self.velocity_scalar(left)) / (radius - left)
        raise FrozenExactFlowReferenceError("could not form a same-branch growth-rate derivative diagnostic")

    def branch_diagnostics(self, radii_m: NDArray[np.float64] | np.ndarray) -> list[dict[str, Any]]:
        radii = _as_edges(radii_m)
        velocity = self.velocity(radii)
        maximum = max(float(np.max(np.abs(velocity))), 1.0e-300)
        rows: list[dict[str, Any]] = []
        for radius, value in zip(radii, velocity):
            branch = self.branch_name(float(radius))
            rows.append(
                {
                    "radius_m": float(radius),
                    "growth_velocity_m_s": float(value),
                    "dG_dR_s_inv": float(self.derivative_diagnostic(float(radius))),
                    "branch": branch,
                    "velocity_sign": int(np.sign(value)),
                    "near_zero_relative_to_domain_max": abs(float(value)) / maximum,
                    "critical_radius_m": self.critical_radius_m,
                }
            )
        self._validate_branch_structure(radii, velocity)
        return rows

    def _validate_branch_structure(self, edges: NDArray[np.float64], velocity: NDArray[np.float64]) -> None:
        if not np.all(np.isfinite(velocity)) or np.any(velocity == 0.0):
            raise FrozenExactFlowReferenceError("frozen growth law contains a non-finite or unclassified zero velocity")
        signs = np.sign(velocity).astype(np.int8)
        transitions = np.flatnonzero(signs[1:] != signs[:-1])
        if self.has_domain_critical_radius and self.critical_radius_m is not None:
            critical = self.critical_radius_m
            if transitions.size != 1:
                raise FrozenExactFlowReferenceError("analytic critical radius does not match one continuous sign transition")
            index = int(transitions[0])
            if not edges[index] < critical < edges[index + 1]:
                raise FrozenExactFlowReferenceError("analytic critical radius is outside observed sign-transition interval")
            if not (signs[0] < 0 and signs[-1] > 0):
                raise FrozenExactFlowReferenceError("frozen critical branches have unexpected physical signs")
        elif transitions.size != 0:
            raise FrozenExactFlowReferenceError("frozen growth law has an unregistered sign transition")


@dataclass(frozen=True)
class _BranchCoordinate:
    name: str
    sign: int
    radii_m: NDArray[np.float64]
    tau_s: NDArray[np.float64]
    outflow_radius_m: float
    reverse_boundary_radius_m: float | None
    critical_radius_m: float | None


@dataclass(frozen=True)
class FlowEvaluation:
    radius_m: float
    status: str
    event_time_s: float | None
    tau_s: float | None


@dataclass(frozen=True)
class FlowBatch:
    radius_m: NDArray[np.float64]
    status: NDArray[np.str_]
    event_time_s: NDArray[np.float64]
    tau_s: NDArray[np.float64]


@dataclass(frozen=True)
class ExactPushforwardState:
    """A single direct U0-to-t cumulative-measure pushforward."""

    time_s: float
    cell_number_m3: NDArray[np.float64]
    departure_faces_m: NDArray[np.float64]
    lower_number_loss_m3: float
    upper_number_loss_m3: float
    initial_number_m3: float
    final_number_m3: float
    conservation_residual_m3: float
    representation: str

    def metrics(self, edges_m: NDArray[np.float64] | np.ndarray) -> PopulationMetrics:
        return metrics_from_piecewise_constant_cells(edges_m, self.cell_number_m3)


class FrozenAutonomousExactFlow:
    """Independent time-of-flight exact/semi-exact flow for frozen ``G(R)``."""

    def __init__(
        self,
        law: FrozenAutonomousGrowthLaw,
        reference_knots_m: NDArray[np.float64] | np.ndarray,
        *,
        quad_epsabs_s: float = 2.0e-14,
        quad_epsrel: float = 2.0e-12,
    ) -> None:
        self.law = law
        self.reference_knots_m = _as_edges(reference_knots_m)
        if self.reference_knots_m[0] != law.lower_radius_m or self.reference_knots_m[-1] != law.upper_radius_m:
            raise FrozenExactFlowReferenceError("time-coordinate knots must preserve the exact physical domain")
        self.quad_epsabs_s = float(quad_epsabs_s)
        self.quad_epsrel = float(quad_epsrel)
        if not math.isfinite(self.quad_epsabs_s) or self.quad_epsabs_s <= 0.0 or not math.isfinite(self.quad_epsrel) or self.quad_epsrel <= 0.0:
            raise FrozenExactFlowReferenceError("quadrature tolerances must be positive and finite")
        self._quad_error_max_s = 0.0
        self._quad_evaluation_count = 0
        # This validates the analytic branch law before a time coordinate is
        # built.  It is also the complete-domain growth-law reporting source.
        self.growth_law_rows = tuple(self.law.branch_diagnostics(self.reference_knots_m))
        self._coordinates = self._build_coordinates()
        self._tau_cache: dict[float, float] = {}
        for coordinate in self._coordinates.values():
            self._tau_cache.update(
                {float(radius): float(tau) for radius, tau in zip(coordinate.radii_m, coordinate.tau_s)}
            )

    @property
    def quadrature_error_max_s(self) -> float:
        return float(self._quad_error_max_s)

    @property
    def quadrature_evaluation_count(self) -> int:
        return int(self._quad_evaluation_count)

    @property
    def critical_radius_m(self) -> float | None:
        return self.law.critical_radius_m if self.law.has_domain_critical_radius else None

    def _inverse_velocity(self, radius_m: float) -> float:
        velocity = self.law.velocity_scalar(float(radius_m))
        if not math.isfinite(velocity) or velocity == 0.0:
            raise FrozenExactFlowReferenceError("time-of-flight quadrature reached a stationary or invalid velocity")
        return 1.0 / velocity

    def _quad_tau(self, left_m: float, right_m: float) -> float:
        left = float(left_m)
        right = float(right_m)
        if left == right:
            return 0.0
        midpoint = 0.5 * (left + right)
        left_branch = self.law.branch_name(left)
        right_branch = self.law.branch_name(right)
        mid_branch = self.law.branch_name(midpoint)
        if "stationary" in {left_branch, right_branch, mid_branch} or len({left_branch, right_branch, mid_branch}) != 1:
            raise FrozenExactFlowReferenceError("time-of-flight quadrature attempted to cross the critical branch")
        with warnings.catch_warnings():
            warnings.simplefilter("error", IntegrationWarning)
            try:
                value, error = quad(
                    self._inverse_velocity,
                    left,
                    right,
                    epsabs=self.quad_epsabs_s,
                    epsrel=self.quad_epsrel,
                    limit=300,
                )
            except (IntegrationWarning, ValueError, FloatingPointError) as exception:
                raise FrozenExactFlowReferenceError("independent time-of-flight quadrature failed") from exception
        if not math.isfinite(float(value)) or not math.isfinite(float(error)):
            raise FrozenExactFlowReferenceError("time-of-flight quadrature returned a non-finite result")
        self._quad_error_max_s = max(self._quad_error_max_s, abs(float(error)))
        self._quad_evaluation_count += 1
        return float(value)

    def _nodes_for_branch(self, name: str) -> NDArray[np.float64]:
        knots = self.reference_knots_m
        if self.law.has_domain_critical_radius and self.law.critical_radius_m is not None:
            critical = self.law.critical_radius_m
            nodes = knots[knots < critical] if name == "shrinking" else knots[knots > critical]
        else:
            nodes = knots
        if nodes.size < 2:
            raise FrozenExactFlowReferenceError(f"{name} branch has fewer than two admissible time-coordinate knots")
        return nodes.copy()

    def _build_coordinate(self, name: str, sign: int) -> _BranchCoordinate:
        radii = self._nodes_for_branch(name)
        tau = np.empty(radii.shape, dtype=np.float64)
        active_critical = self.law.critical_radius_m if self.law.has_domain_critical_radius else None
        if sign < 0:
            tau[0] = 0.0
            for index in range(1, radii.size):
                increment = self._quad_tau(float(radii[index - 1]), float(radii[index]))
                if increment >= 0.0:
                    raise FrozenExactFlowReferenceError("shrinking time coordinate has a non-negative increment")
                tau[index] = tau[index - 1] + increment
            if np.any(np.diff(tau) >= 0.0):
                raise FrozenExactFlowReferenceError("shrinking time coordinate is not strictly monotone")
            return _BranchCoordinate(
                name=name,
                sign=-1,
                radii_m=radii,
                tau_s=tau,
                outflow_radius_m=float(radii[0]),
                reverse_boundary_radius_m=None if active_critical is not None else float(radii[-1]),
                critical_radius_m=active_critical,
            )
        tau[-1] = 0.0
        for index in range(radii.size - 2, -1, -1):
            increment = self._quad_tau(float(radii[index + 1]), float(radii[index]))
            if increment >= 0.0:
                raise FrozenExactFlowReferenceError("growing time coordinate has a non-negative reverse increment")
            tau[index] = tau[index + 1] + increment
        if np.any(np.diff(tau) <= 0.0):
            raise FrozenExactFlowReferenceError("growing time coordinate is not strictly monotone")
        return _BranchCoordinate(
            name=name,
            sign=1,
            radii_m=radii,
            tau_s=tau,
            outflow_radius_m=float(radii[-1]),
            reverse_boundary_radius_m=None if active_critical is not None else float(radii[0]),
            critical_radius_m=active_critical,
        )

    def _build_coordinates(self) -> dict[str, _BranchCoordinate]:
        if self.law.has_domain_critical_radius:
            return {
                "shrinking": self._build_coordinate("shrinking", -1),
                "growing": self._build_coordinate("growing", 1),
            }
        midpoint = math.sqrt(self.law.lower_radius_m * self.law.upper_radius_m)
        sign = -1 if self.law.velocity_scalar(midpoint) < 0.0 else 1
        name = "shrinking" if sign < 0 else "growing"
        return {name: self._build_coordinate(name, sign)}

    def _coordinate_for_radius(self, radius_m: float) -> _BranchCoordinate | None:
        branch = self.law.branch_name(radius_m)
        if branch == "stationary_critical":
            return None
        try:
            return self._coordinates[branch]
        except KeyError as error:  # pragma: no cover - protects future law extensions
            raise FrozenExactFlowReferenceError(f"no time coordinate for branch {branch}") from error

    @staticmethod
    def _node_index(nodes: NDArray[np.float64], radius_m: float) -> int | None:
        index = int(np.searchsorted(nodes, radius_m, side="left"))
        if index < nodes.size and nodes[index] == radius_m:
            return index
        return None

    def tau(self, radius_m: float) -> float:
        """Return the independent time coordinate, never crossing ``G=0``."""

        radius = float(radius_m)
        cached = self._tau_cache.get(radius)
        if cached is not None:
            return cached
        coordinate = self._coordinate_for_radius(radius)
        if coordinate is None:
            return float("-inf")
        index = self._node_index(coordinate.radii_m, radius)
        if index is not None:
            return float(coordinate.tau_s[index])
        insertion = int(np.searchsorted(coordinate.radii_m, radius, side="left"))
        if insertion == 0:
            anchor = float(coordinate.radii_m[0])
            anchor_tau = float(coordinate.tau_s[0])
        elif insertion == coordinate.radii_m.size:
            anchor = float(coordinate.radii_m[-1])
            anchor_tau = float(coordinate.tau_s[-1])
        else:
            left = float(coordinate.radii_m[insertion - 1])
            right = float(coordinate.radii_m[insertion])
            if abs(radius - left) <= abs(right - radius):
                anchor = left
                anchor_tau = float(coordinate.tau_s[insertion - 1])
            else:
                anchor = right
                anchor_tau = float(coordinate.tau_s[insertion])
        if not math.isfinite(anchor_tau):
            raise FrozenExactFlowReferenceError("radius is outside its frozen time-coordinate nodes")
        value = float(anchor_tau + self._quad_tau(anchor, radius))
        if not math.isfinite(value) or value > 512.0 * _FLOAT_EPS * max(1.0, abs(value)):
            raise FrozenExactFlowReferenceError("time coordinate is not finite and non-positive from its outflow boundary")
        value = min(value, 0.0)
        self._tau_cache[radius] = value
        return value

    def _critical_extension(self, coordinate: _BranchCoordinate, target_tau_s: float) -> float:
        """Find a same-branch endpoint closer to Rc that brackets ``target``."""

        critical = coordinate.critical_radius_m
        if critical is None:
            raise FrozenExactFlowReferenceError("cannot extend a finite-domain branch toward a missing critical radius")
        if coordinate.sign < 0:
            candidate = float(coordinate.radii_m[-1])
            direction = 1.0
            predicate = lambda value: value <= target_tau_s
        else:
            candidate = float(coordinate.radii_m[0])
            direction = -1.0
            predicate = lambda value: value <= target_tau_s
        for _ in range(96):
            candidate = critical + 0.25 * (candidate - critical)
            if candidate == critical:
                candidate = float(np.nextafter(critical, math.inf if direction > 0.0 else -math.inf))
            if not (self.law.lower_radius_m < candidate < self.law.upper_radius_m):
                raise FrozenExactFlowReferenceError("critical extension escaped the physical radius domain")
            if self.law.branch_name(candidate) != coordinate.name:
                continue
            tau = self.tau(candidate)
            if predicate(tau):
                return candidate
        raise FrozenExactFlowReferenceError("finite target time could not be bracketed before the critical asymptote")

    def _inverse_tau(
        self,
        coordinate: _BranchCoordinate,
        target_tau_s: float,
        *,
        start_radius_m: float,
        duration_s: float,
    ) -> float:
        target = float(target_tau_s)
        if not math.isfinite(target) or target > 0.0:
            raise FrozenExactFlowReferenceError("inverse flow requires a finite non-positive time coordinate")
        nodes = coordinate.radii_m
        taus = coordinate.tau_s
        if coordinate.sign < 0:
            values = -taus
            query = -target
            index = int(np.searchsorted(values, query, side="left"))
            if index < values.size and target == float(taus[index]):
                return float(nodes[index])
            if index == 0:
                return float(nodes[0])
            if index < values.size:
                left, right = float(nodes[index - 1]), float(nodes[index])
            elif coordinate.reverse_boundary_radius_m is not None:
                return float(coordinate.reverse_boundary_radius_m)
            else:
                left, right = float(nodes[-1]), self._critical_extension(coordinate, target)
        else:
            index = int(np.searchsorted(taus, target, side="left"))
            if index < taus.size and target == float(taus[index]):
                return float(nodes[index])
            if index == taus.size:
                return float(nodes[-1])
            if index > 0:
                left, right = float(nodes[index - 1]), float(nodes[index])
            elif coordinate.reverse_boundary_radius_m is not None:
                return float(coordinate.reverse_boundary_radius_m)
            else:
                left, right = self._critical_extension(coordinate, target), float(nodes[0])
        f_left = self.tau(left) - target
        f_right = self.tau(right) - target
        if f_left == 0.0:
            return left
        if f_right == 0.0:
            return right
        if f_left * f_right > 0.0:
            raise FrozenExactFlowReferenceError("time-coordinate inversion interval does not bracket the target")
        # The time-coordinate table supplies an exact-quadrature bracket.
        # Refine it with a safeguarded secant/Newton sequence whose residuals
        # are *always* evaluated by the independent quadrature.  This avoids
        # replacing the reference with an interpolated map while keeping the
        # 3200-face diagnostic computationally practical.
        value = self._safeguarded_inverse_tau(
            float(start_radius_m), left, right, float(duration_s)
        )
        if self.law.branch_name(value) != coordinate.name:
            raise FrozenExactFlowReferenceError("inverse flow crossed a critical branch")
        return value

    def _safeguarded_inverse_tau(
        self,
        start_radius_m: float,
        left_m: float,
        right_m: float,
        duration_s: float,
    ) -> float:
        left = float(left_m)
        right = float(right_m)
        start = float(start_radius_m)
        duration = float(duration_s)
        # Use a *local* flight-time residual.  A global Tau coordinate may be
        # thousands of seconds close to Rc while the diagnostic step is only
        # milliseconds; subtracting those two values would throw away exactly
        # the fine-time information this reference is meant to retain.
        f_left = self._quad_tau(start, left) - duration
        f_right = self._quad_tau(start, right) - duration
        tolerance_s = max(
            16.0 * self.quad_epsabs_s,
            32.0 * self.quad_epsrel * max(abs(duration), 1.0e-12),
        )

        def residual_tolerance(radius: float) -> float:
            # A binary64 radius cannot resolve a flight time below dR/|G|.
            # Treat that transparent representational floor separately from
            # quadrature accuracy; the returned map remains radius-accurate
            # to the available binary64 coordinate rather than looping after
            # a physically unrepresentable sub-ULP correction.
            velocity = abs(self.law.velocity_scalar(radius))
            radius_floor = 32.0 * _FLOAT_EPS * max(abs(radius), abs(start), 1.0e-300) / max(velocity, 1.0e-300)
            return max(tolerance_s, radius_floor)

        if abs(f_left) <= residual_tolerance(left):
            return left
        if abs(f_right) <= residual_tolerance(right):
            return right
        if f_left * f_right > 0.0:
            raise FrozenExactFlowReferenceError("safeguarded inverse has no sign-changing time bracket")
        value = 0.5 * (left + right)
        for _ in range(18):
            # A secant proposal uses exact endpoint coordinates.  The guarded
            # Newton proposal below normally reduces the remaining iterations
            # to two or three for a smooth same-branch characteristic.
            denominator = f_right - f_left
            if denominator != 0.0:
                proposal = right - f_right * (right - left) / denominator
            else:
                proposal = 0.5 * (left + right)
            # Near a slowly moving critical-neighbour face, the physically
            # correct root can be very close to one bracket endpoint.  Do not
            # impose an arbitrary interior fraction: that would turn a
            # legitimate small displacement into a long bisection sequence.
            if not (left < proposal < right):
                proposal = 0.5 * (left + right)
            value = float(proposal)
            f_value = self._quad_tau(start, value) - duration
            if abs(f_value) <= residual_tolerance(value):
                return value
            if f_left * f_value < 0.0:
                right, f_right = value, f_value
            else:
                left, f_left = value, f_value
            velocity = self.law.velocity_scalar(value)
            newton = value - f_value * velocity
            if left < newton < right:
                newton_value = float(newton)
                newton_residual = self._quad_tau(start, newton_value) - duration
                if abs(newton_residual) <= residual_tolerance(newton_value):
                    return newton_value
                if f_left * newton_residual < 0.0:
                    right, f_right = newton_value, newton_residual
                else:
                    left, f_left = newton_value, newton_residual
            if right - left <= max(32.0 * _FLOAT_EPS * max(abs(left), abs(right), 1.0e-300), 1.0e-30):
                midpoint = 0.5 * (left + right)
                if abs(self._quad_tau(start, midpoint) - duration) <= 4.0 * residual_tolerance(midpoint):
                    return midpoint
        raise FrozenExactFlowReferenceError("safeguarded inverse time-of-flight solve did not reach its residual tolerance")

    def evaluate(self, radius_m: float, duration_s: float) -> FlowEvaluation:
        """Evaluate a forward or backward exact-flow characteristic.

        Forward reaches of Rmin/Rmax are reported as physical events.  A
        backward reach of a finite opposite boundary is a no-inflow domain
        coverage event.  In either case the returned radius is the physical
        boundary, never an extrapolated or negative value.
        """

        radius = float(radius_m)
        duration = float(duration_s)
        if not math.isfinite(duration):
            raise FrozenExactFlowReferenceError("flow duration must be finite")
        if duration == 0.0:
            return FlowEvaluation(radius_m=radius, status="IDENTITY", event_time_s=None, tau_s=self.tau(radius) if self.law.branch_name(radius) != "stationary_critical" else float("-inf"))
        coordinate = self._coordinate_for_radius(radius)
        if coordinate is None:
            return FlowEvaluation(radius_m=radius, status="STATIONARY_CRITICAL", event_time_s=None, tau_s=float("-inf"))
        start_tau = self.tau(radius)
        target = start_tau + duration
        if target >= 0.0:
            if coordinate.sign < 0:
                return FlowEvaluation(
                    radius_m=coordinate.outflow_radius_m,
                    status="ABSORBED_RMIN",
                    event_time_s=max(-start_tau, 0.0),
                    tau_s=0.0,
                )
            return FlowEvaluation(
                radius_m=coordinate.outflow_radius_m,
                status="EXITED_RMAX",
                event_time_s=max(-start_tau, 0.0),
                tau_s=0.0,
            )
        # A finite reverse boundary exists only when there is no internal
        # stationary point.  Do not extrapolate a departure beyond it.
        finite_reverse_tau: float | None = None
        if coordinate.reverse_boundary_radius_m is not None:
            finite_reverse_tau = self.tau(coordinate.reverse_boundary_radius_m)
        if finite_reverse_tau is not None and target < finite_reverse_tau:
            status = "NO_INFLOW_UPPER" if coordinate.sign < 0 else "NO_INFLOW_LOWER"
            return FlowEvaluation(
                radius_m=coordinate.reverse_boundary_radius_m,
                status=status,
                event_time_s=abs(finite_reverse_tau - start_tau),
                tau_s=finite_reverse_tau,
            )
        result = self._inverse_tau(
            coordinate,
            target,
            start_radius_m=radius,
            duration_s=duration,
        )
        return FlowEvaluation(radius_m=result, status="INTERIOR", event_time_s=None, tau_s=target)

    def flow(self, radii_m: NDArray[np.float64] | np.ndarray, duration_s: float) -> FlowBatch:
        radii = np.asarray(radii_m, dtype=np.float64)
        if radii.ndim != 1 or not np.all(np.isfinite(radii)):
            raise FrozenExactFlowReferenceError("flow radii must be a finite one-dimensional array")
        values: list[FlowEvaluation] = [self.evaluate(float(radius), float(duration_s)) for radius in radii]
        return FlowBatch(
            radius_m=np.asarray([value.radius_m for value in values], dtype=np.float64),
            status=np.asarray([value.status for value in values], dtype=np.str_),
            event_time_s=np.asarray([
                np.nan if value.event_time_s is None else float(value.event_time_s) for value in values
            ], dtype=np.float64),
            tau_s=np.asarray([value.tau_s if value.tau_s is not None else np.nan for value in values], dtype=np.float64),
        )

    def departure_faces(self, arrival_faces_m: NDArray[np.float64] | np.ndarray, duration_s: float) -> FlowBatch:
        duration = float(duration_s)
        if not math.isfinite(duration) or duration < 0.0:
            raise FrozenExactFlowReferenceError("departure duration must be finite and non-negative")
        if duration == 0.0:
            faces = np.asarray(arrival_faces_m, dtype=np.float64)
            return FlowBatch(
                radius_m=faces.copy(),
                status=np.full(faces.shape, "IDENTITY", dtype=np.str_),
                event_time_s=np.full(faces.shape, np.nan, dtype=np.float64),
                tau_s=np.asarray([self.tau(float(face)) if self.law.branch_name(float(face)) != "stationary_critical" else -math.inf for face in faces], dtype=np.float64),
            )
        result = self.flow(arrival_faces_m, -duration)
        if np.any(np.diff(result.radius_m) < -128.0 * _FLOAT_EPS * max(float(np.max(np.abs(result.radius_m))), 1.0e-300)):
            raise FrozenExactFlowReferenceError("independent exact backward face map is non-monotone")
        return result

    def semigroup_check(self, radii_m: NDArray[np.float64] | np.ndarray, duration_s: float) -> dict[str, Any]:
        radii = np.asarray(radii_m, dtype=np.float64)
        half = self.flow(radii, 0.5 * float(duration_s))
        composed = self.flow(half.radius_m, 0.5 * float(duration_s))
        direct = self.flow(radii, float(duration_s))
        delta = composed.radius_m - direct.radius_m
        scale = np.maximum(np.maximum(np.abs(direct.radius_m), np.abs(composed.radius_m)), 1.0e-300)
        return {
            "max_radius_error_m": float(np.max(np.abs(delta))) if delta.size else 0.0,
            "rms_radius_error_m": float(math.sqrt(float(np.mean(np.square(delta)))) if delta.size else 0.0),
            "max_relative_radius_error": float(np.max(np.abs(delta) / scale)) if delta.size else 0.0,
            "status_mismatch_count": int(np.count_nonzero(composed.status != direct.status)),
            "face_count": int(radii.size),
        }

    def roundtrip_check(self, radii_m: NDArray[np.float64] | np.ndarray, duration_s: float) -> dict[str, Any]:
        radii = np.asarray(radii_m, dtype=np.float64)
        forward = self.flow(radii, float(duration_s))
        interior = np.isin(forward.status, np.asarray(["INTERIOR", "IDENTITY", "STATIONARY_CRITICAL"], dtype=np.str_))
        if not np.any(interior):
            return {"tested_count": 0, "max_radius_error_m": None, "max_relative_radius_error": None}
        recovered = self.flow(forward.radius_m[interior], -float(duration_s))
        delta = recovered.radius_m - radii[interior]
        scale = np.maximum(np.maximum(np.abs(recovered.radius_m), np.abs(radii[interior])), 1.0e-300)
        return {
            "tested_count": int(np.count_nonzero(interior)),
            "max_radius_error_m": float(np.max(np.abs(delta))),
            "max_relative_radius_error": float(np.max(np.abs(delta) / scale)),
            "status_mismatch_count": int(np.count_nonzero(~np.isin(recovered.status, np.asarray(["INTERIOR", "IDENTITY", "STATIONARY_CRITICAL"], dtype=np.str_)))),
        }

    def tau_table_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for coordinate in self._coordinates.values():
            for radius, tau in zip(coordinate.radii_m, coordinate.tau_s):
                rows.append(
                    {
                        "radius_m": float(radius),
                        "tau_s": float(tau),
                        "branch": coordinate.name,
                        "branch_sign": coordinate.sign,
                        "quadrature_epsabs_s": self.quad_epsabs_s,
                        "quadrature_epsrel": self.quad_epsrel,
                    }
                )
        return rows

    def rmin_crossing_rows(self, radii_m: NDArray[np.float64] | np.ndarray) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for radius in np.asarray(radii_m, dtype=np.float64):
            if self.law.branch_name(float(radius)) != "shrinking":
                continue
            tau = self.tau(float(radius))
            rows.append(
                {
                    "radius_m": float(radius),
                    "t_to_Rmin_s": max(-tau, 0.0),
                    "event": "ABSORBING_RMIN",
                    "Rmin_m": self.law.lower_radius_m,
                }
            )
        return rows

    def mpmath_tau(self, left_m: float, right_m: float, *, dps: int = 80) -> float:
        """Independent high-precision shadow integral from frozen binary64 scalars.

        The function imports :mod:`mpmath` lazily so local environments can
        test the SciPy reference without silently substituting a lower-quality
        shadow.  Formal runs must provide hash-pinned mpmath and fail closed if
        it is unavailable.
        """

        try:
            import mpmath as mp  # type: ignore[import-not-found]
        except ImportError as error:
            raise FrozenExactFlowReferenceError("mpmath high-precision shadow is unavailable") from error
        left = float(left_m)
        right = float(right_m)
        if left == right:
            return 0.0
        midpoint = 0.5 * (left + right)
        branches = {self.law.branch_name(left), self.law.branch_name(right), self.law.branch_name(midpoint)}
        if len(branches) != 1 or "stationary_critical" in branches:
            raise FrozenExactFlowReferenceError("mpmath shadow integral would cross a critical branch")
        with mp.workdps(int(dps)):
            mpf = mp.mpf
            diffusivity = mpf(float(self.law.parameters.diffusivity_m2_s))
            xb = mpf(float(self.law.matrix_xb))
            xp = mpf(float(self.law.parameters.x_b))
            planar = mpf(float(self.law.planar_xb))
            capillary = mpf(float(self.law.capillary_length_m))
            elastic = mpf(float(self.law.elastic_exponent))
            shape = mpf(float(self.law.parameters.shape_factor))

            def inverse_velocity(radius: Any) -> Any:
                equilibrium = planar * mp.exp(capillary / radius + elastic)
                velocity = diffusivity / radius * (xb - equilibrium) / (xp - equilibrium) * shape
                return 1 / velocity

            value = mp.quad(inverse_velocity, [mpf(left), mpf(right)])
        result = float(value)
        if not math.isfinite(result):
            raise FrozenExactFlowReferenceError("mpmath shadow integral returned a non-finite value")
        return result

    def tau_shadow_rows(self, *, sample_count_per_branch: int = 5, dps: int = 80) -> list[dict[str, Any]]:
        if sample_count_per_branch < 2:
            raise ValueError("at least two shadow samples per branch are required")
        rows: list[dict[str, Any]] = []
        for coordinate in self._coordinates.values():
            count = coordinate.radii_m.size
            indices = np.unique(np.linspace(0, count - 1, num=sample_count_per_branch + 1, dtype=int))
            for left_index, right_index in zip(indices[:-1], indices[1:]):
                left = float(coordinate.radii_m[int(left_index)])
                right = float(coordinate.radii_m[int(right_index)])
                scipy_value = self._quad_tau(left, right)
                mpmath_value = self.mpmath_tau(left, right, dps=dps)
                rows.append(
                    {
                        "branch": coordinate.name,
                        "left_radius_m": left,
                        "right_radius_m": right,
                        "scipy_tau_increment_s": scipy_value,
                        "mpmath_tau_increment_s": mpmath_value,
                        "absolute_discrepancy_s": abs(scipy_value - mpmath_value),
                        "relative_discrepancy": abs(scipy_value - mpmath_value) / max(abs(mpmath_value), 1.0e-300),
                        "mpmath_dps": int(dps),
                    }
                )
        return rows


class PiecewiseConstantCumulativeMeasure:
    """Authoritative REF-PC cumulative measure built from original U0 cells."""

    representation = "REF_PC_AUTHORITATIVE"

    def __init__(self, edges_m: NDArray[np.float64] | np.ndarray, cell_number_m3: NDArray[np.float64] | np.ndarray) -> None:
        self.edges_m = _as_edges(edges_m)
        self.cell_number_m3 = _as_cells(cell_number_m3, expected_size=self.edges_m.size - 1)
        self._cumulative = np.concatenate((np.asarray([0.0], dtype=np.float64), np.cumsum(self.cell_number_m3, dtype=np.float64)))
        self.total_number_m3 = float(self._cumulative[-1])

    def cdf(self, radii_m: NDArray[np.float64] | np.ndarray) -> NDArray[np.float64]:
        radii = np.asarray(radii_m, dtype=np.float64)
        if radii.ndim != 1 or not np.all(np.isfinite(radii)):
            raise FrozenExactFlowReferenceError("CDF radii must be finite and one-dimensional")
        if np.any(radii < self.edges_m[0]) or np.any(radii > self.edges_m[-1]):
            raise FrozenExactFlowReferenceError("CDF query lies outside the initial physical measure domain")
        indices = np.searchsorted(self.edges_m, radii, side="right") - 1
        indices = np.clip(indices, 0, self.cell_number_m3.size - 1)
        widths = np.diff(self.edges_m)
        fractions = (radii - self.edges_m[indices]) / widths[indices]
        values = self._cumulative[indices] + self.cell_number_m3[indices] * fractions
        values = np.asarray(values, dtype=np.float64)
        values[radii == self.edges_m[0]] = 0.0
        values[radii == self.edges_m[-1]] = self.total_number_m3
        if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > self.total_number_m3):
            raise FrozenExactFlowReferenceError("REF-PC CDF evaluation is invalid")
        return values

    def pushforward(self, flow: FrozenAutonomousExactFlow, time_s: float) -> ExactPushforwardState:
        time_value = float(time_s)
        if not math.isfinite(time_value) or time_value < 0.0:
            raise FrozenExactFlowReferenceError("pushforward time must be finite and non-negative")
        if time_value == 0.0:
            # Preserve the initial discrete measure bit-for-bit.  This is an
            # identity contract, not a CDF/reduction coincidence.
            return ExactPushforwardState(
                time_s=0.0,
                cell_number_m3=self.cell_number_m3.copy(),
                departure_faces_m=self.edges_m.copy(),
                lower_number_loss_m3=0.0,
                upper_number_loss_m3=0.0,
                initial_number_m3=self.total_number_m3,
                final_number_m3=float(np.sum(self.cell_number_m3, dtype=np.float64)),
                conservation_residual_m3=0.0,
                representation=self.representation,
            )
        departure = flow.departure_faces(self.edges_m, time_value)
        cdf = self.cdf(departure.radius_m)
        cells = np.diff(cdf)
        tolerance = 1024.0 * _FLOAT_EPS * max(self.total_number_m3, 1.0e-300)
        if not np.all(np.isfinite(cells)) or np.any(cells < -tolerance):
            raise FrozenExactFlowReferenceError("exact pushforward generated a materially negative target cell measure")
        # A negative value here could only be a last-bit CDF subtraction.  It
        # is not silently repaired: an exact value below zero has already
        # failed above, while signed zero is harmless as a stored measure.
        cells = np.asarray(cells, dtype=np.float64)
        cells[cells == -0.0] = 0.0
        if np.any(cells < 0.0):
            raise FrozenExactFlowReferenceError("exact pushforward has a negative CDF-difference cell")
        lower_loss = float(cdf[0])
        upper_loss = float(self.total_number_m3 - cdf[-1])
        final_number = float(np.sum(cells, dtype=np.float64))
        residual = self.total_number_m3 - lower_loss - upper_loss - final_number
        return ExactPushforwardState(
            time_s=time_value,
            cell_number_m3=cells,
            departure_faces_m=departure.radius_m.copy(),
            lower_number_loss_m3=lower_loss,
            upper_number_loss_m3=upper_loss,
            initial_number_m3=self.total_number_m3,
            final_number_m3=final_number,
            conservation_residual_m3=float(residual),
            representation=self.representation,
        )


class PiecewiseLinearCumulativeMeasure(PiecewiseConstantCumulativeMeasure):
    """Non-negative conservative PL initial-measure sensitivity diagnostic.

    Every cell retains precisely its original number measure.  Slopes are
    limited so the reconstructed density remains non-negative; therefore its
    cumulative measure is monotone.  This class is never an authority for a
    CR1 acceptance or method decision.
    """

    representation = "REF_PL_DIAGNOSTIC_ONLY"

    def __init__(self, edges_m: NDArray[np.float64] | np.ndarray, cell_number_m3: NDArray[np.float64] | np.ndarray) -> None:
        super().__init__(edges_m, cell_number_m3)
        widths = np.diff(self.edges_m)
        density = self.cell_number_m3 / widths
        centres = 0.5 * (self.edges_m[:-1] + self.edges_m[1:])
        slope = np.zeros_like(density)
        if density.size >= 3:
            left_gradient = (density[1:-1] - density[:-2]) / (centres[1:-1] - centres[:-2])
            right_gradient = (density[2:] - density[1:-1]) / (centres[2:] - centres[1:-1])
            same_sign = left_gradient * right_gradient > 0.0
            harmonic = np.zeros_like(left_gradient)
            denominator = left_gradient + right_gradient
            valid = same_sign & (denominator != 0.0)
            harmonic[valid] = 2.0 * left_gradient[valid] * right_gradient[valid] / denominator[valid]
            slope[1:-1] = harmonic
        limits = np.where(density > 0.0, 2.0 * density / widths, 0.0)
        slope = np.clip(slope, -limits, limits)
        self._density = density
        self._slope = slope
        self._left_density = density - 0.5 * slope * widths
        if np.any(self._left_density < -1024.0 * _FLOAT_EPS * np.maximum(density, 1.0e-300)):
            raise FrozenExactFlowReferenceError("limited REF-PL density is not non-negative")

    def cdf(self, radii_m: NDArray[np.float64] | np.ndarray) -> NDArray[np.float64]:
        radii = np.asarray(radii_m, dtype=np.float64)
        if radii.ndim != 1 or not np.all(np.isfinite(radii)):
            raise FrozenExactFlowReferenceError("CDF radii must be finite and one-dimensional")
        if np.any(radii < self.edges_m[0]) or np.any(radii > self.edges_m[-1]):
            raise FrozenExactFlowReferenceError("CDF query lies outside the initial physical measure domain")
        indices = np.searchsorted(self.edges_m, radii, side="right") - 1
        indices = np.clip(indices, 0, self.cell_number_m3.size - 1)
        local = radii - self.edges_m[indices]
        values = self._cumulative[indices] + self._left_density[indices] * local + 0.5 * self._slope[indices] * local**2
        values = np.asarray(values, dtype=np.float64)
        values[radii == self.edges_m[0]] = 0.0
        values[radii == self.edges_m[-1]] = self.total_number_m3
        tolerance = 2048.0 * _FLOAT_EPS * max(self.total_number_m3, 1.0e-300)
        if not np.all(np.isfinite(values)) or np.any(values < -tolerance) or np.any(values > self.total_number_m3 + tolerance):
            raise FrozenExactFlowReferenceError("REF-PL CDF evaluation is invalid")
        return values


def measure_error_metrics(
    left_cells: NDArray[np.float64] | np.ndarray,
    right_cells: NDArray[np.float64] | np.ndarray,
    *,
    edges_m: NDArray[np.float64] | np.ndarray,
) -> dict[str, float | list[float]]:
    """Independent population, CDF, W1, and moment error metrics.

    The W1 calculation integrates the absolute difference of the normalized
    piecewise-linear CDFs exactly on the common target grid.  It does not call
    the CR1 CDF or a cohort quadrature helper.
    """

    edges = _as_edges(edges_m)
    left = _as_cells(left_cells, expected_size=edges.size - 1)
    right = _as_cells(right_cells, expected_size=edges.size - 1)
    delta = left - right
    absolute = np.abs(delta)
    left_total = float(np.sum(left, dtype=np.float64))
    right_total = float(np.sum(right, dtype=np.float64))
    widths = np.diff(edges)
    l1 = float(np.sum(absolute, dtype=np.float64))
    linf = float(np.max(absolute)) if absolute.size else 0.0
    left_cdf = np.concatenate((np.asarray([0.0]), np.cumsum(left, dtype=np.float64)))
    right_cdf = np.concatenate((np.asarray([0.0]), np.cumsum(right, dtype=np.float64)))
    if left_total > 0.0:
        left_cdf = left_cdf / left_total
    if right_total > 0.0:
        right_cdf = right_cdf / right_total
    difference = left_cdf - right_cdf
    wasserstein = 0.0
    for index, width in enumerate(widths):
        a = float(difference[index])
        b = float(difference[index + 1])
        if a == 0.0 and b == 0.0:
            continue
        if a * b >= 0.0:
            wasserstein += 0.5 * (abs(a) + abs(b)) * float(width)
        else:
            crossing = abs(a) / (abs(a) + abs(b))
            wasserstein += 0.5 * abs(a) * crossing * float(width)
            wasserstein += 0.5 * abs(b) * (1.0 - crossing) * float(width)
    moment_weights = np.empty((4, left.size), dtype=np.float64)
    for order in range(4):
        moment_weights[order] = (edges[1:] ** (order + 1) - edges[:-1] ** (order + 1)) / ((order + 1) * widths)
    signed = np.sum(moment_weights * delta[np.newaxis, :], axis=1, dtype=np.float64)
    weighted_absolute = np.sum(moment_weights * absolute[np.newaxis, :], axis=1, dtype=np.float64)
    result: dict[str, float | list[float]] = {
        "population_L1_abs": l1,
        "population_Linf_abs": linf,
        "population_relative_L1": l1 / max(right_total, 1.0e-300),
        "population_relative_Linf": linf / max(float(np.max(right)), 1.0e-300),
        "CDF_max_error": float(np.max(np.abs(difference))),
        "Wasserstein_m": float(wasserstein),
        "signed_moments_M0_to_M3": [float(value) for value in signed],
        "absolute_weighted_M0_to_M3": [float(value) for value in weighted_absolute],
    }
    for order in range(4):
        result[f"signed_M{order}"] = float(signed[order])
        result[f"absolute_weighted_M{order}"] = float(weighted_absolute[order])
    return result


def observed_orders(
    h_s: Sequence[float],
    errors: Sequence[float],
    *,
    metric: str,
) -> dict[str, Any]:
    """Calculate pairwise and late-level observed orders without assuming one."""

    if len(h_s) != len(errors):
        raise ValueError("h and error series must have equal length")
    rows: list[dict[str, Any]] = []
    finite_orders: list[float] = []
    for coarse_h, fine_h, coarse_error, fine_error in zip(h_s[:-1], h_s[1:], errors[:-1], errors[1:]):
        coarse = float(coarse_error)
        fine = float(fine_error)
        ratio_h = float(coarse_h) / float(fine_h)
        order: float | None
        if not math.isfinite(coarse) or not math.isfinite(fine) or coarse < 0.0 or fine < 0.0 or ratio_h <= 1.0:
            order = None
        elif coarse == 0.0 and fine == 0.0:
            order = None
        elif coarse == 0.0 or fine == 0.0:
            order = math.inf
        else:
            order = math.log(coarse / fine) / math.log(ratio_h)
            if math.isfinite(order):
                finite_orders.append(order)
        rows.append(
            {
                "metric": metric,
                "h_coarse_s": float(coarse_h),
                "h_fine_s": float(fine_h),
                "error_coarse": coarse,
                "error_fine": fine,
                "refinement_ratio": ratio_h,
                "observed_order": order,
            }
        )
    late = finite_orders[-3:]
    return {
        "metric": metric,
        "pair_rows": rows,
        "late_level_median_order": None if not late else float(np.median(np.asarray(late, dtype=np.float64))),
        "late_level_order_min": None if not late else float(np.min(np.asarray(late, dtype=np.float64))),
        "late_level_order_max": None if not late else float(np.max(np.asarray(late, dtype=np.float64))),
        "late_level_order_spread": None if not late else float(np.max(late) - np.min(late)),
        "finite_pair_count": len(finite_orders),
    }


def classify_refinement_against_reference(h_s: Sequence[float], errors: Sequence[float]) -> dict[str, Any]:
    """Classify CR1 refinement only from errors against the exact reference."""

    values = [float(value) for value in errors]
    if len(values) < 4 or any(not math.isfinite(value) or value < 0.0 for value in values):
        return {"classification": "INSUFFICIENT_REFINEMENT_RANGE", "reason": "fewer than four finite registered errors"}
    if all(value == 0.0 for value in values):
        return {"classification": "CR1_ASYMPTOTICALLY_CONSISTENT_FIRST_ORDER_LIKE", "reason": "all registered exact-reference errors are zero"}
    pairwise = observed_orders(h_s, values, metric="primary_population_relative_L1")
    late_orders = [
        row["observed_order"]
        for row in pairwise["pair_rows"][-3:]
        if isinstance(row["observed_order"], float) and math.isfinite(float(row["observed_order"]))
    ]
    late_errors = values[-4:]
    strictly_decreasing = all(right < left for left, right in zip(late_errors, late_errors[1:]))
    nonincreasing = all(right <= left for left, right in zip(late_errors, late_errors[1:]))
    if strictly_decreasing and len(late_orders) >= 2:
        median = float(np.median(np.asarray(late_orders, dtype=np.float64)))
        spread = float(np.max(late_orders) - np.min(late_orders))
        if 0.75 <= median <= 1.25 and spread <= 0.75:
            classification = "CR1_ASYMPTOTICALLY_CONSISTENT_FIRST_ORDER_LIKE"
        elif 0.0 < median < 0.75 and spread <= 0.75:
            classification = "CR1_ASYMPTOTICALLY_CONSISTENT_SUBFIRST_ORDER"
        else:
            classification = "CR1_CONVERGENT_BUT_PREASYMPTOTIC"
        return {
            "classification": classification,
            "reason": "last four exact-reference errors decrease strictly",
            "late_median_order": median,
            "late_order_spread": spread,
            "pairwise": pairwise,
        }
    if values[-1] >= values[-2] and values[-2] >= values[-3]:
        return {
            "classification": "CR1_ERROR_STAGNATION_AGAINST_EXACT_REFERENCE",
            "reason": "last two exact-reference refinements do not reduce the primary error",
            "pairwise": pairwise,
        }
    if nonincreasing and values[-1] < values[0]:
        return {
            "classification": "CR1_CONVERGENT_BUT_PREASYMPTOTIC",
            "reason": "registered errors decrease overall but lack a stable late order",
            "pairwise": pairwise,
        }
    return {
        "classification": "CR1_NONMONOTONE_NONCONVERGENT",
        "reason": "registered exact-reference errors do not exhibit a convergent late sequence",
        "pairwise": pairwise,
    }
