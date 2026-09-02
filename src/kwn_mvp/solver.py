"""Conservative finite-volume effective KWN solver with an exact inventory ledger."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import ConfigDocument, ConfigurationError, config_hash, require_mapping, require_number
from .growth import growth_rate_m_s
from .ledger import InventoryError, InventoryLedger, InventorySnapshot
from .lower_boundary import (
    boundary_growth_velocity,
    boundary_inventory_diagnostic,
    boundary_number_flux_diagnostic,
    boundary_radius,
    particle_inventory_at_radius,
)
from .nucleation import NucleationResult, nucleation_rate
from .populations import Population, PopulationParameters
from .radius_grid import RadiusGrid
from .thermo_adapter import (
    DiluteEquilibriumAdapter,
    ValidationContractEquilibriumAdapter,
    build_equilibrium_adapter,
)


class RadiusGridOverflowError(RuntimeError):
    """Raised when material would leave Rmax instead of being silently discarded."""


class SolverStateError(RuntimeError):
    """Raised when a finite-volume update violates positivity or the ledger contract."""


@dataclass(frozen=True)
class SolverConfig:
    """Fully explicit solver configuration in SI units."""

    temperature_k: float
    grid: RadiusGrid
    matrix_molar_volume_m3_mol: float
    initial_matrix_xb: float
    total_b_mol_m3: float | None
    inventory_tolerance_relative: float
    max_dt_s: float
    min_dt_s: float
    size_cfl: float
    accuracy_radius_cfl: float | None
    accuracy_active_radius_cfl: float | None
    active_cfl_support_fraction: float
    active_cfl_tail_lower_fraction: float
    active_cfl_tail_upper_fraction: float
    population_measure: str
    positivity_safety: float
    cfl_active_inventory_relative_threshold: float
    rmax_outflow_relative_tolerance: float
    thermo_mode: str
    planar_reference_xb: float | None
    validation_contract_path: str | None
    validation_contract_hash: str | None
    populations: Tuple[PopulationParameters, PopulationParameters]
    initial_population: Dict[str, Dict[str, Any]]
    source_config_hash: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SolverConfig":
        """Construct a validated configuration from a JSON-subset YAML mapping."""

        root = dict(data)
        simulation = require_mapping(root, "simulation")
        matrix = require_mapping(root, "matrix")
        grid_data = require_mapping(root, "radius_grid")
        thermo = require_mapping(root, "thermodynamics")
        populations_data = require_mapping(root, "populations")
        try:
            grid = RadiusGrid.logarithmic(
                require_number(grid_data, "minimum_m", positive=True),
                require_number(grid_data, "maximum_m", positive=True),
                int(require_number(grid_data, "bins", positive=True)),
            )
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid radius_grid: {exc}") from exc
        parsed: List[PopulationParameters] = []
        initial: Dict[str, Dict[str, Any]] = {}
        for name in ("g", "beta"):
            section = populations_data.get(name)
            if not isinstance(section, dict):
                raise ConfigurationError(f"populations.{name} must be a mapping")
            nucleation = section.get("nucleation", {"mode": "off"})
            if not isinstance(nucleation, dict):
                raise ConfigurationError(f"populations.{name}.nucleation must be a mapping")
            parsed.append(
                PopulationParameters(
                    name=name,
                    x_b=require_number(section, "xB", positive=True),
                    molar_volume_m3_mol=require_number(section, "molar_volume_m3_mol", positive=True),
                    diffusivity_m2_s=require_number(section, "diffusivity_m2_s"),
                    gamma_j_m2=require_number(section, "gamma_j_m2"),
                    xeq_infinity=require_number(section, "xeq_infinity", positive=True),
                    shape_factor=float(section.get("shape_factor", 1.0)),
                    elastic_penalty_j_m3=float(section.get("elastic_penalty_j_m3", 0.0)),
                    nucleation=dict(nucleation),
                )
            )
            population_initial = section.get("initial", {"kind": "empty"})
            if not isinstance(population_initial, dict):
                raise ConfigurationError(f"populations.{name}.initial must be a mapping")
            initial[name] = dict(population_initial)
        if parsed[1].nucleation.get("mode", "off") == "effective_cnt":
            raise ConfigurationError("MVP beta nucleation must remain off; direct g->beta is forbidden")
        total_value = matrix.get("total_b_mol_m3")
        if total_value is not None and not isinstance(total_value, (int, float)):
            raise ConfigurationError("matrix.total_b_mol_m3 must be numeric or null")
        temperature = require_number(simulation, "temperature_K", positive=True)
        initial_xb = require_number(matrix, "initial_xB")
        if not 0.0 <= initial_xb <= 1.0:
            raise ConfigurationError("matrix.initial_xB must be in [0, 1]")
        positivity_safety = float(simulation.get("positivity_safety", 0.95))
        if not 0.0 < positivity_safety <= 1.0:
            raise ConfigurationError("simulation.positivity_safety must lie in (0, 1]")
        cfl_support_threshold = float(
            simulation.get("cfl_active_inventory_relative_threshold", 1.0e-30)
        )
        if not 0.0 < cfl_support_threshold <= 1.0:
            raise ConfigurationError(
                "simulation.cfl_active_inventory_relative_threshold must lie in (0, 1]"
            )
        accuracy_radius_cfl_value = simulation.get("accuracy_radius_cfl")
        if accuracy_radius_cfl_value is not None and (
            not isinstance(accuracy_radius_cfl_value, (int, float))
            or float(accuracy_radius_cfl_value) <= 0.0
        ):
            raise ConfigurationError("simulation.accuracy_radius_cfl must be positive or null")
        accuracy_active_radius_cfl_value = simulation.get("accuracy_active_radius_cfl")
        if accuracy_active_radius_cfl_value is not None and (
            not isinstance(accuracy_active_radius_cfl_value, (int, float))
            or float(accuracy_active_radius_cfl_value) <= 0.0
        ):
            raise ConfigurationError(
                "simulation.accuracy_active_radius_cfl must be positive or null"
            )
        active_cfl_support_fraction = float(
            simulation.get("active_cfl_support_fraction", 1.0e-6)
        )
        if not 0.0 < active_cfl_support_fraction < 1.0:
            raise ConfigurationError("simulation.active_cfl_support_fraction must lie in (0, 1)")
        active_cfl_tail_lower_fraction = float(
            simulation.get("active_cfl_tail_lower_fraction", 1.0e-8)
        )
        active_cfl_tail_upper_fraction = float(
            simulation.get("active_cfl_tail_upper_fraction", 1.0e-6)
        )
        if not (
            0.0 <= active_cfl_tail_lower_fraction
            < active_cfl_tail_upper_fraction
            <= 1.0
        ):
            raise ConfigurationError(
                "active lower-tail fractions must satisfy 0 <= lower < upper <= 1"
            )
        population_measure = str(simulation.get("population_measure", "fixed_pivot"))
        if population_measure not in {"fixed_pivot", "cell_integrated"}:
            raise ConfigurationError(
                "simulation.population_measure must be 'fixed_pivot' or 'cell_integrated'"
            )
        return cls(
            temperature_k=temperature,
            grid=grid,
            matrix_molar_volume_m3_mol=require_number(matrix, "molar_volume_m3_mol", positive=True),
            initial_matrix_xb=initial_xb,
            total_b_mol_m3=None if total_value is None else float(total_value),
            inventory_tolerance_relative=require_number(matrix, "inventory_tolerance_relative", positive=True),
            max_dt_s=require_number(simulation, "max_dt_s", positive=True),
            min_dt_s=require_number(simulation, "min_dt_s", positive=True),
            size_cfl=require_number(simulation, "size_cfl", positive=True),
            accuracy_radius_cfl=(
                None if accuracy_radius_cfl_value is None else float(accuracy_radius_cfl_value)
            ),
            accuracy_active_radius_cfl=(
                None
                if accuracy_active_radius_cfl_value is None
                else float(accuracy_active_radius_cfl_value)
            ),
            active_cfl_support_fraction=active_cfl_support_fraction,
            active_cfl_tail_lower_fraction=active_cfl_tail_lower_fraction,
            active_cfl_tail_upper_fraction=active_cfl_tail_upper_fraction,
            population_measure=population_measure,
            positivity_safety=positivity_safety,
            cfl_active_inventory_relative_threshold=cfl_support_threshold,
            rmax_outflow_relative_tolerance=require_number(
                simulation, "rmax_outflow_relative_tolerance", positive=True
            ),
            thermo_mode=str(thermo.get("mode", "approximate_dilute")),
            planar_reference_xb=(
                None
                if thermo.get("planar_reference_xB") is None
                else float(thermo["planar_reference_xB"])
            ),
            validation_contract_path=(
                None
                if thermo.get("contract_path") is None
                else str(thermo["contract_path"])
            ),
            validation_contract_hash=(
                None
                if thermo.get("contract_hash") is None
                else str(thermo["contract_hash"])
            ),
            populations=(parsed[0], parsed[1]),
            initial_population=initial,
            source_config_hash=config_hash(root),
        )

    @classmethod
    def from_document(cls, document: ConfigDocument) -> "SolverConfig":
        """Construct a config while retaining the source document hash."""

        value = cls.from_mapping(document.data)
        return cls(
            **{**value.__dict__, "source_config_hash": document.sha256},
        )


@dataclass(frozen=True)
class StepDiagnostics:
    """One accepted KWN macro-step diagnostic record."""

    step: int
    time_s: float
    dt_s: float
    size_cfl: float
    positivity_utilization: float
    roundoff_zeroed_bin_count: int
    matrix_xb: float
    inventory: InventorySnapshot
    gp_nucleation_rate_m3_s: float
    beta_nucleation_rate_m3_s: float
    rmin_dissolution_flux_m3_s: float
    rmax_outflow_flux_m3_s: float
    beta_rmin_number_flux_m3_s: float = 0.0
    beta_rmin_volume_flux_s: float = 0.0
    beta_rmin_mol_b_flux_mol_m3_s: float = 0.0
    beta_boundary_radius_m: float = 0.0
    beta_boundary_growth_velocity_m_s: float = 0.0
    radius_courant_max: float = 0.0
    radius_courant_beta_max: float = 0.0
    radius_courant_face_index: int = -1
    radius_courant_cell_index: int = -1
    radius_courant_face_velocity_m_s: float = 0.0
    radius_courant_cell_width_m: float = 0.0
    timestep_limiter: str = "legacy_active_cell_cfl"
    active_population_courant: float = 0.0
    active_m0_courant: float = 0.0
    active_m3_courant: float = 0.0
    lower_tail_active_courant: float = 0.0
    boundary_candidate_courant: float = 0.0
    boundary_active_courant: float = 0.0
    boundary_face_is_active: bool = False
    active_m0_support_i_lo: int = -1
    active_m0_support_i_hi: int = -1
    active_m3_support_i_lo: int = -1
    active_m3_support_i_hi: int = -1
    lower_tail_i_lo: int = -1
    lower_tail_i_hi: int = -1
    lower_tail_m0_fraction: float = 0.0


@dataclass(frozen=True)
class ActiveCFLDiagnostics:
    """Fixed, measure-aware radius-space CFL audit for one FV population.

    The raw global face rate remains available for stability/telemetry, but
    accuracy qualification is tied to physically occupied M0/M3 support and
    a declared lower-tail CDF band.  The lower physical face is only an
    active accuracy limiter when the tail can actually reach it in the
    frozen-step characteristic geometry; its flux still executes every step.
    """

    population_name: str
    global_max_rate_s_inv: float
    population_active_rate_s_inv: float
    m0_rate_s_inv: float
    m3_rate_s_inv: float
    lower_tail_rate_s_inv: float
    boundary_candidate_rate_s_inv: float
    boundary_active_rate_s_inv: float
    boundary_face_is_active: bool
    m0_support_i_lo: int
    m0_support_i_hi: int
    m3_support_i_lo: int
    m3_support_i_hi: int
    lower_tail_i_lo: int
    lower_tail_i_hi: int
    m0_covered_fraction: float
    m3_covered_fraction: float
    lower_tail_m0_fraction: float
    active_face_indices: tuple[int, ...]
    lower_tail_face_indices: tuple[int, ...]


def _normalised_lognormal_density(
    *, grid: RadiusGrid, number_density_m3: float, median_radius_m: float, log_sigma: float
) -> NDArray[np.float64]:
    """Return a cell-integrated-normalised lognormal n(R) in m^-4."""

    if number_density_m3 < 0.0 or median_radius_m <= 0.0 or log_sigma <= 0.0:
        raise ConfigurationError("lognormal initial state requires N>=0, median>0, log_sigma>0")
    radii = grid.centres_m
    pdf = np.exp(-0.5 * (np.log(radii / median_radius_m) / log_sigma) ** 2)
    pdf /= radii * log_sigma * np.sqrt(2.0 * np.pi)
    normalisation = float(np.sum(pdf * grid.widths_m))
    if normalisation <= 0.0:
        raise ConfigurationError("Initial lognormal population has zero support on configured grid")
    return number_density_m3 * pdf / normalisation


def _build_initial_population(
    *,
    parameters: PopulationParameters,
    grid: RadiusGrid,
    definition: Mapping[str, Any],
    production_quadrature: str,
) -> Population:
    """Construct an initial PSD from an explicit config-owned definition."""

    kind = str(definition.get("kind", "empty"))
    population = Population.empty(
        parameters, grid, production_quadrature=production_quadrature
    )
    if kind == "empty":
        return population
    if kind == "monodisperse":
        radius = float(definition["radius_m"])
        count = float(definition["number_density_m3"])
        population.add_number_at_radius(radius, count)
        return population
    if kind == "lognormal":
        population.number_density_per_m4[:] = _normalised_lognormal_density(
            grid=grid,
            number_density_m3=float(definition["number_density_m3"]),
            median_radius_m=float(definition["median_radius_m"]),
            log_sigma=float(definition["log_sigma"]),
        )
        return population
    if kind == "discrete":
        entries = definition.get("entries")
        if not isinstance(entries, list):
            raise ConfigurationError("discrete initial population requires list 'entries'")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ConfigurationError("Each discrete initial entry must be a mapping")
            population.add_number_at_radius(float(entry["radius_m"]), float(entry["number_density_m3"]))
        return population
    if kind == "cell_integrated":
        values = definition.get("cell_number_density_m3")
        if not isinstance(values, list) or len(values) != grid.bins:
            raise ConfigurationError(
                "cell_integrated initial population requires one cell_number_density_m3 value per grid bin"
            )
        declared_edges = definition.get("radius_edges_m")
        if declared_edges is not None:
            edges = np.asarray(declared_edges, dtype=np.float64)
            if edges.shape != grid.edges_m.shape or not np.array_equal(edges, grid.edges_m):
                raise ConfigurationError(
                    "cell_integrated initial population radius_edges_m must exactly match the solver grid"
                )
        numbers = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(numbers)) or np.any(numbers < 0.0):
            raise ConfigurationError("cell_integrated initial population numbers must be finite and non-negative")
        population.number_density_per_m4[:] = numbers / grid.widths_m
        return population
    raise ConfigurationError(f"Unsupported initial population kind {kind!r}")


class KWNSolver:
    """Two-population KWN solver with conservative radius-space fluxes.

    The solver is a mean-field model.  It never claims equivalence to the
    spatially resolved, elastic phase-field trajectory and never includes a
    direct g-to-beta conversion term.
    """

    solver_version = "internal_kwn_finite_volume_v1"

    def __init__(self, config: SolverConfig) -> None:
        """Initialise state from a fully validated configuration."""

        if not 0.0 < config.size_cfl <= 0.4:
            raise ConfigurationError("size_cfl must lie in (0, 0.4] for positivity preservation")
        if not 0.0 < config.positivity_safety <= 1.0:
            raise ConfigurationError("positivity_safety must lie in (0, 1]")
        if not 0.0 < config.cfl_active_inventory_relative_threshold <= 1.0:
            raise ConfigurationError("cfl_active_inventory_relative_threshold must lie in (0, 1]")
        if config.min_dt_s > config.max_dt_s:
            raise ConfigurationError("min_dt_s cannot exceed max_dt_s")
        self.config = config
        self.populations: Dict[str, Population] = {
            parameters.name: _build_initial_population(
                parameters=parameters,
                grid=config.grid,
                definition=config.initial_population[parameters.name],
                production_quadrature=config.population_measure,
            )
            for parameters in config.populations
        }
        self.equilibrium_adapter: DiluteEquilibriumAdapter | ValidationContractEquilibriumAdapter = build_equilibrium_adapter(
            mode=config.thermo_mode,
            temperature_k=config.temperature_k,
            planar_reference_xb=config.planar_reference_xb,
            contract_path=config.validation_contract_path,
            expected_contract_hash=config.validation_contract_hash,
        )
        if config.total_b_mol_m3 is None:
            total = self._derive_initial_total_b_mol_m3(config.initial_matrix_xb)
        else:
            total = config.total_b_mol_m3
        self.ledger = InventoryLedger(
            matrix_molar_volume_m3_mol=config.matrix_molar_volume_m3_mol,
            total_b_mol_m3=total,
            tolerance_relative=config.inventory_tolerance_relative,
        )
        self.matrix_xb = float(config.initial_matrix_xb)
        self.ledger.snapshot(matrix_xb=self.matrix_xb, populations=self.population_list())
        self.time_s = 0.0
        self.step = 0
        self.roundoff_zeroed_bin_count = 0
        self.history: List[StepDiagnostics] = []
        self._last_timestep_telemetry: dict[str, Any] = {
            "limiter": "maximum_dt",
            "overall": (0.0, -1, -1, 0.0, 0.0),
            "beta": (0.0, -1, -1, 0.0, 0.0),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "KWNSolver":
        """Create a solver from raw JSON-subset YAML data."""

        return cls(SolverConfig.from_mapping(data))

    @classmethod
    def from_document(cls, document: ConfigDocument) -> "KWNSolver":
        """Create a solver from parsed configuration provenance."""

        return cls(SolverConfig.from_document(document))

    def population_list(self) -> List[Population]:
        """Return g and beta in the ledger's canonical order."""

        return [self.populations["g"], self.populations["beta"]]

    def _derive_initial_total_b_mol_m3(self, matrix_xb: float) -> float:
        """Derive a fixed total inventory from the explicitly configured initial state."""

        gp = self.populations["g"]
        beta = self.populations["beta"]
        matrix_fraction = 1.0 - gp.volume_fraction() - beta.volume_fraction()
        if matrix_fraction <= 0.0:
            raise InventoryError("Cannot derive total inventory without positive matrix fraction")
        return (
            matrix_fraction * matrix_xb / self.config.matrix_molar_volume_m3_mol
            + gp.b_inventory_mol_m3()
            + beta.b_inventory_mol_m3()
        )

    def population(self, name: str) -> Population:
        """Return named population state."""

        try:
            return self.populations[name]
        except KeyError as exc:
            raise KeyError(f"No KWN population named {name!r}") from exc

    @property
    def contract_hash(self) -> str | None:
        """Return the hash governing this KWN run, if it uses the validation path."""

        return getattr(self.equilibrium_adapter, "contract_hash", None)

    def equilibrium_xb(self, population: Population) -> NDArray[np.float64]:
        """Return the current curvature-corrected equilibrium for one population."""

        return self.equilibrium_adapter.equilibrium_xb(population.grid.centres_m, population.parameters)

    def growth_rates(self) -> Dict[str, NDArray[np.float64]]:
        """Return current growth/dissolution velocities for both populations."""

        rates: Dict[str, NDArray[np.float64]] = {}
        for name, population in self.populations.items():
            if (
                population.number_density_m3() == 0.0
                and population.parameters.nucleation.get("mode", "off") == "off"
            ):
                # An absent, non-nucleating population must not require an
                # equilibrium evaluation outside the grid actually in use.
                rates[name] = np.zeros(population.grid.bins, dtype=np.float64)
                continue
            rates[name] = growth_rate_m_s(
                radii_m=population.grid.centres_m,
                matrix_xb=self.matrix_xb,
                equilibrium_xb=self.equilibrium_xb(population),
                parameters=population.parameters,
            )
        return rates

    def lower_boundary_growth_velocity(self, population: Population | str) -> float:
        """Return the physical shared-kernel velocity at this population's ``Rmin``."""

        item = self.population(population) if isinstance(population, str) else population
        return boundary_growth_velocity(
            radius_m=boundary_radius(item.grid),
            matrix_xb=self.matrix_xb,
            parameters=item.parameters,
            equilibrium_adapter=self.equilibrium_adapter,
        )

    def lower_boundary_particle_inventory(self, population: Population | str):
        """Return the shared per-particle inventory price at physical ``Rmin``."""

        item = self.population(population) if isinstance(population, str) else population
        return particle_inventory_at_radius(
            boundary_radius(item.grid),
            x_b=item.parameters.x_b,
            molar_volume_m3_mol=item.parameters.molar_volume_m3_mol,
        )

    def face_velocities(
        self, population: Population | str, velocity_m_s: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Return the FV face operator with its lower face at physical ``Rmin``.

        An empty, non-nucleating population has no resolved donor and therefore
        no active lower-bound transport operator.  This does not create an
        external inflow; it merely prevents an absent population from
        contributing a meaningless raw CFL diagnostic.
        """

        item = self.population(population) if isinstance(population, str) else population
        if velocity_m_s is None:
            velocity_m_s = self.growth_rates()[item.parameters.name]
        if (
            item.number_density_m3() == 0.0
            and item.parameters.nucleation.get("mode", "off") == "off"
        ):
            lower = 0.0
        else:
            lower = self.lower_boundary_growth_velocity(item)
        return self._face_velocities(velocity_m_s, lower_boundary_velocity_m_s=lower)

    @staticmethod
    def _face_operator_rates(
        face_velocity_m_s: NDArray[np.float64], widths_m: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Return literal face rates and the cell-width owner of each rate."""

        widths = np.asarray(widths_m, dtype=np.float64)
        face_velocity = np.asarray(face_velocity_m_s, dtype=np.float64)
        if (
            widths.ndim != 1
            or widths.size < 1
            or face_velocity.ndim != 1
            or face_velocity.size != widths.size + 1
            or not np.all(np.isfinite(face_velocity))
            or not np.all(np.isfinite(widths))
            or np.any(widths <= 0.0)
        ):
            raise SolverStateError("face velocity and radius widths are inconsistent")
        rates = np.empty(face_velocity.size, dtype=np.float64)
        cells = np.empty(face_velocity.size, dtype=np.int64)
        rates[0] = abs(face_velocity[0]) / widths[0]
        cells[0] = 0
        rates[-1] = abs(face_velocity[-1]) / widths[-1]
        cells[-1] = widths.size - 1
        if widths.size > 1:
            left = abs(face_velocity[1:-1]) / widths[:-1]
            right = abs(face_velocity[1:-1]) / widths[1:]
            use_left = left >= right
            rates[1:-1] = np.where(use_left, left, right)
            cells[1:-1] = np.where(
                use_left, np.arange(widths.size - 1), np.arange(1, widths.size)
            )
        return rates, cells

    @staticmethod
    def _shortest_contiguous_support(
        weights: NDArray[np.float64], edges_m: NDArray[np.float64], coverage: float
    ) -> tuple[int, int, float] | None:
        """Find the minimum log-radius interval carrying a fixed mass fraction.

        The exact discrete cell measure is the authority.  A deterministic
        ``(log span, cell count, lower index)`` tie-break makes the active
        support stable across repeated diagnostics.
        """

        mass = np.asarray(weights, dtype=np.float64)
        edges = np.asarray(edges_m, dtype=np.float64)
        if mass.ndim != 1 or edges.shape != (mass.size + 1,):
            raise SolverStateError("support weights and radius edges are inconsistent")
        if np.any(~np.isfinite(mass)) or np.any(mass < 0.0):
            raise SolverStateError("support weights must be finite and non-negative")
        total = math.fsum(float(value) for value in mass)
        if total <= 0.0:
            return None
        if not 0.0 < coverage <= 1.0:
            raise SolverStateError("support coverage must lie in (0, 1]")
        target = coverage * total
        best: tuple[float, int, int, int, float] | None = None
        left = 0
        running = 0.0
        for right, value in enumerate(mass):
            running += float(value)
            while left < right and running - float(mass[left]) >= target:
                running -= float(mass[left])
                left += 1
            if running + 1.0e-15 * total < target:
                continue
            span = math.log(float(edges[right + 1]) / float(edges[left]))
            candidate = (span, right - left + 1, left, right, running / total)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
        if best is None:
            # A finite non-negative vector must hit its own total, but retain
            # a clear failure rather than silently changing the support rule.
            raise SolverStateError("unable to form requested active-support coverage")
        return int(best[2]), int(best[3]), float(best[4])

    @staticmethod
    def _lower_tail_interval(
        weights: NDArray[np.float64], *, lower_fraction: float, upper_fraction: float
    ) -> tuple[int, int, float] | None:
        """Return the fixed lower-CDF cell band used for boundary accuracy."""

        mass = np.asarray(weights, dtype=np.float64)
        if mass.ndim != 1 or np.any(~np.isfinite(mass)) or np.any(mass < 0.0):
            raise SolverStateError("lower-tail weights must be finite and non-negative")
        total = math.fsum(float(value) for value in mass)
        if total <= 0.0:
            return None
        if not 0.0 <= lower_fraction < upper_fraction <= 1.0:
            raise SolverStateError("invalid lower-tail CDF interval")
        cdf = np.cumsum(mass, dtype=np.float64)
        low_target = lower_fraction * total
        high_target = upper_fraction * total
        lo = int(np.searchsorted(cdf, low_target, side="left"))
        hi = int(np.searchsorted(cdf, high_target, side="left"))
        lo = min(max(lo, 0), mass.size - 1)
        hi = min(max(hi, lo), mass.size - 1)
        fraction = math.fsum(float(value) for value in mass[lo : hi + 1]) / total
        return lo, hi, fraction

    @staticmethod
    def _faces_incident_to_interval(i_lo: int, i_hi: int) -> set[int]:
        """Return all FV faces incident on a closed cell interval."""

        return set(range(i_lo, i_hi + 2))

    @staticmethod
    def _reachable_faces_from_interval(
        *,
        i_lo: int,
        i_hi: int,
        face_velocity_m_s: NDArray[np.float64],
        widths_m: NDArray[np.float64],
        dt_s: float,
    ) -> set[int]:
        """Add frozen-characteristic downstream faces reachable in ``dt_s``.

        This is deliberately a cumulative traversal-time closure rather than
        an arbitrary halo.  It keeps an empty far tail out of the active-CFL
        policy while including every face a support-boundary characteristic
        can physically cross during the proposed macro-step.
        """

        face_velocity = np.asarray(face_velocity_m_s, dtype=np.float64)
        widths = np.asarray(widths_m, dtype=np.float64)
        if dt_s < 0.0 or not math.isfinite(float(dt_s)):
            raise SolverStateError("candidate timestep must be finite and non-negative")
        faces = KWNSolver._faces_incident_to_interval(i_lo, i_hi)
        # Negative transport moves toward lower R.  Starting at the lower
        # support edge, it first crosses face i_lo and then accumulates the
        # time needed to traverse each lower cell to its next left face.
        if face_velocity[i_lo] < 0.0:
            faces.add(i_lo)
            elapsed = 0.0
            cell = i_lo - 1
            while cell >= 0 and face_velocity[cell] < 0.0:
                elapsed += float(widths[cell]) / abs(float(face_velocity[cell]))
                if elapsed > dt_s:
                    break
                faces.add(cell)
                cell -= 1
        # Positive transport moves toward larger R by the symmetric rule.
        right_face = i_hi + 1
        if face_velocity[right_face] > 0.0:
            faces.add(right_face)
            elapsed = 0.0
            cell = i_hi + 1
            while cell < widths.size and face_velocity[cell + 1] > 0.0:
                elapsed += float(widths[cell]) / abs(float(face_velocity[cell + 1]))
                if elapsed > dt_s:
                    break
                faces.add(cell + 1)
                cell += 1
        return faces

    @staticmethod
    def _lower_boundary_reachable(
        *,
        tail_i_lo: int,
        face_velocity_m_s: NDArray[np.float64],
        widths_m: NDArray[np.float64],
        dt_s: float,
    ) -> bool:
        """Whether a lower-tail characteristic can reach physical ``Rmin``."""

        face_velocity = np.asarray(face_velocity_m_s, dtype=np.float64)
        widths = np.asarray(widths_m, dtype=np.float64)
        if face_velocity[0] >= 0.0:
            return False
        if tail_i_lo == 0:
            return True
        # Entering the first cell requires each intervening face to carry
        # negative transport.  The tail lower edge is used, so we never
        # claim the entire tail crosses merely because its cell has width.
        if face_velocity[tail_i_lo] >= 0.0:
            return False
        elapsed = 0.0
        for cell in range(tail_i_lo - 1, -1, -1):
            velocity = float(face_velocity[cell])
            if velocity >= 0.0:
                return False
            elapsed += float(widths[cell]) / abs(velocity)
            if elapsed > dt_s:
                return False
        return True

    def active_cfl_diagnostics(
        self,
        population: Population | str,
        *,
        face_velocity_m_s: NDArray[np.float64] | None = None,
        dt_s: float = 0.0,
    ) -> ActiveCFLDiagnostics:
        """Evaluate the fixed V1 active-population CFL contract.

        This diagnostic has no effect on flux assembly.  It is used for
        accuracy timestep selection only when ``accuracy_active_radius_cfl``
        is configured, and otherwise is recorded as telemetry.
        """

        item = self.population(population) if isinstance(population, str) else population
        density = np.asarray(item.number_density_per_m4, dtype=np.float64)
        if face_velocity_m_s is None:
            face_velocity_m_s = self.face_velocities(item)
        face_velocity = np.asarray(face_velocity_m_s, dtype=np.float64)
        rates, _ = self._face_operator_rates(face_velocity, item.grid.widths_m)
        m0 = density * item.grid.widths_m
        m3 = density * (item.grid.edges_m[1:] ** 4 - item.grid.edges_m[:-1] ** 4) / 4.0
        coverage = 1.0 - self.config.active_cfl_support_fraction
        m0_support = self._shortest_contiguous_support(m0, item.grid.edges_m, coverage)
        m3_support = self._shortest_contiguous_support(m3, item.grid.edges_m, coverage)
        tail = self._lower_tail_interval(
            m0,
            lower_fraction=self.config.active_cfl_tail_lower_fraction,
            upper_fraction=self.config.active_cfl_tail_upper_fraction,
        )
        active_faces: set[int] = set()
        m0_faces: set[int] = set()
        m3_faces: set[int] = set()
        tail_faces: set[int] = set()
        if m0_support is not None:
            m0_faces = self._reachable_faces_from_interval(
                i_lo=m0_support[0],
                i_hi=m0_support[1],
                face_velocity_m_s=face_velocity,
                widths_m=item.grid.widths_m,
                dt_s=dt_s,
            )
            active_faces.update(m0_faces)
        if m3_support is not None:
            m3_faces = self._reachable_faces_from_interval(
                i_lo=m3_support[0],
                i_hi=m3_support[1],
                face_velocity_m_s=face_velocity,
                widths_m=item.grid.widths_m,
                dt_s=dt_s,
            )
            active_faces.update(m3_faces)
        if tail is not None:
            tail_faces = self._reachable_faces_from_interval(
                i_lo=tail[0],
                i_hi=tail[1],
                face_velocity_m_s=face_velocity,
                widths_m=item.grid.widths_m,
                dt_s=dt_s,
            )
        boundary_active = bool(
            tail is not None
            and self._lower_boundary_reachable(
                tail_i_lo=tail[0],
                face_velocity_m_s=face_velocity,
                widths_m=item.grid.widths_m,
                dt_s=dt_s,
            )
        )

        def maximum(face_indices: set[int]) -> float:
            return 0.0 if not face_indices else float(np.max(rates[sorted(face_indices)]))

        return ActiveCFLDiagnostics(
            population_name=item.parameters.name,
            global_max_rate_s_inv=float(np.max(rates)),
            population_active_rate_s_inv=maximum(active_faces),
            m0_rate_s_inv=maximum(m0_faces),
            m3_rate_s_inv=maximum(m3_faces),
            lower_tail_rate_s_inv=maximum(tail_faces),
            boundary_candidate_rate_s_inv=float(rates[0]),
            boundary_active_rate_s_inv=float(rates[0]) if boundary_active else 0.0,
            boundary_face_is_active=boundary_active,
            m0_support_i_lo=-1 if m0_support is None else m0_support[0],
            m0_support_i_hi=-1 if m0_support is None else m0_support[1],
            m3_support_i_lo=-1 if m3_support is None else m3_support[0],
            m3_support_i_hi=-1 if m3_support is None else m3_support[1],
            lower_tail_i_lo=-1 if tail is None else tail[0],
            lower_tail_i_hi=-1 if tail is None else tail[1],
            m0_covered_fraction=0.0 if m0_support is None else m0_support[2],
            m3_covered_fraction=0.0 if m3_support is None else m3_support[2],
            lower_tail_m0_fraction=0.0 if tail is None else tail[2],
            active_face_indices=tuple(sorted(active_faces)),
            lower_tail_face_indices=tuple(sorted(tail_faces)),
        )

    @staticmethod
    def _face_velocities(
        velocity_m_s: NDArray[np.float64], *, lower_boundary_velocity_m_s: float | None = None
    ) -> NDArray[np.float64]:
        """Return FV face velocities, optionally fixing the lower face at ``Rmin``."""

        velocity = np.asarray(velocity_m_s, dtype=np.float64)
        if velocity.ndim != 1 or velocity.size < 2:
            raise SolverStateError("finite-volume velocity must have at least two cell values")
        faces = np.empty(velocity.size + 1, dtype=np.float64)
        lower = float(velocity[0]) if lower_boundary_velocity_m_s is None else float(lower_boundary_velocity_m_s)
        if not np.isfinite(lower):
            raise SolverStateError("lower finite-volume face velocity must be finite")
        faces[0] = lower
        faces[1:-1] = 0.5 * (velocity[:-1] + velocity[1:])
        faces[-1] = velocity[-1]
        return faces

    @classmethod
    def _upwind_face_fluxes(
        cls,
        density_per_m4: NDArray[np.float64],
        velocity_m_s: NDArray[np.float64],
        *,
        face_velocity_m_s: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Return the exact first-order face fluxes used by the FV update."""

        faces = np.empty(density_per_m4.size + 1, dtype=np.float64)
        face_velocity = (
            cls._face_velocities(velocity_m_s)
            if face_velocity_m_s is None
            else np.asarray(face_velocity_m_s, dtype=np.float64)
        )
        if face_velocity.shape != faces.shape or not np.all(np.isfinite(face_velocity)):
            raise SolverStateError("finite-volume face velocity shape or values are invalid")
        faces[0] = (
            face_velocity[0] * density_per_m4[0] if face_velocity[0] < 0.0 else 0.0
        )
        internal_velocity = face_velocity[1:-1]
        faces[1:-1] = np.where(
            internal_velocity >= 0.0,
            internal_velocity * density_per_m4[:-1],
            internal_velocity * density_per_m4[1:],
        )
        faces[-1] = (
            face_velocity[-1] * density_per_m4[-1] if face_velocity[-1] > 0.0 else 0.0
        )
        return faces

    @classmethod
    def _raw_face_operator_rate(
        cls,
        velocity_m_s: NDArray[np.float64],
        widths_m: NDArray[np.float64],
        *,
        face_velocity_m_s: NDArray[np.float64] | None = None,
    ) -> tuple[float, int, int, float, float]:
        """Return the all-grid face-Courant rate of the actual FV operator.

        Each internal face is incident on two cells, so its rate is measured
        against the smaller of the two relevant radius-cell widths.  Unlike
        the legacy active-inventory CFL this intentionally includes every
        grid face, even where the current density is extremely small.
        """

        widths = np.asarray(widths_m, dtype=np.float64)
        face_velocity = (
            cls._face_velocities(velocity_m_s)
            if face_velocity_m_s is None
            else np.asarray(face_velocity_m_s, dtype=np.float64)
        )
        rates, cells = cls._face_operator_rates(face_velocity, widths)
        index = int(np.argmax(rates))
        cell = int(cells[index])
        return (
            float(rates[index]),
            index,
            cell,
            float(face_velocity[index]),
            float(widths[cell]),
        )

    def _choose_dt(
        self,
        velocities: Mapping[str, NDArray[np.float64]],
        maximum_s: float | None,
        *,
        face_velocities: Mapping[str, NDArray[np.float64]] | None = None,
    ) -> Tuple[float, float]:
        """Choose an adaptive step and record raw plus physical active CFLs.

        ``size_cfl`` preserves the historical inventory-weighted implicit
        policy.  The optional ``accuracy_active_radius_cfl`` is a separate,
        fixed V1 accuracy limiter: it deliberately never applies the raw
        all-grid face maximum to empty bins.
        """

        cfl_rates: List[float] = []
        raw_by_population: dict[str, tuple[float, int, int, float, float]] = {}
        for name, velocity in velocities.items():
            population = self.populations[name]
            raw_by_population[name] = self._raw_face_operator_rate(
                velocity,
                population.grid.widths_m,
                face_velocity_m_s=(
                    None if face_velocities is None else face_velocities.get(name)
                ),
            )
            cell_number = population.number_density_per_m4 * population.grid.widths_m
            cell_volume = cell_number * (4.0 * np.pi / 3.0) * population.grid.centres_m**3
            total_volume = float(np.sum(cell_volume))
            if total_volume == 0.0:
                continue
            # The CFL support is based on conserved precipitate volume, not
            # particle count: a numerically transported tiny-radius tail can
            # carry many count-weighted particles but negligible inventory.
            # It remains in the conservative face update; it simply cannot
            # impose a global timestep unrelated to its B inventory.
            active = cell_volume > max(
                total_volume * self.config.cfl_active_inventory_relative_threshold, 1.0e-300
            )
            # Include immediate receivers/donors at each active face.  The
            # conservative implicit face solve below advances the remaining
            # low-inventory tail without allowing it to impose a global
            # explicit-CFL micro-step.
            active[:-1] |= active[1:]
            active[1:] |= active[:-1]
            if np.any(active):
                cfl_rates.append(
                    float(np.max(np.abs(velocity[active]) / population.grid.widths_m[active]))
                )
        max_rate = max(cfl_rates, default=0.0)
        cap = self.config.max_dt_s if maximum_s is None else min(self.config.max_dt_s, maximum_s)
        if max_rate == 0.0:
            cfl_dt = cap
        else:
            cfl_dt = min(cap, self.config.size_cfl / max_rate)
        dt = cfl_dt
        limiter = "legacy_active_cell_cfl" if cfl_dt < cap else "maximum_dt"
        overall = max(raw_by_population.values(), key=lambda value: value[0], default=(0.0, -1, -1, 0.0, 0.0))
        if self.config.accuracy_radius_cfl is not None and overall[0] > 0.0:
            accuracy_dt = self.config.accuracy_radius_cfl / overall[0]
            if accuracy_dt < dt:
                dt = accuracy_dt
                limiter = "accuracy_radius_cfl"
        active_by_population: dict[str, ActiveCFLDiagnostics] = {}
        if self.config.accuracy_active_radius_cfl is not None:
            # Reachability is defined at the *candidate* timestep.  A lower
            # physical face can therefore be active for the legacy macrostep
            # but inactive after a smaller candidate subdivides the same
            # characteristic path.  The fixed point must be allowed to grow
            # again in that situation; a shrink-only iteration would retain
            # an obsolete Rmin face and manufacture a false micro-step.
            base_dt = dt

            def active_diagnostics_at(candidate_dt: float) -> dict[str, ActiveCFLDiagnostics]:
                return {
                    name: self.active_cfl_diagnostics(
                        self.populations[name],
                        face_velocity_m_s=(
                            self._face_velocities(velocity)
                            if face_velocities is None
                            else face_velocities[name]
                        ),
                        dt_s=candidate_dt,
                    )
                    for name, velocity in velocities.items()
                }

            def active_rate_for(
                diagnostics: Mapping[str, ActiveCFLDiagnostics]
            ) -> float:
                return max(
                    (
                        max(
                            item.population_active_rate_s_inv,
                            item.lower_tail_rate_s_inv,
                            item.boundary_active_rate_s_inv,
                        )
                        for item in diagnostics.values()
                    ),
                    default=0.0,
                )

            def active_candidate_dt(candidate_dt: float) -> tuple[float, dict[str, ActiveCFLDiagnostics]]:
                diagnostics = active_diagnostics_at(candidate_dt)
                rate = active_rate_for(diagnostics)
                limited = base_dt if rate == 0.0 else min(
                    base_dt, self.config.accuracy_active_radius_cfl / rate
                )
                return limited, diagnostics

            # Picard iteration is fast in the normal case and, unlike the
            # former implementation, reaches the self-consistent subdivided
            # state after a candidate Rmin face deactivates.  A discontinuous
            # reachability threshold can make a short cycle; the conservative
            # bisection fallback then returns the largest admissible step.
            seen: list[float] = []
            converged = False
            for _ in range(12):
                candidate, active_by_population = active_candidate_dt(dt)
                if math.isclose(candidate, dt, rel_tol=1.0e-13, abs_tol=0.0):
                    dt = candidate
                    converged = True
                    break
                if any(math.isclose(candidate, previous, rel_tol=1.0e-13, abs_tol=0.0) for previous in seen):
                    break
                seen.append(dt)
                dt = candidate
            if not converged:
                # ``rate(dt)`` is non-decreasing as the reachable-face set
                # grows with dt, so dt*rate(dt) has a monotone admissible
                # predicate.  Bisection avoids committing a non-self-
                # consistent reachability classification at a discontinuity.
                lower_dt = 0.0
                upper_dt = base_dt
                for _ in range(64):
                    middle_dt = 0.5 * (lower_dt + upper_dt)
                    middle_diagnostics = active_diagnostics_at(middle_dt)
                    middle_rate = active_rate_for(middle_diagnostics)
                    if middle_dt * middle_rate <= self.config.accuracy_active_radius_cfl:
                        lower_dt = middle_dt
                    else:
                        upper_dt = middle_dt
                dt = lower_dt
                active_by_population = active_diagnostics_at(dt)
            # Record masks at the exact accepted candidate and make the
            # contract explicit rather than relying on an iteration history.
            active_by_population = active_diagnostics_at(dt)
            final_active_rate = active_rate_for(active_by_population)
            if dt * final_active_rate > self.config.accuracy_active_radius_cfl * (1.0 + 1.0e-12):
                raise SolverStateError("active-CFL fixed point did not satisfy its declared candidate-step contract")
            if dt < base_dt * (1.0 - 1.0e-14):
                limiter = "accuracy_active_radius_cfl"
        else:
            active_by_population = {
                name: self.active_cfl_diagnostics(
                    self.populations[name],
                    face_velocity_m_s=(
                        self._face_velocities(velocity)
                        if face_velocities is None
                        else face_velocities[name]
                    ),
                    dt_s=dt,
                )
                for name, velocity in velocities.items()
            }
        if dt < self.config.min_dt_s:
            raise SolverStateError(
                f"CFL-limited step {dt:.3e} s is below configured min_dt_s={self.config.min_dt_s:.3e} s"
            )
        cfl = max_rate * dt
        self._last_timestep_telemetry = {
            "limiter": limiter,
            "overall": overall,
            "beta": raw_by_population.get("beta", (0.0, -1, -1, 0.0, 0.0)),
            "active": active_by_population,
        }
        return dt, cfl

    def _advect_population(
        self,
        population: Population,
        velocity_m_s: NDArray[np.float64],
        dt_s: float,
        *,
        face_velocity_m_s: NDArray[np.float64] | None = None,
    ) -> Tuple[float, float, float, int]:
        """Advance one conservative first-order implicit upwind face update.

        Returns lower-boundary dissolution and Rmax outflow number fluxes in
        m^-3 s^-1.  Face velocities are frozen at the accepted start state,
        while donor densities are solved at the end state.  This is an
        M-matrix update: it is conservative across every internal face and
        positive without a negative-bin clamp, including the stiff Rmin tail.
        Rmax outflow is rejected before material can disappear.
        """

        density = population.number_density_per_m4
        widths = population.grid.widths_m
        if not np.any(density):
            return 0.0, 0.0, 0.0, 0
        face_velocity = (
            self._face_velocities(velocity_m_s)
            if face_velocity_m_s is None
            else np.asarray(face_velocity_m_s, dtype=np.float64)
        )
        if face_velocity.shape != (density.size + 1,) or not np.all(np.isfinite(face_velocity)):
            raise SolverStateError("finite-volume face velocity shape or values are invalid")
        raw_faces = self._upwind_face_fluxes(
            density, velocity_m_s, face_velocity_m_s=face_velocity
        )
        raw_rmax_outflow_flux = max(raw_faces[-1], 0.0)
        existing_number = max(population.number_density_m3(), 1.0e-300)
        relative_outflow = raw_rmax_outflow_flux * dt_s / existing_number
        if relative_outflow > self.config.rmax_outflow_relative_tolerance:
            raise RadiusGridOverflowError(
                f"{population.parameters.name} would lose {relative_outflow:.3e} of its number density "
                "through Rmax. Expand the configured radius grid; material was not discarded."
            )

        # Assemble ``n_new + dt * div(F_new) = n_old``.  Each upwind face has
        # one donor, so the system is tridiagonal with positive diagonal and
        # non-positive off-diagonal entries.  The Thomas solve below preserves
        # the literal shared-face sign in both neighbouring cells.
        bin_count = density.size
        lower = np.zeros(bin_count, dtype=np.float64)
        diagonal = np.ones(bin_count, dtype=np.float64)
        upper = np.zeros(bin_count, dtype=np.float64)
        if face_velocity[0] < 0.0:
            diagonal[0] -= dt_s * face_velocity[0] / widths[0]
        internal_face_velocity = face_velocity[1:-1]
        positive_faces = np.flatnonzero(internal_face_velocity >= 0.0)
        negative_faces = np.flatnonzero(internal_face_velocity < 0.0)
        if positive_faces.size:
            values = internal_face_velocity[positive_faces]
            diagonal[positive_faces] += dt_s * values / widths[positive_faces]
            lower[positive_faces + 1] -= dt_s * values / widths[positive_faces + 1]
        if negative_faces.size:
            values = internal_face_velocity[negative_faces]
            diagonal[negative_faces + 1] -= dt_s * values / widths[negative_faces + 1]
            upper[negative_faces] += dt_s * values / widths[negative_faces]
        if face_velocity[-1] > 0.0:
            diagonal[-1] += dt_s * face_velocity[-1] / widths[-1]

        upper_reduced = np.zeros(bin_count, dtype=np.float64)
        rhs_reduced = np.empty(bin_count, dtype=np.float64)
        pivot = diagonal[0]
        if pivot <= 0.0 or not np.isfinite(pivot):
            raise SolverStateError("implicit finite-volume solve has an invalid first pivot")
        upper_reduced[0] = upper[0] / pivot
        rhs_reduced[0] = density[0] / pivot
        for index in range(1, bin_count):
            pivot = diagonal[index] - lower[index] * upper_reduced[index - 1]
            if pivot <= 0.0 or not np.isfinite(pivot):
                raise SolverStateError(
                    f"implicit finite-volume solve has an invalid pivot at bin {index}"
                )
            if index < bin_count - 1:
                upper_reduced[index] = upper[index] / pivot
            rhs_reduced[index] = (
                density[index] - lower[index] * rhs_reduced[index - 1]
            ) / pivot
        updated = np.empty_like(density)
        updated[-1] = rhs_reduced[-1]
        for index in range(bin_count - 2, -1, -1):
            updated[index] = rhs_reduced[index] - upper_reduced[index] * updated[index + 1]
        roundoff_floor = -1.0e-280
        if np.any(updated < roundoff_floor) or not np.all(np.isfinite(updated)):
            raise SolverStateError(
                f"{population.parameters.name} finite-volume update violated positivity or finiteness"
            )
        # Only eliminate subnormal round-off; this is not a mass-correction path.
        roundoff_mask = (updated < 0.0) & (updated >= roundoff_floor)
        roundoff_zeroed = int(np.count_nonzero(roundoff_mask))
        updated[roundoff_mask] = 0.0
        faces = self._upwind_face_fluxes(
            updated, velocity_m_s, face_velocity_m_s=face_velocity
        )
        lower_dissolution_flux = boundary_number_flux_diagnostic(face_velocity[0], updated[0])
        rmax_outflow_flux = max(faces[-1], 0.0)
        depletion = np.divide(
            np.maximum(density - updated, 0.0),
            density,
            out=np.zeros_like(density),
            where=density > 0.0,
        )
        population.number_density_per_m4[:] = updated
        return lower_dissolution_flux, rmax_outflow_flux, float(np.max(depletion)), roundoff_zeroed

    def _inject_nucleation(self, population: Population, result: NucleationResult, dt_s: float) -> None:
        """Insert a source into one finite-volume bin; mass is drawn via the ledger."""

        if result.rate_m3_s == 0.0:
            return
        if result.rate_m3_s < 0.0 or result.radius_m <= 0.0:
            raise SolverStateError("Nucleation result must have non-negative rate and positive radius")
        population.add_number_at_radius(result.radius_m, result.rate_m3_s * dt_s)

    def advance_one(self, maximum_dt_s: float | None = None) -> StepDiagnostics:
        """Advance exactly one adaptive, conservative KWN macro-step."""

        velocities = self.growth_rates()
        face_velocities = {
            name: self.face_velocities(population, velocities[name])
            for name, population in self.populations.items()
        }
        dt_s, cfl = self._choose_dt(
            velocities, maximum_dt_s, face_velocities=face_velocities
        )
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
        lower_flux = 0.0
        upper_flux = 0.0
        beta_lower_flux = 0.0
        positivity_utilization = 0.0
        roundoff_zeroed_bin_count = 0
        for name in ("g", "beta"):
            lower, upper, utilization, zeroed = self._advect_population(
                self.populations[name],
                velocities[name],
                dt_s,
                face_velocity_m_s=face_velocities[name],
            )
            lower_flux += lower
            upper_flux += upper
            if name == "beta":
                beta_lower_flux = lower
            positivity_utilization = max(positivity_utilization, utilization)
            roundoff_zeroed_bin_count += zeroed
        self.roundoff_zeroed_bin_count += roundoff_zeroed_bin_count
        self._inject_nucleation(self.populations["g"], g_source, dt_s)
        self._inject_nucleation(self.populations["beta"], beta_source, dt_s)
        state_changed = (
            any(np.any(value != 0.0) for value in velocities.values())
            or g_source.rate_m3_s != 0.0
            or beta_source.rate_m3_s != 0.0
        )
        # In the exact D=J=0 invariant case retain the input float bit pattern.
        # This is not a mass correction: it avoids injecting a round-off pulse
        # through an otherwise identity operation.
        if state_changed:
            self.matrix_xb = self.ledger.recover_matrix_xb(self.population_list())
        self.time_s += dt_s
        self.step += 1
        inventory = self.ledger.snapshot(matrix_xb=self.matrix_xb, populations=self.population_list())
        beta = self.populations["beta"]
        beta_boundary_radius = boundary_radius(beta.grid)
        beta_boundary_inventory = particle_inventory_at_radius(
            beta_boundary_radius,
            x_b=beta.parameters.x_b,
            molar_volume_m3_mol=beta.parameters.molar_volume_m3_mol,
        )
        beta_boundary_flux = boundary_inventory_diagnostic(
            beta_lower_flux, beta_boundary_inventory
        )
        raw_rate, face_index, cell_index, face_velocity, cell_width = self._last_timestep_telemetry[
            "overall"
        ]
        beta_raw_rate = float(self._last_timestep_telemetry["beta"][0])
        beta_active = self._last_timestep_telemetry["active"].get("beta")
        if beta_active is None:
            raise SolverStateError("accepted step is missing beta active-CFL telemetry")
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
            beta_rmin_number_flux_m3_s=beta_lower_flux,
            beta_rmin_volume_flux_s=beta_boundary_flux.beta_volume_flux_out_s,
            beta_rmin_mol_b_flux_mol_m3_s=beta_boundary_flux.b_mol_flux_out_mol_m3_s,
            beta_boundary_radius_m=beta_boundary_radius,
            beta_boundary_growth_velocity_m_s=float(face_velocities["beta"][0]),
            radius_courant_max=raw_rate * dt_s,
            radius_courant_beta_max=beta_raw_rate * dt_s,
            radius_courant_face_index=int(face_index),
            radius_courant_cell_index=int(cell_index),
            radius_courant_face_velocity_m_s=float(face_velocity),
            radius_courant_cell_width_m=float(cell_width),
            timestep_limiter=str(self._last_timestep_telemetry["limiter"]),
            active_population_courant=beta_active.population_active_rate_s_inv * dt_s,
            active_m0_courant=beta_active.m0_rate_s_inv * dt_s,
            active_m3_courant=beta_active.m3_rate_s_inv * dt_s,
            lower_tail_active_courant=beta_active.lower_tail_rate_s_inv * dt_s,
            boundary_candidate_courant=beta_active.boundary_candidate_rate_s_inv * dt_s,
            boundary_active_courant=beta_active.boundary_active_rate_s_inv * dt_s,
            boundary_face_is_active=beta_active.boundary_face_is_active,
            active_m0_support_i_lo=beta_active.m0_support_i_lo,
            active_m0_support_i_hi=beta_active.m0_support_i_hi,
            active_m3_support_i_lo=beta_active.m3_support_i_lo,
            active_m3_support_i_hi=beta_active.m3_support_i_hi,
            lower_tail_i_lo=beta_active.lower_tail_i_lo,
            lower_tail_i_hi=beta_active.lower_tail_i_hi,
            lower_tail_m0_fraction=beta_active.lower_tail_m0_fraction,
        )
        self.history.append(diagnostic)
        return diagnostic

    def run_steps(self, steps: int) -> List[StepDiagnostics]:
        """Advance a deterministic number of accepted macro-steps."""

        if steps < 0:
            raise ValueError("steps must be non-negative")
        return [self.advance_one() for _ in range(int(steps))]

    def run_to_time(self, end_time_s: float) -> List[StepDiagnostics]:
        """Advance until an exact requested physical time in seconds."""

        if end_time_s < self.time_s:
            raise ValueError("end_time_s precedes the current solver time")
        accepted: List[StepDiagnostics] = []
        while self.time_s < end_time_s:
            remaining = end_time_s - self.time_s
            accepted.append(self.advance_one(maximum_dt_s=remaining))
        return accepted

    def state_arrays(self) -> Dict[str, NDArray[np.float64]]:
        """Return independent arrays sufficient to compare deterministic state."""

        return {
            "g_number_density_per_m4": self.populations["g"].number_density_per_m4.copy(),
            "beta_number_density_per_m4": self.populations["beta"].number_density_per_m4.copy(),
            "matrix_xb": np.asarray([self.matrix_xb], dtype=np.float64),
            "time_s": np.asarray([self.time_s], dtype=np.float64),
            "step": np.asarray([self.step], dtype=np.int64),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        """Write a restartable NumPy checkpoint with config provenance."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "solver_version": self.solver_version,
            "source_config_hash": self.config.source_config_hash,
            "temperature_K": self.config.temperature_k,
            "total_b_mol_m3": self.ledger.total_b_mol_m3,
            "validation_contract_hash": self.contract_hash,
        }
        arrays = self.state_arrays()
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
        np.savez(output, **arrays)

    @classmethod
    def load_checkpoint(cls, *, config: SolverConfig, path: str | Path) -> "KWNSolver":
        """Restore an exact solver state after verifying configuration provenance."""

        checkpoint_path = Path(path)
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("solver_version") != cls.solver_version:
                raise SolverStateError("Checkpoint solver version differs from this KWN implementation")
            if metadata.get("source_config_hash") != config.source_config_hash:
                raise SolverStateError("Checkpoint config hash differs from the requested restart config")
            solver = cls(config)
            if metadata.get("validation_contract_hash") != solver.contract_hash:
                raise SolverStateError(
                    "Checkpoint validation contract hash differs from the requested restart config"
                )
            if not np.isclose(float(metadata["total_b_mol_m3"]), solver.ledger.total_b_mol_m3, rtol=0.0, atol=0.0):
                raise SolverStateError("Checkpoint total inventory differs from configuration")
            solver.populations["g"].number_density_per_m4[:] = archive["g_number_density_per_m4"]
            solver.populations["beta"].number_density_per_m4[:] = archive["beta_number_density_per_m4"]
            solver.matrix_xb = float(archive["matrix_xb"][0])
            solver.time_s = float(archive["time_s"][0])
            solver.step = int(archive["step"][0])
            solver.ledger.snapshot(matrix_xb=solver.matrix_xb, populations=solver.population_list())
            return solver
