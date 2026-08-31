"""Conservative finite-volume effective KWN solver with an exact inventory ledger."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import ConfigDocument, ConfigurationError, config_hash, require_mapping, require_number
from .growth import growth_rate_m_s
from .ledger import InventoryError, InventoryLedger, InventorySnapshot
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
    matrix_xb: float
    inventory: InventorySnapshot
    gp_nucleation_rate_m3_s: float
    beta_nucleation_rate_m3_s: float
    rmin_dissolution_flux_m3_s: float
    rmax_outflow_flux_m3_s: float


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
    *, parameters: PopulationParameters, grid: RadiusGrid, definition: Mapping[str, Any]
) -> Population:
    """Construct an initial PSD from an explicit config-owned definition."""

    kind = str(definition.get("kind", "empty"))
    population = Population.empty(parameters, grid)
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
        if config.min_dt_s > config.max_dt_s:
            raise ConfigurationError("min_dt_s cannot exceed max_dt_s")
        self.config = config
        self.populations: Dict[str, Population] = {
            parameters.name: _build_initial_population(
                parameters=parameters,
                grid=config.grid,
                definition=config.initial_population[parameters.name],
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
        self.history: List[StepDiagnostics] = []

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

    def _choose_dt(self, velocities: Mapping[str, NDArray[np.float64]], maximum_s: float | None) -> Tuple[float, float]:
        """Choose a CFL-limited adaptive macro-step and report its actual CFL."""

        cfl_rates: List[float] = []
        for name, velocity in velocities.items():
            population = self.populations[name]
            cell_number = population.number_density_per_m4 * population.grid.widths_m
            total_number = float(np.sum(cell_number))
            if total_number == 0.0:
                continue
            # Empty/round-off cells do not carry a flux and must not force an
            # artificial micro-step merely because Gibbs--Thomson velocity is
            # large at an unused Rmin.  The threshold is relative to the
            # represented population and has no mass-correction role.
            active = cell_number > max(total_number * 1.0e-30, 1.0e-300)
            # Include immediate receivers/donors at each active face.  This
            # retains the positivity guarantee while avoiding unused Rmin/Rmax
            # velocities imposing a CFL restriction on an empty population.
            active[:-1] |= active[1:]
            active[1:] |= active[:-1]
            if np.any(active):
                cfl_rates.append(
                    float(np.max(np.abs(velocity[active]) / population.grid.widths_m[active]))
                )
        max_rate = max(cfl_rates, default=0.0)
        cap = self.config.max_dt_s if maximum_s is None else min(self.config.max_dt_s, maximum_s)
        if max_rate == 0.0:
            dt = cap
            return dt, 0.0
        dt = min(cap, self.config.size_cfl / max_rate)
        if dt < self.config.min_dt_s:
            raise SolverStateError(
                f"CFL-limited step {dt:.3e} s is below configured min_dt_s={self.config.min_dt_s:.3e} s"
            )
        cfl = max_rate * dt
        return dt, cfl

    def _advect_population(
        self, population: Population, velocity_m_s: NDArray[np.float64], dt_s: float
    ) -> Tuple[float, float]:
        """Apply one first-order upwind finite-volume update.

        Returns lower-boundary dissolution and Rmax outflow number fluxes in
        m^-3 s^-1.  Rmax outflow is rejected before material can disappear.
        """

        density = population.number_density_per_m4
        widths = population.grid.widths_m
        faces = np.empty(population.grid.bins + 1, dtype=np.float64)
        faces[0] = velocity_m_s[0] * density[0] if velocity_m_s[0] < 0.0 else 0.0
        internal_velocity = 0.5 * (velocity_m_s[:-1] + velocity_m_s[1:])
        faces[1:-1] = np.where(
            internal_velocity >= 0.0,
            internal_velocity * density[:-1],
            internal_velocity * density[1:],
        )
        faces[-1] = velocity_m_s[-1] * density[-1] if velocity_m_s[-1] > 0.0 else 0.0
        lower_dissolution_flux = max(-faces[0], 0.0)
        rmax_outflow_flux = max(faces[-1], 0.0)
        existing_number = max(population.number_density_m3(), 1.0e-300)
        relative_outflow = rmax_outflow_flux * dt_s / existing_number
        if relative_outflow > self.config.rmax_outflow_relative_tolerance:
            raise RadiusGridOverflowError(
                f"{population.parameters.name} would lose {relative_outflow:.3e} of its number density "
                "through Rmax. Expand the configured radius grid; material was not discarded."
            )
        updated = density - dt_s * (faces[1:] - faces[:-1]) / widths
        roundoff_floor = -1.0e-280
        if np.any(updated < roundoff_floor) or not np.all(np.isfinite(updated)):
            raise SolverStateError(
                f"{population.parameters.name} finite-volume update violated positivity or finiteness"
            )
        # Only eliminate subnormal round-off; this is not a mass-correction path.
        updated[(updated < 0.0) & (updated >= roundoff_floor)] = 0.0
        population.number_density_per_m4[:] = updated
        return lower_dissolution_flux, rmax_outflow_flux

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
        dt_s, cfl = self._choose_dt(velocities, maximum_dt_s)
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
        for name in ("g", "beta"):
            lower, upper = self._advect_population(self.populations[name], velocities[name], dt_s)
            lower_flux += lower
            upper_flux += upper
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
        diagnostic = StepDiagnostics(
            step=self.step,
            time_s=self.time_s,
            dt_s=dt_s,
            size_cfl=cfl,
            matrix_xb=self.matrix_xb,
            inventory=inventory,
            gp_nucleation_rate_m3_s=g_source.rate_m3_s,
            beta_nucleation_rate_m3_s=beta_source.rate_m3_s,
            rmin_dissolution_flux_m3_s=lower_flux,
            rmax_outflow_flux_m3_s=upper_flux,
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
