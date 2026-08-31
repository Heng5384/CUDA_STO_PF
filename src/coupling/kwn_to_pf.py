"""One-way KWN snapshot adapter for the currently qualified PF contract.

This adapter intentionally produces a *plan*, not a mutated PF input file.  A
KWN package can carry four conserved inventory buckets, while the current
production-qualified PF initialization can represent matrix composition and
resolved beta geometry only.  Non-zero GP or sub-grid beta inventory therefore
remains explicit in the ledger and makes the plan fail closed as
``PARTIAL_PF_STATE_NOT_CLOSED``.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .kwn_pf_schema import (
    DEFAULT_MASS_RELATIVE_TOLERANCE,
    HandoffPackage,
    SchemaValidationError,
    ValidationReport,
    make_handoff_package,
    validate_handoff_package,
)


PARTIAL_PF_STATE_NOT_CLOSED = "PARTIAL_PF_STATE_NOT_CLOSED"
READY_FOR_PF_HANDOFF = "READY_FOR_PF_HANDOFF"
FAIL_PF_COUPLING_CONTRACT = "FAIL_PF_COUPLING_CONTRACT"


class PFStateNotClosedError(RuntimeError):
    """Raised if a caller attempts to materialize a plan with unmapped solute."""


class KwnHandoffExportError(ValueError):
    """Raised when a KWN state cannot be exactly classified for a snapshot."""


@dataclass(frozen=True)
class PfCapabilities:
    """Declared PF initialization capabilities, not merely dormant code paths.

    ``technical_gp_eta_field_available`` is informational.  It does not turn a
    legacy/optional eta field into a current-contract qualified KWN inventory
    target.  That distinction prevents silently converting a KWN GP population
    into a different PF physics path.
    """

    supports_matrix_composition: bool = True
    supports_resolved_beta_geometry: bool = True
    supports_gp_inventory: bool = False
    supports_beta_subgrid_inventory: bool = False
    technical_gp_eta_field_available: bool = True
    current_contract_gp_inventory_eligible: bool = False
    description: str = (
        "Current qualified PF initialization is matrix plus resolved beta. "
        "Its optional GP eta storage path is not a qualified KWN handoff state, "
        "and it has no independent sub-grid beta reservoir."
    )


CURRENT_PF_CAPABILITIES = PfCapabilities()


def assess_current_pf_state() -> Dict[str, Dict[str, Any]]:
    """State the currently qualified PF handoff contract without probing PF code.

    The result distinguishes a technically present legacy eta field from a
    state variable that the active, qualified two-phase PF workflow may accept
    as a KWN inventory.  It is intentionally a contract declaration rather
    than a heuristic inference from optional command-line flags.
    """

    return {
        "matrix_xB_alpha": {
            "runtime_support": "AVAILABLE",
            "primary_handoff_eligible": True,
            "action": "MAP_TARGET_BASELINE_WITH_QUALIFIED_PROFILE_MATERIALIZER",
        },
        "resolved_beta_phi": {
            "runtime_support": "AVAILABLE",
            "primary_handoff_eligible": True,
            "action": "PRESERVE_EXISTING_GEOMETRY",
        },
        "gp_inventory": {
            "runtime_support": "OPTIONAL_ETA_FIELD_ONLY",
            "primary_handoff_eligible": False,
            "action": "RETAIN_IN_PACKAGE_ONLY",
            "reason": "GP_ZONE_PATH_NOT_CURRENTLY_QUALIFIED",
        },
        "beta_subgrid_inventory": {
            "runtime_support": "NO_PERSISTENT_PF_STATE",
            "primary_handoff_eligible": False,
            "action": "RETAIN_IN_PACKAGE_ONLY",
        },
    }


def _volume_and_inventory(
    density_per_m4: np.ndarray,
    widths_m: np.ndarray,
    radii_m: np.ndarray,
    x_b: float,
    molar_volume_m3_per_mol: float,
) -> tuple:
    """Return a discrete KWN volume fraction and B concentration in mol m^-3."""

    sphere_volume_m3 = (4.0 * math.pi / 3.0) * radii_m**3
    volume_fraction = float(np.sum(density_per_m4 * widths_m * sphere_volume_m3))
    return volume_fraction, volume_fraction * float(x_b) / float(molar_volume_m3_per_mol)


def _split_index_at_grid_edge(edges_m: np.ndarray, handoff_radius_m: float) -> int:
    """Return the first resolved-bin index, refusing an ambiguous split cell."""

    if handoff_radius_m <= float(edges_m[0]):
        return 0
    if handoff_radius_m >= float(edges_m[-1]):
        return int(edges_m.size - 1)
    matched = np.flatnonzero(
        np.isclose(edges_m, handoff_radius_m, rtol=1.0e-12, atol=0.0)
    )
    if matched.size != 1:
        raise KwnHandoffExportError(
            "beta_handoff_radius_m must coincide with a KWN radius-bin edge; "
            "the MVP does not silently split a finite-volume cell"
        )
    return int(matched[0])


def build_handoff_from_kwn_solver(
    solver: Any,
    *,
    source_git_commit: str,
    pf_box_lengths_m: Sequence[float],
    observation_dataset_role: str,
    beta_handoff_radius_m: float,
    assumptions: Sequence[str],
    resolved_shape_orientation_metadata: Optional[Mapping[str, Any]] = None,
    source_binary_hash: Optional[str] = None,
    source_fixture_hash: Optional[str] = None,
    source_analysis_hash: Optional[str] = None,
) -> HandoffPackage:
    """Export one conservative KWN state as a versioned handoff package.

    The beta radius cutoff is a numerical PF-resolution classification only:
    beta bins below it remain in the sub-grid bucket and bins at/above it are
    converted to expected PF-box counts.  It is never interpreted as a direct
    GP-to-beta conversion event.  To retain finite-volume conservation exactly,
    v1 requires the cutoff to be a KWN bin edge.
    """

    if not math.isfinite(beta_handoff_radius_m) or beta_handoff_radius_m <= 0.0:
        raise KwnHandoffExportError("beta_handoff_radius_m must be finite and positive")
    try:
        gp = solver.population("g")
        beta = solver.population("beta")
        config = solver.config
        ledger = solver.ledger
        matrix_xb = float(solver.matrix_xb)
        handoff_time_s = float(solver.time_s)
    except (AttributeError, KeyError) as error:
        raise KwnHandoffExportError(
            "solver must expose KWN populations g/beta, config, ledger, matrix_xb, and time_s"
        ) from error

    beta_edges = np.asarray(beta.grid.edges_m, dtype=float)
    beta_density = np.asarray(beta.number_density_per_m4, dtype=float)
    beta_widths = np.asarray(beta.grid.widths_m, dtype=float)
    beta_radii = np.asarray(beta.grid.centres_m, dtype=float)
    gp_edges = np.asarray(gp.grid.edges_m, dtype=float)
    gp_density = np.asarray(gp.number_density_per_m4, dtype=float)
    split_index = _split_index_at_grid_edge(beta_edges, float(beta_handoff_radius_m))
    resolved_mask = np.arange(beta_density.size) >= split_index
    subgrid_density = beta_density.copy()
    subgrid_density[resolved_mask] = 0.0
    resolved_density = beta_density.copy()
    resolved_density[~resolved_mask] = 0.0

    subgrid_volume_fraction, subgrid_inventory = _volume_and_inventory(
        subgrid_density,
        beta_widths,
        beta_radii,
        beta.parameters.x_b,
        beta.parameters.molar_volume_m3_mol,
    )
    resolved_volume_fraction, resolved_inventory = _volume_and_inventory(
        resolved_density,
        beta_widths,
        beta_radii,
        beta.parameters.x_b,
        beta.parameters.molar_volume_m3_mol,
    )
    gp_volume_fraction, gp_inventory = _volume_and_inventory(
        gp_density,
        np.asarray(gp.grid.widths_m, dtype=float),
        np.asarray(gp.grid.centres_m, dtype=float),
        gp.parameters.x_b,
        gp.parameters.molar_volume_m3_mol,
    )
    matrix_fraction = 1.0 - gp_volume_fraction - subgrid_volume_fraction - resolved_volume_fraction
    if matrix_fraction <= 0.0:
        raise KwnHandoffExportError("KWN populations leave no positive matrix volume for handoff")
    matrix_inventory = matrix_fraction * matrix_xb / float(config.matrix_molar_volume_m3_mol)
    total_inventory = float(ledger.total_b_mol_m3)
    residual = total_inventory - (
        matrix_inventory + gp_inventory + subgrid_inventory + resolved_inventory
    )

    box_lengths = np.asarray(pf_box_lengths_m, dtype=float)
    if box_lengths.shape != (3,) or np.any(~np.isfinite(box_lengths)) or np.any(box_lengths <= 0.0):
        raise KwnHandoffExportError("pf_box_lengths_m must contain three finite positive values")
    expected_count = resolved_density * beta_widths * float(np.prod(box_lengths))
    source: Dict[str, Any] = {
        "git_commit": source_git_commit,
        "config_hash": str(config.source_config_hash),
        "kwn_backend": str(getattr(solver, "solver_version", "internal_kwn_finite_volume_v1")),
        "kwn_backend_version": "v1",
    }
    for key, value in (
        ("binary_hash", source_binary_hash),
        ("fixture_hash", source_fixture_hash),
        ("analysis_hash", source_analysis_hash),
    ):
        if value:
            source[key] = value
    assumptions_list = list(assumptions)
    assumptions_list.extend(
        [
            "Resolved beta is classified from KWN beta bins at or above the numerical handoff radius; this is not direct GP-to-beta conversion.",
            "GP and beta-subgrid inventories remain separate ledger buckets during PF handoff.",
        ]
    )
    resolved_population: Dict[str, Any] = {
        "xB_beta": float(beta.parameters.x_b),
        "Vm_m3_per_mol": float(beta.parameters.molar_volume_m3_mol),
        "volume_fraction": resolved_volume_fraction,
        "B_inventory": resolved_inventory,
    }
    if resolved_shape_orientation_metadata is not None:
        resolved_population["shape_orientation_metadata"] = dict(
            resolved_shape_orientation_metadata
        )
    metadata: Dict[str, Any] = {
        "schema_version": "kwn_pf_handoff_v1",
        "source": source,
        "temperature_K": float(config.temperature_k),
        "handoff_time_s": handoff_time_s,
        "pf_box": {"lengths_m": box_lengths.tolist(), "periodic": True},
        "unit_system": "SI",
        "inventory_basis": "mol_B_per_m3",
        "total_pseudo_binary_inventory": total_inventory,
        "assumptions": assumptions_list,
        "observation_dataset_role": observation_dataset_role,
        "matrix": {
            "xB_alpha": matrix_xb,
            "Vm_m3_per_mol": float(config.matrix_molar_volume_m3_mol),
            "B_inventory": matrix_inventory,
        },
        "gp_population": {
            "xB_g": float(gp.parameters.x_b),
            "Vm_m3_per_mol": float(gp.parameters.molar_volume_m3_mol),
            "volume_fraction": gp_volume_fraction,
            "B_inventory": gp_inventory,
        },
        "beta_subgrid_population": {
            "xB_beta": float(beta.parameters.x_b),
            "Vm_m3_per_mol": float(beta.parameters.molar_volume_m3_mol),
            "volume_fraction": subgrid_volume_fraction,
            "B_inventory": subgrid_inventory,
        },
        "beta_resolved_population": resolved_population,
        "ledger": {
            "C_B_total": total_inventory,
            "C_B_matrix": matrix_inventory,
            "C_B_GP": gp_inventory,
            "C_B_beta_subgrid": subgrid_inventory,
            "C_B_beta_resolved": resolved_inventory,
            "residual": residual,
        },
        "extensions": {"beta_handoff_radius_m": float(beta_handoff_radius_m)},
    }
    arrays = {
        "gp_radius_bin_edges_m": gp_edges,
        "gp_number_density_per_m4": gp_density,
        "beta_subgrid_radius_bin_edges_m": beta_edges,
        "beta_subgrid_number_density_per_m4": subgrid_density,
        "beta_resolved_radius_bin_edges_m": beta_edges,
        "beta_resolved_expected_count": expected_count,
    }
    return make_handoff_package(metadata, arrays)


@dataclass(frozen=True)
class PfInitializationPlan:
    """A serializable plan for a future PF initializer or smoke-test wrapper."""

    status: str
    mode: str
    source_validation: ValidationReport
    capabilities: PfCapabilities
    materialized_state: Dict[str, Any]
    source_ledger: Dict[str, float]
    unmapped_inventory: Dict[str, float]
    missing_pf_state_variables: List[str]
    notes: List[str]

    @property
    def pf_state_closed(self) -> bool:
        """Whether every non-zero source inventory has a declared PF target."""

        return self.status == READY_FOR_PF_HANDOFF

    def require_pf_state_closed(self) -> None:
        """Prevent a caller from treating an explicit partial state as usable PF input."""

        if not self.pf_state_closed:
            details = "; ".join(self.missing_pf_state_variables) or self.status
            raise PFStateNotClosedError(
                f"PF state is not closed ({self.status}): {details}"
            )

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible representation without mutating source state."""

        return {
            "status": self.status,
            "mode": self.mode,
            "pf_state_closed": self.pf_state_closed,
            "pf_raw_initialization_emitted": False,
            "pf_raw_initialization_allowed": self.pf_state_closed,
            "capabilities": {
                "supports_matrix_composition": self.capabilities.supports_matrix_composition,
                "supports_resolved_beta_geometry": self.capabilities.supports_resolved_beta_geometry,
                "supports_gp_inventory": self.capabilities.supports_gp_inventory,
                "supports_beta_subgrid_inventory": self.capabilities.supports_beta_subgrid_inventory,
                "technical_gp_eta_field_available": self.capabilities.technical_gp_eta_field_available,
                "current_contract_gp_inventory_eligible": self.capabilities.current_contract_gp_inventory_eligible,
                "description": self.capabilities.description,
            },
            "source_validation": self.source_validation.as_dict(),
            "materialized_state": copy.deepcopy(self.materialized_state),
            "source_ledger": dict(self.source_ledger),
            "unmapped_inventory": dict(self.unmapped_inventory),
            "missing_pf_state_variables": list(self.missing_pf_state_variables),
            "notes": list(self.notes),
        }


def _ledger_from_metadata(package: HandoffPackage) -> Dict[str, float]:
    ledger = package.metadata["ledger"]
    return {
        "C_B_total": float(ledger["C_B_total"]),
        "C_B_matrix": float(ledger["C_B_matrix"]),
        "C_B_GP": float(ledger["C_B_GP"]),
        "C_B_beta_subgrid": float(ledger["C_B_beta_subgrid"]),
        "C_B_beta_resolved": float(ledger["C_B_beta_resolved"]),
        "residual": float(ledger["residual"]),
    }


def _is_material(value: float, total: float, tolerance: float) -> bool:
    return value > tolerance * max(abs(total), 1.0e-300)


def _resolved_payload(package: HandoffPackage, mode: str) -> Dict[str, Any]:
    population = package.metadata["beta_resolved_population"]
    payload: Dict[str, Any] = {
        "mode": mode,
        "B_inventory": float(population["B_inventory"]),
        "xB_beta": float(population["xB_beta"]),
        "Vm_m3_per_mol": float(population["Vm_m3_per_mol"]),
        "volume_fraction": float(population["volume_fraction"]),
        "radius_bin_edges_m": package.arrays["beta_resolved_radius_bin_edges_m"].tolist(),
        "expected_count": package.arrays["beta_resolved_expected_count"].tolist(),
    }
    if "shape_orientation_metadata" in population:
        payload["shape_orientation_metadata"] = copy.deepcopy(
            population["shape_orientation_metadata"]
        )
    if mode == "sampled_geometry":
        if "beta_resolved_sampled_radii_m" not in package.arrays:
            raise SchemaValidationError(
                "sampled_geometry mode requires beta_resolved_sampled_radii_m"
            )
        if "beta_resolved_centers_m" not in package.arrays:
            raise SchemaValidationError(
                "sampled_geometry mode requires beta_resolved_centers_m"
            )
        payload["sampled_radii_m"] = package.arrays["beta_resolved_sampled_radii_m"].tolist()
        payload["centers_m"] = package.arrays["beta_resolved_centers_m"].tolist()
    else:
        payload[
            "geometry_instruction"
        ] = "Preserve the authoritative PF resolved-beta geometry; do not resample positions."
    return payload


def adapt_kwn_to_pf(
    package: HandoffPackage,
    mode: str = "existing_geometry",
    capabilities: PfCapabilities = CURRENT_PF_CAPABILITIES,
    mass_relative_tolerance: float = DEFAULT_MASS_RELATIVE_TOLERANCE,
) -> PfInitializationPlan:
    """Build a no-loss KWN-to-PF handoff plan.

    The function never adds GP or beta-subgrid inventory to the matrix bucket.
    With :data:`CURRENT_PF_CAPABILITIES`, any material amount in either bucket
    returns :data:`PARTIAL_PF_STATE_NOT_CLOSED` and is retained verbatim in
    ``unmapped_inventory``.
    """

    if mode not in ("existing_geometry", "sampled_geometry"):
        raise ValueError("mode must be 'existing_geometry' or 'sampled_geometry'")
    if not isinstance(capabilities, PfCapabilities):
        raise TypeError("capabilities must be a PfCapabilities instance")
    validation = validate_handoff_package(package, mass_relative_tolerance)
    ledger = _ledger_from_metadata(package)
    total = ledger["C_B_total"]
    unmapped: Dict[str, float] = {}
    missing: List[str] = []
    notes: List[str] = [
        "This is an offline snapshot plan; it does not implement concurrent KWN-PF coupling."
    ]
    if mode == "existing_geometry":
        notes.append(
            "The supplied matrix xB_alpha is a target baseline, not permission to overwrite the existing diffuse profile or delta_C_relaxation field."
        )

    if not capabilities.supports_matrix_composition and _is_material(
        ledger["C_B_matrix"], total, mass_relative_tolerance
    ):
        missing.append("matrix composition initialization state")
    if not capabilities.supports_resolved_beta_geometry and _is_material(
        ledger["C_B_beta_resolved"], total, mass_relative_tolerance
    ):
        missing.append("resolved-beta geometry initialization state")

    if _is_material(ledger["C_B_GP"], total, mass_relative_tolerance):
        if not capabilities.supports_gp_inventory:
            unmapped["C_B_GP"] = ledger["C_B_GP"]
            missing.append(
                "independent GP inventory state (current qualified PF eta path is not a KWN handoff target)"
            )
    if _is_material(ledger["C_B_beta_subgrid"], total, mass_relative_tolerance):
        if not capabilities.supports_beta_subgrid_inventory:
            unmapped["C_B_beta_subgrid"] = ledger["C_B_beta_subgrid"]
            missing.append("independent sub-grid beta inventory state")

    materialized_state: Dict[str, Any] = {
        "matrix": {
            "xB_alpha": float(package.metadata["matrix"]["xB_alpha"]),
            "Vm_m3_per_mol": float(package.metadata["matrix"]["Vm_m3_per_mol"]),
            "B_inventory": ledger["C_B_matrix"],
        },
        "resolved_beta": _resolved_payload(package, mode),
    }
    if capabilities.supports_gp_inventory:
        materialized_state["gp_population"] = {
            "B_inventory": ledger["C_B_GP"],
            "xB_g": float(package.metadata["gp_population"]["xB_g"]),
            "Vm_m3_per_mol": float(package.metadata["gp_population"]["Vm_m3_per_mol"]),
            "volume_fraction": float(package.metadata["gp_population"]["volume_fraction"]),
            "radius_bin_edges_m": package.arrays["gp_radius_bin_edges_m"].tolist(),
            "number_density_per_m4": package.arrays["gp_number_density_per_m4"].tolist(),
        }
    if capabilities.supports_beta_subgrid_inventory:
        materialized_state["beta_subgrid_population"] = {
            "B_inventory": ledger["C_B_beta_subgrid"],
            "xB_beta": float(package.metadata["beta_subgrid_population"]["xB_beta"]),
            "Vm_m3_per_mol": float(package.metadata["beta_subgrid_population"]["Vm_m3_per_mol"]),
            "volume_fraction": float(package.metadata["beta_subgrid_population"]["volume_fraction"]),
            "radius_bin_edges_m": package.arrays["beta_subgrid_radius_bin_edges_m"].tolist(),
            "number_density_per_m4": package.arrays[
                "beta_subgrid_number_density_per_m4"
            ].tolist(),
        }

    if missing and unmapped:
        status = PARTIAL_PF_STATE_NOT_CLOSED
        notes.append(
            "GP and/or beta-subgrid inventory is preserved in the handoff ledger, not added to xB_alpha."
        )
    elif missing:
        status = FAIL_PF_COUPLING_CONTRACT
        notes.append("The target PF lacks a required matrix or resolved-beta initialization state.")
    else:
        status = READY_FOR_PF_HANDOFF
        notes.append("Every material inventory bucket has a declared PF target.")

    return PfInitializationPlan(
        status=status,
        mode=mode,
        source_validation=validation,
        capabilities=capabilities,
        materialized_state=materialized_state,
        source_ledger=ledger,
        unmapped_inventory=unmapped,
        missing_pf_state_variables=missing,
        notes=notes,
    )


build_pf_initialization_plan = adapt_kwn_to_pf
