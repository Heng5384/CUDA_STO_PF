"""Convert an already extracted PF resolved-beta snapshot into the v1 schema.

PF snapshot import is intentionally conservative: the current qualified PF
state has a matrix field and resolved-beta geometry, but no persistent,
independent KWN GP or sub-grid beta bucket.  The importer records those two
buckets as exactly zero rather than inventing them from matrix solute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .kwn_pf_schema import HandoffPackage, make_handoff_package


@dataclass(frozen=True)
class PfResolvedSnapshot:
    """PF observables already extracted from one resolved-beta snapshot.

    All inventories use the same ``mol_B_per_m3`` basis required by the
    handoff schema.  This adapter does not derive inventories from raw fields;
    that conversion belongs to the source PF extraction/audit contract.
    """

    source_git_commit: str
    config_hash: str
    temperature_K: float
    handoff_time_s: float
    pf_box_lengths_m: Sequence[float]
    matrix_xB_alpha: float
    matrix_B_inventory: float
    matrix_Vm_m3_per_mol: float
    resolved_xB_beta: float
    resolved_Vm_m3_per_mol: float
    resolved_volume_fraction: float
    resolved_B_inventory: float
    resolved_radius_bin_edges_m: np.ndarray
    resolved_expected_count: np.ndarray
    gp_xB_g: float
    gp_Vm_m3_per_mol: float
    beta_subgrid_Vm_m3_per_mol: float
    gp_radius_bin_edges_m: np.ndarray
    beta_subgrid_radius_bin_edges_m: np.ndarray
    observation_dataset_role: str
    assumptions: Sequence[str]
    resolved_sampled_radii_m: Optional[np.ndarray] = None
    resolved_centers_m: Optional[np.ndarray] = None
    shape_orientation_metadata: Optional[Mapping[str, Any]] = None
    source_binary_hash: Optional[str] = None
    source_fixture_hash: Optional[str] = None
    source_analysis_hash: Optional[str] = None


def _source(snapshot: PfResolvedSnapshot) -> Dict[str, Any]:
    source: Dict[str, Any] = {
        "git_commit": snapshot.source_git_commit,
        "config_hash": snapshot.config_hash,
        "kwn_backend": "PF_snapshot_import",
        "kwn_backend_version": "kwn_pf_handoff_v1",
    }
    if snapshot.source_binary_hash:
        source["binary_hash"] = snapshot.source_binary_hash
    if snapshot.source_fixture_hash:
        source["fixture_hash"] = snapshot.source_fixture_hash
    if snapshot.source_analysis_hash:
        source["analysis_hash"] = snapshot.source_analysis_hash
    return source


def pf_snapshot_to_handoff(snapshot: PfResolvedSnapshot) -> HandoffPackage:
    """Create a schema-valid package whose GP/sub-grid inventories are zero.

    This function is appropriate only for a PF snapshot whose source audit has
    already established matrix and resolved-beta inventories on a common basis.
    It never treats unresolved matrix solute as an omitted GP/beta population.
    """

    zero_gp_density = np.zeros(len(snapshot.gp_radius_bin_edges_m) - 1, dtype=float)
    zero_subgrid_density = np.zeros(
        len(snapshot.beta_subgrid_radius_bin_edges_m) - 1, dtype=float
    )
    total = float(snapshot.matrix_B_inventory) + float(snapshot.resolved_B_inventory)
    assumptions = list(snapshot.assumptions)
    assumptions.append(
        "PF snapshot import exposes matrix plus resolved beta only; GP and beta-subgrid buckets are exactly zero in this source package."
    )
    resolved_population: Dict[str, Any] = {
        "xB_beta": snapshot.resolved_xB_beta,
        "Vm_m3_per_mol": snapshot.resolved_Vm_m3_per_mol,
        "volume_fraction": snapshot.resolved_volume_fraction,
        "B_inventory": snapshot.resolved_B_inventory,
    }
    if snapshot.shape_orientation_metadata is not None:
        resolved_population["shape_orientation_metadata"] = dict(
            snapshot.shape_orientation_metadata
        )
    metadata: Dict[str, Any] = {
        "schema_version": "kwn_pf_handoff_v1",
        "source": _source(snapshot),
        "temperature_K": snapshot.temperature_K,
        "handoff_time_s": snapshot.handoff_time_s,
        "pf_box": {"lengths_m": list(snapshot.pf_box_lengths_m), "periodic": True},
        "unit_system": "SI",
        "inventory_basis": "mol_B_per_m3",
        "total_pseudo_binary_inventory": total,
        "assumptions": assumptions,
        "observation_dataset_role": snapshot.observation_dataset_role,
        "matrix": {
            "xB_alpha": snapshot.matrix_xB_alpha,
            "Vm_m3_per_mol": snapshot.matrix_Vm_m3_per_mol,
            "B_inventory": snapshot.matrix_B_inventory,
        },
        "gp_population": {
            "xB_g": snapshot.gp_xB_g,
            "Vm_m3_per_mol": snapshot.gp_Vm_m3_per_mol,
            "volume_fraction": 0.0,
            "B_inventory": 0.0,
        },
        "beta_subgrid_population": {
            "xB_beta": 1.0,
            "Vm_m3_per_mol": snapshot.beta_subgrid_Vm_m3_per_mol,
            "volume_fraction": 0.0,
            "B_inventory": 0.0,
        },
        "beta_resolved_population": resolved_population,
        "ledger": {
            "C_B_total": total,
            "C_B_matrix": snapshot.matrix_B_inventory,
            "C_B_GP": 0.0,
            "C_B_beta_subgrid": 0.0,
            "C_B_beta_resolved": snapshot.resolved_B_inventory,
            "residual": 0.0,
        },
    }
    arrays: Dict[str, np.ndarray] = {
        "gp_radius_bin_edges_m": np.asarray(snapshot.gp_radius_bin_edges_m, dtype=float),
        "gp_number_density_per_m4": zero_gp_density,
        "beta_subgrid_radius_bin_edges_m": np.asarray(
            snapshot.beta_subgrid_radius_bin_edges_m, dtype=float
        ),
        "beta_subgrid_number_density_per_m4": zero_subgrid_density,
        "beta_resolved_radius_bin_edges_m": np.asarray(
            snapshot.resolved_radius_bin_edges_m, dtype=float
        ),
        "beta_resolved_expected_count": np.asarray(
            snapshot.resolved_expected_count, dtype=float
        ),
    }
    if snapshot.resolved_sampled_radii_m is not None:
        arrays["beta_resolved_sampled_radii_m"] = np.asarray(
            snapshot.resolved_sampled_radii_m, dtype=float
        )
    if snapshot.resolved_centers_m is not None:
        arrays["beta_resolved_centers_m"] = np.asarray(
            snapshot.resolved_centers_m, dtype=float
        )
    return make_handoff_package(metadata, arrays)


build_handoff_from_pf_snapshot = pf_snapshot_to_handoff
