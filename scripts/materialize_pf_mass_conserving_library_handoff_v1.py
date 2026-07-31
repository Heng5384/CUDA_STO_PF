#!/usr/bin/env python3
"""Materialize the conditional six-hour handoff from the frozen V1 library.

The profile-library fields are translated only on their native periodic grid.
The portable composition field is the registered
``delta_C_relaxation=(1-h)(xB-xB_far)``.  A single matrix baseline is solved
from the target canonical inventory; no particle field is scaled, clipped,
interpolated, rotated, or pre-relaxed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import materialize_pf_elastic_multi_particle_6h_fixture_v1 as library_tools


INITIAL_STATE_CLASS = (
    "MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1"
)
SPEC_SCHEMA = "PF_MASS_CONSERVING_LIBRARY_HANDOFF_SPEC_V1"
MANIFEST_SCHEMA = "PF_MASS_CONSERVING_LIBRARY_HANDOFF_MANIFEST_V1"
RAW_META_SCHEMA = "PF_MASS_CONSERVING_LIBRARY_HANDOFF_RAW_INIT_META_V1"
FROZEN_LIBRARY_SHA256 = (
    "58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe"
)
FROZEN_SELECTION_SHA256 = (
    "56c44d8f72b27bb462dffe89b59cb2fcb2ff0bf8807dec9d9cac31d2bd7fcbe3"
)
FROZEN_SOURCE_TREE_SHA256 = (
    "f7855699addf98f9d5aed03af876d851d524fb62c98f5a48df78d1fe561a2a75"
)
FROZEN_BINARY_SHA256 = (
    "55cf917df94fcf01373d62f54f9ab99715975863ad5dcfb95baa8a517460cda1"
)
REGISTERED_RADII_NM = (8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5)


def write_raw(path: Path, value: np.ndarray) -> None:
    np.asarray(value, dtype="<f8").ravel(order="C").tofile(path)


def materialize(args: argparse.Namespace) -> Dict[str, Any]:
    if args.out.exists():
        raise ValueError(f"refusing to overwrite output: {args.out}")
    spec = library_tools.load_json(args.spec)
    if spec.get("schema") != SPEC_SCHEMA:
        raise ValueError("wrong conditional-handoff spec schema")
    if spec.get("initial_state_class") != INITIAL_STATE_CLASS:
        raise ValueError("wrong initial_state_class")
    if spec.get("profile_library_manifest_sha256") != FROZEN_LIBRARY_SHA256:
        raise ValueError("spec does not pin the frozen V1 library")
    if spec.get("selection_provenance_sha256") != FROZEN_SELECTION_SHA256:
        raise ValueError("spec does not pin the frozen V1 selection")

    library_path, library, profiles, freeze = library_tools.verify_library(
        args.library_root,
        FROZEN_LIBRARY_SHA256,
        args.selection_provenance,
        FROZEN_SELECTION_SHA256,
    )
    if freeze.get("source_tree_sha256") != FROZEN_SOURCE_TREE_SHA256:
        raise ValueError("frozen source-tree identity mismatch")
    if freeze.get("binary_sha256") != FROZEN_BINARY_SHA256:
        raise ValueError("frozen profile-library binary identity mismatch")

    structural = dict(spec)
    structural["schema"] = library_tools.SPEC_SCHEMA
    grid, dx_nm, lambda_sm_nm, particles = library_tools.validate_spec(
        structural, library, profiles
    )
    target = spec["target"]
    target_mean = float(target["mean_C_B_tot"])
    target_total = target_mean * math.prod(grid)
    xB_max_safe = float(target.get("xB_max_safe", 0.499999))
    threshold = float(target.get("component_h_threshold", 1.0e-4))
    if not math.isfinite(target_mean) or not (0.0 < target_mean < 1.0):
        raise ValueError("target mean inventory is invalid")
    if not math.isfinite(threshold) or not (0.0 < threshold < 1.0):
        raise ValueError("component threshold is invalid")

    phi_complement = np.ones(grid, dtype=np.float64)
    delta_x_total = np.zeros(grid, dtype=np.float64)
    particle_fields: List[Dict[str, Any]] = []
    for particle in particles:
        radius = float(particle["registered_radius_nm"])
        manifest_path, profile = profiles[radius]
        phi_path = library_tools.check_field(
            profile, manifest_path, "phi", grid
        )
        h_path = library_tools.check_field(
            profile, manifest_path, "h_phi", grid
        )
        delta_path = library_tools.check_field(
            profile, manifest_path, "delta_C_relaxation", grid
        )
        phi = library_tools.translated(
            library_tools.read_raw(phi_path, grid, "phi"),
            particle["center_grid"],
        )
        h_profile = library_tools.translated(
            library_tools.read_raw(h_path, grid, "h_phi"),
            particle["center_grid"],
        )
        delta_c = library_tools.translated(
            library_tools.read_raw(
                delta_path, grid, "delta_C_relaxation"
            ),
            particle["center_grid"],
        )
        if not all(
            np.all(np.isfinite(field))
            for field in (phi, h_profile, delta_c)
        ):
            raise ValueError(
                f"{particle['particle_id']}: non-finite profile field"
            )
        if float(np.max(np.abs(h_profile - library_tools.h_of_phi(phi)))) > 5.0e-14:
            raise ValueError(
                f"{particle['particle_id']}: frozen h(phi) identity mismatch"
            )
        phi_complement *= 1.0 - phi
        source_alpha = 1.0 - h_profile
        if float(np.min(source_alpha)) <= 0.0:
            raise ValueError(
                f"{particle['particle_id']}: source profile has zero alpha"
            )
        delta_x = delta_c / source_alpha
        if not np.all(np.isfinite(delta_x)):
            raise ValueError(
                f"{particle['particle_id']}: non-finite portable "
                "delta_C/alpha reconstruction"
            )
        delta_x_total += delta_x
        particle_fields.append(
            {
                "particle": particle,
                "manifest_path": manifest_path,
                "profile": profile,
                "h_profile": h_profile,
                "delta_x": delta_x,
            }
        )

    phi_total = 1.0 - phi_complement
    if (
        not np.all(np.isfinite(phi_total))
        or float(np.min(phi_total)) < 0.0
        or float(np.max(phi_total)) > 1.0
    ):
        raise ValueError("assembled phi is non-finite or out of [0,1]")
    h_total = library_tools.h_of_phi(phi_total)
    alpha = 1.0 - h_total
    effective_matrix_volume = float(np.sum(alpha, dtype=np.float64))
    beta_inventory = float(np.sum(h_total, dtype=np.float64))
    delta_c_total = alpha * delta_x_total
    relaxation_inventory = float(np.sum(delta_c_total, dtype=np.float64))
    if not math.isfinite(effective_matrix_volume) or effective_matrix_volume <= 0.0:
        raise ValueError("effective matrix volume is invalid")
    baseline = (
        target_total - beta_inventory - relaxation_inventory
    ) / effective_matrix_volume
    if not math.isfinite(baseline) or not (0.0 < baseline < xB_max_safe):
        raise ValueError("derived matrix baseline is invalid")
    if float(np.min(alpha)) <= 0.0:
        raise ValueError("assembled field contains an unresolved zero-alpha cell")
    xB = baseline + delta_c_total / alpha
    if (
        not np.all(np.isfinite(xB))
        or float(np.min(xB)) <= 0.0
        or float(np.max(xB)) >= xB_max_safe
    ):
        raise ValueError(
            "assembled xB is non-finite/out of range; clipping is forbidden"
        )
    C_total = h_total + alpha * xB
    Y = library_tools.logit(xB)
    dY_dt_prev = np.zeros(grid, dtype=np.float64)
    actual_total = float(np.sum(C_total, dtype=np.float64))
    relative_error = abs(actual_total - target_total) / max(
        abs(target_total), 1.0
    )
    if relative_error > 1.0e-14:
        raise ValueError("initial canonical inventory does not close")

    h_profile_sum = np.zeros(grid, dtype=np.float64)
    for row in particle_fields:
        h_profile_sum += row["h_profile"]
    positive = h_profile_sum > 0.0
    if np.any((h_total > 0.0) & ~positive):
        raise ValueError("cannot assign assembled beta inventory to library entries")

    mappings: List[Dict[str, Any]] = []
    assigned_total = 0.0
    for row in particle_fields:
        particle = row["particle"]
        profile = row["profile"]
        beta_share = np.zeros(grid, dtype=np.float64)
        beta_share[positive] = (
            h_total[positive]
            * row["h_profile"][positive]
            / h_profile_sum[positive]
        )
        assembled_h = float(np.sum(beta_share, dtype=np.float64))
        local_relaxation = float(
            np.sum(alpha * row["delta_x"], dtype=np.float64)
        )
        canonical_inventory = assembled_h + local_relaxation
        assigned_total += canonical_inventory
        radius = float(particle["registered_radius_nm"])
        mappings.append(
            {
                "particle_id": particle["particle_id"],
                "library_entry_id": f"R{radius:.1f}".replace(".", "p"),
                "registered_radius_nm": radius,
                "library_entry_sha256": library_tools.sha256(
                    row["manifest_path"]
                ),
                "center": [int(v) for v in particle["center_grid"]],
                "orientation": particle["orientation_label"],
                "phi_profile_hash": profile["fields"]["phi"]["sha256"],
                "delta_C_relaxation_hash": profile["fields"][
                    "delta_C_relaxation"
                ]["sha256"],
                "effective_h_volume": float(
                    profile["geometry"].get(
                        "final_h_volume_nm3",
                        profile["geometry"]["h_volume_nm3"],
                    )
                ),
                "assembled_effective_h_volume": assembled_h * dx_nm**3,
                "local_relaxation_inventory": local_relaxation,
                "canonical_particle_inventory": canonical_inventory,
            }
        )
    matrix_baseline_inventory = baseline * effective_matrix_volume
    decomposed_total = matrix_baseline_inventory + assigned_total
    decomposition_error = abs(decomposed_total - target_total) / max(
        abs(target_total), 1.0
    )
    if decomposition_error > 1.0e-14:
        raise ValueError("per-particle canonical decomposition does not close")

    component_labels, component_rows = library_tools.components(
        h_total, threshold, dx_nm
    )
    if len(component_rows) != len(particles):
        raise ValueError("assembled connected-component count mismatch")
    particle_to_component: Dict[str, int] = {}
    used_components = set()
    for particle in particles:
        center = tuple(int(v) for v in particle["center_grid"])
        component = int(component_labels[center])
        if component < 0 or component in used_components:
            raise ValueError("particle-to-component mapping is not bijective")
        used_components.add(component)
        particle_to_component[particle["particle_id"]] = component
    for component in component_rows:
        component_id = int(component["component_label"])
        component["particle_id"] = next(
            key
            for key, value in particle_to_component.items()
            if value == component_id
        )

    args.out.mkdir(parents=True)
    arrays = {
        "phi": phi_total,
        "h_phi": h_total,
        "delta_C_relaxation_total": delta_c_total,
        "C_B_tot": C_total,
        "xB_alpha": xB,
        "Y": Y,
        "dY_dt_prev": dY_dt_prev,
    }
    field_rows: Dict[str, Dict[str, str]] = {}
    for name, value in arrays.items():
        path = args.out / f"{name}.raw.f64"
        write_raw(path, value)
        field_rows[name] = {
            "path": path.name,
            "sha256": library_tools.sha256(path),
            "dtype": "float64-le",
            "order": "C",
        }

    meta = {
        "schema": RAW_META_SCHEMA,
        "Nx": grid[0],
        "Ny": grid[1],
        "Nz": grid[2],
        "dx_nm": dx_nm,
        "interface_width_nm": lambda_sm_nm,
        "dt_recommended": float(target.get("dt_code", 0.02)),
        "mean_xBtot": actual_total / math.prod(grid),
        "xB_max_safe": xB_max_safe,
        "dtype": "float64",
        "order": "C",
        "phi_path": field_rows["phi"]["path"],
        "xB_path": field_rows["xB_alpha"]["path"],
        "fresh_dY_dt_prev_contract": "zero_for_fresh_dynamic_start",
        "initial_state_class": INITIAL_STATE_CLASS,
    }
    library_tools.write_json(args.out / "init_meta.json", meta)

    with (args.out / "initial_particles.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        columns = list(mappings[0])
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(mappings)
    with (args.out / "initial_components.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        columns = [
            "particle_id",
            "component_label",
            "threshold_voxel_count",
            "h_volume_nm3",
            "equivalent_radius_nm",
            "centroid_nm",
            "semi_axes_nm",
            "axis_ratio",
            "principal_axes_rows",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            [{key: row.get(key) for key in columns} for row in component_rows]
        )

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "initial_state_class": INITIAL_STATE_CLASS,
        "fixture_id": spec["fixture_id"],
        "validation_only": True,
        "scientific_semantics": (
            "mass-conserving, profile-library-assembled conditional 6 h "
            "handoff state"
        ),
        "profile_library_manifest_sha256": FROZEN_LIBRARY_SHA256,
        "selection_provenance_sha256": FROZEN_SELECTION_SHA256,
        "source_tree_sha256": FROZEN_SOURCE_TREE_SHA256,
        "profile_library_binary_sha256": FROZEN_BINARY_SHA256,
        "source_spec_canonical_sha256": library_tools.canonical_spec_digest(
            spec
        ),
        "canonical_particle_order": [
            row["particle_id"] for row in mappings
        ],
        "particle_library_mappings": mappings,
        "target_global_inventory": {
            "mean_C_B_tot": target_mean,
            "total_C_B_tot_code": target_total,
            "source": "hash_pinned_manifest",
        },
        "derived_matrix_baseline": {
            "xB_alpha": baseline,
            "effective_matrix_volume_code": effective_matrix_volume,
            "matrix_baseline_inventory_code": matrix_baseline_inventory,
            "equation": (
                "(M_target-sum_j(canonical_particle_inventory_j))/"
                "V_effective_matrix"
            ),
            "clipping_used": False,
            "normalization_used": False,
        },
        "initial_canonical_inventory": {
            "target_total_code": target_total,
            "actual_total_code": actual_total,
            "matrix_baseline_inventory_code": matrix_baseline_inventory,
            "particle_inventory_sum_code": assigned_total,
            "assembled_beta_phase_inventory_code": beta_inventory,
            "local_relaxation_inventory_code": relaxation_inventory,
            "decomposed_total_code": decomposed_total,
            "field_relative_error": relative_error,
            "decomposition_relative_error": decomposition_error,
            "status": "PASS_MACHINE_PRECISION",
        },
        "initial_zero_mode_provenance": {
            "zero_mode": "PF_CONSERVED_Y_ZERO_MODE_V1",
            "backend": "HOST_NEWTON_BISECTION_V1",
            "explicit_context": "SM_EXPLICIT_CONTEXT_N_V1",
            "reaction_discretization": "SM_TANGENT_N_V1",
            "target_mass_source": "initial_canonical_inventory",
            "dY_dt_prev": "zero_for_fresh_dynamic_start",
        },
        "common_multi_particle_equilibrium_required": False,
        "common_multi_particle_equilibrium_claim": False,
        "initial_relaxation_is_physical_evolution": True,
        "particle_profiles_locked_after_t0": False,
        "full_field_xB_dt_MAE_blocking": False,
        "grid": {
            "Nx": grid[0],
            "Ny": grid[1],
            "Nz": grid[2],
            "dx_nm": dx_nm,
            "lambda_sm_nm": lambda_sm_nm,
        },
        "placement": spec["placement"],
        "physical_contract": {
            "temperature_C": float(target["temperature_C"]),
            "elasticity_enabled": True,
            "elastic_boundary": "periodic_fixed_cell",
            "orientation_label": "variant_100_identity",
            "eigenstrain": [0.046, -0.022, -0.017, 0.0, 0.0, 0.0],
            "GP_enabled": False,
            "GP_birth_enabled": False,
            "GP_release_enabled": False,
            "external_source_enabled": False,
            "new_beta_nucleation_enabled": False,
        },
        "assembly_contract": {
            "phi": "1-product_j(1-phi_j)",
            "portable_composition_field": (
                "source delta_C_relaxation_j reconstructed as "
                "delta_x_j=delta_C_j/(1-h_j), then stored in the target "
                "phase as (1-h_total)*sum_j(delta_x_j)"
            ),
            "absolute_xB_copied_across_boxes": False,
            "radial_interpolation_used": False,
            "profile_scaling_used": False,
            "rotation_used": False,
            "analytic_tanh_used": False,
            "clipping_used": False,
            "normalization_used": False,
            "optimizer_invoked": False,
            "common_multi_particle_pre_relaxation_run": False,
        },
        "component_contract": {
            "h_threshold": threshold,
            "expected_count": len(particles),
            "actual_count": len(component_rows),
            "particle_to_component": particle_to_component,
        },
        "fields": field_rows,
        "init_meta": {
            "path": "init_meta.json",
            "sha256": library_tools.sha256(args.out / "init_meta.json"),
        },
        "initial_particles": {
            "path": "initial_particles.csv",
            "sha256": library_tools.sha256(
                args.out / "initial_particles.csv"
            ),
        },
        "initial_components": {
            "path": "initial_components.csv",
            "sha256": library_tools.sha256(
                args.out / "initial_components.csv"
            ),
        },
        "library_manifest_path_used_for_materialization": str(library_path),
    }
    library_tools.write_json(args.out / "fixture_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--selection-provenance", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = materialize(args)
    except ValueError as exc:
        raise SystemExit(f"[fatal] {exc}") from exc
    print(
        json.dumps(
            {
                "status": (
                    "PASS_MASS_CONSERVING_LIBRARY_ASSEMBLED_"
                    "CONDITIONAL_HANDOFF_V1"
                ),
                "fixture_manifest_sha256": library_tools.sha256(
                    args.out / "fixture_manifest.json"
                ),
                "particle_count": len(
                    manifest["particle_library_mappings"]
                ),
                "initial_inventory_relative_error": manifest[
                    "initial_canonical_inventory"
                ]["field_relative_error"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
