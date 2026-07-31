#!/usr/bin/env python3
"""Freeze an already materialized, exact-library V2 field as the V1 handoff.

This is an evidence-preserving schema promotion: every raw field is copied
byte-for-byte.  The source V2 manifest must already prove the frozen library
identity, native-grid profile mapping, phase-consistent use of
``delta_C_relaxation``, deterministic placement, and machine-precision global
inventory.  No physical field is regenerated or modified here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import materialize_pf_elastic_multi_particle_6h_fixture_v1 as tools
import materialize_pf_mass_conserving_library_handoff_v1 as contract


SOURCE_SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_6H_FIXTURE_V2"
SOURCE_CONTRACT = "PHASE_CONSISTENT_DELTA_X_ALPHA_V2"


def checked_field(
    source: Path, manifest: Dict[str, Any], name: str, shape: tuple[int, int, int]
) -> np.ndarray:
    row = manifest.get("fields", {}).get(name)
    if not isinstance(row, dict):
        raise ValueError(f"missing source field: {name}")
    path = source / str(row.get("path", ""))
    if not path.is_file() or tools.sha256(path) != row.get("sha256"):
        raise ValueError(f"source field hash mismatch: {name}")
    value = np.fromfile(path, dtype="<f8")
    if value.size != math.prod(shape) or not np.all(np.isfinite(value)):
        raise ValueError(f"source field is malformed/non-finite: {name}")
    return value.reshape(shape, order="C")


def freeze(args: argparse.Namespace) -> Dict[str, Any]:
    if args.out.exists():
        raise ValueError(f"refusing to overwrite output: {args.out}")
    source_manifest_path = args.source_fixture / "fixture_manifest.json"
    source_manifest = tools.load_json(source_manifest_path)
    if source_manifest.get("schema") != SOURCE_SCHEMA:
        raise ValueError("source is not the qualified V2 fixture schema")
    if source_manifest.get("fixture_kind") != "E2":
        raise ValueError("only the E2 source fixture can be promoted")
    if source_manifest.get("validation_only") is not True:
        raise ValueError("source fixture is not validation-only")
    selected = source_manifest.get("selected_library", {})
    expected_pairs = {
        "selected_library_manifest_sha256": contract.FROZEN_LIBRARY_SHA256,
        "selection_provenance_sha256": contract.FROZEN_SELECTION_SHA256,
        "source_tree_sha256": contract.FROZEN_SOURCE_TREE_SHA256,
        "binary_sha256": contract.FROZEN_BINARY_SHA256,
    }
    for key, expected in expected_pairs.items():
        if selected.get(key) != expected:
            raise ValueError(f"source frozen identity mismatch: {key}")
    assembly = source_manifest.get("combination", {})
    if assembly.get("composition_contract") != SOURCE_CONTRACT:
        raise ValueError("source lacks the phase-consistent delta-C contract")
    if (
        assembly.get("absolute_xB_superposed") is not False
        or assembly.get("radial_interpolation_used") is not False
        or assembly.get("spatial_resampling_used") is not False
        or assembly.get("rotated_variant_used") is not False
        or assembly.get("clipping_used") is not False
        or assembly.get("normalization_used") is not False
    ):
        raise ValueError("source used a forbidden field transformation")
    physical = source_manifest.get("physical_contract", {})
    forbidden = (
        "GP_enabled",
        "GP_birth_enabled",
        "GP_release_enabled",
        "external_source_enabled",
        "new_beta_nucleation_enabled",
    )
    if any(physical.get(key) is not False for key in forbidden):
        raise ValueError("source enables a prohibited GP/source/nucleation path")

    grid_row = source_manifest["grid"]
    shape = tuple(int(grid_row[key]) for key in ("Nx", "Ny", "Nz"))
    phi = checked_field(args.source_fixture, source_manifest, "phi", shape)
    h = checked_field(args.source_fixture, source_manifest, "h_phi", shape)
    xB = checked_field(args.source_fixture, source_manifest, "xB_alpha", shape)
    C = checked_field(args.source_fixture, source_manifest, "C_B_tot", shape)
    checked_field(args.source_fixture, source_manifest, "Y", shape)
    checked_field(args.source_fixture, source_manifest, "dY_dt_prev", shape)
    delta_c = checked_field(
        args.source_fixture,
        source_manifest,
        "delta_C_relaxation_total",
        shape,
    )
    if (
        float(np.min(phi)) < 0.0
        or float(np.max(phi)) > 1.0
        or float(np.min(xB)) <= 0.0
        or float(np.max(xB)) >= 1.0
    ):
        raise ValueError("source fields lie outside their legal domains")
    h_recomputed = tools.h_of_phi(phi)
    if float(np.max(np.abs(h - h_recomputed))) > 5.0e-14:
        raise ValueError("source h(phi) mismatch")
    if float(np.max(np.abs(C - (h + (1.0 - h) * xB)))) > 1.0e-14:
        raise ValueError("source canonical field identity mismatch")

    inventory = source_manifest["inventory"]
    target_total = float(inventory["target_total_C_B_tot"])
    actual_total = float(np.sum(C, dtype=np.float64))
    field_relative_error = abs(actual_total - target_total) / max(
        abs(target_total), 1.0
    )
    if field_relative_error > 1.0e-14:
        raise ValueError("source canonical inventory does not close")
    beta_total = float(np.sum(h, dtype=np.float64))
    matrix_total = float(np.sum((1.0 - h) * xB, dtype=np.float64))
    if abs(matrix_total + beta_total - target_total) / max(
        abs(target_total), 1.0
    ) > 1.0e-14:
        raise ValueError("source matrix/beta inventory decomposition fails")

    library_rows = {
        float(row["registered_radius_nm"]): row
        for row in selected.get("profiles", [])
    }
    particles = sorted(
        (dict(row) for row in source_manifest.get("particles", [])),
        key=lambda row: str(row.get("particle_id", "")),
    )
    if not particles or len({row["particle_id"] for row in particles}) != len(
        particles
    ):
        raise ValueError("source particle ids are missing/duplicated")
    source_h_sum = sum(float(row["source_h_volume_nm3"]) for row in particles)
    mappings: List[Dict[str, Any]] = []
    allocated = 0.0
    for index, particle in enumerate(particles):
        radius = float(particle["registered_radius_nm"])
        if radius not in contract.REGISTERED_RADII_NM:
            raise ValueError(f"unregistered radius: {radius}")
        library_row = library_rows.get(radius)
        if not library_row:
            raise ValueError(f"missing library entry: {radius}")
        identity_pairs = {
            "source_profile_manifest_sha256": "profile_manifest_sha256",
            "source_phi_sha256": "phi_sha256",
            "source_delta_C_relaxation_sha256": (
                "delta_C_relaxation_sha256"
            ),
        }
        for particle_key, library_key in identity_pairs.items():
            if particle.get(particle_key) != library_row.get(library_key):
                raise ValueError(
                    f"{particle['particle_id']}: {particle_key} mismatch"
                )
        if particle.get("orientation_label") != "variant_100_identity":
            raise ValueError("non-identity orientation is not qualified")
        center = particle.get("center_grid")
        if (
            not isinstance(center, list)
            or len(center) != 3
            or any(int(value) != value for value in center)
            or any(
                not (0 <= int(value) < limit)
                for value, limit in zip(center, shape)
            )
        ):
            raise ValueError("illegal particle center")
        # The bounded-union field differs from the sum of isolated h tails by
        # a tiny deterministic assembly term.  Allocate the exact assembled
        # beta inventory by the frozen isolated h-volume weights.  This keeps
        # the global ledger exact without changing any profile field.
        if index + 1 == len(particles):
            canonical_particle_inventory = beta_total - allocated
        else:
            canonical_particle_inventory = (
                beta_total
                * float(particle["source_h_volume_nm3"])
                / source_h_sum
            )
            allocated += canonical_particle_inventory
        mappings.append(
            {
                "particle_id": particle["particle_id"],
                "library_entry_id": f"R{radius:.1f}".replace(".", "p"),
                "registered_radius_nm": radius,
                "library_entry_sha256": particle[
                    "source_profile_manifest_sha256"
                ],
                "center": [int(value) for value in center],
                "orientation": particle["orientation_label"],
                "phi_profile_hash": particle["source_phi_sha256"],
                "delta_C_relaxation_hash": particle[
                    "source_delta_C_relaxation_sha256"
                ],
                "effective_h_volume": float(
                    particle["source_h_volume_nm3"]
                ),
                "canonical_particle_inventory": (
                    canonical_particle_inventory
                ),
            }
        )
    allocated_beta = sum(
        row["canonical_particle_inventory"] for row in mappings
    )
    decomposition_relative_error = abs(
        matrix_total + allocated_beta - target_total
    ) / max(abs(target_total), 1.0)
    if decomposition_relative_error > 1.0e-14:
        raise ValueError("promoted particle ledger does not close")

    args.out.mkdir(parents=True)
    copied_fields: Dict[str, Dict[str, str]] = {}
    for name, row in source_manifest["fields"].items():
        source_path = args.source_fixture / row["path"]
        destination = args.out / row["path"]
        shutil.copyfile(source_path, destination)
        if tools.sha256(destination) != row["sha256"]:
            raise ValueError(f"bytewise copy failed: {name}")
        copied_fields[name] = dict(row)
    for name in ("initial_particles.csv", "initial_components.csv"):
        shutil.copyfile(args.source_fixture / name, args.out / name)

    source_meta = tools.load_json(args.source_fixture / "init_meta.json")
    meta = dict(source_meta)
    meta["schema"] = contract.RAW_META_SCHEMA
    meta["initial_state_class"] = contract.INITIAL_STATE_CLASS
    tools.write_json(args.out / "init_meta.json", meta)

    manifest = {
        "schema": contract.MANIFEST_SCHEMA,
        "initial_state_class": contract.INITIAL_STATE_CLASS,
        "fixture_id": source_manifest["fixture_id"],
        "validation_only": True,
        "scientific_semantics": (
            "mass-conserving, profile-library-assembled conditional 6 h "
            "handoff state"
        ),
        "profile_library_manifest_sha256": contract.FROZEN_LIBRARY_SHA256,
        "selection_provenance_sha256": contract.FROZEN_SELECTION_SHA256,
        "source_tree_sha256": contract.FROZEN_SOURCE_TREE_SHA256,
        "profile_library_binary_sha256": contract.FROZEN_BINARY_SHA256,
        "source_fixture_manifest_sha256": tools.sha256(
            source_manifest_path
        ),
        "canonical_particle_order": [
            row["particle_id"] for row in mappings
        ],
        "particle_library_mappings": mappings,
        "target_global_inventory": {
            "mean_C_B_tot": float(inventory["target_mean_C_B_tot"]),
            "total_C_B_tot_code": target_total,
            "source": "hash_pinned_manifest",
        },
        "derived_matrix_baseline": {
            "xB_alpha": float(inventory["matrix_xB"]),
            "effective_matrix_volume_code": float(
                np.sum(1.0 - h, dtype=np.float64)
            ),
            "matrix_inventory_code": matrix_total,
            "equation": (
                "(M_target-sum_j(canonical_particle_inventory_j))/"
                "V_effective_matrix, with the frozen local "
                "delta_C_relaxation retained in the matrix field"
            ),
            "clipping_used": False,
            "normalization_used": False,
        },
        "initial_canonical_inventory": {
            "target_total_code": target_total,
            "actual_total_code": actual_total,
            "matrix_inventory_code": matrix_total,
            "particle_inventory_sum_code": allocated_beta,
            "local_relaxation_inventory_code": float(
                np.sum(delta_c, dtype=np.float64)
            ),
            "field_relative_error": field_relative_error,
            "decomposition_relative_error": decomposition_relative_error,
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
            **grid_row,
            "lambda_sm_nm": float(physical["lambda_sm_nm"]),
        },
        "placement": source_manifest["placement"],
        "physical_contract": physical,
        "assembly_contract": {
            "phi": source_manifest["combination"]["phi"],
            "portable_composition_field": (
                "frozen per-entry delta_C_relaxation; phase-consistent "
                "target storage documented by source V2 contract"
            ),
            "source_composition_contract": SOURCE_CONTRACT,
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
        "component_contract": source_manifest["component_contract"],
        "fields": copied_fields,
        "init_meta": {
            "path": "init_meta.json",
            "sha256": tools.sha256(args.out / "init_meta.json"),
        },
        "initial_particles": {
            "path": "initial_particles.csv",
            "sha256": tools.sha256(args.out / "initial_particles.csv"),
        },
        "initial_components": {
            "path": "initial_components.csv",
            "sha256": tools.sha256(args.out / "initial_components.csv"),
        },
    }
    tools.write_json(args.out / "fixture_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = freeze(args)
    except ValueError as exc:
        raise SystemExit(f"[fatal] {exc}") from exc
    print(
        json.dumps(
            {
                "status": (
                    "PASS_MASS_CONSERVING_LIBRARY_ASSEMBLED_"
                    "CONDITIONAL_HANDOFF_V1"
                ),
                "fixture_manifest_sha256": tools.sha256(
                    args.out / "fixture_manifest.json"
                ),
                "particle_count": len(
                    manifest["particle_library_mappings"]
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
