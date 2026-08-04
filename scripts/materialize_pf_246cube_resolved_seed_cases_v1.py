#!/usr/bin/env python3
"""Materialize one R1/R2 246-cube resolved-seed conditional fixture.

Only complete, byte-pinned entries of the frozen 15-entry target-profile
library are embedded.  This is deliberately a field assembly operation, not
a profile generator: no resampling, interpolation, radius scaling, or
single-particle profile edit is available here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

import materialize_pf_246cube_library_handoff_v1 as legacy
import materialize_pf_elastic_multi_particle_6h_fixture_v1 as library_tools


SCHEMA = "PF_246CUBE_LIBRARY_HANDOFF_MANIFEST_V1"
INITIAL_STATE_CLASS = "MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1"
SELECTION_SCHEMA = "PF_246CUBE_RESOLVED_SEED_THERMAL_CASE_INTEGER_SELECTION_V1"
LIBRARY_SHA256 = "de4142e0268e379f70fd1c860aab9e004421d07df4f8d872f4eaeb3dbef7af5b"
GRID = (246, 246, 246)
NATIVE_GRID = (96, 96, 96)
TARGET_MEAN_C = 0.03
COMPONENT_H_THRESHOLD = 1.0e-4
MATRIX_XAG_CENTER = 0.0062
MATRIX_XAG_INTERVAL = (0.0058, 0.0066)
PHI_NUMERICAL_MIN = -1.0e-6
PHI_NUMERICAL_MAX = 1.0 + 1.0e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def xag_from_xb(value: float) -> float:
    return 2.0 * value / (2.0 + value)


def candidate_particles(case: dict[str, Any]) -> list[dict[str, Any]]:
    particles: list[dict[str, Any]] = []
    for radius_text, count_value in sorted(case["histogram"].items(), key=lambda item: float(item[0])):
        radius = float(radius_text)
        for _ in range(int(count_value)):
            particles.append({
                "particle_id": f"P{len(particles):03d}",
                "registered_radius_nm": radius,
                "orientation_label": "variant_100_identity",
            })
    if len(particles) != int(case["particle_count"]):
        raise ValueError("selection histogram/count mismatch")
    return particles


def load_profiles(library_root: Path, expected_radii: list[float]) -> tuple[dict[float, dict[str, Any]], dict[str, Any]]:
    _path, library, profiles, identity = library_tools.verify_library(
        library_root, LIBRARY_SHA256, None, None, expected_radii
    )
    if tuple(int(library["grid"][key]) for key in ("Nx", "Ny", "Nz")) != NATIVE_GRID:
        raise ValueError("unexpected library grid")
    if float(library["lambda_sm_nm"]) != 4.0 or float(library["temperature_C"]) != 380.0:
        raise ValueError("library physical identity mismatch")
    profile_data: dict[float, dict[str, Any]] = {}
    coordinates = np.indices(NATIVE_GRID).transpose(1, 2, 3, 0)
    offsets = (coordinates - np.asarray([48, 48, 48]) + 48) % 96 - 48
    radial = np.sqrt(np.sum(offsets.astype(np.float64) ** 2, axis=-1))
    for radius, (manifest_path, manifest) in profiles.items():
        phi = library_tools.read_raw(library_tools.check_field(manifest, manifest_path, "phi", NATIVE_GRID), NATIVE_GRID, "phi")
        h = library_tools.read_raw(library_tools.check_field(manifest, manifest_path, "h_phi", NATIVE_GRID), NATIVE_GRID, "h_phi")
        delta_c = library_tools.read_raw(library_tools.check_field(manifest, manifest_path, "delta_C_relaxation", NATIVE_GRID), NATIVE_GRID, "delta_C_relaxation")
        if float(np.max(np.abs(h - library_tools.h_of_phi(phi)))) > 5.0e-14:
            raise ValueError(f"R={radius}: h(phi) identity mismatch")
        alpha = 1.0 - h
        if float(np.min(alpha)) <= 0.0:
            raise ValueError(f"R={radius}: singular alpha")
        support = h > COMPONENT_H_THRESHOLD
        if not np.any(support):
            raise ValueError(f"R={radius}: empty support")
        profile_data[float(radius)] = {
            "manifest": manifest,
            "manifest_path": manifest_path,
            "phi": phi,
            "h": h,
            "delta_x": delta_c / alpha,
            "support": support,
            "support_radius_nm": float(np.max(radial[support])),
            "profile_manifest_sha256": sha256(manifest_path),
            "source_phi_sha256": manifest["fields"]["phi"]["sha256"],
            "source_delta_C_relaxation_sha256": manifest["fields"]["delta_C_relaxation"]["sha256"],
        }
    return profile_data, identity


def build(case_name: str, selection: dict[str, Any], profile_data: dict[float, dict[str, Any]], library_identity: dict[str, Any], order: str) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    case = selection["cases"][case_name]
    particles = candidate_particles(case)
    expected_radii = set(profile_data)
    if not {float(row["registered_radius_nm"]) for row in particles}.issubset(expected_radii):
        raise ValueError("selection calls an unavailable profile radius")
    for row in particles:
        profile = profile_data[float(row["registered_radius_nm"])]
        row["source_profile_manifest_sha256"] = profile["profile_manifest_sha256"]
        row["source_phi_sha256"] = profile["source_phi_sha256"]
        row["source_delta_C_relaxation_sha256"] = profile["source_delta_C_relaxation_sha256"]
    if order == "reverse":
        particles.reverse()
    elif order != "canonical":
        raise ValueError("input order must be canonical or reverse")
    # Placement order is canonicalized inside the legacy hard-core routine.
    placed, spatial = legacy.place_particles(
        particles,
        int(case["placement_seed_unsigned64"]),
        {radius: data["support_radius_nm"] for radius, data in profile_data.items()},
    )
    phi_complement = np.ones(GRID, dtype=np.float64)
    delta_x_total = np.zeros(GRID, dtype=np.float64)
    source_h_sum = np.zeros(GRID, dtype=np.float64)
    support_owner = np.zeros(GRID, dtype=np.uint8)
    for particle in placed:
        profile = profile_data[float(particle["registered_radius_nm"])]
        indices = np.ix_(*legacy.source_window_indices(particle["center_grid"]))
        local_owner = support_owner[indices]
        if np.any(local_owner[profile["support"]]):
            raise ValueError("qualified profile supports overlap")
        local_owner[profile["support"]] = 1
        support_owner[indices] = local_owner
        local = phi_complement[indices]
        local *= 1.0 - profile["phi"]
        phi_complement[indices] = local
        local_delta = delta_x_total[indices]
        local_delta += profile["delta_x"]
        delta_x_total[indices] = local_delta
        local_h = source_h_sum[indices]
        local_h += profile["h"]
        source_h_sum[indices] = local_h
    phi = 1.0 - phi_complement
    h = library_tools.h_of_phi(phi)
    if not np.all(np.isfinite(phi)) or float(np.min(phi)) < 0.0 or float(np.max(phi)) > 1.0:
        raise ValueError("assembled phi violates fresh-field bounds")
    alpha = 1.0 - h
    if float(np.min(alpha)) <= 0.0 or np.any((h > 0.0) & (source_h_sum <= 0.0)):
        raise ValueError("assembled alpha/tail allocation is invalid")
    delta_c = alpha * delta_x_total
    target_total = TARGET_MEAN_C * math.prod(GRID)
    beta_inventory = float(np.sum(h, dtype=np.float64))
    relaxation_inventory = float(np.sum(delta_c, dtype=np.float64))
    matrix_volume = float(np.sum(alpha, dtype=np.float64))
    matrix_xb = (target_total - beta_inventory - relaxation_inventory) / matrix_volume
    if not math.isfinite(matrix_xb) or not 0.0 < matrix_xb < 0.499999:
        raise ValueError("derived matrix baseline lies outside domain")
    x_b = matrix_xb + delta_x_total
    if not np.all(np.isfinite(x_b)) or float(np.min(x_b)) <= 0.0 or float(np.max(x_b)) >= 0.499999:
        raise ValueError("assembled composition would require clipping")
    c_total = h + alpha * x_b
    actual_total = float(np.sum(c_total, dtype=np.float64))
    mass_error = abs(actual_total - target_total) / target_total
    if mass_error > 1.0e-14:
        raise ValueError("canonical total inventory does not close")
    y = library_tools.logit(x_b)
    d_y = np.zeros(GRID, dtype=np.float64)
    labels, component_count = legacy.periodic_labels(h > COMPONENT_H_THRESHOLD)
    if component_count != len(placed):
        raise ValueError(f"component count {component_count} differs from selected particles {len(placed)}")
    groups = legacy.group_component_indices(labels)
    component_rows: list[dict[str, Any]] = []
    used: set[int] = set()
    mapping_rows: list[dict[str, Any]] = []
    assigned_beta = 0.0
    assigned_relaxation = 0.0
    for index, particle in enumerate(placed):
        profile = profile_data[float(particle["registered_radius_nm"])]
        indices = np.ix_(*legacy.source_window_indices(particle["center_grid"]))
        denominator = source_h_sum[indices]
        share = np.zeros(NATIVE_GRID, dtype=np.float64)
        active = denominator > 0.0
        share[active] = h[indices][active] * profile["h"][active] / denominator[active]
        beta_share = float(np.sum(share, dtype=np.float64))
        relaxation_share = float(np.sum(alpha[indices] * profile["delta_x"], dtype=np.float64))
        if index + 1 == len(placed):
            beta_share = beta_inventory - assigned_beta
            relaxation_share = relaxation_inventory - assigned_relaxation
        else:
            assigned_beta += beta_share
            assigned_relaxation += relaxation_share
        center = tuple(int(value) for value in particle["center_grid"])
        component = int(labels[center])
        if component <= 0 or component in used:
            raise ValueError("particle/component map is not bijective")
        used.add(component)
        metrics = legacy.component_metrics(groups[component], component, h)
        metrics["particle_id"] = particle["particle_id"]
        component_rows.append(metrics)
        radius = float(particle["registered_radius_nm"])
        mapping_rows.append({
            **particle,
            "replicate_id": f"thermal_{case_name}",
            "center_nm": [float(value) for value in particle["center_grid"]],
            "orientation": "variant_100_identity",
            "library_entry_id": f"R{radius:g}".replace(".", "p"),
            "library_entry_sha256": profile["profile_manifest_sha256"],
            "effective_h_volume": float(profile["manifest"]["geometry"]["h_volume_nm3"]),
            "assembled_beta_inventory": beta_share,
            "assembled_relaxation_inventory": relaxation_share,
            "canonical_particle_inventory": beta_share + relaxation_share,
        })
    if len(used) != len(placed):
        raise ValueError("incomplete component mapping")
    particle_inventory = float(sum(row["canonical_particle_inventory"] for row in mapping_rows))
    matrix_inventory = matrix_xb * matrix_volume
    decomposition_error = abs(matrix_inventory + particle_inventory - target_total) / target_total
    if decomposition_error > 1.0e-14:
        raise ValueError("particle/matrix ledger does not close")
    observed_mask = h < 0.005
    observed_xb = float(np.mean(x_b[observed_mask], dtype=np.float64))
    observed_xag = xag_from_xb(observed_xb)
    if not MATRIX_XAG_INTERVAL[0] <= observed_xag <= MATRIX_XAG_INTERVAL[1]:
        raise ValueError("assembled far-field matrix xAg lies outside the experimental interval")
    field_h_error = beta_inventory - float(case["target_h_volume_nm3"])
    if abs(field_h_error) > 2.0:
        raise ValueError("assembled field h-volume no longer closes to selected-library target")
    fields = {
        "phi": phi,
        "h_phi": h,
        "delta_C_relaxation_total": delta_c,
        "C_B_tot": c_total,
        "xB_alpha": x_b,
        "Y": y,
        "dY_dt_prev": d_y,
    }
    manifest = {
        "schema": SCHEMA,
        "initial_state_class": INITIAL_STATE_CLASS,
        "fixture_id": f"pf_246cube_resolved_seed_thermal_{case_name}_v1",
        "replicate_id": f"thermal_{case_name}",
        "validation_only": True,
        "scientific_semantics": "fixed-inventory exact-profile conditional 6 h resolved-seed sensitivity state",
        "source_psd": {**case, "selection_file_sha256": selection["selection_file_sha256"]},
        "inventory_selection": {
            "schema": selection["schema"],
            "selection_file_sha256": selection["selection_file_sha256"],
            "optimizer": selection["optimizer"],
        },
        "profile_library_manifest_sha256": LIBRARY_SHA256,
        "source_tree_sha256": library_identity["source_tree_sha256"],
        "profile_library_binary_sha256": library_identity["binary_sha256"],
        "profile_library_binary_sha256_set": library_identity.get("binary_sha256_set", [library_identity["binary_sha256"]]),
        "profile_library_mixed_binary_contract": library_identity.get("mixed_binary_contract", "SINGLE_BINARY"),
        "canonical_particle_order": [row["particle_id"] for row in mapping_rows],
        "particle_library_mappings": mapping_rows,
        "target_global_inventory": {"mean_C_B_tot": TARGET_MEAN_C, "total_C_B_tot_code": target_total, "source": "hash_pinned_selection"},
        "derived_matrix_baseline": {
            "xB_alpha": matrix_xb,
            "xAg": xag_from_xb(matrix_xb),
            "observed_matrix_xB_h_lt_0p005": observed_xb,
            "observed_matrix_xAg_h_lt_0p005": observed_xag,
            "experimental_xAg_center": MATRIX_XAG_CENTER,
            "experimental_xAg_interval": list(MATRIX_XAG_INTERVAL),
            "effective_matrix_volume_code": matrix_volume,
            "matrix_baseline_inventory_code": matrix_inventory,
            "clipping_used": False,
            "normalization_used": False,
        },
        "initial_canonical_inventory": {
            "target_total_code": target_total,
            "actual_total_code": actual_total,
            "matrix_baseline_inventory_code": matrix_inventory,
            "particle_inventory_sum_code": particle_inventory,
            "assembled_beta_phase_inventory_code": beta_inventory,
            "local_relaxation_inventory_code": relaxation_inventory,
            "field_relative_error": mass_error,
            "decomposition_relative_error": decomposition_error,
            "h_volume_relative_error_from_selection_target": abs(field_h_error) / float(case["target_h_volume_nm3"]),
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
        "grid": {"Nx": 246, "Ny": 246, "Nz": 246, "dx_nm": 1.0, "lambda_sm_nm": 4.0},
        "placement": {
            **spatial,
            "seed_unsigned64": int(case["placement_seed_unsigned64"]),
            "periodic_separation_rule": "distance_nm > Ri_nm + Rj_nm + 4*lambda_sm_nm",
            "physical_support_definition": "h(phi)>1e-4",
        },
        "physical_contract": {
            "temperature_C": 380.0,
            "elasticity_enabled": True,
            "elastic_boundary": "periodic_fixed_cell",
            "orientation_label": "variant_100_identity",
            "eigenstrain": [0.046, -0.022, -0.017, 0.0, 0.0, 0.0],
            "external_strain": [0.0] * 6,
            "external_stress": [0.0] * 6,
            "initial_age_h": 6.0,
        },
        "assembly_contract": {
            "phi": "1-product_j(1-phi_j), canonical particle-id order",
            "portable_composition_field": "native delta_C_relaxation_j embedded without resampling and rebuilt as (1-h_total)*sum_j(delta_x_j)",
            "native_window_embedding": "complete 96^3 native binary64 field at each deterministic center",
            "absolute_xB_copied_across_boxes": False,
            "radial_interpolation_used": False,
            "profile_scaling_used": False,
            "spatial_resampling_used": False,
            "rotation_used": False,
            "analytic_tanh_used": False,
            "clipping_used": False,
            "normalization_used": False,
            "optimizer_invoked": True,
            "common_multi_particle_pre_relaxation_run": False,
            "input_manifest_order": order,
            "canonical_assembly_order_enforced": True,
        },
        "component_contract": {
            "h_threshold": COMPONENT_H_THRESHOLD,
            "expected_count": len(placed),
            "actual_count": component_count,
            "particle_to_component": {row["particle_id"]: next(metric["component_label"] for metric in component_rows if metric["particle_id"] == row["particle_id"]) for row in mapping_rows},
        },
        "field_bounds": {"phi_min": float(np.min(phi)), "phi_max": float(np.max(phi)), "xB_min": float(np.min(x_b)), "xB_max": float(np.max(x_b)), "phi_numerical_contract": [PHI_NUMERICAL_MIN, PHI_NUMERICAL_MAX]},
        "selected_library": library_identity,
    }
    return fields, manifest, mapping_rows, component_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--case", choices=("R1", "R2"), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--input-order", choices=("canonical", "reverse"), default="canonical")
    args = parser.parse_args()
    if (args.out is None) == (args.compare_to is None):
        raise SystemExit("[fatal] provide exactly one of --out or --compare-to")
    selection = load_json(args.selection)
    if selection.get("schema") != SELECTION_SCHEMA or selection.get("status") != "PASS_RESOLVED_SEED_INTEGER_PSD_SELECTION_V1":
        raise SystemExit("[fatal] selection schema/status mismatch")
    if selection.get("profile_library_manifest_sha256") != LIBRARY_SHA256:
        raise SystemExit("[fatal] selection does not pin the exact 15-entry library")
    selection["selection_file_sha256"] = sha256(args.selection)
    expected_radii = [8.0 + 0.25 * index for index in range(15)]
    profiles, identity = load_profiles(args.library_root.resolve(), expected_radii)
    fields, manifest, mappings, components = build(args.case, selection, profiles, identity, args.input_order)
    if args.out is not None:
        if args.out.exists():
            raise SystemExit(f"[fatal] refusing to overwrite fixture root: {args.out}")
        identity_out = legacy.write_fixture(args.out, fields, manifest, mappings, components)
        print(json.dumps({"status": "PASS_246CUBE_RESOLVED_SEED_THERMAL_FIXTURE_MATERIALIZED_V1", "case": args.case, "particle_count": len(mappings), "matrix_xAg": manifest["derived_matrix_baseline"]["observed_matrix_xAg_h_lt_0p005"], **identity_out}, sort_keys=True))
    else:
        result = legacy.compare_fixture(args.compare_to, fields, manifest)
        if not result["raw_field_hashes_equal"] or not result["semantic_manifest_equal"]:
            raise SystemExit("[fatal] deterministic comparison failed")
        print(json.dumps({"status": "PASS_246CUBE_RESOLVED_SEED_THERMAL_FIXTURE_DETERMINISM_V1", "case": args.case, **result}, sort_keys=True))


if __name__ == "__main__":
    main()
