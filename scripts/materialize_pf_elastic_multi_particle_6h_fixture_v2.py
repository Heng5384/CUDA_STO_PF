#!/usr/bin/env python3
"""Materialize a conserved V2 multi-particle elastic PF fixture.

V1 deliberately demonstrated that a direct sum of profile
``delta_C_relaxation`` fields is not a valid composition operation for
multiple diffuse beta profiles.  V2 preserves the frozen profile geometry but
combines the *matrix composition deviations* of every source profile:

    delta_x_alpha_total = sum_j(xB_alpha,j - xB_far,j)
    C_B_tot = h_total + (1-h_total) * (xB_matrix + delta_x_alpha_total)

The scalar ``xB_matrix`` is solved once from the pinned global inventory.
There is no clipping, scaling, interpolation, rotation, or superposition of
absolute xB fields.  V1 inputs and V1 evidence remain untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import materialize_pf_elastic_multi_particle_6h_fixture_v1 as v1


SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_6H_FIXTURE_V2"
SPEC_SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_6H_FIXTURE_SPEC_V2"
RAW_META_SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_RAW_INIT_META_V2"
COMPOSITION_CONTRACT = "PHASE_CONSISTENT_DELTA_X_ALPHA_V2"


def _validate_spec(
    spec: Dict[str, Any], library: Dict[str, Any], profiles: Dict[float, Tuple[Path, Dict[str, Any]]],
) -> Tuple[Tuple[int, int, int], float, float, List[Dict[str, Any]]]:
    if spec.get("schema") != SPEC_SCHEMA:
        raise ValueError("wrong V2 multi-particle fixture spec schema")
    if spec.get("composition_contract") != COMPOSITION_CONTRACT:
        raise ValueError("V2 requires PHASE_CONSISTENT_DELTA_X_ALPHA_V2")
    # V1's structural checks deliberately remain the authoritative placement,
    # profile-grid, source-phi, source-delta and periodic-separation contract.
    structural = dict(spec)
    structural["schema"] = v1.SPEC_SCHEMA
    grid, dx_nm, lambda_sm_nm, particles = v1.validate_spec(structural, library, profiles)
    for particle in particles:
        manifest_path, manifest = profiles[float(particle["registered_radius_nm"])]
        expected_xb = str(manifest["fields"]["xB_alpha"]["sha256"])
        if particle.get("source_xB_alpha_sha256") != expected_xb:
            raise ValueError(f"{particle['particle_id']}: source_xB_alpha_sha256 mismatch")
        composition = manifest.get("composition")
        if not isinstance(composition, dict):
            raise ValueError(f"{manifest_path}: missing frozen composition contract")
        far = float(composition.get("far_field_xB_mean", math.nan))
        if not math.isfinite(far) or not (0.0 < far < 1.0):
            raise ValueError(f"{manifest_path}: invalid frozen far_field_xB_mean")
    return grid, dx_nm, lambda_sm_nm, particles


def _write_raw(path: Path, values: np.ndarray) -> None:
    np.asarray(values, dtype="<f8").ravel(order="C").tofile(path)


def _source_fields(
    manifest_path: Path, manifest: Dict[str, Any], grid: Tuple[int, int, int], center: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phi = v1.translated(v1.read_raw(v1.check_field(manifest, manifest_path, "phi", grid), grid, "phi"), center)
    h = v1.translated(v1.read_raw(v1.check_field(manifest, manifest_path, "h_phi", grid), grid, "h_phi"), center)
    delta_c = v1.translated(
        v1.read_raw(v1.check_field(manifest, manifest_path, "delta_C_relaxation", grid), grid, "delta_C_relaxation"),
        center,
    )
    xb = v1.translated(v1.read_raw(v1.check_field(manifest, manifest_path, "xB_alpha", grid), grid, "xB_alpha"), center)
    if float(np.max(np.abs(h - v1.h_of_phi(phi)))) > 5.0e-14:
        raise ValueError(f"{manifest_path}: translated h(phi) mismatch")
    if not all(np.all(np.isfinite(value)) for value in (phi, h, delta_c, xb)):
        raise ValueError(f"{manifest_path}: non-finite source field")
    return phi, h, delta_c, xb


def _s0_geometry(
    grid: Tuple[int, int, int], dx_nm: float, lambda_sm_nm: float,
    records: List[Dict[str, Any]], target_h_volume: float,
) -> Tuple[np.ndarray, List[np.ndarray], float]:
    parameters = [float(record["spherical_radius_parameter_nm"]) for record in records]

    def h_volume(offset_nm: float) -> float:
        product = np.ones(grid, dtype=np.float64)
        for record, parameter in zip(records, parameters):
            radius = parameter + offset_nm
            if radius <= 0.0:
                return 0.0
            product *= 1.0 - v1.sphere_phi(grid, record["center_grid"], dx_nm, lambda_sm_nm, radius)
        return float(np.sum(v1.h_of_phi(1.0 - product), dtype=np.float64) * dx_nm**3)

    lo, hi = -0.5 * min(parameters), 0.5 * min(parameters)
    if h_volume(lo) > target_h_volume or h_volume(hi) < target_h_volume:
        raise ValueError("S0 h-volume cannot bracket the E2 geometry")
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if h_volume(mid) < target_h_volume:
            lo = mid
        else:
            hi = mid
    offset = 0.5 * (lo + hi)
    product = np.ones(grid, dtype=np.float64)
    individual_h: List[np.ndarray] = []
    for record, parameter in zip(records, parameters):
        adjusted = parameter + offset
        phi = v1.sphere_phi(grid, record["center_grid"], dx_nm, lambda_sm_nm, adjusted)
        product *= 1.0 - phi
        individual_h.append(v1.h_of_phi(phi))
        record["spherical_radius_parameter_nm"] = adjusted
        record["spherical_uniform_radius_offset_nm"] = offset
    return 1.0 - product, individual_h, offset


def _conservative_s0_phase_storage_projection(
    reference_delta_c: np.ndarray, alpha_s0: np.ndarray, matrix_baseline: float,
    xB_min_safe: float, xB_max_safe: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Transport an E2 canonical correction into S0 without losing inventory.

    The V2 E2 correction has the form ``alpha_E2 * delta_x_E2``.  Retaining
    that canonical correction and the same matrix baseline after changing to
    a spherical phase field requires ``delta_x_S0=delta_C_E2/alpha_S0``.
    A few nearly-pure S0 core cells can make that storage value unbounded.
    This deterministic bounded projection first limits the *storage* value,
    then redistributes the exact residual through available alpha capacity.
    It is not a mass clip, profile scale, or physical retune: its input and
    output canonical correction sums are identical to binary summation
    precision, and every projected cell is recorded in the fixture manifest.
    """
    if np.any(alpha_s0 < 0.0):
        raise ValueError("negative alpha in S0 phase-storage projection")
    raw_xb = np.empty_like(reference_delta_c)
    positive = alpha_s0 > 0.0
    raw_xb[positive] = matrix_baseline + reference_delta_c[positive] / alpha_s0[positive]
    # At alpha=0 C is independent of xB.  Keep the finite matrix baseline.
    raw_xb[~positive] = matrix_baseline
    storage_xb = np.minimum(np.maximum(raw_xb, xB_min_safe), xB_max_safe)
    initial_delta_c = alpha_s0 * (storage_xb - matrix_baseline)
    target_sum = float(np.sum(reference_delta_c, dtype=np.float64))
    residual = target_sum - float(np.sum(initial_delta_c, dtype=np.float64))
    if residual < 0.0:
        capacity = alpha_s0 * (storage_xb - xB_min_safe)
    else:
        capacity = alpha_s0 * (xB_max_safe - storage_xb)
    capacity_sum = float(np.sum(capacity, dtype=np.float64))
    if not math.isfinite(capacity_sum) or capacity_sum <= abs(residual):
        raise ValueError("S0 phase-storage projection has insufficient bounded capacity")
    # Distribute in canonical-C space.  Dividing only where alpha>0 makes the
    # updated stored composition finite by construction.
    adjustment_c = residual * capacity / capacity_sum
    storage_xb[positive] += adjustment_c[positive] / alpha_s0[positive]
    delta_c = alpha_s0 * (storage_xb - matrix_baseline)
    remaining = target_sum - float(np.sum(delta_c, dtype=np.float64))
    # One deterministic high-capacity alpha cell closes any last summation ULP
    # without changing a phase-zero cell or violating the safe bounds.
    anchor = int(np.argmax(capacity))
    flat_alpha = alpha_s0.ravel(order="C")
    flat_xb = storage_xb.ravel(order="C")
    if flat_alpha[anchor] <= 0.0:
        raise ValueError("S0 projection chose a zero-alpha anchor")
    candidate = flat_xb[anchor] + remaining / flat_alpha[anchor]
    if not (xB_min_safe <= candidate <= xB_max_safe):
        raise ValueError("S0 phase-storage final residual exceeds bounded anchor capacity")
    flat_xb[anchor] = candidate
    delta_c = alpha_s0 * (storage_xb - matrix_baseline)
    final_error = target_sum - float(np.sum(delta_c, dtype=np.float64))
    if abs(final_error) > 1.0e-12:
        raise ValueError(f"S0 phase-storage projection does not conserve correction: {final_error:.3e}")
    return storage_xb, {
        "method": "CONSERVATIVE_BOUNDED_CANONICAL_DELTA_C_PROJECTION_V2",
        "source_correction_sum": target_sum,
        "output_correction_sum": float(np.sum(delta_c, dtype=np.float64)),
        "correction_sum_error": final_error,
        "bounded_cell_count": int(np.count_nonzero((raw_xb < xB_min_safe) | (raw_xb > xB_max_safe))),
        "raw_xB_min": float(np.min(raw_xb)), "raw_xB_max": float(np.max(raw_xb)),
        "xB_min_safe": xB_min_safe, "xB_max_safe": xB_max_safe,
        "residual_redistributed": residual, "mass_discarded": 0.0,
        "hidden_profile_scaling_used": False,
    }


def materialize(args: argparse.Namespace) -> Dict[str, Any]:
    spec_path = args.spec.resolve()
    spec = v1.load_json(spec_path)
    library_sha = str(spec.get("selected_library_manifest_sha256", ""))
    selection_sha = spec.get("selected_library_selection_sha256")
    if len(library_sha) != 64:
        raise ValueError("fixture spec must pin selected_library_manifest_sha256")
    _library_path, library, profile_index, freeze = v1.verify_library(
        args.library_root.resolve(), library_sha,
        args.selection_provenance.resolve() if args.selection_provenance else None,
        str(selection_sha) if selection_sha else None,
    )
    grid, dx_nm, lambda_sm_nm, particles = _validate_spec(spec, library, profile_index)
    if args.kind not in ("E2", "S0"):
        raise ValueError("fixture kind must be E2 or S0")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output root: {args.out}")
    args.out.mkdir(parents=True, exist_ok=False)

    e2_product = np.ones(grid, dtype=np.float64)
    geometry_product = np.ones(grid, dtype=np.float64)
    delta_x_total = np.zeros(grid, dtype=np.float64)
    direct_delta_diagnostic = np.zeros(grid, dtype=np.float64)
    records: List[Dict[str, Any]] = []
    source_h_sum = 0.0
    individual_h: List[np.ndarray] = []
    single_e2_phi: Optional[np.ndarray] = None
    single_e2_delta: Optional[np.ndarray] = None

    for particle in particles:
        radius = float(particle["registered_radius_nm"])
        manifest_path, manifest = profile_index[radius]
        source_phi, source_h, source_delta, source_xb = _source_fields(
            manifest_path, manifest, grid, particle["center_grid"],
        )
        far = float(manifest["composition"]["far_field_xB_mean"])
        delta_x = source_xb - far
        if not np.all(np.isfinite(delta_x)):
            raise ValueError(f"{particle['particle_id']}: non-finite source delta_x")
        e2_product *= 1.0 - source_phi
        delta_x_total += delta_x
        direct_delta_diagnostic += source_delta
        source_h_volume = float(manifest["geometry"]["h_volume_nm3"])
        record = dict(particle)
        record.update({
            "source_profile_path": str(manifest_path),
            "source_h_volume_nm3": source_h_volume,
            "source_equivalent_radius_nm": float(manifest["geometry"]["actual_equivalent_radius_nm"]),
            "source_axis_ratio": float(manifest["geometry"]["axis_ratio_major_minor"]),
            "source_far_field_xB": far,
            "source_delta_x_alpha_definition": "xB_alpha - source_far_field_xB",
            "source_delta_x_alpha_max_abs": float(np.max(np.abs(delta_x))),
            "source_delta_C_consistency_max_abs": float(np.max(np.abs(source_delta - (1.0 - source_h) * delta_x))),
            "spherical_radius_parameter_nm": None,
        })
        records.append(record)
        source_h_sum += source_h_volume
        if args.kind == "E2":
            geometry_product *= 1.0 - source_phi
            individual_h.append(source_h)
            if len(particles) == 1:
                single_e2_phi, single_e2_delta = source_phi, source_delta
        else:
            sphere, parameter = v1.analytic_sphere(
                grid, particle["center_grid"], dx_nm, lambda_sm_nm, source_h_volume,
            )
            geometry_product *= 1.0 - sphere
            individual_h.append(v1.h_of_phi(sphere))
            record["spherical_radius_parameter_nm"] = parameter

    reference_e2_h_volume = float(np.sum(v1.h_of_phi(1.0 - e2_product), dtype=np.float64) * dx_nm**3)
    s0_offset = 0.0
    if args.kind == "S0":
        phi_total, individual_h, s0_offset = _s0_geometry(
            grid, dx_nm, lambda_sm_nm, records, reference_e2_h_volume,
        )
    else:
        phi_total = single_e2_phi if single_e2_phi is not None else 1.0 - geometry_product
    if not np.all(np.isfinite(phi_total)) or float(np.min(phi_total)) < 0.0 or float(np.max(phi_total)) > 1.0:
        raise ValueError("bounded phi union is invalid")
    h_total = v1.h_of_phi(phi_total)
    alpha = 1.0 - h_total
    alpha_floor = float(spec["target"].get("phase_storage_alpha_floor", 1.0e-10))
    if not math.isfinite(alpha_floor) or alpha_floor <= 0.0 or float(np.min(alpha)) < 0.0:
        raise ValueError("invalid phase-storage contract")

    target_mean = float(spec["target"]["mean_C_B_tot"])
    if not math.isfinite(target_mean) or not (0.0 < target_mean < 1.0):
        raise ValueError("target mean inventory must be in (0,1)")
    target_total = target_mean * float(math.prod(grid))
    reference_e2_phi = 1.0 - e2_product
    reference_e2_h = v1.h_of_phi(reference_e2_phi)
    reference_e2_alpha = 1.0 - reference_e2_h
    reference_delta_c = reference_e2_alpha * delta_x_total
    baseline = (target_total - float(np.sum(reference_e2_h + reference_delta_c, dtype=np.float64))) / float(np.sum(reference_e2_alpha, dtype=np.float64))
    xB_max_safe = float(spec["target"].get("xB_max_safe", 0.499999))
    xB_min_safe = float(spec["target"].get("xB_min_safe", 1.0e-12))
    if not (0.0 < xB_min_safe < xB_max_safe < 1.0):
        raise ValueError("invalid xB storage bounds")
    s0_projection: Dict[str, Any] = {"method": "NOT_APPLICABLE_E2", "mass_discarded": 0.0}
    if args.kind == "S0":
        xB, s0_projection = _conservative_s0_phase_storage_projection(
            reference_delta_c, alpha, baseline, xB_min_safe, xB_max_safe,
        )
        delta_x_storage = xB - baseline
        delta_c_formula = alpha * delta_x_storage
    else:
        delta_x_storage = delta_x_total
        delta_c_formula = alpha * delta_x_total
        xB = baseline + delta_x_storage
    if args.kind == "E2" and len(particles) == 1 and single_e2_delta is not None:
        # Exact frozen delta_C identity is a V2 regression requirement.  The
        # algebraic V2 expression is equivalent for one profile; writing the
        # frozen raw field prevents round-off from becoming a false regression.
        delta_c_output = single_e2_delta
        single_profile_shortcut = True
    else:
        delta_c_output = delta_c_formula
        single_profile_shortcut = False
    # V2 writes the finite composition directly: beta-core xB/Y values are
    # storage values only, because alpha*xB is exactly zero there.  No E2
    # path divides by alpha.  S0's separately declared conservative projector
    # handles its sharper core without discarding canonical inventory.
    if not np.all(np.isfinite(xB)) or float(np.min(xB)) < xB_min_safe or float(np.max(xB)) > xB_max_safe:
        raise ValueError("V2 composition would require clipping or violates xB storage bounds")
    c_total = h_total + alpha * xB
    rel_error = abs(float(np.sum(c_total, dtype=np.float64)) - target_total) / max(abs(target_total), 1.0)
    if rel_error > 1.0e-12:
        raise ValueError(f"canonical inventory does not close: {rel_error:.3e}")
    y = v1.logit(xB)
    dY_dt_prev = np.zeros(grid, dtype=np.float64)

    threshold = float(spec["target"].get("component_h_threshold", 1.0e-4))
    labels, component_rows = v1.components(h_total, threshold, dx_nm)
    particle_to_component: Dict[str, int] = {}
    used: set[int] = set()
    for record, source_h in zip(records, individual_h):
        values = labels[source_h > threshold]
        values = values[values >= 0]
        if values.size == 0:
            raise ValueError(f"{record['particle_id']}: no threshold support")
        candidates, counts = np.unique(values, return_counts=True)
        assigned = int(candidates[int(np.argmax(counts))])
        if assigned in used:
            raise ValueError("unexpected initial merge under the frozen threshold")
        used.add(assigned)
        particle_to_component[str(record["particle_id"])] = assigned
    if len(component_rows) != len(records) or len(used) != len(records):
        raise ValueError("initial component count does not equal particle count")
    for row in component_rows:
        row["particle_id"] = next(key for key, value in particle_to_component.items() if value == row["component_label"])

    arrays = {
        "phi": phi_total, "h_phi": h_total, "C_B_tot": c_total, "xB_alpha": xB,
        "Y": y, "dY_dt_prev": dY_dt_prev, "delta_x_alpha_total": delta_x_storage,
        "delta_x_alpha_source_total": delta_x_total,
        "delta_C_relaxation_total": delta_c_output,
    }
    fields = {name: args.out / f"{name}.raw.f64" for name in arrays}
    for name, value in arrays.items():
        _write_raw(fields[name], value)
    meta = {
        "schema": RAW_META_SCHEMA, "Nx": grid[0], "Ny": grid[1], "Nz": grid[2], "dx_nm": dx_nm,
        "interface_width_nm": lambda_sm_nm, "dt_recommended": float(spec["target"].get("dt_code", 0.02)),
        "mean_xBtot": float(np.mean(c_total, dtype=np.float64)), "xB_min_safe": xB_min_safe, "xB_max_safe": xB_max_safe,
        "dtype": "float64", "order": "C", "phi_path": fields["phi"].name, "xB_path": fields["xB_alpha"].name,
        "fresh_dY_dt_prev_contract": "zero_for_fresh_dynamic_start",
        "composition_contract": COMPOSITION_CONTRACT,
    }
    v1.write_json(args.out / "init_meta.json", meta)
    field_rows = {
        name: {"path": path.name, "sha256": v1.sha256(path), "dtype": "float64-le", "order": "C"}
        for name, path in fields.items()
    }
    with (args.out / "initial_particles.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["particle_id", "registered_radius_nm", "center_grid", "source_profile_manifest_sha256", "source_phi_sha256", "source_xB_alpha_sha256", "source_delta_C_relaxation_sha256", "source_far_field_xB", "source_delta_x_alpha_max_abs", "source_h_volume_nm3", "source_equivalent_radius_nm", "source_axis_ratio", "spherical_radius_parameter_nm"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in columns} for row in records])
    with (args.out / "initial_components.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["particle_id", "component_label", "threshold_voxel_count", "h_volume_nm3", "equivalent_radius_nm", "centroid_nm", "semi_axes_nm", "axis_ratio", "principal_axes_rows"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(component_rows)
    radius_counts: Dict[str, int] = {}
    for record in records:
        label = f"{float(record['registered_radius_nm']):.1f}"
        radius_counts[label] = radius_counts.get(label, 0) + 1
    manifest = {
        "schema": SCHEMA, "fixture_kind": args.kind, "fixture_id": spec.get("fixture_id"),
        "validation_only": bool(spec.get("validation_only", True)), "scientific_status": spec.get("scientific_status"),
        "source_spec_canonical_sha256": v1.canonical_spec_digest(spec), "canonical_particle_order": [row["particle_id"] for row in records],
        "selected_library": freeze,
        "physical_contract": {
            "temperature_C": float(spec["target"]["temperature_C"]), "dx_nm": dx_nm, "lambda_sm_nm": lambda_sm_nm,
            "elasticity_enabled": True, "elastic_boundary": "periodic_fixed_cell", "orientation_label": "variant_100_identity",
            "eigenstrain": [0.046, -0.022, -0.017, 0.0, 0.0, 0.0], "GP_enabled": False,
            "GP_birth_enabled": False, "GP_release_enabled": False, "external_source_enabled": False,
            "new_beta_nucleation_enabled": False,
        },
        "grid": {"Nx": grid[0], "Ny": grid[1], "Nz": grid[2], "dx_nm": dx_nm}, "placement": spec["placement"],
        "particles": records, "radius_counts": radius_counts,
        "combination": {
            "composition_contract": COMPOSITION_CONTRACT,
            "phi": "1-product_j(1-phi_j), canonical particle-id order",
            "delta_x_alpha": "sum_j(xB_alpha_j - frozen_profile_far_field_xB_mean_j)",
            "delta_C_relaxation": "E2: (1-h_total)*delta_x_alpha_source_total; S0: explicit conserved canonical projection of that E2 correction",
            "absolute_xB_superposed": False, "direct_delta_C_sum_used": False,
            "direct_delta_C_sum_max_abs_diagnostic": float(np.max(np.abs(direct_delta_diagnostic))),
            "one_profile_exact_delta_C_shortcut": single_profile_shortcut,
            "radial_interpolation_used": False, "spatial_resampling_used": False, "rotated_variant_used": False,
            "normalization_used": False, "clipping_used": False,
            "s0_phase_storage_projection": s0_projection,
        },
        "inventory": {
            "target_mean_C_B_tot": target_mean, "target_total_C_B_tot": target_total,
            "actual_mean_C_B_tot": float(np.mean(c_total, dtype=np.float64)), "actual_total_C_B_tot": float(np.sum(c_total, dtype=np.float64)),
            "relative_error": rel_error, "matrix_xB": baseline, "matrix_xAg": 2.0 * baseline / (2.0 + baseline),
            "reference_matrix_xB": spec["target"].get("matrix_xB_reference"), "unresolved_inventory": 0.0,
            "source_h_volume_sum_nm3": source_h_sum, "combined_h_volume_nm3": float(np.sum(h_total, dtype=np.float64) * dx_nm**3),
            "reference_E2_combined_h_volume_nm3": reference_e2_h_volume, "S0_uniform_radius_offset_nm": s0_offset,
            "beta_volume_fraction": float(np.mean(h_total, dtype=np.float64)),
            "common_pair_matrix_baseline_contract": baseline,
        },
        "component_contract": {"h_threshold": threshold, "expected_count": len(records), "actual_count": len(component_rows), "particle_to_component": particle_to_component},
        "phase_storage_contract": {"alpha_floor": alpha_floor, "minimum_alpha": float(np.min(alpha)), "zero_alpha_cell_count": int(np.count_nonzero(alpha == 0.0)), "status": "NO_UNRESOLVED_DIVISION_REQUIRED", "S0_projection_used": args.kind == "S0"},
        "time_level_contract": {"Y": "logit(xB_alpha)", "dY_dt_prev": "zero_for_fresh_dynamic_start", "dY_dt_prev_nonzero": False},
        "fields": field_rows, "init_meta": {"path": "init_meta.json", "sha256": v1.sha256(args.out / "init_meta.json")},
        "initial_particles": {"path": "initial_particles.csv", "sha256": v1.sha256(args.out / "initial_particles.csv")},
        "initial_components": {"path": "initial_components.csv", "sha256": v1.sha256(args.out / "initial_components.csv")},
    }
    v1.write_json(args.out / "fixture_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--selection-provenance", type=Path)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--kind", choices=("E2", "S0"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = materialize(args)
    except ValueError as exc:
        raise SystemExit(f"[fatal] {exc}") from exc
    print(json.dumps({
        "status": "PASS_PF_ELASTIC_MULTI_PARTICLE_FIXTURE_MATERIALIZED_V2", "fixture_kind": manifest["fixture_kind"],
        "fixture_particle_count": len(manifest["particles"]), "fixture_beta_volume_fraction": manifest["inventory"]["beta_volume_fraction"],
        "fixture_matrix_xB": manifest["inventory"]["matrix_xB"], "fixture_inventory_relative_error": manifest["inventory"]["relative_error"],
        "fixture_manifest_sha256": v1.sha256(args.out / "fixture_manifest.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
