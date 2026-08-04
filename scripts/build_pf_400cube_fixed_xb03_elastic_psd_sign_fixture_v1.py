#!/usr/bin/env python3
"""Materialize the independent 400-cube fixed-inventory elastic PSD pilot.

This is deliberately separate from the 246-cube production materializers.  It
only embeds complete, hash-pinned elastic target profiles at the native grid
declared by the fixture specification.  No profile is scaled, interpolated,
rotated, or replaced by an analytic sphere.  The matrix baseline is solved
once from the assembled canonical inventory ledger; it is never projected
after field construction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import materialize_pf_elastic_multi_particle_6h_fixture_v1 as library_tools


SCHEMA = "PF_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_FIXTURE_V1"
META_SCHEMA = "PF_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_RAW_INIT_META_V1"
INITIAL_STATE_CLASS = "MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def h_of_phi(phi: np.ndarray) -> np.ndarray:
    """Evaluate the quintic interpolation without endpoint round-off.

    The direct polynomial can produce h=1+O(eps) at phi=1.  The complement
    form is exact at that pure-beta endpoint, so use it above the midpoint.
    This changes only floating-point evaluation, never clips a field.
    """
    direct = phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)
    q = 1.0 - phi
    complement = 1.0 - q**3 * (1.0 + 3.0 * phi + 6.0 * phi**2)
    return np.where(phi <= 0.5, direct, complement)


def xag_from_xb(xb: float) -> float:
    return 2.0 * xb / (2.0 + xb)


def periodic_distance(a: np.ndarray, b: np.ndarray, n: int) -> float:
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    delta = np.minimum(delta, float(n) - delta)
    return float(np.linalg.norm(delta))


def window_indices(center: np.ndarray, native_n: int, target_n: int) -> np.ndarray:
    offsets = np.arange(native_n, dtype=np.int64) - native_n // 2
    return (center[0] + offsets) % target_n, (center[1] + offsets) % target_n, (center[2] + offsets) % target_n


def read_profile_arrays(manifest_path: Path, manifest: dict[str, Any], native: tuple[int, int, int]) -> dict[str, np.ndarray]:
    phi_path = library_tools.check_field(manifest, manifest_path, "phi", native)
    h_path = library_tools.check_field(manifest, manifest_path, "h_phi", native)
    delta_path = library_tools.check_field(manifest, manifest_path, "delta_C_relaxation", native)
    phi = library_tools.read_raw(phi_path, native, "phi")
    h = library_tools.read_raw(h_path, native, "h_phi")
    delta_c = library_tools.read_raw(delta_path, native, "delta_C_relaxation")
    if float(np.max(np.abs(h - h_of_phi(phi)))) > 5.0e-14:
        raise ValueError(f"{manifest_path}: h(phi) identity mismatch")
    alpha = 1.0 - h
    matrix_support = alpha > 1.0e-12
    if float(np.max(np.abs(delta_c[~matrix_support]), initial=0.0)) > 1.0e-14:
        raise ValueError(f"{manifest_path}: nonzero matrix-excess field inside pure beta core")
    delta_x = np.zeros_like(delta_c)
    np.divide(delta_c, alpha, out=delta_x, where=matrix_support)
    return {"phi": phi, "h": h, "delta_x": delta_x}


def profile_support_radius(h: np.ndarray, threshold: float, dx_nm: float) -> float:
    n = h.shape[0]
    center = n // 2
    coords = np.indices(h.shape, dtype=np.int64)
    delta = (coords - center + n // 2) % n - n // 2
    radius = np.sqrt(np.sum(delta.astype(np.float64) ** 2, axis=0)) * dx_nm
    support = h > threshold
    if not bool(np.any(support)):
        raise ValueError("empty h-threshold profile support")
    return float(np.max(radius[support]))


def build_particles(spec: dict[str, Any]) -> list[dict[str, Any]]:
    histogram = spec["discrete_psd"]["registered_radius_histogram"]
    result: list[dict[str, Any]] = []
    for radius_text in sorted(histogram, key=float):
        for _ in range(int(histogram[radius_text])):
            result.append({"particle_id": f"P{len(result):03d}", "registered_radius_nm": float(radius_text)})
    if len(result) != int(spec["discrete_psd"]["target_particle_count"]):
        raise ValueError("frozen PSD histogram does not have the declared particle count")
    return result


def place_particles(particles: list[dict[str, Any]], support_radii: dict[float, float], spec: dict[str, Any]) -> list[dict[str, Any]]:
    n = int(spec["grid"]["Nx"])
    lam = float(spec["grid"]["lambda_sm_nm"])
    max_trials = int(spec["placement_contract"]["max_trials_per_particle"])
    seed = int(spec["placement_contract"]["seed_unsigned64"])
    rng = np.random.Generator(np.random.PCG64(seed))
    placed: list[dict[str, Any]] = []
    for row in particles:
        radius = float(row["registered_radius_nm"])
        candidate: np.ndarray | None = None
        for _ in range(max_trials):
            trial = rng.integers(0, n, size=3, dtype=np.int64)
            good = True
            for other in placed:
                other_radius = float(other["registered_radius_nm"])
                distance = periodic_distance(trial, np.asarray(other["center_grid"], dtype=np.int64), n)
                declared_min = radius + other_radius + 4.0 * lam
                support_min = support_radii[radius] + support_radii[other_radius]
                if distance <= max(declared_min, support_min):
                    good = False
                    break
            if good:
                candidate = trial
                break
        if candidate is None:
            raise ValueError(f"hard-core placement failed for {row['particle_id']}")
        row = dict(row)
        row["center_grid"] = [int(value) for value in candidate]
        placed.append(row)
    return placed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"[fatal] refusing to overwrite fixture root: {args.out}")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if spec.get("schema") != "PF_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_FIXTURE_SPEC_V1":
        raise SystemExit("[fatal] wrong fixture spec schema")
    grid_data = spec["grid"]
    grid = tuple(int(grid_data[key]) for key in ("Nx", "Ny", "Nz"))
    if grid != (400, 400, 400) or float(grid_data["dx_nm"]) != 1.0:
        raise SystemExit("[fatal] this independent materializer only accepts the frozen 400^3/dx=1 contract")
    expected_radii = [float(value) for value in spec["profile_library_contract"]["expected_radii_nm"]]
    native = tuple(int(value) for value in spec["profile_library_contract"]["native_grid"])
    library_manifest_path, library, profile_index, _identity = library_tools.verify_library(
        args.library_root.resolve(),
        library_tools.sha256(library_tools.native_library_paths(args.library_root.resolve())[0]),
        None,
        None,
        expected_radii,
        native,
    )
    if tuple(int(library["grid"][key]) for key in ("Nx", "Ny", "Nz")) != native:
        raise SystemExit("[fatal] profile library native grid differs from fixture contract")
    if abs(float(library["lambda_sm_nm"]) - float(grid_data["lambda_sm_nm"])) > 1.0e-14:
        raise SystemExit("[fatal] profile library interface width differs from fixture contract")
    profiles: dict[float, dict[str, Any]] = {}
    threshold = float(spec["placement_contract"]["h_threshold"])
    for radius in expected_radii:
        manifest_path, manifest = profile_index[radius]
        arrays = read_profile_arrays(manifest_path, manifest, native)
        profiles[radius] = {
            **arrays,
            "manifest": manifest,
            "manifest_path": manifest_path,
            "manifest_sha256": sha256(manifest_path),
            "support_radius_nm": profile_support_radius(arrays["h"], threshold, 1.0),
        }
    particles = place_particles(build_particles(spec), {key: float(value["support_radius_nm"]) for key, value in profiles.items()}, spec)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.mkdir()
    complement = np.ones(grid, dtype=np.float64)
    delta_x_total = np.zeros(grid, dtype=np.float64)
    support_owner = np.zeros(grid, dtype=np.uint8)
    source_h_volume_sum = 0.0
    particle_rows: list[dict[str, Any]] = []
    for row in particles:
        radius = float(row["registered_radius_nm"])
        profile = profiles[radius]
        ix = np.ix_(*window_indices(np.asarray(row["center_grid"], dtype=np.int64), native[0], grid[0]))
        owner_window = support_owner[ix]
        support = profile["h"] > threshold
        if bool(np.any(owner_window[support])):
            raise SystemExit(f"[fatal] source support overlap at {row['particle_id']}")
        owner_window[support] = 1
        support_owner[ix] = owner_window
        local = complement[ix]
        local *= 1.0 - profile["phi"]
        complement[ix] = local
        delta = delta_x_total[ix]
        delta += profile["delta_x"]
        delta_x_total[ix] = delta
        geometry = profile["manifest"]["geometry"]
        h_volume = float(geometry.get("final_h_volume_nm3", geometry["h_volume_nm3"]))
        source_h_volume_sum += h_volume
        particle_rows.append({
            **row,
            "orientation_label": "variant_100_identity",
            "source_profile_manifest_path": str(profile["manifest_path"]),
            "source_profile_manifest_sha256": profile["manifest_sha256"],
            "source_phi_sha256": profile["manifest"]["fields"]["phi"]["sha256"],
            "source_delta_C_relaxation_sha256": profile["manifest"]["fields"]["delta_C_relaxation"]["sha256"],
            "source_h_volume_nm3": h_volume,
            "source_equivalent_radius_nm": float(geometry["actual_equivalent_radius_nm"]),
            "source_support_radius_nm": float(profile["support_radius_nm"]),
        })
    phi = 1.0 - complement
    del complement, support_owner
    if not np.all(np.isfinite(phi)) or float(np.min(phi)) < 0.0 or float(np.max(phi)) > 1.0:
        raise SystemExit("[fatal] bounded profile union violated phi bounds")
    h = h_of_phi(phi)
    alpha = 1.0 - h
    if float(np.min(alpha)) < -1.0e-14:
        raise SystemExit("[fatal] h(phi) escaped its bounded interpolation range")
    v_b = float(library["v_B"])
    target_mean = float(spec["inventory_contract"]["mean_C_B_tot"])
    target_total = target_mean * math.prod(grid)
    baseline = (target_total - float(np.sum(v_b * h + alpha * delta_x_total, dtype=np.float64))) / float(np.sum(alpha, dtype=np.float64))
    x_b = baseline + delta_x_total
    if not np.all(np.isfinite(x_b)) or float(np.min(x_b)) <= 0.0 or float(np.max(x_b)) >= 0.499999:
        raise SystemExit("[fatal] composition reconstruction requires clipping or leaves physical bounds")
    c_total = v_b * h + alpha * x_b
    inventory_relative_error = abs(float(np.sum(c_total, dtype=np.float64)) - target_total) / max(abs(target_total), 1.0)
    if inventory_relative_error > 1.0e-12:
        raise SystemExit(f"[fatal] canonical inventory did not close: {inventory_relative_error:.3e}")
    beta_h_volume = float(np.sum(h, dtype=np.float64))
    target_beta_volume = float(spec["inventory_contract"]["target_beta_inventory_nm3"])
    beta_relative_error = abs(beta_h_volume - target_beta_volume) / target_beta_volume
    if beta_relative_error > 1.0e-4:
        raise SystemExit(f"[fatal] discrete beta inventory misses frozen target: {beta_relative_error:.3e}")
    matrix_mask = h < threshold
    if not bool(np.any(matrix_mask)):
        raise SystemExit("[fatal] no far-field matrix voxels at frozen threshold")
    matrix_xb = float(np.mean(x_b[matrix_mask], dtype=np.float64))
    matrix_xag = xag_from_xb(matrix_xb)
    lo, hi = (float(value) for value in spec["inventory_contract"]["matrix_xAg_allowed_interval"])
    if not lo <= matrix_xag <= hi:
        raise SystemExit(f"[fatal] assembled far-field xAg={matrix_xag:.9g} outside [{lo}, {hi}]")
    y = np.log(x_b / (1.0 - x_b))
    fields = {
        "phi": phi,
        "h_phi": h,
        "xB_alpha": x_b,
        "Y": y,
        "dY_dt_prev": np.zeros(grid, dtype=np.float64),
        "C_B_tot": c_total,
        "delta_C_relaxation_total": alpha * delta_x_total,
    }
    field_rows: dict[str, dict[str, str]] = {}
    for name, values in fields.items():
        path = args.out / f"{name}.raw.f64"
        np.asarray(values, dtype="<f8").ravel(order="C").tofile(path)
        field_rows[name] = {"path": path.name, "sha256": sha256(path), "dtype": "float64-le", "order": "C"}
    init_meta = {
        "schema": META_SCHEMA,
        "Nx": 400, "Ny": 400, "Nz": 400, "dx_nm": 1.0,
        "interface_width_nm": float(grid_data["lambda_sm_nm"]),
        "dt_recommended": float(spec["time_contract"]["dt_code"]),
        "mean_xBtot": target_mean, "xB_max_safe": 0.499999,
        "dtype": "float64", "order": "C",
        "phi_path": field_rows["phi"]["path"], "xB_path": field_rows["xB_alpha"]["path"],
        "fresh_dY_dt_prev_contract": "zero_for_fresh_dynamic_start",
    }
    write_json(args.out / "init_meta.json", init_meta)
    with (args.out / "initial_particles.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = list(particle_rows[0])
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(particle_rows)
    for left, row in enumerate(particle_rows):
        for other in particle_rows[left + 1:]:
            distance = periodic_distance(np.asarray(row["center_grid"]), np.asarray(other["center_grid"]), 400)
            required = float(row["registered_radius_nm"]) + float(other["registered_radius_nm"]) + 4.0 * float(grid_data["lambda_sm_nm"])
            if distance <= required:
                raise SystemExit("[fatal] final placement violates declared periodic separation")
    manifest = {
        "schema": SCHEMA,
        "fixture_id": spec["fixture_id"],
        "initial_state_class": INITIAL_STATE_CLASS,
        "scientific_semantics": "conditional 6h narrow-PSD positive-kappa-increment feasibility pilot; not an experimental PSD fit",
        "validation_only": False,
        "spec_path": str(args.spec.resolve()), "spec_sha256": sha256(args.spec),
        "profile_library_manifest_path": str(library_manifest_path),
        "profile_library_manifest_sha256": sha256(library_manifest_path),
        "profile_library_source_tree_sha256": library.get("source_tree_sha256"),
        "profile_library_binary_sha256": library.get("binary_sha256"),
        "grid": grid_data,
        "physical_contract": spec["physical_contract"],
        "time_contract": spec["time_contract"],
        "placement": {**spec["placement_contract"], "canonical_particle_order": [row["particle_id"] for row in particle_rows]},
        "particle_library_mappings": particle_rows,
        "component_contract": {"h_threshold": threshold, "expected_count": len(particle_rows), "actual_count": len(particle_rows), "support_overlap_status": "PASS_NO_OVERLAP"},
        "initial_canonical_inventory": {
            "target_mean_C_B_tot": target_mean, "target_total_code": target_total,
            "actual_total_code": float(np.sum(c_total, dtype=np.float64)), "field_relative_error": inventory_relative_error,
            "status": "PASS_MACHINE_PRECISION",
        },
        "initial_beta_inventory": {
            "target_beta_inventory_nm3": target_beta_volume, "assembled_beta_h_volume_nm3": beta_h_volume,
            "source_profile_h_volume_sum_nm3": source_h_volume_sum, "relative_error": beta_relative_error,
        },
        "derived_matrix_baseline": {
            "xB_alpha": baseline, "xAg": xag_from_xb(baseline),
            "observed_far_field_xB_h_lt_1e-4": matrix_xb, "observed_far_field_xAg_h_lt_1e-4": matrix_xag,
            "target_xAg": float(spec["inventory_contract"]["matrix_xAg_target"]),
        },
        "assembly_contract": {
            "phi": "1-product_j(1-phi_j), canonical particle-id order",
            "portable_composition_field": "native delta_C_relaxation/(1-h) embedded without resampling; one matrix baseline closes the exact canonical ledger",
            "profile_scaling_used": False, "profile_interpolation_used": False,
            "profile_rotation_used": False, "analytic_profile_used": False,
            "clipping_used": False, "normalization_used": False,
            "common_multi_particle_equilibrium_claim": False,
            "initial_relaxation_is_physical_evolution": True,
        },
        "initial_zero_mode_provenance": {
            "zero_mode": "PF_CONSERVED_Y_ZERO_MODE_V1", "backend": "HOST_NEWTON_BISECTION_V1",
            "explicit_context": "SM_EXPLICIT_CONTEXT_N_V1", "reaction_discretization": "SM_TANGENT_N_V1",
            "target_mass_source": "initial_canonical_inventory", "dY_dt_prev": "zero_for_fresh_dynamic_start",
        },
        "field_bounds": {"phi_min": float(np.min(phi)), "phi_max": float(np.max(phi)), "xB_min": float(np.min(x_b)), "xB_max": float(np.max(x_b))},
        "fields": field_rows,
    }
    write_json(args.out / "fixture_manifest.json", manifest)
    manifest_sha = sha256(args.out / "fixture_manifest.json")
    (args.out / "fixture_hashes.sha256").write_text("\n".join(f"{row['sha256']}  {name}" for name, row in sorted(field_rows.items())) + f"\n{manifest_sha}  fixture_manifest.json\n", encoding="utf-8")
    (args.out / "status.txt").write_text("PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_FIXTURE_V1\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_FIXTURE_V1", "fixture_manifest_sha256": manifest_sha, "particle_count": len(particle_rows), "mean_C_Btot": float(np.mean(c_total)), "far_field_xAg": matrix_xag, "beta_h_volume_nm3": beta_h_volume}, sort_keys=True))


if __name__ == "__main__":
    main()
