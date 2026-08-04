#!/usr/bin/env python3
"""Materialize one V2 400-cube native-profile conditional handoff fixture.

This program is deliberately separate from the running V5 pilot.  Its only
input shape is an immutable case specification from the V2 integer planner.
It combines full, hash-pinned 96^3 elastic target profiles using the accepted
bounded union, then derives exactly one matrix baseline from the canonical
inventory ledger.  No radius/profile interpolation, scaling, clipping, or
case-specific physical parameter is permitted.
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


SCHEMA = "PF_400CUBE_PSD_SPATIAL_DENSITY_INTEGER_GATE_FIXTURE_V2"
META_SCHEMA = "PF_400CUBE_PSD_SPATIAL_DENSITY_INTEGER_GATE_RAW_INIT_META_V2"
INITIAL_STATE_CLASS = "MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1"
TARGET_H = 1531490.9025410344
H_REL_TOL = 1.0e-7
TARGET_MEAN = 0.03
THRESHOLD = 1.0e-4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def h_of_phi(phi: np.ndarray) -> np.ndarray:
    direct = phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)
    q = 1.0 - phi
    complement = 1.0 - q**3 * (1.0 + 3.0 * phi + 6.0 * phi**2)
    return np.where(phi <= 0.5, direct, complement)


def xag_from_xb(xb: float) -> float:
    return 2.0 * xb / (2.0 + xb)


def periodic_delta(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    delta = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return np.minimum(delta, float(n) - delta)


def periodic_distance(a: np.ndarray, b: np.ndarray, n: int) -> float:
    return float(np.linalg.norm(periodic_delta(a, b, n)))


def window_indices(center: np.ndarray, native_n: int, target_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.arange(native_n, dtype=np.int64) - native_n // 2
    return tuple((int(center[axis]) + offsets) % target_n for axis in range(3))  # type: ignore[return-value]


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
        raise ValueError(f"{manifest_path}: pure-beta core has nonzero matrix delta_C")
    delta_x = np.zeros_like(delta_c)
    np.divide(delta_c, alpha, out=delta_x, where=matrix_support)
    return {"phi": phi, "h": h, "delta_x": delta_x}


def support_radius_nm(h: np.ndarray, threshold: float) -> float:
    n = h.shape[0]
    center = n // 2
    coords = np.indices(h.shape, dtype=np.int64)
    delta = (coords - center + n // 2) % n - n // 2
    radius = np.sqrt(np.sum(delta.astype(np.float64) ** 2, axis=0))
    mask = h > threshold
    if not bool(np.any(mask)):
        raise ValueError("profile has empty h-threshold support")
    return float(np.max(radius[mask]))


def deterministic_centers(count: int, seed: int, n: int, minimum_distance: float, max_trials: int) -> list[np.ndarray]:
    """Generate centers using only a count/replicate seed, never the PSD.

    The same N/replicate center set is consequently paired across the four
    N=512 PSD families.  The final actual-radius/support gate is stricter and
    checked independently after profile assignment.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    centers: list[np.ndarray] = []
    for particle_index in range(count):
        for _ in range(max_trials):
            candidate = rng.integers(0, n, size=3, dtype=np.int64)
            if all(periodic_distance(candidate, existing, n) > minimum_distance for existing in centers):
                centers.append(candidate)
                break
        else:
            raise RuntimeError(f"periodic hard-core placement failed at particle {particle_index}/{count}")
    return centers


def deterministic_particles(spec: dict[str, Any], profiles: dict[float, dict[str, Any]]) -> list[dict[str, Any]]:
    psd = spec["discrete_psd"]
    count = int(psd["particle_count"])
    repeated: list[float] = []
    for radius_text, number in sorted(psd["registered_radius_histogram"].items(), key=lambda item: float(item[0])):
        radius = float(radius_text)
        if radius not in profiles:
            raise ValueError(f"spec requests radius absent from verified library: {radius}")
        repeated.extend([radius] * int(number))
    if len(repeated) != count:
        raise ValueError("integer histogram does not close declared particle count")
    rng = np.random.Generator(np.random.PCG64(int(spec["placement_contract"]["assignment_seed_unsigned64"])))
    assignment = np.asarray(repeated, dtype=np.float64)
    rng.shuffle(assignment)
    centers = deterministic_centers(
        count, int(spec["placement_contract"]["center_seed_unsigned64"]), int(spec["grid"]["Nx"]),
        float(spec["placement_contract"]["minimum_declared_center_distance_nm"]), int(spec["placement_contract"]["max_trials_per_particle"]),
    )
    result: list[dict[str, Any]] = []
    for index, (radius, center) in enumerate(zip(assignment, centers)):
        result.append({"particle_id": f"P{index:04d}", "registered_radius_nm": float(radius), "center_grid": [int(value) for value in center]})
    return result


def validate_particle_separation(particles: list[dict[str, Any]], profiles: dict[float, dict[str, Any]], n: int, lambda_nm: float) -> tuple[float, float]:
    nearest_center = float("inf")
    nearest_gap = float("inf")
    for left, particle in enumerate(particles):
        for other in particles[left + 1:]:
            distance = periodic_distance(np.asarray(particle["center_grid"]), np.asarray(other["center_grid"]), n)
            radius = float(particle["registered_radius_nm"])
            other_radius = float(other["registered_radius_nm"])
            declared = radius + other_radius + 4.0 * lambda_nm
            support = float(profiles[radius]["support_radius_nm"]) + float(profiles[other_radius]["support_radius_nm"])
            if distance <= max(declared, support):
                raise ValueError(f"periodic overlap/separation failure: {particle['particle_id']} / {other['particle_id']}")
            nearest_center = min(nearest_center, distance)
            nearest_gap = min(nearest_gap, distance - max(declared, support))
    return nearest_center, nearest_gap


def profile_row(particle: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    geometry = profile["manifest"]["geometry"]
    return {
        **particle,
        "orientation_label": "variant_100_identity",
        "source_profile_manifest_path": str(profile["manifest_path"]),
        "source_profile_manifest_sha256": profile["manifest_sha256"],
        "source_phi_sha256": profile["manifest"]["fields"]["phi"]["sha256"],
        "source_delta_C_relaxation_sha256": profile["manifest"]["fields"]["delta_C_relaxation"]["sha256"],
        "source_h_volume_nm3": float(geometry.get("final_h_volume_nm3", geometry["h_volume_nm3"])),
        "source_equivalent_radius_nm": float(geometry["actual_equivalent_radius_nm"]),
        "source_support_radius_nm": float(profile["support_radius_nm"]),
    }


def compose(particles: list[dict[str, Any]], profiles: dict[float, dict[str, Any]], n: int, native: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], float]:
    """Compose fields in canonical particle-id order, regardless of input order."""
    canonical = sorted(particles, key=lambda row: row["particle_id"])
    complement = np.ones((n, n, n), dtype=np.float64)
    delta_x_total = np.zeros((n, n, n), dtype=np.float64)
    support_owner = np.zeros((n, n, n), dtype=np.uint8)
    rows: list[dict[str, Any]] = []
    source_h_sum = 0.0
    for particle in canonical:
        radius = float(particle["registered_radius_nm"])
        profile = profiles[radius]
        indices = np.ix_(*window_indices(np.asarray(particle["center_grid"], dtype=np.int64), native, n))
        support = profile["h"] > THRESHOLD
        owned = support_owner[indices]
        if bool(np.any(owned[support])):
            raise ValueError(f"h-threshold profile support overlap at {particle['particle_id']}")
        owned[support] = 1
        support_owner[indices] = owned
        local = complement[indices]
        local *= 1.0 - profile["phi"]
        complement[indices] = local
        local_delta = delta_x_total[indices]
        local_delta += profile["delta_x"]
        delta_x_total[indices] = local_delta
        row = profile_row(particle, profile)
        source_h_sum += float(row["source_h_volume_nm3"])
        rows.append(row)
    phi = 1.0 - complement
    del complement, support_owner
    if not np.all(np.isfinite(phi)) or float(np.min(phi)) < 0.0 or float(np.max(phi)) > 1.0:
        raise ValueError("bounded union escaped [0, 1]")
    return phi, delta_x_total, rows, source_h_sum


def center_statistics(particles: list[dict[str, Any]], n: int) -> dict[str, Any]:
    centers = np.asarray([row["center_grid"] for row in particles], dtype=np.float64)
    radii = np.asarray([row["registered_radius_nm"] for row in particles], dtype=np.float64)
    count = len(particles)
    distances: list[float] = []
    nearest = np.full(count, np.inf, dtype=np.float64)
    neighbours_50 = np.zeros(count, dtype=np.int64)
    for left in range(count):
        for right in range(left + 1, count):
            distance = periodic_distance(centers[left], centers[right], n)
            distances.append(distance)
            nearest[left] = min(nearest[left], distance); nearest[right] = min(nearest[right], distance)
            if distance <= 50.0:
                neighbours_50[left] += 1; neighbours_50[right] += 1
    distances_array = np.asarray(distances, dtype=np.float64)
    bin_edges = np.linspace(0.0, n / 2.0, 41)
    counts, _ = np.histogram(distances_array, bins=bin_edges)
    shell = (4.0 * math.pi / 3.0) * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)
    expected_pairs = 0.5 * count * (count / float(n ** 3)) * shell
    g = np.divide(counts, expected_pairs, out=np.zeros_like(shell), where=expected_pairs > 0.0)
    wavevectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    factors: dict[str, float] = {}
    for vector in wavevectors:
        k = 2.0 * math.pi * np.asarray(vector, dtype=np.float64) / float(n)
        amplitude = np.sum(np.exp(-1j * centers.dot(k)))
        factors["_".join(str(value) for value in vector)] = float((amplitude.conjugate() * amplitude).real / count)
    sf_values = list(factors.values())
    return {
        "particle_count": count,
        "nearest_center_distance_nm_min": float(np.min(nearest)),
        "nearest_center_distance_nm_mean": float(np.mean(nearest)),
        "local_neighbor_count_r50nm": {"min": int(np.min(neighbours_50)), "mean": float(np.mean(neighbours_50)), "max": int(np.max(neighbours_50)), "std": float(np.std(neighbours_50))},
        "pair_correlation": {"bin_edges_nm": bin_edges.tolist(), "g_r": g.tolist(), "pair_counts": counts.astype(int).tolist()},
        "directional_structure_factor": {"fundamental_modes": factors, "anisotropy_max_min_ratio": float(max(sf_values) / max(min(sf_values), 1.0e-30))},
        "periodic_isotropy_audit": "REPORTED_DIRECTIONAL_STRUCTURE_FACTOR_NO_CRYSTAL_LATTICE_PLACEMENT",
        "radius_center_assignment_hash": hashlib.sha256(json.dumps({"centers": centers.astype(int).tolist(), "radii": radii.tolist()}, sort_keys=True).encode("utf-8")).hexdigest(),
    }


def write_fields(out: Path, fields: dict[str, np.ndarray]) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for name, values in fields.items():
        path = out / f"{name}.raw.f64"
        np.asarray(values, dtype="<f8").ravel(order="C").tofile(path)
        rows[name] = {"path": path.name, "sha256": sha256(path), "dtype": "float64-le", "order": "C"}
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reverse-order-check", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"[fatal] refusing to overwrite fixture root: {args.out}")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if spec.get("schema") != "PF_400CUBE_PSD_SPATIAL_DENSITY_INTEGER_GATE_PLAN_V2":
        raise SystemExit("[fatal] V2 fixture requires a V2 integer-gate plan spec")
    grid = tuple(int(spec["grid"][key]) for key in ("Nx", "Ny", "Nz"))
    if grid != (400, 400, 400) or float(spec["grid"]["dx_nm"]) != 1.0:
        raise SystemExit("[fatal] V2 materializer accepts only frozen 400^3/dx=1")
    if float(spec["inventory_contract"]["h_volume_relative_tolerance"]) != H_REL_TOL:
        raise SystemExit("[fatal] unexpected h-volume gate")
    expected_radii = [float(value) for value in spec["profile_library_contract"]["expected_radii_nm"]]
    native = tuple(int(value) for value in spec["profile_library_contract"]["native_grid"])
    if native != (96, 96, 96):
        raise SystemExit("[fatal] V2 accepts only the native 96^3 library profiles")
    library_manifest_path, library, profile_index, _identity = library_tools.verify_library(
        args.library_root.resolve(), str(spec["profile_library_contract"]["library_manifest_sha256"]), None, None, expected_radii, native,
    )
    if abs(float(library["lambda_sm_nm"]) - float(spec["grid"]["lambda_sm_nm"])) > 1.0e-14:
        raise SystemExit("[fatal] library interface-width mismatch")
    profiles: dict[float, dict[str, Any]] = {}
    for radius in expected_radii:
        manifest_path, manifest = profile_index[radius]
        arrays = read_profile_arrays(manifest_path, manifest, native)
        profiles[radius] = {**arrays, "manifest": manifest, "manifest_path": manifest_path, "manifest_sha256": sha256(manifest_path), "support_radius_nm": support_radius_nm(arrays["h"], THRESHOLD)}
    particles = deterministic_particles(spec, profiles)
    n = grid[0]
    nearest_center, nearest_gap = validate_particle_separation(particles, profiles, n, float(spec["grid"]["lambda_sm_nm"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.mkdir()
    phi, delta_x_total, particle_rows, source_h_sum = compose(particles, profiles, n, native[0])
    h = h_of_phi(phi)
    alpha = 1.0 - h
    if float(np.min(alpha)) < -1.0e-14:
        raise SystemExit("[fatal] bounded h(phi) contract failed")
    target_total = TARGET_MEAN * float(n ** 3)
    baseline = (target_total - float(np.sum(h + alpha * delta_x_total, dtype=np.float64))) / float(np.sum(alpha, dtype=np.float64))
    x_b = baseline + delta_x_total
    if not np.all(np.isfinite(x_b)) or float(np.min(x_b)) <= 0.0 or float(np.max(x_b)) >= 0.499999:
        raise SystemExit("[fatal] composition reconstruction would require clipping")
    c_total = h + alpha * x_b
    inventory_rel = abs(float(np.sum(c_total, dtype=np.float64)) - target_total) / target_total
    if inventory_rel > 1.0e-12:
        raise SystemExit(f"[fatal] canonical inventory closure {inventory_rel:.3e}")
    # The integer gate is explicitly a profile-inventory gate: the task asks
    # for the *recorded native profile h-volumes*, and its lattice proof is
    # written in those quantities.  The bounded union's sub-threshold diffuse
    # tails are still reported as a field diagnostic below, but cannot replace
    # the profile ledger (nor can they be eliminated without forbidden field
    # clipping/renormalization).  This is the same distinction used by the
    # frozen Method-1 inventory selection.
    actual_h = float(np.sum(h, dtype=np.float64))
    h_rel = abs(source_h_sum - TARGET_H) / TARGET_H
    if h_rel > H_REL_TOL:
        raise SystemExit(f"[fatal] recorded native-profile h-volume misses V2 gate: {h_rel:.3e}")
    assembled_union_h_rel = abs(actual_h - TARGET_H) / TARGET_H
    matrix_mask = h < THRESHOLD
    if not bool(np.any(matrix_mask)):
        raise SystemExit("[fatal] no h<1e-4 matrix observation region")
    matrix_xb = float(np.mean(x_b[matrix_mask], dtype=np.float64))
    matrix_xag = xag_from_xb(matrix_xb)
    if abs(matrix_xag - float(spec["inventory_contract"]["matrix_xAg_target"])) > float(spec["inventory_contract"]["matrix_xAg_absolute_tolerance"]):
        raise SystemExit(f"[fatal] matrix xAg V2 gate failed: {matrix_xag:.12g}")
    y = np.log(x_b / (1.0 - x_b))
    fields = {"phi": phi, "h_phi": h, "xB_alpha": x_b, "Y": y, "dY_dt_prev": np.zeros_like(phi), "C_B_tot": c_total, "delta_C_relaxation_total": alpha * delta_x_total}
    field_rows = write_fields(args.out, fields)
    reverse_status = "NOT_REQUESTED"
    if args.reverse_order_check:
        # The raw spec could be read in reverse, but the physics contract
        # canonicalizes by particle ID before the bounded union.  Recompose to
        # verify that both phi and the independently derived composition are
        # byte-identical; every other fresh-start field is deterministic from
        # those two fields.
        reverse_phi, reverse_delta, _reverse_rows, _reverse_source = compose(list(reversed(particles)), profiles, n, native[0])
        reverse_h = h_of_phi(reverse_phi)
        reverse_alpha = 1.0 - reverse_h
        reverse_baseline = (target_total - float(np.sum(reverse_h + reverse_alpha * reverse_delta, dtype=np.float64))) / float(np.sum(reverse_alpha, dtype=np.float64))
        reverse_xb = reverse_baseline + reverse_delta
        if not np.array_equal(reverse_phi, phi) or not np.array_equal(reverse_xb, x_b):
            raise SystemExit("[fatal] reversed logical manifest changes raw fields")
        reverse_status = "PASS_BITWISE_CANONICAL_ORDER_INVARIANCE"
    init_meta = {"schema": META_SCHEMA, "Nx": 400, "Ny": 400, "Nz": 400, "dx_nm": 1.0, "interface_width_nm": float(spec["grid"]["lambda_sm_nm"]), "dt_recommended": float(spec["time_contract"]["dt_code"]), "mean_xBtot": TARGET_MEAN, "xB_max_safe": 0.499999, "dtype": "float64", "order": "C", "phi_path": field_rows["phi"]["path"], "xB_path": field_rows["xB_alpha"]["path"], "fresh_dY_dt_prev_contract": "zero_for_fresh_dynamic_start"}
    write_json(args.out / "init_meta.json", init_meta)
    with (args.out / "initial_particles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(particle_rows[0]))
        writer.writeheader(); writer.writerows(particle_rows)
    spatial = center_statistics(particle_rows, n)
    spatial.update({"minimum_required_extra_gap_nm": nearest_gap, "minimum_declared_center_distance_nm": nearest_center})
    write_json(args.out / "spatial_statistics.json", spatial)
    manifest = {
        "schema": SCHEMA, "fixture_id": spec["fixture_id"], "case": spec["case"], "initial_state_class": INITIAL_STATE_CLASS,
        "scientific_semantics": "conditional 6h resolved-beta PSD/spatial/density ladder; not a common multi-particle equilibrium claim or fitted experimental PSD",
        "spec_path": str(args.spec.resolve()), "spec_sha256": sha256(args.spec),
        "profile_library_manifest_path": str(library_manifest_path), "profile_library_manifest_sha256": sha256(library_manifest_path),
        "profile_library_source_tree_sha256": library.get("source_tree_sha256"), "profile_library_binary_sha256": library.get("binary_sha256"),
        "grid": spec["grid"], "physical_contract": spec["physical_contract"], "time_contract": spec["time_contract"],
        "placement": {**spec["placement_contract"], "canonical_particle_order": [row["particle_id"] for row in particle_rows], "spatial_statistics_path": "spatial_statistics.json"},
        "particle_library_mappings": particle_rows,
        "component_contract": {"h_threshold": THRESHOLD, "expected_count": len(particles), "actual_count": len(particles), "support_overlap_status": "PASS_NO_OVERLAP", "periodic_separation_status": "PASS"},
        "initial_canonical_inventory": {"target_mean_C_B_tot": TARGET_MEAN, "target_total_code": target_total, "actual_total_code": float(np.sum(c_total, dtype=np.float64)), "field_relative_error": inventory_rel, "status": "PASS_MACHINE_PRECISION"},
        "initial_beta_inventory": {"original_target_h_volume_nm3": TARGET_H, "selected_common_target_h_volume_nm3": float(spec["inventory_contract"]["selected_common_target_h_volume_nm3"]), "source_profile_h_volume_sum_nm3": source_h_sum, "profile_inventory_relative_error": h_rel, "assembled_bounded_union_h_volume_nm3": actual_h, "assembled_bounded_union_relative_to_target": assembled_union_h_rel, "h_volume_relative_tolerance": H_REL_TOL, "inventory_policy": str(spec["inventory_contract"]["inventory_policy"]), "gate_definition": "sum of recorded native profile h-volumes; bounded-union diffuse-tail integral separately reported without clipping or normalization"},
        "derived_matrix_baseline": {"xB_alpha": baseline, "xAg": xag_from_xb(baseline), "observed_far_field_xB_h_lt_1e-4": matrix_xb, "observed_far_field_xAg_h_lt_1e-4": matrix_xag, "target_xAg": float(spec["inventory_contract"]["matrix_xAg_target"]), "target_xAg_tolerance": float(spec["inventory_contract"]["matrix_xAg_absolute_tolerance"]), "experimental_xAg_interval": spec["inventory_contract"]["experimental_matrix_xAg_interval"]},
        "assembly_contract": {"phi": "1-product_j(1-phi_j), canonical particle-id order", "portable_composition_field": "native delta_C_relaxation/(1-h), no resampling; one baseline closes canonical inventory", "profile_scaling_used": False, "profile_interpolation_used": False, "profile_rotation_used": False, "analytic_profile_used": False, "clipping_used": False, "normalization_used": False, "common_multi_particle_equilibrium_claim": False, "initial_relaxation_is_physical_evolution": True, "reverse_order_invariance": reverse_status},
        "initial_zero_mode_provenance": {"zero_mode": "PF_CONSERVED_Y_ZERO_MODE_V1", "backend": "HOST_NEWTON_BISECTION_V1", "explicit_context": "SM_EXPLICIT_CONTEXT_N_V1", "reaction_discretization": "SM_TANGENT_N_V1", "target_mass_source": "initial_canonical_inventory", "dY_dt_prev": "zero_for_fresh_dynamic_start"},
        "field_bounds": {"phi_min": float(np.min(phi)), "phi_max": float(np.max(phi)), "xB_min": float(np.min(x_b)), "xB_max": float(np.max(x_b))}, "fields": field_rows,
    }
    write_json(args.out / "fixture_manifest.json", manifest)
    manifest_sha = sha256(args.out / "fixture_manifest.json")
    hashes = [f"{row['sha256']}  {row['path']}" for _name, row in sorted(field_rows.items())]
    for name in ("fixture_manifest.json", "init_meta.json", "initial_particles.csv", "spatial_statistics.json"):
        hashes.append(f"{sha256(args.out / name)}  {name}")
    (args.out / "fixture_hashes.sha256").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    status = "PASS_400CUBE_PSD_CAMPAIGN_INTEGER_GATE_FIXTURE_V2"
    (args.out / "status.txt").write_text(status + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "case_id": spec["case"]["case_id"], "fixture_manifest_sha256": manifest_sha, "particle_count": len(particles), "mean_C_Btot": float(np.mean(c_total)), "matrix_xAg": matrix_xag, "profile_h_volume_nm3": source_h_sum, "profile_h_volume_relative_error": h_rel, "assembled_bounded_union_h_volume_nm3": actual_h, "reverse_order_status": reverse_status}, sort_keys=True))


if __name__ == "__main__":
    main()
