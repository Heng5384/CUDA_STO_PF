#!/usr/bin/env python3
"""Materialize a deterministic conserved multi-particle elastic PF fixture.

V1 only translates complete selected-library fields on their native periodic
grid.  It intentionally refuses resampling, radial interpolation, rotated
variants, and an absolute-xB superposition.  The portable quantity transferred
from each library entry is delta_C_relaxation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_6H_FIXTURE_V1"
SPEC_SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_6H_FIXTURE_SPEC_V1"
LIBRARY_SCHEMA = "PF_ELASTIC_TARGET_PROFILE_LIBRARY_V1"
PROFILE_SCHEMA = "PF_ELASTIC_TARGET_PROFILE_V1"
RAW_META_SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_RAW_INIT_META_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_spec_digest(spec: Dict[str, Any]) -> str:
    """Hash semantic spec content without treating manifest particle order as physics."""
    canonical = dict(spec)
    particles = canonical.get("particles", [])
    canonical["particles"] = sorted(
        (dict(item) for item in particles), key=lambda item: str(item.get("particle_id", ""))
    )
    return canonical_json_digest(canonical)


def h_of_phi(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def logit(xb: np.ndarray) -> np.ndarray:
    return np.log(xb / (1.0 - xb))


def load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_finite(value: Any, label: str) -> None:
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label}: non-finite scalar")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite(item, f"{label}[{index}]")
    else:
        raise ValueError(f"{label}: unsupported finite-value type")


def native_library_paths(root: Path) -> Tuple[Path, Path]:
    direct = root / "library_manifest.json"
    nested = root / "library" / "library_manifest.json"
    if direct.is_file():
        manifest = direct
    elif nested.is_file():
        manifest = nested
    else:
        raise ValueError(f"{root}: missing selected library_manifest.json")
    profiles = root / "profiles"
    if not profiles.is_dir():
        profiles = manifest.parent.parent / "profiles"
    if not profiles.is_dir():
        raise ValueError(f"{root}: missing profiles directory")
    return manifest, profiles


def read_raw(path: Path, shape: Tuple[int, int, int], label: str) -> np.ndarray:
    count = math.prod(shape)
    value = np.fromfile(path, dtype="<f8")
    if value.size != count:
        raise ValueError(f"{path}: {label} count={value.size}, expected={count}")
    return value.reshape(shape, order="C")


def periodic_distance(a: Sequence[int], b: Sequence[int], shape: Sequence[int], dx_nm: float) -> float:
    squared = 0.0
    for lhs, rhs, period in zip(a, b, shape):
        delta = abs(int(lhs) - int(rhs))
        delta = min(delta, int(period) - delta)
        squared += (delta * dx_nm) ** 2
    return math.sqrt(squared)


def periodic_delta_vector(length: int, center: int, dx_nm: float) -> np.ndarray:
    values = np.arange(length, dtype=np.float64)
    return ((values - float(center) + 0.5 * length) % length - 0.5 * length) * dx_nm


def replay_hard_core_centers(
    particles: Sequence[Dict[str, Any]], shape: Tuple[int, int, int], dx_nm: float,
    lambda_sm_nm: float, seed: int, max_trials: int,
) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    placed: List[Tuple[List[int], float]] = []
    result: List[List[int]] = []
    for particle in particles:
        radius = float(particle["registered_radius_nm"])
        chosen: Optional[List[int]] = None
        for _ in range(max_trials):
            candidate = [int(v) for v in rng.integers(0, shape[0], size=3)]
            if all(
                periodic_distance(candidate, other_center, shape, dx_nm)
                > radius + other_radius + 4.0 * lambda_sm_nm
                for other_center, other_radius in placed
            ):
                chosen = candidate
                break
        if chosen is None:
            raise ValueError(f"hard-core placement could not place particle {particle['particle_id']}")
        placed.append((chosen, radius))
        result.append(chosen)
    return result


def source_manifest_path(library_manifest_path: Path, profile_entry: Dict[str, Any]) -> Path:
    path = (library_manifest_path.parent / str(profile_entry["profile_manifest_path"])).resolve()
    if not path.is_file():
        raise ValueError(f"selected source profile manifest missing: {path}")
    return path


def check_field(manifest: Dict[str, Any], manifest_path: Path, name: str, shape: Tuple[int, int, int]) -> Path:
    field = manifest.get("fields", {}).get(name)
    if not isinstance(field, dict):
        raise ValueError(f"{manifest_path}: missing field declaration {name}")
    path = manifest_path.parent / str(field.get("path", ""))
    expected = str(field.get("sha256", ""))
    if len(expected) != 64 or not path.is_file():
        raise ValueError(f"{manifest_path}: missing field {name}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{path}: {name} SHA-256 mismatch")
    if path.stat().st_size != math.prod(shape) * 8:
        raise ValueError(f"{path}: {name} byte size does not match native grid")
    return path


def verify_library(
    library_root: Path, expected_library_sha: str, selection_provenance: Optional[Path],
    expected_selection_sha: Optional[str],
) -> Tuple[Path, Dict[str, Any], Dict[float, Tuple[Path, Dict[str, Any]]], Dict[str, Any]]:
    library_manifest_path, _profiles_root = native_library_paths(library_root)
    if sha256(library_manifest_path) != expected_library_sha:
        raise ValueError("selected library manifest SHA-256 mismatch")
    library = load_json(library_manifest_path)
    if library.get("schema") != LIBRARY_SCHEMA:
        raise ValueError("wrong selected library schema")
    if int(library.get("profile_count", -1)) != 8:
        raise ValueError("selected library must contain all eight registered profiles")
    shape_data = library.get("grid", {})
    shape = (int(shape_data.get("Nx", 0)), int(shape_data.get("Ny", 0)), int(shape_data.get("Nz", 0)))
    if shape != (96, 96, 96):
        raise ValueError(f"unexpected selected library native shape {shape}")
    profiles: Dict[float, Tuple[Path, Dict[str, Any]]] = {}
    selected_profile_rows: List[Dict[str, Any]] = []
    for entry in library.get("profiles", []):
        radius = float(entry["target_radius_nm"])
        manifest_path = source_manifest_path(library_manifest_path, entry)
        manifest = load_json(manifest_path)
        if manifest.get("schema") != PROFILE_SCHEMA:
            raise ValueError(f"{manifest_path}: wrong profile schema")
        if sha256(manifest_path) != str(entry["profile_manifest_sha256"]):
            raise ValueError(f"{manifest_path}: manifest SHA-256 mismatch")
        if tuple(int(manifest["grid"][axis]) for axis in ("Nx", "Ny", "Nz")) != shape:
            raise ValueError(f"{manifest_path}: profile grid mismatch")
        if float(manifest["geometry"]["target_equivalent_radius_nm"]) != radius:
            raise ValueError(f"{manifest_path}: registered radius mismatch")
        if manifest.get("orientation_label") != "variant_100_identity":
            raise ValueError(f"{manifest_path}: rotated variant is not allowed")
        for name in ("phi", "h_phi", "delta_C_relaxation", "dY_dt_prev", "Y", "C_B_tot", "xB_alpha"):
            check_field(manifest, manifest_path, name, shape)
        profiles[radius] = (manifest_path, manifest)
        selected_profile_rows.append({
            "registered_radius_nm": radius,
            "profile_manifest_path": str(manifest_path),
            "profile_manifest_sha256": sha256(manifest_path),
            **{f"{name}_sha256": manifest["fields"][name]["sha256"] for name in manifest["fields"]},
        })
    if sorted(profiles) != [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5]:
        raise ValueError("selected library radius ladder differs from the frozen V1 ladder")
    selection: Dict[str, Any] = {}
    if selection_provenance is not None:
        if not selection_provenance.is_file():
            raise ValueError(f"selection provenance missing: {selection_provenance}")
        actual_selection = sha256(selection_provenance)
        if expected_selection_sha and actual_selection != expected_selection_sha:
            raise ValueError("selection provenance SHA-256 mismatch")
        selection = load_json(selection_provenance)
        if selection.get("library_manifest_sha256") != expected_library_sha:
            raise ValueError("selection provenance does not pin the selected library")
    return library_manifest_path, library, profiles, {
        "selected_library_manifest_sha256": expected_library_sha,
        "selected_library_manifest_path": str(library_manifest_path),
        "selection_provenance_path": str(selection_provenance) if selection_provenance else None,
        "selection_provenance_sha256": sha256(selection_provenance) if selection_provenance else None,
        "source_tree_sha256": library.get("source_tree_sha256"),
        "binary_sha256": library.get("binary_sha256"),
        "profiles": selected_profile_rows,
        "selection": selection,
    }


def validate_spec(
    spec: Dict[str, Any], library: Dict[str, Any], profiles: Dict[float, Tuple[Path, Dict[str, Any]]],
) -> Tuple[Tuple[int, int, int], float, float, List[Dict[str, Any]]]:
    if spec.get("schema") != SPEC_SCHEMA:
        raise ValueError("wrong multi-particle fixture spec schema")
    target = spec.get("target", {})
    grid = tuple(int(v) for v in target.get("grid", []))
    if len(grid) != 3 or any(v <= 0 for v in grid):
        raise ValueError("fixture target must declare a positive three-dimensional grid")
    native = library["grid"]
    native_grid = (int(native["Nx"]), int(native["Ny"]), int(native["Nz"]))
    if grid != native_grid:
        raise ValueError(
            "V1 requires the target grid to equal the native selected-profile grid; "
            "cross-box field extension/resampling is not qualified"
        )
    dx_nm = float(target.get("dx_nm", math.nan))
    lambda_sm_nm = float(target.get("lambda_sm_nm", math.nan))
    if not math.isfinite(dx_nm) or dx_nm <= 0.0 or abs(dx_nm - float(native["dx_nm"])) > 1e-12:
        raise ValueError("fixture dx does not equal selected-library dx")
    if not math.isfinite(lambda_sm_nm) or abs(lambda_sm_nm - float(library["lambda_sm_nm"])) > 1e-12:
        raise ValueError("fixture lambda does not equal selected-library lambda")
    if float(target.get("temperature_C", math.nan)) != float(library["temperature_C"]):
        raise ValueError("fixture temperature does not equal selected-library temperature")
    particles_data = spec.get("particles")
    if not isinstance(particles_data, list) or not particles_data:
        raise ValueError("fixture spec has no particles")
    particles = sorted((dict(item) for item in particles_data), key=lambda item: str(item.get("particle_id", "")))
    ids = [str(item.get("particle_id", "")) for item in particles]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("particle ids must be non-empty and unique")
    placement = spec.get("placement", {})
    if placement.get("method") != "deterministic_hard_core_rejection_integer_grid_v1":
        raise ValueError("V1 requires hash-pinned deterministic integer-grid hard-core placement")
    seed = int(placement.get("seed", -1))
    trials = int(placement.get("max_trials_per_particle", 0))
    if seed < 0 or trials < 1:
        raise ValueError("placement seed/max_trials missing")
    for item in particles:
        radius = float(item.get("registered_radius_nm", math.nan))
        if radius not in profiles:
            raise ValueError(f"{item['particle_id']}: unregistered radius {radius}")
        if item.get("orientation_label") != "variant_100_identity":
            raise ValueError(f"{item['particle_id']}: rotated variant is rejected")
        center = item.get("center_grid")
        if not isinstance(center, list) or len(center) != 3 or any(int(v) != v for v in center):
            raise ValueError(f"{item['particle_id']}: center must be a three-component integer grid coordinate")
        if any(int(v) < 0 or int(v) >= limit for v, limit in zip(center, grid)):
            raise ValueError(f"{item['particle_id']}: center lies outside periodic domain")
        manifest_path, manifest = profiles[radius]
        expected = {
            "source_profile_manifest_sha256": sha256(manifest_path),
            "source_phi_sha256": manifest["fields"]["phi"]["sha256"],
            "source_delta_C_relaxation_sha256": manifest["fields"]["delta_C_relaxation"]["sha256"],
        }
        for key, value in expected.items():
            if item.get(key) != value:
                raise ValueError(f"{item['particle_id']}: {key} mismatch")
    replay = replay_hard_core_centers(particles, grid, dx_nm, lambda_sm_nm, seed, trials)
    for item, replay_center in zip(particles, replay):
        if [int(v) for v in item["center_grid"]] != replay_center:
            raise ValueError(f"{item['particle_id']}: center does not reproduce hash-pinned hard-core placement")
    for index, lhs in enumerate(particles):
        for rhs in particles[:index]:
            distance = periodic_distance(lhs["center_grid"], rhs["center_grid"], grid, dx_nm)
            required = float(lhs["registered_radius_nm"]) + float(rhs["registered_radius_nm"]) + 4.0 * lambda_sm_nm
            if not distance > required:
                raise ValueError(f"periodic overlap: {lhs['particle_id']} / {rhs['particle_id']}")
    return grid, dx_nm, lambda_sm_nm, particles


def translated(array: np.ndarray, center_grid: Sequence[int]) -> np.ndarray:
    source_center = tuple(length // 2 for length in array.shape)
    shift = tuple(int(target) - source for target, source in zip(center_grid, source_center))
    return np.roll(array, shift=shift, axis=(0, 1, 2))


def analytic_sphere(
    shape: Tuple[int, int, int], center: Sequence[int], dx_nm: float, lambda_sm_nm: float,
    target_h_volume: float,
) -> Tuple[np.ndarray, float]:
    lo, hi = 0.0, max(shape) * dx_nm / 2.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        phi = sphere_phi(shape, center, dx_nm, lambda_sm_nm, mid)
        h_volume = float(np.sum(h_of_phi(phi), dtype=np.float64) * dx_nm**3)
        if h_volume < target_h_volume:
            lo = mid
        else:
            hi = mid
    radius = 0.5 * (lo + hi)
    return sphere_phi(shape, center, dx_nm, lambda_sm_nm, radius), radius


def sphere_phi(
    shape: Tuple[int, int, int], center: Sequence[int], dx_nm: float,
    lambda_sm_nm: float, radius_nm: float,
) -> np.ndarray:
    coordinates = [periodic_delta_vector(length, int(c), dx_nm) for length, c in zip(shape, center)]
    radius_field = np.sqrt(
        coordinates[0][:, None, None] ** 2
        + coordinates[1][None, :, None] ** 2
        + coordinates[2][None, None, :] ** 2
    )
    width = lambda_sm_nm / 2.0
    return 0.5 * (1.0 + np.tanh((radius_nm - radius_field) / width))


def components(h: np.ndarray, threshold: float, dx_nm: float) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    mask = h > threshold
    shape = h.shape
    labels = np.full(shape, -1, dtype=np.int32)
    rows: List[Dict[str, Any]] = []
    label = 0
    for start in zip(*np.nonzero(mask & (labels < 0))):
        if labels[start] >= 0:
            continue
        queue: deque[Tuple[int, int, int]] = deque([tuple(int(v) for v in start)])
        labels[start] = label
        cells: List[Tuple[int, int, int]] = []
        while queue:
            i, j, k = queue.popleft()
            cells.append((i, j, k))
            for ni, nj, nk in (
                ((i + 1) % shape[0], j, k), ((i - 1) % shape[0], j, k),
                (i, (j + 1) % shape[1], k), (i, (j - 1) % shape[1], k),
                (i, j, (k + 1) % shape[2]), (i, j, (k - 1) % shape[2]),
            ):
                if mask[ni, nj, nk] and labels[ni, nj, nk] < 0:
                    labels[ni, nj, nk] = label
                    queue.append((ni, nj, nk))
        indices = np.asarray(cells, dtype=np.int64)
        support = np.zeros(shape, dtype=np.float64)
        support[indices[:, 0], indices[:, 1], indices[:, 2]] = h[indices[:, 0], indices[:, 1], indices[:, 2]]
        volume = float(np.sum(support, dtype=np.float64) * dx_nm**3)
        total = float(np.sum(support, dtype=np.float64))
        centroid: List[float] = []
        displacements: List[np.ndarray] = []
        for axis, length in enumerate(shape):
            weights = np.sum(support, axis=tuple(x for x in range(3) if x != axis), dtype=np.float64)
            angles = 2.0 * math.pi * np.arange(length, dtype=np.float64) / length
            angle = math.atan2(float(np.dot(weights, np.sin(angles))), float(np.dot(weights, np.cos(angles)))) % (2.0 * math.pi)
            center = angle * length / (2.0 * math.pi)
            centroid.append(center * dx_nm)
            values = periodic_delta_vector(length, int(round(center)) % length, dx_nm)
            # Preserve the fractional circular centroid in the displacement.
            values = ((np.arange(length, dtype=np.float64) - center + 0.5 * length) % length - 0.5 * length) * dx_nm
            reshape = [1, 1, 1]
            reshape[axis] = length
            displacements.append(values.reshape(reshape))
        covariance = np.empty((3, 3), dtype=np.float64)
        for a in range(3):
            for b in range(a, 3):
                value = float(np.sum(support * displacements[a] * displacements[b], dtype=np.float64) / total)
                covariance[a, b] = covariance[b, a] = value
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        values = values[order]
        vectors = vectors[:, order]
        if np.any(values <= 0.0):
            raise ValueError(
                "component shape tensor is not positive definite: "
                f"cells={indices.shape[0]} volume={volume:.17e} eigenvalues={values.tolist()}"
            )
        axes = np.sqrt(5.0 * values)
        for column in range(3):
            pivot = int(np.argmax(np.abs(vectors[:, column])))
            if vectors[pivot, column] < 0.0:
                vectors[:, column] *= -1.0
        rows.append({
            "component_label": label,
            "threshold_voxel_count": int(indices.shape[0]),
            "h_volume_nm3": volume,
            "equivalent_radius_nm": (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0),
            "centroid_nm": centroid,
            "semi_axes_nm": axes.tolist(),
            "axis_ratio": float(axes[0] / axes[2]),
            "principal_axes_rows": vectors.T.tolist(),
        })
        label += 1
    return labels, rows


def write_raw(path: Path, value: np.ndarray) -> None:
    np.asarray(value, dtype="<f8").ravel(order="C").tofile(path)


def materialize(args: argparse.Namespace) -> Dict[str, Any]:
    spec_path = args.spec.resolve()
    spec = load_json(spec_path)
    expected_library_sha = str(spec.get("selected_library_manifest_sha256", ""))
    expected_selection_sha = spec.get("selected_library_selection_sha256")
    if len(expected_library_sha) != 64:
        raise ValueError("fixture spec must pin selected_library_manifest_sha256")
    library_path, library, profile_index, freeze = verify_library(
        args.library_root.resolve(), expected_library_sha,
        args.selection_provenance.resolve() if args.selection_provenance else None,
        str(expected_selection_sha) if expected_selection_sha else None,
    )
    grid, dx_nm, lambda_sm_nm, particles = validate_spec(spec, library, profile_index)
    mode = args.kind
    if mode not in ("E2", "S0"):
        raise ValueError("fixture kind must be E2 or S0")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output root: {args.out}")
    args.out.mkdir(parents=True, exist_ok=False)
    product = np.ones(grid, dtype=np.float64)
    reference_e2_product = np.ones(grid, dtype=np.float64)
    delta_total = np.zeros(grid, dtype=np.float64)
    single_phi: Optional[np.ndarray] = None
    individual_h: List[np.ndarray] = []
    particle_records: List[Dict[str, Any]] = []
    source_h_sum = 0.0
    for particle in particles:
        radius = float(particle["registered_radius_nm"])
        manifest_path, manifest = profile_index[radius]
        source_h_volume = float(manifest["geometry"]["h_volume_nm3"])
        source_phi = translated(
            read_raw(check_field(manifest, manifest_path, "phi", grid), grid, "phi"),
            particle["center_grid"],
        )
        source_h = translated(
            read_raw(check_field(manifest, manifest_path, "h_phi", grid), grid, "h_phi"),
            particle["center_grid"],
        )
        source_delta = translated(
            read_raw(check_field(manifest, manifest_path, "delta_C_relaxation", grid), grid, "delta_C_relaxation"),
            particle["center_grid"],
        )
        if float(np.max(np.abs(source_h - h_of_phi(source_phi)))) > 5.0e-14:
            raise ValueError(f"{particle['particle_id']}: translated h(phi) mismatch")
        reference_e2_product *= 1.0 - source_phi
        if mode == "E2":
            phi = source_phi
            h = source_h
            local_delta = source_delta
            sphere_parameter = None
        else:
            phi, sphere_parameter = analytic_sphere(grid, particle["center_grid"], dx_nm, lambda_sm_nm, source_h_volume)
            h = h_of_phi(phi)
            # A target-profile delta_C is phase-weighted by that profile's
            # h(phi).  Directly attaching it to a different spherical h(phi)
            # would violate the bounded raw-field storage contract.  The S0
            # sensitivity reference therefore uses a uniform matrix field.
            local_delta = np.zeros(grid, dtype=np.float64)
        if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(h)) or not np.all(np.isfinite(local_delta)):
            raise ValueError(f"{particle['particle_id']}: non-finite translated field")
        if float(np.min(phi)) < 0.0 or float(np.max(phi)) > 1.0:
            raise ValueError(f"{particle['particle_id']}: phi is not bounded without clipping")
        product *= 1.0 - phi
        delta_total += local_delta
        if len(particles) == 1:
            # Preserve the registered profile byte-for-byte for the one-particle
            # E2 contract.  Evaluating 1 - (1 - phi) is mathematically identical
            # but can round differently in binary64.
            single_phi = phi
        individual_h.append(h)
        source_h_sum += source_h_volume
        record = dict(particle)
        record.update({
            "source_profile_path": str(manifest_path),
            "source_h_volume_nm3": source_h_volume,
            "source_equivalent_radius_nm": float(manifest["geometry"]["actual_equivalent_radius_nm"]),
            "source_axis_ratio": float(manifest["geometry"]["axis_ratio_major_minor"]),
            "spherical_radius_parameter_nm": sphere_parameter,
        })
        particle_records.append(record)
    reference_e2_h_volume = float(
        np.sum(h_of_phi(1.0 - reference_e2_product), dtype=np.float64) * dx_nm**3
    )
    s0_uniform_radius_offset_nm = 0.0
    if mode == "S0":
        # The S0/E2 comparison is required to carry the same portable
        # composition correction and the same total h-volume.  A common
        # sub-grid adjustment of the analytic-sphere radius parameters makes
        # the latter exact without resampling any selected E2 profile.
        initial_parameters = [float(item["spherical_radius_parameter_nm"]) for item in particle_records]

        def s0_h_volume(offset_nm: float) -> float:
            trial_product = np.ones(grid, dtype=np.float64)
            for record, parameter in zip(particle_records, initial_parameters):
                radius_nm = parameter + offset_nm
                if radius_nm <= 0.0:
                    return 0.0
                trial_product *= 1.0 - sphere_phi(
                    grid, record["center_grid"], dx_nm, lambda_sm_nm, radius_nm,
                )
            return float(np.sum(h_of_phi(1.0 - trial_product), dtype=np.float64) * dx_nm**3)

        lo = -0.5 * min(initial_parameters)
        hi = 0.5 * min(initial_parameters)
        if s0_h_volume(lo) > reference_e2_h_volume or s0_h_volume(hi) < reference_e2_h_volume:
            raise ValueError("S0 total h-volume cannot bracket the E2 reference")
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            if s0_h_volume(mid) < reference_e2_h_volume:
                lo = mid
            else:
                hi = mid
        s0_uniform_radius_offset_nm = 0.5 * (lo + hi)
        product = np.ones(grid, dtype=np.float64)
        individual_h = []
        single_phi = None
        for record, parameter in zip(particle_records, initial_parameters):
            adjusted = parameter + s0_uniform_radius_offset_nm
            phi = sphere_phi(grid, record["center_grid"], dx_nm, lambda_sm_nm, adjusted)
            product *= 1.0 - phi
            individual_h.append(h_of_phi(phi))
            record["spherical_radius_parameter_nm"] = adjusted
            record["spherical_uniform_radius_offset_nm"] = s0_uniform_radius_offset_nm
            if len(particles) == 1:
                single_phi = phi
    phi_total = single_phi if single_phi is not None else 1.0 - product
    if not np.all(np.isfinite(phi_total)) or float(np.min(phi_total)) < 0.0 or float(np.max(phi_total)) > 1.0:
        raise ValueError("bounded union produced an invalid phi field")
    h_total = h_of_phi(phi_total)
    alpha = 1.0 - h_total
    phase_floor = float(spec.get("target", {}).get("phase_storage_alpha_floor", 1.0e-10))
    if not math.isfinite(phase_floor) or phase_floor <= 0.0:
        raise ValueError("invalid phase_storage_alpha_floor")
    if float(np.min(alpha)) <= phase_floor:
        raise ValueError("unresolved matrix fraction encountered; V1 fails closed rather than divide or clip")
    target_mean = float(spec["target"]["mean_C_B_tot"])
    if not math.isfinite(target_mean) or not (0.0 < target_mean < 1.0):
        raise ValueError("target mean canonical inventory must be in (0,1)")
    v_b = float(library["v_B"])
    target_total = target_mean * float(math.prod(grid))
    baseline = (target_total - float(np.sum(v_b * h_total + delta_total, dtype=np.float64))) / float(np.sum(alpha, dtype=np.float64))
    xB_max_safe = float(spec["target"].get("xB_max_safe", 0.499999))
    xB = baseline + delta_total / alpha
    if not np.all(np.isfinite(xB)) or float(np.min(xB)) <= 0.0 or float(np.max(xB)) >= xB_max_safe:
        raise ValueError("portable composition reconstruction would require clipping or invalid xB")
    c_total = alpha * xB + v_b * h_total
    inventory_relative_error = abs(float(np.sum(c_total, dtype=np.float64)) - target_total) / max(abs(target_total), 1.0)
    if inventory_relative_error > 1.0e-12:
        raise ValueError(f"canonical inventory does not close: {inventory_relative_error:.3e}")
    y = logit(xB)
    # The frozen raw-field runtime contract explicitly initializes fresh dY/dt
    # history to zero.  It is written and hashed rather than left implicit.
    dY_dt_prev = np.zeros(grid, dtype=np.float64)
    threshold = float(spec["target"].get("component_h_threshold", 1.0e-4))
    labels, component_rows = components(h_total, threshold, dx_nm)
    support_to_component: Dict[str, int] = {}
    used_components: set[int] = set()
    for particle, source_h in zip(particle_records, individual_h):
        support = source_h > threshold
        candidate_labels = labels[support]
        candidate_labels = candidate_labels[candidate_labels >= 0]
        if candidate_labels.size == 0:
            raise ValueError(f"{particle['particle_id']}: no connected support at registered threshold")
        values, counts = np.unique(candidate_labels, return_counts=True)
        assigned = int(values[int(np.argmax(counts))])
        support_to_component[str(particle["particle_id"])] = assigned
        if assigned in used_components:
            raise ValueError("unexpected merged initial components")
        used_components.add(assigned)
    if len(component_rows) != len(particle_records) or len(used_components) != len(particle_records):
        raise ValueError("initial connected-component count does not equal particle count")
    for row in component_rows:
        row["particle_id"] = next(key for key, value in support_to_component.items() if value == row["component_label"])
    fields = {
        "phi": args.out / "phi.raw.f64",
        "h_phi": args.out / "h_phi.raw.f64",
        "C_B_tot": args.out / "C_B_tot.raw.f64",
        "xB_alpha": args.out / "xB_alpha.raw.f64",
        "Y": args.out / "Y.raw.f64",
        "dY_dt_prev": args.out / "dY_dt_prev.raw.f64",
        "delta_C_relaxation_total": args.out / "delta_C_relaxation_total.raw.f64",
    }
    for name, path in fields.items():
        write_raw(path, {
            "phi": phi_total, "h_phi": h_total, "C_B_tot": c_total, "xB_alpha": xB,
            "Y": y, "dY_dt_prev": dY_dt_prev, "delta_C_relaxation_total": delta_total,
        }[name])
    meta = {
        "schema": RAW_META_SCHEMA,
        "Nx": grid[0], "Ny": grid[1], "Nz": grid[2], "dx_nm": dx_nm,
        "interface_width_nm": lambda_sm_nm, "dt_recommended": float(spec["target"].get("dt_code", 0.02)),
        "mean_xBtot": float(np.mean(c_total, dtype=np.float64)), "xB_max_safe": xB_max_safe,
        "dtype": "float64", "order": "C", "phi_path": fields["phi"].name,
        "xB_path": fields["xB_alpha"].name, "fresh_dY_dt_prev_contract": "zero_for_fresh_dynamic_start",
    }
    write_json(args.out / "init_meta.json", meta)
    field_rows = {name: {"path": path.name, "sha256": sha256(path), "dtype": "float64-le", "order": "C"} for name, path in fields.items()}
    with (args.out / "initial_particles.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["particle_id", "registered_radius_nm", "center_grid", "source_profile_manifest_sha256", "source_phi_sha256", "source_delta_C_relaxation_sha256", "source_h_volume_nm3", "source_equivalent_radius_nm", "source_axis_ratio", "spherical_radius_parameter_nm"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in columns} for row in particle_records])
    with (args.out / "initial_components.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["particle_id", "component_label", "threshold_voxel_count", "h_volume_nm3", "equivalent_radius_nm", "centroid_nm", "semi_axes_nm", "axis_ratio", "principal_axes_rows"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(component_rows)
    counts: Dict[str, int] = {}
    for item in particle_records:
        label = f"{float(item['registered_radius_nm']):.1f}"
        counts[label] = counts.get(label, 0) + 1
    manifest = {
        "schema": SCHEMA, "fixture_kind": mode, "fixture_id": spec.get("fixture_id"),
        "validation_only": bool(spec.get("validation_only", True)),
        "source_spec_canonical_sha256": canonical_spec_digest(spec),
        "canonical_particle_order": [item["particle_id"] for item in particle_records],
        "selected_library": freeze, "physical_contract": {
            "temperature_C": float(spec["target"]["temperature_C"]), "dx_nm": dx_nm, "lambda_sm_nm": lambda_sm_nm,
            "elasticity_enabled": True, "elastic_boundary": "periodic_fixed_cell",
            "orientation_label": "variant_100_identity", "eigenstrain": [0.046, -0.022, -0.017, 0.0, 0.0, 0.0],
            "GP_enabled": False, "GP_birth_enabled": False, "GP_release_enabled": False,
            "external_source_enabled": False, "new_beta_nucleation_enabled": False,
        },
        "grid": {"Nx": grid[0], "Ny": grid[1], "Nz": grid[2], "dx_nm": dx_nm},
        "placement": spec["placement"], "particles": particle_records, "radius_counts": counts,
        "combination": {
            "phi": "1-product_j(1-phi_j), canonical particle-id order",
            "delta_C_relaxation": "sum_j(delta_C_relaxation_j), canonical particle-id order",
            "s0_composition_reference": "uniform_matrix_without_elastic_delta_C" if mode == "S0" else "not_applicable",
            "absolute_xB_superposed": False, "radial_interpolation_used": False,
            "spatial_resampling_used": False, "rotated_variant_used": False, "normalization_used": False,
        },
        "inventory": {
            "target_mean_C_B_tot": target_mean, "target_total_C_B_tot": target_total,
            "actual_mean_C_B_tot": float(np.mean(c_total, dtype=np.float64)),
            "actual_total_C_B_tot": float(np.sum(c_total, dtype=np.float64)),
            "relative_error": inventory_relative_error, "matrix_xB": baseline,
            "matrix_xAg": 2.0 * baseline / (2.0 + baseline), "reference_matrix_xB": spec["target"].get("matrix_xB_reference"),
            "unresolved_inventory": 0.0, "source_h_volume_sum_nm3": source_h_sum,
            "combined_h_volume_nm3": float(np.sum(h_total, dtype=np.float64) * dx_nm**3),
            "reference_E2_combined_h_volume_nm3": reference_e2_h_volume,
            "S0_uniform_radius_offset_nm": s0_uniform_radius_offset_nm,
            "beta_volume_fraction": float(np.mean(h_total, dtype=np.float64)),
        },
        "component_contract": {"h_threshold": threshold, "expected_count": len(particle_records), "actual_count": len(component_rows), "particle_to_component": support_to_component},
        "phase_storage_contract": {"alpha_floor": phase_floor, "minimum_alpha": float(np.min(alpha)), "status": "NO_UNRESOLVED_DIVISION_REQUIRED"},
        "time_level_contract": {"Y": "logit(xB_alpha)", "dY_dt_prev": "zero_for_fresh_dynamic_start", "dY_dt_prev_nonzero": False},
        "fields": field_rows,
        "init_meta": {"path": "init_meta.json", "sha256": sha256(args.out / "init_meta.json")},
        "initial_particles": {"path": "initial_particles.csv", "sha256": sha256(args.out / "initial_particles.csv")},
        "initial_components": {"path": "initial_components.csv", "sha256": sha256(args.out / "initial_components.csv")},
    }
    write_json(args.out / "fixture_manifest.json", manifest)
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
        "status": "PASS_PF_ELASTIC_MULTI_PARTICLE_FIXTURE_MATERIALIZED_V1",
        "fixture_kind": manifest["fixture_kind"], "fixture_particle_count": len(manifest["particles"]),
        "fixture_beta_volume_fraction": manifest["inventory"]["beta_volume_fraction"],
        "fixture_matrix_xB": manifest["inventory"]["matrix_xB"],
        "fixture_inventory_relative_error": manifest["inventory"]["relative_error"],
        "fixture_manifest_sha256": sha256(args.out / "fixture_manifest.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
