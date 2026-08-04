#!/usr/bin/env python3
"""Materialize one hash-pinned 246^3/96-particle conditional handoff.

Only exact entries from the frozen 96^3 elastic target-profile library are
used.  Each native binary64 profile is embedded without interpolation,
resampling, scaling, rotation, or analytic replacement.  The physical support
contract is h(phi)>1e-4; the complete native 96^3 phi and portable
delta_C_relaxation windows are nevertheless retained so their small elastic
tails enter the conditional post-handoff evolution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import ndimage

import materialize_pf_elastic_multi_particle_6h_fixture_v1 as library_tools


SCHEMA = "PF_246CUBE_LIBRARY_HANDOFF_MANIFEST_V1"
INITIAL_STATE_CLASS = (
    "MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1"
)
HISTORICAL_FILE_SHA256 = (
    "f03ca43771200490cc0ff83646a86c8041fc65427b44ec2c3e71772f5b1d3f50"
)
HISTORICAL_CANONICAL_SHA256 = (
    "8ce52733e6755b23f4f7d4ba53f0414ef927bb863c3eb4ce9d43cb09d4d5de45"
)
DEFAULT_LIBRARY_SHA256 = (
    "58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe"
)
DEFAULT_SELECTION_SHA256 = (
    "56c44d8f72b27bb462dffe89b59cb2fcb2ff0bf8807dec9d9cac31d2bd7fcbe3"
)
LIBRARY_SOURCE_TREE_SHA256 = (
    "f7855699addf98f9d5aed03af876d851d524fb62c98f5a48df78d1fe561a2a75"
)
DEFAULT_REGISTERED_RADII_NM = (8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5)
INVENTORY_SELECTION_SCHEMA = "PF_246CUBE_LIBRARY_INVENTORY_SELECTION_V1"
EXPECTED_HISTOGRAM = {
    "8.0": 4,
    "8.5": 19,
    "9.0": 19,
    "9.5": 17,
    "10.0": 17,
    "10.5": 16,
    "11.0": 4,
    "11.5": 0,
}
REPLICATES = ("replicate_A", "replicate_B", "replicate_C")
GRID = (246, 246, 246)
DX_NM = 1.0
LAMBDA_SM_NM = 4.0
TARGET_MEAN_C = 0.03
COMPONENT_H_THRESHOLD = 1.0e-4
X_B_MAX_SAFE = 0.499999
PHI_NUMERICAL_MIN = -1.0e-6
PHI_NUMERICAL_MAX = 1.0 + 1.0e-6
NATIVE_GRID = (96, 96, 96)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    raw = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(memoryview(raw).cast("B")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_raw(path: Path, value: np.ndarray) -> None:
    np.asarray(value, dtype="<f8").ravel(order="C").tofile(path)


def radius_key(radius_nm: float) -> str:
    return str(float(radius_nm))


def quantize_radius(
    radius_nm: float, registered_radii_nm: Sequence[float]
) -> float:
    return min(
        registered_radii_nm,
        key=lambda candidate: (abs(radius_nm - candidate), candidate),
    )


def periodic_vector(
    a: Sequence[float], b: Sequence[float], lengths: Sequence[float]
) -> np.ndarray:
    delta = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    periods = np.asarray(lengths, dtype=np.float64)
    return (delta + 0.5 * periods) % periods - 0.5 * periods


def periodic_distance(
    a: Sequence[float], b: Sequence[float], lengths: Sequence[float]
) -> float:
    return float(np.linalg.norm(periodic_vector(a, b, lengths)))


def replicate_seed(label: str) -> Tuple[str, str, int]:
    material = f"{HISTORICAL_CANONICAL_SHA256}|{label}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return material, digest, int(digest[:16], 16)


def load_historical(
    path: Path, registered_radii_nm: Sequence[float]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    if sha256(path) != HISTORICAL_FILE_SHA256:
        raise ValueError("historical fixture file SHA-256 mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("canonical_manifest_sha256") != HISTORICAL_CANONICAL_SHA256:
        raise ValueError("historical canonical manifest identity mismatch")
    source = sorted(
        (dict(row) for row in manifest.get("particles", [])),
        key=lambda row: int(row["particle_id"]),
    )
    if len(source) != 96:
        raise ValueError("historical PSD does not contain 96 particles")
    particles: List[Dict[str, Any]] = []
    original_r3 = 0.0
    quantized_r3 = 0.0
    for index, row in enumerate(source):
        radius = float(row["radius_nm"])
        h_volume = float(row["h_volume_nm3"])
        if not math.isfinite(radius):
            raise ValueError("historical PSD contains a non-finite radius")
        if not math.isfinite(h_volume) or h_volume <= 0.0:
            raise ValueError("historical PSD contains an invalid h-volume")
        registered = quantize_radius(radius, registered_radii_nm)
        original_r3 += radius**3
        quantized_r3 += registered**3
        particles.append(
            {
                "particle_id": f"P{index:03d}",
                "historical_particle_id": int(row["particle_id"]),
                "historical_radius_nm": radius,
                "historical_h_volume_nm3": h_volume,
                "historical_effective_h_radius_nm": (
                    3.0 * h_volume / (4.0 * math.pi)
                ) ** (1.0 / 3.0),
                "nearest_registered_radius_nm": registered,
                "registered_radius_nm": registered,
                "orientation_label": "variant_100_identity",
            }
        )
    histogram = Counter(radius_key(row["registered_radius_nm"]) for row in particles)
    full_histogram = {
        radius_key(radius): int(histogram.get(radius_key(radius), 0))
        for radius in registered_radii_nm
    }
    if (
        tuple(registered_radii_nm) == DEFAULT_REGISTERED_RADII_NM
        and full_histogram != EXPECTED_HISTOGRAM
    ):
        raise ValueError(
            f"quantized PSD histogram mismatch: {full_histogram}"
        )
    audit = {
        "historical_fixture_file_sha256": HISTORICAL_FILE_SHA256,
        "historical_canonical_manifest_sha256": HISTORICAL_CANONICAL_SHA256,
        "historical_radius_min_nm": min(row["historical_radius_nm"] for row in particles),
        "historical_radius_max_nm": max(row["historical_radius_nm"] for row in particles),
        "historical_mean_R3_nm3": original_r3 / len(particles),
        "discrete_mean_R3_nm3": quantized_r3 / len(particles),
        "relative_change_of_mean_R3": (
            quantized_r3 - original_r3
        ) / original_r3,
        "histogram": full_histogram,
    }
    return manifest, particles, audit


def apply_inventory_selection(
    particles: List[Dict[str, Any]],
    psd_audit: Dict[str, Any],
    path: Path | None,
    registered_radii_nm: Sequence[float],
    library_sha256: str,
    selection_sha256: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any] | None]:
    """Apply a hash-pinned integer radius-bin selection without profile scaling.

    Transitions are applied to the particles with the largest historical
    h-equivalent radius in each source bin.  This makes the particle-level
    assignment deterministic and minimizes the radius mismatch within the
    frozen histogram transition plan.
    """
    if path is None:
        return particles, psd_audit, None
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("schema") != INVENTORY_SELECTION_SCHEMA:
        raise ValueError("inventory-selection schema mismatch")
    identities = selection.get("source_identities", {})
    expected_identities = {
        "historical_fixture_file_sha256": HISTORICAL_FILE_SHA256,
        "historical_canonical_manifest_sha256": HISTORICAL_CANONICAL_SHA256,
        "profile_library_manifest_sha256": library_sha256,
    }
    if "profile_library_selection_provenance_sha256" in identities:
        expected_identities[
            "profile_library_selection_provenance_sha256"
        ] = selection_sha256
    for key, expected in expected_identities.items():
        if identities.get(key) != expected:
            raise ValueError(
                f"inventory-selection source identity mismatch for {key}"
            )
    if selection.get("particle_count") != len(particles):
        raise ValueError("inventory-selection particle count mismatch")
    declared_radii = tuple(
        float(value)
        for value in selection.get("registered_radii_nm", registered_radii_nm)
    )
    if declared_radii != tuple(registered_radii_nm):
        raise ValueError("inventory-selection registered radius ladder mismatch")
    declared_histogram = {
        radius_key(float(key)): int(value)
        for key, value in selection["selected_histogram"].items()
    }
    if sum(declared_histogram.values()) != len(particles):
        raise ValueError("inventory-selection histogram particle count mismatch")
    if set(declared_histogram) != {radius_key(value) for value in registered_radii_nm}:
        raise ValueError("inventory-selection histogram radius set mismatch")

    applied: List[Dict[str, Any]] = []
    if selection.get("transitions"):
        source_histogram = {
            radius_key(float(key)): int(value)
            for key, value in selection.get("source_histogram", {}).items()
        }
        if source_histogram != psd_audit["histogram"]:
            raise ValueError("inventory-selection source histogram mismatch")
        selected = [dict(row) for row in particles]
        moved: set[str] = set()
        for transition in selection["transitions"]:
            source = float(transition["from_radius_nm"])
            target = float(transition["to_radius_nm"])
            count = int(transition["count"])
            if (
                source not in registered_radii_nm
                or target not in registered_radii_nm
                or count <= 0
            ):
                raise ValueError("invalid inventory-selection transition")
            candidates = sorted(
                (
                    row
                    for row in selected
                    if float(row["nearest_registered_radius_nm"]) == source
                    and row["particle_id"] not in moved
                ),
                key=lambda row: (
                    -float(row["historical_effective_h_radius_nm"]),
                    row["particle_id"],
                ),
            )
            if len(candidates) < count:
                raise ValueError(
                    "inventory-selection transition exceeds source bin"
                )
            chosen = candidates[:count]
            for row in chosen:
                row["registered_radius_nm"] = target
                moved.add(str(row["particle_id"]))
            applied.append(
                {
                    "from_radius_nm": source,
                    "to_radius_nm": target,
                    "count": count,
                    "selected_particle_ids": [
                        row["particle_id"] for row in chosen
                    ],
                }
            )
    else:
        selected = sorted(
            (dict(row) for row in particles),
            key=lambda row: (
                float(row["historical_effective_h_radius_nm"]),
                row["particle_id"],
            ),
        )
        cursor = 0
        for radius in registered_radii_nm:
            count = declared_histogram[radius_key(radius)]
            for row in selected[cursor : cursor + count]:
                row["registered_radius_nm"] = float(radius)
            cursor += count
        if cursor != len(selected):
            raise ValueError(
                "inventory-selection histogram assignment is incomplete"
            )
        selected.sort(key=lambda row: row["particle_id"])

    histogram = Counter(radius_key(row["registered_radius_nm"]) for row in selected)
    full_histogram = {
        radius_key(radius): int(histogram.get(radius_key(radius), 0))
        for radius in registered_radii_nm
    }
    if full_histogram != declared_histogram:
        raise ValueError("inventory-selection selected histogram mismatch")
    selected_r3 = sum(float(row["registered_radius_nm"]) ** 3 for row in selected)
    selected_h_volume = 4.0 * math.pi * selected_r3 / 3.0
    target_h_volume = float(selection["target_effective_h_volume_nm3"])
    declared_selected_h_volume = float(selection["selected_effective_h_volume_nm3"])
    if abs(selected_h_volume - declared_selected_h_volume) > 1.0e-6:
        raise ValueError("inventory-selection selected h-volume identity mismatch")
    updated_audit = {
        **psd_audit,
        "nearest_radius_histogram": psd_audit["histogram"],
        "histogram": full_histogram,
        "inventory_selection_applied": True,
        "inventory_selection_sha256": sha256(path),
        "inventory_selection_policy": selection["selection_policy"],
        "inventory_selection_assignment_rule": selection[
            "particle_assignment_rule"
        ],
        "inventory_selection_transitions": applied,
        "selected_discrete_mean_R3_nm3": selected_r3 / len(selected),
        "selected_effective_h_volume_nm3": selected_h_volume,
        "target_effective_h_volume_nm3": target_h_volume,
        "effective_h_volume_error_nm3": selected_h_volume - target_h_volume,
    }
    identity = {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "schema": selection["schema"],
        "selection_policy": selection["selection_policy"],
        "target_mean_C_B_tot": float(selection["target_mean_C_B_tot"]),
        "experimental_matrix_xAg_center": float(
            selection["experimental_matrix_xAg_center"]
        ),
        "experimental_matrix_xAg_interval": selection[
            "experimental_matrix_xAg_interval"
        ],
        "profile_scaling_used": False,
        "interpolation_used": False,
    }
    return selected, updated_audit, identity


def source_window_indices(center: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.arange(NATIVE_GRID[0], dtype=np.int64) - NATIVE_GRID[0] // 2
    return tuple(
        (int(c) + offsets) % target for c, target in zip(center, GRID)
    )  # type: ignore[return-value]


def place_particles(
    particles: List[Dict[str, Any]],
    seed: int,
    support_radius_by_registered: Dict[float, float],
    max_trials: int = 200000,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    placement_order = sorted(
        particles,
        key=lambda row: (-float(row["registered_radius_nm"]), row["particle_id"]),
    )
    placed: List[Dict[str, Any]] = []
    for row in placement_order:
        radius = float(row["registered_radius_nm"])
        center = None
        trial_used = None
        for trial in range(1, max_trials + 1):
            candidate = [
                int(value)
                for value in rng.integers(0, GRID[0], size=3)
            ]
            if all(
                periodic_distance(candidate, other["center_grid"], GRID)
                > radius
                + float(other["registered_radius_nm"])
                + 4.0 * LAMBDA_SM_NM
                for other in placed
            ):
                center = candidate
                trial_used = trial
                break
        if center is None:
            raise ValueError(f"could not place {row['particle_id']}")
        item = dict(row)
        item["center_grid"] = center
        item["placement_trial"] = int(trial_used)
        item["support_radius_nm"] = support_radius_by_registered[radius]
        placed.append(item)
    canonical = sorted(placed, key=lambda row: row["particle_id"])
    distances: List[float] = []
    margins: List[float] = []
    support_margins: List[float] = []
    directions: List[np.ndarray] = []
    nearest = [math.inf] * len(canonical)
    for right in range(len(canonical)):
        for left in range(right):
            vector = periodic_vector(
                canonical[left]["center_grid"],
                canonical[right]["center_grid"],
                GRID,
            )
            distance = float(np.linalg.norm(vector))
            required = (
                float(canonical[left]["registered_radius_nm"])
                + float(canonical[right]["registered_radius_nm"])
                + 4.0 * LAMBDA_SM_NM
            )
            support_required = (
                float(canonical[left]["support_radius_nm"])
                + float(canonical[right]["support_radius_nm"])
            )
            if not distance > required:
                raise ValueError("periodic hard-core separation failed")
            if not distance > support_required:
                raise ValueError("physical profile supports overlap")
            distances.append(distance)
            margins.append(distance - required)
            support_margins.append(distance - support_required)
            directions.append(vector / distance)
            nearest[left] = min(nearest[left], distance)
            nearest[right] = min(nearest[right], distance)
    centers = np.asarray(
        [row["center_grid"] for row in canonical], dtype=np.float64
    )
    covariance = np.cov(centers, rowvar=False, ddof=1)
    covariance_eigenvalues = np.linalg.eigvalsh(covariance)
    nearest_array = np.asarray(nearest)
    direction_mean = np.mean(np.asarray(directions), axis=0)
    stats = {
        "method": "deterministic_periodic_hard_core_rejection_integer_grid_v1",
        "placement_order": "descending_registered_radius_then_particle_id",
        "max_trials_per_particle": max_trials,
        "minimum_periodic_pair_distance_nm": min(distances),
        "mean_periodic_pair_distance_nm": float(np.mean(distances)),
        "std_periodic_pair_distance_nm": float(np.std(distances)),
        "minimum_registered_separation_margin_nm": min(margins),
        "minimum_actual_support_margin_nm": min(support_margins),
        "nearest_neighbor_mean_nm": float(np.mean(nearest_array)),
        "nearest_neighbor_std_nm": float(np.std(nearest_array)),
        "nearest_neighbor_cv": float(
            np.std(nearest_array) / np.mean(nearest_array)
        ),
        "pair_direction_mean": direction_mean.tolist(),
        "center_covariance_matrix_grid2": covariance.tolist(),
        "center_covariance_eigenvalues_grid2": covariance_eigenvalues.tolist(),
        "regular_lattice_rejected": bool(
            np.std(nearest_array) > 1.0e-6
            and np.min(covariance_eigenvalues) > 0.0
        ),
        "periodic_image_overlap_detected": False,
        "periodic_image_overlap_status": "PASS",
    }
    if not stats["regular_lattice_rejected"]:
        raise ValueError("center set fails the non-lattice spatial contract")
    return canonical, stats


def periodic_labels(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    structure = np.zeros((3, 3, 3), dtype=np.int8)
    structure[1, 1, 1] = 1
    structure[0, 1, 1] = structure[2, 1, 1] = 1
    structure[1, 0, 1] = structure[1, 2, 1] = 1
    structure[1, 1, 0] = structure[1, 1, 2] = 1
    labels, count = ndimage.label(mask, structure=structure)
    parent = np.arange(count + 1, dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union_faces(left: np.ndarray, right: np.ndarray) -> None:
        active = (left > 0) & (right > 0)
        if not np.any(active):
            return
        pairs = np.unique(
            np.stack((left[active], right[active]), axis=1),
            axis=0,
        )
        for lhs, rhs in pairs:
            a, b = find(int(lhs)), find(int(rhs))
            if a != b:
                parent[b] = a

    union_faces(labels[0, :, :], labels[-1, :, :])
    union_faces(labels[:, 0, :], labels[:, -1, :])
    union_faces(labels[:, :, 0], labels[:, :, -1])
    root_to_dense: Dict[int, int] = {}
    lookup = np.zeros(count + 1, dtype=np.int32)
    for old in range(1, count + 1):
        root = find(old)
        if root not in root_to_dense:
            root_to_dense[root] = len(root_to_dense) + 1
        lookup[old] = root_to_dense[root]
    dense = lookup[labels]
    return dense, len(root_to_dense)


def group_component_indices(labels: np.ndarray) -> Dict[int, np.ndarray]:
    flat_labels = labels.ravel(order="C")
    active = np.flatnonzero(flat_labels)
    order = np.argsort(flat_labels[active], kind="stable")
    active = active[order]
    sorted_labels = flat_labels[active]
    boundaries = np.flatnonzero(np.diff(sorted_labels)) + 1
    groups = np.split(active, boundaries)
    return {
        int(flat_labels[indices[0]]): indices
        for indices in groups
        if indices.size
    }


def component_metrics(
    indices: np.ndarray,
    component: int,
    h: np.ndarray,
) -> Dict[str, Any]:
    if not indices.size:
        raise ValueError("empty component")
    weights = h.ravel(order="C")[indices]
    coords = np.column_stack(np.unravel_index(indices, GRID)).astype(np.float64)
    centroid: List[float] = []
    displacements: List[np.ndarray] = []
    for axis, length in enumerate(GRID):
        angles = 2.0 * math.pi * coords[:, axis] / length
        angle = math.atan2(
            float(np.sum(weights * np.sin(angles))),
            float(np.sum(weights * np.cos(angles))),
        ) % (2.0 * math.pi)
        center = angle * length / (2.0 * math.pi)
        centroid.append(center)
        displacements.append(
            (coords[:, axis] - center + 0.5 * length) % length
            - 0.5 * length
        )
    covariance = np.empty((3, 3), dtype=np.float64)
    weight_sum = float(np.sum(weights, dtype=np.float64))
    for left in range(3):
        for right in range(left, 3):
            value = float(
                np.sum(
                    weights * displacements[left] * displacements[right],
                    dtype=np.float64,
                )
                / weight_sum
            )
            covariance[left, right] = covariance[right, left] = value
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if np.any(eigenvalues <= 0.0):
        raise ValueError("component shape tensor is not positive definite")
    semiaxes = np.sqrt(5.0 * eigenvalues)
    volume = weight_sum * DX_NM**3
    return {
        "component_label": component - 1,
        "threshold_voxel_count": int(indices.size),
        "h_volume_nm3": volume,
        "equivalent_radius_nm": (
            3.0 * volume / (4.0 * math.pi)
        ) ** (1.0 / 3.0),
        "centroid_nm": [value * DX_NM for value in centroid],
        "semi_axes_nm": (semiaxes * DX_NM).tolist(),
        "axis_ratio": float(semiaxes[0] / semiaxes[-1]),
        "principal_axes_rows": eigenvectors.T.tolist(),
    }


def load_library_profiles(
    library_root: Path,
    selection_provenance: Path,
    library_sha256: str,
    selection_sha256: str,
    registered_radii_nm: Sequence[float],
    expected_source_tree_sha256: str,
) -> Tuple[Dict[float, Dict[str, Any]], Dict[str, Any]]:
    _library_path, library, profiles, freeze = library_tools.verify_library(
        library_root,
        library_sha256,
        selection_provenance,
        selection_sha256,
        registered_radii_nm,
    )
    if freeze.get("source_tree_sha256") != expected_source_tree_sha256:
        raise ValueError("frozen library source-tree mismatch")
    cache: Dict[float, Dict[str, Any]] = {}
    native_coords = np.indices(NATIVE_GRID).transpose(1, 2, 3, 0)
    native_center = np.asarray([48, 48, 48])
    native_delta = (
        native_coords - native_center + 48
    ) % 96 - 48
    native_radius = np.sqrt(
        np.sum(native_delta.astype(np.float64) ** 2, axis=-1)
    )
    for radius, (manifest_path, manifest) in profiles.items():
        phi_path = library_tools.check_field(
            manifest, manifest_path, "phi", NATIVE_GRID
        )
        h_path = library_tools.check_field(
            manifest, manifest_path, "h_phi", NATIVE_GRID
        )
        delta_path = library_tools.check_field(
            manifest, manifest_path, "delta_C_relaxation", NATIVE_GRID
        )
        phi = library_tools.read_raw(phi_path, NATIVE_GRID, "phi")
        h = library_tools.read_raw(h_path, NATIVE_GRID, "h_phi")
        delta_c = library_tools.read_raw(
            delta_path, NATIVE_GRID, "delta_C_relaxation"
        )
        if float(np.max(np.abs(h - library_tools.h_of_phi(phi)))) > 5.0e-14:
            raise ValueError(f"R={radius}: source h(phi) mismatch")
        alpha = 1.0 - h
        if float(np.min(alpha)) <= 0.0:
            raise ValueError(f"R={radius}: zero source alpha")
        support = h > COMPONENT_H_THRESHOLD
        if not np.any(support):
            raise ValueError(f"R={radius}: empty registered support")
        cache[radius] = {
            "manifest_path": manifest_path,
            "manifest": manifest,
            "phi": phi,
            "h": h,
            "delta_x": delta_c / alpha,
            "support": support,
            "support_radius_nm": float(np.max(native_radius[support])),
            "profile_manifest_sha256": sha256(manifest_path),
            "source_phi_sha256": manifest["fields"]["phi"]["sha256"],
            "source_delta_C_relaxation_sha256": manifest["fields"][
                "delta_C_relaxation"
            ]["sha256"],
        }
    return cache, {
        "selected_library_manifest_sha256": library_sha256,
        "selection_provenance_sha256": selection_sha256,
        "source_tree_sha256": library["source_tree_sha256"],
        "binary_sha256": library["binary_sha256"],
        "binary_sha256_set": library.get(
            "binary_sha256_set", [library["binary_sha256"]]
        ),
        "mixed_binary_profiles": library.get("mixed_binary_profiles", False),
        "mixed_binary_contract": library.get(
            "mixed_binary_contract", "SINGLE_BINARY"
        ),
    }


def ordered_particles(
    particles: List[Dict[str, Any]], input_order: str
) -> List[Dict[str, Any]]:
    if input_order == "canonical":
        supplied = list(particles)
    elif input_order == "reverse":
        supplied = list(reversed(particles))
    else:
        raise ValueError("unknown input-order probe")
    # Manifest order is never physics: assembly always uses this canonical key.
    return sorted(supplied, key=lambda row: row["particle_id"])


def assemble(
    historical_manifest: Path,
    library_root: Path,
    selection_provenance: Path,
    replicate: str,
    input_order: str,
    inventory_selection: Path | None = None,
    library_sha256: str = DEFAULT_LIBRARY_SHA256,
    selection_sha256: str = DEFAULT_SELECTION_SHA256,
    registered_radii_nm: Sequence[float] = DEFAULT_REGISTERED_RADII_NM,
    expected_source_tree_sha256: str = LIBRARY_SOURCE_TREE_SHA256,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if replicate not in REPLICATES:
        raise ValueError(f"unknown replicate: {replicate}")
    _history, source_particles, psd_audit = load_historical(
        historical_manifest, registered_radii_nm
    )
    source_particles, psd_audit, inventory_selection_identity = (
        apply_inventory_selection(
            source_particles,
            psd_audit,
            inventory_selection,
            registered_radii_nm,
            library_sha256,
            selection_sha256,
        )
    )
    profiles, library_identity = load_library_profiles(
        library_root,
        selection_provenance,
        library_sha256,
        selection_sha256,
        registered_radii_nm,
        expected_source_tree_sha256,
    )
    support_radii = {
        radius: float(row["support_radius_nm"])
        for radius, row in profiles.items()
    }
    seed_material, seed_sha256, seed = replicate_seed(replicate)
    placed, spatial = place_particles(
        source_particles, seed, support_radii
    )
    particles = ordered_particles(placed, input_order)

    phi_complement = np.ones(GRID, dtype=np.float64)
    delta_x_total = np.zeros(GRID, dtype=np.float64)
    source_h_sum = np.zeros(GRID, dtype=np.float64)
    support_owner = np.zeros(GRID, dtype=np.uint8)
    for particle in particles:
        profile = profiles[float(particle["registered_radius_nm"])]
        ix = np.ix_(*source_window_indices(particle["center_grid"]))
        local_owner = support_owner[ix]
        support = profile["support"]
        if np.any(local_owner[support]):
            raise ValueError("embedded physical profile supports overlap")
        local_owner[support] = 1
        support_owner[ix] = local_owner
        local_product = phi_complement[ix]
        local_product *= 1.0 - profile["phi"]
        phi_complement[ix] = local_product
        local_delta = delta_x_total[ix]
        local_delta += profile["delta_x"]
        delta_x_total[ix] = local_delta
        local_h_sum = source_h_sum[ix]
        local_h_sum += profile["h"]
        source_h_sum[ix] = local_h_sum

    phi = 1.0 - phi_complement
    h = library_tools.h_of_phi(phi)
    if (
        not np.all(np.isfinite(phi))
        or float(np.min(phi)) < 0.0
        or float(np.max(phi)) > 1.0
    ):
        raise ValueError("assembled phi violates the fresh-field contract")
    alpha = 1.0 - h
    if float(np.min(alpha)) <= 0.0:
        raise ValueError("assembled field has zero unresolved alpha")
    if np.any((h > 0.0) & (source_h_sum <= 0.0)):
        raise ValueError("assembled beta tail cannot be allocated")
    delta_c = alpha * delta_x_total
    target_total = TARGET_MEAN_C * math.prod(GRID)
    beta_inventory = float(np.sum(h, dtype=np.float64))
    relaxation_inventory = float(np.sum(delta_c, dtype=np.float64))
    effective_matrix_volume = float(np.sum(alpha, dtype=np.float64))
    baseline = (
        target_total - beta_inventory - relaxation_inventory
    ) / effective_matrix_volume
    if not math.isfinite(baseline) or not (0.0 < baseline < X_B_MAX_SAFE):
        raise ValueError("derived matrix baseline is outside the variable domain")
    x_b = baseline + delta_x_total
    if (
        not np.all(np.isfinite(x_b))
        or float(np.min(x_b)) <= 0.0
        or float(np.max(x_b)) >= X_B_MAX_SAFE
    ):
        raise ValueError("assembled xB would require clipping")
    c_total = h + alpha * x_b
    actual_total = float(np.sum(c_total, dtype=np.float64))
    inventory_error = abs(actual_total - target_total) / max(
        abs(target_total), 1.0
    )
    if inventory_error > 1.0e-14:
        raise ValueError("global canonical inventory does not close")
    y = library_tools.logit(x_b)
    d_y = np.zeros(GRID, dtype=np.float64)

    mapping_rows: List[Dict[str, Any]] = []
    assigned_beta = 0.0
    assigned_relaxation = 0.0
    for index, particle in enumerate(particles):
        profile = profiles[float(particle["registered_radius_nm"])]
        ix = np.ix_(*source_window_indices(particle["center_grid"]))
        denominator = source_h_sum[ix]
        share = np.zeros(NATIVE_GRID, dtype=np.float64)
        positive = denominator > 0.0
        share[positive] = (
            h[ix][positive] * profile["h"][positive] / denominator[positive]
        )
        beta_share = float(np.sum(share, dtype=np.float64))
        relaxation_share = float(
            np.sum(alpha[ix] * profile["delta_x"], dtype=np.float64)
        )
        # Close the particle decomposition exactly with the last canonical row
        # without changing any physical field.
        if index + 1 == len(particles):
            beta_share = beta_inventory - assigned_beta
            relaxation_share = relaxation_inventory - assigned_relaxation
        else:
            assigned_beta += beta_share
            assigned_relaxation += relaxation_share
        radius = float(particle["registered_radius_nm"])
        geometry = profile["manifest"]["geometry"]
        mapping_rows.append(
            {
                **particle,
                "replicate_id": replicate,
                "center_nm": [
                    float(value) * DX_NM
                    for value in particle["center_grid"]
                ],
                "orientation": particle["orientation_label"],
                "library_entry_id": f"R{radius_key(radius)}".replace(".", "p"),
                "library_entry_sha256": profile[
                    "profile_manifest_sha256"
                ],
                "source_phi_sha256": profile["source_phi_sha256"],
                "source_delta_C_relaxation_sha256": profile[
                    "source_delta_C_relaxation_sha256"
                ],
                "effective_h_volume": float(
                    geometry.get(
                        "final_h_volume_nm3",
                        geometry["h_volume_nm3"],
                    )
                ),
                "assembled_beta_inventory": beta_share,
                "assembled_relaxation_inventory": relaxation_share,
                "canonical_particle_inventory": beta_share
                + relaxation_share,
            }
        )
    particle_inventory = sum(
        row["canonical_particle_inventory"] for row in mapping_rows
    )
    matrix_baseline_inventory = baseline * effective_matrix_volume
    decomposition_error = abs(
        matrix_baseline_inventory + particle_inventory - target_total
    ) / max(abs(target_total), 1.0)
    if decomposition_error > 1.0e-14:
        raise ValueError("particle/matrix ledger decomposition does not close")

    labels, count = periodic_labels(h > COMPONENT_H_THRESHOLD)
    if count != 96:
        raise ValueError(f"initial component count={count}, expected=96")
    indices_by_component = group_component_indices(labels)
    if set(indices_by_component) != set(range(1, count + 1)):
        raise ValueError("component index grouping is incomplete")
    used_components: set[int] = set()
    components: List[Dict[str, Any]] = []
    for row in mapping_rows:
        center = tuple(int(value) for value in row["center_grid"])
        component = int(labels[center])
        if component <= 0 or component in used_components:
            raise ValueError("particle/component mapping is not bijective")
        used_components.add(component)
        metrics = component_metrics(
            indices_by_component[component], component, h
        )
        metrics["particle_id"] = row["particle_id"]
        components.append(metrics)
    if len(used_components) != 96:
        raise ValueError("particle/component mapping is incomplete")

    matrix_mask = h < 0.005
    matrix_x_b = float(np.mean(x_b[matrix_mask], dtype=np.float64))
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
        "fixture_id": (
            f"pf_246cube_library_handoff_{replicate}_inventory_selected_v1"
            if inventory_selection_identity is not None
            else f"pf_246cube_library_handoff_{replicate}_v1"
        ),
        "replicate_id": replicate,
        "validation_only": True,
        "scientific_semantics": (
            "mass-conserving, exact-profile-library-assembled conditional "
            "6 h handoff state"
        ),
        "source_psd": psd_audit,
        "inventory_selection": inventory_selection_identity,
        "profile_library_manifest_sha256": library_sha256,
        "selection_provenance_sha256": selection_sha256,
        "source_tree_sha256": expected_source_tree_sha256,
        "profile_library_binary_sha256": library_identity["binary_sha256"],
        "profile_library_binary_sha256_set": library_identity[
            "binary_sha256_set"
        ],
        "profile_library_mixed_binary_contract": library_identity[
            "mixed_binary_contract"
        ],
        "canonical_particle_order": [
            row["particle_id"] for row in mapping_rows
        ],
        "particle_library_mappings": mapping_rows,
        "target_global_inventory": {
            "mean_C_B_tot": TARGET_MEAN_C,
            "total_C_B_tot_code": target_total,
            "source": "hash_pinned_manifest",
        },
        "derived_matrix_baseline": {
            "xB_alpha": baseline,
            "xAg": 2.0 * baseline / (2.0 + baseline),
            "observed_matrix_xB_h_lt_0p005": matrix_x_b,
            "observed_matrix_xAg_h_lt_0p005": (
                2.0 * matrix_x_b / (2.0 + matrix_x_b)
            ),
            "effective_matrix_volume_code": effective_matrix_volume,
            "matrix_baseline_inventory_code": matrix_baseline_inventory,
            "clipping_used": False,
            "normalization_used": False,
        },
        "initial_canonical_inventory": {
            "target_total_code": target_total,
            "actual_total_code": actual_total,
            "matrix_baseline_inventory_code": matrix_baseline_inventory,
            "particle_inventory_sum_code": particle_inventory,
            "assembled_beta_phase_inventory_code": beta_inventory,
            "local_relaxation_inventory_code": relaxation_inventory,
            "field_relative_error": inventory_error,
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
            "Nx": GRID[0],
            "Ny": GRID[1],
            "Nz": GRID[2],
            "dx_nm": DX_NM,
            "lambda_sm_nm": LAMBDA_SM_NM,
        },
        "placement": {
            **spatial,
            "seed_material": seed_material,
            "seed_material_sha256": seed_sha256,
            "seed_unsigned64": seed,
            "periodic_separation_rule": (
                "distance_nm > Ri_nm + Rj_nm + 4*lambda_sm_nm"
            ),
            "physical_support_definition": (
                f"h(phi)>{COMPONENT_H_THRESHOLD:.1e}"
            ),
        },
        "physical_contract": {
            "temperature_C": 380.0,
            "elasticity_enabled": True,
            "elastic_boundary": "periodic_fixed_cell",
            "orientation_label": "variant_100_identity",
            "eigenstrain": [0.046, -0.022, -0.017, 0.0, 0.0, 0.0],
            "external_strain": [0.0] * 6,
            "external_stress": [0.0] * 6,
            "GP_enabled": False,
            "GP_birth_enabled": False,
            "GP_release_enabled": False,
            "external_source_enabled": False,
            "new_beta_nucleation_enabled": False,
            "initial_age_h": 6.0,
        },
        "assembly_contract": {
            "phi": "1-product_j(1-phi_j), canonical particle-id order",
            "portable_composition_field": (
                "native delta_C_relaxation_j reconstructed as "
                "delta_x_j=delta_C_j/(1-h_j), embedded without resampling, "
                "then stored as (1-h_total)*sum_j(delta_x_j)"
            ),
            "native_window_embedding": (
                "all 96^3 source values, minimum-image offsets [-48,47]"
            ),
            "absolute_xB_copied_across_boxes": False,
            "radial_interpolation_used": False,
            "profile_scaling_used": False,
            "spatial_resampling_used": False,
            "rotation_used": False,
            "analytic_tanh_used": False,
            "clipping_used": False,
            "normalization_used": False,
            "optimizer_invoked": inventory_selection_identity is not None,
            "common_multi_particle_pre_relaxation_run": False,
            "input_manifest_order": input_order,
            "canonical_assembly_order_enforced": True,
        },
        "component_contract": {
            "h_threshold": COMPONENT_H_THRESHOLD,
            "expected_count": 96,
            "actual_count": count,
            "particle_to_component": {
                row["particle_id"]: int(
                    next(
                        component["component_label"]
                        for component in components
                        if component["particle_id"] == row["particle_id"]
                    )
                )
                for row in mapping_rows
            },
        },
        "field_bounds": {
            "phi_min": float(np.min(phi)),
            "phi_max": float(np.max(phi)),
            "xB_min": float(np.min(x_b)),
            "xB_max": float(np.max(x_b)),
            "phi_numerical_contract": [
                PHI_NUMERICAL_MIN,
                PHI_NUMERICAL_MAX,
            ],
        },
        "selected_library": library_identity,
    }
    return fields, manifest, mapping_rows, components


def write_fixture(
    out: Path,
    fields: Dict[str, np.ndarray],
    manifest: Dict[str, Any],
    mappings: List[Dict[str, Any]],
    components: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if out.exists():
        raise ValueError(f"refusing to overwrite output: {out}")
    out.mkdir(parents=True)
    field_rows: Dict[str, Dict[str, str]] = {}
    for name, value in fields.items():
        path = out / f"{name}.raw.f64"
        write_raw(path, value)
        field_rows[name] = {
            "path": path.name,
            "sha256": sha256(path),
            "dtype": "float64-le",
            "order": "C",
        }
    manifest["fields"] = field_rows
    meta = {
        "schema": "PF_246CUBE_LIBRARY_HANDOFF_RAW_INIT_META_V1",
        "Nx": GRID[0],
        "Ny": GRID[1],
        "Nz": GRID[2],
        "dx_nm": DX_NM,
        "interface_width_nm": LAMBDA_SM_NM,
        "dt_recommended": 0.02,
        "mean_xBtot": manifest["target_global_inventory"]["mean_C_B_tot"],
        "xB_max_safe": X_B_MAX_SAFE,
        "dtype": "float64",
        "order": "C",
        "phi_path": field_rows["phi"]["path"],
        "xB_path": field_rows["xB_alpha"]["path"],
        "fresh_dY_dt_prev_contract": "zero_for_fresh_dynamic_start",
        "initial_state_class": INITIAL_STATE_CLASS,
    }
    write_json(out / "init_meta.json", meta)
    with (out / "initial_particles.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        columns = list(mappings[0])
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(mappings)
    with (out / "initial_components.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        columns = list(components[0])
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(components)
    manifest["init_meta"] = {
        "path": "init_meta.json",
        "sha256": sha256(out / "init_meta.json"),
    }
    manifest["initial_particles"] = {
        "path": "initial_particles.csv",
        "sha256": sha256(out / "initial_particles.csv"),
    }
    manifest["initial_components"] = {
        "path": "initial_components.csv",
        "sha256": sha256(out / "initial_components.csv"),
    }
    write_json(out / "fixture_manifest.json", manifest)
    return {
        "fixture_manifest_sha256": sha256(out / "fixture_manifest.json"),
        "field_hashes": {
            name: row["sha256"] for name, row in field_rows.items()
        },
    }


def compare_fixture(
    path: Path,
    fields: Dict[str, np.ndarray],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("schema") != SCHEMA:
        raise ValueError("comparison fixture uses wrong schema")
    mismatches = []
    for name, value in fields.items():
        actual = array_sha256(value)
        expected = frozen["fields"][name]["sha256"]
        if actual != expected:
            mismatches.append(name)
    semantic_keys = (
        "replicate_id",
        "canonical_particle_order",
        "source_psd",
        "target_global_inventory",
        "derived_matrix_baseline",
        "initial_canonical_inventory",
        "placement",
        "physical_contract",
        "component_contract",
        "field_bounds",
    )
    semantic_equal = all(
        manifest.get(key) == frozen.get(key) for key in semantic_keys
    )
    if mismatches or not semantic_equal:
        raise ValueError(
            f"deterministic comparison failed fields={mismatches} "
            f"semantic_equal={semantic_equal}"
        )
    return {
        "status": "PASS_DETERMINISTIC_MATERIALIZATION",
        "fixture_manifest_sha256": sha256(path),
        "input_order_probe": manifest["assembly_contract"][
            "input_manifest_order"
        ],
        "raw_field_hashes_equal": True,
        "semantic_manifest_equal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--selection-provenance", type=Path, required=True)
    parser.add_argument("--inventory-selection", type=Path)
    parser.add_argument(
        "--library-manifest-sha256", default=DEFAULT_LIBRARY_SHA256
    )
    parser.add_argument(
        "--selection-provenance-sha256", default=DEFAULT_SELECTION_SHA256
    )
    parser.add_argument(
        "--registered-radii-nm",
        type=float,
        nargs="+",
        default=DEFAULT_REGISTERED_RADII_NM,
    )
    parser.add_argument(
        "--expected-source-tree-sha256", default=LIBRARY_SOURCE_TREE_SHA256
    )
    parser.add_argument("--replicate", choices=REPLICATES, required=True)
    parser.add_argument(
        "--input-order", choices=("canonical", "reverse"), default="canonical"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path)
    group.add_argument("--compare-to-manifest", type=Path)
    args = parser.parse_args()
    try:
        fields, manifest, mappings, components = assemble(
            args.historical_manifest,
            args.library_root,
            args.selection_provenance,
            args.replicate,
            args.input_order,
            args.inventory_selection,
            args.library_manifest_sha256,
            args.selection_provenance_sha256,
            tuple(sorted(args.registered_radii_nm)),
            args.expected_source_tree_sha256,
        )
        if args.out is not None:
            identity = write_fixture(
                args.out, fields, manifest, mappings, components
            )
            result = {
                "status": "PASS_246CUBE_LIBRARY_HANDOFF_MATERIALIZED_V1",
                "replicate": args.replicate,
                "particle_count": len(mappings),
                "matrix_xB": manifest["derived_matrix_baseline"]["xB_alpha"],
                "inventory_relative_error": manifest[
                    "initial_canonical_inventory"
                ]["field_relative_error"],
                **identity,
            }
        else:
            result = compare_fixture(
                args.compare_to_manifest, fields, manifest
            )
    except ValueError as exc:
        raise SystemExit(f"[fatal] {exc}") from exc
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
