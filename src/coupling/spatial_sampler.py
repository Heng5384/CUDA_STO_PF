"""Deterministic periodic sampling for the optional resolved-beta handoff mode.

The sampler is deliberately limited to spherical-equivalent resolved beta
seeds.  It does not implement GP-to-beta conversion, profile initialization,
or any PF materialization.  It simply turns a KWN expected-count histogram into
a reproducible, non-overlapping set of candidate seed centres.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


PF_RESOLUTION_MISMATCH = "PF_RESOLUTION_MISMATCH"


class SpatialSamplingError(RuntimeError):
    """A sampling failure whose ``status`` is safe to surface in an audit."""

    status = PF_RESOLUTION_MISMATCH


@dataclass(frozen=True)
class SampledResolvedBetaGeometry:
    """Resolved seed candidate geometry sampled from one KWN beta histogram."""

    radii_m: np.ndarray
    centers_m: np.ndarray
    source_bin_index: np.ndarray
    seed: int
    attempts: int
    expected_count_total: float
    integer_count_total: int
    target_volume_proxy_m3: float
    sampled_volume_proxy_m3: float
    uniform_radius_scale: float
    radius_scale_relative_change: float

    def as_arrays(self) -> dict:
        """Return array names directly compatible with the handoff schema."""

        return {
            "beta_resolved_sampled_radii_m": self.radii_m.copy(),
            "beta_resolved_centers_m": self.centers_m.copy(),
        }


def periodic_distance_m(
    first: Sequence[float], second: Sequence[float], box_lengths_m: Sequence[float]
) -> float:
    """Return the minimum-image Euclidean distance in a periodic cuboid."""

    if len(first) != 3 or len(second) != 3 or len(box_lengths_m) != 3:
        raise ValueError("positions and box_lengths_m must each have length three")
    squared = 0.0
    for lhs, rhs, length in zip(first, second, box_lengths_m):
        if not math.isfinite(float(length)) or float(length) <= 0.0:
            raise ValueError("box_lengths_m must be finite and positive")
        delta = abs(float(lhs) - float(rhs)) % float(length)
        delta = min(delta, float(length) - delta)
        squared += delta * delta
    return math.sqrt(squared)


def validate_periodic_nonoverlap(
    radii_m: Sequence[float],
    centers_m: Sequence[Sequence[float]],
    box_lengths_m: Sequence[float],
    clearance_m: float = 0.0,
) -> None:
    """Raise if any two spherical-equivalent seeds overlap across a periodic face."""

    if len(radii_m) != len(centers_m):
        raise ValueError("radii_m and centers_m must have the same length")
    if clearance_m < 0.0 or not math.isfinite(clearance_m):
        raise ValueError("clearance_m must be finite and non-negative")
    for index, radius in enumerate(radii_m):
        if not math.isfinite(float(radius)) or float(radius) <= 0.0:
            raise ValueError("radii_m must be finite and positive")
        if len(centers_m[index]) != 3:
            raise ValueError("every centre must have three coordinates")
        for coordinate, length in zip(centers_m[index], box_lengths_m):
            if not math.isfinite(float(coordinate)) or not 0.0 <= float(coordinate) < float(length):
                raise ValueError("every centre must lie inside the periodic box")
    for left in range(len(radii_m)):
        for right in range(left + 1, len(radii_m)):
            separation = periodic_distance_m(
                centers_m[left], centers_m[right], box_lengths_m
            )
            required = float(radii_m[left]) + float(radii_m[right]) + clearance_m
            if separation <= required:
                raise SpatialSamplingError(
                    f"periodic overlap between sampled seeds {left} and {right}: "
                    f"separation={separation:.6e} m, required>{required:.6e} m"
                )


def _validate_histogram(
    radius_bin_edges_m: Sequence[float], expected_count: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(radius_bin_edges_m, dtype=float)
    counts = np.asarray(expected_count, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("radius_bin_edges_m must contain at least two one-dimensional edges")
    if counts.ndim != 1 or counts.size != edges.size - 1:
        raise ValueError("expected_count must have one value per radius bin")
    if not np.all(np.isfinite(edges)) or np.any(edges <= 0.0) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("radius_bin_edges_m must be finite, positive, and strictly increasing")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("expected_count must be finite and non-negative")
    return edges, counts


def integerize_expected_counts(expected_count: Sequence[float]) -> np.ndarray:
    """Use deterministic largest-remainder integerization without superparticles."""

    expected = np.asarray(expected_count, dtype=float)
    if expected.ndim != 1 or not np.all(np.isfinite(expected)) or np.any(expected < 0.0):
        raise ValueError("expected_count must be a finite non-negative one-dimensional array")
    floors = np.floor(expected).astype(int)
    desired_total = int(math.floor(float(expected.sum()) + 0.5))
    remaining = desired_total - int(floors.sum())
    if remaining > 0:
        fractions = expected - floors
        # Stable bin-index tie break keeps this reproducible across platforms.
        order = sorted(range(expected.size), key=lambda index: (-fractions[index], index))
        for index in order[:remaining]:
            floors[index] += 1
    return floors


def _geometric_midpoints(edges: np.ndarray) -> np.ndarray:
    return np.sqrt(edges[:-1] * edges[1:])


def sample_resolved_beta_geometry(
    radius_bin_edges_m: Sequence[float],
    expected_count: Sequence[float],
    box_lengths_m: Sequence[float],
    seed: int,
    clearance_m: float = 0.0,
    max_attempts_per_particle: int = 10_000,
    max_particle_count: Optional[int] = None,
    max_radius_scale_relative_change: float = 0.02,
) -> SampledResolvedBetaGeometry:
    """Sample periodic, hard-core centres for a resolved KWN beta population.

    Radius-bin integerization is followed by one uniform radius correction that
    preserves the KWN spherical volume proxy.  A correction above two percent
    (by default), a count beyond the declared PF limit, or placement failure is
    reported as :data:`PF_RESOLUTION_MISMATCH`; no particles are silently
    deleted and no representative superparticles are created.
    """

    edges, expected = _validate_histogram(radius_bin_edges_m, expected_count)
    lengths = tuple(float(value) for value in box_lengths_m)
    if len(lengths) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in lengths):
        raise ValueError("box_lengths_m must contain three finite positive values")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if clearance_m < 0.0 or not math.isfinite(clearance_m):
        raise ValueError("clearance_m must be finite and non-negative")
    if max_attempts_per_particle <= 0:
        raise ValueError("max_attempts_per_particle must be positive")
    if max_particle_count is not None and max_particle_count < 0:
        raise ValueError("max_particle_count must be non-negative when supplied")
    if max_radius_scale_relative_change < 0.0:
        raise ValueError("max_radius_scale_relative_change must be non-negative")

    integer_counts = integerize_expected_counts(expected)
    expected_total = float(expected.sum())
    integer_total = int(integer_counts.sum())
    midpoints = _geometric_midpoints(edges)
    target_proxy = float((4.0 * math.pi / 3.0) * np.sum(expected * midpoints ** 3))
    unscaled_radii = np.repeat(midpoints, integer_counts)
    unscaled_proxy = float((4.0 * math.pi / 3.0) * np.sum(unscaled_radii ** 3))

    if target_proxy > 0.0 and integer_total == 0:
        raise SpatialSamplingError(
            "a non-zero resolved inventory rounds to zero PF particles; "
            "use a larger PF box or retain it in the sub-grid ledger"
        )
    if max_particle_count is not None and integer_total > max_particle_count:
        raise SpatialSamplingError(
            f"integerized resolved count {integer_total} exceeds PF limit {max_particle_count}"
        )
    if target_proxy == 0.0:
        return SampledResolvedBetaGeometry(
            radii_m=np.empty(0, dtype=float),
            centers_m=np.empty((0, 3), dtype=float),
            source_bin_index=np.empty(0, dtype=int),
            seed=seed,
            attempts=0,
            expected_count_total=expected_total,
            integer_count_total=integer_total,
            target_volume_proxy_m3=0.0,
            sampled_volume_proxy_m3=0.0,
            uniform_radius_scale=1.0,
            radius_scale_relative_change=0.0,
        )

    scale = (target_proxy / unscaled_proxy) ** (1.0 / 3.0)
    relative_change = abs(scale - 1.0)
    if relative_change > max_radius_scale_relative_change:
        raise SpatialSamplingError(
            "integerization would require a uniform radius correction of "
            f"{relative_change:.3%}, exceeding the {max_radius_scale_relative_change:.3%} limit"
        )
    radii = unscaled_radii * scale
    if np.any(radii < edges[0]) or np.any(radii > edges[-1]):
        raise SpatialSamplingError(
            "uniform inventory-preserving radius correction leaves the declared KWN radius range"
        )
    if any(2.0 * float(radius) + clearance_m > min(lengths) for radius in radii):
        raise SpatialSamplingError("at least one sampled seed self-overlaps through the periodic box")

    source_bin_index = np.repeat(np.arange(integer_counts.size, dtype=int), integer_counts)
    # Place large seeds first, then restore bin order in the final output.
    placement_order = sorted(range(integer_total), key=lambda index: -float(radii[index]))
    rng = random.Random(seed)
    placed_centers: List[Optional[Tuple[float, float, float]]] = [None] * integer_total
    placed_indices: List[int] = []
    attempts = 0
    for index in placement_order:
        found = False
        for _ in range(max_attempts_per_particle):
            attempts += 1
            candidate = tuple(rng.random() * length for length in lengths)
            if all(
                periodic_distance_m(candidate, placed_centers[other], lengths)
                > float(radii[index]) + float(radii[other]) + clearance_m
                for other in placed_indices
            ):
                placed_centers[index] = candidate
                placed_indices.append(index)
                found = True
                break
        if not found:
            raise SpatialSamplingError(
                "could not place all resolved seeds without periodic overlap; "
                "this PF box/PSD combination is not resolvable under the declared hard-core rule"
            )
    centers = np.asarray(placed_centers, dtype=float)
    validate_periodic_nonoverlap(radii, centers, lengths, clearance_m)
    sampled_proxy = float((4.0 * math.pi / 3.0) * np.sum(radii ** 3))
    return SampledResolvedBetaGeometry(
        radii_m=radii,
        centers_m=centers,
        source_bin_index=source_bin_index,
        seed=seed,
        attempts=attempts,
        expected_count_total=expected_total,
        integer_count_total=integer_total,
        target_volume_proxy_m3=target_proxy,
        sampled_volume_proxy_m3=sampled_proxy,
        uniform_radius_scale=scale,
        radius_scale_relative_change=relative_change,
    )
