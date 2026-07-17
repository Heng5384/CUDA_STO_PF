#!/usr/bin/env python3
"""Conservative PF snapshot to particle-ledger handoff.

This is an offline, default-off analysis tool. It does not modify a PF field or
claim a calibrated KWN model. Ctot remains authoritative and the diffuse beta
inventory is counted exactly once through integral h(phi) * v_B.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage


def h_of_phi(phi: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(phi, dtype=np.float64), 0.0, 1.0)
    return p**3 * (6.0 * p**2 - 15.0 * p + 10.0)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n + 1))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        if a == 0 or b == 0:
            return
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def periodic_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    labels, count = ndimage.label(mask, structure=structure)
    uf = UnionFind(count)
    for axis in range(mask.ndim):
        lo = np.take(labels, 0, axis=axis)
        hi = np.take(labels, -1, axis=axis)
        for a, b in zip(lo.ravel(), hi.ravel()):
            uf.union(int(a), int(b))
    roots = sorted({uf.find(i) for i in range(1, count + 1)})
    dense = {root: i + 1 for i, root in enumerate(roots)}
    lut = np.zeros(count + 1, dtype=np.int32)
    for i in range(1, count + 1):
        lut[i] = dense[uf.find(i)]
    return lut[labels], len(roots)


def periodic_weighted_center(weights: np.ndarray, dx_nm: float) -> np.ndarray:
    center = []
    for axis, n in enumerate(weights.shape):
        marginal = weights.sum(axis=tuple(i for i in range(3) if i != axis))
        angle = 2.0 * math.pi * np.arange(n, dtype=np.float64) / n
        z = np.sum(marginal * np.exp(1j * angle))
        a = math.atan2(z.imag, z.real) % (2.0 * math.pi)
        center.append(a * n * dx_nm / (2.0 * math.pi))
    return np.asarray(center)


def periodic_delta(coords_nm: np.ndarray, center_nm: np.ndarray,
                   lengths_nm: np.ndarray) -> np.ndarray:
    delta = coords_nm - center_nm
    return (delta + 0.5 * lengths_nm) % lengths_nm - 0.5 * lengths_nm


@dataclass
class ParticleRecord:
    particle_id: int
    center_nm: list[float]
    h_volume_nm3: float
    beta_inventory: float
    equivalent_radius_nm: float
    threshold_volume_nm3: float
    surface_area_nm2: float
    shape_tensor_nm2: list[list[float]]
    principal_variances_nm2: list[float]
    orientation_columns: list[list[float]]
    local_matrix_xB: float
    local_chemical_potential: float | None
    elastic_environment_tag: str
    nearby_GP_inventory: float
    nearest_neighbor_distance_nm: float | None


def extract_handoff(
    ctot: np.ndarray,
    phi: np.ndarray,
    *,
    dx_nm: float,
    v_B: float = 1.0,
    gp_inventory: float = 0.0,
    threshold: float = 0.5,
    tail_eps: float = 1.0e-12,
    chemical_potential: np.ndarray | None = None,
) -> dict:
    ctot = np.asarray(ctot, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    if ctot.shape != phi.shape or ctot.ndim != 3:
        raise ValueError("Ctot and phi must be same-shape 3-D arrays")
    if not np.all(np.isfinite(ctot)) or not np.all(np.isfinite(phi)):
        raise ValueError("nonfinite PF snapshot")
    if not (dx_nm > 0.0) or not (v_B > 0.0):
        raise ValueError("dx_nm and v_B must be positive")

    h = h_of_phi(phi)
    qalpha = ctot - h * v_B
    bound_tol = 1.0e-12
    if float(np.min(qalpha)) < -bound_tol:
        raise ValueError(f"negative q_alpha in snapshot: {np.min(qalpha):.17e}")

    core_labels, particle_count = periodic_components(phi >= threshold)
    if particle_count == 0:
        raise ValueError("no resolved beta object at the selected threshold")

    # Assign every diffuse h contribution to exactly one resolved core. This
    # creates a particle inventory partition without changing the PF fields.
    grid = np.indices(phi.shape, dtype=np.float64)
    coords = np.stack([grid[i].ravel() * dx_nm for i in range(3)], axis=1)
    lengths = np.asarray(phi.shape, dtype=np.float64) * dx_nm
    centers = []
    for pid in range(1, particle_count + 1):
        centers.append(periodic_weighted_center((core_labels == pid).astype(float), dx_nm))
    centers_array = np.asarray(centers)
    assigned = np.zeros(phi.size, dtype=np.int32)
    active = h.ravel() > tail_eps
    active_coords = coords[active]
    distance2 = np.empty((active_coords.shape[0], particle_count), dtype=np.float64)
    for j, center in enumerate(centers_array):
        delta = periodic_delta(active_coords, center, lengths)
        distance2[:, j] = np.einsum("ij,ij->i", delta, delta)
    assigned[active] = np.argmin(distance2, axis=1).astype(np.int32) + 1
    assigned = assigned.reshape(phi.shape)

    dv = dx_nm**3
    face_area = dx_nm**2
    particles: list[ParticleRecord] = []
    for pid in range(1, particle_count + 1):
        core = core_labels == pid
        partition = assigned == pid
        weights = np.where(partition, h, 0.0)
        h_sum = float(np.sum(weights, dtype=np.float64))
        h_volume = h_sum * dv
        center = periodic_weighted_center(weights, dx_nm)
        delta = periodic_delta(coords, center, lengths)
        w = weights.ravel()
        tensor = np.einsum(
            "ni,n,nj->ij", delta, w, delta, optimize=True
        ) / max(h_sum, np.finfo(float).tiny)
        eigenvalues, eigenvectors = np.linalg.eigh(tensor)
        exposed_faces = 0
        for axis in range(3):
            exposed_faces += int(np.count_nonzero(core & ~np.roll(core, 1, axis=axis)))
            exposed_faces += int(np.count_nonzero(core & ~np.roll(core, -1, axis=axis)))
        shell = ndimage.binary_dilation(core, iterations=2) & ~core
        matrix_weight = np.where(shell, np.maximum(1.0 - h, 0.0), 0.0)
        matrix_den = float(np.sum(matrix_weight))
        xalpha = np.divide(qalpha, np.maximum(1.0 - h, np.finfo(float).tiny))
        local_x = (float(np.sum(matrix_weight * xalpha)) / matrix_den
                   if matrix_den > 0.0 else float("nan"))
        local_mu = None
        if chemical_potential is not None and matrix_den > 0.0:
            local_mu = float(np.sum(matrix_weight * chemical_potential) / matrix_den)
        particles.append(ParticleRecord(
            particle_id=pid,
            center_nm=center.tolist(),
            h_volume_nm3=h_volume,
            beta_inventory=h_sum * v_B,
            equivalent_radius_nm=(3.0 * h_volume / (4.0 * math.pi)) ** (1.0 / 3.0),
            threshold_volume_nm3=float(np.count_nonzero(core)) * dv,
            surface_area_nm2=exposed_faces * face_area,
            shape_tensor_nm2=tensor.tolist(),
            principal_variances_nm2=eigenvalues.tolist(),
            orientation_columns=eigenvectors.tolist(),
            local_matrix_xB=local_x,
            local_chemical_potential=local_mu,
            elastic_environment_tag="not_supplied",
            nearby_GP_inventory=0.0,
            nearest_neighbor_distance_nm=None,
        ))

    for i, particle in enumerate(particles):
        if len(particles) > 1:
            distances = []
            for j, other in enumerate(particles):
                if i == j:
                    continue
                delta = periodic_delta(np.asarray(other.center_nm)[None, :],
                                       np.asarray(particle.center_nm), lengths)[0]
                distances.append(float(np.linalg.norm(delta)))
            particle.nearest_neighbor_distance_nm = min(distances)

    matrix_inventory = float(np.sum(qalpha, dtype=np.float64))
    beta_inventory = float(np.sum(h, dtype=np.float64) * v_B)
    field_inventory = float(np.sum(ctot, dtype=np.float64))
    partitioned_beta = sum(p.beta_inventory for p in particles)
    total_with_gp = field_inventory + gp_inventory
    roundtrip = matrix_inventory + partitioned_beta + gp_inventory
    thresholds = {}
    for value in (0.4, 0.5, 0.6):
        labels, count = periodic_components(phi >= value)
        radii = []
        for pid in range(1, count + 1):
            volume = float(np.count_nonzero(labels == pid)) * dv
            radii.append((3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0))
        thresholds[f"phi_ge_{value:.1f}"] = {"particle_count": count, "radii_nm": radii}

    return {
        "schema": "PF_PARTICLE_HANDOFF_LEDGER_V1",
        "authoritative_pf_state": "Ctot",
        "diffuse_interface_policy": "partition_integral_h_phi_vB_once",
        "grid_shape": list(ctot.shape),
        "dx_nm": dx_nm,
        "cell_volume_nm3": dv,
        "v_B": v_B,
        "particle_threshold": threshold,
        "particle_count": particle_count,
        "particles": [asdict(p) for p in particles],
        "ledger": {
            "matrix_inventory_qalpha": matrix_inventory,
            "beta_inventory_hvB": beta_inventory,
            "partitioned_particle_beta_inventory": partitioned_beta,
            "GP_inventory": gp_inventory,
            "PF_field_inventory_Ctot": field_inventory,
            "total_inventory_with_GP": total_with_gp,
            "roundtrip_inventory": roundtrip,
            "roundtrip_error_abs": roundtrip - total_with_gp,
            "roundtrip_error_rel": abs(roundtrip - total_with_gp) /
                max(abs(total_with_gp), np.finfo(float).tiny),
            "particle_partition_error_rel": abs(partitioned_beta - beta_inventory) /
                max(abs(beta_inventory), np.finfo(float).tiny),
        },
        "threshold_stability": thresholds,
        "physics_claim": "mass-conserving state mapping only; no calibrated KWN claim",
    }


def read_raw(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float64)
    if data.size != math.prod(shape):
        raise ValueError(f"{path}: expected {math.prod(shape)} doubles, got {data.size}")
    return data.reshape(shape)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ctot", type=Path, required=True)
    parser.add_argument("--phi", type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs=3, required=True)
    parser.add_argument("--dx-nm", type=float, required=True)
    parser.add_argument("--v-B", type=float, default=1.0)
    parser.add_argument("--gp-inventory", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shape = tuple(args.shape)
    result = extract_handoff(
        read_raw(args.Ctot, shape), read_raw(args.phi, shape),
        dx_nm=args.dx_nm, v_B=args.v_B,
        gp_inventory=args.gp_inventory, threshold=args.threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["ledger"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
