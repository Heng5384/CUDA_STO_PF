#!/usr/bin/env python3
"""Periodic six-neighbour objects and per-object level-set geometry.

This module is read-only.  The resolved-object contract is h(phi)>1e-4 with
six-neighbour connectivity.  Interface geometry is measured independently at
phi=0.45, 0.50, and 0.55.  Periodic objects are unwrapped into object-local
boxes before marching cubes, so a seam-crossing object is meshed once.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import ndimage


SCHEMA = "PF400_PERIODIC_TRUE_AREA_GEOMETRY_V1"
OBJECT_H_THRESHOLD = 1.0e-4
PRIMARY_LEVEL = 0.50
AREA_LEVELS = (0.45, 0.50, 0.55)
V4_HEADER_BYTES = 904


def h_of_phi(phi: np.ndarray) -> np.ndarray:
    value = np.asarray(phi, dtype=np.float64)
    return value**3 * (6.0 * value**2 - 15.0 * value + 10.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class DisjointSet:
    def __init__(self, count: int) -> None:
        self.parent = np.arange(count + 1, dtype=np.int32)

    def find(self, value: int) -> int:
        parent = self.parent
        root = value
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[value]) != value:
            nxt = int(parent[value])
            parent[value] = root
            value = nxt
        return root

    def union(self, left: int, right: int) -> None:
        if left == 0 or right == 0:
            return
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def periodic_labels_six(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label a periodic Boolean volume with exactly six-neighbour connectivity."""
    if mask.ndim != 3:
        raise ValueError("periodic_labels_six requires a three-dimensional mask")
    structure = ndimage.generate_binary_structure(3, 1)
    labels, count = ndimage.label(mask, structure=structure)
    dsu = DisjointSet(int(count))
    for axis in range(3):
        first = np.take(labels, 0, axis=axis)
        last = np.take(labels, -1, axis=axis)
        select = (first > 0) & (last > 0)
        if np.any(select):
            pairs = np.column_stack((first[select], last[select]))
            for left, right in np.unique(pairs, axis=0):
                dsu.union(int(left), int(right))
    root_for_label = np.arange(count + 1, dtype=np.int32)
    for label in range(1, count + 1):
        root_for_label[label] = dsu.find(label)
    roots = np.unique(root_for_label[1:])
    compact_root = np.zeros(count + 1, dtype=np.int32)
    compact_root[roots] = np.arange(1, roots.size + 1, dtype=np.int32)
    compact_label = compact_root[root_for_label]
    return compact_label[labels], int(roots.size)


def circular_centres(
    labels: np.ndarray, weights: np.ndarray, count: int
) -> np.ndarray:
    centres = np.empty((count + 1, 3), dtype=np.float64)
    centres[0] = np.nan
    flat_labels = labels.ravel()
    flat_weights = weights.ravel()
    for axis, period in enumerate(labels.shape):
        coordinates = np.indices(labels.shape, sparse=True)[axis]
        angles = 2.0 * math.pi * coordinates / period
        sin_sum = np.bincount(
            flat_labels,
            weights=(weights * np.sin(angles)).ravel(),
            minlength=count + 1,
        )
        cos_sum = np.bincount(
            flat_labels,
            weights=(weights * np.cos(angles)).ravel(),
            minlength=count + 1,
        )
        centres[:, axis] = (
            np.mod(np.arctan2(sin_sum, cos_sum), 2.0 * math.pi)
            * period
            / (2.0 * math.pi)
        )
    centres[0] = np.nan
    return centres


def periodic_extents(labels: np.ndarray, centres: np.ndarray, count: int) -> np.ndarray:
    extents = np.zeros((count + 1, 3), dtype=np.float64)
    for axis, period in enumerate(labels.shape):
        for coordinate in range(period):
            present = np.unique(np.take(labels, coordinate, axis=axis))
            present = present[present > 0]
            if present.size == 0:
                continue
            delta = np.abs(
                (coordinate - centres[present, axis] + period / 2.0) % period
                - period / 2.0
            )
            extents[present, axis] = np.maximum(extents[present, axis], delta)
    return extents


def _local_object(
    phi: np.ndarray,
    labels: np.ndarray,
    object_id: int,
    centre: np.ndarray,
    extent: np.ndarray,
    padding: int = 3,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    indices: list[np.ndarray] = []
    unwrapped: list[np.ndarray] = []
    for axis, period in enumerate(phi.shape):
        half_width = int(math.ceil(extent[axis])) + padding
        offsets = np.arange(-half_width, half_width + 1, dtype=np.int64)
        origin = int(round(float(centre[axis])))
        indices.append((origin + offsets) % period)
        unwrapped.append((origin + offsets).astype(np.float64))
    selector = np.ix_(*indices)
    local_labels = np.asarray(labels[selector])
    support = local_labels == object_id
    local_phi = np.where(support, np.asarray(phi[selector], dtype=np.float32), 0.0)
    return local_phi, support, (unwrapped[0], unwrapped[1], unwrapped[2])


def level_set_mesh_metrics(
    scalar: np.ndarray, level: float, spacing: float = 1.0
) -> dict[str, float | int | bool]:
    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    except ImportError as exc:
        raise RuntimeError("VTK Python bindings are required") from exc
    if not float(np.min(scalar)) <= level <= float(np.max(scalar)):
        raise ValueError(f"level {level} is outside local scalar range")
    image = vtk.vtkImageData()
    image.SetDimensions(*scalar.shape)
    image.SetSpacing(spacing, spacing, spacing)
    values = numpy_to_vtk(
        np.asarray(scalar, dtype=np.float32).ravel(order="F"), deep=True
    )
    values.SetName("phi")
    image.GetPointData().SetScalars(values)
    contour = vtk.vtkMarchingCubes()
    contour.SetInputData(image)
    contour.SetValue(0, float(level))
    contour.ComputeNormalsOff()
    contour.ComputeGradientsOff()
    contour.Update()
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(contour.GetOutputPort())
    cleaner.PointMergingOn()
    cleaner.Update()
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(cleaner.GetOutputPort())
    triangles.Update()
    mesh = triangles.GetOutput()
    if mesh.GetNumberOfCells() == 0:
        raise ValueError("empty level-set mesh")
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(mesh)
    edges.BoundaryEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    # vtkTriangleFilter can retain vertex/line cells generated by degenerate
    # fragments.  MassProperties must receive a polys-only data object.
    surface = vtk.vtkPolyData()
    surface.SetPoints(mesh.GetPoints())
    surface.SetPolys(mesh.GetPolys())
    mass = vtk.vtkMassProperties()
    mass.SetInputData(surface)
    mass.Update()
    points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float64, copy=False)
    polys_raw = vtk_to_numpy(mesh.GetPolys().GetData())
    cells = polys_raw.reshape(-1, 4)[:, 1:]
    p0, p1, p2 = points[cells[:, 0]], points[cells[:, 1]], points[cells[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    twice_area = np.linalg.norm(cross, axis=1)
    area_weights = 0.5 * twice_area
    normals = np.divide(
        cross,
        twice_area[:, None],
        out=np.zeros_like(cross),
        where=twice_area[:, None] > 0.0,
    )
    normal_tensor = np.einsum("i,ij,ik->jk", area_weights, normals, normals)
    normal_tensor /= float(np.sum(area_weights))
    normal_eigen = np.linalg.eigvalsh(normal_tensor)[::-1]
    return {
        "area_nm2": float(np.sum(area_weights)),
        "mesh_volume_nm3": float(mass.GetVolume()),
        "mesh_points": int(mesh.GetNumberOfPoints()),
        "mesh_triangles": int(mesh.GetNumberOfCells()),
        "boundary_or_nonmanifold_edges": int(edges.GetOutput().GetNumberOfCells()),
        "watertight": edges.GetOutput().GetNumberOfCells() == 0,
        "normal_tensor_eig1": float(normal_eigen[0]),
        "normal_tensor_eig2": float(normal_eigen[1]),
        "normal_tensor_eig3": float(normal_eigen[2]),
    }


def _weighted_shape(
    local_phi: np.ndarray,
    support: np.ndarray,
    coordinate_axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    centre: np.ndarray,
    dx_nm: float,
) -> dict[str, Any]:
    h = np.where(support, h_of_phi(local_phi), 0.0)
    total = float(np.sum(h))
    coords = np.meshgrid(*coordinate_axes, indexing="ij", sparse=True)
    deltas = [
        (np.asarray(coords[axis], dtype=np.float64) - centre[axis]) * dx_nm
        for axis in range(3)
    ]
    covariance = np.empty((3, 3), dtype=np.float64)
    for left in range(3):
        for right in range(left, 3):
            value = float(np.sum(h * deltas[left] * deltas[right]) / total)
            covariance[left, right] = covariance[right, left] = value
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    # Uniform-ellipsoid semi-axes satisfy <x_i^2>=a_i^2/5.
    semi_axes = np.sqrt(5.0 * eigenvalues)
    inertia = total * (np.trace(covariance) * np.eye(3) - covariance)
    return {
        "principal_axis_1_nm": float(semi_axes[0]),
        "principal_axis_2_nm": float(semi_axes[1]),
        "principal_axis_3_nm": float(semi_axes[2]),
        "aspect_ratio_1_3": float(semi_axes[0] / max(semi_axes[2], 1.0e-30)),
        "aspect_ratio_1_2": float(semi_axes[0] / max(semi_axes[1], 1.0e-30)),
        "shape_tensor_xx_nm2": float(covariance[0, 0]),
        "shape_tensor_xy_nm2": float(covariance[0, 1]),
        "shape_tensor_xz_nm2": float(covariance[0, 2]),
        "shape_tensor_yy_nm2": float(covariance[1, 1]),
        "shape_tensor_yz_nm2": float(covariance[1, 2]),
        "shape_tensor_zz_nm2": float(covariance[2, 2]),
        "inertia_xx_nm5": float(inertia[0, 0] * dx_nm**3),
        "inertia_xy_nm5": float(inertia[0, 1] * dx_nm**3),
        "inertia_xz_nm5": float(inertia[0, 2] * dx_nm**3),
        "inertia_yy_nm5": float(inertia[1, 1] * dx_nm**3),
        "inertia_yz_nm5": float(inertia[1, 2] * dx_nm**3),
        "inertia_zz_nm5": float(inertia[2, 2] * dx_nm**3),
        "principal_vector_1_x": float(eigenvectors[0, 0]),
        "principal_vector_1_y": float(eigenvectors[1, 0]),
        "principal_vector_1_z": float(eigenvectors[2, 0]),
    }


def extract_geometry(
    phi: np.ndarray,
    x_b: np.ndarray | None = None,
    *,
    case_id: str = "synthetic",
    time_h: float = 0.0,
    step: int = 0,
    dx_nm: float = 1.0,
    levels: Sequence[float] = AREA_LEVELS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if phi.ndim != 3:
        raise ValueError("phi must be three-dimensional")
    if x_b is not None and x_b.shape != phi.shape:
        raise ValueError("xB shape differs from phi")
    h = h_of_phi(phi)
    labels, count = periodic_labels_six(h > OBJECT_H_THRESHOLD)
    if count == 0:
        raise ValueError("no resolved beta objects")
    flat_labels = labels.ravel()
    voxel_counts = np.bincount(flat_labels, minlength=count + 1)
    h_volumes = np.bincount(flat_labels, weights=h.ravel(), minlength=count + 1)
    centres_grid = circular_centres(labels, h, count)
    extents = periodic_extents(labels, centres_grid, count)
    rows: list[dict[str, Any]] = []
    box_volume_nm3 = float(math.prod(phi.shape) * dx_nm**3)
    for object_id in range(1, count + 1):
        local_phi, support, coordinate_axes = _local_object(
            phi, labels, object_id, centres_grid[object_id], extents[object_id]
        )
        volume_nm3 = float(h_volumes[object_id] * dx_nm**3)
        radius_nm = (3.0 * volume_nm3 / (4.0 * math.pi)) ** (1.0 / 3.0)
        sphere_area_nm2 = 4.0 * math.pi * radius_nm**2
        row: dict[str, Any] = {
            "schema": SCHEMA,
            "case_id": case_id,
            "time_h": time_h,
            "step": step,
            "object_id": object_id,
            "periodic_object_id": f"{case_id}_{time_h:g}h_O{object_id:04d}",
            "lineage_id": "UNAVAILABLE_SNAPSHOT_ONLY",
            "lineage_trusted": False,
            "threshold_voxel_count": int(voxel_counts[object_id]),
            "h_volume_nm3": volume_nm3,
            "voxel_volume_nm3": float(voxel_counts[object_id] * dx_nm**3),
            "equivalent_radius_nm": radius_nm,
            "center_x_nm": float(centres_grid[object_id, 0] * dx_nm),
            "center_y_nm": float(centres_grid[object_id, 1] * dx_nm),
            "center_z_nm": float(centres_grid[object_id, 2] * dx_nm),
            "sphere_area_nm2": sphere_area_nm2,
            "periodic_wrap_x": bool(extents[object_id, 0] > min(centres_grid[object_id, 0], phi.shape[0] - centres_grid[object_id, 0] - 1)),
            "periodic_wrap_y": bool(extents[object_id, 1] > min(centres_grid[object_id, 1], phi.shape[1] - centres_grid[object_id, 1] - 1)),
            "periodic_wrap_z": bool(extents[object_id, 2] > min(centres_grid[object_id, 2], phi.shape[2] - centres_grid[object_id, 2] - 1)),
            "local_elastic_descriptor": "",
            "local_elastic_descriptor_status": "UNAVAILABLE_ACCEPTED_TIME_FIELD_NOT_RETAINED",
        }
        row.update(_weighted_shape(local_phi, support, coordinate_axes, centres_grid[object_id], dx_nm))
        core_labels, core_count = ndimage.label(
            local_phi >= PRIMARY_LEVEL,
            structure=ndimage.generate_binary_structure(3, 1),
        )
        del core_labels
        row["core_component_count"] = int(core_count)
        row["merged_or_necked"] = bool(core_count > 1)
        row["topology_warning"] = "MULTIPLE_CORE_LOBES_IN_ONE_LOW_SUPPORT_OBJECT" if core_count > 1 else ""
        if x_b is not None:
            indices = tuple((np.asarray(axis, dtype=np.int64) % phi.shape[i]) for i, axis in enumerate(coordinate_axes))
            local_xb = np.asarray(x_b[np.ix_(*indices)], dtype=np.float64)
            local_h = np.where(support, h_of_phi(local_phi), 0.0)
            row["local_object_xB_h_weighted"] = float(np.sum(local_h * local_xb) / np.sum(local_h))
            row["local_object_xAg_h_weighted"] = float(0.9977236 * row["local_object_xB_h_weighted"])
        for level in levels:
            level_present = float(np.min(local_phi)) <= float(level) <= float(np.max(local_phi))
            if level_present:
                metrics = level_set_mesh_metrics(local_phi, float(level), dx_nm)
            else:
                metrics = {
                    "area_nm2": 0.0,
                    "mesh_volume_nm3": 0.0,
                    "mesh_points": 0,
                    "mesh_triangles": 0,
                    "boundary_or_nonmanifold_edges": 0,
                    "watertight": True,
                    "normal_tensor_eig1": "",
                    "normal_tensor_eig2": "",
                    "normal_tensor_eig3": "",
                }
            tag = f"phi{int(round(100 * level)):02d}"
            row[f"levelset_present_{tag}"] = level_present
            for key, value in metrics.items():
                row[f"{key}_{tag}"] = value
        area = float(row["area_nm2_phi50"])
        mesh_volume = float(row["mesh_volume_nm3_phi50"])
        row["true_area_nm2"] = area
        row["true_to_sphere_area_ratio"] = area / sphere_area_nm2
        row["sphericity"] = sphere_area_nm2 / area if area > 0.0 else ""
        row["compactness"] = 36.0 * math.pi * mesh_volume**2 / area**3 if area > 0.0 else ""
        row["mesh_to_h_volume_relative_difference"] = abs(mesh_volume - volume_nm3) / volume_nm3
        meshes_watertight = all(bool(row[f"watertight_phi{int(round(100 * level)):02d}"]) for level in levels)
        if not meshes_watertight:
            row["geometry_status"] = "FAIL_NON_WATERTIGHT"
        elif not bool(row["levelset_present_phi50"]):
            row["geometry_status"] = "PASS_LOW_AMPLITUDE_NO_CANONICAL_LEVELSET"
            row["topology_warning"] = "NO_CANONICAL_PHI_0P5_ISOSURFACE_LOW_AMPLITUDE_OBJECT"
        else:
            row["geometry_status"] = "PASS"
        rows.append(row)
    rows.sort(key=lambda item: int(item["object_id"]))
    aggregate: dict[str, Any] = {
        "schema": SCHEMA,
        "case_id": case_id,
        "time_h": time_h,
        "step": step,
        "grid": list(phi.shape),
        "dx_nm": dx_nm,
        "box_volume_nm3": box_volume_nm3,
        "object_h_threshold": OBJECT_H_THRESHOLD,
        "canonical_area_level": PRIMARY_LEVEL,
        "area_levels": list(levels),
        "particle_count": count,
        "beta_h_volume_fraction": float(np.sum(h) * dx_nm**3 / box_volume_nm3),
        "Sv_sph_nm-1": float(sum(float(row["sphere_area_nm2"]) for row in rows) / box_volume_nm3),
        "Sv_true_nm-1": float(sum(float(row["true_area_nm2"]) for row in rows) / box_volume_nm3),
        "M6_nm3": float(sum(float(row["equivalent_radius_nm"]) ** 6 for row in rows) / box_volume_nm3),
        "all_meshes_watertight": all(str(row["geometry_status"]).startswith("PASS") for row in rows),
        "no_canonical_levelset_object_count": sum(not bool(row["levelset_present_phi50"]) for row in rows),
        "wrapped_object_count": sum(bool(row["periodic_wrap_x"] or row["periodic_wrap_y"] or row["periodic_wrap_z"]) for row in rows),
        "necked_object_count": sum(bool(row["merged_or_necked"]) for row in rows),
    }
    for level in levels:
        tag = f"phi{int(round(100 * level)):02d}"
        aggregate[f"Sv_true_{tag}_nm-1"] = float(sum(float(row[f"area_nm2_{tag}"]) for row in rows) / box_volume_nm3)
    return rows, aggregate


@dataclass(frozen=True)
class V4Header:
    path: str
    magic: str
    version: int
    header_bytes: int
    count: int
    step: int
    shape: tuple[int, int, int]
    dt_code: float
    temperature_K: float
    target_mass_code: float


def read_v4_header(path: Path) -> V4Header:
    with path.open("rb") as handle:
        raw = handle.read(V4_HEADER_BYTES)
    if len(raw) != V4_HEADER_BYTES:
        raise ValueError(f"short V4 header: {path}")
    magic = raw[:8].decode("ascii", errors="replace")
    version = struct.unpack_from("<I", raw, 8)[0]
    header_bytes = struct.unpack_from("<I", raw, 12)[0]
    count = struct.unpack_from("<Q", raw, 16)[0]
    step = struct.unpack_from("<Q", raw, 32)[0]
    shape = struct.unpack_from("<iii", raw, 40)
    dt_code = struct.unpack_from("<d", raw, 56)[0]
    temperature = struct.unpack_from("<d", raw, 64)[0]
    target = struct.unpack_from("<d", raw, 72)[0]
    if magic != "PFZMCHK4" or version != 4 or header_bytes != V4_HEADER_BYTES:
        raise ValueError(f"unsupported checkpoint schema: {path}")
    if count != math.prod(shape):
        raise ValueError(f"checkpoint count/grid mismatch: {path}")
    return V4Header(str(path), magic, version, header_bytes, count, step, shape, dt_code, temperature, target)


def checkpoint_fields(path: Path) -> tuple[V4Header, np.memmap, np.memmap]:
    header = read_v4_header(path)
    phi = np.memmap(path, dtype="<f8", mode="r", offset=header.header_bytes, shape=header.shape, order="C")
    xb_offset = header.header_bytes + 2 * header.count * 8
    x_b = np.memmap(path, dtype="<f8", mode="r", offset=xb_offset, shape=header.shape, order="C")
    return header, phi, x_b


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--fixture", type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--time-h", required=True, type=float)
    parser.add_argument("--step", type=int)
    parser.add_argument("--grid", type=int, default=400)
    parser.add_argument("--dx-nm", type=float, default=1.0)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)
    inputs: dict[str, Any] = {}
    if args.checkpoint:
        header, phi, x_b = checkpoint_fields(args.checkpoint)
        step = header.step
        inputs = {
            "source_type": "PFZMCHK4",
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256(args.checkpoint),
            "header": header.__dict__,
        }
    else:
        fixture = args.fixture.resolve()
        shape = (args.grid, args.grid, args.grid)
        phi_path = fixture / "phi.raw.f64"
        xb_path = fixture / "C_B_tot.raw.f64"
        phi = np.memmap(phi_path, dtype="<f8", mode="r", shape=shape, order="C")
        x_b = np.memmap(xb_path, dtype="<f8", mode="r", shape=shape, order="C")
        step = int(args.step or 0)
        inputs = {
            "source_type": "INITIAL_FIXTURE_RAW_FIELDS",
            "fixture": str(fixture),
            "phi_path": str(phi_path),
            "phi_sha256": sha256(phi_path),
            "xB_path": str(xb_path),
            "xB_sha256": sha256(xb_path),
            "fixture_manifest_sha256": sha256(fixture / "fixture_manifest.json"),
        }
    rows, aggregate = extract_geometry(
        phi,
        x_b,
        case_id=args.case_id,
        time_h=args.time_h,
        step=step,
        dx_nm=args.dx_nm,
    )
    write_csv(args.out / "particle_geometry.csv", rows)
    (args.out / "snapshot_geometry.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "schema": SCHEMA,
        "status": "PASS" if aggregate["all_meshes_watertight"] else "FAIL_NON_WATERTIGHT",
        "inputs": inputs,
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "contracts": {
            "resolved_object": "periodic six-neighbour on h(phi)>1e-4",
            "area": "object-local periodic unwrap; VTK marching cubes",
            "levels": list(AREA_LEVELS),
            "physical_particle_identity_claim": False,
        },
    }
    (args.out / "geometry_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.out / "status.txt").write_text(str(provenance["status"]) + "\n", encoding="utf-8")
    print(provenance["status"])
    return 0 if provenance["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
