#!/usr/bin/env python3
"""Periodic interface, strain, stress, and spectrum audit for replay fields."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage


AGES = (6, 12, 18, 24, 36, 48)
REPLICATES = ("A", "B", "C")
COMPONENTS = ("xx", "yy", "zz", "xy", "xz", "yz")
DX_NM = 1.0
LAMBDA_NM = 4.0
PHI_LEVEL = 0.5
NORMAL_AZIMUTH_BINS = 36
NORMAL_COS_POLAR_BINS = 18
CURVATURE_EDGES_NM_INV = np.linspace(-2.0, 2.0, 201)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: Iterable[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    fieldnames = list(fields) if fields is not None else list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_params(path: Path) -> dict[str, float | str]:
    values: dict[str, float | str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        try:
            values[key] = float(value)
        except ValueError:
            values[key] = value
    return values


def load_field(root: Path, name: str, dtype: str, shape: tuple[int, int, int]) -> np.ndarray:
    path = root / name
    field = np.fromfile(path, dtype=dtype)
    if field.size != math.prod(shape):
        raise ValueError(f"{path}: expected {math.prod(shape)} values, found {field.size}")
    return field.reshape(shape)


def periodic_marching_cubes_area_nm2(phi: np.ndarray) -> tuple[float, int, int]:
    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    except ImportError as exc:
        raise RuntimeError("VTK Python bindings are required for true marching cubes") from exc
    periodic = np.empty(tuple(size + 1 for size in phi.shape), dtype=np.float32)
    periodic[:-1, :-1, :-1] = phi.astype(np.float32, copy=False)
    periodic[-1, :-1, :-1] = periodic[0, :-1, :-1]
    periodic[:, -1, :-1] = periodic[:, 0, :-1]
    periodic[:, :, -1] = periodic[:, :, 0]
    image = vtk.vtkImageData()
    image.SetDimensions(*periodic.shape)
    image.SetSpacing(DX_NM, DX_NM, DX_NM)
    scalars = numpy_to_vtk(periodic.ravel(order="F"), deep=True)
    scalars.SetName("phi")
    image.GetPointData().SetScalars(scalars)
    contour = vtk.vtkMarchingCubes()
    contour.SetInputData(image)
    contour.SetValue(0, PHI_LEVEL)
    contour.ComputeNormalsOff()
    contour.ComputeGradientsOff()
    contour.Update()
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(contour.GetOutputPort())
    triangles.Update()
    mesh = triangles.GetOutput()
    points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float64, copy=False)
    cells = vtk_to_numpy(mesh.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    p0, p1, p2 = points[cells[:, 0]], points[cells[:, 1]], points[cells[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1).sum()
    return float(area), int(points.shape[0]), int(cells.shape[0])


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size + 1, dtype=np.int32)

    def find(self, value: int) -> int:
        parent = self.parent
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(self, left: int, right: int) -> None:
        if left == 0 or right == 0:
            return
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def periodic_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    dsu = DisjointSet(count)
    for axis in range(3):
        first = np.take(labels, 0, axis=axis)
        last = np.take(labels, -1, axis=axis)
        for shift0 in (-1, 0, 1):
            for shift1 in (-1, 0, 1):
                shifted = np.roll(last, (shift0, shift1), axis=(0, 1))
                select = (first > 0) & (shifted > 0)
                for left, right in zip(first[select], shifted[select]):
                    dsu.union(int(left), int(right))
    lookup = np.arange(count + 1, dtype=np.int32)
    for label in range(1, count + 1):
        lookup[label] = dsu.find(label)
    merged = lookup[labels]
    roots = np.unique(merged[merged > 0])
    compact = np.zeros(count + 1, dtype=np.int32)
    compact[roots] = np.arange(1, roots.size + 1, dtype=np.int32)
    return compact[merged], int(roots.size)


def circular_unwrap(values: np.ndarray, period: int) -> tuple[np.ndarray, float]:
    angle = 2.0 * np.pi * values / period
    center_angle = math.atan2(float(np.sin(angle).mean()), float(np.cos(angle).mean()))
    center = (center_angle % (2.0 * np.pi)) * period / (2.0 * np.pi)
    delta = (values - center + period / 2.0) % period - period / 2.0
    return delta * DX_NM, center * DX_NM


def particle_shapes(phi: np.ndarray, replicate: str, age_h: int) -> list[dict]:
    labels, count = periodic_labels(phi >= PHI_LEVEL)
    coords = np.argwhere(labels > 0)
    ids = labels[labels > 0]
    order = np.argsort(ids, kind="stable")
    coords, ids = coords[order], ids[order]
    boundaries = np.flatnonzero(np.diff(ids)) + 1
    groups = np.split(coords, boundaries)
    rows: list[dict] = []
    for particle_id, group in enumerate(groups, start=1):
        unwrapped = []
        centers = []
        for axis, period in enumerate(phi.shape):
            delta, center = circular_unwrap(group[:, axis].astype(np.float64), period)
            unwrapped.append(delta)
            centers.append(center)
        xyz = np.column_stack(unwrapped)
        covariance = xyz.T @ xyz / max(xyz.shape[0], 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        order_eigen = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order_eigen]
        eigenvectors = eigenvectors[:, order_eigen]
        axes = np.sqrt(eigenvalues)
        principal = eigenvectors[:, 0]
        if principal[np.argmax(np.abs(principal))] < 0:
            principal = -principal
        volume_nm3 = float(group.shape[0]) * DX_NM**3
        rows.append(
            {
                "replicate": replicate,
                "age_h": age_h,
                "particle_id": particle_id,
                "voxel_count": int(group.shape[0]),
                "threshold_volume_nm3": volume_nm3,
                "equivalent_radius_nm": (3.0 * volume_nm3 / (4.0 * np.pi)) ** (1.0 / 3.0),
                "axis_rms_1_nm": axes[0],
                "axis_rms_2_nm": axes[1],
                "axis_rms_3_nm": axes[2],
                "aspect_1_over_2": axes[0] / max(axes[1], np.finfo(float).tiny),
                "aspect_1_over_3": axes[0] / max(axes[2], np.finfo(float).tiny),
                "principal_x": principal[0],
                "principal_y": principal[1],
                "principal_z": principal[2],
                "periodic_center_x_nm": centers[0],
                "periodic_center_y_nm": centers[1],
                "periodic_center_z_nm": centers[2],
                "periodic_label_count": count,
            }
        )
    return rows


def interface_geometry(phi: np.ndarray, replicate: str, age_h: int) -> tuple[dict, list[dict], list[dict], np.ndarray]:
    gx = (np.roll(phi, -1, 0) - np.roll(phi, 1, 0)) / (2.0 * DX_NM)
    gy = (np.roll(phi, -1, 1) - np.roll(phi, 1, 1)) / (2.0 * DX_NM)
    gz = (np.roll(phi, -1, 2) - np.roll(phi, 1, 2)) / (2.0 * DX_NM)
    gradient = np.sqrt(gx * gx + gy * gy + gz * gz)
    shell = np.abs(phi - PHI_LEVEL) <= 0.05
    valid = shell & (gradient > 1.0e-12)
    nx, ny, nz = gx[valid] / gradient[valid], gy[valid] / gradient[valid], gz[valid] / gradient[valid]
    weights = gradient[valid]
    azimuth = np.arctan2(ny, nx)
    cos_polar = np.clip(nz, -1.0, 1.0)
    histogram, az_edges, mu_edges = np.histogram2d(
        azimuth,
        cos_polar,
        bins=(NORMAL_AZIMUTH_BINS, NORMAL_COS_POLAR_BINS),
        range=((-np.pi, np.pi), (-1.0, 1.0)),
        weights=weights,
    )
    probability = histogram / max(float(histogram.sum()), np.finfo(float).tiny)
    normal_rows: list[dict] = []
    for ia in range(NORMAL_AZIMUTH_BINS):
        for im in range(NORMAL_COS_POLAR_BINS):
            normal_rows.append(
                {
                    "replicate": replicate,
                    "age_h": age_h,
                    "azimuth_low_rad": az_edges[ia],
                    "azimuth_high_rad": az_edges[ia + 1],
                    "cos_polar_low": mu_edges[im],
                    "cos_polar_high": mu_edges[im + 1],
                    "area_weighted_probability": probability[ia, im],
                }
            )
    normal_tensor = np.array(
        [
            [np.average(nx * nx, weights=weights), np.average(nx * ny, weights=weights), np.average(nx * nz, weights=weights)],
            [np.average(nx * ny, weights=weights), np.average(ny * ny, weights=weights), np.average(ny * nz, weights=weights)],
            [np.average(nx * nz, weights=weights), np.average(ny * nz, weights=weights), np.average(nz * nz, weights=weights)],
        ]
    )
    normal_eigenvalues = np.linalg.eigvalsh(normal_tensor)[::-1]
    safe_gradient = np.maximum(gradient, 1.0e-12)
    nx_all, ny_all, nz_all = gx / safe_gradient, gy / safe_gradient, gz / safe_gradient
    divergence = (
        np.roll(nx_all, -1, 0) - np.roll(nx_all, 1, 0)
        + np.roll(ny_all, -1, 1) - np.roll(ny_all, 1, 1)
        + np.roll(nz_all, -1, 2) - np.roll(nz_all, 1, 2)
    ) / (2.0 * DX_NM)
    mean_curvature = 0.5 * divergence[valid]
    curvature_hist, curvature_edges = np.histogram(
        mean_curvature, bins=CURVATURE_EDGES_NM_INV, weights=weights
    )
    curvature_probability = curvature_hist / max(float(curvature_hist.sum()), np.finfo(float).tiny)
    curvature_rows = [
        {
            "replicate": replicate,
            "age_h": age_h,
            "curvature_low_nm_inv": curvature_edges[index],
            "curvature_high_nm_inv": curvature_edges[index + 1],
            "area_weighted_probability": curvature_probability[index],
        }
        for index in range(curvature_hist.size)
    ]
    area_nm2, vertex_count, triangle_count = periodic_marching_cubes_area_nm2(phi)
    box_volume_nm3 = math.prod(phi.shape) * DX_NM**3
    delta_halfwidth = 0.05
    coarea_nm2 = float(np.sum(gradient[shell]) * DX_NM**3 / (2.0 * delta_halfwidth))
    pad = int(math.ceil(4.0 * LAMBDA_NM / DX_NM)) + 1
    beta = phi >= PHI_LEVEL
    beta_padded = np.pad(beta, pad, mode="wrap")
    distance_in = ndimage.distance_transform_edt(beta_padded, sampling=DX_NM)
    distance_out = ndimage.distance_transform_edt(~beta_padded, sampling=DX_NM)
    crop = tuple(slice(pad, -pad) for _ in range(3))
    signed_distance_nm = (distance_out - distance_in)[crop]
    abs_distance = np.abs(signed_distance_nm)
    geometry = {
        "replicate": replicate,
        "age_h": age_h,
        "marching_cubes_interface_area_nm2": area_nm2,
        "Sv_m_inv": area_nm2 * 1.0e-18 / (box_volume_nm3 * 1.0e-27),
        "coarea_interface_area_nm2": coarea_nm2,
        "coarea_to_marching_cubes_ratio": coarea_nm2 / area_nm2,
        "marching_cubes_vertex_count": vertex_count,
        "marching_cubes_triangle_count": triangle_count,
        "normal_tensor_eigenvalue_1": normal_eigenvalues[0],
        "normal_tensor_eigenvalue_2": normal_eigenvalues[1],
        "normal_tensor_eigenvalue_3": normal_eigenvalues[2],
        "normal_anisotropy_l1_minus_l3": normal_eigenvalues[0] - normal_eigenvalues[2],
        "curvature_mean_nm_inv": np.average(mean_curvature, weights=weights),
        "curvature_abs_mean_nm_inv": np.average(np.abs(mean_curvature), weights=weights),
        "curvature_rms_nm_inv": math.sqrt(np.average(mean_curvature**2, weights=weights)),
        "interface_shell_0_lambda_fraction": float(np.mean(abs_distance < LAMBDA_NM)),
        "interface_shell_lambda_2lambda_fraction": float(np.mean((abs_distance >= LAMBDA_NM) & (abs_distance < 2.0 * LAMBDA_NM))),
        "interface_shell_2lambda_4lambda_fraction": float(np.mean((abs_distance >= 2.0 * LAMBDA_NM) & (abs_distance < 4.0 * LAMBDA_NM))),
    }
    return geometry, normal_rows, curvature_rows, signed_distance_nm


def field_statistics(
    replicate: str,
    age_h: int,
    phi: np.ndarray,
    signed_distance_nm: np.ndarray,
    strain: dict[str, np.ndarray],
    stress_pa: dict[str, np.ndarray],
    energy_j_m3: np.ndarray,
) -> tuple[list[dict], list[dict], dict[str, np.ndarray]]:
    epsilon_h = (strain["xx"] + strain["yy"] + strain["zz"]) / 3.0
    dev_xx, dev_yy, dev_zz = strain["xx"] - epsilon_h, strain["yy"] - epsilon_h, strain["zz"] - epsilon_h
    dev_contract = (
        dev_xx**2 + dev_yy**2 + dev_zz**2
        + 2.0 * (strain["xy"]**2 + strain["xz"]**2 + strain["yz"]**2)
    )
    equivalent_dev = np.sqrt((2.0 / 3.0) * dev_contract)
    sigma_h = (stress_pa["xx"] + stress_pa["yy"] + stress_pa["zz"]) / 3.0
    distance = np.abs(signed_distance_nm)
    regions = {
        "whole_box": np.ones(phi.shape, dtype=bool),
        "matrix_core_phi_le_0p1": phi <= 0.1,
        "beta_core_phi_ge_0p9": phi >= 0.9,
        "interface_shell_0_lambda": distance < LAMBDA_NM,
        "interface_shell_lambda_2lambda": (distance >= LAMBDA_NM) & (distance < 2.0 * LAMBDA_NM),
        "interface_shell_2lambda_4lambda": (distance >= 2.0 * LAMBDA_NM) & (distance < 4.0 * LAMBDA_NM),
    }
    strain_rows: list[dict] = []
    stress_rows: list[dict] = []
    for region, select in regions.items():
        if not np.any(select):
            raise ValueError(f"empty region {replicate} {age_h} {region}")
        eh, ed, dc = epsilon_h[select], equivalent_dev[select], dev_contract[select]
        sh, en = sigma_h[select], energy_j_m3[select]
        strain_rows.append(
            {
                "replicate": replicate,
                "age_h": age_h,
                "region": region,
                "voxel_count": int(select.sum()),
                "epsilon_h_mean": float(eh.mean()),
                "epsilon_h_mean_square": float(np.mean(eh**2)),
                "epsilon_h_variance": float(np.var(eh)),
                "epsilon_dev_contract_mean": float(np.mean(dc)),
                "epsilon_dev_equivalent_mean": float(ed.mean()),
                "epsilon_dev_equivalent_variance": float(np.var(ed)),
            }
        )
        stress_rows.append(
            {
                "replicate": replicate,
                "age_h": age_h,
                "region": region,
                "voxel_count": int(select.sum()),
                "sigma_h_mean_Pa": float(sh.mean()),
                "sigma_h_mean_square_Pa2": float(np.mean(sh**2)),
                "sigma_h_variance_Pa2": float(np.var(sh)),
                "elastic_energy_mean_J_m3": float(en.mean()),
                "elastic_energy_variance_J2_m6": float(np.var(en)),
            }
        )
    spectral_fields = {
        "epsilon_h": epsilon_h,
        "epsilon_dev_equivalent": equivalent_dev,
        "sigma_h_Pa": sigma_h,
        "elastic_energy_fluctuation_J_m3": energy_j_m3,
    }
    return strain_rows, stress_rows, spectral_fields


def spectral_analysis(
    replicate: str, age_h: int, fields: dict[str, np.ndarray]
) -> tuple[list[dict], list[dict]]:
    shape = next(iter(fields.values())).shape
    nvox = math.prod(shape)
    qx = (2.0 * np.pi * np.fft.fftfreq(shape[0], d=DX_NM)).astype(np.float32)
    qy = (2.0 * np.pi * np.fft.fftfreq(shape[1], d=DX_NM)).astype(np.float32)
    qz = (2.0 * np.pi * np.fft.rfftfreq(shape[2], d=DX_NM)).astype(np.float32)
    qmag = np.sqrt(qx[:, None, None] ** 2 + qy[None, :, None] ** 2 + qz[None, None, :] ** 2)
    q_fundamental = 2.0 * np.pi / (shape[0] * DX_NM)
    q_max = np.pi / DX_NM
    q_edges = np.arange(0.0, q_max + 1.000001 * q_fundamental, q_fundamental)
    low_high = np.pi / (4.0 * LAMBDA_NM)
    mid_high = np.pi / LAMBDA_NM
    weights_z = np.full(qz.shape, 2.0)
    weights_z[0] = 1.0
    if shape[2] % 2 == 0:
        weights_z[-1] = 1.0
    spectral_rows: list[dict] = []
    band_rows: list[dict] = []
    box_volume_m3 = nvox * (DX_NM * 1.0e-9) ** 3
    q_select = (qmag > 0.0) & (qmag <= q_max)
    shell_index = np.searchsorted(q_edges, qmag[q_select], side="right") - 1
    shell_count = np.bincount(
        shell_index,
        weights=np.broadcast_to(weights_z, qmag.shape)[q_select],
        minlength=q_edges.size - 1,
    )
    for field_name, field in fields.items():
        fluctuation = field.astype(np.float64, copy=False) - float(np.mean(field))
        transform = np.fft.rfftn(fluctuation) / nvox
        power = np.abs(transform) ** 2
        weighted_power = power * weights_z[None, None, :]
        shell_power = np.bincount(
            shell_index,
            weights=weighted_power[q_select],
            minlength=q_edges.size - 1,
        )
        for index in range(q_edges.size - 1):
            spectral_rows.append(
                {
                    "replicate": replicate,
                    "age_h": age_h,
                    "field": field_name,
                    "direction": "isotropic_radial",
                    "q_low_nm_inv": q_edges[index],
                    "q_high_nm_inv": q_edges[index + 1],
                    "q_center_nm_inv": 0.5 * (q_edges[index] + q_edges[index + 1]),
                    "weighted_mode_count": shell_count[index],
                    "integrated_variance": shell_power[index],
                    "mean_mode_power": shell_power[index] / max(shell_count[index], 1.0),
                    "correlation_spectral_density_m3": box_volume_m3 * shell_power[index] / max(shell_count[index], 1.0),
                }
            )
        for direction, values, qvalues in (
            ("[100]", transform[:, 0, 0], np.abs(qx)),
            ("[010]", transform[0, :, 0], np.abs(qy)),
            ("[001]", transform[0, 0, :], np.abs(qz)),
        ):
            direction_power = np.abs(values) ** 2
            direction_bins = np.searchsorted(q_edges, qvalues, side="right") - 1
            for index in range(q_edges.size - 1):
                selected = direction_bins == index
                if not np.any(selected):
                    continue
                spectral_rows.append(
                    {
                        "replicate": replicate,
                        "age_h": age_h,
                        "field": field_name,
                        "direction": direction,
                        "q_low_nm_inv": q_edges[index],
                        "q_high_nm_inv": q_edges[index + 1],
                        "q_center_nm_inv": 0.5 * (q_edges[index] + q_edges[index + 1]),
                        "weighted_mode_count": int(selected.sum()),
                        "integrated_variance": float(direction_power[selected].sum()),
                        "mean_mode_power": float(direction_power[selected].mean()),
                        "correlation_spectral_density_m3": box_volume_m3 * float(direction_power[selected].mean()),
                    }
                )
        for band, lower, upper in (
            ("low_q", 0.0, low_high),
            ("mid_q", low_high, mid_high),
            ("high_q", mid_high, q_max + np.finfo(float).eps),
        ):
            selected = (qmag > lower) & (qmag <= upper)
            band_rows.append(
                {
                    "replicate": replicate,
                    "age_h": age_h,
                    "field": field_name,
                    "band": band,
                    "q_low_nm_inv": lower,
                    "q_high_nm_inv": min(upper, q_max),
                    "integrated_variance": float(weighted_power[selected].sum()),
                    "fraction_of_total_variance": float(weighted_power[selected].sum() / max(weighted_power.sum(), np.finfo(float).tiny)),
                    "parseval_total_variance": float(weighted_power.sum()),
                    "real_space_variance": float(np.var(fluctuation)),
                    "parseval_relative_error": abs(float(weighted_power.sum()) - float(np.var(fluctuation))) / max(float(np.var(fluctuation)), np.finfo(float).tiny),
                }
            )
    return spectral_rows, band_rows


def ensemble_ratio_rows(interface_rows: list[dict], strain_rows: list[dict], stress_rows: list[dict], band_rows: list[dict]) -> list[dict]:
    descriptors: dict[tuple[str, str], dict[int, float]] = {}
    for row in interface_rows:
        key = (row["replicate"], "Sv")
        descriptors.setdefault(key, {})[int(row["age_h"])] = float(row["Sv_m_inv"])
    for row in strain_rows:
        if row["region"] == "whole_box":
            descriptors.setdefault((row["replicate"], "hydrostatic_strain_variance"), {})[int(row["age_h"])] = float(row["epsilon_h_variance"])
            descriptors.setdefault((row["replicate"], "deviatoric_strain_variance"), {})[int(row["age_h"])] = float(row["epsilon_dev_equivalent_variance"])
    for row in stress_rows:
        if row["region"] == "whole_box":
            descriptors.setdefault((row["replicate"], "hydrostatic_stress_variance"), {})[int(row["age_h"])] = float(row["sigma_h_variance_Pa2"])
            descriptors.setdefault((row["replicate"], "elastic_energy_mean"), {})[int(row["age_h"])] = float(row["elastic_energy_mean_J_m3"])
    for row in band_rows:
        if row["field"] in ("epsilon_h", "epsilon_dev_equivalent"):
            name = f"{row['field']}__{row['band']}"
            descriptors.setdefault((row["replicate"], name), {})[int(row["age_h"])] = float(row["integrated_variance"])
    per_replicate: list[dict] = []
    for (replicate, descriptor), ages in sorted(descriptors.items()):
        if 6 not in ages or 48 not in ages:
            continue
        per_replicate.append(
            {
                "scope": "replicate",
                "replicate": replicate,
                "descriptor": descriptor,
                "value_6h": ages[6],
                "value_48h": ages[48],
                "ratio_48h_over_6h": ages[48] / max(ages[6], np.finfo(float).tiny),
                "trend_sign": "decrease" if ages[48] < ages[6] else ("increase" if ages[48] > ages[6] else "flat"),
                "mean": "",
                "sample_standard_deviation": "",
                "minimum": "",
                "maximum": "",
                "trend_sign_consistent": "",
            }
        )
    output = list(per_replicate)
    for descriptor in sorted({row["descriptor"] for row in per_replicate}):
        selected = [row for row in per_replicate if row["descriptor"] == descriptor]
        ratios = np.array([row["ratio_48h_over_6h"] for row in selected], dtype=float)
        signs = {row["trend_sign"] for row in selected}
        output.append(
            {
                "scope": "ensemble",
                "replicate": "A+B+C",
                "descriptor": descriptor,
                "value_6h": "",
                "value_48h": "",
                "ratio_48h_over_6h": "",
                "trend_sign": "",
                "mean": float(ratios.mean()),
                "sample_standard_deviation": float(ratios.std(ddof=1)),
                "minimum": float(ratios.min()),
                "maximum": float(ratios.max()),
                "trend_sign_consistent": len(signs) == 1,
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", type=Path, required=True, help="root containing A/age_6h ... C/age_48h")
    parser.add_argument("--parameter-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)
    params = read_params(args.parameter_file)
    gamma = float(params["gamma_Jm2"])
    lambda_m = float(params["lambda_sm_m"])
    stress_scale_pa = 12.0 * gamma / lambda_m
    shape = (246, 246, 246)

    interface_rows: list[dict] = []
    normal_rows: list[dict] = []
    curvature_rows: list[dict] = []
    shape_rows: list[dict] = []
    strain_rows: list[dict] = []
    stress_rows: list[dict] = []
    spectrum_rows: list[dict] = []
    band_rows: list[dict] = []
    source_rows: list[dict] = []
    for replicate in REPLICATES:
        for age_h in AGES:
            field_root = args.replay_root / replicate / f"age_{age_h}h" / "fields"
            summary_path = field_root / "replay_summary.txt"
            if not summary_path.is_file():
                raise FileNotFoundError(summary_path)
            phi = load_field(field_root, "accepted_phi.raw.f64", "<f8", shape)
            source_rows.append(
                {
                    "replicate": replicate,
                    "age_h": age_h,
                    "field_root": str(field_root),
                    "accepted_phi_sha256": sha256(field_root / "accepted_phi.raw.f64"),
                    "accepted_xB_sha256": sha256(field_root / "accepted_xB.raw.f64"),
                    "replay_summary_sha256": sha256(summary_path),
                }
            )
            geometry, normals, curvatures, signed_distance = interface_geometry(phi, replicate, age_h)
            shapes = particle_shapes(phi, replicate, age_h)
            geometry["particle_count_phi_ge_0p5"] = len(shapes)
            geometry["aspect_1_over_3_mean"] = float(np.mean([row["aspect_1_over_3"] for row in shapes]))
            geometry["aspect_1_over_3_max"] = float(np.max([row["aspect_1_over_3"] for row in shapes]))
            interface_rows.append(geometry)
            normal_rows.extend(normals)
            curvature_rows.extend(curvatures)
            shape_rows.extend(shapes)
            strain = {component: load_field(field_root, f"strain_{component}.raw.f32", "<f4", shape).astype(np.float64) for component in COMPONENTS}
            stress_pa = {component: load_field(field_root, f"stress_{component}.raw.f32", "<f4", shape).astype(np.float64) * stress_scale_pa for component in COMPONENTS}
            energy = load_field(field_root, "elastic_energy_density.raw.f64", "<f8", shape) * stress_scale_pa
            case_strain, case_stress, spectral_fields = field_statistics(
                replicate, age_h, phi, signed_distance, strain, stress_pa, energy
            )
            case_spectrum, case_bands = spectral_analysis(replicate, age_h, spectral_fields)
            strain_rows.extend(case_strain)
            stress_rows.extend(case_stress)
            spectrum_rows.extend(case_spectrum)
            band_rows.extend(case_bands)

    ratio_rows = ensemble_ratio_rows(interface_rows, strain_rows, stress_rows, band_rows)
    write_csv(args.out / "interface_statistics_time_series.csv", interface_rows)
    write_csv(args.out / "interface_normal_distribution.csv", normal_rows)
    write_csv(args.out / "interface_curvature_distribution.csv", curvature_rows)
    write_csv(args.out / "particle_shape_statistics.csv", shape_rows)
    write_csv(args.out / "strain_statistics_time_series.csv", strain_rows)
    write_csv(args.out / "stress_statistics_time_series.csv", stress_rows)
    write_csv(args.out / "strain_power_spectrum.csv", spectrum_rows)
    write_csv(args.out / "strain_band_integrals.csv", band_rows)
    write_csv(args.out / "structural_relaxation_ratios.csv", ratio_rows)
    write_csv(args.out / "descriptor_source_manifest.csv", source_rows)

    manifest = {
        "schema": "PF_PERIODIC_INTERFACE_STRAIN_DESCRIPTOR_AUDIT_V1",
        "status": "PASS_PF_PERIODIC_INTERFACE_STRAIN_DESCRIPTOR_AUDIT_V1",
        "grid": list(shape),
        "dx_nm": DX_NM,
        "lambda_nm": LAMBDA_NM,
        "marching_cubes_level": PHI_LEVEL,
        "periodic_marching_cubes_contract": "N+1 wrapped endpoint lattice over exactly N cells",
        "core_region_contract": {"matrix": "phi<=0.1", "beta": "phi>=0.9"},
        "shell_contract_nm": [0.0, LAMBDA_NM, 2.0 * LAMBDA_NM, 4.0 * LAMBDA_NM],
        "q_bin_width_nm_inv": 2.0 * np.pi / (shape[0] * DX_NM),
        "q_max_nm_inv": np.pi / DX_NM,
        "q_bands_nm_inv": {
            "low": [0.0, np.pi / (4.0 * LAMBDA_NM)],
            "mid": [np.pi / (4.0 * LAMBDA_NM), np.pi / LAMBDA_NM],
            "high": [np.pi / LAMBDA_NM, np.pi / DX_NM],
        },
        "zero_mode_removed": True,
        "fft_normalization": "rfftn(field-mean)/Nvox with Hermitian multiplicity and Parseval audit",
        "stress_scale_Pa_per_code": stress_scale_pa,
        "parameter_file": str(args.parameter_file),
        "parameter_sha256": sha256(args.parameter_file),
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
    }
    (args.out / "descriptor_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_files = sorted(path for path in args.out.iterdir() if path.is_file())
    with (args.out / "descriptor_outputs.sha256").open("w", encoding="utf-8") as stream:
        for path in output_files:
            stream.write(f"{sha256(path)}  {path.name}\n")
    (args.out / "status.txt").write_text(manifest["status"] + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
