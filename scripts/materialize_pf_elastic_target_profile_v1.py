#!/usr/bin/env python3
"""Materialize one constrained elastic PF target profile.

The source run must be a single resolved-beta, full-model elastic
minimization with both constraints active:

* fixed ``sum(h(phi))`` (particle h-volume);
* fixed canonical B inventory
  ``sum((1-h(phi))*xB_alpha + v_B*h(phi))``.

The script converts the final VTK pair into deterministic little-endian raw
fields, computes shape/composition audits, and writes a hash-pinned manifest.
It never edits or replaces the source run.
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


SCHEMA = "PF_ELASTIC_TARGET_PROFILE_V1"
CONSTRAINT_MODE = "MINIMIZE_CONSERVED_MASS_CONSTRAINT_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def h_of_phi(phi: np.ndarray) -> np.ndarray:
    p2 = phi * phi
    return p2 * phi * (6.0 * p2 - 15.0 * phi + 10.0)


def read_legacy_ascii_vtk(path: Path) -> tuple[np.ndarray, tuple[int, int, int], str]:
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) < 10 or lines[2].strip().upper() != "ASCII":
        raise ValueError(f"{path}: only legacy ASCII VTK is accepted")
    dims: tuple[int, int, int] | None = None
    scalar_name = ""
    data_line = -1
    for index, line in enumerate(lines):
        fields = line.split()
        if fields and fields[0].upper() == "DIMENSIONS" and len(fields) == 4:
            dims = tuple(int(value) for value in fields[1:4])
        elif fields and fields[0].upper() == "SCALARS" and len(fields) >= 2:
            scalar_name = fields[1]
        elif fields and fields[0].upper() == "LOOKUP_TABLE":
            data_line = index + 1
            break
    if dims is None or data_line < 0 or not scalar_name:
        raise ValueError(f"{path}: incomplete VTK scalar header")
    values = np.fromstring("\n".join(lines[data_line:]), sep=" ", dtype=np.float64)
    expected = int(np.prod(dims))
    if values.size != expected:
        raise ValueError(f"{path}: expected {expected} scalars, found {values.size}")
    # write_vtk_cuda emits x as the fastest scalar-stream coordinate.  A
    # Fortran reshape restores the runtime array indexed as [i,j,k].
    return values.reshape(dims, order="F"), dims, scalar_name


def read_raw_f64(
    path: Path, dims: tuple[int, int, int], field_name: str
) -> tuple[np.ndarray, tuple[int, int, int], str]:
    values = np.fromfile(path, dtype="<f8")
    expected = int(np.prod(dims))
    if values.size != expected:
        raise ValueError(f"{path}: expected {expected} float64 values, found {values.size}")
    return values.reshape(dims, order="C"), dims, field_name


def parse_param_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def parse_last_constraint_trace(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: empty constraint trace")
    required = {
        "iter",
        "target_mass_code",
        "current_mass_code",
        "lambda",
        "residual_code",
        "residual_relative",
        "accepted_constraint_steps",
    }
    missing = required.difference(rows[-1])
    if missing:
        raise ValueError(f"{path}: missing trace columns {sorted(missing)}")
    return rows[-1]


def require_final_pass(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    convergence_candidates = [
        line
        for line in text.splitlines()
        if line.startswith(
            "MINIMIZE_TARGET_PROFILE_CONVERGENCE_FINAL_AUDIT "
        )
    ]
    if not convergence_candidates:
        raise ValueError(
            f"{path}: target-profile convergence marker is absent"
        )
    convergence_tokens: dict[str, str] = {}
    for item in convergence_candidates[-1].split()[1:]:
        if "=" in item:
            key, value = item.split("=", 1)
            convergence_tokens[key] = value
    if (
        convergence_tokens.get("status") != "PASS"
        or convergence_tokens.get("converged") != "true"
    ):
        raise ValueError(
            f"{path}: target-profile convergence marker is not an exact PASS"
        )
    candidates = [
        line
        for line in text.splitlines()
        if line.startswith("MINIMIZE_MASS_CONSTRAINT_FINAL_AUDIT ")
    ]
    if not candidates:
        raise ValueError(f"{path}: final mass-constraint marker is absent")
    marker = candidates[-1]
    tokens: dict[str, str] = {}
    for item in marker.split()[1:]:
        if "=" in item:
            key, value = item.split("=", 1)
            tokens[key] = value
    if tokens.get("status") != "PASS" or tokens.get("mode") != CONSTRAINT_MODE:
        raise ValueError(f"{path}: final constraint marker is not an exact PASS")
    prohibited = (
        "enable_gp_assisted_beta_nucleation=1",
        "enable_gp_runtime_library_nucleation=1",
        "gp_initial_population_enabled=1",
        "gp_birth_model=poisson",
    )
    if any(item in text for item in prohibited):
        raise ValueError(f"{path}: prohibited GP/source path appears active")
    for key, value in convergence_tokens.items():
        tokens[f"convergence_{key}"] = value
    return tokens


def periodic_centroid_and_shape(
    h: np.ndarray, dx_nm: float
) -> tuple[list[float], list[float], list[list[float]], list[float]]:
    total = float(np.sum(h, dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("non-positive h-volume")
    shape = h.shape
    centroid_grid: list[float] = []
    for axis, count in enumerate(shape):
        marginal_axes = tuple(i for i in range(3) if i != axis)
        weights = np.sum(h, axis=marginal_axes, dtype=np.float64)
        angles = 2.0 * math.pi * np.arange(count, dtype=np.float64) / count
        sine = float(np.dot(weights, np.sin(angles)))
        cosine = float(np.dot(weights, np.cos(angles)))
        angle = math.atan2(sine, cosine) % (2.0 * math.pi)
        centroid_grid.append(angle * count / (2.0 * math.pi))

    displacements: list[np.ndarray] = []
    for axis, count in enumerate(shape):
        coordinate = np.arange(count, dtype=np.float64)
        delta = (coordinate - centroid_grid[axis] + 0.5 * count) % count - 0.5 * count
        reshape = [1, 1, 1]
        reshape[axis] = count
        displacements.append(delta.reshape(reshape) * dx_nm)

    covariance = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(i, 3):
            value = float(np.sum(h * displacements[i] * displacements[j], dtype=np.float64) / total)
            covariance[i, j] = covariance[j, i] = value
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if np.any(eigenvalues <= 0.0):
        raise ValueError("shape tensor is not positive definite")
    semi_axes = np.sqrt(5.0 * eigenvalues)
    # Eigenvector signs are arbitrary.  Canonicalize each unoriented axis for
    # stable manifests without changing its physical meaning.
    for column in range(3):
        vector = eigenvectors[:, column]
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0.0:
            eigenvectors[:, column] *= -1.0
    centroid_nm = [value * dx_nm for value in centroid_grid]
    return (
        centroid_nm,
        semi_axes.tolist(),
        eigenvectors.T.tolist(),
        eigenvalues.tolist(),
    )


def write_raw(path: Path, array: np.ndarray) -> None:
    array.astype("<f8", copy=False).ravel(order="C").tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phi-vtk", type=Path)
    parser.add_argument("--xb-vtk", type=Path)
    parser.add_argument("--phi-raw", type=Path)
    parser.add_argument("--xb-raw", type=Path)
    parser.add_argument("--grid-nx", type=int)
    parser.add_argument("--grid-ny", type=int)
    parser.add_argument("--grid-nz", type=int)
    parser.add_argument("--constraint-trace", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--param-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-radius-nm", type=float, required=True)
    parser.add_argument("--temperature-c", type=float, default=380.0)
    parser.add_argument("--dx-nm", type=float, default=1.0)
    parser.add_argument("--lambda-sm-nm", type=float, default=4.0)
    parser.add_argument("--v-b", type=float, default=1.0)
    parser.add_argument("--dt-recommended", type=float, default=0.02)
    parser.add_argument("--orientation-label", default="variant_100_identity")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--radius-relative-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--mass-relative-tolerance", type=float, default=1.0e-12)
    args = parser.parse_args()

    vtk_mode = args.phi_vtk is not None or args.xb_vtk is not None
    raw_mode = args.phi_raw is not None or args.xb_raw is not None
    if vtk_mode == raw_mode:
        raise SystemExit("provide exactly one complete phi/xB source pair: VTK or raw")
    if vtk_mode and (args.phi_vtk is None or args.xb_vtk is None):
        raise SystemExit("both --phi-vtk and --xb-vtk are required")
    if raw_mode and (
        args.phi_raw is None
        or args.xb_raw is None
        or args.grid_nx is None
        or args.grid_ny is None
        or args.grid_nz is None
    ):
        raise SystemExit("raw input requires phi/xB plus all three grid dimensions")
    field_inputs = (
        (args.phi_vtk, args.xb_vtk)
        if vtk_mode
        else (args.phi_raw, args.xb_raw)
    )
    inputs = field_inputs + (
        args.constraint_trace,
        args.run_log,
        args.param_file,
    )
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty profile root: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    final_marker = require_final_pass(args.run_log)
    trace = parse_last_constraint_trace(args.constraint_trace)
    params = parse_param_file(args.param_file)
    if vtk_mode:
        phi, dims, phi_name = read_legacy_ascii_vtk(args.phi_vtk)
        xb, xb_dims, xb_name = read_legacy_ascii_vtk(args.xb_vtk)
    else:
        raw_dims = (args.grid_nx, args.grid_ny, args.grid_nz)
        phi, dims, phi_name = read_raw_f64(args.phi_raw, raw_dims, "phi")
        xb, xb_dims, xb_name = read_raw_f64(args.xb_raw, raw_dims, "xB")
    if dims != xb_dims:
        raise SystemExit("phi/xB VTK dimensions differ")
    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(xb)):
        raise SystemExit("non-finite profile field")
    bound_tol = 5.0e-12
    if float(np.min(phi)) < -bound_tol or float(np.max(phi)) > 1.0 + bound_tol:
        raise SystemExit("phi lies outside [0,1]")
    if float(np.min(xb)) <= 0.0 or float(np.max(xb)) >= 1.0:
        raise SystemExit("xB_alpha lies outside (0,1)")
    phi = np.clip(phi, 0.0, 1.0)

    h = h_of_phi(phi)
    ctot = (1.0 - h) * xb + args.v_b * h
    y = np.log(xb / (1.0 - xb))
    h_volume_nm3 = float(np.sum(h, dtype=np.float64) * args.dx_nm**3)
    equivalent_radius_nm = (3.0 * h_volume_nm3 / (4.0 * math.pi)) ** (1.0 / 3.0)
    radius_error_relative = abs(equivalent_radius_nm - args.target_radius_nm) / args.target_radius_nm
    if radius_error_relative > args.radius_relative_tolerance:
        raise SystemExit(
            f"equivalent radius mismatch: target={args.target_radius_nm:.12g}, "
            f"actual={equivalent_radius_nm:.12g}, rel={radius_error_relative:.3e}"
        )

    target_mass = float(trace["target_mass_code"])
    current_mass = float(np.sum(ctot, dtype=np.float64))
    mass_error = current_mass - target_mass
    mass_error_relative = abs(mass_error) / max(abs(target_mass), 1.0)
    if mass_error_relative > args.mass_relative_tolerance:
        raise SystemExit(
            f"canonical mass mismatch: target={target_mass:.17e}, "
            f"materialized={current_mass:.17e}, rel={mass_error_relative:.3e}"
        )
    if abs(float(trace["current_mass_code"]) - current_mass) / max(abs(target_mass), 1.0) > 5.0e-13:
        raise SystemExit("VTK materialization does not match the final constraint trace")

    centroid_nm, semi_axes_nm, principal_axes, shape_eigenvalues = (
        periodic_centroid_and_shape(h, args.dx_nm)
    )
    axis_ratio_major_minor = semi_axes_nm[0] / semi_axes_nm[2]
    far_mask = h <= 1.0e-8
    if int(np.count_nonzero(far_mask)) < max(64, int(0.05 * h.size)):
        raise SystemExit("insufficient far-field support")
    far_xb_mean = float(np.mean(xb[far_mask], dtype=np.float64))
    far_xb_std = float(np.std(xb[far_mask], dtype=np.float64))
    far_xb_min = float(np.min(xb[far_mask]))
    far_xb_max = float(np.max(xb[far_mask]))
    relaxation_correction = (1.0 - h) * (xb - far_xb_mean)

    field_paths = {
        "phi": args.out / "phi.raw.f64",
        "xB_alpha": args.out / "xB_alpha.raw.f64",
        "Y": args.out / "Y.raw.f64",
        "C_B_tot": args.out / "C_B_tot.raw.f64",
        "h_phi": args.out / "h_phi.raw.f64",
        "delta_C_relaxation": args.out / "delta_C_relaxation.raw.f64",
        "dY_dt_prev": args.out / "dY_dt_prev.raw.f64",
    }
    write_raw(field_paths["phi"], phi)
    write_raw(field_paths["xB_alpha"], xb)
    write_raw(field_paths["Y"], y)
    write_raw(field_paths["C_B_tot"], ctot)
    write_raw(field_paths["h_phi"], h)
    write_raw(field_paths["delta_C_relaxation"], relaxation_correction)
    write_raw(field_paths["dY_dt_prev"], np.zeros_like(phi))

    source_hashes = {path.name: sha256(path) for path in inputs}
    fields: dict[str, dict[str, Any]] = {}
    for name, path in field_paths.items():
        fields[name] = {
            "path": path.name,
            "sha256": sha256(path),
            "dtype": "float64-le",
            "order": "C",
        }
    xB_max_safe = float(params.get("minimize_xB_max_safe", "0.499999"))
    init_meta = {
        "schema": "PF_RAW_INIT_META_V1",
        "Nx": dims[0],
        "Ny": dims[1],
        "Nz": dims[2],
        "dx_nm": args.dx_nm,
        "interface_width_nm": args.lambda_sm_nm,
        "dt_recommended": args.dt_recommended,
        "mean_xBtot": current_mass / float(h.size),
        "xB_max_safe": xB_max_safe,
        "dtype": "float64",
        "order": "C",
        "phi_path": field_paths["phi"].name,
        "xB_path": field_paths["xB_alpha"].name,
        "phi_sha256": fields["phi"]["sha256"],
        "xB_sha256": fields["xB_alpha"]["sha256"],
        "source_profile_schema": SCHEMA,
    }
    init_meta_path = args.out / "init_meta.json"
    init_meta_path.write_text(
        json.dumps(init_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "scientific_role": "single_resolved_beta_elastic_constrained_target_profile",
        "not_a_dynamic_ageing_result": True,
        "temperature_C": args.temperature_c,
        "grid": {"Nx": dims[0], "Ny": dims[1], "Nz": dims[2], "dx_nm": args.dx_nm},
        "lambda_sm_nm": args.lambda_sm_nm,
        "v_B": args.v_b,
        "orientation_label": args.orientation_label,
        "orientation_scope": "single_frozen_identity_variant",
        "source_commit": args.source_commit,
        "source_tree_sha256": args.source_tree_sha256,
        "binary_sha256": args.binary_sha256,
        "constraint_contract": {
            "volume": "fixed_sum_h_phi",
            "mass": CONSTRAINT_MODE,
            "canonical_field": "(1-h(phi))*xB_alpha+v_B*h(phi)",
            "target_mass_code": target_mass,
            "materialized_mass_code": current_mass,
            "mass_error_code": mass_error,
            "mass_error_relative": mass_error_relative,
            "last_lambda": float(trace["lambda"]),
            "accepted_constraint_steps": int(trace["accepted_constraint_steps"]),
            "final_marker": final_marker,
        },
        "geometry": {
            "target_equivalent_radius_nm": args.target_radius_nm,
            "actual_equivalent_radius_nm": equivalent_radius_nm,
            "radius_error_relative": radius_error_relative,
            "h_volume_nm3": h_volume_nm3,
            "periodic_h_centroid_nm": centroid_nm,
            "shape_tensor_eigenvalues_nm2": shape_eigenvalues,
            "ellipsoid_semi_axes_nm": semi_axes_nm,
            "axis_ratio_major_minor": axis_ratio_major_minor,
            "principal_axes_rows": principal_axes,
        },
        "composition": {
            "xB_alpha_min": float(np.min(xb)),
            "xB_alpha_max": float(np.max(xb)),
            "xB_alpha_mean": float(np.mean(xb, dtype=np.float64)),
            "far_field_h_threshold": 1.0e-8,
            "far_field_voxel_count": int(np.count_nonzero(far_mask)),
            "far_field_xB_mean": far_xb_mean,
            "far_field_xB_std": far_xb_std,
            "far_field_xB_min": far_xb_min,
            "far_field_xB_max": far_xb_max,
            "relaxation_correction_integral_code": float(
                np.sum(relaxation_correction, dtype=np.float64)
            ),
        },
        "source_field_format": "legacy_ascii_vtk" if vtk_mode else "float64_le_raw_C",
        "source_field_names": {"phi": phi_name, "xB_alpha": xb_name},
        "source_hashes": source_hashes,
        "parameter_values": params,
        "fields": fields,
        "runtime_load_contract": {
            "init_mode": "raw_fields",
            "init_phi_raw": field_paths["phi"].name,
            "init_xB_raw": field_paths["xB_alpha"].name,
            "init_meta": init_meta_path.name,
            "init_meta_sha256": sha256(init_meta_path),
            "eta_field": "implicit_zero",
            "dY_dt_prev_initialization": "zero_for_fresh_dynamic_start",
        },
    }
    manifest_path = args.out / "profile_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_sha = sha256(manifest_path)
    terminal = {
        "profile_status": "PASS_ELASTIC_CONSTRAINED_TARGET_PROFILE_V1",
        "profile_manifest_sha256": manifest_sha,
        "target_radius_nm": args.target_radius_nm,
        "actual_radius_nm": equivalent_radius_nm,
        "axis_ratio_major_minor": axis_ratio_major_minor,
        "mass_error_relative": mass_error_relative,
        "elastic_enabled": int(float(params.get("elastic_enabled", "1"))),
        "gp_enabled": False,
    }
    (args.out / "final_terminal_output.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in terminal.items()) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(terminal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
