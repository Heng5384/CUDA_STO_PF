#!/usr/bin/env python3
"""Qualify one elastic profile under short production dynamics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analyze_pf_zero_mode_checkpoints import read_checkpoint
from materialize_pf_elastic_target_profile_v1 import (
    h_of_phi,
    periodic_centroid_and_shape,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_profile_field(
    manifest: dict[str, Any], manifest_path: Path, name: str
) -> np.ndarray:
    field = manifest["fields"][name]
    path = manifest_path.parent / field["path"]
    if sha256(path) != field["sha256"]:
        raise ValueError(f"{path}: field hash mismatch")
    shape = (
        int(manifest["grid"]["Nx"]),
        int(manifest["grid"]["Ny"]),
        int(manifest["grid"]["Nz"]),
    )
    array = np.fromfile(path, dtype="<f8")
    if array.size != math.prod(shape):
        raise ValueError(f"{path}: field size mismatch")
    return array.reshape(shape, order="C")


def geometry(phi: np.ndarray, dx_nm: float) -> dict[str, Any]:
    h = h_of_phi(np.clip(phi, 0.0, 1.0))
    volume = float(np.sum(h, dtype=np.float64) * dx_nm**3)
    radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    centroid, axes, directions, eigenvalues = periodic_centroid_and_shape(
        h, dx_nm
    )
    return {
        "h_volume_nm3": volume,
        "equivalent_radius_nm": radius,
        "centroid_nm": centroid,
        "semi_axes_nm": axes,
        "axis_ratio": axes[0] / axes[2],
        "principal_axes_rows": directions,
        "shape_eigenvalues_nm2": eigenvalues,
    }


def periodic_centroid_error(
    lhs: list[float], rhs: list[float], periods: list[float]
) -> float:
    values = []
    for a, b, period in zip(lhs, rhs, periods):
        delta = abs(float(a) - float(b))
        values.append(min(delta, period - delta))
    return max(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-manifest", type=Path, required=True)
    parser.add_argument("--initial-probe-checkpoint", type=Path, required=True)
    parser.add_argument("--continuous-checkpoint", type=Path, required=True)
    parser.add_argument("--restart-checkpoint", type=Path, required=True)
    parser.add_argument("--refined-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-probe-stdout", type=Path, required=True)
    parser.add_argument("--continuous-stdout", type=Path, required=True)
    parser.add_argument("--restart-stdout", type=Path, required=True)
    parser.add_argument("--refined-stdout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--initial-normalized-axis-relative-max",
        type=float,
        default=2.0e-2,
    )
    parser.add_argument(
        "--initial-axis-ratio-relative-max",
        type=float,
        default=2.0e-2,
    )
    parser.add_argument(
        "--initial-centroid-shift-max-nm",
        type=float,
        default=1.0e-1,
    )
    parser.add_argument("--dt-phi-l1-max", type=float, default=2.0e-2)
    parser.add_argument("--dt-axis-relative-max", type=float, default=2.0e-2)
    parser.add_argument(
        "--dt-xB-mean-absolute-preferred",
        type=float,
        default=2.0e-5,
    )
    parser.add_argument(
        "--dt-xB-mean-absolute-hard",
        type=float,
        default=5.0e-5,
    )
    parser.add_argument("--mass-relative-max", type=float, default=1.0e-10)
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.profile_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "PF_ELASTIC_TARGET_PROFILE_V1":
        raise SystemExit("wrong profile schema")
    phi_initial = load_profile_field(
        manifest, args.profile_manifest, "phi"
    )
    xb_initial = load_profile_field(
        manifest, args.profile_manifest, "xB_alpha"
    )
    c_initial = load_profile_field(
        manifest, args.profile_manifest, "C_B_tot"
    )
    dx_nm = float(manifest["grid"]["dx_nm"])
    shape = phi_initial.shape

    (
        initial_probe_step,
        initial_probe_shape,
        initial_probe_dt,
        _initial_probe_temp,
        _initial_probe_target,
        phi_initial_probe_flat,
        _Y_initial_probe,
        xb_initial_probe_flat,
        _dY_initial_probe,
    ) = read_checkpoint(args.initial_probe_checkpoint)
    (
        continuous_step,
        continuous_shape,
        continuous_dt,
        _continuous_temp,
        _continuous_target,
        phi_cont_flat,
        _Y_cont,
        xb_cont_flat,
        _dY_cont,
    ) = read_checkpoint(args.continuous_checkpoint)
    (
        restart_step,
        restart_shape,
        restart_dt,
        _restart_temp,
        _restart_target,
        phi_restart_flat,
        _Y_restart,
        xb_restart_flat,
        _dY_restart,
    ) = read_checkpoint(args.restart_checkpoint)
    (
        refined_step,
        refined_shape,
        refined_dt,
        _refined_temp,
        _refined_target,
        phi_refined_flat,
        _Y_refined,
        xb_refined_flat,
        _dY_refined,
    ) = read_checkpoint(args.refined_checkpoint)
    if tuple(initial_probe_shape) != shape:
        raise SystemExit("initial-probe checkpoint/profile shape mismatch")
    if initial_probe_step != 1 or initial_probe_dt != continuous_dt:
        raise SystemExit("initial probe is not exactly one production step")
    if tuple(continuous_shape) != shape or tuple(restart_shape) != shape:
        raise SystemExit("checkpoint/profile shape mismatch")
    if tuple(refined_shape) != shape:
        raise SystemExit("refined checkpoint/profile shape mismatch")
    if abs(continuous_step * continuous_dt - refined_step * refined_dt) > 1e-12:
        raise SystemExit("dt-refinement endpoints differ")
    if continuous_step != restart_step or continuous_dt != restart_dt:
        raise SystemExit("continuous/restart endpoints differ")

    phi_initial_probe = np.asarray(phi_initial_probe_flat).reshape(
        shape, order="C"
    )
    xb_initial_probe = np.asarray(xb_initial_probe_flat).reshape(
        shape, order="C"
    )
    phi_cont = np.asarray(phi_cont_flat).reshape(shape, order="C")
    xb_cont = np.asarray(xb_cont_flat).reshape(shape, order="C")
    phi_restart = np.asarray(phi_restart_flat).reshape(shape, order="C")
    xb_restart = np.asarray(xb_restart_flat).reshape(shape, order="C")
    phi_refined = np.asarray(phi_refined_flat).reshape(shape, order="C")
    xb_refined = np.asarray(xb_refined_flat).reshape(shape, order="C")
    h_initial = h_of_phi(phi_initial)
    h_initial_probe = h_of_phi(phi_initial_probe)
    h_cont = h_of_phi(phi_cont)
    h_refined = h_of_phi(phi_refined)
    c_cont = (1.0 - h_cont) * xb_cont + float(manifest["v_B"]) * h_cont
    c_refined = (
        (1.0 - h_refined) * xb_refined
        + float(manifest["v_B"]) * h_refined
    )
    initial_mass = float(np.sum(c_initial, dtype=np.float64))
    continuous_mass = float(np.sum(c_cont, dtype=np.float64))
    refined_mass = float(np.sum(c_refined, dtype=np.float64))
    mass_relative = max(
        abs(continuous_mass - initial_mass),
        abs(refined_mass - initial_mass),
    ) / max(abs(initial_mass), 1.0)

    initial_geometry = geometry(phi_initial, dx_nm)
    initial_probe_geometry = geometry(phi_initial_probe, dx_nm)
    continuous_geometry = geometry(phi_cont, dx_nm)
    refined_geometry = geometry(phi_refined, dx_nm)
    initial_probe_phi_l1 = float(
        np.sum(np.abs(phi_initial_probe - phi_initial), dtype=np.float64)
        / max(float(np.sum(h_initial, dtype=np.float64)), 1.0)
    )
    initial_probe_xB_mean_absolute = float(
        np.mean(np.abs(xb_initial_probe - xb_initial), dtype=np.float64)
    )
    initial_probe_volume_relative = abs(
        float(np.sum(h_initial_probe, dtype=np.float64))
        - float(np.sum(h_initial, dtype=np.float64))
    ) / max(float(np.sum(h_initial, dtype=np.float64)), 1.0)
    dt_phi_l1 = float(
        np.sum(np.abs(phi_cont - phi_refined), dtype=np.float64)
        / max(
            0.5
            * float(
                np.sum(h_cont, dtype=np.float64)
                + np.sum(h_refined, dtype=np.float64)
            ),
            1.0,
        )
    )
    initial_axes = np.asarray(initial_geometry["semi_axes_nm"])
    initial_probe_axes = np.asarray(
        initial_probe_geometry["semi_axes_nm"]
    )
    continuous_axes = np.asarray(continuous_geometry["semi_axes_nm"])
    refined_axes = np.asarray(refined_geometry["semi_axes_nm"])
    initial_normalized_axes = (
        initial_axes / float(initial_geometry["equivalent_radius_nm"])
    )
    initial_probe_normalized_axes = (
        initial_probe_axes
        / float(initial_probe_geometry["equivalent_radius_nm"])
    )
    initial_normalized_axis_relative = float(
        np.max(
            np.abs(
                initial_probe_normalized_axes - initial_normalized_axes
            )
            / np.maximum(np.abs(initial_normalized_axes), 1e-30)
        )
    )
    initial_axis_ratio_relative = abs(
        float(initial_probe_geometry["axis_ratio"])
        - float(initial_geometry["axis_ratio"])
    ) / max(abs(float(initial_geometry["axis_ratio"])), 1e-30)
    dt_axis_relative = float(
        np.max(
            np.abs(continuous_axes - refined_axes)
            / np.maximum(np.abs(continuous_axes), 1e-30)
        )
    )
    dt_xB_mean_absolute = float(
        np.mean(np.abs(xb_cont - xb_refined), dtype=np.float64)
    )
    periods = [
        shape[0] * dx_nm,
        shape[1] * dx_nm,
        shape[2] * dx_nm,
    ]
    initial_probe_centroid_shift_nm = periodic_centroid_error(
        initial_geometry["centroid_nm"],
        initial_probe_geometry["centroid_nm"],
        periods,
    )
    restart_bytewise = (
        args.continuous_checkpoint.read_bytes()
        == args.restart_checkpoint.read_bytes()
    )
    restart_array_exact = (
        np.array_equal(phi_cont, phi_restart)
        and np.array_equal(xb_cont, xb_restart)
    )
    logs = [
        args.initial_probe_stdout.read_text(
            encoding="utf-8", errors="replace"
        ),
        args.continuous_stdout.read_text(encoding="utf-8", errors="replace"),
        args.restart_stdout.read_text(encoding="utf-8", errors="replace"),
        args.refined_stdout.read_text(encoding="utf-8", errors="replace"),
    ]
    zero_mode_pass = all(
        "PF_ZERO_MODE_FINAL_AUDIT status=PASS" in text for text in logs
    )
    prohibited_paths_absent = all(
        marker not in "\n".join(logs)
        for marker in (
            "GP_EVENT",
            "GP_BIRTH",
            "BETA_NUCLEATION_EVENT",
        )
    )
    gates = {
        "restart_bytewise": restart_bytewise,
        "restart_array_exact": restart_array_exact,
        "zero_mode": zero_mode_pass,
        "prohibited_paths_absent": prohibited_paths_absent,
        "mass": mass_relative <= args.mass_relative_max,
        "initial_normalized_shape": (
            initial_normalized_axis_relative
            <= args.initial_normalized_axis_relative_max
        ),
        "initial_axis_ratio": (
            initial_axis_ratio_relative
            <= args.initial_axis_ratio_relative_max
        ),
        "initial_centroid": (
            initial_probe_centroid_shift_nm
            <= args.initial_centroid_shift_max_nm
        ),
        "dt_phi": dt_phi_l1 <= args.dt_phi_l1_max,
        "dt_shape": dt_axis_relative <= args.dt_axis_relative_max,
        "dt_xB_hard": (
            dt_xB_mean_absolute <= args.dt_xB_mean_absolute_hard
        ),
    }
    passed = all(gates.values())
    audit = {
        "schema": "PF_ELASTIC_TARGET_PROFILE_DYNAMIC_QUALIFICATION_V1",
        "status": (
            "PASS_ELASTIC_TARGET_PROFILE_DYNAMIC_RESTART_AND_DT_V1"
            if passed
            else "FAIL_ELASTIC_TARGET_PROFILE_DYNAMIC_RESTART_OR_DT_V1"
        ),
        "profile_manifest": str(args.profile_manifest),
        "profile_manifest_sha256": sha256(args.profile_manifest),
        "initial_probe_checkpoint_sha256": sha256(
            args.initial_probe_checkpoint
        ),
        "continuous_checkpoint_sha256": sha256(args.continuous_checkpoint),
        "restart_checkpoint_sha256": sha256(args.restart_checkpoint),
        "refined_checkpoint_sha256": sha256(args.refined_checkpoint),
        "initial_probe_step": initial_probe_step,
        "initial_probe_dt_code": initial_probe_dt,
        "continuous_step": continuous_step,
        "continuous_dt_code": continuous_dt,
        "refined_step": refined_step,
        "refined_dt_code": refined_dt,
        "mass_relative": mass_relative,
        "initial_probe_phi_l1_normalized": initial_probe_phi_l1,
        "initial_probe_xB_mean_absolute": initial_probe_xB_mean_absolute,
        "initial_probe_volume_relative": initial_probe_volume_relative,
        "initial_normalized_axis_relative_max": (
            initial_normalized_axis_relative
        ),
        "initial_axis_ratio_relative": initial_axis_ratio_relative,
        "initial_probe_centroid_shift_nm": (
            initial_probe_centroid_shift_nm
        ),
        "dt_phi_l1_normalized": dt_phi_l1,
        "dt_axis_relative_max": dt_axis_relative,
        "dt_xB_mean_absolute": dt_xB_mean_absolute,
        "dt_xB_preferred_status": (
            "PASS"
            if dt_xB_mean_absolute
            <= args.dt_xB_mean_absolute_preferred
            else "NOT_MET"
        ),
        "thresholds": {
            "initial_normalized_axis_relative_max": (
                args.initial_normalized_axis_relative_max
            ),
            "initial_axis_ratio_relative_max": (
                args.initial_axis_ratio_relative_max
            ),
            "initial_centroid_shift_max_nm": (
                args.initial_centroid_shift_max_nm
            ),
            "dt_phi_l1_max": args.dt_phi_l1_max,
            "dt_axis_relative_max": args.dt_axis_relative_max,
            "dt_xB_mean_absolute_preferred": (
                args.dt_xB_mean_absolute_preferred
            ),
            "dt_xB_mean_absolute_hard": (
                args.dt_xB_mean_absolute_hard
            ),
            "mass_relative_max": args.mass_relative_max,
        },
        "initial_geometry": initial_geometry,
        "initial_probe_geometry": initial_probe_geometry,
        "continuous_geometry": continuous_geometry,
        "refined_geometry": refined_geometry,
        "gates": gates,
    }
    audit_path = args.out / "dynamic_qualification.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out / "final_terminal_output.txt").write_text(
        f"dynamic_profile_status={audit['status']}\n"
        f"restart_bytewise={str(restart_bytewise).lower()}\n"
        f"mass_relative={mass_relative:.17e}\n"
        "initial_profile_handoff_status="
        f"{'PASS' if gates['initial_normalized_shape'] and gates['initial_axis_ratio'] and gates['initial_centroid'] else 'FAIL'}\n"
        "initial_probe_normalized_axis_relative_max="
        f"{initial_normalized_axis_relative:.17e}\n"
        f"dt_phi_l1_normalized={dt_phi_l1:.17e}\n"
        f"dt_xB_mean_absolute={dt_xB_mean_absolute:.17e}\n"
        f"dt_xB_preferred_status={audit['dt_xB_preferred_status']}\n"
        f"audit_sha256={sha256(audit_path)}\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
