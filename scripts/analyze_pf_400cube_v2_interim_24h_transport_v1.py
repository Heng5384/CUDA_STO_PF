#!/usr/bin/env python3
"""Apply the immutable transport-V2 equations to a new 400^3 6--24 h input.

V2 itself is not edited or re-fitted.  Its 18 AQ-background members and the
corresponding interface alphas frozen from the historical 6 h calibration are
propagated unchanged.  This program produces a diagnostic on a new PF input,
not a new V2 freeze and not an absolute experimental-kappa reproduction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss


PASS = "PASS_400CUBE_V2_INTERIM_24H_EQUATION_REPLAY_V1"
V2_MANIFEST_SHA256 = "1c815a6277d46ec4c5537966de402f0e190545b6433761291bf46c08ce725856"
GRID_SHAPE = (400, 400, 400)
DX_NM = 1.0
BOX_VOLUME_M3 = math.prod(GRID_SHAPE) * 1.0e-27
AGES = (6, 24)
STEPS = {6: 0, 24: 65393}
MODELS = ("M0", "MI", "MS", "MIS")
REFERENCE_BACKGROUND = "central__all_small_at_reported_max"
GAUSS_ORDER = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"empty output: {path}")
    names = list(fields) if fields is not None else list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_raw(path: Path, dtype: str) -> np.ndarray:
    values = np.fromfile(path, dtype=dtype)
    if values.size != math.prod(GRID_SHAPE):
        raise ValueError(f"{path}: expected {math.prod(GRID_SHAPE)} values, found {values.size}")
    return values.reshape(GRID_SHAPE)


def periodic_marching_cubes_area_nm2(phi: np.ndarray) -> tuple[float, int, int]:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

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
    contour.SetValue(0, 0.5)
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


def epsilon_h_spectrum(field_root: Path, age_h: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
    components = [load_raw(field_root / f"strain_{name}.raw.f32", "<f4").astype(np.float64) for name in ("xx", "yy", "zz")]
    epsilon_h = (components[0] + components[1] + components[2]) / 3.0
    del components
    fluctuation = epsilon_h - float(np.mean(epsilon_h))
    real_variance = float(np.var(fluctuation))
    nvox = math.prod(GRID_SHAPE)
    transform = np.fft.rfftn(fluctuation) / nvox
    del fluctuation, epsilon_h
    power = np.abs(transform) ** 2
    del transform
    qx = (2.0 * np.pi * np.fft.fftfreq(GRID_SHAPE[0], d=DX_NM)).astype(np.float32)
    qy = (2.0 * np.pi * np.fft.fftfreq(GRID_SHAPE[1], d=DX_NM)).astype(np.float32)
    qz = (2.0 * np.pi * np.fft.rfftfreq(GRID_SHAPE[2], d=DX_NM)).astype(np.float32)
    qmag = np.sqrt(qx[:, None, None] ** 2 + qy[None, :, None] ** 2 + qz[None, None, :] ** 2)
    q_fundamental = 2.0 * np.pi / (GRID_SHAPE[0] * DX_NM)
    q_max = np.pi / DX_NM
    q_edges = np.arange(0.0, q_max + 1.000001 * q_fundamental, q_fundamental)
    weights_z = np.full(qz.shape, 2.0)
    weights_z[0] = 1.0
    weights_z[-1] = 1.0
    weighted_power = power * weights_z[None, None, :]
    select = (qmag > 0.0) & (qmag <= q_max)
    shell_index = np.searchsorted(q_edges, qmag[select], side="right") - 1
    mode_weights = np.broadcast_to(weights_z, qmag.shape)[select]
    shell_count = np.bincount(shell_index, weights=mode_weights, minlength=q_edges.size - 1)
    shell_power = np.bincount(shell_index, weights=weighted_power[select], minlength=q_edges.size - 1)
    box_volume_m3 = BOX_VOLUME_M3
    rows = []
    for index in range(q_edges.size - 1):
        epsilon_spectral_density = box_volume_m3 * shell_power[index] / max(shell_count[index], 1.0)
        rows.append(
            {
                "replicate": "P400",
                "age_h": age_h,
                "field": "epsilon_h",
                "direction": "isotropic_radial",
                "q_low_nm_inv": q_edges[index],
                "q_high_nm_inv": q_edges[index + 1],
                "weighted_mode_count": shell_count[index],
                "integrated_variance": shell_power[index],
                "correlation_spectral_density_m3": epsilon_spectral_density,
                "trace_strain_spectral_density_m3": 9.0 * epsilon_spectral_density,
            }
        )
    parseval = float(weighted_power.sum())
    return rows, {
        "age_h": age_h,
        "epsilon_h_variance": real_variance,
        "parseval_variance": parseval,
        "parseval_relative_error": abs(parseval - real_variance) / max(real_variance, np.finfo(float).tiny),
    }


def strain_rate(omega: np.ndarray, spectrum: list[dict[str, Any]], velocity: float, gamma: float) -> np.ndarray:
    q_low = np.asarray([float(row["q_low_nm_inv"]) for row in spectrum]) * 1.0e9
    q_high = np.asarray([float(row["q_high_nm_inv"]) for row in spectrum]) * 1.0e9
    density = np.asarray([float(row["trace_strain_spectral_density_m3"]) for row in spectrum])
    q_cut = np.minimum(2.0 * omega / velocity, math.pi / 1.0e-9)
    upper = np.minimum(q_high[None, :], q_cut[:, None])
    active = upper > q_low[None, :]
    integral = np.sum(density[None, :] * np.where(active, (upper**4 - q_low[None, :] ** 4) / 4.0, 0.0), axis=1)
    return gamma**2 * velocity * integral / (4.0 * math.pi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", required=True, type=Path)
    parser.add_argument("--lineage-root", required=True, type=Path)
    parser.add_argument("--merge-audit", required=True, type=Path)
    parser.add_argument("--v2-manifest", required=True, type=Path)
    parser.add_argument("--v1-manifest", required=True, type=Path)
    parser.add_argument("--transport-module", required=True, type=Path)
    parser.add_argument("--yu-config", required=True, type=Path)
    parser.add_argument("--transport-contract", required=True, type=Path)
    parser.add_argument("--v2-calibration", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)
    try:
        if sha256(args.v2_manifest) != V2_MANIFEST_SHA256:
            raise ValueError("immutable V2 manifest hash mismatch")
        v2_manifest = json.loads(args.v2_manifest.read_text(encoding="utf-8"))
        v1_manifest = json.loads(args.v1_manifest.read_text(encoding="utf-8"))
        expected_calibration = v2_manifest["all_output_sha256"]["reports/sheskin_pf_interface_strain_blind_prediction_v1/interface_model_6h_calibration.csv"]
        if sha256(args.v2_calibration) != expected_calibration:
            raise ValueError("V2 calibration hash mismatch")
        expected_v1 = {
            args.transport_module: v1_manifest["all_script_sha256"]["scripts/pf_full_psd_no_dislocation_transport_v1.py"],
            args.yu_config: v1_manifest["all_parameter_sha256"]["data/qualification/yu2024_transport_v1/yu_48h_parameters.json"],
            args.transport_contract: v1_manifest["all_parameter_sha256"]["data/qualification/pf_full_psd_no_dislocation_transport_v1/transport_parameter_contract.json"],
        }
        for path, expected in expected_v1.items():
            if sha256(path) != expected:
                raise ValueError(f"V1 dependency hash mismatch: {path}")
        if (args.replay_root / "status.txt").read_text().strip() != "PASS_400CUBE_V2_INTERIM_24H_ACCEPTED_FIELD_REPLAY_V1":
            raise ValueError("accepted-field replay lacks exact PASS")
        merge = json.loads(args.merge_audit.read_text(encoding="utf-8"))
        if not str(merge.get("status", "")).startswith("PASS_") or not all(merge.get("gates", {}).values()):
            raise ValueError("hourly merge-aware audit lacks complete PASS gates")

        transport = load_module("frozen_v1_transport", args.transport_module)
        config = json.loads(args.yu_config.read_text(encoding="utf-8"))
        contract = json.loads(args.transport_contract.read_text(encoding="utf-8"))
        transport.validate_yu_base_config(config, args.yu_config)
        transport.validate_interface_contract(contract, args.transport_contract, args.yu_config)
        if any(contract["disabled_inputs"].get(key) is not True for key in ("S11_dislocation_core", "S13_dislocation_strain", "yu_refit_scale_0p1172768")):
            raise ValueError("no-dislocation contract is not frozen")

        lineage = read_csv(args.lineage_root / "particle_lineage.csv")
        observables = read_csv(args.lineage_root / "ensemble_observables.csv")
        populations: dict[int, dict[str, Any]] = {}
        for age_h in AGES:
            step = STEPS[age_h]
            radii = np.asarray(sorted(float(row["equivalent_radius_nm"]) for row in lineage if int(row["step"]) == step), dtype=float)
            observed = [row for row in observables if int(row["step"]) == step]
            if len(observed) != 1 or radii.size != int(observed[0]["particle_count"]):
                raise ValueError(f"PSD/observable mismatch at {age_h} h")
            populations[age_h] = {"radii_nm": radii, "matrix_xAg": float(observed[0]["far_field_matrix_xAg"])}

        interface_rows: list[dict[str, Any]] = []
        spectrum_rows: list[dict[str, Any]] = []
        strain_stats: list[dict[str, Any]] = []
        spectra: dict[int, list[dict[str, Any]]] = {}
        for age_h in AGES:
            fields = args.replay_root / f"age_{age_h}h/fields"
            phi_path = fields / "accepted_phi.raw.f64"
            phi = load_raw(phi_path, "<f8")
            area_nm2, vertices, triangles = periodic_marching_cubes_area_nm2(phi)
            del phi
            interface_rows.append({"age_h": age_h, "area_nm2": area_nm2, "Sv_m_inv": area_nm2 * 1.0e-18 / BOX_VOLUME_M3, "vertex_count": vertices, "triangle_count": triangles, "accepted_phi_sha256": sha256(phi_path)})
            case_spectrum, case_stats = epsilon_h_spectrum(fields, age_h)
            spectra[age_h] = case_spectrum
            spectrum_rows.extend(case_spectrum)
            strain_stats.append(case_stats)

        calibration_rows = read_csv(args.v2_calibration)
        calibrations: dict[str, tuple[float, float]] = {}
        temperatures = sorted({float(row["temperature_K"]) for row in calibration_rows})
        for row in calibration_rows:
            value = (float(row["background_A2_s"]), float(row["interface_alpha"]))
            previous = calibrations.setdefault(row["background_case_id"], value)
            if previous != value:
                raise ValueError("nonunique frozen V2 calibration member")
        if len(calibrations) != 18:
            raise ValueError("expected exactly 18 frozen V2 members")

        constants = config["physical_constants"]
        shared = config["shared_parameters"]
        velocity = float(shared["average_sound_velocity_m_s"])
        gamma = float(shared["gruneisen_gamma"])
        omega_d = float(constants["k_B_J_K"]) * float(shared["debye_temperature_K"]) / float(constants["hbar_J_s"])
        nodes, weights = leggauss(GAUSS_ORDER)
        sv = {int(row["age_h"]): float(row["Sv_m_inv"]) for row in interface_rows}
        prediction_rows: list[dict[str, Any]] = []
        for case_id, (a2, alpha) in sorted(calibrations.items()):
            for age_h in AGES:
                snapshot = populations[age_h]
                for temperature in temperatures:
                    xmax = float(shared["debye_temperature_K"]) / temperature
                    x = 0.5 * (nodes + 1.0) * xmax
                    omega = x * float(constants["k_B_J_K"]) * temperature / float(constants["hbar_J_s"])
                    base = transport.base_scattering_rates(omega, temperature, snapshot["matrix_xAg"], config)
                    density = transport.full_psd_precipitate_rate(omega, snapshot["radii_nm"] * 1.0e-9, BOX_VOLUME_M3, config)
                    background = a2 * omega**2
                    interface = alpha * (2.0 / 3.0) * velocity * sv[age_h] * omega / omega_d
                    strain = strain_rate(omega, spectra[age_h], velocity, gamma)
                    prefactor = float(constants["k_B_J_K"]) / (2.0 * math.pi**2 * velocity) * (float(constants["k_B_J_K"]) * temperature / float(constants["hbar_J_s"])) ** 3
                    m0_rate = base["phonon_phonon"] + base["boundary"] + base["point_defect"] + density + background
                    rates = {"M0": m0_rate, "MI": m0_rate + interface, "MS": m0_rate + strain, "MIS": m0_rate + interface + strain}
                    for model, rate in rates.items():
                        kappa = float(0.5 * xmax * np.sum(weights * prefactor * transport.bose_weight(x) / rate))
                        prediction_rows.append({"background_case_id": case_id, "background_A2_s": a2, "interface_alpha_frozen_from_historical_6h": alpha, "age_h": age_h, "temperature_K": temperature, "model": model, "kappa_W_mK": kappa, "matrix_xAg": snapshot["matrix_xAg"], "particle_count": snapshot["radii_nm"].size})

        grouped: dict[tuple[float, str, int], list[float]] = defaultdict(list)
        lookup: dict[tuple[str, float, str, int], float] = {}
        for row in prediction_rows:
            key = (float(row["temperature_K"]), str(row["model"]), int(row["age_h"]))
            grouped[key].append(float(row["kappa_W_mK"]))
            lookup[(str(row["background_case_id"]), float(row["temperature_K"]), str(row["model"]), int(row["age_h"]))] = float(row["kappa_W_mK"])
        summary_rows: list[dict[str, Any]] = []
        for temperature in temperatures:
            for model in MODELS:
                values6 = np.asarray(grouped[(temperature, model, 6)])
                values24 = np.asarray(grouped[(temperature, model, 24)])
                deltas = values24 - values6
                central6 = lookup[(REFERENCE_BACKGROUND, temperature, model, 6)]
                central24 = lookup[(REFERENCE_BACKGROUND, temperature, model, 24)]
                summary_rows.append({"temperature_K": temperature, "model": model, "kappa_6h_member_mean_W_mK": float(values6.mean()), "kappa_24h_member_mean_W_mK": float(values24.mean()), "delta_24h_minus_6h_member_mean_W_mK": float(deltas.mean()), "relative_change_member_mean_percent": 100.0 * float(deltas.mean()) / float(values6.mean()), "delta_member_min_W_mK": float(deltas.min()), "delta_member_max_W_mK": float(deltas.max()), "central_kappa_6h_W_mK": central6, "central_kappa_24h_W_mK": central24, "central_delta_W_mK": central24 - central6})

        descriptor_rows = []
        for age_h in AGES:
            radii = populations[age_h]["radii_nm"]
            descriptor_rows.append({"age_h": age_h, "particle_count": radii.size, "Nv_m-3": radii.size / BOX_VOLUME_M3, "mean_radius_nm": float(radii.mean()), "std_radius_nm": float(radii.std()), "Sv_true_m-1": sv[age_h], "matrix_xAg": populations[age_h]["matrix_xAg"], "epsilon_h_variance": next(float(row["epsilon_h_variance"]) for row in strain_stats if int(row["age_h"]) == age_h), "full_psd_sha256": hashlib.sha256(np.ascontiguousarray(radii, dtype="<f8").tobytes()).hexdigest()})

        write_csv(args.out / "interface_statistics.csv", interface_rows)
        write_csv(args.out / "strain_power_spectrum.csv", spectrum_rows)
        write_csv(args.out / "strain_parseval_audit.csv", strain_stats)
        write_csv(args.out / "pf_descriptors_6h_24h.csv", descriptor_rows)
        write_csv(args.out / "v2_member_predictions.csv", prediction_rows)
        write_csv(args.out / "v2_prediction_summary.csv", summary_rows)

        for model in MODELS:
            selected = [row for row in summary_rows if row["model"] == model]
            plt.plot([row["temperature_K"] for row in selected], [row["delta_24h_minus_6h_member_mean_W_mK"] for row in selected], marker="o", label=model)
        plt.axhline(0.0, color="black", lw=0.8)
        plt.xlabel("Temperature (K)")
        plt.ylabel("κ(24 h) − κ(6 h) (W m⁻¹ K⁻¹)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out / "v2_delta_kappa_6h_to_24h.png", dpi=180)
        plt.close()

        focus = [row for row in summary_rows if abs(float(row["temperature_K"]) - 573.15) < 1.0e-9]
        focus_map = {row["model"]: row for row in focus}
        max_parseval = max(float(row["parseval_relative_error"]) for row in strain_stats)
        manifest = {
            "schema": "PF_400CUBE_V2_INTERIM_24H_EQUATION_REPLAY_V1",
            "status": PASS,
            "scientific_identity": "immutable V2 equations and frozen coefficients applied to new 400-cube PF input",
            "immutable_v2_baseline_modified": False,
            "absolute_experimental_kappa_claim": False,
            "A_N": 1.5,
            "S11_rate": 0.0,
            "S13_rate": 0.0,
            "yu_refit_scale_0p1172768_used": False,
            "background_member_count": 18,
            "maximum_parseval_relative_error": max_parseval,
            "input_sha256": {str(path): sha256(path) for path in (args.v2_manifest, args.v1_manifest, args.transport_module, args.yu_config, args.transport_contract, args.v2_calibration, args.merge_audit)},
        }
        (args.out / "audit.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (args.out / "status.txt").write_text(PASS + "\n", encoding="utf-8")
        report = f"""# 400 nm PF transport-V2 interim audit through 24 h

Status: `{PASS}`.

This is a new-input diagnostic using the immutable V2 equations and all 18 frozen AQ/interface-calibration members. It does not modify V2, re-fit 24 h data, infer dislocation density, use Yu's 0.1172768 scale, or claim absolute experimental conductivity reproduction.

At 573.15 K, the member-mean predictions are:

| model | κ(6 h) | κ(24 h) | Δκ(24−6 h) | relative change |
|---|---:|---:|---:|---:|
"""
        for model in MODELS:
            row = focus_map[model]
            report += f"| {model} | {row['kappa_6h_member_mean_W_mK']:.6f} | {row['kappa_24h_member_mean_W_mK']:.6f} | {row['delta_24h_minus_6h_member_mean_W_mK']:+.6f} | {row['relative_change_member_mean_percent']:+.3f}% |\n"
        report += "\n`M0` is AQ background + no-dislocation Yu host/full-PSD scattering; `MI` adds V2 interface scattering; `MS` adds the parameter-free V2 scalar Born strain spectrum; `MIS` adds both.\n"
        (args.out / "report.md").write_text(report, encoding="utf-8")
        outputs = sorted(path for path in args.out.iterdir() if path.is_file())
        with (args.out / "outputs.sha256").open("w", encoding="utf-8") as stream:
            for path in outputs:
                stream.write(f"{sha256(path)}  {path.name}\n")
        print(PASS)
    except Exception as exc:
        (args.out / "status.txt").write_text("BLOCKED_400CUBE_V2_INTERIM_24H_EQUATION_REPLAY_V1\n", encoding="utf-8")
        (args.out / "first_failure.txt").write_text(str(exc) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
