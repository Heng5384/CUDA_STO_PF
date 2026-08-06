#!/usr/bin/env python3
"""Model A--H comparison using the frozen no-dislocation PF transport core.

The direct precipitate reference is imported from
pf_full_psd_no_dislocation_transport_v1.py; it is not reimplemented here.
Only the descriptor selection and the frozen V2 A2/interface additions are
assembled in this driver.
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

import numpy as np
from numpy.polynomial.legendre import leggauss


PASS = "PASS_PF400_DISCRETE_PSD_TRUE_AREA_TRANSPORT_V1"
BOX_VOLUME_M3 = (400.0e-9) ** 3
MATRIX_MODES = ("fixed_6h_matrix", "pf_time_varying_matrix")
MODEL_TO_DESCRIPTOR = {
    "A_Nv_Rmean": "Nv_plus_mean_R_monodisperse",
    "B_Sv_sph": "Sv_geometric_limit",
    "C_M6": "M6_rayleigh_limit",
    "D_Sv_sph_plus_M6": "Sv_plus_M6_moment_reconstruction",
    "E_direct_discrete_PSD": "full_psd",
}
MODELS = (*MODEL_TO_DESCRIPTOR, "F_direct_PSD_plus_sphere_MI", "G_direct_PSD_plus_true_MI", "H_true_area_envelope")
GAUSS_ORDER = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("p1_no_dis_transport_core", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def integrate_from_rate(
    x: np.ndarray,
    weights: np.ndarray,
    x_max: float,
    rate: np.ndarray,
    temperature_K: float,
    config: dict[str, Any],
    transport: Any,
) -> float:
    constants = config["physical_constants"]
    shared = config["shared_parameters"]
    velocity = float(shared["average_sound_velocity_m_s"])
    prefactor = (
        float(constants["k_B_J_K"])
        / (2.0 * math.pi**2 * velocity)
        * (float(constants["k_B_J_K"]) * temperature_K / float(constants["hbar_J_s"])) ** 3
    )
    return float(0.5 * x_max * np.sum(weights * prefactor * transport.bose_weight(x) / rate))


def geometry_rows(root: Path) -> list[dict[str, str]]:
    paths = sorted(root.glob("case_*/*h/particle_geometry.csv"))
    if len(paths) != 28:
        raise ValueError(f"expected 28 geometry snapshots, found {len(paths)}")
    rows: list[dict[str, str]] = []
    for path in paths:
        values = read_csv(path)
        if not values:
            raise ValueError(f"empty geometry table: {path}")
        if any(not row.get("geometry_status", "").startswith("PASS") for row in values):
            raise ValueError(f"geometry gate failed: {path}")
        rows.extend(values)
    return rows


def calibration_members(path: Path) -> tuple[dict[str, tuple[float, float]], list[float]]:
    rows = read_csv(path)
    members: dict[str, tuple[float, float]] = {}
    temperatures: set[float] = set()
    for row in rows:
        member = row["background_case_id"]
        value = (float(row["background_A2_s"]), float(row["interface_alpha"]))
        if member in members and members[member] != value:
            raise ValueError(f"nonunique calibration member: {member}")
        members[member] = value
        temperatures.add(float(row["temperature_K"]))
    if len(members) != 18:
        raise ValueError(f"expected 18 calibration members, found {len(members)}")
    return members, sorted(temperatures)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-root", required=True, type=Path)
    parser.add_argument("--observables", required=True, type=Path)
    parser.add_argument("--transport-core", required=True, type=Path)
    parser.add_argument("--yu-config", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--interface-calibration", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)
    transport = load_module(args.transport_core.resolve())
    config = json.loads(args.yu_config.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    transport.validate_yu_base_config(config, args.yu_config)
    transport.validate_interface_contract(contract, args.contract, args.yu_config)
    members, temperatures = calibration_members(args.interface_calibration)
    observations = read_csv(args.observables)
    observation_map: dict[tuple[str, int], dict[str, str]] = {}
    for row in observations:
        hour = int(row["hour_h"])
        if hour in (6, 12, 24, 48):
            observation_map[(row["case_id"], hour)] = row
    if len(observation_map) != 28:
        raise ValueError(f"expected 28 primary observations, found {len(observation_map)}")
    grouped_geometry: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in geometry_rows(args.geometry_root):
        grouped_geometry[(row["case_id"], int(float(row["time_h"])))].append(row)
    if set(grouped_geometry) != set(observation_map):
        raise ValueError("geometry and observation snapshot keys differ")

    fixed_matrix = {
        case_id: float(observation_map[(case_id, 6)]["far_field_matrix_xAg"])
        for case_id, _ in observation_map
    }
    constants = config["physical_constants"]
    shared = config["shared_parameters"]
    velocity = float(shared["average_sound_velocity_m_s"])
    omega_debye = float(constants["k_B_J_K"]) * float(shared["debye_temperature_K"]) / float(constants["hbar_J_s"])
    nodes, weights = leggauss(GAUSS_ORDER)
    member_rows: list[dict[str, Any]] = []
    descriptor_rows: list[dict[str, Any]] = []

    for key in sorted(grouped_geometry, key=lambda item: (item[0], item[1])):
        case_id, hour = key
        objects = sorted(grouped_geometry[key], key=lambda row: int(row["object_id"]))
        observed = observation_map[key]
        radii_nm = np.asarray([float(row["equivalent_radius_nm"]) for row in objects], dtype=np.float64)
        radii_m = radii_nm * 1.0e-9
        sphere_areas = np.asarray([float(row["sphere_area_nm2"]) for row in objects])
        true_areas = {
            45: np.asarray([float(row["area_nm2_phi45"]) for row in objects]),
            50: np.asarray([float(row["area_nm2_phi50"]) for row in objects]),
            55: np.asarray([float(row["area_nm2_phi55"]) for row in objects]),
        }
        sv_sph = float(np.sum(sphere_areas) / 400.0**3 * 1.0e9)
        sv_true = {level: float(np.sum(area) / 400.0**3 * 1.0e9) for level, area in true_areas.items()}
        moments = transport.moments_from_radii(radii_m, BOX_VOLUME_M3)
        particle_count_match = len(objects) == int(observed["particle_count"])
        sv_closure_relative = abs(sv_sph - float(observed["Sv_nm_inv"]) * 1.0e9) / (float(observed["Sv_nm_inv"]) * 1.0e9)
        descriptor_rows.append(
            {
                "case_id": case_id,
                "family": observed["family"],
                "replicate": observed["replicate"],
                "time_h": hour,
                "handoff_sensitive": hour < 12,
                "particle_count": len(objects),
                "particle_count_matches_registered": particle_count_match,
                "Nv_m-3": moments["Nv_m-3"],
                "mean_radius_nm": float(np.mean(radii_nm)),
                "std_radius_nm": float(np.std(radii_nm)),
                "Sv_sph_m-1": sv_sph,
                "Sv_true_phi45_m-1": sv_true[45],
                "Sv_true_phi50_m-1": sv_true[50],
                "Sv_true_phi55_m-1": sv_true[55],
                "M6_m3": moments["M6_m3"],
                "matrix_xAg": float(observed["far_field_matrix_xAg"]),
                "registered_Sv_closure_relative": sv_closure_relative,
                "full_psd_sha256": transport.canonical_sha256(sorted(radii_nm.tolist())),
                "authority": "CONDITIONAL_COMPLETE_CHECKPOINT_EVIDENCE",
            }
        )
        if not particle_count_match or sv_closure_relative > 5.0e-6:
            raise ValueError(f"registered descriptor closure failed for {key}: count={particle_count_match}, Sv={sv_closure_relative}")
        for matrix_mode in MATRIX_MODES:
            matrix_xag = fixed_matrix[case_id] if matrix_mode == "fixed_6h_matrix" else float(observed["far_field_matrix_xAg"])
            for temperature in temperatures:
                x_max = float(shared["debye_temperature_K"]) / temperature
                x = 0.5 * (nodes + 1.0) * x_max
                omega = x * float(constants["k_B_J_K"]) * temperature / float(constants["hbar_J_s"])
                base_parts = transport.base_scattering_rates(omega, temperature, matrix_xag, config)
                base = base_parts["phonon_phonon"] + base_parts["boundary"] + base_parts["point_defect"]
                descriptor_rates = {
                    model: transport.descriptor_precipitate_rate(omega, radii_m, BOX_VOLUME_M3, config, descriptor)
                    for model, descriptor in MODEL_TO_DESCRIPTOR.items()
                }
                for member, (a2, alpha) in sorted(members.items()):
                    background = base + a2 * omega**2
                    kappas: dict[str, tuple[float, float, float, float]] = {}
                    for model, precipitate in descriptor_rates.items():
                        value = integrate_from_rate(x, weights, x_max, background + precipitate, temperature, config, transport)
                        kappas[model] = (value, value, value, 0.0)
                    direct_rate = descriptor_rates["E_direct_discrete_PSD"]
                    sphere_interface = alpha * (2.0 / 3.0) * velocity * sv_sph * omega / omega_debye
                    true_interface = {
                        level: alpha * (2.0 / 3.0) * velocity * area * omega / omega_debye
                        for level, area in sv_true.items()
                    }
                    f_value = integrate_from_rate(x, weights, x_max, background + direct_rate + sphere_interface, temperature, config, transport)
                    g_value = integrate_from_rate(x, weights, x_max, background + direct_rate + true_interface[50], temperature, config, transport)
                    envelope_values = [integrate_from_rate(x, weights, x_max, background + direct_rate + true_interface[level], temperature, config, transport) for level in (45, 50, 55)]
                    kappas["F_direct_PSD_plus_sphere_MI"] = (f_value, f_value, f_value, 0.0)
                    kappas["G_direct_PSD_plus_true_MI"] = (g_value, g_value, g_value, 0.0)
                    kappas["H_true_area_envelope"] = (
                        g_value,
                        min(envelope_values),
                        max(envelope_values),
                        max(envelope_values) - min(envelope_values),
                    )
                    for model in MODELS:
                        value, lower, upper, width = kappas[model]
                        member_rows.append(
                            {
                                "case_id": case_id,
                                "family": observed["family"],
                                "replicate": observed["replicate"],
                                "time_h": hour,
                                "handoff_sensitive": hour < 12,
                                "matrix_mode": matrix_mode,
                                "matrix_xAg_used": matrix_xag,
                                "temperature_K": temperature,
                                "calibration_member": member,
                                "background_A2_s": a2,
                                "interface_alpha_frozen_6h": alpha,
                                "model": model,
                                "kappa_conditional_lattice_no_dis_W_mK": value,
                                "kappa_levelset_min_W_mK": lower,
                                "kappa_levelset_max_W_mK": upper,
                                "kappa_levelset_width_W_mK": width,
                                "A_N": 1.5,
                                "S11_rate": 0.0,
                                "S13_rate": 0.0,
                                "yu_scale_0p1172768_used": False,
                            }
                        )

    group_values: dict[tuple[str, int, str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in member_rows:
        group_values[(row["case_id"], int(row["time_h"]), row["matrix_mode"], float(row["temperature_K"]), row["model"])].append(row)
    summary_rows: list[dict[str, Any]] = []
    reference: dict[tuple[str, str, float, str], float] = {}
    descriptor_map = {(row["case_id"], int(row["time_h"])): row for row in descriptor_rows}
    for key in sorted(group_values):
        case_id, hour, matrix_mode, temperature, model = key
        values = group_values[key]
        kappas = np.asarray([float(row["kappa_conditional_lattice_no_dis_W_mK"]) for row in values])
        lower = min(float(row["kappa_levelset_min_W_mK"]) for row in values)
        upper = max(float(row["kappa_levelset_max_W_mK"]) for row in values)
        descriptor = descriptor_map[(case_id, hour)]
        ref_key = (case_id, matrix_mode, temperature, model)
        if hour == 6:
            reference[ref_key] = float(np.mean(kappas))
        summary_rows.append(
            {
                **descriptor,
                "matrix_mode": matrix_mode,
                "temperature_K": temperature,
                "model": model,
                "calibration_member_count": len(values),
                "kappa_member_mean_W_mK": float(np.mean(kappas)),
                "kappa_member_min_W_mK": float(np.min(kappas)),
                "kappa_member_max_W_mK": float(np.max(kappas)),
                "kappa_levelset_and_member_min_W_mK": lower,
                "kappa_levelset_and_member_max_W_mK": upper,
                "delta_kappa_from_6h_member_mean_W_mK": float(np.mean(kappas)) - reference[ref_key],
                "relative_change_from_6h_percent": 100.0 * (float(np.mean(kappas)) - reference[ref_key]) / reference[ref_key],
            }
        )

    direct_lookup = {
        (row["case_id"], row["time_h"], row["matrix_mode"], row["temperature_K"]): float(row["kappa_member_mean_W_mK"])
        for row in summary_rows
        if row["model"] == "E_direct_discrete_PSD"
    }
    compression_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        if row["model"] not in MODEL_TO_DESCRIPTOR or row["model"] == "E_direct_discrete_PSD":
            continue
        key = (row["case_id"], row["time_h"], row["matrix_mode"], row["temperature_K"])
        reference_value = direct_lookup[key]
        error = float(row["kappa_member_mean_W_mK"]) - reference_value
        compression_rows.append(
            {
                "case_id": row["case_id"],
                "family": row["family"],
                "replicate": row["replicate"],
                "time_h": row["time_h"],
                "matrix_mode": row["matrix_mode"],
                "temperature_K": row["temperature_K"],
                "compressed_model": row["model"],
                "reference_model": "E_direct_discrete_PSD",
                "kappa_compressed_W_mK": row["kappa_member_mean_W_mK"],
                "kappa_direct_W_mK": reference_value,
                "signed_error_W_mK": error,
                "absolute_error_W_mK": abs(error),
                "relative_error_percent": 100.0 * abs(error) / abs(reference_value),
            }
        )

    # Operator qualification uses the imported direct-sum code path.
    sample = sorted(grouped_geometry)[0]
    sample_radii = np.asarray([float(row["equivalent_radius_nm"]) for row in grouped_geometry[sample]]) * 1.0e-9
    omega_test = np.geomspace(1.0e8, omega_debye, 257)
    vector = transport.full_psd_precipitate_rate(omega_test, sample_radii, BOX_VOLUME_M3, config)
    scalar = transport.direct_scalar_sum_precipitate_rate(omega_test, sample_radii, BOX_VOLUME_M3, config)
    scalar_vector_relative = float(np.max(np.abs(vector - scalar) / np.maximum(np.abs(scalar), 1.0e-300)))
    duplicated = transport.full_psd_precipitate_rate(omega_test, np.tile(sample_radii, 2), 2.0 * BOX_VOLUME_M3, config)
    permuted = transport.full_psd_precipitate_rate(omega_test, sample_radii[::-1], BOX_VOLUME_M3, config)
    mono = np.full(100, 10.0e-9)
    mono_full = transport.full_psd_precipitate_rate(omega_test, mono, BOX_VOLUME_M3, config)
    mono_a = transport.descriptor_precipitate_rate(omega_test, mono, BOX_VOLUME_M3, config, "Nv_plus_mean_R_monodisperse")
    zero = transport.full_psd_precipitate_rate(omega_test, np.asarray([], dtype=float), BOX_VOLUME_M3, config)
    quadrature = {
        order: transport.integrate_kappa_gauss(303.15, 0.0062, sample_radii, BOX_VOLUME_M3, config, "full_psd", order=order)
        for order in (256, 512, 1024)
    }
    adaptive_value, adaptive_error = transport.integrate_kappa_adaptive(
        303.15, 0.0062, sample_radii, BOX_VOLUME_M3, config, "full_psd"
    )
    bin_refinement = []
    for width_nm in (2.0, 1.0, 0.5, 0.25, 0.125):
        binned = transport.binned_center_radii(sample_radii, width_nm * 1.0e-9)
        rate = transport.full_psd_precipitate_rate(omega_test, binned, BOX_VOLUME_M3, config)
        bin_refinement.append(
            {
                "bin_width_nm": width_nm,
                "max_relative_rate_error": float(np.max(np.abs(rate - vector) / np.maximum(np.abs(vector), 1.0e-300))),
            }
        )
    sample_moments = transport.moments_from_radii(sample_radii, BOX_VOLUME_M3)
    reconstructed_radius = (4.0 * math.pi * sample_moments["M6_m3"] / sample_moments["Sv_m-1"]) ** 0.25
    reconstructed_nv = sample_moments["Sv_m-1"] / (4.0 * math.pi * reconstructed_radius**2)
    reconstructed_sv = reconstructed_nv * 4.0 * math.pi * reconstructed_radius**2
    reconstructed_m6 = reconstructed_nv * reconstructed_radius**6
    six_h_matrix_closure = 0.0
    summary_lookup = {
        (row["case_id"], int(row["time_h"]), row["matrix_mode"], float(row["temperature_K"]), row["model"]): float(row["kappa_member_mean_W_mK"])
        for row in summary_rows
    }
    for case_id in sorted({row["case_id"] for row in summary_rows}):
        for temperature in temperatures:
            for model in MODELS:
                fixed = summary_lookup[(case_id, 6, "fixed_6h_matrix", temperature, model)]
                varying = summary_lookup[(case_id, 6, "pf_time_varying_matrix", temperature, model)]
                six_h_matrix_closure = max(six_h_matrix_closure, abs(fixed - varying))
    qualification = {
        "status": PASS,
        "direct_reference": "imported full_psd_precipitate_rate",
        "scalar_vector_max_relative_error": scalar_vector_relative,
        "scalar_vector_pass_1e-12": scalar_vector_relative <= 1.0e-12,
        "duplicate_population_double_volume_max_relative": float(np.max(np.abs(vector - duplicated) / np.maximum(np.abs(vector), 1.0e-300))),
        "permutation_max_relative": float(np.max(np.abs(vector - permuted) / np.maximum(np.abs(vector), 1.0e-300))),
        "monodisperse_A_max_relative_error": float(np.max(np.abs(mono_full - mono_a) / np.maximum(np.abs(mono_full), 1.0e-300))),
        "zero_population_max_abs": float(np.max(np.abs(zero))),
        "quadrature_gauss_W_mK": quadrature,
        "quadrature_adaptive_W_mK": adaptive_value,
        "quadrature_adaptive_reported_error": adaptive_error,
        "quadrature_512_vs_1024_relative": abs(quadrature[512] - quadrature[1024]) / abs(quadrature[1024]),
        "quadrature_512_vs_adaptive_relative": abs(quadrature[512] - adaptive_value) / abs(adaptive_value),
        "bin_refinement": bin_refinement,
        "bin_refinement_pass": bin_refinement[-1]["max_relative_rate_error"] < bin_refinement[0]["max_relative_rate_error"],
        "Sv_M6_moment_reconstruction_Sv_relative": abs(reconstructed_sv - sample_moments["Sv_m-1"]) / sample_moments["Sv_m-1"],
        "Sv_M6_moment_reconstruction_M6_relative": abs(reconstructed_m6 - sample_moments["M6_m3"]) / sample_moments["M6_m3"],
        "fixed_vs_time_varying_matrix_6h_max_abs_W_mK": six_h_matrix_closure,
        "A_N": 1.5,
        "S11_rate": 0.0,
        "S13_rate": 0.0,
        "yu_scale_0p1172768_used": False,
        "absolute_experimental_kappa_claim": False,
    }
    if not all((qualification["scalar_vector_pass_1e-12"], qualification["duplicate_population_double_volume_max_relative"] <= 1.0e-14, qualification["permutation_max_relative"] <= 1.0e-14, qualification["monodisperse_A_max_relative_error"] <= 1.0e-14, qualification["zero_population_max_abs"] == 0.0, qualification["quadrature_512_vs_1024_relative"] <= 1.0e-10, qualification["quadrature_512_vs_adaptive_relative"] <= 1.0e-9, qualification["bin_refinement_pass"], qualification["Sv_M6_moment_reconstruction_Sv_relative"] <= 1.0e-14, qualification["Sv_M6_moment_reconstruction_M6_relative"] <= 1.0e-14, six_h_matrix_closure == 0.0)):
        qualification["status"] = "FAIL_PF400_DISCRETE_PSD_TRUE_AREA_TRANSPORT_V1"
        raise ValueError(f"transport operator qualification failed: {qualification}")

    write_csv(args.out / "transport_member_predictions.csv", member_rows)
    write_csv(args.out / "transport_model_comparison.csv", summary_rows)
    write_csv(args.out / "snapshot_descriptors.csv", descriptor_rows)
    write_csv(args.out / "descriptor_compression_error.csv", compression_rows)
    (args.out / "transport_qualification.json").write_text(json.dumps(qualification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance = {
        "schema": "PF400_DISCRETE_PSD_TRUE_AREA_TRANSPORT_PROVENANCE_V1",
        "status": qualification["status"],
        "models": list(MODELS),
        "matrix_modes": list(MATRIX_MODES),
        "temperature_grid_K": temperatures,
        "calibration_member_count": len(members),
        "input_sha256": {str(path.resolve()): sha256(path.resolve()) for path in (args.observables, args.transport_core, args.yu_config, args.contract, args.interface_calibration)},
        "driver_sha256": sha256(Path(__file__).resolve()),
        "conductivity_semantics": "conditional_lattice_no_dislocation; not experimental total kappa",
    }
    (args.out / "transport_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "status.txt").write_text(qualification["status"] + "\n", encoding="utf-8")
    print(qualification["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
