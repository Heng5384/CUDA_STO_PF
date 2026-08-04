#!/usr/bin/env python3
"""PF full-PSD lattice transport with the dislocation channel frozen off.

This module extends Yu et al. 2024 SI Eqs. S7--S10 from one average radius
to a resolved particle population,

    tau_Pre^-1 = v / V_box * sum_i sigma_eff(R_i, omega).

It deliberately freezes A_N=1.5, never reads the Yu refit contract, and sets
both S11 and S13 rates identically to zero.  The output is a conditional,
resolved-precipitate transport trajectory; it is not an absolute experimental
thermal-conductivity reproduction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad


SCHEMA = "PF_FULL_PSD_NO_DISLOCATION_TRANSPORT_V1"
FINAL_STATUS = "PASS_PF_FULL_PSD_NO_DISLOCATION_TRANSPORT_INTERFACE_V1"
DISLOCATION_MODE = "DISLOCATION_OFF"
FROZEN_A_N = 1.5
DEFAULT_TEMPERATURES_K = (300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0)
DEFAULT_INTERFACE_CONTRACT = Path(
    "data/qualification/pf_full_psd_no_dislocation_transport_v1/"
    "transport_parameter_contract.json"
)
DESCRIPTOR_MODELS = (
    "full_psd",
    "Nv_plus_mean_R_monodisperse",
    "Sv_geometric_limit",
    "M6_rayleigh_limit",
    "Sv_plus_M6_moment_reconstruction",
)
MATRIX_MODES = ("fixed_6h_matrix", "pf_time_varying_matrix")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_yu_base_config(config: dict[str, Any], config_path: Path | None = None) -> None:
    if config_path is not None and "refit" in str(config_path).lower():
        raise ValueError("Yu refit inputs are forbidden in the no-dislocation V1 interface")
    if config.get("source_doi") != "10.1002/aenm.202304442":
        raise ValueError("Unexpected transport source DOI")
    shared = config["shared_parameters"]
    if not math.isclose(float(shared["A_N"]), FROZEN_A_N, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(f"A_N must be frozen exactly at {FROZEN_A_N}")
    contract = config["source_contract"]
    if contract.get("parameter_refit_authorized", False):
        raise ValueError("A refitted Yu parameter contract is forbidden")
    if not contract.get("single_relaxation_time_integral", False):
        raise ValueError("Yu main-text Eq. 3 single-rate integral is required")
    if contract.get("callaway_second_term_present", False):
        raise ValueError("The source does not contain a Callaway second term")


def validate_interface_contract(
    contract: dict[str, Any], contract_path: Path, yu_config_path: Path
) -> None:
    if contract.get("schema") != "PF_FULL_PSD_NO_DISLOCATION_TRANSPORT_PARAMETER_CONTRACT_V1":
        raise ValueError("Unexpected PF transport parameter contract schema")
    if contract.get("status") != "FROZEN":
        raise ValueError("PF transport parameter contract is not frozen")
    if float(contract["frozen_parameters"]["A_N"]) != FROZEN_A_N:
        raise ValueError("PF transport contract does not freeze A_N=1.5")
    disabled = contract["disabled_inputs"]
    required_disabled = (
        "S11_dislocation_core",
        "S13_dislocation_strain",
        "yu_refit_scale_0p1172768",
        "yu_small_precipitate_population",
        "yu_big_precipitate_population",
    )
    if not all(disabled.get(name) is True for name in required_disabled):
        raise ValueError("PF transport contract has a forbidden scattering input enabled")
    if contract["yu_public_parameter_config_sha256"] != file_sha256(yu_config_path):
        raise ValueError("Yu public parameter file does not match the frozen interface contract")
    if not contract_path.is_file():
        raise ValueError("PF transport interface contract is missing")


def material_derived(config: dict[str, Any], matrix_xag: float) -> dict[str, float]:
    if not math.isfinite(matrix_xag) or not 0.0 <= matrix_xag < 1.0:
        raise ValueError(f"Invalid matrix xAg: {matrix_xag}")
    shared = config["shared_parameters"]
    state = config["state_parameters"]
    lattice_m = float(state["solid_solution_lattice_constant_angstrom"]) * 1.0e-10
    atomic_volume = lattice_m**3 / 8.0
    mass_ratio = (
        float(shared["published_delta_M_i_g_mol"])
        / float(shared["matrix_atom_mass_g_mol"])
    )
    radius_ratio = (
        float(shared["impurity_atomic_radius_pm"])
        - float(shared["matrix_atomic_radius_pm"])
    ) / float(shared["matrix_atomic_radius_pm"])
    point_gamma = matrix_xag * (
        mass_ratio**2 + float(shared["point_defect_epsilon"]) * radius_ratio**2
    )
    return {
        "atomic_volume_m3": atomic_volume,
        "point_defect_Gamma": point_gamma,
        "density_contrast_ratio": (
            float(shared["density_difference_kg_m3"])
            / float(shared["matrix_density_kg_m3"])
        ),
    }


def base_scattering_rates(
    omega_rad_s: np.ndarray | float,
    temperature_K: float,
    matrix_xag: float,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    omega = np.asarray(omega_rad_s, dtype=float)
    constants = config["physical_constants"]
    shared = config["shared_parameters"]
    state = config["state_parameters"]
    derived = material_derived(config, matrix_xag)
    v = float(shared["average_sound_velocity_m_s"])
    phonon_phonon = (
        FROZEN_A_N
        * 2.0
        / (6.0 * math.pi**2) ** (1.0 / 3.0)
        * (
            float(constants["k_B_J_K"])
            * derived["atomic_volume_m3"] ** (1.0 / 3.0)
            * float(shared["gruneisen_gamma"]) ** 2
            * omega**2
            * temperature_K
            / (float(shared["average_atomic_mass_kg"]) * v**3)
        )
    )
    boundary = np.full_like(omega, v / float(state["grain_size_m"]))
    point_defect = (
        derived["atomic_volume_m3"]
        * omega**4
        * derived["point_defect_Gamma"]
        / (4.0 * math.pi * v**3)
    )
    zeros = np.zeros_like(omega)
    return {
        "phonon_phonon": phonon_phonon,
        "boundary": boundary,
        "point_defect": point_defect,
        "dislocation_core": zeros.copy(),
        "dislocation_strain": zeros.copy(),
    }


def precipitate_cross_section(
    omega_rad_s: np.ndarray | float,
    radius_m: np.ndarray | float,
    config: dict[str, Any],
) -> np.ndarray:
    omega = np.asarray(omega_rad_s, dtype=float)
    radius = np.asarray(radius_m, dtype=float)
    if np.any(radius <= 0.0) or np.any(~np.isfinite(radius)):
        raise ValueError("All particle radii must be finite and positive")
    shared = config["shared_parameters"]
    contrast = material_derived(config, 0.0)["density_contrast_ratio"]
    v = float(shared["average_sound_velocity_m_s"])
    sigma_short = 2.0 * math.pi * radius**2
    sigma_long = (
        4.0
        / 9.0
        * math.pi
        * radius**2
        * contrast**2
        * (np.expand_dims(omega, axis=-1) * radius / v) ** 4
    )
    sigma_short_broadcast = np.broadcast_to(sigma_short, sigma_long.shape)
    denominator = sigma_short_broadcast + sigma_long
    return np.divide(
        sigma_short_broadcast * sigma_long,
        denominator,
        out=np.zeros_like(sigma_long),
        where=denominator > 0.0,
    )


def moments_from_radii(radii_m: np.ndarray, box_volume_m3: float) -> dict[str, float]:
    radii = np.asarray(radii_m, dtype=float)
    if radii.ndim != 1 or radii.size == 0:
        raise ValueError("A non-empty one-dimensional radius population is required")
    if box_volume_m3 <= 0.0 or not math.isfinite(box_volume_m3):
        raise ValueError("box_volume_m3 must be finite and positive")
    if np.any(radii <= 0.0) or np.any(~np.isfinite(radii)):
        raise ValueError("All particle radii must be finite and positive")
    nv = float(radii.size / box_volume_m3)
    sv = float(4.0 * math.pi * np.sum(radii**2) / box_volume_m3)
    m6 = float(np.sum(radii**6) / box_volume_m3)
    return {
        "particle_count": int(radii.size),
        "Nv_m-3": nv,
        "mean_radius_m": float(np.mean(radii)),
        "Sv_m-1": sv,
        "M6_m3": m6,
    }


def full_psd_precipitate_rate(
    omega_rad_s: np.ndarray | float,
    radii_m: np.ndarray,
    box_volume_m3: float,
    config: dict[str, Any],
) -> np.ndarray:
    omega = np.atleast_1d(np.asarray(omega_rad_s, dtype=float))
    cross_sections = precipitate_cross_section(omega, radii_m, config)
    v = float(config["shared_parameters"]["average_sound_velocity_m_s"])
    result = v * np.sum(cross_sections, axis=-1) / box_volume_m3
    return result


def direct_scalar_sum_precipitate_rate(
    omega_rad_s: np.ndarray | float,
    radii_m: np.ndarray,
    box_volume_m3: float,
    config: dict[str, Any],
) -> np.ndarray:
    omega = np.atleast_1d(np.asarray(omega_rad_s, dtype=float))
    v = float(config["shared_parameters"]["average_sound_velocity_m_s"])
    total = np.zeros_like(omega)
    for radius in np.asarray(radii_m, dtype=float):
        total += precipitate_cross_section(omega, np.array([radius]), config)[:, 0]
    return v * total / box_volume_m3


def descriptor_precipitate_rate(
    omega_rad_s: np.ndarray | float,
    radii_m: np.ndarray,
    box_volume_m3: float,
    config: dict[str, Any],
    model: str,
) -> np.ndarray:
    omega = np.atleast_1d(np.asarray(omega_rad_s, dtype=float))
    if model not in DESCRIPTOR_MODELS:
        raise ValueError(f"Unknown descriptor model: {model}")
    moments = moments_from_radii(radii_m, box_volume_m3)
    shared = config["shared_parameters"]
    v = float(shared["average_sound_velocity_m_s"])
    contrast = material_derived(config, 0.0)["density_contrast_ratio"]
    if model == "full_psd":
        return full_psd_precipitate_rate(omega, radii_m, box_volume_m3, config)
    if model == "Nv_plus_mean_R_monodisperse":
        cross = precipitate_cross_section(
            omega, np.array([moments["mean_radius_m"]]), config
        )[:, 0]
        return v * moments["Nv_m-3"] * cross
    if model == "Sv_geometric_limit":
        return np.full_like(omega, v * moments["Sv_m-1"] / 2.0)
    if model == "M6_rayleigh_limit":
        return (
            v
            * (4.0 / 9.0)
            * math.pi
            * contrast**2
            * (omega / v) ** 4
            * moments["M6_m3"]
        )
    # Equivalent monodisperse population matching both Sv and M6 exactly.
    radius_eq = (4.0 * math.pi * moments["M6_m3"] / moments["Sv_m-1"]) ** 0.25
    nv_eq = moments["Sv_m-1"] / (4.0 * math.pi * radius_eq**2)
    cross = precipitate_cross_section(omega, np.array([radius_eq]), config)[:, 0]
    return v * nv_eq * cross


def total_rate(
    omega_rad_s: np.ndarray | float,
    temperature_K: float,
    matrix_xag: float,
    radii_m: np.ndarray,
    box_volume_m3: float,
    config: dict[str, Any],
    descriptor_model: str,
) -> np.ndarray:
    base = base_scattering_rates(omega_rad_s, temperature_K, matrix_xag, config)
    precipitate = descriptor_precipitate_rate(
        omega_rad_s, radii_m, box_volume_m3, config, descriptor_model
    )
    return (
        base["phonon_phonon"]
        + base["boundary"]
        + base["point_defect"]
        + precipitate
    )


def bose_weight(x: np.ndarray | float) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    result = np.zeros_like(values)
    positive = values > 0.0
    xp = values[positive]
    result[positive] = xp**4 * np.exp(xp) / np.expm1(xp) ** 2
    return result


def dkappa_dx(
    x: np.ndarray | float,
    temperature_K: float,
    matrix_xag: float,
    radii_m: np.ndarray,
    box_volume_m3: float,
    config: dict[str, Any],
    descriptor_model: str,
) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    safe_x = np.where(values > 0.0, values, 1.0e-14)
    constants = config["physical_constants"]
    shared = config["shared_parameters"]
    omega = (
        safe_x
        * float(constants["k_B_J_K"])
        * temperature_K
        / float(constants["hbar_J_s"])
    )
    rate = total_rate(
        omega,
        temperature_K,
        matrix_xag,
        radii_m,
        box_volume_m3,
        config,
        descriptor_model,
    )
    prefactor = (
        float(constants["k_B_J_K"])
        / (2.0 * math.pi**2 * float(shared["average_sound_velocity_m_s"]))
        * (
            float(constants["k_B_J_K"])
            * temperature_K
            / float(constants["hbar_J_s"])
        )
        ** 3
    )
    return prefactor * bose_weight(safe_x) / rate


def integrate_kappa_gauss(
    temperature_K: float,
    matrix_xag: float,
    radii_m: np.ndarray,
    box_volume_m3: float,
    config: dict[str, Any],
    descriptor_model: str = "full_psd",
    order: int | None = None,
) -> float:
    n = int(order or config["numerics"]["gauss_legendre_order"])
    nodes, weights = leggauss(n)
    x_max = float(config["shared_parameters"]["debye_temperature_K"]) / temperature_K
    x = 0.5 * (nodes + 1.0) * x_max
    values = dkappa_dx(
        x,
        temperature_K,
        matrix_xag,
        radii_m,
        box_volume_m3,
        config,
        descriptor_model,
    )
    return float(0.5 * x_max * np.sum(weights * values))


def integrate_kappa_adaptive(
    temperature_K: float,
    matrix_xag: float,
    radii_m: np.ndarray,
    box_volume_m3: float,
    config: dict[str, Any],
    descriptor_model: str = "full_psd",
) -> tuple[float, float]:
    x_max = float(config["shared_parameters"]["debye_temperature_K"]) / temperature_K
    numerics = config["numerics"]

    def integrand(value: float) -> float:
        return float(
            dkappa_dx(
                np.array([value]),
                temperature_K,
                matrix_xag,
                radii_m,
                box_volume_m3,
                config,
                descriptor_model,
            )[0]
        )

    result, error = quad(
        integrand,
        0.0,
        x_max,
        epsabs=float(numerics["quad_epsabs"]),
        epsrel=float(numerics["quad_epsrel"]),
        limit=int(numerics["quad_limit"]),
    )
    return float(result), float(error)


def binned_center_radii(radii_m: np.ndarray, bin_width_m: float) -> np.ndarray:
    if bin_width_m <= 0.0:
        raise ValueError("bin_width_m must be positive")
    radii = np.asarray(radii_m, dtype=float)
    indices = np.floor(radii / bin_width_m)
    return (indices + 0.5) * bin_width_m


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / abs(reference) if reference != 0.0 else math.inf


def _float_text(value: float) -> str:
    return f"{value:.16g}"


def load_transport_snapshots(
    microstructure_csv: Path,
    particle_psd_csv: Path,
    replicate: str,
    matrix_xag_column: str = "matrix_xAg",
    matrix_xb_column: str = "matrix_xB",
    matrix_observation_contract: str = "upstream-defined matrix observation",
) -> tuple[list[dict[str, Any]], float, dict[str, float]]:
    micro_rows = read_csv(microstructure_csv)
    particle_rows = read_csv(particle_psd_csv)
    particles_by_step: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in particle_rows:
        particles_by_step[int(row["step"])].append(row)
    snapshots: list[dict[str, Any]] = []
    box_volumes_nm3: list[float] = []
    closure_max = {"Nv": 0.0, "Sv": 0.0, "M6": 0.0, "mean_R": 0.0}
    for row in micro_rows:
        if matrix_xag_column not in row or matrix_xb_column not in row:
            raise ValueError(
                "Requested matrix columns are absent: "
                f"{matrix_xag_column!r}, {matrix_xb_column!r}"
            )
        step = int(row["step"])
        particle_group = particles_by_step.get(step, [])
        expected_count = int(row["particle_count"])
        if len(particle_group) != expected_count:
            raise ValueError(
                f"PSD count mismatch at step {step}: {len(particle_group)} != {expected_count}"
            )
        if expected_count <= 0:
            raise ValueError(f"No resolved particles at step {step}")
        nv_nm3 = float(row["particle_number_density_nm^-3"])
        box_volume_nm3 = expected_count / nv_nm3
        box_volumes_nm3.append(box_volume_nm3)
        radii_nm = np.array(
            [float(item["equivalent_radius_nm"]) for item in particle_group], dtype=float
        )
        moments = moments_from_radii(radii_nm * 1.0e-9, box_volume_nm3 * 1.0e-27)
        source_values = {
            "Nv": nv_nm3 * 1.0e27,
            "Sv": float(row["Sv_spherical_equivalent_nm^-1"]) * 1.0e9,
            "M6": float(row["M6_integral_nm^3"]) * 1.0e-27,
            "mean_R": float(row["mean_radius_nm"]) * 1.0e-9,
        }
        calculated_values = {
            "Nv": moments["Nv_m-3"],
            "Sv": moments["Sv_m-1"],
            "M6": moments["M6_m3"],
            "mean_R": moments["mean_radius_m"],
        }
        for name in closure_max:
            closure_max[name] = max(
                closure_max[name],
                relative_error(calculated_values[name], source_values[name]),
            )
        sorted_particle_rows = sorted(
            particle_group, key=lambda item: int(item["particle_id"])
        )
        snapshot = {
            "snapshot_id": f"{replicate}_step_{step:06d}",
            "replicate": replicate,
            "step": step,
            "registered_age_h": float(row["registered_age_h"]),
            "age_h": float(row["age_h"]),
            "elapsed_physical_time_s": float(row["elapsed_physical_time_s"]),
            "particle_count": expected_count,
            "box_volume_nm3": box_volume_nm3,
            "Nv_nm-3": nv_nm3,
            "mean_radius_nm": float(row["mean_radius_nm"]),
            "Sv_nm-1": float(row["Sv_spherical_equivalent_nm^-1"]),
            "M6_nm3": float(row["M6_integral_nm^3"]),
            "beta_volume_fraction": float(row["beta_volume_fraction"]),
            "matrix_xAg": float(row[matrix_xag_column]),
            "matrix_xB": float(row[matrix_xb_column]),
            "matrix_xAg_source_column": matrix_xag_column,
            "matrix_xB_source_column": matrix_xb_column,
            "matrix_observation_contract": matrix_observation_contract,
            "radii_nm": [float(item["equivalent_radius_nm"]) for item in sorted_particle_rows],
            "particle_ids": [int(item["particle_id"]) for item in sorted_particle_rows],
        }
        snapshot["full_psd_canonical_sha256"] = canonical_sha256(
            list(zip(snapshot["particle_ids"], snapshot["radii_nm"]))
        )
        snapshots.append(snapshot)
    if not snapshots:
        raise ValueError("No transport snapshots were loaded")
    snapshots.sort(key=lambda item: (item["age_h"], item["step"]))
    reference_box_volume = float(np.mean(box_volumes_nm3))
    box_spread = max(abs(value - reference_box_volume) for value in box_volumes_nm3)
    if box_spread / reference_box_volume > 1.0e-12:
        raise ValueError("PF box volume is inconsistent across snapshots")
    if any(value > 5.0e-12 for value in closure_max.values()):
        raise ValueError(f"PF descriptor closure failed: {closure_max}")
    return snapshots, reference_box_volume, closure_max


def build_provenance(
    *,
    microstructure_csv: Path,
    particle_psd_csv: Path,
    upstream_audit_json: Path,
    upstream_analysis_manifest: Path,
    fixture_manifest: Path,
    yu_config: Path,
    interface_contract: Path,
    trajectory_class: str,
    replicate: str,
    source_commit: str | None,
    supplemental_provenance: Sequence[Path] = (),
) -> dict[str, Any]:
    audit = load_json(upstream_audit_json)
    observational_status = audit.get("observational_metrics_status")
    if observational_status != "PASS":
        raise ValueError(
            f"Upstream observational metrics must PASS, got {observational_status!r}"
        )
    upstream_input = audit.get("input", {})
    provenance = {
        "trajectory_class": trajectory_class,
        "replicate": replicate,
        "source_commit": source_commit,
        "upstream_audit_status": audit.get("status"),
        "upstream_observational_metrics_status": observational_status,
        "upstream_particle_identity_status": audit.get("gates", {}).get(
            "particle_identity_tracking_status"
        ),
        "identity_use_contract": (
            "overall snapshot PSD only; no independent post-merge particle lineage claim"
        ),
        "source_binary_sha256": upstream_input.get("source_binary_sha256"),
        "source_parameter_sha256": upstream_input.get("source_parameter_sha256"),
        "microstructure_csv": str(microstructure_csv.resolve()),
        "microstructure_csv_sha256": file_sha256(microstructure_csv),
        "particle_psd_csv": str(particle_psd_csv.resolve()),
        "particle_psd_csv_sha256": file_sha256(particle_psd_csv),
        "upstream_audit_json": str(upstream_audit_json.resolve()),
        "upstream_audit_json_sha256": file_sha256(upstream_audit_json),
        "upstream_analysis_manifest": str(upstream_analysis_manifest.resolve()),
        "upstream_analysis_manifest_sha256": file_sha256(upstream_analysis_manifest),
        "fixture_manifest": str(fixture_manifest.resolve()),
        "fixture_manifest_sha256": file_sha256(fixture_manifest),
        "yu_public_parameter_config": str(yu_config.resolve()),
        "yu_public_parameter_config_sha256": file_sha256(yu_config),
        "transport_parameter_contract": str(interface_contract.resolve()),
        "transport_parameter_contract_sha256": file_sha256(interface_contract),
        "transport_interface_script": str(Path(__file__).resolve()),
        "transport_interface_script_sha256": file_sha256(Path(__file__)),
        "supplemental_provenance": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in supplemental_provenance
        ],
    }
    required_hashes = (
        "source_binary_sha256",
        "source_parameter_sha256",
        "fixture_manifest_sha256",
        "upstream_analysis_manifest_sha256",
        "microstructure_csv_sha256",
        "particle_psd_csv_sha256",
        "yu_public_parameter_config_sha256",
        "transport_parameter_contract_sha256",
        "transport_interface_script_sha256",
    )
    missing = [name for name in required_hashes if not provenance.get(name)]
    if missing:
        raise ValueError(f"Required provenance is missing: {missing}")
    return provenance


def compute_transport_trajectory(
    snapshots: list[dict[str, Any]],
    box_volume_nm3: float,
    config: dict[str, Any],
    temperatures_K: Sequence[float],
    fixed_matrix_xag: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    first = snapshots[0]
    fixed_xag = (
        float(fixed_matrix_xag)
        if fixed_matrix_xag is not None
        else float(first["matrix_xAg"])
    )
    box_volume_m3 = box_volume_nm3 * 1.0e-27
    baseline_radii_m = np.asarray(first["radii_nm"], dtype=float) * 1.0e-9
    full_rows: list[dict[str, Any]] = []
    descriptor_rows: list[dict[str, Any]] = []
    baseline_by_temperature: dict[float, float] = {}
    for temperature in temperatures_K:
        baseline_by_temperature[temperature] = integrate_kappa_gauss(
            temperature,
            fixed_xag,
            baseline_radii_m,
            box_volume_m3,
            config,
            "full_psd",
        )
    for snapshot in snapshots:
        radii_m = np.asarray(snapshot["radii_nm"], dtype=float) * 1.0e-9
        for temperature in temperatures_K:
            baseline = baseline_by_temperature[temperature]
            full_values: dict[str, float] = {}
            for matrix_mode in MATRIX_MODES:
                matrix_xag = (
                    fixed_xag
                    if matrix_mode == "fixed_6h_matrix"
                    else float(snapshot["matrix_xAg"])
                )
                full_value = integrate_kappa_gauss(
                    temperature,
                    matrix_xag,
                    radii_m,
                    box_volume_m3,
                    config,
                    "full_psd",
                )
                full_values[matrix_mode] = full_value
                for model in DESCRIPTOR_MODELS:
                    value = (
                        full_value
                        if model == "full_psd"
                        else integrate_kappa_gauss(
                            temperature,
                            matrix_xag,
                            radii_m,
                            box_volume_m3,
                            config,
                            model,
                        )
                    )
                    descriptor_rows.append(
                        {
                            "replicate": snapshot["replicate"],
                            "snapshot_id": snapshot["snapshot_id"],
                            "step": snapshot["step"],
                            "age_h": _float_text(snapshot["age_h"]),
                            "temperature_K": _float_text(temperature),
                            "matrix_mode": matrix_mode,
                            "matrix_xAg": _float_text(matrix_xag),
                            "descriptor_model": model,
                            "kappa_L_no_dis_W_mK": _float_text(value),
                            "full_psd_kappa_L_no_dis_W_mK": _float_text(full_value),
                            "absolute_error_vs_full_psd_W_mK": _float_text(
                                abs(value - full_value)
                            ),
                            "relative_error_vs_full_psd": _float_text(
                                relative_error(value, full_value)
                            ),
                        }
                    )
            pure = full_values["fixed_6h_matrix"]
            coupled = full_values["pf_time_varying_matrix"]
            full_rows.append(
                {
                    "replicate": snapshot["replicate"],
                    "snapshot_id": snapshot["snapshot_id"],
                    "step": snapshot["step"],
                    "age_h": _float_text(snapshot["age_h"]),
                    "temperature_K": _float_text(temperature),
                    "fixed_6h_matrix_xAg": _float_text(fixed_xag),
                    "pf_time_varying_matrix_xAg": _float_text(
                        snapshot["matrix_xAg"]
                    ),
                    "kappa_L_no_dis_fixed_matrix_W_mK": _float_text(pure),
                    "kappa_L_no_dis_time_varying_matrix_W_mK": _float_text(coupled),
                    "delta_kappa_PSD_W_mK": _float_text(pure - baseline),
                    "delta_kappa_PSD_plus_matrix_W_mK": _float_text(
                        coupled - baseline
                    ),
                    "delta_kappa_matrix_given_PSD_W_mK": _float_text(coupled - pure),
                    "reference_6h_full_psd_fixed_matrix_W_mK": _float_text(baseline),
                }
            )
    error_summary: list[dict[str, Any]] = []
    for matrix_mode in MATRIX_MODES:
        for model in DESCRIPTOR_MODELS:
            selected = [
                row
                for row in descriptor_rows
                if row["matrix_mode"] == matrix_mode
                and row["descriptor_model"] == model
            ]
            errors = np.array(
                [float(row["relative_error_vs_full_psd"]) for row in selected]
            )
            absolute = np.array(
                [float(row["absolute_error_vs_full_psd_W_mK"]) for row in selected]
            )
            error_summary.append(
                {
                    "matrix_mode": matrix_mode,
                    "descriptor_model": model,
                    "observations": len(selected),
                    "MAPE_fraction_vs_full_psd": _float_text(float(np.mean(errors))),
                    "max_relative_error_vs_full_psd": _float_text(float(np.max(errors))),
                    "RMSE_W_mK_vs_full_psd": _float_text(
                        float(np.sqrt(np.mean(absolute**2)))
                    ),
                }
            )
    return full_rows, descriptor_rows, error_summary


def export_transport_snapshot(
    *,
    output_dir: Path,
    snapshots: list[dict[str, Any]],
    box_volume_nm3: float,
    closure: dict[str, float],
    provenance: dict[str, Any],
    config: dict[str, Any],
    temperatures_K: Sequence[float],
    fixed_matrix_xag: float | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_rows, descriptor_rows, error_rows = compute_transport_trajectory(
        snapshots,
        box_volume_nm3,
        config,
        temperatures_K,
        fixed_matrix_xag,
    )
    snapshot_rows: list[dict[str, Any]] = []
    particle_rows: list[dict[str, Any]] = []
    per_particle_density_m3 = 1.0 / (box_volume_nm3 * 1.0e-27)
    for snapshot in snapshots:
        snapshot_rows.append(
            {
                key: snapshot[key]
                for key in (
                    "snapshot_id",
                    "replicate",
                    "step",
                    "registered_age_h",
                    "age_h",
                    "elapsed_physical_time_s",
                    "particle_count",
                    "box_volume_nm3",
                    "Nv_nm-3",
                    "mean_radius_nm",
                    "Sv_nm-1",
                    "M6_nm3",
                    "beta_volume_fraction",
                    "matrix_xAg",
                    "matrix_xB",
                    "matrix_xAg_source_column",
                    "matrix_xB_source_column",
                    "matrix_observation_contract",
                    "full_psd_canonical_sha256",
                )
            }
        )
        for particle_id, radius_nm in zip(
            snapshot["particle_ids"], snapshot["radii_nm"]
        ):
            particle_rows.append(
                {
                    "snapshot_id": snapshot["snapshot_id"],
                    "replicate": snapshot["replicate"],
                    "step": snapshot["step"],
                    "age_h": _float_text(snapshot["age_h"]),
                    "particle_id": particle_id,
                    "equivalent_radius_nm": _float_text(radius_nm),
                    "per_particle_number_density_weight_m-3": _float_text(
                        per_particle_density_m3
                    ),
                }
            )
    write_csv(
        output_dir / "transport_snapshots.csv",
        list(snapshot_rows[0].keys()),
        snapshot_rows,
    )
    write_csv(
        output_dir / "full_psd_particles.csv",
        list(particle_rows[0].keys()),
        particle_rows,
    )
    write_csv(
        output_dir / "kappa_time_temperature.csv",
        list(full_rows[0].keys()),
        full_rows,
    )
    write_csv(
        output_dir / "descriptor_predictions.csv",
        list(descriptor_rows[0].keys()),
        descriptor_rows,
    )
    write_csv(
        output_dir / "descriptor_error_summary.csv",
        list(error_rows[0].keys()),
        error_rows,
    )
    fixed_xag = (
        float(fixed_matrix_xag)
        if fixed_matrix_xag is not None
        else float(snapshots[0]["matrix_xAg"])
    )
    manifest = {
        "schema": SCHEMA,
        "status": FINAL_STATUS,
        "scientific_claim": (
            "conditional resolved-PF-PSD no-dislocation lattice transport; "
            "not absolute experimental kappa reproduction"
        ),
        "transport_contract": {
            "A_N": FROZEN_A_N,
            "dislocation_mode": DISLOCATION_MODE,
            "S11_rate": 0.0,
            "S13_rate": 0.0,
            "yu_refit_scale_used": False,
            "full_psd_equation": (
                "tau_Pre^-1 = v/V_box * sum_i "
                "(sigma_S^-1 + sigma_l^-1)^-1"
            ),
            "matrix_coupling": (
                "PF far-field xAg changes point-defect Gamma only; lattice, grain, "
                "sound, Debye and all other material parameters remain frozen"
            ),
            "fixed_matrix_xAg": fixed_xag,
            "matrix_modes": list(MATRIX_MODES),
            "descriptor_models": list(DESCRIPTOR_MODELS),
            "temperature_grid_K": list(temperatures_K),
        },
        "snapshot_contract": {
            "count": len(snapshots),
            "first_age_h": snapshots[0]["age_h"],
            "last_age_h": snapshots[-1]["age_h"],
            "box_volume_nm3": box_volume_nm3,
            "full_psd_particle_rows": len(particle_rows),
            "descriptor_closure_max_relative": closure,
        },
        "provenance": provenance,
        "outputs": {},
    }
    for name in (
        "transport_snapshots.csv",
        "full_psd_particles.csv",
        "kappa_time_temperature.csv",
        "descriptor_predictions.csv",
        "descriptor_error_summary.csv",
    ):
        manifest["outputs"][name] = file_sha256(output_dir / name)
    write_json(output_dir / "transport_snapshot_manifest.json", manifest)
    return manifest


def parse_temperatures(value: str) -> tuple[float, ...]:
    temperatures = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not temperatures or any(item <= 0.0 for item in temperatures):
        raise ValueError("At least one positive temperature is required")
    return temperatures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--microstructure-csv", type=Path, required=True)
    parser.add_argument("--particle-psd-csv", type=Path, required=True)
    parser.add_argument("--upstream-audit-json", type=Path, required=True)
    parser.add_argument("--upstream-analysis-manifest", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument(
        "--yu-config",
        type=Path,
        default=Path("data/qualification/yu2024_transport_v1/yu_48h_parameters.json"),
    )
    parser.add_argument(
        "--interface-contract", type=Path, default=DEFAULT_INTERFACE_CONTRACT
    )
    parser.add_argument("--replicate", required=True)
    parser.add_argument("--trajectory-class", required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--matrix-xag-column", default="matrix_xAg")
    parser.add_argument("--matrix-xb-column", default="matrix_xB")
    parser.add_argument(
        "--matrix-observation-contract",
        default="upstream-defined matrix observation",
    )
    parser.add_argument(
        "--supplemental-provenance", type=Path, action="append", default=[]
    )
    parser.add_argument("--fixed-matrix-xag", type=float)
    parser.add_argument(
        "--temperatures-K",
        default=",".join(str(value) for value in DEFAULT_TEMPERATURES_K),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.yu_config)
    validate_yu_base_config(config, args.yu_config)
    validate_interface_contract(
        load_json(args.interface_contract), args.interface_contract, args.yu_config
    )
    snapshots, box_volume_nm3, closure = load_transport_snapshots(
        args.microstructure_csv,
        args.particle_psd_csv,
        args.replicate,
        args.matrix_xag_column,
        args.matrix_xb_column,
        args.matrix_observation_contract,
    )
    provenance = build_provenance(
        microstructure_csv=args.microstructure_csv,
        particle_psd_csv=args.particle_psd_csv,
        upstream_audit_json=args.upstream_audit_json,
        upstream_analysis_manifest=args.upstream_analysis_manifest,
        fixture_manifest=args.fixture_manifest,
        yu_config=args.yu_config,
        interface_contract=args.interface_contract,
        trajectory_class=args.trajectory_class,
        replicate=args.replicate,
        source_commit=args.source_commit,
        supplemental_provenance=args.supplemental_provenance,
    )
    export_transport_snapshot(
        output_dir=args.output_dir,
        snapshots=snapshots,
        box_volume_nm3=box_volume_nm3,
        closure=closure,
        provenance=provenance,
        config=config,
        temperatures_K=parse_temperatures(args.temperatures_K),
        fixed_matrix_xag=args.fixed_matrix_xag,
    )


if __name__ == "__main__":
    main()
