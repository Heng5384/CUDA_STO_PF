#!/usr/bin/env python3
"""Independent host-only reproduction of Yu et al. 2024 thermal transport.

This implementation follows main-text Eq. (3), Eq. (4), SI Eqs. (S1)-(S16),
and SI Table S2 of doi:10.1002/aenm.202304442. It deliberately does not read
phase-field outputs and does not fit any parameter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad


MECHANISMS = (
    "phonon_phonon",
    "boundary",
    "point_defect",
    "precipitate_small",
    "precipitate_big",
    "dislocation_core",
    "dislocation_strain",
)

STAGES = {
    "U_N_combined_only": ("phonon_phonon",),
    "U_N_plus_GB": ("phonon_phonon", "boundary"),
    "U_N_plus_GB_PD": ("phonon_phonon", "boundary", "point_defect"),
    "U_N_plus_GB_PD_Pre_small": (
        "phonon_phonon",
        "boundary",
        "point_defect",
        "precipitate_small",
    ),
    "U_N_plus_GB_PD_Pre_all": (
        "phonon_phonon",
        "boundary",
        "point_defect",
        "precipitate_small",
        "precipitate_big",
    ),
    "full": MECHANISMS,
}

GROUPS = {
    "phonon_phonon_combined": ("phonon_phonon",),
    "boundary": ("boundary",),
    "point_defect": ("point_defect",),
    "precipitate": ("precipitate_small", "precipitate_big"),
    "dislocation": ("dislocation_core", "dislocation_strain"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aq-config",
        type=Path,
        default=Path("data/qualification/yu2024_transport_v1/yu_AQ_parameters.json"),
    )
    parser.add_argument(
        "--h48-config",
        type=Path,
        default=Path("data/qualification/yu2024_transport_v1/yu_48h_parameters.json"),
    )
    parser.add_argument(
        "--experimental-dir",
        type=Path,
        default=Path("data/qualification/yu2024_transport_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/yu2024_debye_callaway_reproduction_v1"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def validate_config_contract(aq: dict[str, Any], h48: dict[str, Any]) -> None:
    for config in (aq, h48):
        contract = config["source_contract"]
        if not contract["single_relaxation_time_integral"]:
            raise ValueError("Yu main-text Eq. (3) must remain the single-rate integral")
        if contract["normal_and_umklapp_separable"]:
            raise ValueError("SI Eq. (S1) does not separate Normal and Umklapp rates")
        if contract["callaway_second_term_present"]:
            raise ValueError("Yu 2024 does not publish a Callaway second term")
        if contract["parameter_refit_authorized"]:
            raise ValueError("Parameter refitting is forbidden")
    if aq["shared_parameters"] != h48["shared_parameters"]:
        raise ValueError("AQ and 48 h shared parameter blocks differ")
    if aq["physical_constants"] != h48["physical_constants"]:
        raise ValueError("AQ and 48 h physical constants differ")
    if aq["source_doi"] != h48["source_doi"]:
        raise ValueError("AQ and 48 h source DOI differs")


def derived_parameters(config: dict[str, Any]) -> dict[str, float]:
    c = config["physical_constants"]
    s = config["shared_parameters"]
    q = config["state_parameters"]
    lattice_m = q["solid_solution_lattice_constant_angstrom"] * 1.0e-10
    atomic_volume = lattice_m**3 / 8.0
    alpha = (
        s["impurity_atomic_volume_m3"] - s["matrix_atomic_volume_m3"]
    ) / s["matrix_atomic_volume_m3"]
    beta = 0.5 * (
        s["matrix_atom_mass_g_mol"] - s["impurity_atom_mass_g_mol"]
    ) / s["matrix_atom_mass_g_mol"]
    gamma_prime_derived = (
        s["matrix_atomic_volume_m3"]
        * s["impurity_fraction_near_dislocations"]
        * s["bulk_modulus_Pa"]
        / (c["k_B_J_K"] * s["annealing_temperature_K_for_gamma_prime"])
        * (s["gruneisen_gamma"] * alpha**2 - alpha * beta)
    )
    mass_ratio = (
        s["published_delta_M_i_g_mol"] / s["matrix_atom_mass_g_mol"]
    )
    radius_ratio = (
        s["impurity_atomic_radius_pm"] - s["matrix_atomic_radius_pm"]
    ) / s["matrix_atomic_radius_pm"]
    point_gamma = q["interstitial_Ag_fraction"] * (
        mass_ratio**2 + s["point_defect_epsilon"] * radius_ratio**2
    )
    return {
        "atomic_volume_m3": atomic_volume,
        "alpha": alpha,
        "beta": beta,
        "gamma_prime_derived": gamma_prime_derived,
        "point_defect_mass_ratio": mass_ratio,
        "point_defect_radius_ratio": radius_ratio,
        "point_defect_Gamma": point_gamma,
        "density_contrast_ratio": (
            s["density_difference_kg_m3"] / s["matrix_density_kg_m3"]
        ),
    }


def scattering_rates(
    omega_rad_s: np.ndarray | float,
    temperature_K: float,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    omega = np.asarray(omega_rad_s, dtype=float)
    c = config["physical_constants"]
    s = config["shared_parameters"]
    q = config["state_parameters"]
    d = derived_parameters(config)
    v = s["average_sound_velocity_m_s"]

    phonon_phonon = (
        s["A_N"]
        * 2.0
        / (6.0 * math.pi**2) ** (1.0 / 3.0)
        * (
            c["k_B_J_K"]
            * d["atomic_volume_m3"] ** (1.0 / 3.0)
            * s["gruneisen_gamma"] ** 2
            * omega**2
            * temperature_K
            / (s["average_atomic_mass_kg"] * v**3)
        )
    )
    boundary = np.full_like(omega, v / q["grain_size_m"])
    point_defect = (
        d["atomic_volume_m3"]
        * omega**4
        * d["point_defect_Gamma"]
        / (4.0 * math.pi * v**3)
    )

    def precipitate_rate(radius_m: float, number_density_m3: float) -> np.ndarray:
        sigma_short = 2.0 * math.pi * radius_m**2
        sigma_long = (
            4.0
            / 9.0
            * math.pi
            * radius_m**2
            * d["density_contrast_ratio"] ** 2
            * (omega * radius_m / v) ** 4
        )
        sigma_effective = np.divide(
            sigma_short * sigma_long,
            sigma_short + sigma_long,
            out=np.zeros_like(omega),
            where=(sigma_short + sigma_long) > 0.0,
        )
        return v * number_density_m3 * sigma_effective

    precipitate_small = precipitate_rate(
        q["small_precipitate_radius_m"],
        q["small_precipitate_number_density_m3"],
    )
    precipitate_big = precipitate_rate(
        q["big_precipitate_radius_m"],
        q["big_precipitate_number_density_m3"],
    )

    dislocation_core = (
        q["dislocation_density_m2"]
        * d["atomic_volume_m3"] ** (4.0 / 3.0)
        / v**2
        * omega**3
    )
    bracket = 0.5 + (
        1.0
        / 24.0
        * ((1.0 - 2.0 * s["poisson_ratio"]) / (1.0 - s["poisson_ratio"]))
        ** 2
        * (
            1.0
            + math.sqrt(2.0)
            * (
                s["longitudinal_sound_velocity_m_s"]
                / s["transverse_sound_velocity_m_s"]
            )
            ** 2
        )
        ** 2
    )
    dislocation_strain = (
        s["dislocation_prefactor_C"]
        * s["burgers_vector_m"] ** 2
        * q["dislocation_density_m2"]
        * (s["gruneisen_gamma"] + s["published_gamma_prime"]) ** 2
        * omega
        * bracket
    )
    return {
        "phonon_phonon": phonon_phonon,
        "boundary": boundary,
        "point_defect": point_defect,
        "precipitate_small": precipitate_small,
        "precipitate_big": precipitate_big,
        "dislocation_core": dislocation_core,
        "dislocation_strain": dislocation_strain,
    }


def total_rate(
    omega_rad_s: np.ndarray | float,
    temperature_K: float,
    config: dict[str, Any],
    mechanisms: Iterable[str],
) -> np.ndarray:
    rates = scattering_rates(omega_rad_s, temperature_K, config)
    selected = tuple(mechanisms)
    if not selected:
        raise ValueError("At least one scattering mechanism is required")
    unknown = set(selected) - set(MECHANISMS)
    if unknown:
        raise ValueError(f"Unknown scattering mechanisms: {sorted(unknown)}")
    total = np.zeros_like(np.asarray(omega_rad_s, dtype=float))
    for mechanism in selected:
        total = total + rates[mechanism]
    return total


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
    config: dict[str, Any],
    mechanisms: Iterable[str],
) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    safe_x = np.where(values > 0.0, values, 1.0e-14)
    c = config["physical_constants"]
    s = config["shared_parameters"]
    omega = safe_x * c["k_B_J_K"] * temperature_K / c["hbar_J_s"]
    rate = total_rate(omega, temperature_K, config, mechanisms)
    tau = np.divide(1.0, rate, out=np.full_like(rate, np.inf), where=rate > 0.0)
    prefactor = (
        c["k_B_J_K"]
        / (2.0 * math.pi**2 * s["average_sound_velocity_m_s"])
        * (c["k_B_J_K"] * temperature_K / c["hbar_J_s"]) ** 3
    )
    result = prefactor * tau * bose_weight(safe_x)
    if np.any(values <= 0.0):
        near_zero = np.full(np.count_nonzero(values <= 0.0), 1.0e-12)
        omega0 = (
            near_zero
            * c["k_B_J_K"]
            * temperature_K
            / c["hbar_J_s"]
        )
        rate0 = total_rate(omega0, temperature_K, config, mechanisms)
        result = np.asarray(result)
        result[values <= 0.0] = (
            prefactor * bose_weight(near_zero) / rate0
        )
    return result


def integrate_gauss_legendre(
    temperature_K: float,
    config: dict[str, Any],
    mechanisms: Iterable[str] = MECHANISMS,
    order: int | None = None,
) -> float:
    s = config["shared_parameters"]
    n = order or config["numerics"]["gauss_legendre_order"]
    nodes, weights = leggauss(n)
    x_max = s["debye_temperature_K"] / temperature_K
    x = 0.5 * (nodes + 1.0) * x_max
    return float(
        0.5
        * x_max
        * np.sum(weights * dkappa_dx(x, temperature_K, config, mechanisms))
    )


def integrate_adaptive_quad(
    temperature_K: float,
    config: dict[str, Any],
    mechanisms: Iterable[str] = MECHANISMS,
) -> tuple[float, float]:
    s = config["shared_parameters"]
    n = config["numerics"]
    x_max = s["debye_temperature_K"] / temperature_K

    def scalar_integrand(x: float) -> float:
        return float(dkappa_dx(np.array([x]), temperature_K, config, mechanisms)[0])

    value, error = quad(
        scalar_integrand,
        0.0,
        x_max,
        epsabs=n["quad_epsabs"],
        epsrel=n["quad_epsrel"],
        limit=n["quad_limit"],
    )
    return float(value), float(error)


def read_experimental(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "state": row["state"],
                    "T_K": float(row["T_K"]),
                    "kappa_lat_W_mK": float(row["kappa_lat_W_mK"]),
                    "data_origin": row["data_origin"],
                    "source_location": row["source_location"],
                    "digitization_uncertainty_W_mK": float(
                        row["digitization_uncertainty_W_mK"]
                    ),
                }
            )
    return rows


def read_digitized_model(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "state": row["state"],
                    "T_K": float(row["T_K"]),
                    "yu_model_kappa_lat_W_mK": float(
                        row["yu_model_kappa_lat_W_mK"]
                    ),
                    "digitization_uncertainty_W_mK": float(
                        row["digitization_uncertainty_W_mK"]
                    ),
                }
            )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def weighted_mechanism_statistics(
    temperature_K: float, config: dict[str, Any]
) -> dict[str, Any]:
    s = config["shared_parameters"]
    n = config["numerics"]["gauss_legendre_order"]
    nodes, weights = leggauss(n)
    x_max = s["debye_temperature_K"] / temperature_K
    x = 0.5 * (nodes + 1.0) * x_max
    quadrature_weights = 0.5 * x_max * weights
    kernel = dkappa_dx(x, temperature_K, config, MECHANISMS)
    contributions = quadrature_weights * kernel
    kappa = float(np.sum(contributions))
    normalized = contributions / kappa
    c = config["physical_constants"]
    omega = x * c["k_B_J_K"] * temperature_K / c["hbar_J_s"]
    rates = scattering_rates(omega, temperature_K, config)
    weighted_rates = {
        name: float(np.sum(normalized * values)) for name, values in rates.items()
    }
    cumulative = np.cumsum(normalized)
    i10 = min(int(np.searchsorted(cumulative, 0.10)), len(omega) - 1)
    i90 = min(int(np.searchsorted(cumulative, 0.90)), len(omega) - 1)
    dominant = max(weighted_rates, key=weighted_rates.get)
    return {
        "kappa_W_mK": kappa,
        "kernel_weighted_total_rate_s-1": float(sum(weighted_rates.values())),
        "weighted_rates": weighted_rates,
        "dominant_omega_low_10pct_1e12_rad_s": float(omega[i10] / 1.0e12),
        "dominant_omega_high_90pct_1e12_rad_s": float(omega[i90] / 1.0e12),
        "dominant_mechanism": dominant,
    }


def compute_budget(
    experiments: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        temperature = experiment["T_K"]
        stats = weighted_mechanism_statistics(temperature, config)
        stage_values = {
            stage: integrate_gauss_legendre(temperature, config, mechanisms)
            for stage, mechanisms in STAGES.items()
        }
        without: dict[str, float] = {}
        delta: dict[str, float] = {}
        for group, members in GROUPS.items():
            mechanisms = tuple(m for m in MECHANISMS if m not in members)
            value = integrate_gauss_legendre(temperature, config, mechanisms)
            without[group] = value
            delta[group] = value - stats["kappa_W_mK"]
        row: dict[str, Any] = {
            "state": config["state"],
            "T_K": f"{temperature:.8g}",
            "kappa_full_W_mK": f"{stats['kappa_W_mK']:.12g}",
            "kernel_weighted_total_rate_s-1": (
                f"{stats['kernel_weighted_total_rate_s-1']:.12g}"
            ),
            "dominant_omega_low_10pct_1e12_rad_s": (
                f"{stats['dominant_omega_low_10pct_1e12_rad_s']:.12g}"
            ),
            "dominant_omega_high_90pct_1e12_rad_s": (
                f"{stats['dominant_omega_high_90pct_1e12_rad_s']:.12g}"
            ),
            "dominant_mechanism": stats["dominant_mechanism"],
        }
        for stage, value in stage_values.items():
            row[f"kappa_{stage}_W_mK"] = f"{value:.12g}"
        for group in GROUPS:
            row[f"kappa_without_{group}_W_mK"] = f"{without[group]:.12g}"
            row[f"delta_kappa_off_{group}_W_mK"] = f"{delta[group]:.12g}"
        for mechanism in MECHANISMS:
            row[f"weighted_rate_{mechanism}_s-1"] = (
                f"{stats['weighted_rates'][mechanism]:.12g}"
            )
        rows.append(row)
    return rows


def compute_spectral_rows(
    experiments: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    c = config["physical_constants"]
    s = config["shared_parameters"]
    points = config["numerics"]["spectral_grid_points"]
    for experiment in experiments:
        temperature = experiment["T_K"]
        x = np.linspace(0.0, s["debye_temperature_K"] / temperature, points)
        omega = x * c["k_B_J_K"] * temperature / c["hbar_J_s"]
        rates = scattering_rates(omega, temperature, config)
        total = sum(rates.values())
        kernel_x = dkappa_dx(x, temperature, config, MECHANISMS)
        spectral_pW_s_mK = (
            kernel_x * c["hbar_J_s"] / (c["k_B_J_K"] * temperature) * 1.0e12
        )
        for index in range(points):
            row: dict[str, Any] = {
                "state": config["state"],
                "T_K": f"{temperature:.8g}",
                "x": f"{x[index]:.12g}",
                "omega_rad_s": f"{omega[index]:.12g}",
                "omega_1e12_rad_s_paper_frequency_axis": (
                    f"{omega[index] / 1.0e12:.12g}"
                ),
                "frequency_THz_cycles_s": (
                    f"{omega[index] / (2.0 * math.pi * 1.0e12):.12g}"
                ),
                "dkappa_dx_W_mK": f"{kernel_x[index]:.12g}",
                "spectral_kappa_pW_s_mK": f"{spectral_pW_s_mK[index]:.12g}",
                "total_rate_s-1": f"{total[index]:.12g}",
            }
            for mechanism in MECHANISMS:
                row[f"{mechanism}_rate_s-1"] = f"{rates[mechanism][index]:.12g}"
            rows.append(row)
    return rows


def relative_difference(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b != 0.0 else math.inf


def log_slope(rate_a: float, rate_b: float, omega_a: float, omega_b: float) -> float:
    return math.log(rate_b / rate_a) / math.log(omega_b / omega_a)


def validate_numerics(
    experiments_by_state: dict[str, list[dict[str, Any]]],
    configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    convergence_rows: list[dict[str, Any]] = []
    positivity_pass = True
    kappa_pass = True
    repeat_signatures: dict[str, dict[str, str]] = {}
    for state, config in configs.items():
        signature_values_first: list[float] = []
        signature_values_second: list[float] = []
        for experiment in experiments_by_state[state]:
            temperature = experiment["T_K"]
            base_order = config["numerics"]["gauss_legendre_order"]
            refined_order = config["numerics"]["gauss_legendre_refined_order"]
            baseline = integrate_gauss_legendre(
                temperature, config, MECHANISMS, base_order
            )
            refined = integrate_gauss_legendre(
                temperature, config, MECHANISMS, refined_order
            )
            adaptive, adaptive_error = integrate_adaptive_quad(
                temperature, config, MECHANISMS
            )
            signature_values_first.append(baseline)
            signature_values_second.append(
                integrate_gauss_legendre(
                    temperature, config, MECHANISMS, base_order
                )
            )
            nodes, _ = leggauss(base_order)
            x = 0.5 * (nodes + 1.0) * (
                config["shared_parameters"]["debye_temperature_K"] / temperature
            )
            omega = (
                x
                * config["physical_constants"]["k_B_J_K"]
                * temperature
                / config["physical_constants"]["hbar_J_s"]
            )
            all_rates = total_rate(omega, temperature, config, MECHANISMS)
            positivity_pass = positivity_pass and bool(
                np.all(np.isfinite(all_rates)) and np.all(all_rates > 0.0)
            )
            kappa_pass = kappa_pass and all(
                math.isfinite(value) and value > 0.0
                for value in (baseline, refined, adaptive)
            )
            convergence_rows.append(
                {
                    "state": state,
                    "T_K": temperature,
                    "gauss_order": base_order,
                    "gauss_refined_order": refined_order,
                    "kappa_gauss_W_mK": baseline,
                    "kappa_gauss_refined_W_mK": refined,
                    "kappa_quad_W_mK": adaptive,
                    "quad_reported_abs_error": adaptive_error,
                    "gauss_refinement_relative_difference": relative_difference(
                        baseline, refined
                    ),
                    "gauss_vs_quad_relative_difference": relative_difference(
                        baseline, adaptive
                    ),
                }
            )
        repeat_signatures[state] = {
            "first": canonical_sha256(signature_values_first),
            "second": canonical_sha256(signature_values_second),
        }

    sample = configs["AQ"]
    rates_low_a = scattering_rates(1.0e7, 300.0, sample)
    rates_low_b = scattering_rates(1.0e8, 300.0, sample)
    rates_high_a = scattering_rates(1.0e16, 300.0, sample)
    rates_high_b = scattering_rates(1.0e17, 300.0, sample)
    slopes = {
        "phonon_phonon_low_frequency": log_slope(
            float(rates_low_a["phonon_phonon"]),
            float(rates_low_b["phonon_phonon"]),
            1.0e7,
            1.0e8,
        ),
        "point_defect_low_frequency": log_slope(
            float(rates_low_a["point_defect"]),
            float(rates_low_b["point_defect"]),
            1.0e7,
            1.0e8,
        ),
        "precipitate_small_rayleigh": log_slope(
            float(rates_low_a["precipitate_small"]),
            float(rates_low_b["precipitate_small"]),
            1.0e7,
            1.0e8,
        ),
        "precipitate_small_geometric": log_slope(
            float(rates_high_a["precipitate_small"]),
            float(rates_high_b["precipitate_small"]),
            1.0e16,
            1.0e17,
        ),
    }
    d = derived_parameters(sample)
    s = sample["shared_parameters"]
    radius = sample["state_parameters"]["small_precipitate_radius_m"]
    crossover_dimensionless = (
        4.5 / d["density_contrast_ratio"] ** 2
    ) ** 0.25
    crossover_omega = (
        crossover_dimensionless * s["average_sound_velocity_m_s"] / radius
    )
    rate_below = float(
        scattering_rates(crossover_omega * 0.999, 300.0, sample)[
            "precipitate_small"
        ]
    )
    rate_above = float(
        scattering_rates(crossover_omega * 1.001, 300.0, sample)[
            "precipitate_small"
        ]
    )
    max_grid_refinement = max(
        row["gauss_refinement_relative_difference"] for row in convergence_rows
    )
    max_quad_difference = max(
        row["gauss_vs_quad_relative_difference"] for row in convergence_rows
    )
    gamma_prime_relative_difference = relative_difference(
        d["gamma_prime_derived"], s["published_gamma_prime"]
    )
    deterministic = all(
        value["first"] == value["second"] for value in repeat_signatures.values()
    )
    return {
        "source_contract": {
            "normal_umklapp_separable": False,
            "callaway_second_term_present": False,
            "refit_performed": False,
        },
        "unit_checks": {
            "angstrom_to_m": 1.0e-10,
            "cm^-2_to_m^-2": 1.0e4,
            "g_cm^-3_to_kg_m^-3": 1.0e3,
            "published_Vm_Vi_unit_typo": (
                "Table S2 prints m^-3; values are used as m^3/atom because "
                "Eq. S15 and dimensional consistency require volumes"
            ),
            "gamma_prime_derived": d["gamma_prime_derived"],
            "gamma_prime_published": s["published_gamma_prime"],
            "gamma_prime_relative_difference": gamma_prime_relative_difference,
            "pass": gamma_prime_relative_difference <= 0.01,
        },
        "convergence": {
            "rows": convergence_rows,
            "max_grid_refinement_relative_difference": max_grid_refinement,
            "max_gauss_vs_quad_relative_difference": max_quad_difference,
            "grid_threshold": 0.005,
            "quad_threshold": 0.002,
            "pass": max_grid_refinement <= 0.005
            and max_quad_difference <= 0.002,
        },
        "positivity": {
            "full_total_rates_finite_positive": positivity_pass,
            "kappa_finite_positive": kappa_pass,
            "note": (
                "Mechanisms with a published zero density have zero rate and "
                "infinite individual relaxation time by construction."
            ),
            "pass": positivity_pass and kappa_pass,
        },
        "asymptotic_slopes": {
            **slopes,
            "expected": {
                "phonon_phonon": 2.0,
                "point_defect": 4.0,
                "precipitate_rayleigh": 4.0,
                "precipitate_geometric": 0.0,
            },
            "pass": (
                abs(slopes["phonon_phonon_low_frequency"] - 2.0) < 1.0e-10
                and abs(slopes["point_defect_low_frequency"] - 4.0) < 1.0e-10
                and abs(slopes["precipitate_small_rayleigh"] - 4.0) < 1.0e-6
                and abs(slopes["precipitate_small_geometric"]) < 1.0e-6
            ),
        },
        "precipitate_continuity": {
            "crossover_omega_rad_s": crossover_omega,
            "rate_below_s-1": rate_below,
            "rate_above_s-1": rate_above,
            "relative_change_across_0.2_percent_interval": relative_difference(
                rate_below, rate_above
            ),
            "pass": relative_difference(rate_below, rate_above) < 0.01,
        },
        "determinism": {
            "in_process_numeric_signatures": repeat_signatures,
            "pass": deterministic,
        },
        "state_isolation": {
            "shared_parameters_identical": (
                configs["AQ"]["shared_parameters"]
                == configs["48h"]["shared_parameters"]
            ),
            "state_specific_keys": sorted(
                configs["AQ"]["state_parameters"].keys()
            ),
            "pass": True,
        },
        "input_isolation": {
            "PF_data_used": False,
            "Sheskin_data_used": False,
            "loaded_scientific_inputs": [
                "yu_AQ_parameters.json",
                "yu_48h_parameters.json",
                "yu_AQ_kappaL_digitized.csv (SI Table S1)",
                "yu_48h_kappaL_digitized.csv (SI Table S1)",
            ],
            "pass": True,
        },
    }


def comparison_rows(
    experiments_by_state: dict[str, list[dict[str, Any]]],
    digitized_model_by_state: dict[str, list[dict[str, Any]]],
    configs: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, float]] = {}
    for state, experiments in experiments_by_state.items():
        absolute_errors: list[float] = []
        relative_errors: list[float] = []
        squared_errors: list[float] = []
        model_absolute_errors: list[float] = []
        model_relative_errors: list[float] = []
        model_squared_errors: list[float] = []
        digitized_by_temperature = {
            model_row["T_K"]: model_row
            for model_row in digitized_model_by_state[state]
        }
        for experiment in experiments:
            digitized = digitized_by_temperature[experiment["T_K"]]
            reproduced = integrate_gauss_legendre(
                experiment["T_K"], configs[state], MECHANISMS
            )
            absolute = abs(reproduced - experiment["kappa_lat_W_mK"])
            relative = absolute / experiment["kappa_lat_W_mK"]
            absolute_errors.append(absolute)
            relative_errors.append(relative)
            squared_errors.append(absolute**2)
            model_absolute = abs(
                reproduced - digitized["yu_model_kappa_lat_W_mK"]
            )
            model_relative = (
                model_absolute / digitized["yu_model_kappa_lat_W_mK"]
            )
            model_absolute_errors.append(model_absolute)
            model_relative_errors.append(model_relative)
            model_squared_errors.append(model_absolute**2)
            rows.append(
                {
                    "state": state,
                    "T_K": f"{experiment['T_K']:.8g}",
                    "experimental_kappa_W_mK": (
                        f"{experiment['kappa_lat_W_mK']:.12g}"
                    ),
                    "experimental_source": experiment["data_origin"],
                    "yu_model_digitized_kappa_W_mK": (
                        f"{digitized['yu_model_kappa_lat_W_mK']:.12g}"
                    ),
                    "yu_model_digitization_uncertainty_W_mK": (
                        f"{digitized['digitization_uncertainty_W_mK']:.12g}"
                    ),
                    "reproduced_kappa_W_mK": f"{reproduced:.12g}",
                    "absolute_error_vs_experiment_W_mK": f"{absolute:.12g}",
                    "relative_error_vs_experiment": f"{relative:.12g}",
                    "absolute_error_vs_yu_model_W_mK": f"{model_absolute:.12g}",
                    "relative_error_vs_yu_model": f"{model_relative:.12g}",
                }
            )
        metrics[state] = {
            "experiment_MAE_W_mK": float(np.mean(absolute_errors)),
            "experiment_MAPE": float(np.mean(relative_errors)),
            "experiment_max_relative_error": float(max(relative_errors)),
            "experiment_RMSE_W_mK": float(math.sqrt(np.mean(squared_errors))),
            "yu_model_MAE_W_mK": float(np.mean(model_absolute_errors)),
            "yu_model_MAPE": float(np.mean(model_relative_errors)),
            "yu_model_max_relative_error": float(max(model_relative_errors)),
            "yu_model_RMSE_W_mK": float(
                math.sqrt(np.mean(model_squared_errors))
            ),
        }
    return rows, metrics


def plot_state(
    state: str,
    experiments: list[dict[str, Any]],
    digitized_model: list[dict[str, Any]],
    config: dict[str, Any],
    path: Path,
) -> None:
    temperatures = np.linspace(
        min(row["T_K"] for row in experiments),
        max(row["T_K"] for row in experiments),
        181,
    )
    reproduced = [
        integrate_gauss_legendre(float(t), config, MECHANISMS) for t in temperatures
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.5), dpi=160)
    ax.plot(temperatures, reproduced, color="#173b8f", lw=2.0, label="Reproduction")
    ax.scatter(
        [row["T_K"] for row in experiments],
        [row["kappa_lat_W_mK"] for row in experiments],
        color="#d62728" if state == "AQ" else "#2ca02c",
        edgecolor="black",
        linewidth=0.4,
        s=34,
        zorder=3,
        label="Yu experiment (SI Table S1)",
    )
    ax.plot(
        [row["T_K"] for row in digitized_model],
        [row["yu_model_kappa_lat_W_mK"] for row in digitized_model],
        color="#24306e",
        ls="--",
        lw=1.5,
        marker="x",
        ms=4,
        label="Yu model (Figure 6b digitized)",
    )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$\kappa_{\mathrm{lat}}$ (W m$^{-1}$ K$^{-1}$)")
    ax.set_title(f"Yu 2024 {state}: published-parameter reproduction")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": "reproduce_yu2024_debye_callaway.py"})
    plt.close(fig)


def plot_combined(
    experiments_by_state: dict[str, list[dict[str, Any]]],
    digitized_model_by_state: dict[str, list[dict[str, Any]]],
    configs: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=160)
    styles = {
        "AQ": ("#d62728", "o"),
        "48h": ("#2ca02c", "s"),
    }
    for state in ("AQ", "48h"):
        experiments = experiments_by_state[state]
        temperatures = np.linspace(
            min(row["T_K"] for row in experiments),
            max(row["T_K"] for row in experiments),
            181,
        )
        reproduced = [
            integrate_gauss_legendre(float(t), configs[state], MECHANISMS)
            for t in temperatures
        ]
        color, marker = styles[state]
        ax.plot(
            temperatures,
            reproduced,
            color=color,
            lw=2.0,
            label=f"{state} reproduction",
        )
        ax.scatter(
            [row["T_K"] for row in experiments],
            [row["kappa_lat_W_mK"] for row in experiments],
            color=color,
            edgecolor="black",
            linewidth=0.4,
            marker=marker,
            s=32,
            zorder=3,
            label=f"{state} experiment",
        )
        ax.plot(
            [row["T_K"] for row in digitized_model_by_state[state]],
            [
                row["yu_model_kappa_lat_W_mK"]
                for row in digitized_model_by_state[state]
            ],
            color=color,
            ls="--",
            lw=1.2,
            alpha=0.8,
            label=f"{state} Yu model digitized",
        )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$\kappa_{\mathrm{lat}}$ (W m$^{-1}$ K$^{-1}$)")
    ax.set_title("Yu 2024 AQ and 48 h reproduction")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": "reproduce_yu2024_debye_callaway.py"})
    plt.close(fig)


def write_validation_markdown(path: Path, validation: dict[str, Any]) -> None:
    convergence = validation["convergence"]
    lines = [
        "# Numerical validation",
        "",
        "This file is generated from the frozen JSON inputs. No fit is performed.",
        "",
        "## Summary",
        "",
        f"- Full rates finite and positive: `{validation['positivity']['full_total_rates_finite_positive']}`",
        f"- Conductivities finite and positive: `{validation['positivity']['kappa_finite_positive']}`",
        f"- Maximum 256/512-point refinement difference: `{convergence['max_grid_refinement_relative_difference']:.6e}`",
        f"- Maximum Gauss/adaptive-quad difference: `{convergence['max_gauss_vs_quad_relative_difference']:.6e}`",
        f"- Gamma-prime independent relative check: `{validation['unit_checks']['gamma_prime_relative_difference']:.6e}`",
        f"- In-process deterministic repeat: `{validation['determinism']['pass']}`",
        f"- Asymptotic rate powers pass: `{validation['asymptotic_slopes']['pass']}`",
        f"- Precipitate crossover continuity pass: `{validation['precipitate_continuity']['pass']}`",
        "",
        "## Source limitations",
        "",
        "- SI Eq. (S1) publishes only the combined Normal-plus-Umklapp rate.",
        "- The main text publishes a single relaxation-time integral; no Callaway second term is present.",
        "- Table S2 prints `m^-3` for atomic volumes `Vm` and `Vi`. The implementation uses `m^3/atom`, as required by Eq. (S15) and dimensional consistency.",
        "- A zero published defect density produces zero rate and infinite individual relaxation time; positivity is tested on the full active total rate.",
        "",
        "## Isolation",
        "",
        "- PF data used: `false`",
        "- Sheskin data used: `false`",
        "- Parameter refit: `false`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    aq_config = load_json(args.aq_config)
    h48_config = load_json(args.h48_config)
    validate_config_contract(aq_config, h48_config)
    configs = {"AQ": aq_config, "48h": h48_config}
    experiments_by_state = {
        "AQ": read_experimental(args.experimental_dir / "yu_AQ_kappaL_digitized.csv"),
        "48h": read_experimental(
            args.experimental_dir / "yu_48h_kappaL_digitized.csv"
        ),
    }
    digitized_model_by_state = {
        "AQ": read_digitized_model(
            args.experimental_dir / "yu_AQ_model_digitized.csv"
        ),
        "48h": read_digitized_model(
            args.experimental_dir / "yu_48h_model_digitized.csv"
        ),
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    budget_rows_by_state = {
        state: compute_budget(experiments_by_state[state], configs[state])
        for state in ("AQ", "48h")
    }
    for state, filename in (
        ("AQ", "AQ_scattering_budget.csv"),
        ("48h", "48h_scattering_budget.csv"),
    ):
        rows = budget_rows_by_state[state]
        write_csv(output / filename, list(rows[0].keys()), rows)

    for state, filename in (
        ("AQ", "AQ_frequency_resolved.csv"),
        ("48h", "48h_frequency_resolved.csv"),
    ):
        rows = compute_spectral_rows(experiments_by_state[state], configs[state])
        write_csv(output / filename, list(rows[0].keys()), rows)

    comparison, metrics = comparison_rows(
        experiments_by_state, digitized_model_by_state, configs
    )
    write_csv(output / "reproduction_comparison.csv", list(comparison[0]), comparison)
    validation = validate_numerics(experiments_by_state, configs)
    validation["input_hashes"] = {
        "AQ_parameters": file_sha256(args.aq_config),
        "48h_parameters": file_sha256(args.h48_config),
        "AQ_experiment": file_sha256(
            args.experimental_dir / "yu_AQ_kappaL_digitized.csv"
        ),
        "48h_experiment": file_sha256(
            args.experimental_dir / "yu_48h_kappaL_digitized.csv"
        ),
        "AQ_digitized_model": file_sha256(
            args.experimental_dir / "yu_AQ_model_digitized.csv"
        ),
        "48h_digitized_model": file_sha256(
            args.experimental_dir / "yu_48h_model_digitized.csv"
        ),
        "figure6b_digitization_contract": file_sha256(
            args.experimental_dir / "figure6b_digitization_contract.json"
        ),
        "implementation": file_sha256(Path(__file__)),
    }
    validation["experiment_comparison_metrics"] = metrics
    validation["overall_numerical_pass"] = all(
        (
            validation["unit_checks"]["pass"],
            validation["convergence"]["pass"],
            validation["positivity"]["pass"],
            validation["asymptotic_slopes"]["pass"],
            validation["precipitate_continuity"]["pass"],
            validation["determinism"]["pass"],
            validation["state_isolation"]["pass"],
            validation["input_isolation"]["pass"],
        )
    )
    (output / "numerical_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_validation_markdown(output / "numerical_validation.md", validation)
    plot_state(
        "AQ",
        experiments_by_state["AQ"],
        digitized_model_by_state["AQ"],
        aq_config,
        output / "AQ_reproduction.png",
    )
    plot_state(
        "48h",
        experiments_by_state["48h"],
        digitized_model_by_state["48h"],
        h48_config,
        output / "48h_reproduction.png",
    )
    plot_combined(
        experiments_by_state,
        digitized_model_by_state,
        configs,
        output / "combined_reproduction.png",
    )
    summary = {
        "model": "Yu 2024 published single-relaxation-time Debye integral",
        "doi": aq_config["source_doi"],
        "metrics": metrics,
        "numerical_validation_pass": validation["overall_numerical_pass"],
        "PF_data_used": False,
        "Sheskin_data_used": False,
        "refit_performed": False,
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
