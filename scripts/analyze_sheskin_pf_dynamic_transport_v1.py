#!/usr/bin/env python3
"""AQ/6 h calibration and frozen-parameter 48 h transport evaluation."""

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
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar


REPLICATES = ("A", "B", "C")
AGES = (6.0, 48.0)
MODELS = ("M0", "MI", "MS", "MIS")
GAUSS_ORDER = 512
BOX_VOLUME_M3 = 246.0**3 * 1.0e-27
ALPHA_BOUNDS = (0.1, 10.0)
REFERENCE_BACKGROUND = "central__all_small_at_reported_max"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"empty output: {path}")
    fields = list(rows[0])
    require(all(list(row) == fields for row in rows), f"schema mismatch: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def observations(path: Path, state: str) -> list[dict[str, float]]:
    selected: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (
                row["sample_state"] == state
                and row["analysis_selection"] == "SELECTED_FOR_TEMPERATURE_CURVE"
                and row["thermal_quantity_identity"] == "MEASURED_TOTAL_KAPPA"
            ):
                selected.append(
                    {
                        "temperature_K": float(row["temperature_K"]),
                        "kappa_W_mK": float(row["kappa_W_mK"]),
                        "sigma_W_mK": float(row["analysis_uncertainty_W_mK"]),
                    }
                )
    selected.sort(key=lambda row: row["temperature_K"])
    require(len(selected) == 7, f"expected seven {state} observations")
    return selected


class DynamicEngine:
    def __init__(
        self,
        transport: Any,
        config: dict[str, Any],
        populations: dict[str, dict[float, dict[str, Any]]],
        descriptors: Path,
    ) -> None:
        self.transport = transport
        self.config = config
        self.populations = populations
        self.nodes, self.weights = leggauss(GAUSS_ORDER)
        self.constants = config["physical_constants"]
        self.shared = config["shared_parameters"]
        self.v = float(self.shared["average_sound_velocity_m_s"])
        self.gamma = float(self.shared["gruneisen_gamma"])
        self.omega_D = (
            float(self.constants["k_B_J_K"])
            * float(self.shared["debye_temperature_K"])
            / float(self.constants["hbar_J_s"])
        )
        interface = read_csv(descriptors / "interface_statistics_time_series.csv")
        self.sv = {
            (row["replicate"], float(row["age_h"])): float(row["Sv_m_inv"])
            for row in interface
            if float(row["age_h"]) in AGES
        }
        spectrum = read_csv(descriptors / "strain_power_spectrum.csv")
        grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
        for row in spectrum:
            if (
                row["field"] == "epsilon_h"
                and row["direction"] == "isotropic_radial"
                and float(row["age_h"]) in AGES
            ):
                grouped[(row["replicate"], float(row["age_h"]))].append(row)
        self.spectrum: dict[tuple[str, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for key, rows in grouped.items():
            rows.sort(key=lambda row: float(row["q_low_nm_inv"]))
            q_low = np.asarray([float(row["q_low_nm_inv"]) for row in rows]) * 1.0e9
            q_high = np.asarray([float(row["q_high_nm_inv"]) for row in rows]) * 1.0e9
            # e=trace(epsilon)=3 epsilon_h, hence S_e=9 S_epsilon_h.
            trace_spectrum = 9.0 * np.asarray(
                [float(row["correlation_spectral_density_m3"]) for row in rows]
            )
            self.spectrum[key] = (q_low, q_high, trace_spectrum)
        require(
            set(self.sv) == set(self.spectrum) == {(rep, age) for rep in REPLICATES for age in AGES},
            "descriptor endpoints are incomplete",
        )
        self._cache: dict[tuple[str, float, float], dict[str, np.ndarray | float]] = {}

    def strain_rate(self, omega: np.ndarray, replicate: str, age_h: float) -> np.ndarray:
        q_low, q_high, spectral_density = self.spectrum[(replicate, age_h)]
        q_cut = np.minimum(2.0 * omega / self.v, math.pi / 1.0e-9)
        upper = np.minimum(q_high[None, :], q_cut[:, None])
        active = upper > q_low[None, :]
        integral = np.sum(
            spectral_density[None, :]
            * np.where(active, (upper**4 - q_low[None, :] ** 4) / 4.0, 0.0),
            axis=1,
        )
        return self.gamma**2 * self.v * integral / (4.0 * math.pi)

    def components(self, replicate: str, age_h: float, temperature_K: float) -> dict[str, np.ndarray | float]:
        key = (replicate, age_h, temperature_K)
        if key in self._cache:
            return self._cache[key]
        xmax = float(self.shared["debye_temperature_K"]) / temperature_K
        x = 0.5 * (self.nodes + 1.0) * xmax
        omega = (
            x
            * float(self.constants["k_B_J_K"])
            * temperature_K
            / float(self.constants["hbar_J_s"])
        )
        snapshot = self.populations[replicate][age_h]
        base = self.transport.base_scattering_rates(
            omega, temperature_K, snapshot["matrix_xAg"], self.config
        )
        density = self.transport.full_psd_precipitate_rate(
            omega,
            snapshot["radii_nm"] * 1.0e-9,
            BOX_VOLUME_M3,
            self.config,
        )
        interface_unit = (
            (2.0 / 3.0)
            * self.v
            * self.sv[(replicate, age_h)]
            * omega
            / self.omega_D
        )
        prefactor = (
            float(self.constants["k_B_J_K"])
            / (2.0 * math.pi**2 * self.v)
            * (
                float(self.constants["k_B_J_K"])
                * temperature_K
                / float(self.constants["hbar_J_s"])
            )
            ** 3
        )
        result: dict[str, np.ndarray | float] = {
            "x": x,
            "omega": omega,
            "xmax": xmax,
            "prefactor": prefactor,
            "phonon_phonon": base["phonon_phonon"],
            "boundary": base["boundary"],
            "point_defect": base["point_defect"],
            "density": density,
            "interface_unit": interface_unit,
            "strain": self.strain_rate(omega, replicate, age_h),
        }
        self._cache[key] = result
        return result

    def kappa(
        self,
        replicate: str,
        age_h: float,
        temperature_K: float,
        background_A2_s: float,
        alpha: float,
        model: str,
    ) -> float:
        c = self.components(replicate, age_h, temperature_K)
        omega = np.asarray(c["omega"])
        rate = (
            np.asarray(c["phonon_phonon"])
            + np.asarray(c["boundary"])
            + np.asarray(c["point_defect"])
            + np.asarray(c["density"])
            + background_A2_s * omega**2
        )
        if model in ("MI", "MIS"):
            rate = rate + alpha * np.asarray(c["interface_unit"])
        if model in ("MS", "MIS"):
            rate = rate + np.asarray(c["strain"])
        integrand = float(c["prefactor"]) * self.transport.bose_weight(np.asarray(c["x"])) / rate
        return float(0.5 * float(c["xmax"]) * np.sum(self.weights * integrand))

    def rates_at(
        self,
        omega: np.ndarray,
        replicate: str,
        age_h: float,
        temperature_K: float,
        background_A2_s: float,
        alpha: float,
    ) -> dict[str, np.ndarray]:
        snapshot = self.populations[replicate][age_h]
        base = self.transport.base_scattering_rates(
            omega, temperature_K, snapshot["matrix_xAg"], self.config
        )
        result = {
            "phonon_phonon": base["phonon_phonon"],
            "boundary": base["boundary"],
            "point_defect": base["point_defect"],
            "density": self.transport.full_psd_precipitate_rate(
                omega, snapshot["radii_nm"] * 1.0e-9, BOX_VOLUME_M3, self.config
            ),
            "background": background_A2_s * omega**2,
            "interface": alpha
            * (2.0 / 3.0)
            * self.v
            * self.sv[(replicate, age_h)]
            * omega
            / self.omega_D,
            "strain": self.strain_rate(omega, replicate, age_h),
        }
        result["total_M0"] = sum(result[name] for name in ("phonon_phonon", "boundary", "point_defect", "density", "background"))
        result["total_MI"] = result["total_M0"] + result["interface"]
        result["total_MS"] = result["total_M0"] + result["strain"]
        result["total_MIS"] = result["total_M0"] + result["interface"] + result["strain"]
        return result


def load_background_members(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require(manifest["status"] == "SHESKIN_BACKGROUND_IDENTIFIED", "background not identified")
    members = [member for member in manifest["members"] if member["model"] == "H2_omega2"]
    require(len(members) == 18, "expected 18 retained H2 members")
    require(len({member["case_id"] for member in members}) == 18, "background members not unique")
    return members


def calibration_phase(args: argparse.Namespace, engine: DynamicEngine, authority_provenance: dict[str, Any]) -> int:
    obs = observations(args.experimental_csv, "6h")
    backgrounds = load_background_members(args.background_manifest)
    interface_rows: list[dict[str, Any]] = []
    strain_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    calibrations: list[dict[str, Any]] = []
    for member in backgrounds:
        A2 = float(member["parameter"])

        def objective(alpha: float) -> float:
            return float(
                sum(
                    (
                        (
                            np.mean(
                                [engine.kappa(rep, 6.0, row["temperature_K"], A2, alpha, "MI") for rep in REPLICATES]
                            )
                            - row["kappa_W_mK"]
                        )
                        / row["sigma_W_mK"]
                    )
                    ** 2
                    for row in obs
                )
            )

        result = minimize_scalar(
            objective,
            bounds=ALPHA_BOUNDS,
            method="bounded",
            options={"xatol": 1.0e-12, "maxiter": 500},
        )
        require(result.success, f"interface calibration failed: {member['case_id']}")
        alpha = float(result.x)
        physical = (
            alpha > ALPHA_BOUNDS[0] * (1.0 + 1.0e-6)
            and alpha < ALPHA_BOUNDS[1] * (1.0 - 1.0e-6)
        )
        calibrations.append(
            {
                "case_id": member["case_id"],
                "background_A2_s": A2,
                "interface_alpha": alpha,
                "chi2_6h": float(result.fun),
                "parameter_physicality": "PASS_WITHIN_FROZEN_BOUNDS" if physical else "REJECT_AT_FROZEN_BOUND",
            }
        )
        for row in obs:
            temperature = row["temperature_K"]
            values_by_model: dict[str, list[float]] = {}
            for model in MODELS:
                values = [engine.kappa(rep, 6.0, temperature, A2, alpha, model) for rep in REPLICATES]
                values_by_model[model] = values
                for rep, value in zip(REPLICATES, values):
                    prediction_rows.append(
                        {
                            "background_case_id": member["case_id"],
                            "background_A2_s": A2,
                            "interface_alpha": alpha,
                            "replicate": rep,
                            "age_h": 6,
                            "temperature_K": temperature,
                            "model": model,
                            "predicted_kappa_W_mK": value,
                            "experimental_kappa_W_mK": row["kappa_W_mK"],
                            "calibration_identity": "AQ_BACKGROUND_PLUS_6H_ONLY_NO_48H",
                        }
                    )
            interface_rows.append(
                {
                    "background_case_id": member["case_id"],
                    "background_A2_s": A2,
                    "interface_alpha": alpha,
                    "alpha_lower_bound": ALPHA_BOUNDS[0],
                    "alpha_upper_bound": ALPHA_BOUNDS[1],
                    "parameter_physicality": calibrations[-1]["parameter_physicality"],
                    "temperature_K": temperature,
                    "experimental_6h_kappa_W_mK": row["kappa_W_mK"],
                    "MI_ensemble_mean_W_mK": float(np.mean(values_by_model["MI"])),
                    "MI_replicate_min_W_mK": float(np.min(values_by_model["MI"])),
                    "MI_replicate_max_W_mK": float(np.max(values_by_model["MI"])),
                    "residual_W_mK": float(np.mean(values_by_model["MI"])) - row["kappa_W_mK"],
                    "fit_data_identity": "6H_MEASURED_TOTAL_ONLY_ENSEMBLE_MEAN",
                }
            )
            strain_rows.append(
                {
                    "background_case_id": member["case_id"],
                    "background_A2_s": A2,
                    "strain_parameter": "NONE_PARAMETER_FREE",
                    "strain_gamma": engine.gamma,
                    "temperature_K": temperature,
                    "experimental_6h_kappa_W_mK": row["kappa_W_mK"],
                    "MS_ensemble_mean_W_mK": float(np.mean(values_by_model["MS"])),
                    "MS_replicate_min_W_mK": float(np.min(values_by_model["MS"])),
                    "MS_replicate_max_W_mK": float(np.max(values_by_model["MS"])),
                    "residual_W_mK": float(np.mean(values_by_model["MS"])) - row["kappa_W_mK"],
                    "calibration_identity": "NO_DYNAMIC_FIT_6H_DIAGNOSTIC_ONLY",
                }
            )

    write_csv(args.out / "interface_model_6h_calibration.csv", interface_rows)
    write_csv(args.out / "strain_model_6h_calibration.csv", strain_rows)
    write_csv(args.out / "six_hour_model_predictions.csv", prediction_rows)
    all_physical = all(row["parameter_physicality"] == "PASS_WITHIN_FROZEN_BOUNDS" for row in calibrations)
    alpha_values = [row["interface_alpha"] for row in calibrations]
    interface_audit = f"""# Interface-parameter physicality audit

- Model: `I1_spectral_transmissivity`.
- Calibration data: 6 h measured-total temperature curve only; A/B/C share one coefficient within each AQ-background member.
- Frozen bound: `{ALPHA_BOUNDS[0]} <= alpha <= {ALPHA_BOUNDS[1]}`.
- Retained AQ-background members: `{len(calibrations)}`.
- Fitted alpha range: `{min(alpha_values):.17g}` to `{max(alpha_values):.17g}`.
- All members strictly inside bounds: `{str(all_physical).upper()}`.
- 48 h data used: `FALSE`.
"""
    (args.out / "interface_parameter_physicality_audit.md").write_text(interface_audit, encoding="utf-8")
    (args.out / "strain_parameter_physicality_audit.md").write_text(
        "# Strain-parameter physicality audit\n\nThe scalar Born strain-spectrum model has no calibrated dynamic parameter. It uses the frozen Yu `gamma=1.96`, the accepted-field trace-strain spectrum, zero high-q extrapolation, and no 48 h observation. Status: `PASS_PARAMETER_FREE_SCALAR_BORN_CONTRACT`.\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    temperatures = np.asarray([row["temperature_K"] for row in obs])
    measured = np.asarray([row["kappa_W_mK"] for row in obs])
    sigma = np.asarray([row["sigma_W_mK"] for row in obs])
    ax.errorbar(temperatures, measured, yerr=sigma, fmt="ko", label="Sheskin 6 h measured total")
    for model, style in zip(MODELS, (":", "-", "--", "-.")):
        center, lower, upper = [], [], []
        for temperature in temperatures:
            member_means = []
            for calibration in calibrations:
                member_means.append(
                    np.mean(
                        [
                            engine.kappa(rep, 6.0, float(temperature), calibration["background_A2_s"], calibration["interface_alpha"], model)
                            for rep in REPLICATES
                        ]
                    )
                )
            center.append(float(np.median(member_means)))
            lower.append(float(np.min(member_means)))
            upper.append(float(np.max(member_means)))
        ax.plot(temperatures, center, style, linewidth=1.8, label=model)
        ax.fill_between(temperatures, lower, upper, alpha=0.08)
    ax.set(xlabel="Temperature (K)", ylabel=r"$\kappa$ (W m$^{-1}$ K$^{-1}$)", title="AQ-background envelope and 6 h-only dynamic calibration")
    ax.grid(alpha=0.25); ax.legend()
    fig.savefig(args.out / "6h_calibration_curve.png", dpi=200); plt.close(fig)

    hashed_inputs = {
        "transport_script": sha256(args.transport_script),
        "transport_config": sha256(args.transport_config),
        "baseline_authority_reader": sha256(args.baseline_script),
        "experimental_csv": sha256(args.experimental_csv),
        "background_manifest": sha256(args.background_manifest),
        "descriptor_manifest": sha256(args.descriptors / "descriptor_manifest.json"),
        "interface_statistics": sha256(args.descriptors / "interface_statistics_time_series.csv"),
        "strain_power_spectrum": sha256(args.descriptors / "strain_power_spectrum.csv"),
        "analysis_script": sha256(Path(__file__).resolve()),
    }
    frozen = {
        "schema": "SHESKIN_DYNAMIC_TRANSPORT_FROZEN_BEFORE_48H_V1",
        "status": "PASS_FROZEN_BEFORE_48H_PREDICTION_V1",
        "calibration_data": "AQ background plus 6 h measured-total curve only",
        "experimental_quantity_identity": "MEASURED_TOTAL_KAPPA_EFFECTIVE_DEBYE_RECONSTRUCTION_NOT_WF_LATTICE",
        "background_model": "H2_omega2_18_member_source_literal_envelope",
        "background_absolute_fit_warning": "POOR_ABSOLUTE_FIT_ALL_CANDIDATES_CHI2_P_LT_0P01",
        "interface_model": "I1_spectral_transmissivity",
        "strain_model": "S1_PF_scalar_Born_parameter_free",
        "joint_model_contract": "MIS uses the already MI-calibrated alpha unchanged plus parameter-free S1; no joint refit",
        "alpha_bounds": list(ALPHA_BOUNDS),
        "calibrations": calibrations,
        "numerics": {
            "gauss_legendre_order": GAUSS_ORDER,
            "Debye_upper_limit": "Theta_D/T",
            "strain_q_integration": "piecewise_constant_radial_S_trace_exact_q3_bin_integral",
            "q_PF_max_m_inv": math.pi / 1.0e-9,
            "zero_mode_removed": True,
            "high_q_extrapolation": "ZERO",
        },
        "frozen_material_parameters": {
            "A_N": 1.5,
            "S11": 0.0,
            "S13": 0.0,
            "average_sound_velocity_m_s": engine.v,
            "gruneisen_gamma": engine.gamma,
            "debye_temperature_K": float(engine.shared["debye_temperature_K"]),
            "Yu_dislocation_scale_used": False,
        },
        "authority_provenance": authority_provenance,
        "input_sha256": hashed_inputs,
        "read_48h_experiment_during_calibration": False,
        "all_interface_parameters_physical": all_physical,
    }
    (args.out / "frozen_before_48h_prediction_manifest.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_paths = sorted(path for path in args.out.iterdir() if path.is_file())
    with (args.out / "calibration_outputs.sha256").open("w", encoding="utf-8") as stream:
        for path in output_paths:
            stream.write(f"{sha256(path)}  {path.name}\n")
    print(frozen["status"])
    print(f"interface_alpha_range={min(alpha_values):.17g},{max(alpha_values):.17g}")
    return 0


def blind_phase(args: argparse.Namespace, engine: DynamicEngine) -> int:
    frozen = json.loads(args.frozen_manifest.read_text(encoding="utf-8"))
    require(frozen["status"] == "PASS_FROZEN_BEFORE_48H_PREDICTION_V1", "models not frozen")
    current_hashes = {
        "transport_script": sha256(args.transport_script),
        "transport_config": sha256(args.transport_config),
        "baseline_authority_reader": sha256(args.baseline_script),
        "experimental_csv": sha256(args.experimental_csv),
        "background_manifest": sha256(args.background_manifest),
        "descriptor_manifest": sha256(args.descriptors / "descriptor_manifest.json"),
        "interface_statistics": sha256(args.descriptors / "interface_statistics_time_series.csv"),
        "strain_power_spectrum": sha256(args.descriptors / "strain_power_spectrum.csv"),
        "analysis_script": sha256(Path(__file__).resolve()),
    }
    require(current_hashes == frozen["input_sha256"], "frozen input hash mismatch")
    obs6 = observations(args.experimental_csv, "6h")
    obs48 = observations(args.experimental_csv, "48h")
    obs_by_age = {6.0: {row["temperature_K"]: row for row in obs6}, 48.0: {row["temperature_K"]: row for row in obs48}}
    calibrations = frozen["calibrations"]
    rows: list[dict[str, Any]] = []
    for calibration in calibrations:
        for age_h in AGES:
            for temperature, observation in obs_by_age[age_h].items():
                for model in MODELS:
                    for rep in REPLICATES:
                        value = engine.kappa(
                            rep,
                            age_h,
                            temperature,
                            float(calibration["background_A2_s"]),
                            float(calibration["interface_alpha"]),
                            model,
                        )
                        rows.append(
                            {
                                "background_case_id": calibration["case_id"],
                                "background_A2_s": calibration["background_A2_s"],
                                "interface_alpha_frozen_from_6h": calibration["interface_alpha"],
                                "replicate": rep,
                                "age_h": age_h,
                                "temperature_K": temperature,
                                "model": model,
                                "predicted_kappa_W_mK": value,
                                "experimental_measured_total_kappa_W_mK": observation["kappa_W_mK"],
                                "prediction_identity": "FROZEN_AFTER_AQ_AND_6H_BEFORE_48H_COMPARISON",
                            }
                        )
    write_csv(args.out / "blind_48h_predictions.csv", rows)

    aggregate_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for model in MODELS:
        for temperature in sorted(obs_by_age[6.0]):
            member_values: dict[str, dict[float, list[float]]] = {}
            for calibration in calibrations:
                member_values[calibration["case_id"]] = {}
                for age_h in AGES:
                    member_values[calibration["case_id"]][age_h] = [
                        float(row["predicted_kappa_W_mK"])
                        for row in rows
                        if row["background_case_id"] == calibration["case_id"]
                        and row["model"] == model
                        and float(row["temperature_K"]) == temperature
                        and float(row["age_h"]) == age_h
                    ]
            for case_id, values in member_values.items():
                mean6, mean48 = float(np.mean(values[6.0])), float(np.mean(values[48.0]))
                aggregate_rows.append(
                    {
                        "scope": "background_member_A_B_C_ensemble",
                        "background_case_id": case_id,
                        "model": model,
                        "temperature_K": temperature,
                        "predicted_6h_mean_W_mK": mean6,
                        "predicted_48h_mean_W_mK": mean48,
                        "predicted_delta_W_mK": mean48 - mean6,
                        "predicted_relative_change_percent": 100.0 * (mean48 - mean6) / mean6,
                        "replicate_6h_min_W_mK": min(values[6.0]),
                        "replicate_6h_max_W_mK": max(values[6.0]),
                        "replicate_48h_min_W_mK": min(values[48.0]),
                        "replicate_48h_max_W_mK": max(values[48.0]),
                        "experiment_6h_W_mK": obs_by_age[6.0][temperature]["kappa_W_mK"],
                        "experiment_48h_W_mK": obs_by_age[48.0][temperature]["kappa_W_mK"],
                    }
                )
            means6 = [float(np.mean(values[6.0])) for values in member_values.values()]
            means48 = [float(np.mean(values[48.0])) for values in member_values.values()]
            aggregate_rows.append(
                {
                    "scope": "nonprobabilistic_background_envelope",
                    "background_case_id": "ALL_18_H2_MEMBERS",
                    "model": model,
                    "temperature_K": temperature,
                    "predicted_6h_mean_W_mK": float(np.median(means6)),
                    "predicted_48h_mean_W_mK": float(np.median(means48)),
                    "predicted_delta_W_mK": float(np.median(np.asarray(means48) - np.asarray(means6))),
                    "predicted_relative_change_percent": float(np.median(100.0 * (np.asarray(means48) - np.asarray(means6)) / np.asarray(means6))),
                    "replicate_6h_min_W_mK": min(means6),
                    "replicate_6h_max_W_mK": max(means6),
                    "replicate_48h_min_W_mK": min(means48),
                    "replicate_48h_max_W_mK": max(means48),
                    "experiment_6h_W_mK": obs_by_age[6.0][temperature]["kappa_W_mK"],
                    "experiment_48h_W_mK": obs_by_age[48.0][temperature]["kappa_W_mK"],
                }
            )
    write_csv(args.out / "full_temperature_comparison.csv", aggregate_rows)

    focus_temperature = 573.15
    for model in MODELS:
        for calibration in calibrations:
            selected = [row for row in aggregate_rows if row["scope"] == "background_member_A_B_C_ensemble" and row["background_case_id"] == calibration["case_id"] and row["model"] == model]
            focus = next(row for row in selected if float(row["temperature_K"]) == focus_temperature)
            trends = []
            for rep in REPLICATES:
                value6 = next(float(row["predicted_kappa_W_mK"]) for row in rows if row["background_case_id"] == calibration["case_id"] and row["model"] == model and row["replicate"] == rep and float(row["age_h"]) == 6.0 and float(row["temperature_K"]) == focus_temperature)
                value48 = next(float(row["predicted_kappa_W_mK"]) for row in rows if row["background_case_id"] == calibration["case_id"] and row["model"] == model and row["replicate"] == rep and float(row["age_h"]) == 48.0 and float(row["temperature_K"]) == focus_temperature)
                trends.append(value48 > value6)
            temperature_trends = [float(row["predicted_48h_mean_W_mK"]) > float(row["predicted_6h_mean_W_mK"]) for row in selected]
            relative = float(focus["predicted_relative_change_percent"])
            endpoint_error = abs(float(focus["predicted_48h_mean_W_mK"]) - 1.03) / 1.03
            gate_rows.append(
                {
                    "model": model,
                    "background_case_id": calibration["case_id"],
                    "trend_mean_positive": float(focus["predicted_48h_mean_W_mK"]) > float(focus["predicted_6h_mean_W_mK"]),
                    "replicates_positive_count": sum(trends),
                    "trend_gate_pass": sum(trends) >= 2 and float(focus["predicted_48h_mean_W_mK"]) > float(focus["predicted_6h_mean_W_mK"]),
                    "relative_change_percent_573p15K": relative,
                    "relative_change_gate_10_to_30_percent": 10.0 <= relative <= 30.0,
                    "predicted_48h_573p15K_W_mK": focus["predicted_48h_mean_W_mK"],
                    "endpoint_relative_error": endpoint_error,
                    "endpoint_gate_10_percent": endpoint_error <= 0.10,
                    "endpoint_ideal_gate_5_percent": endpoint_error <= 0.05,
                    "full_temperature_positive_count": sum(temperature_trends),
                    "full_temperature_majority_pass": sum(temperature_trends) >= 4,
                    "parameter_physicality": calibration["parameter_physicality"] if model in ("MI", "MIS") else "PASS_NO_DYNAMIC_PARAMETER",
                }
            )
    write_csv(args.out / "blind_prediction_acceptance_gates.csv", gate_rows)

    # Spectral outputs use a pre-registered representative background member;
    # prediction/acceptance tables above retain the complete background envelope.
    reference = next(row for row in calibrations if row["case_id"] == REFERENCE_BACKGROUND)
    omega = np.geomspace(engine.omega_D * 1.0e-5, engine.omega_D, 256)
    decomposition_rows: list[dict[str, Any]] = []
    for age_h in AGES:
        per_rep = {rep: engine.rates_at(omega, rep, age_h, 573.15, float(reference["background_A2_s"]), float(reference["interface_alpha"])) for rep in REPLICATES}
        names = list(next(iter(per_rep.values())))
        for index, value in enumerate(omega):
            means = {name: float(np.mean([per_rep[rep][name][index] for rep in REPLICATES])) for name in names}
            dynamic_channels = {name: means[name] for name in ("phonon_phonon", "background", "point_defect", "density", "interface", "strain", "boundary")}
            dominant = max(dynamic_channels, key=dynamic_channels.get)
            decomposition_rows.append(
                {
                    "background_case_id": REFERENCE_BACKGROUND,
                    "scope": "A_B_C_ensemble_mean",
                    "age_h": age_h,
                    "temperature_K": 573.15,
                    "omega_rad_s": value,
                    "frequency_THz": value / (2.0 * math.pi * 1.0e12),
                    "tau_host_inverse_s-1": means["phonon_phonon"] + means["boundary"],
                    "tau_background_inverse_s-1": means["background"],
                    "tau_point_defect_inverse_s-1": means["point_defect"],
                    "tau_density_inverse_s-1": means["density"],
                    "tau_interface_inverse_s-1": means["interface"],
                    "tau_strain_inverse_s-1": means["strain"],
                    "tau_total_M0_inverse_s-1": means["total_M0"],
                    "tau_total_MI_inverse_s-1": means["total_MI"],
                    "tau_total_MS_inverse_s-1": means["total_MS"],
                    "tau_total_MIS_inverse_s-1": means["total_MIS"],
                    "dominant_scattering_channel": dominant,
                }
            )
    write_csv(args.out / "scattering_rate_decomposition.csv", decomposition_rows)

    cumulative_rows: list[dict[str, Any]] = []
    temperature = 573.15
    xmax = float(engine.shared["debye_temperature_K"]) / temperature
    x_dense = np.linspace(1.0e-10, xmax, 2048)
    omega_dense = x_dense * float(engine.constants["k_B_J_K"]) * temperature / float(engine.constants["hbar_J_s"])
    prefactor = float(engine.constants["k_B_J_K"]) / (2.0 * math.pi**2 * engine.v) * (float(engine.constants["k_B_J_K"]) * temperature / float(engine.constants["hbar_J_s"])) ** 3
    for age_h in AGES:
        per_rep_rates = {rep: engine.rates_at(omega_dense, rep, age_h, temperature, float(reference["background_A2_s"]), float(reference["interface_alpha"])) for rep in REPLICATES}
        for model in MODELS:
            key = f"total_{model}"
            spectra = [prefactor * engine.transport.bose_weight(x_dense) / per_rep_rates[rep][key] for rep in REPLICATES]
            spectral_mean = np.mean(spectra, axis=0)
            cumulative = cumulative_trapezoid(spectral_mean, x_dense, initial=0.0)
            for index in range(x_dense.size):
                cumulative_rows.append(
                    {
                        "background_case_id": REFERENCE_BACKGROUND,
                        "scope": "A_B_C_ensemble_mean",
                        "age_h": age_h,
                        "temperature_K": temperature,
                        "model": model,
                        "x_hbaromega_over_kBT": x_dense[index],
                        "omega_rad_s": omega_dense[index],
                        "frequency_THz": omega_dense[index] / (2.0 * math.pi * 1.0e12),
                        "spectral_kappa_density_dkappa_dx": spectral_mean[index],
                        "cumulative_kappa_W_mK": cumulative[index],
                    }
                )
    write_csv(args.out / "cumulative_kappa_spectrum.csv", cumulative_rows)

    envelope = [row for row in aggregate_rows if row["scope"] == "nonprobabilistic_background_envelope"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for model, marker in zip(MODELS, ("o", "s", "^", "D")):
        selected = sorted([row for row in envelope if row["model"] == model], key=lambda row: float(row["temperature_K"]))
        t = np.asarray([float(row["temperature_K"]) for row in selected])
        ax.plot(t, [float(row["predicted_6h_mean_W_mK"]) for row in selected], marker + "--", label=f"{model} 6 h")
        ax.plot(t, [float(row["predicted_48h_mean_W_mK"]) for row in selected], marker + "-", label=f"{model} 48 h")
    ax.plot([row["temperature_K"] for row in obs6], [row["kappa_W_mK"] for row in obs6], "k+--", label="experiment 6 h")
    ax.plot([row["temperature_K"] for row in obs48], [row["kappa_W_mK"] for row in obs48], "kx-", label="experiment 48 h")
    ax.set(xlabel="Temperature (K)", ylabel=r"$\kappa$ (W m$^{-1}$ K$^{-1}$)", title="Frozen-coefficient 48 h comparison")
    ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=2)
    fig.savefig(args.out / "6h_48h_kappa_comparison.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    for model in MODELS:
        selected = sorted([row for row in envelope if row["model"] == model], key=lambda row: float(row["temperature_K"]))
        ax.plot([float(row["temperature_K"]) for row in selected], [float(row["predicted_delta_W_mK"]) for row in selected], "o-", label=model)
    ax.plot([row["temperature_K"] for row in obs6], [b["kappa_W_mK"] - a["kappa_W_mK"] for a, b in zip(obs6, obs48)], "kx--", label="experiment")
    ax.axhline(0.0, color="0.4", linewidth=1)
    ax.set(xlabel="Temperature (K)", ylabel=r"$\Delta\kappa_{48-6}$ (W m$^{-1}$ K$^{-1}$)", title="Blind recovery after coefficient freeze")
    ax.grid(alpha=0.25); ax.legend()
    fig.savefig(args.out / "delta_kappa_vs_temperature.png", dpi=200); plt.close(fig)

    for age_h in AGES:
        fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
        selected = [row for row in decomposition_rows if float(row["age_h"]) == age_h]
        for field, label in (
            ("tau_host_inverse_s-1", "host+boundary"),
            ("tau_background_inverse_s-1", "Sheskin background"),
            ("tau_point_defect_inverse_s-1", "point defect"),
            ("tau_density_inverse_s-1", "PF density"),
            ("tau_interface_inverse_s-1", "interface"),
            ("tau_strain_inverse_s-1", "strain"),
        ):
            ax.loglog([float(row["frequency_THz"]) for row in selected], [float(row[field]) for row in selected], label=label)
        ax.set(xlabel="Frequency (THz)", ylabel=r"$\tau^{-1}$ (s$^{-1}$)", title=f"Scattering-rate decomposition, {age_h:g} h at 573.15 K")
        ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8)
        fig.savefig(args.out / f"scattering_rate_decomposition_{int(age_h)}h.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    for age_h, style in ((6.0, "--"), (48.0, "-")):
        for model in MODELS:
            selected = [row for row in cumulative_rows if float(row["age_h"]) == age_h and row["model"] == model]
            ax.plot([float(row["frequency_THz"]) for row in selected], [float(row["cumulative_kappa_W_mK"]) for row in selected], style, label=f"{model} {age_h:g} h")
    ax.set(xlabel="Frequency (THz)", ylabel=r"Cumulative $\kappa$ (W m$^{-1}$ K$^{-1}$)", title="Frozen-model cumulative conductivity at 573.15 K")
    ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=2)
    fig.savefig(args.out / "cumulative_kappa_6h_48h.png", dpi=200); plt.close(fig)

    model_summary: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        selected = [row for row in gate_rows if row["model"] == model]
        model_summary[model] = {
            "background_member_count": len(selected),
            "all_trend_gates_pass": all(str(row["trend_gate_pass"]) == "True" for row in selected),
            "all_relative_change_gates_pass": all(str(row["relative_change_gate_10_to_30_percent"]) == "True" for row in selected),
            "all_endpoint_10pct_gates_pass": all(str(row["endpoint_gate_10_percent"]) == "True" for row in selected),
            "all_full_temperature_majority_pass": all(str(row["full_temperature_majority_pass"]) == "True" for row in selected),
            "all_parameter_physicality_pass": all(not str(row["parameter_physicality"]).startswith("REJECT") for row in selected),
        }
        model_summary[model]["all_registered_gates_pass"] = all(model_summary[model][key] for key in ("all_trend_gates_pass", "all_relative_change_gates_pass", "all_endpoint_10pct_gates_pass", "all_full_temperature_majority_pass", "all_parameter_physicality_pass"))
    if model_summary["MI"]["all_registered_gates_pass"]:
        final_status = "PASS_INTERFACE_SCATTERING_BLIND_48H_PREDICTION"
    elif model_summary["MS"]["all_registered_gates_pass"]:
        final_status = "PASS_STRAIN_SCATTERING_BLIND_48H_PREDICTION"
    elif any(
        model_summary[model]["all_trend_gates_pass"]
        and model_summary[model]["all_relative_change_gates_pass"]
        and model_summary[model]["all_parameter_physicality_pass"]
        for model in ("MI", "MS")
    ):
        final_status = "PASS_INTERFACE_OR_STRAIN_SEMIQUANTITATIVE_RECONSTRUCTION"
    else:
        final_status = "FAIL_RESOLVED_ONLY_AFTER_INTERFACE_AND_STRAIN_TEST"
    summary = {
        "schema": "SHESKIN_FROZEN_48H_BLIND_EVALUATION_V1",
        "status": final_status,
        "frozen_manifest_sha256": sha256(args.frozen_manifest),
        "model_gates": model_summary,
        "background_envelope_is_nonprobabilistic": True,
        "no_48h_refit": True,
        "absolute_experimental_lattice_reproduction_claimed": False,
    }
    (args.out / "blind_evaluation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_paths = sorted(path for path in args.out.iterdir() if path.is_file())
    with (args.out / "blind_outputs.sha256").open("w", encoding="utf-8") as stream:
        for path in output_paths:
            stream.write(f"{sha256(path)}  {path.name}\n")
    print(final_status)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("calibrate", "blind"), required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--transport-script", type=Path, required=True)
    parser.add_argument("--transport-config", type=Path, required=True)
    parser.add_argument("--baseline-script", type=Path, required=True)
    parser.add_argument("--experimental-csv", type=Path, required=True)
    parser.add_argument("--background-manifest", type=Path, required=True)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)
    transport = load_module(args.transport_script, "sheskin_dynamic_transport_core")
    baseline = load_module(args.baseline_script, "sheskin_dynamic_authority_reader")
    config = json.loads(args.transport_config.read_text(encoding="utf-8"))
    transport.validate_yu_base_config(config, args.transport_config)
    populations, authority_provenance, _ = baseline.collect_authority(args.authority_root)
    engine = DynamicEngine(transport, config, populations, args.descriptors)
    if args.phase == "calibrate":
        require(args.frozen_manifest is None, "calibration must not receive a frozen manifest")
        return calibration_phase(args, engine, authority_provenance)
    require(args.frozen_manifest is not None and args.frozen_manifest.is_file(), "blind phase requires frozen manifest")
    return blind_phase(args, engine)


if __name__ == "__main__":
    raise SystemExit(main())
