#!/usr/bin/env python3
"""Fit the pre-registered one-parameter Sheskin AQ background candidates.

Only selected AQ measured-total kappa values are used.  The Sheskin AQ object
data do not define a unique PSD, so the calculation propagates a finite,
source-literal envelope: the reported TEM/APT density interval, a zero-size
particle-scattering limit, the reported <=10 nm small-object diameter bound,
and the single-reconstruction 37:6 small/elongated count diagnostic.  Yu AQ
particle populations are never imported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar
from scipy.stats import chi2 as chi2_distribution


SCHEMA = "SHESKIN_AQ_BACKGROUND_FIT_V1"
MODELS = ("H1_constant", "H2_omega2", "H3_omega4", "H4_host_scale")
GAUSS_ORDER = 512
AQ_XAG = 0.0078
AQ_NV_CENTRAL = 3.61e21
AQ_NV_SIGMA = 1.51e21
COUNT_SMALL = 37
COUNT_LARGE = 6
COUNT_TOTAL = COUNT_SMALL + COUNT_LARGE


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    require(bool(values), f"refusing to write empty CSV: {path}")
    fields = list(values[0])
    require(all(list(row) == fields for row in values), f"CSV schema mismatch: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("sheskin_aq_transport", path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def population_cases() -> list[dict[str, Any]]:
    density_levels = (
        ("lower_1sigma", AQ_NV_CENTRAL - AQ_NV_SIGMA),
        ("central", AQ_NV_CENTRAL),
        ("upper_1sigma", AQ_NV_CENTRAL + AQ_NV_SIGMA),
    )
    small_fraction = COUNT_SMALL / COUNT_TOTAL
    large_fraction = COUNT_LARGE / COUNT_TOTAL
    morphologies: list[tuple[str, list[tuple[float, float]], str]] = [
        (
            "zero_particle_scattering_limit",
            [],
            "Missing lower size bound represented by R->0; mathematical envelope limit.",
        ),
        (
            "all_small_at_reported_max",
            [(1.0, 5.0)],
            "All TEM/APT objects assigned the reported small-object maximum radius 5 nm.",
        ),
    ]
    for aspect_ratio in (10.0, 5.0, 2.0):
        large_radius = 50.0 / aspect_ratio ** (2.0 / 3.0)
        morphologies.append(
            (
                f"single_APT_37to6_prolate_AR{aspect_ratio:g}",
                [(small_fraction, 5.0), (large_fraction, large_radius)],
                "Single-reconstruction 37:6 count diagnostic; small radius 5 nm; "
                f"100 nm long prolate large object with AR={aspect_ratio:g} and volume-equivalent radius.",
            )
        )
    morphologies.append(
        (
            "single_APT_37to6_large_Lover2_sphere_bound",
            [(small_fraction, 5.0), (large_fraction, 50.0)],
            "Deliberately conservative upper bound treating 100 nm length as a 50 nm sphere radius.",
        )
    )
    cases: list[dict[str, Any]] = []
    for density_label, nv in density_levels:
        require(nv > 0.0, "AQ density lower bound must be positive")
        for morphology, populations, note in morphologies:
            cases.append(
                {
                    "case_id": f"{density_label}__{morphology}",
                    "density_level": density_label,
                    "total_Nv_m-3": nv,
                    "morphology": morphology,
                    "populations": populations,
                    "source_status": (
                        "SOURCE_LITERAL_BOUND"
                        if morphology in {"zero_particle_scattering_limit", "all_small_at_reported_max"}
                        else "SINGLE_APT_COUNT_FRACTION_DIAGNOSTIC"
                    ),
                    "notes": note,
                }
            )
    return cases


class Evaluator:
    def __init__(self, transport: Any, config: dict[str, Any]):
        self.transport = transport
        self.config = config
        self.nodes, self.weights = leggauss(GAUSS_ORDER)

    def particle_rate(self, omega: np.ndarray, case: dict[str, Any]) -> np.ndarray:
        result = np.zeros_like(omega)
        velocity = float(self.config["shared_parameters"]["average_sound_velocity_m_s"])
        for fraction, radius_nm in case["populations"]:
            cross = self.transport.precipitate_cross_section(
                omega, np.asarray([radius_nm * 1.0e-9]), self.config
            )[:, 0]
            result += velocity * float(case["total_Nv_m-3"]) * fraction * cross
        return result

    def kappa(self, temperature_K: float, case: dict[str, Any], model: str,
              parameter: float) -> float:
        constants = self.config["physical_constants"]
        shared = self.config["shared_parameters"]
        xmax = float(shared["debye_temperature_K"]) / temperature_K
        x = 0.5 * (self.nodes + 1.0) * xmax
        omega = x * float(constants["k_B_J_K"]) * temperature_K / float(constants["hbar_J_s"])
        base = self.transport.base_scattering_rates(omega, temperature_K, AQ_XAG, self.config)
        host = base["phonon_phonon"]
        rate = base["boundary"] + base["point_defect"] + self.particle_rate(omega, case)
        if model == "H1_constant":
            rate = rate + host + parameter
        elif model == "H2_omega2":
            rate = rate + host + parameter * omega**2
        elif model == "H3_omega4":
            rate = rate + host + parameter * omega**4
        elif model == "H4_host_scale":
            rate = rate + parameter * host
        else:
            raise ValueError(model)
        prefactor = (
            float(constants["k_B_J_K"])
            / (2.0 * math.pi**2 * float(shared["average_sound_velocity_m_s"]))
            * (float(constants["k_B_J_K"]) * temperature_K / float(constants["hbar_J_s"])) ** 3
        )
        values = prefactor * self.transport.bose_weight(x) / rate
        return float(0.5 * xmax * np.sum(self.weights * values))


def parameterization(model: str) -> tuple[tuple[float, float], Callable[[float], float], str]:
    if model == "H1_constant":
        return (-2.0, 16.0), lambda z: 10.0**z, "s^-1"
    if model == "H2_omega2":
        return (-32.0, -6.0), lambda z: 10.0**z, "s"
    if model == "H3_omega4":
        return (-58.0, -24.0), lambda z: 10.0**z, "s^3"
    if model == "H4_host_scale":
        return (0.0, 4.0), lambda z: 10.0**z, "dimensionless"
    raise ValueError(model)


def fit_model(evaluator: Evaluator, case: dict[str, Any], model: str,
              observations: list[dict[str, float]]) -> dict[str, Any]:
    bounds, decode, unit = parameterization(model)

    def objective(encoded: float) -> float:
        parameter = decode(encoded)
        return float(
            sum(
                ((evaluator.kappa(item["temperature_K"], case, model, parameter) - item["kappa_W_mK"]) / item["sigma_W_mK"]) ** 2
                for item in observations
            )
        )

    result = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1.0e-12, "maxiter": 500})
    require(result.success, f"fit failed: {case['case_id']} {model}: {result.message}")
    parameter = decode(float(result.x))
    prediction = np.asarray(
        [evaluator.kappa(item["temperature_K"], case, model, parameter) for item in observations]
    )
    measured = np.asarray([item["kappa_W_mK"] for item in observations])
    residual = prediction - measured
    chi2 = float(result.fun)
    n = len(observations)
    k = 1
    aicc = chi2 + 2 * k + 2 * k * (k + 1) / (n - k - 1)
    degrees_of_freedom = n - k
    return {
        "parameter": parameter,
        "parameter_unit": unit,
        "chi2": chi2,
        "AICc": aicc,
        "degrees_of_freedom": degrees_of_freedom,
        "chi2_survival_probability": float(
            chi2_distribution.sf(chi2, degrees_of_freedom)
        ),
        "RMSE_W_mK": float(np.sqrt(np.mean(residual**2))),
        "MAPE_percent": float(100.0 * np.mean(np.abs(residual / measured))),
        "max_abs_residual_W_mK": float(np.max(np.abs(residual))),
        "prediction": prediction,
        "encoded_at_bound": bool(abs(result.x - bounds[0]) < 1.0e-6 or abs(result.x - bounds[1]) < 1.0e-6),
    }


def load_aq_observations(path: Path) -> list[dict[str, float]]:
    rows = read_csv(path)
    selected = [
        row for row in rows
        if row["sample_state"] == "AQ"
        and row["analysis_selection"] == "SELECTED_FOR_TEMPERATURE_CURVE"
        and row["thermal_quantity_identity"] == "MEASURED_TOTAL_KAPPA"
    ]
    require(len(selected) == 7, f"expected seven selected AQ points, found {len(selected)}")
    observations = [
        {
            "temperature_K": float(row["temperature_K"]),
            "kappa_W_mK": float(row["kappa_W_mK"]),
            "sigma_W_mK": float(row["analysis_uncertainty_W_mK"]),
        }
        for row in selected
    ]
    observations.sort(key=lambda item: item["temperature_K"])
    return observations


def save_plot(path: Path, observations: list[dict[str, float]], cases: list[dict[str, Any]],
              fits: list[dict[str, Any]], evaluator: Evaluator) -> None:
    temperatures = np.asarray([item["temperature_K"] for item in observations])
    measured = np.asarray([item["kappa_W_mK"] for item in observations])
    sigma = np.asarray([item["sigma_W_mK"] for item in observations])
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.errorbar(temperatures, measured, yerr=sigma, fmt="o", color="black", label="Sheskin AQ measured total")
    best_by_model: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        subset = [item for item in fits if item["model"] == model]
        best_by_model[model] = min(subset, key=lambda item: item["AICc"])
    case_by_id = {case["case_id"]: case for case in cases}
    styles = {"H1_constant": "-", "H2_omega2": "--", "H3_omega4": "-.", "H4_host_scale": ":"}
    for model in MODELS:
        best = best_by_model[model]
        case = case_by_id[best["case_id"]]
        pred = [evaluator.kappa(t, case, model, best["parameter"]) for t in temperatures]
        ax.plot(temperatures, pred, styles[model], linewidth=1.8, label=f"{model} (AQ-only)")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$\kappa$ (W m$^{-1}$ K$^{-1}$)")
    ax.set_title("Sheskin AQ one-parameter background candidates")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180, metadata={"Software": "CUDA_STO_PF", "Creation Time": "2026-08-02"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--experimental-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    experimental_csv = args.experimental_csv.resolve()
    out = args.out.resolve()
    require(experimental_csv.is_file(), f"missing experimental data: {experimental_csv}")
    require(not out.exists(), f"refusing to overwrite output root: {out}")
    out.mkdir(parents=True)

    transport_path = root / "scripts/pf_full_psd_no_dislocation_transport_v1.py"
    config_path = root / "data/qualification/yu2024_transport_v1/yu_48h_parameters.json"
    contract_path = root / "data/qualification/pf_full_psd_no_dislocation_transport_v1/transport_parameter_contract.json"
    transport = load_module(transport_path)
    config = transport.load_json(config_path)
    contract = transport.load_json(contract_path)
    transport.validate_yu_base_config(config, config_path)
    transport.validate_interface_contract(contract, contract_path, config_path)
    require(float(config["shared_parameters"]["A_N"]) == 1.5, "A_N is not frozen")

    observations = load_aq_observations(experimental_csv)
    cases = population_cases()
    evaluator = Evaluator(transport, config)

    known_rows: list[dict[str, Any]] = []
    for case in cases:
        for item in observations:
            host_only = evaluator.kappa(item["temperature_K"], case, "H1_constant", 0.0)
            zero_case = dict(case)
            zero_case["populations"] = []
            without_particle = evaluator.kappa(item["temperature_K"], zero_case, "H1_constant", 0.0)
            known_rows.append(
                {
                    "case_id": case["case_id"],
                    "density_level": case["density_level"],
                    "total_Nv_m-3": case["total_Nv_m-3"],
                    "morphology": case["morphology"],
                    "source_status": case["source_status"],
                    "temperature_K": item["temperature_K"],
                    "matrix_xAg": AQ_XAG,
                    "measured_total_kappa_W_mK": item["kappa_W_mK"],
                    "known_channel_kappa_W_mK": host_only,
                    "known_channel_without_AQ_particles_kappa_W_mK": without_particle,
                    "AQ_particle_delta_kappa_W_mK": host_only - without_particle,
                    "notes": case["notes"],
                }
            )
    write_csv(out / "sheskin_AQ_known_channel_kappa.csv", known_rows)

    fit_records: list[dict[str, Any]] = []
    fit_internal: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []
    for case in cases:
        for model in MODELS:
            fit = fit_model(evaluator, case, model, observations)
            record = {
                "case_id": case["case_id"],
                "density_level": case["density_level"],
                "total_Nv_m-3": case["total_Nv_m-3"],
                "morphology": case["morphology"],
                "source_status": case["source_status"],
                "model": model,
                "parameter": fit["parameter"],
                "parameter_unit": fit["parameter_unit"],
                "chi2": fit["chi2"],
                "AICc": fit["AICc"],
                "degrees_of_freedom": fit["degrees_of_freedom"],
                "chi2_survival_probability": fit["chi2_survival_probability"],
                "RMSE_W_mK": fit["RMSE_W_mK"],
                "MAPE_percent": fit["MAPE_percent"],
                "max_abs_residual_W_mK": fit["max_abs_residual_W_mK"],
                "parameter_physicality": "PASS_NONNEGATIVE_EMPIRICAL_NO_INDEPENDENT_UPPER_BOUND" if not fit["encoded_at_bound"] else "FAIL_OPTIMIZER_BOUND",
                "fit_data_identity": "AQ_MEASURED_TOTAL_ONLY_NO_6H_NO_48H",
            }
            fit_records.append(record)
            fit_internal.append({**record, "prediction": fit["prediction"]})
            for item, predicted in zip(observations, fit["prediction"]):
                prediction_rows.append(
                    {
                        "case_id": case["case_id"],
                        "model": model,
                        "temperature_K": item["temperature_K"],
                        "measured_total_kappa_W_mK": item["kappa_W_mK"],
                        "analysis_uncertainty_W_mK": item["sigma_W_mK"],
                        "predicted_kappa_W_mK": float(predicted),
                        "residual_W_mK": float(predicted) - item["kappa_W_mK"],
                        "standardized_residual": (
                            float(predicted) - item["kappa_W_mK"]
                        )
                        / item["sigma_W_mK"],
                        "fit_data_identity": "AQ_ONLY_NO_6H_NO_48H",
                    }
                )
            for held_index, held in enumerate(observations):
                training = [item for index, item in enumerate(observations) if index != held_index]
                loto = fit_model(evaluator, case, model, training)
                predicted = evaluator.kappa(held["temperature_K"], case, model, loto["parameter"])
                cv_rows.append(
                    {
                        "case_id": case["case_id"],
                        "model": model,
                        "held_out_temperature_K": held["temperature_K"],
                        "training_points": len(training),
                        "fitted_parameter": loto["parameter"],
                        "parameter_unit": loto["parameter_unit"],
                        "measured_total_kappa_W_mK": held["kappa_W_mK"],
                        "predicted_kappa_W_mK": predicted,
                        "residual_W_mK": predicted - held["kappa_W_mK"],
                        "absolute_percentage_error": abs(predicted - held["kappa_W_mK"]) / held["kappa_W_mK"] * 100.0,
                        "fit_data_identity": "AQ_LOTO_ONLY_NO_6H_NO_48H",
                    }
                )
    write_csv(out / "sheskin_background_candidate_fits.csv", fit_records)
    write_csv(out / "sheskin_background_fit_predictions.csv", prediction_rows)
    write_csv(out / "sheskin_background_cross_validation.csv", cv_rows)

    summary: list[dict[str, Any]] = []
    for model in MODELS:
        subset = [item for item in fit_records if item["model"] == model]
        cv_subset = [item for item in cv_rows if item["model"] == model]
        summary.append(
            {
                "model": model,
                "case_count": len(subset),
                "AICc_min": min(item["AICc"] for item in subset),
                "AICc_median": float(np.median([item["AICc"] for item in subset])),
                "AICc_max": max(item["AICc"] for item in subset),
                "chi2_survival_probability_max": max(
                    item["chi2_survival_probability"] for item in subset
                ),
                "AQ_MAPE_percent_min": min(item["MAPE_percent"] for item in subset),
                "AQ_MAPE_percent_median": float(np.median([item["MAPE_percent"] for item in subset])),
                "AQ_MAPE_percent_max": max(item["MAPE_percent"] for item in subset),
                "LOTO_MAPE_percent": float(np.mean([item["absolute_percentage_error"] for item in cv_subset])),
                "parameter_min": min(item["parameter"] for item in subset),
                "parameter_max": max(item["parameter"] for item in subset),
                "parameter_unit": subset[0]["parameter_unit"],
            }
        )
    write_csv(out / "sheskin_background_model_summary.csv", summary)

    # A model must be stable against the source-literal AQ envelope and win by
    # Delta-AICc > 2 to be called uniquely identified.  Otherwise all models
    # within 2 of the best AICc in any admissible case are propagated.
    winners: set[str] = set()
    retained: set[str] = set()
    for case in cases:
        subset = [item for item in fit_records if item["case_id"] == case["case_id"]]
        best = min(item["AICc"] for item in subset)
        winners.add(min(subset, key=lambda item: item["AICc"])["model"])
        retained.update(item["model"] for item in subset if item["AICc"] - best <= 2.0)
    unique = len(winners) == 1 and len(retained) == 1
    background_status = "SHESKIN_BACKGROUND_IDENTIFIED" if unique else "SHESKIN_BACKGROUND_ENVELOPE_ONLY"
    absolute_fit_status = (
        "PASS_CHI2_P_GE_0P01_IN_AT_LEAST_ONE_ENVELOPE_CASE"
        if any(item["chi2_survival_probability"] >= 0.01 for item in fit_records)
        else "POOR_ABSOLUTE_FIT_ALL_CANDIDATES_CHI2_P_LT_0P01"
    )

    manifest_members = [
        {
            "case_id": item["case_id"],
            "model": item["model"],
            "parameter": item["parameter"],
            "parameter_unit": item["parameter_unit"],
            "AICc": item["AICc"],
            "AQ_MAPE_percent": item["MAPE_percent"],
        }
        for item in fit_records
        if item["model"] in retained
    ]
    uncertainty = {
        "schema": SCHEMA,
        "status": background_status,
        "absolute_fit_status": absolute_fit_status,
        "selection_data": "AQ measured-total temperature curve only",
        "A_N": 1.5,
        "S11_rate": 0.0,
        "S13_rate": 0.0,
        "yu_refit_scale_used": False,
        "Sheskin_AQ_matrix_xAg": AQ_XAG,
        "AQ_particle_population": "Sheskin source-literal envelope; Yu AQ populations forbidden",
        "winning_models_by_case": sorted(winners),
        "retained_models_delta_AICc_le_2_in_any_case": sorted(retained),
        "members": manifest_members,
        "source_hashes": {
            "experimental_csv": sha256(experimental_csv),
            "transport_script": sha256(transport_path),
            "Yu_parameter_config": sha256(config_path),
            "transport_parameter_contract": sha256(contract_path),
            "analysis_script": sha256(Path(__file__).resolve()),
        },
    }
    (out / "sheskin_background_uncertainty_manifest.json").write_text(
        json.dumps(uncertainty, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    save_plot(out / "AQ_background_fit.png", observations, cases, fit_internal, evaluator)

    summary_lines = [
        "# Sheskin AQ background model selection",
        "",
        f"Status: `{background_status}`.",
        "",
        "Only the seven selected AQ measured-total points were used. No 6 h or 48 h value was loaded by the fitter. Each candidate has exactly one common, temperature-independent parameter.",
        f"Absolute goodness-of-fit audit: `{absolute_fit_status}`. AICc identifies the best member of the pre-registered candidate set; it does not by itself establish that the winning one-parameter curve explains residuals at the assigned 0.015 W m^-1 K^-1 digitization scale.",
        "",
        "The Sheskin AQ measurements do not define a unique PSD. The retained envelope uses the Table 1 TEM/APT density at central and +/-1 sigma values, the reported <=10 nm small-object diameter bound, and the 37:6 small/elongated count only as a single-reconstruction diagnostic. SE/FIB density is not added. Yu AQ particle populations are not used.",
        "",
        "## Candidate summary",
        "",
        "| model | AICc min/median/max | AQ MAPE min/median/max (%) | LOTO MAPE (%) |",
        "|---|---:|---:|---:|",
    ]
    for item in summary:
        summary_lines.append(
            f"| {item['model']} | {item['AICc_min']:.3f} / {item['AICc_median']:.3f} / {item['AICc_max']:.3f} | "
            f"{item['AQ_MAPE_percent_min']:.3f} / {item['AQ_MAPE_percent_median']:.3f} / {item['AQ_MAPE_percent_max']:.3f} | {item['LOTO_MAPE_percent']:.3f} |"
        )
    summary_lines.extend(
        [
            "",
            f"Winning models across AQ structure cases: `{', '.join(sorted(winners))}`.",
            f"Models retained by the pre-registered Delta-AICc <= 2 rule in at least one admissible case: `{', '.join(sorted(retained))}`.",
            "",
            "H1-H3 are empirical frequency forms and are not assigned to a specific defect. H4 scales the frozen Yu phonon-phonon term but is not interpreted as a revised Yu parameter. Every fitted amplitude is nonnegative; no independent experimental upper bound is available, so physicality is limited to sign and optimizer-bound checks.",
        ]
    )
    (out / "sheskin_background_model_selection.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    outputs = {path.name: sha256(path) for path in sorted(out.iterdir()) if path.is_file()}
    (out / "stage2_manifest.json").write_text(
        json.dumps({"schema": SCHEMA, "status": background_status, "outputs": outputs}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(background_status)


if __name__ == "__main__":
    main()
