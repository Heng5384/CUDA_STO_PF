#!/usr/bin/env python3
"""User-authorized minimal refit to Yu 2024 Figure 6b dashed curves.

The original published-parameter reproduction remains untouched. This script
fits only two positive parameters:
  1. shared A_N for both As-quenched and Annealed;
  2. a multiplier on the Annealed dislocation density, which is equivalent to
     multiplying both SI S11 and S13 rates because both are linear in N_D.
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
REPRODUCTION_SCRIPT = ROOT / "scripts/reproduce_yu2024_debye_callaway.py"
DATA_SOURCE = ROOT / "data/qualification/yu2024_transport_v1"
DATA_OUTPUT = ROOT / "data/qualification/yu2024_transport_refit_v1"
REPORT_OUTPUT = ROOT / "reports/yu2024_debye_callaway_refit_v1"


def load_reproduction_module():
    spec = importlib.util.spec_from_file_location(
        "yu2024_reproduction", REPRODUCTION_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {REPRODUCTION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fit_configs(
    base_configs: dict[str, dict[str, Any]],
    fitted_A_N: float,
    annealed_dislocation_scale: float,
) -> dict[str, dict[str, Any]]:
    configs = copy.deepcopy(base_configs)
    for config in configs.values():
        config["shared_parameters"]["A_N"] = fitted_A_N
    configs["Annealed"]["state_parameters"]["dislocation_density_m2"] *= (
        annealed_dislocation_scale
    )
    return configs


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    absolute = np.abs(prediction - target)
    relative = absolute / target
    return {
        "MAE_W_mK": float(np.mean(absolute)),
        "MAPE_fraction": float(np.mean(relative)),
        "max_relative_error_fraction": float(np.max(relative)),
        "RMSE_W_mK": float(np.sqrt(np.mean(absolute**2))),
    }


def main() -> None:
    model = load_reproduction_module()
    base_configs = {
        "AQ": load_json(DATA_SOURCE / "yu_AQ_parameters.json"),
        "Annealed": load_json(DATA_SOURCE / "yu_48h_parameters.json"),
    }
    digitized = {
        "AQ": read_csv(DATA_SOURCE / "yu_AQ_model_digitized.csv"),
        "Annealed": read_csv(DATA_SOURCE / "yu_48h_model_digitized.csv"),
    }
    experiments = {
        "AQ": read_csv(DATA_SOURCE / "yu_AQ_kappaL_digitized.csv"),
        "Annealed": read_csv(DATA_SOURCE / "yu_48h_kappaL_digitized.csv"),
    }

    def residuals(log_parameters: np.ndarray) -> np.ndarray:
        fitted_A_N, dislocation_scale = np.exp(log_parameters)
        configs = fit_configs(base_configs, fitted_A_N, dislocation_scale)
        values: list[float] = []
        for state in ("AQ", "Annealed"):
            for row in digitized[state]:
                prediction = model.integrate_gauss_legendre(
                    float(row["T_K"]), configs[state]
                )
                target = float(row["yu_model_kappa_lat_W_mK"])
                uncertainty = float(row["digitization_uncertainty_W_mK"])
                values.append((prediction - target) / uncertainty)
        return np.asarray(values)

    result = least_squares(
        residuals,
        np.log([1.5, 0.1]),
        bounds=(np.log([0.1, 1.0e-6]), np.log([5.0, 10.0])),
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=1000,
    )
    if not result.success:
        raise RuntimeError(result.message)
    fitted_A_N, dislocation_scale = np.exp(result.x)
    configs = fit_configs(base_configs, fitted_A_N, dislocation_scale)

    n_observations = len(result.fun)
    n_parameters = len(result.x)
    degrees_of_freedom = n_observations - n_parameters
    chi_square = float(np.sum(result.fun**2))
    reduced_chi_square = chi_square / degrees_of_freedom
    covariance_log = (
        np.linalg.inv(result.jac.T @ result.jac) * reduced_chi_square
    )
    standard_error_log = np.sqrt(np.diag(covariance_log))
    critical_t = float(student_t.ppf(0.975, degrees_of_freedom))
    lower = np.exp(result.x - critical_t * standard_error_log)
    upper = np.exp(result.x + critical_t * standard_error_log)
    jacobian_condition = float(np.linalg.cond(result.jac))

    DATA_OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.mkdir(parents=True, exist_ok=True)
    fitted_contract = {
        "status": "USER_AUTHORIZED_REFIT_NOT_YU_PUBLISHED_PARAMETER_SET",
        "fit_target": "Yu 2024 main Figure 6b dashed Callaway curves",
        "state_labels": {
            "AQ": "As-quenched",
            "Annealed": "Annealed at 653 K for 48 h",
        },
        "frozen_parameters": "all Yu Table S2 inputs except the two listed fitted parameters",
        "fitted_parameters": {
            "A_N": {
                "published": 1.5,
                "fitted": float(fitted_A_N),
                "formal_95pct_interval": [float(lower[0]), float(upper[0])],
            },
            "Annealed_dislocation_rate_scale": {
                "published_reference": 1.0,
                "fitted": float(dislocation_scale),
                "formal_95pct_interval": [float(lower[1]), float(upper[1])],
                "applies_to": "both SI S11 and S13 through their common linear N_D dependence",
            },
            "Annealed_effective_dislocation_density_m-2": {
                "published": base_configs["Annealed"]["state_parameters"][
                    "dislocation_density_m2"
                ],
                "fitted": configs["Annealed"]["state_parameters"][
                    "dislocation_density_m2"
                ],
            },
            "Annealed_effective_dislocation_density_cm-2": {
                "published": base_configs["Annealed"]["state_parameters"][
                    "dislocation_density_m2"
                ]
                / 1.0e4,
                "fitted": configs["Annealed"]["state_parameters"][
                    "dislocation_density_m2"
                ]
                / 1.0e4,
            },
        },
        "fit_method": {
            "algorithm": "bounded nonlinear least squares in log-parameter space",
            "residual": "(calculated - digitized Yu model) / digitization uncertainty",
            "observations": n_observations,
            "parameters": n_parameters,
            "degrees_of_freedom": degrees_of_freedom,
            "chi_square": chi_square,
            "reduced_chi_square": reduced_chi_square,
            "jacobian_condition_number": jacobian_condition,
            "function_evaluations": result.nfev,
            "optimality": float(result.optimality),
            "formal_interval_warning": (
                "Intervals are local least-squares intervals. Digitized curve "
                "points are correlated, so they are not physical confidence bounds."
            ),
        },
        "isolation": {
            "PF_data_used": False,
            "Sheskin_data_used": False,
            "cluster_used": False,
            "GPU_used": False,
        },
    }
    (DATA_OUTPUT / "fitted_parameter_contract.json").write_text(
        json.dumps(fitted_contract, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for state, filename in (
        ("AQ", "yu_AQ_refitted_parameters.json"),
        ("Annealed", "yu_Annealed_refitted_parameters.json"),
    ):
        output_config = copy.deepcopy(configs[state])
        output_config["refit_status"] = (
            "USER_AUTHORIZED_REFIT_NOT_YU_PUBLISHED_PARAMETER_SET"
        )
        output_config["fit_target"] = "Yu 2024 Figure 6b dashed curve"
        output_config["state"] = state
        (DATA_OUTPUT / filename).write_text(
            json.dumps(output_config, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    comparison_rows: list[dict[str, Any]] = []
    fit_metrics: dict[str, Any] = {}
    for state in ("AQ", "Annealed"):
        model_target = np.asarray(
            [float(row["yu_model_kappa_lat_W_mK"]) for row in digitized[state]]
        )
        experimental_target = np.asarray(
            [float(row["kappa_lat_W_mK"]) for row in experiments[state]]
        )
        temperature = np.asarray([float(row["T_K"]) for row in digitized[state]])
        published_prediction = np.asarray(
            [
                model.integrate_gauss_legendre(float(t), base_configs[state])
                for t in temperature
            ]
        )
        fitted_prediction = np.asarray(
            [
                model.integrate_gauss_legendre(float(t), configs[state])
                for t in temperature
            ]
        )
        fit_metrics[state] = {
            "published_vs_yu_model": metrics(published_prediction, model_target),
            "refitted_vs_yu_model": metrics(fitted_prediction, model_target),
            "refitted_vs_experiment": metrics(
                fitted_prediction, experimental_target
            ),
        }
        for index, t_value in enumerate(temperature):
            comparison_rows.append(
                {
                    "state": state,
                    "T_K": f"{t_value:.8g}",
                    "yu_model_digitized_W_mK": f"{model_target[index]:.12g}",
                    "experiment_Table_S1_W_mK": (
                        f"{experimental_target[index]:.12g}"
                    ),
                    "published_parameter_prediction_W_mK": (
                        f"{published_prediction[index]:.12g}"
                    ),
                    "refitted_prediction_W_mK": (
                        f"{fitted_prediction[index]:.12g}"
                    ),
                    "refitted_relative_error_vs_yu_model": (
                        f"{abs(fitted_prediction[index]-model_target[index])/model_target[index]:.12g}"
                    ),
                    "refitted_relative_error_vs_experiment": (
                        f"{abs(fitted_prediction[index]-experimental_target[index])/experimental_target[index]:.12g}"
                    ),
                }
            )
    write_csv(REPORT_OUTPUT / "refit_comparison.csv", comparison_rows)

    covariance_rows = [
        {
            "parameter_i": name_i,
            "parameter_j": name_j,
            "covariance_log_space": f"{covariance_log[i, j]:.12g}",
        }
        for i, name_i in enumerate(("A_N", "Annealed_dislocation_scale"))
        for j, name_j in enumerate(("A_N", "Annealed_dislocation_scale"))
    ]
    write_csv(REPORT_OUTPUT / "parameter_covariance.csv", covariance_rows)

    summary = {
        "status": "REFIT_COMPLETED",
        "fitted_parameters": fitted_contract["fitted_parameters"],
        "fit_statistics": fitted_contract["fit_method"],
        "metrics": fit_metrics,
        "PF_data_used": False,
        "Sheskin_data_used": False,
        "original_reproduction_overwritten": False,
    }
    (REPORT_OUTPUT / "refit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    colors = {"AQ": "#d62728", "Annealed": "#2ca02c"}
    markers = {"AQ": "o", "Annealed": "s"}
    for state in ("AQ", "Annealed"):
        temperature_data = np.asarray(
            [float(row["T_K"]) for row in digitized[state]]
        )
        smooth_temperature = np.linspace(
            float(temperature_data.min()), float(temperature_data.max()), 181
        )
        fitted_smooth = np.asarray(
            [
                model.integrate_gauss_legendre(float(t), configs[state])
                for t in smooth_temperature
            ]
        )
        fig, ax = plt.subplots(figsize=(6.4, 4.5), dpi=160)
        ax.plot(
            smooth_temperature,
            fitted_smooth,
            color="#173b8f",
            lw=2.0,
            label="Refitted model",
        )
        ax.plot(
            temperature_data,
            [float(row["yu_model_kappa_lat_W_mK"]) for row in digitized[state]],
            color="#24306e",
            ls="--",
            marker="x",
            ms=4,
            label="Yu model (Figure 6b digitized)",
        )
        ax.scatter(
            temperature_data,
            [float(row["kappa_lat_W_mK"]) for row in experiments[state]],
            color=colors[state],
            edgecolor="black",
            linewidth=0.4,
            s=34,
            label=f"{state} experiment",
            zorder=3,
        )
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(r"$\kappa_{\mathrm{lat}}$ (W m$^{-1}$ K$^{-1}$)")
        ax.set_title(f"Yu 2024 {state}: user-authorized minimal refit")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(
            REPORT_OUTPUT / f"{state}_refit.png",
            metadata={"Software": "refit_yu2024_debye_callaway.py"},
        )
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.9), dpi=160)
    for state in ("AQ", "Annealed"):
        temperature_data = np.asarray(
            [float(row["T_K"]) for row in digitized[state]]
        )
        smooth_temperature = np.linspace(
            float(temperature_data.min()), float(temperature_data.max()), 181
        )
        fitted_smooth = [
            model.integrate_gauss_legendre(float(t), configs[state])
            for t in smooth_temperature
        ]
        ax.plot(
            smooth_temperature,
            fitted_smooth,
            color=colors[state],
            lw=2.0,
            label=f"{state} refit",
        )
        ax.plot(
            temperature_data,
            [float(row["yu_model_kappa_lat_W_mK"]) for row in digitized[state]],
            color=colors[state],
            ls="--",
            marker="x",
            ms=4,
            label=f"{state} Yu model",
        )
        ax.scatter(
            temperature_data,
            [float(row["kappa_lat_W_mK"]) for row in experiments[state]],
            color=colors[state],
            edgecolor="black",
            linewidth=0.4,
            marker=markers[state],
            s=32,
            zorder=3,
            label=f"{state} experiment",
        )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$\kappa_{\mathrm{lat}}$ (W m$^{-1}$ K$^{-1}$)")
    ax.set_title("Yu 2024 As-quenched and Annealed minimal refit")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(
        REPORT_OUTPUT / "combined_refit.png",
        metadata={"Software": "refit_yu2024_debye_callaway.py"},
    )
    plt.close(fig)

    report = f"""# Yu 2024 minimal refit

## Scope

This is a user-authorized refit to the two dashed Callaway curves in main-text
Figure 6b. It does not replace the published-parameter reproduction. The upper
curve is labelled **Annealed** (653 K for 48 h); the lower curve is
**As-quenched**.

All Yu parameters are frozen except:

1. the shared `A_N`;
2. one multiplier on the Annealed dislocation density, equivalently scaling
   both S11 and S13 because both rates are linear in `N_D`.

## Fitted values

| Parameter | Published | Fitted |
|---|---:|---:|
| A_N | 1.5 | {fitted_A_N:.9g} |
| Annealed dislocation-rate scale | 1.0 | {dislocation_scale:.9g} |
| Annealed N_D (m^-2) | {base_configs["Annealed"]["state_parameters"]["dislocation_density_m2"]:.9g} | {configs["Annealed"]["state_parameters"]["dislocation_density_m2"]:.9g} |
| Annealed N_D (cm^-2) | {base_configs["Annealed"]["state_parameters"]["dislocation_density_m2"]/1e4:.9g} | {configs["Annealed"]["state_parameters"]["dislocation_density_m2"]/1e4:.9g} |

The fitted effective dislocation density is {dislocation_scale:.3%} of the
published value. This is an effective curve-fit result, not a replacement
measurement.

## Accuracy against Yu Figure 6b

| State | MAPE | Maximum relative error |
|---|---:|---:|
| As-quenched | {fit_metrics["AQ"]["refitted_vs_yu_model"]["MAPE_fraction"]:.3%} | {fit_metrics["AQ"]["refitted_vs_yu_model"]["max_relative_error_fraction"]:.3%} |
| Annealed | {fit_metrics["Annealed"]["refitted_vs_yu_model"]["MAPE_fraction"]:.3%} | {fit_metrics["Annealed"]["refitted_vs_yu_model"]["max_relative_error_fraction"]:.3%} |

## Fit diagnostics

- observations: {n_observations}
- fitted parameters: {n_parameters}
- degrees of freedom: {degrees_of_freedom}
- chi-square using the digitization uncertainty: {chi_square:.6g}
- reduced chi-square: {reduced_chi_square:.6g}
- Jacobian condition number: {jacobian_condition:.6g}
- PF data used: `false`
- Sheskin data used: `false`

The formal intervals in `fitted_parameter_contract.json` are local
least-squares intervals only. Digitized points along the same drawn curve are
correlated, so those intervals must not be interpreted as physical confidence
bounds.

## Scientific interpretation

The shared intrinsic factor changes only slightly from 1.5 to
{fitted_A_N:.6f}. Nearly all of the original Annealed mismatch is absorbed by
reducing the effective dislocation strength to {dislocation_scale:.6f} of the
published S11/S13 value. This confirms that the discrepancy is localized to
the quantitative dislocation contract, but it does not identify whether the
unreported difference lies in `N_D`, Eq. S13 implementation, `gamma_prime`,
or another prefactor.
"""
    (REPORT_OUTPUT / "refit_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
