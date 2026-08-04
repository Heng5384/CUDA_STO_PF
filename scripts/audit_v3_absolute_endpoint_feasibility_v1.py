#!/usr/bin/env python3
"""Audit V3 endpoint feasibility without changing any PF or atomistic state.

The AQ background is selected before this program loads any 6 h/48 h PF or
experimental endpoint.  H-P0 is the frozen Yu scalar host with no inherited V2
background.  H-P2 members are explicitly provisional branch-resolved controls
derived from the 303.2 K PbTe tensor; they are not temperature-qualified V3
host authorities.  The optional material-leverage stage uses the separately
qualified V3 single-particle numerical kernel and source-defined scalar Ag2Te
controls, but cannot promote them past the existing high-temperature material
gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2 as chi2_distribution


SCHEMA = "V3_ABSOLUTE_ENDPOINT_FEASIBILITY_AND_BACKGROUND_PORTABILITY_AUDIT_V1"
TEMPERATURES_K = (303.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15)
ENDPOINT_T_K = 573.15
EXPERIMENT_6H = 0.85
EXPERIMENT_48H = 1.03
REPLICATES = ("A", "B", "C")
AGES_H = (6.0, 48.0)
AQ_XAG = 0.0078
GAUSS_FIT = 512
GAUSS_ENDPOINT = 512
PF_SV_RATIO = 0.39530647550891224
PF_MID_Q_RATIO = 0.3062474378035795
PF_HIGH_Q_RATIO = 0.4583384203816745
V3_G2_G0_6H = 0.997002235
V3_G2_G0_48H = 1.07359111
V3_TENSOR_MID_Q_RATIO = 0.3239
V3_TENSOR_HIGH_Q_RATIO = 0.4312
V1_EXPECTED = {6.0: 1.2980916621210676, 48.0: 1.2948991542326225}
V2_EXPECTED = {
    ("M0", 6.0): 0.8733452621717084,
    ("M0", 48.0): 0.8716818275173842,
    ("MIS", 6.0): 0.8208832450913344,
    ("MIS", 48.0): 0.8490371923264635,
}
V1_MANIFEST_SHA256 = "653dfb0def383ffe01134ed609a02b382394841520cb72e51131c76a8ef3cbfc"
V2_MANIFEST_SHA256 = "1c815a6277d46ec4c5537966de402f0e190545b6433761291bf46c08ce725856"
PRODUCTION_STATUS = "PASS_246CUBE_6H48H_CONDITIONAL_PRODUCTION_V1"
BACKGROUND_MODELS = ("B0_none", "B1_constant", "B2_omega2", "B4_omega4", "BU_host_scale")
REQUIRED_OUTPUTS = (
    "provenance.md",
    "v1_v2_reproduction.csv",
    "inherited_v2_background_upper_bound.csv",
    "inherited_v2_background_no_go_proof.md",
    "v2_background_portability_audit.md",
    "background_identity_matrix.csv",
    "v3_aq_background_candidate_fits.csv",
    "v3_aq_background_cross_validation.csv",
    "v3_aq_background_selection.md",
    "v3_aq_background_envelope.json",
    "V3_AQ_BACKGROUND_FROZEN_BEFORE_48H_MANIFEST.json",
    "v3_absolute_endpoint_upper_bounds.csv",
    "v3_upper_bound_full_temperature.csv",
    "v3_endpoint_necessary_condition_audit.md",
    "required_dynamic_resistance_decay.csv",
    "required_spectral_decay_D0_D2_D4.csv",
    "required_vs_pf_structural_decay.md",
    "v3_material_envelope_scan.csv",
    "v3_feasible_parameter_region.csv",
    "v3_envelope_leverage_report.md",
    "final_v3_feasibility_report.md",
    "final_terminal_output.txt",
    "inherited_v2_background_hard_cap.png",
    "v3_upper_bound_vs_experiment.png",
    "v3_upper_bound_full_temperature.png",
    "required_dynamic_decay_vs_pf_ratios.png",
    "v3_material_envelope_feasible_region.png",
    "v1_v2_v3_upper_bound_comparison.png",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


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


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


@dataclass(frozen=True)
class HostMember:
    host_id: str
    family: str
    velocities_m_s: tuple[float, float, float]
    direction_label: str
    status: str
    qualification: str


def cubic_branch_velocities(direction: Iterable[float]) -> tuple[float, float, float]:
    c11, c12, c44, density = 107.95e9, 7.63e9, 13.44e9, 8383.0
    n = np.asarray(tuple(direction), dtype=float)
    n /= np.linalg.norm(n)
    gamma = np.empty((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            if i == j:
                gamma[i, j] = c11 * n[i] ** 2 + c44 * (1.0 - n[i] ** 2)
            else:
                gamma[i, j] = (c12 + c44) * n[i] * n[j]
    return tuple(float(value) for value in np.sqrt(np.linalg.eigvalsh(gamma) / density))


def angular_mean_velocities() -> tuple[float, float, float]:
    golden = math.pi * (3.0 - math.sqrt(5.0))
    values = []
    for index in range(512):
        z = 1.0 - 2.0 * (index + 0.5) / 512.0
        radial = math.sqrt(max(0.0, 1.0 - z * z))
        phi = golden * index
        values.append(cubic_branch_velocities((radial * math.cos(phi), radial * math.sin(phi), z)))
    return tuple(float(x) for x in np.mean(np.asarray(values), axis=0))


def host_members() -> list[HostMember]:
    return [
        HostMember("H-P0_YU_SCALAR", "H-P0", (1770.0, 1770.0, 1770.0), "scalar", "AVAILABLE_CONTROL", "FROZEN_YU_HOST_WITHOUT_V2_BACKGROUND"),
        HostMember("H-P2_PBTE_303K_100", "H-P2", cubic_branch_velocities((1, 0, 0)), "[100]", "PROVISIONAL_HOST_ENVELOPE", "303P2K_TENSOR_EXTRAPOLATED_OVER_TRANSPORT_RANGE"),
        HostMember("H-P2_PBTE_303K_110", "H-P2", cubic_branch_velocities((1, 1, 0)), "[110]", "PROVISIONAL_HOST_ENVELOPE", "303P2K_TENSOR_EXTRAPOLATED_OVER_TRANSPORT_RANGE"),
        HostMember("H-P2_PBTE_303K_111", "H-P2", cubic_branch_velocities((1, 1, 1)), "[111]", "PROVISIONAL_HOST_ENVELOPE", "303P2K_TENSOR_EXTRAPOLATED_OVER_TRANSPORT_RANGE"),
        HostMember("H-P2_PBTE_303K_ANGULAR_MEAN", "H-P2", angular_mean_velocities(), "angular_mean", "PROVISIONAL_HOST_ENVELOPE", "303P2K_TENSOR_DIRECTION_AVERAGE_EXTRAPOLATED_OVER_TRANSPORT_RANGE"),
    ]


def population_cases(fit_module: Any) -> list[dict[str, Any]]:
    cases = fit_module.population_cases()
    require(len(cases) == 18, "AQ source-literal envelope is not 18 members")
    return cases


class BranchDebyeEvaluator:
    def __init__(self, config: dict[str, Any], order: int = GAUSS_FIT):
        self.config = config
        self.nodes, self.weights = leggauss(order)
        self.order = order
        shared = config["shared_parameters"]
        state = config["state_parameters"]
        self.kb = float(config["physical_constants"]["k_B_J_K"])
        self.hbar = float(config["physical_constants"]["hbar_J_s"])
        self.theta = float(shared["debye_temperature_K"])
        self.gamma = float(shared["gruneisen_gamma"])
        self.mass = float(shared["average_atomic_mass_kg"])
        self.grain = float(state["grain_size_m"])
        lattice = float(state["solid_solution_lattice_constant_angstrom"]) * 1e-10
        self.atomic_volume = lattice**3 / 8.0
        self.mass_ratio_sq = (float(shared["published_delta_M_i_g_mol"]) / float(shared["matrix_atom_mass_g_mol"])) ** 2
        self.radius_ratio_sq = ((float(shared["impurity_atomic_radius_pm"]) - float(shared["matrix_atomic_radius_pm"])) / float(shared["matrix_atomic_radius_pm"])) ** 2
        self.epsilon = float(shared["point_defect_epsilon"])
        self.density_contrast = float(shared["density_difference_kg_m3"]) / float(shared["matrix_density_kg_m3"])

    def quadrature(self, temperature_K: float) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        xmax = self.theta / temperature_K
        x = 0.5 * (self.nodes + 1.0) * xmax
        omega = x * self.kb * temperature_K / self.hbar
        bose = x**4 * np.exp(x) / np.expm1(x) ** 2
        return x, omega, xmax, bose

    def host_rate(self, omega: np.ndarray, temperature_K: float, velocity: float) -> np.ndarray:
        return 1.5 * 2.0 / (6.0 * math.pi**2) ** (1.0 / 3.0) * self.kb * self.atomic_volume ** (1.0 / 3.0) * self.gamma**2 * omega**2 * temperature_K / (self.mass * velocity**3)

    def cross_section(self, omega: np.ndarray, radius_m: float, velocity: float) -> np.ndarray:
        sigma_short = 2.0 * math.pi * radius_m**2
        sigma_long = 4.0 / 9.0 * math.pi * radius_m**2 * self.density_contrast**2 * (omega * radius_m / velocity) ** 4
        return sigma_short * sigma_long / np.maximum(sigma_short + sigma_long, np.finfo(float).tiny)

    def particle_rate(self, omega: np.ndarray, velocity: float, case: dict[str, Any] | None = None, radii_m: np.ndarray | None = None, box_volume_m3: float | None = None) -> np.ndarray:
        rate = np.zeros_like(omega)
        if case is not None:
            for fraction, radius_nm in case["populations"]:
                rate += velocity * float(case["total_Nv_m-3"]) * float(fraction) * self.cross_section(omega, float(radius_nm) * 1e-9, velocity)
        if radii_m is not None:
            require(box_volume_m3 is not None and box_volume_m3 > 0.0, "invalid PF box volume")
            for radius in np.asarray(radii_m, dtype=float):
                rate += velocity * self.cross_section(omega, float(radius), velocity) / float(box_volume_m3)
        return rate

    def kappa(self, temperature_K: float, matrix_xag: float, host: HostMember, model: str, parameter: float, *, aq_case: dict[str, Any] | None = None, radii_m: np.ndarray | None = None, box_volume_m3: float | None = None, dynamic_exponent: int | None = None, dynamic_amplitude: float = 0.0, branch_dynamic_rates: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> float:
        _, omega, xmax, bose = self.quadrature(temperature_K)
        point_gamma = matrix_xag * (self.mass_ratio_sq + self.epsilon * self.radius_ratio_sq)
        total = 0.0
        for branch, velocity in enumerate(host.velocities_m_s):
            host_rate = self.host_rate(omega, temperature_K, velocity)
            rate = velocity / self.grain + self.atomic_volume * omega**4 * point_gamma / (4.0 * math.pi * velocity**3)
            rate += self.particle_rate(omega, velocity, aq_case, radii_m, box_volume_m3)
            if model == "B0_none":
                rate += host_rate
            elif model == "B1_constant":
                rate += host_rate + parameter
            elif model == "B2_omega2":
                rate += host_rate + parameter * omega**2
            elif model == "B4_omega4":
                rate += host_rate + parameter * omega**4
            elif model == "BU_host_scale":
                rate += parameter * host_rate
            else:
                raise ValueError(model)
            if dynamic_exponent is not None:
                rate += dynamic_amplitude * omega**dynamic_exponent
            if branch_dynamic_rates is not None:
                rate += branch_dynamic_rates[branch]
            require(bool(np.all(rate > 0.0)) and bool(np.all(np.isfinite(rate))), "nonpositive or nonfinite scattering rate")
            prefactor = self.kb / (6.0 * math.pi**2 * velocity) * (self.kb * temperature_K / self.hbar) ** 3
            total += 0.5 * xmax * float(np.sum(self.weights * prefactor * bose / rate))
        return total


def parameterization(model: str) -> tuple[tuple[float, float], Callable[[float], float], str, int]:
    if model == "B0_none":
        return ((0.0, 0.0), lambda _: 0.0, "none", 0)
    if model == "B1_constant":
        return ((-2.0, 16.0), lambda z: 10.0**z, "s^-1", 1)
    if model == "B2_omega2":
        return ((-32.0, -6.0), lambda z: 10.0**z, "s", 1)
    if model == "B4_omega4":
        return ((-58.0, -24.0), lambda z: 10.0**z, "s^3", 1)
    if model == "BU_host_scale":
        return ((-3.0, 3.0), lambda z: 10.0**z, "dimensionless", 1)
    raise ValueError(model)


def load_aq_observations(path: Path) -> list[dict[str, float]]:
    selected = [row for row in read_csv(path) if row["sample_state"] == "AQ" and row["analysis_selection"] == "SELECTED_FOR_TEMPERATURE_CURVE" and row["thermal_quantity_identity"] == "MEASURED_TOTAL_KAPPA"]
    require(len(selected) == 7, f"expected seven AQ points, found {len(selected)}")
    result = [{"temperature_K": float(row["temperature_K"]), "kappa_W_mK": float(row["kappa_W_mK"]), "sigma_W_mK": float(row["analysis_uncertainty_W_mK"])} for row in selected]
    result.sort(key=lambda row: row["temperature_K"])
    require(tuple(row["temperature_K"] for row in result) == TEMPERATURES_K, "unexpected AQ temperature grid")
    return result


def fit_one(evaluator: BranchDebyeEvaluator, host: HostMember, case: dict[str, Any], model: str, observations: list[dict[str, float]]) -> dict[str, Any]:
    bounds, decode, unit, nparameter = parameterization(model)

    def objective(encoded: float) -> float:
        parameter = decode(encoded)
        return float(sum(((evaluator.kappa(item["temperature_K"], AQ_XAG, host, model, parameter, aq_case=case) - item["kappa_W_mK"]) / item["sigma_W_mK"]) ** 2 for item in observations))

    if nparameter == 0:
        encoded, parameter, chi2 = 0.0, 0.0, objective(0.0)
        at_bound = False
    else:
        fitted = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1e-12, "maxiter": 500})
        require(bool(fitted.success), f"fit failed: {host.host_id} {case['case_id']} {model}")
        encoded, parameter, chi2 = float(fitted.x), decode(float(fitted.x)), float(fitted.fun)
        at_bound = abs(encoded - bounds[0]) < 1e-6 or abs(encoded - bounds[1]) < 1e-6
    prediction = np.asarray([evaluator.kappa(item["temperature_K"], AQ_XAG, host, model, parameter, aq_case=case) for item in observations])
    measured = np.asarray([item["kappa_W_mK"] for item in observations])
    residual = prediction - measured
    n = len(observations)
    aicc = chi2 + 2 * nparameter + (2 * nparameter * (nparameter + 1) / (n - nparameter - 1) if nparameter else 0.0)
    return {
        "parameter": parameter,
        "parameter_unit": unit,
        "chi2": chi2,
        "AICc": aicc,
        "degrees_of_freedom": n - nparameter,
        "chi2_survival_probability": float(chi2_distribution.sf(chi2, n - nparameter)),
        "RMSE_W_mK": float(np.sqrt(np.mean(residual**2))),
        "MAPE_percent": float(100.0 * np.mean(np.abs(residual / measured))),
        "max_abs_residual_W_mK": float(np.max(np.abs(residual))),
        "prediction": prediction,
        "encoded_at_bound": at_bound,
    }


def fit_aq_backgrounds(out: Path, evaluator: BranchDebyeEvaluator, hosts: list[HostMember], cases: list[dict[str, Any]], observations: list[dict[str, float]], input_hashes: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fit_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []
    for host in hosts:
        for case in cases:
            for model in BACKGROUND_MODELS:
                fitted = fit_one(evaluator, host, case, model, observations)
                physicality = "PASS_ZERO_BACKGROUND_CONTROL" if model == "B0_none" else ("FAIL_OPTIMIZER_BOUND" if fitted["encoded_at_bound"] else "PASS_NONNEGATIVE_ONE_PARAMETER")
                fit_rows.append({
                    "host_id": host.host_id, "host_family": host.family, "host_status": host.status,
                    "host_direction": host.direction_label, "TA1_m_s": host.velocities_m_s[0], "TA2_m_s": host.velocities_m_s[1], "LA_m_s": host.velocities_m_s[2],
                    "case_id": case["case_id"], "source_status": case["source_status"], "total_Nv_m-3": case["total_Nv_m-3"], "morphology": case["morphology"],
                    "background_model": model, "parameter": fitted["parameter"], "parameter_unit": fitted["parameter_unit"], "chi2": fitted["chi2"], "AICc": fitted["AICc"],
                    "degrees_of_freedom": fitted["degrees_of_freedom"], "chi2_survival_probability": fitted["chi2_survival_probability"], "RMSE_W_mK": fitted["RMSE_W_mK"],
                    "MAPE_percent": fitted["MAPE_percent"], "max_abs_residual_W_mK": fitted["max_abs_residual_W_mK"], "parameter_physicality": physicality,
                    "fit_data_identity": "AQ_MEASURED_TOTAL_ONLY_NO_6H_NO_48H",
                })
                for held_index, held in enumerate(observations):
                    training = [item for index, item in enumerate(observations) if index != held_index]
                    loto = fit_one(evaluator, host, case, model, training)
                    predicted = evaluator.kappa(held["temperature_K"], AQ_XAG, host, model, loto["parameter"], aq_case=case)
                    cv_rows.append({
                        "host_id": host.host_id, "case_id": case["case_id"], "background_model": model,
                        "held_out_temperature_K": held["temperature_K"], "training_points": len(training), "fitted_parameter": loto["parameter"], "parameter_unit": loto["parameter_unit"],
                        "measured_total_kappa_W_mK": held["kappa_W_mK"], "predicted_kappa_W_mK": predicted, "residual_W_mK": predicted - held["kappa_W_mK"],
                        "absolute_percentage_error": abs(predicted - held["kappa_W_mK"]) / held["kappa_W_mK"] * 100.0, "fit_data_identity": "AQ_LOTO_ONLY_NO_6H_NO_48H",
                    })
    write_csv(out / "v3_aq_background_candidate_fits.csv", fit_rows)
    write_csv(out / "v3_aq_background_cross_validation.csv", cv_rows)

    retained: list[dict[str, Any]] = []
    for host in hosts:
        for case in cases:
            subset = [row for row in fit_rows if row["host_id"] == host.host_id and row["case_id"] == case["case_id"] and not str(row["parameter_physicality"]).startswith("FAIL")]
            best = min(float(row["AICc"]) for row in subset)
            for row in subset:
                if float(row["AICc"]) <= best + 2.0:
                    retained.append({
                        "member_id": f"{host.host_id}__{case['case_id']}__{row['background_model']}",
                        "host_id": host.host_id, "host_family": host.family, "host_status": host.status, "host_direction": host.direction_label,
                        "velocities_m_s": list(host.velocities_m_s), "case_id": case["case_id"], "case_source_status": case["source_status"],
                        "background_model": row["background_model"], "parameter": float(row["parameter"]), "parameter_unit": row["parameter_unit"],
                        "AICc": float(row["AICc"]), "delta_AICc_within_host_case": float(row["AICc"]) - best, "AQ_MAPE_percent": float(row["MAPE_percent"]),
                    })
    envelope = {
        "schema": "V3_AQ_BACKGROUND_ENVELOPE_V1", "status": "PASS_AQ_ONLY_BACKGROUND_SELECTION_WITH_PROVISIONAL_HOST_MEMBERS",
        "selection_data": "AQ measured-total seven-point curve only", "member_count": len(retained), "members": retained,
        "background_is_nonprobabilistic": True, "H_P1_status": "UNAVAILABLE_NOT_QUALIFIED",
        "absolute_fit_warning": "POOR_ABSOLUTE_FIT_ALL_RETAINED_MEMBERS_CHI2_P_LT_0P01" if all(float(row["chi2_survival_probability"]) < 0.01 for row in fit_rows) else "MIXED_ABSOLUTE_FIT",
    }
    write_json(out / "v3_aq_background_envelope.json", envelope)

    summary = [
        "# V3 AQ background selection", "", "Status: `PASS_AQ_ONLY_BACKGROUND_SELECTION_WITH_PROVISIONAL_HOST_MEMBERS`.", "",
        "Only the seven selected AQ measured-total points were loaded for this stage. The V3 background was refitted for every host/AQ-structure member; no V2 A2 number was copied into the fit and no 6 h/48 h value was available to model selection.", "",
        "H-P0 is the frozen Yu scalar host with the V2 background removed. H-P1 is unavailable because no branch-resolved PbTe host conductivity contract is qualified. H-P2 uses the 303.2 K PbTe tensor along [100], [110], [111] and a deterministic angular mean, and every H-P2 result is labelled `PROVISIONAL_HOST_ENVELOPE` because temperature dependence and a branch-resolved host lifetime authority are absent.", "",
        f"Retained nonprobabilistic member count: `{len(retained)}` (Delta-AICc <= 2 within each host/AQ structure case).", "",
        "| host | retained models | AQ MAPE range (%) |", "|---|---|---:|",
    ]
    for host in hosts:
        subset = [row for row in retained if row["host_id"] == host.host_id]
        summary.append(f"| {host.host_id} | {', '.join(sorted(set(str(row['background_model']) for row in subset)))} | {min(float(row['AQ_MAPE_percent']) for row in subset):.3f}--{max(float(row['AQ_MAPE_percent']) for row in subset):.3f} |")
    summary.extend(["", "The envelope is not a probability distribution. B1/B2/B4 are empirical nonnegative rates; BU rescales the chosen host rate but is not a revised Yu A_N. B0 is a zero-background upper-control candidate. Poor absolute chi-square fit remains a warning even when AICc selects a relative winner.", ""])
    (out / "v3_aq_background_selection.md").write_text("\n".join(summary), encoding="utf-8")

    manifest = {
        "schema": "V3_AQ_BACKGROUND_FROZEN_BEFORE_48H_MANIFEST_V1", "status": "PASS_V3_AQ_BACKGROUND_FROZEN_BEFORE_48H",
        "experimental_48h_loaded": False, "experimental_6h_loaded": False, "PF_6h_48h_authority_loaded": False,
        "A_N": 1.5, "negative_scattering_allowed": False, "V2_A2_copied": False, "yu_dislocation_scale_used": False,
        "host_members": [host.__dict__ for host in hosts], "candidate_models": list(BACKGROUND_MODELS), "retained_member_count": len(retained),
        "input_sha256": input_hashes,
        "output_sha256": {name: sha256(out / name) for name in ("v3_aq_background_candidate_fits.csv", "v3_aq_background_cross_validation.csv", "v3_aq_background_selection.md", "v3_aq_background_envelope.json")},
    }
    write_json(out / "V3_AQ_BACKGROUND_FROZEN_BEFORE_48H_MANIFEST.json", manifest)
    return retained, manifest


def load_pf_authority(root: Path) -> tuple[dict[str, dict[float, dict[str, Any]]], dict[str, str]]:
    authority = root / "reports/pf_246cube_method1_production_authority_v1"
    result: dict[str, dict[float, dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    for replicate in REPLICATES:
        audit_path = authority / "authority" / replicate / "audit.json"
        status_path = authority / "authority" / replicate / "status.txt"
        snapshots_path = authority / "transport" / replicate / "transport_snapshots.csv"
        particles_path = authority / "transport" / replicate / "full_psd_particles.csv"
        require(status_path.read_text(encoding="utf-8").strip() == PRODUCTION_STATUS, f"{replicate} production status failed")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        require(audit.get("numerical_status") == PRODUCTION_STATUS and all(audit.get("gates", {}).values()), f"{replicate} production audit failed")
        require(len(audit.get("checkpoint_hashes", {})) == 44, f"{replicate} checkpoint chain is incomplete")
        particle_rows = read_csv(particles_path)
        by_snapshot: dict[str, list[float]] = {}
        for row in particle_rows:
            by_snapshot.setdefault(row["snapshot_id"], []).append(float(row["equivalent_radius_nm"]) * 1e-9)
        result[replicate] = {}
        for row in read_csv(snapshots_path):
            age = float(row["registered_age_h"])
            if age not in AGES_H:
                continue
            radii = np.asarray(sorted(by_snapshot[row["snapshot_id"]]), dtype=float)
            require(radii.size == int(row["particle_count"]), "PF PSD count closure failed")
            result[replicate][age] = {"matrix_xAg": float(row["matrix_xAg"]), "radii_m": radii, "box_volume_m3": float(row["box_volume_nm3"]) * 1e-27, "Nv_m-3": float(row["Nv_nm-3"]) * 1e27, "mean_radius_m": float(row["mean_radius_nm"]) * 1e-9}
        require(set(result[replicate]) == set(AGES_H), f"{replicate} endpoint missing")
        for path in (audit_path, status_path, snapshots_path, particles_path):
            hashes[str(path.relative_to(root))] = sha256(path)
    return result, hashes


def reproduce_v1_v2(root: Path, v3_root: Path, out: Path, evaluator: BranchDebyeEvaluator, pf: dict[str, dict[float, dict[str, Any]]], transport: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    v1_manifest = v3_root / "frozen_contracts/transport_v1_resolved_density_only/V1_FROZEN_MANIFEST.json"
    v2_manifest = v3_root / "frozen_contracts/transport_v2_sheskin_interface_scalar_strain/V2_FROZEN_MANIFEST.json"
    require(sha256(v1_manifest) == V1_MANIFEST_SHA256, "V1 frozen manifest mismatch")
    require(sha256(v2_manifest) == V2_MANIFEST_SHA256, "V2 frozen manifest mismatch")
    rows: list[dict[str, Any]] = []
    for age in AGES_H:
        values = [transport.integrate_kappa_gauss(ENDPOINT_T_K, pf[rep][age]["matrix_xAg"], pf[rep][age]["radii_m"], pf[rep][age]["box_volume_m3"], config, "full_psd", 512) for rep in REPLICATES]
        value = float(np.mean(values))
        rows.append({"contract": "V1_density_only", "model": "full_PSD_no_dis", "age_h": age, "temperature_K": ENDPOINT_T_K, "reproduced_kappa_W_mK": value, "frozen_kappa_W_mK": V1_EXPECTED[age], "absolute_difference_W_mK": abs(value - V1_EXPECTED[age]), "reproduction_method": "independent_Debye_reintegration", "status": "PASS" if abs(value - V1_EXPECTED[age]) <= 1e-12 else "FAIL"})
    blind = read_csv(root / "reports/sheskin_pf_interface_strain_blind_prediction_v1/blind_48h_predictions.csv")
    for model in ("M0", "MIS"):
        for age in AGES_H:
            values = [float(row["predicted_kappa_W_mK"]) for row in blind if row["model"] == model and float(row["age_h"]) == age and float(row["temperature_K"]) == ENDPOINT_T_K]
            require(len(values) == 54, f"V2 {model} frozen grid incomplete")
            value = float(np.mean(values))
            expected = V2_EXPECTED[(model, age)]
            rows.append({"contract": "V2_AQ_background_interface_scalar_strain", "model": model, "age_h": age, "temperature_K": ENDPOINT_T_K, "reproduced_kappa_W_mK": value, "frozen_kappa_W_mK": expected, "absolute_difference_W_mK": abs(value - expected), "reproduction_method": "immutable_manifest_verified_full_grid_aggregation", "status": "PASS" if abs(value - expected) <= 1e-14 else "FAIL"})
    require(all(row["status"] == "PASS" for row in rows), "V1/V2 reproduction failed")
    write_csv(out / "v1_v2_reproduction.csv", rows)
    return rows


def inherited_v2_audit(out: Path) -> dict[str, float]:
    upper6 = V2_EXPECTED[("M0", 6.0)]
    upper48 = V2_EXPECTED[("M0", 48.0)]
    gap = upper48 - EXPERIMENT_48H
    recovery = upper48 - EXPERIMENT_6H
    relative = recovery / EXPERIMENT_6H * 100.0
    rows = [
        {"branch": "BRANCH_A_FROZEN_V2_BACKGROUND", "age_h": 6, "temperature_K": ENDPOINT_T_K, "kappa_upper_W_mK": upper6, "experiment_W_mK": EXPERIMENT_6H, "gap_W_mK": upper6 - EXPERIMENT_6H, "dynamic_scattering": "V2_M0_CONTRACT", "status": "REFERENCE"},
        {"branch": "BRANCH_A_FROZEN_V2_BACKGROUND", "age_h": 48, "temperature_K": ENDPOINT_T_K, "kappa_upper_W_mK": upper48, "experiment_W_mK": EXPERIMENT_48H, "gap_W_mK": gap, "dynamic_scattering": "V2_M0_CONTRACT", "status": "NO_GO_V3_IF_V2_BACKGROUND_INHERITED"},
        {"branch": "BRANCH_A_FROZEN_V2_BACKGROUND", "age_h": "6_to_48_max_from_experimental_6h", "temperature_K": ENDPOINT_T_K, "kappa_upper_W_mK": recovery, "experiment_W_mK": 0.18, "gap_W_mK": recovery - 0.18, "dynamic_scattering": "maximum_recovery", "status": "NO_GO_V3_IF_V2_BACKGROUND_INHERITED"},
    ]
    write_csv(out / "inherited_v2_background_upper_bound.csv", rows)
    proof = f"""# Inherited V2 background hard-cap proof

`NO_GO_V3_IF_V2_BACKGROUND_INHERITED`

At {ENDPOINT_T_K:.2f} K the frozen V2 M0 contract gives `kappa_upper_48={upper48:.15f} W m^-1 K^-1`, below the Sheskin endpoint `1.03 W m^-1 K^-1` by `{gap:.15f} W m^-1 K^-1`. Starting from the experimental 6 h value, the largest contract-allowed recovery is only `{recovery:.15f} W m^-1 K^-1` or `{relative:.6f}%`.

For every additional passive V3 channel, `tau_dynamic^-1 >= 0`, so Matthiessen addition cannot raise conductivity above the fixed-base value. 在冻结V2背景上继续加入任何非负V3散射，不可能使48 h热导率高于M0上限；因此不能通过增强极化、界面、应变或Ag2Te阻尼散射将`{upper48:.4f}`提高到`1.03`。This conclusion is independent of the missing high-temperature Ag2Te material parameters.

This is the task-prescribed frozen-V2 contract cap. It does not reinterpret or overwrite the V2 report.
"""
    (out / "inherited_v2_background_no_go_proof.md").write_text(proof, encoding="utf-8")
    return {"upper6": upper6, "upper48": upper48, "gap": gap, "recovery": recovery, "relative": relative}


def portability_audit(out: Path) -> None:
    rows = [
        {"background_treatment": "inherit frozen V2 A2 unchanged", "allowed_as_formal_V3_contract": False, "role": "negative_control_only", "parameter_identity": "MODEL_DEPENDENT_RESIDUAL", "48h_used": False, "risk": "host error and omitted-interface residual can be double counted"},
        {"background_treatment": "refit with AQ under each V3 host", "allowed_as_formal_V3_contract": True, "role": "primary_AQ_only_candidate", "parameter_identity": "HOST_CONTRACT_SPECIFIC_EMPIRICAL_BACKGROUND", "48h_used": False, "risk": "poor absolute AQ fit must remain explicit"},
        {"background_treatment": "no background", "allowed_as_formal_V3_contract": True, "role": "upper_control", "parameter_identity": "B0_ZERO", "48h_used": False, "risk": "not a calibrated description"},
        {"background_treatment": "independently experimental constrained background", "allowed_as_formal_V3_contract": True, "role": "future_authority", "parameter_identity": "INDEPENDENT", "48h_used": False, "risk": "currently unavailable"},
        {"background_treatment": "adjust background with 48h", "allowed_as_formal_V3_contract": False, "role": "forbidden", "parameter_identity": "ENDPOINT_FIT", "48h_used": True, "risk": "destroys blind endpoint test"},
    ]
    write_csv(out / "background_identity_matrix.csv", rows)
    text = """# V2 background portability audit

`V2_A2_IS_MODEL_DEPENDENT_RESIDUAL`

The frozen V2 `tau_bg^-1=A2*omega^2` amplitude was inferred under the Yu single-average-velocity Debye host, the V2 point-defect contract and a density-only AQ object envelope. It omitted branch-resolved PbTe transport, elastic-tensor contrast, polarization conversion and high-temperature Ag2Te damping. Its fitted residual therefore contains the response of that complete model structure; it is not a transferable material constant.

Copying A2 unchanged into V3 can count a V2 host discrepancy or omitted interface physics once inside A2 and again through the new V3 channel. The unchanged A2 branch is retained only as a negative control. Every formal V3 replacement member refits at most one background parameter using AQ alone; 48 h adjustment remains forbidden.
"""
    (out / "v2_background_portability_audit.md").write_text(text, encoding="utf-8")


def host_lookup(hosts: list[HostMember]) -> dict[str, HostMember]:
    return {host.host_id: host for host in hosts}


def endpoint_upper_bounds(out: Path, evaluator: BranchDebyeEvaluator, hosts: list[HostMember], retained: list[dict[str, Any]], pf: dict[str, dict[float, dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = host_lookup(hosts)
    endpoint_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    ensemble_members: list[dict[str, Any]] = []
    for member in retained:
        host = lookup[str(member["host_id"])]
        member_endpoint: dict[float, list[float]] = {6.0: [], 48.0: []}
        for temperature in TEMPERATURES_K:
            for age in AGES_H:
                values = []
                for replicate in REPLICATES:
                    value = evaluator.kappa(temperature, pf[replicate][age]["matrix_xAg"], host, str(member["background_model"]), float(member["parameter"]))
                    values.append(value)
                    full_rows.append({
                        "member_id": member["member_id"], "host_id": host.host_id, "host_status": host.status, "background_model": member["background_model"], "AQ_case_id": member["case_id"],
                        "replicate": replicate, "age_h": age, "temperature_K": temperature, "matrix_xAg": pf[replicate][age]["matrix_xAg"], "kappa_upper_W_mK": value,
                        "dynamic_density_stiffness_interface_strain_damping_rates": 0.0, "upper_bound_identity": "NECESSARY_CONDITION_NOT_ACTUAL_PREDICTION",
                    })
                mean = float(np.mean(values))
                if temperature == ENDPOINT_T_K:
                    member_endpoint[age] = values
                    endpoint_rows.append({
                        "member_id": member["member_id"], "host_id": host.host_id, "host_family": host.family, "host_status": host.status, "background_model": member["background_model"], "AQ_case_id": member["case_id"],
                        "age_h": age, "temperature_K": temperature, "ensemble_mean_kappa_upper_W_mK": mean, "replicate_min_W_mK": min(values), "replicate_max_W_mK": max(values),
                        "experiment_W_mK": EXPERIMENT_6H if age == 6.0 else EXPERIMENT_48H, "endpoint_gate_pass": mean >= (EXPERIMENT_6H if age == 6.0 else EXPERIMENT_48H),
                    })
        k6 = float(np.mean(member_endpoint[6.0]))
        k48 = float(np.mean(member_endpoint[48.0]))
        ensemble_members.append({**member, "kappa_upper_6_W_mK": k6, "kappa_upper_48_W_mK": k48, "necessary_condition_pass": k6 >= EXPERIMENT_6H and k48 >= EXPERIMENT_48H})
    write_csv(out / "v3_absolute_endpoint_upper_bounds.csv", endpoint_rows)
    write_csv(out / "v3_upper_bound_full_temperature.csv", full_rows)
    passing = [row for row in ensemble_members if row["necessary_condition_pass"]]
    source_literal_passing = [row for row in passing if row["case_source_status"] == "SOURCE_LITERAL_BOUND"]
    status = "PASS_V3_ABSOLUTE_ENDPOINT_NECESSARY_CONDITION" if passing else "NO_GO_V3_ABSOLUTE_ENDPOINT_UPPER_BOUND"
    hp0 = [row for row in ensemble_members if row["host_family"] == "H-P0"]
    all6 = np.asarray([float(row["kappa_upper_6_W_mK"]) for row in ensemble_members])
    all48 = np.asarray([float(row["kappa_upper_48_W_mK"]) for row in ensemble_members])
    report = f"""# V3 absolute endpoint necessary-condition audit

`{status}`

All resolved density/stiffness, interface, conversion, roughness, coherent-strain and damping rates were set exactly to zero. The retained base contains only the selected host, grain boundary, matrix point defects, AQ-only time-invariant background and the registered A/B/C matrix composition.

- retained AQ background members: `{len(ensemble_members)}`
- endpoint-passing members: `{len(passing)}`
- endpoint-passing `SOURCE_LITERAL_BOUND` AQ cases: `{len(source_literal_passing)}`
- all-member 573.15 K upper 6 h min/median/max: `{all6.min():.6f} / {np.median(all6):.6f} / {all6.max():.6f}` W m^-1 K^-1
- all-member 573.15 K upper 48 h min/median/max: `{all48.min():.6f} / {np.median(all48):.6f} / {all48.max():.6f}` W m^-1 K^-1
- H-P0 48 h maximum: `{max(float(row['kappa_upper_48_W_mK']) for row in hp0):.6f}` W m^-1 K^-1 (all H-P0 members fail 1.03)

The mathematical hard cap is removed only by some H-P2 members. Every such member remains `PROVISIONAL_HOST_ENVELOPE`, and every passing AQ structure case is a `SINGLE_APT_COUNT_FRACTION_DIAGNOSTIC` rather than a source-literal two-endmember bound. Therefore this PASS means only that the broad replacement-model diagnostic envelope is not excluded by the endpoint inequality. Rebuilding the host/background contract removes the mathematical hard cap, but does not prove that physically realistic Ag2Te parameters will produce the required recovery.
"""
    (out / "v3_endpoint_necessary_condition_audit.md").write_text(report, encoding="utf-8")
    return ensemble_members, passing, full_rows


def solve_positive_amplitude(function: Callable[[float], float], target: float) -> float:
    at_zero = function(0.0)
    require(at_zero >= target - 1e-12, "target exceeds zero-dynamic upper bound")
    if abs(at_zero - target) < 1e-13:
        return 0.0
    # D0, D2 and D4 amplitudes have different dimensions and can differ by
    # more than fifty decades.  Solving directly in amplitude space can make
    # scipy's absolute tolerance return zero for a perfectly valid D2/D4
    # root, so bracket and solve in log10(amplitude) instead.
    lower_log10 = -80.0
    lower_value = function(10.0**lower_log10) - target
    if lower_value < 0.0:
        return float(brentq(lambda value: function(value) - target, 0.0, 10.0**lower_log10, xtol=1e-100, rtol=1e-12, maxiter=300))
    upper_log10 = lower_log10
    while function(10.0**upper_log10) > target:
        upper_log10 += 1.0
        require(upper_log10 <= 80.0, "could not bracket positive scattering amplitude")
    root_log10 = brentq(
        lambda log10_value: function(10.0**log10_value) - target,
        lower_log10,
        upper_log10,
        xtol=1e-12,
        rtol=1e-12,
        maxiter=300,
    )
    return float(10.0**root_log10)


def dynamic_decay(out: Path, evaluator: BranchDebyeEvaluator, hosts: list[HostMember], passing: list[dict[str, Any]], pf: dict[str, dict[float, dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    lookup = host_lookup(hosts)
    resistance_rows: list[dict[str, Any]] = []
    spectral_rows: list[dict[str, Any]] = []
    for member in passing:
        k6 = float(member["kappa_upper_6_W_mK"])
        k48 = float(member["kappa_upper_48_W_mK"])
        dw6 = 1.0 / EXPERIMENT_6H - 1.0 / k6
        dw48 = 1.0 / EXPERIMENT_48H - 1.0 / k48
        require(dw6 >= -1e-14 and dw48 >= -1e-14, f"negative required resistance for {member['member_id']}")
        resistance_rows.append({
            "member_id": member["member_id"], "host_id": member["host_id"], "background_model": member["background_model"], "AQ_case_id": member["case_id"],
            "kappa_upper_6_W_mK": k6, "kappa_upper_48_W_mK": k48, "W_upper_6_mK_W": 1.0 / k6, "W_upper_48_mK_W": 1.0 / k48,
            "W_exp_6_mK_W": 1.0 / EXPERIMENT_6H, "W_exp_48_mK_W": 1.0 / EXPERIMENT_48H, "Delta_W_required_6": dw6, "Delta_W_allowed_48": dw48,
            "r_required_effective_resistance": dw48 / dw6, "diagnostic_only_not_spectral_substitute": True,
        })
        host = lookup[str(member["host_id"])]
        for exponent, label, unit in ((0, "D0", "s^-1"), (2, "D2", "s"), (4, "D4", "s^3")):
            def mean_kappa(age: float, amplitude: float) -> float:
                return float(np.mean([evaluator.kappa(ENDPOINT_T_K, pf[rep][age]["matrix_xAg"], host, str(member["background_model"]), float(member["parameter"]), dynamic_exponent=exponent, dynamic_amplitude=amplitude) for rep in REPLICATES]))
            amplitude6 = solve_positive_amplitude(lambda amplitude: mean_kappa(6.0, amplitude), EXPERIMENT_6H)
            decay = solve_positive_amplitude(lambda ratio: mean_kappa(48.0, amplitude6 * ratio), EXPERIMENT_48H)
            require(0.0 <= decay <= 1.0, f"non-decaying {label} solution for {member['member_id']}")
            spectral_rows.append({
                "member_id": member["member_id"], "host_id": member["host_id"], "background_model": member["background_model"], "AQ_case_id": member["case_id"],
                "test_spectrum": label, "omega_exponent": exponent, "frozen_6h_amplitude": amplitude6, "amplitude_unit": unit, "required_48h_to_6h_rate_scale": decay,
                "kappa_6h_closure_W_mK": mean_kappa(6.0, amplitude6), "kappa_48h_closure_W_mK": mean_kappa(48.0, amplitude6 * decay),
                "r_is_fit_parameter": False, "use": "NECESSARY_SPECTRAL_DECAY_DIAGNOSTIC",
            })
    write_csv(out / "required_dynamic_resistance_decay.csv", resistance_rows)
    write_csv(out / "required_spectral_decay_D0_D2_D4.csv", spectral_rows)
    values = np.asarray([float(row["required_48h_to_6h_rate_scale"]) for row in spectral_rows])
    structural_min = min(PF_SV_RATIO, PF_MID_Q_RATIO, PF_HIGH_Q_RATIO)
    structural_max = max(PF_SV_RATIO, PF_MID_Q_RATIO, PF_HIGH_Q_RATIO)
    overlap = bool(np.any((values >= structural_min) & (values <= structural_max)))
    status = "PASS_REQUIRED_DECAY_OVERLAPS_PF_STRUCTURAL_RATIOS" if overlap else "MISMATCH_REQUIRED_DECAY_OUTSIDE_PF_STRUCTURAL_RATIOS"
    by_label = {label: np.asarray([float(row["required_48h_to_6h_rate_scale"]) for row in spectral_rows if row["test_spectrum"] == label]) for label in ("D0", "D2", "D4")}
    report = f"""# Required dynamic decay versus PF structure

`{status}`

The effective-resistance ratio is only a magnitude diagnostic. Exact Debye integrations independently calibrated a nonnegative D0, D2 or D4 rate at 6 h, froze that amplitude, and solved only for the 48 h/6 h rate ratio needed to reach 1.03 W m^-1 K^-1. The solved ratio is not adopted as a fit parameter.

| quantity | value/range |
|---|---:|
| required D0 rate ratio | {by_label['D0'].min():.4f}--{by_label['D0'].max():.4f} |
| required D2 rate ratio | {by_label['D2'].min():.4f}--{by_label['D2'].max():.4f} |
| required D4 rate ratio | {by_label['D4'].min():.4f}--{by_label['D4'].max():.4f} |
| PF Sv ratio | {PF_SV_RATIO:.6f} |
| PF hydrostatic mid-q ratio | {PF_MID_Q_RATIO:.6f} |
| PF hydrostatic high-q ratio | {PF_HIGH_Q_RATIO:.6f} |
| frozen V1 full-PSD no-dis kappa ratio, 48 h/6 h | {V1_EXPECTED[48.0] / V1_EXPECTED[6.0]:.6f} |
| V3 G2/G0 area ratio-of-ratios, 48 h/6 h | {V3_G2_G0_48H / V3_G2_G0_6H:.6f} |
| V3 full-tensor mid-q ratio | {V3_TENSOR_MID_Q_RATIO:.4f} |
| V3 full-tensor high-q ratio | {V3_TENSOR_HIGH_Q_RATIO:.4f} |

The frozen density-only calculation changes by only `{(V1_EXPECTED[48.0] / V1_EXPECTED[6.0] - 1.0) * 100.0:.4f}%`; it cannot close the experimental amplitude.  G2/G0 grows because the particles become less spherical, while the qualified tensor descriptors show mid/high-q decay.  These are structure descriptors, not a scalar scattering law.  The existing V3 shape/anisotropy kernel has no material-authoritative contrast or damping, so no V3 rate ratio is silently substituted for them.
"""
    (out / "required_vs_pf_structural_decay.md").write_text(report, encoding="utf-8")
    return resistance_rows, spectral_rows, status


def interpolate_log_rate(omega_nodes: np.ndarray, rate_nodes: np.ndarray, omega: np.ndarray) -> np.ndarray:
    floor = np.finfo(float).tiny
    return np.exp(np.interp(np.log(np.maximum(omega, omega_nodes[0])), np.log(omega_nodes), np.log(np.maximum(rate_nodes, floor)), left=np.log(max(rate_nodes[0], floor)), right=np.log(max(rate_nodes[-1], floor))))


def material_kernel_scan(out: Path, evaluator: BranchDebyeEvaluator, hosts: list[HostMember], passing: list[dict[str, Any]], pf: dict[str, dict[float, dict[str, Any]]], v3_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, bool, bool]:
    report_root = v3_root / "reports/transport_v3_polarization_anisotropic_elastic_wave_v1"
    qualification = json.loads((report_root / "single_particle_validation/qualification_manifest.json").read_text(encoding="utf-8"))
    require(qualification.get("status") == "PASS_NUMERICAL_KERNEL_NOT_MATERIAL_AUTHORITY", "V3 numerical kernel is not qualified")
    require(qualification.get("physical_transport_authority") is False, "kernel unexpectedly promoted to material authority")
    require((report_root / "status.txt").read_text(encoding="utf-8").strip() == "BLOCKED_AG2TE_HIGH_TEMPERATURE_PROPERTIES", "V3 material gate identity changed")
    kernel_path = v3_root / "scripts/anisotropic_elastic_wave_scattering_v1.py"
    kernel = load_module(kernel_path, "v3_anisotropic_kernel_endpoint_audit")
    matrix = kernel.ElasticMedium(8383.0, kernel.cubic_tensor(107.95e9, 7.63e9, 13.44e9))
    direction_map = {
        "H-P2_PBTE_303K_100": np.array([1.0, 0.0, 0.0]),
        "H-P2_PBTE_303K_110": np.array([1.0, 1.0, 0.0]),
        "H-P2_PBTE_303K_111": np.array([1.0, 1.0, 1.0]),
    }
    # Source-defined scalar control: B=18.9 GPa at 523 K and the recovered
    # high-phase shear trend evaluated at 421.3, 497.2 and 573.15 K.  It is not
    # promoted to a single-crystal or damping authority.
    control_temperatures = (421.3, 497.225, 573.15)
    shear_values = tuple((7.49129689 - 0.00764928366 * (temperature - 273.15)) * 1e9 for temperature in control_temperatures)
    omega_nodes = np.geomspace(1e8, evaluator.kb * evaluator.theta / evaluator.hbar, 18)
    structure = {}
    for age in AGES_H:
        structure[age] = {
            "Nv_m-3": float(np.mean([pf[rep][age]["Nv_m-3"] for rep in REPLICATES])),
            "radius_m": float(np.mean([pf[rep][age]["mean_radius_m"] for rep in REPLICATES])),
        }
    lookup = host_lookup(hosts)
    kernel_rates: dict[tuple[str, float, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    conversion: dict[tuple[str, float], float] = {}
    applicable_hosts = sorted(set(str(member["host_id"]) for member in passing if str(member["host_id"]) in direction_map))
    for host_id in applicable_hosts:
        direction = direction_map[host_id]
        for control_T, shear in zip(control_temperatures, shear_values):
            bulk = 18.9e9
            inclusion = kernel.ElasticMedium(8200.0, kernel.isotropic_tensor(bulk - 2.0 * shear / 3.0, shear))
            conv_accum = []
            for age in AGES_H:
                radius = structure[age]["radius_m"]
                shape = kernel.Ellipsoid(np.array([radius, radius, radius]), np.eye(3))
                branch_rates = [np.zeros_like(omega_nodes) for _ in range(3)]
                for branch in range(3):
                    for index, omega in enumerate(omega_nodes):
                        result = kernel.integrated_cross_sections(matrix=matrix, inclusion=inclusion, shape=shape, omega_rad_s=float(omega), incident_direction=direction, incident_branch=branch, n_mu=8, n_phi=16, geometric_cap=True)
                        velocity = lookup[host_id].velocities_m_s[branch]
                        branch_rates[branch][index] = structure[age]["Nv_m-3"] * velocity * float(result["transport_cross_section_m2"])
                        if age == 6.0 and index == len(omega_nodes) // 2:
                            total = float(result["total_cross_section_m2"])
                            conv_accum.append(float(result["converting_cross_section_m2"]) / total if total > 0 else 0.0)
                kernel_rates[(host_id, control_T, age)] = tuple(branch_rates)  # type: ignore[assignment]
            conversion[(host_id, control_T)] = float(np.mean(conv_accum))

    rows: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []
    for member in passing:
        host_id = str(member["host_id"])
        if host_id not in applicable_hosts:
            continue
        host = lookup[host_id]
        for control_index, (control_T, shear) in enumerate(zip(control_temperatures, shear_values)):
            def dynamic_arrays(age: float, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                _, omega, _, _ = evaluator.quadrature(ENDPOINT_T_K)
                raw = kernel_rates[(host_id, control_T, age)]
                return tuple(scale * interpolate_log_rate(omega_nodes, raw[branch], omega) for branch in range(3))  # type: ignore[return-value]

            def mean_kappa(age: float, scale: float) -> float:
                arrays = dynamic_arrays(age, scale)
                return float(np.mean([evaluator.kappa(ENDPOINT_T_K, pf[rep][age]["matrix_xAg"], host, str(member["background_model"]), float(member["parameter"]), branch_dynamic_rates=arrays) for rep in REPLICATES]))

            scale6 = solve_positive_amplitude(lambda scale: mean_kappa(6.0, scale), EXPERIMENT_6H)
            k48 = mean_kappa(48.0, scale6)
            recovery = (k48 - EXPERIMENT_6H) / EXPERIMENT_6H * 100.0
            endpoint_error = abs(k48 - EXPERIMENT_48H) / EXPERIMENT_48H * 100.0
            gate = 10.0 <= recovery <= 30.0 and endpoint_error <= 10.0
            rate_ratio = float(np.median([kernel_rates[(host_id, control_T, 48.0)][branch] / np.maximum(kernel_rates[(host_id, control_T, 6.0)][branch], np.finfo(float).tiny) for branch in range(3)]))
            row = {
                "member_id": member["member_id"], "host_id": host_id, "background_model": member["background_model"], "AQ_case_id": member["case_id"],
                "Ag2Te_density_kg_m3": 8200.0, "Ag2Te_bulk_modulus_GPa": 18.9, "Ag2Te_shear_control_temperature_K": control_T, "Ag2Te_shear_modulus_GPa": shear / 1e9,
                "anisotropy_identity": "ISOTROPIC_SCALAR_CONTROL_AZ_EQ_1_NOT_SINGLE_CRYSTAL_AUTHORITY", "damping_identity": "UNAVAILABLE_NOT_NUMERICALLY_SCANNED",
                "interface_transmission_identity": "ELASTIC_INCLUSION_KERNEL_NO_SEPARATE_FIT", "mean_polarization_conversion_fraction": conversion[(host_id, control_T)],
                "kernel_structure_rate_ratio_48_6": rate_ratio, "frozen_6h_global_rate_scale": scale6, "kappa_6h_W_mK": mean_kappa(6.0, scale6), "kappa_48h_W_mK": k48,
                "recovery_percent": recovery, "endpoint_relative_error_percent": endpoint_error, "numerical_feasibility_gate": gate,
                "material_authority_gate": False, "material_status": "BLOCKED_AG2TE_HIGH_TEMPERATURE_PROPERTIES",
            }
            rows.append(row)
            if gate:
                feasible.append({
                    **row,
                    "scalar_track_central_sample": control_index == 1,
                    "qualified_multidimensional_grid_interior": False,
                    "promotion_allowed": False,
                    "region_identity": "PROVISIONAL_ONE_DIMENSIONAL_NUMERICAL_CONTROL_NOT_MATERIAL_REGION",
                })
    if not rows:
        rows = [{
            "member_id": "NONE", "host_id": "NONE", "background_model": "NONE", "AQ_case_id": "NONE", "Ag2Te_density_kg_m3": "", "Ag2Te_bulk_modulus_GPa": "", "Ag2Te_shear_control_temperature_K": "", "Ag2Te_shear_modulus_GPa": "", "anisotropy_identity": "NOT_RUN_NO_APPLICABLE_PASSING_HOST", "damping_identity": "UNAVAILABLE", "interface_transmission_identity": "UNAVAILABLE", "mean_polarization_conversion_fraction": "", "kernel_structure_rate_ratio_48_6": "", "frozen_6h_global_rate_scale": "", "kappa_6h_W_mK": "", "kappa_48h_W_mK": "", "recovery_percent": "", "endpoint_relative_error_percent": "", "numerical_feasibility_gate": False, "material_authority_gate": False, "material_status": "SKIPPED",
        }]
    # The scalar source track does not span the requested density, cubic
    # anisotropy, damping and interface-transmission axes.  Consequently no
    # row is allowed into the *physical* feasible-region table even when the
    # numerical 10--30%/10% gate is met.
    feasible_rows = [{
        **{key: "" for key in rows[0]},
        "member_id": "NO_QUALIFIED_PHYSICAL_PARAMETER_REGION",
        "anisotropy_identity": "UNIDENTIFIED_HIGH_T_C11_C12_C44",
        "damping_identity": "UNIDENTIFIED_BRANCH_LINEWIDTH",
        "interface_transmission_identity": "UNIDENTIFIED_HIGH_T_OR_TRANSMISSION",
        "numerical_feasibility_gate": bool(feasible),
        "material_authority_gate": False,
        "material_status": "BLOCKED_WIDE_SOURCED_MATERIAL_ENVELOPE_UNAVAILABLE",
        "scalar_track_central_sample": "",
        "qualified_multidimensional_grid_interior": False,
        "promotion_allowed": False,
        "region_identity": "MATERIAL_REGION_NOT_IDENTIFIABLE_FROM_AVAILABLE_SOURCES",
    }]
    write_csv(out / "v3_material_envelope_scan.csv", rows)
    write_csv(out / "v3_feasible_parameter_region.csv", feasible_rows)
    # A central point on a three-sample one-dimensional track is not an
    # interior point of the requested multidimensional material envelope.
    provisional_internal = False
    boundary_only = False
    status = "BLOCKED_V3_WIDE_SOURCED_MATERIAL_ENVELOPE_UNAVAILABLE_NUMERICAL_CONTROL_HAS_LEVERAGE" if feasible else "BLOCKED_V3_WIDE_SOURCED_MATERIAL_ENVELOPE_UNAVAILABLE_NO_NUMERICAL_CONTROL_PASS"
    report = f"""# V3 material-envelope leverage audit

`{status}`

The qualified single-particle Christoffel/Born kernel was exercised as a numerical control. For each endpoint-passing base member, one common nonnegative scale was determined from the 6 h endpoint only; the PF-derived 48 h Nv+mean-radius structural change was then propagated blindly. The scan used the source-recovered scalar track `B=18.9 GPa`, `rho=8200 kg m^-3`, and the reported shear trend across the high-temperature interval. It did not read or optimize to the 48 h endpoint.

- provisional numerical gate passes: `{len(feasible)}` of `{len(rows)}`
- central samples on the one-dimensional scalar track: `{sum(bool(row.get('scalar_track_central_sample')) for row in feasible)}`
- physical material-authority passes: `0`

This cannot be labelled `PASS_V3_MATERIAL_ENVELOPE_FEASIBILITY`. The available sources define only a one-dimensional isotropic scalar control (`rho=8200 kg m^-3`, one `B=18.9 GPa` anchor and a shear trend). They do not define independent source-bounded ranges for high-temperature density, cubic anisotropy, branch damping or interface transmission. Thus the requested wide physical envelope is **not identifiable**, and a middle point on the scalar track is not a multidimensional interior solution. Inventing the missing axes would violate the task. The numerical control shows leverage only; it neither establishes a physical feasible region nor predicts the experimental endpoint.
"""
    (out / "v3_envelope_leverage_report.md").write_text(report, encoding="utf-8")
    return rows, feasible, status, provisional_internal, boundary_only


def make_plots(out: Path, inherited: dict[str, float], ensemble: list[dict[str, Any]], full_rows: list[dict[str, Any]], spectral_rows: list[dict[str, Any]], material_rows: list[dict[str, Any]]) -> None:
    metadata = {"Software": "CUDA_STO_PF V3 endpoint audit"}
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    cap_bars = ax.bar(["6 h cap", "48 h cap"], [inherited["upper6"], inherited["upper48"]], color=["#4C78A8", "#72B7B2"])
    ax.bar_label(cap_bars, labels=[f"{inherited['upper6']:.3f}", f"{inherited['upper48']:.3f}"], padding=3)
    ax.scatter([0, 1], [EXPERIMENT_6H, EXPERIMENT_48H], color="#E45756", zorder=3, label="Sheskin endpoint")
    ax.set_ylabel(r"$\kappa$ (W m$^{-1}$ K$^{-1}$)")
    ax.set_title("Frozen V2 background: hard cap")
    ax.legend(frameon=False)
    fig.savefig(out / "inherited_v2_background_hard_cap.png", dpi=180, metadata=metadata)
    plt.close(fig)

    k48 = np.asarray([float(row["kappa_upper_48_W_mK"]) for row in ensemble])
    colors = ["#4C78A8" if row["host_family"] == "H-P0" else "#F58518" for row in ensemble]
    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    ax.scatter(np.arange(len(k48)), np.sort(k48), c=np.asarray(colors)[np.argsort(k48)], s=18)
    ax.axhline(EXPERIMENT_48H, color="#E45756", ls="--", label="Sheskin 48 h")
    ax.scatter([], [], color="#4C78A8", s=24, label="H-P0 Yu control")
    ax.scatter([], [], color="#F58518", s=24, label="H-P2 provisional host")
    ax.text(0.98, 0.94, f"{int(np.sum(k48 >= EXPERIMENT_48H))}/{len(k48)} pass; all H-P2 proxies", ha="right", va="top", transform=ax.transAxes, fontsize=9)
    ax.set_xlabel("retained AQ host/background member (sorted)")
    ax.set_ylabel(r"48 h no-dynamic upper bound (W m$^{-1}$ K$^{-1}$)")
    ax.set_title("V3 replacement-contract endpoint upper bounds")
    ax.legend(frameon=False)
    fig.savefig(out / "v3_upper_bound_vs_experiment.png", dpi=180, metadata=metadata)
    plt.close(fig)

    grouped: dict[float, list[float]] = {temperature: [] for temperature in TEMPERATURES_K}
    for row in full_rows:
        grouped[float(row["temperature_K"])].append(float(row["kappa_upper_W_mK"]))
    temps = np.asarray(TEMPERATURES_K)
    lower = np.asarray([min(grouped[t]) for t in TEMPERATURES_K])
    median = np.asarray([np.median(grouped[t]) for t in TEMPERATURES_K])
    upper = np.asarray([max(grouped[t]) for t in TEMPERATURES_K])
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.fill_between(temps, lower, upper, color="#4C78A8", alpha=0.2, label="retained V3 upper envelope")
    ax.plot(temps, median, color="#4C78A8", lw=2, label="member/replicate median")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"no-dynamic upper bound (W m$^{-1}$ K$^{-1}$)")
    ax.set_title("V3 upper-bound full-temperature envelope")
    ax.legend(frameon=False)
    fig.savefig(out / "v3_upper_bound_full_temperature.png", dpi=180, metadata=metadata)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    positions = {"D0": 0, "D2": 1, "D4": 2}
    for label, position in positions.items():
        values = [float(row["required_48h_to_6h_rate_scale"]) for row in spectral_rows if row["test_spectrum"] == label]
        ax.scatter(np.full(len(values), position), values, alpha=0.65, s=22)
    ax.axhline(PF_SV_RATIO, color="#54A24B", ls="--", label="PF Sv")
    ax.axhline(PF_MID_Q_RATIO, color="#E45756", ls=":", label="PF mid-q")
    ax.axhline(PF_HIGH_Q_RATIO, color="#B279A2", ls="-.", label="PF high-q")
    ax.set_xticks([0, 1, 2], ["D0", "D2", "D4"])
    ax.set_ylabel("required 48 h / 6 h dynamic rate")
    ax.set_title("Required spectral decay versus PF structural ratios")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.savefig(out / "required_dynamic_decay_vs_pf_ratios.png", dpi=180, metadata=metadata)
    plt.close(fig)

    valid_material = [row for row in material_rows if row.get("recovery_percent") not in ("", None)]
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    if valid_material:
        x = [float(row["Ag2Te_shear_modulus_GPa"]) for row in valid_material]
        y = [float(row["recovery_percent"]) for row in valid_material]
        color = [float(row["endpoint_relative_error_percent"]) for row in valid_material]
        scatter = ax.scatter(x, y, c=color, cmap="viridis", s=25)
        fig.colorbar(scatter, ax=ax, label="48 h endpoint error (%)")
        ax.axhspan(10, 30, color="#54A24B", alpha=0.12)
    else:
        ax.text(0.5, 0.5, "No applicable provisional material scan", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Ag2Te scalar-control shear modulus (GPa)")
    ax.set_ylabel("predicted recovery (%)")
    ax.set_title("Provisional V3 kernel controls (not material authority)")
    fig.savefig(out / "v3_material_envelope_feasible_region.png", dpi=180, metadata=metadata)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    labels = ["V1 6h", "V1 48h", "V2 M0 6h", "V2 M0 48h", "V3 upper max 6h", "V3 upper max 48h"]
    values = [V1_EXPECTED[6.0], V1_EXPECTED[48.0], V2_EXPECTED[("M0", 6.0)], V2_EXPECTED[("M0", 48.0)], max(float(row["kappa_upper_6_W_mK"]) for row in ensemble), max(float(row["kappa_upper_48_W_mK"]) for row in ensemble)]
    ax.bar(np.arange(len(values)), values, color=["#4C78A8", "#4C78A8", "#72B7B2", "#72B7B2", "#F58518", "#F58518"])
    ax.axhline(EXPERIMENT_48H, color="#E45756", ls="--", lw=1, label="Sheskin 48 h")
    ax.set_xticks(np.arange(len(values)), labels, rotation=28, ha="right")
    ax.set_ylabel(r"$\kappa$ (W m$^{-1}$ K$^{-1}$)")
    ax.set_title("V1/V2 frozen results and V3 necessary upper bound")
    ax.legend(frameon=False)
    fig.savefig(out / "v1_v2_v3_upper_bound_comparison.png", dpi=180, metadata=metadata)
    plt.close(fig)


def percentile_text(values: list[float]) -> str:
    array = np.asarray(values)
    return f"{array.min():.6f}/{np.median(array):.6f}/{array.max():.6f}"


def final_reports(out: Path, inherited: dict[str, float], retained: list[dict[str, Any]], ensemble: list[dict[str, Any]], passing: list[dict[str, Any]], spectral: list[dict[str, Any]], decay_status: str, material_status: str, material_rows: list[dict[str, Any]], feasible: list[dict[str, Any]], provisional_internal: bool, boundary_only: bool) -> dict[str, Any]:
    k6 = [float(row["kappa_upper_6_W_mK"]) for row in ensemble]
    k48 = [float(row["kappa_upper_48_W_mK"]) for row in ensemble]
    by_label = {label: [float(row["required_48h_to_6h_rate_scale"]) for row in spectral if row["test_spectrum"] == label] for label in ("D0", "D2", "D4")}
    endpoint_status = "PASS_V3_ABSOLUTE_ENDPOINT_NECESSARY_CONDITION" if passing else "NO_GO_V3_ABSOLUTE_ENDPOINT_UPPER_BOUND"
    source_literal_passing = [row for row in passing if row["case_source_status"] == "SOURCE_LITERAL_BOUND"]
    recommendation = "STOP_OR_RESCOPE_FULL_ATOMISTIC_CAMPAIGN_PENDING_IDENTIFIABLE_MATERIAL_ENVELOPE" if passing else "STOP_EXPENSIVE_AG2TE_PROPERTY_CAMPAIGN"
    final_status = "PASS_V3_ABSOLUTE_ENDPOINT_NECESSARY_CONDITION_MATERIAL_AUTHORITY_BLOCKED" if passing else "NO_GO_V3_ABSOLUTE_ENDPOINT_UPPER_BOUND"
    terminal = f"""V1_reproduction_status=PASS
V2_reproduction_status=PASS

inherited_V2_background_status=NEGATIVE_CONTROL_ONLY
inherited_V2_kappa_upper_6_573p15K={inherited['upper6']:.15f}
inherited_V2_kappa_upper_48_573p15K={inherited['upper48']:.15f}
inherited_V2_max_recovery_percent={inherited['relative']:.9f}
inherited_V2_no_go_status=NO_GO_V3_IF_V2_BACKGROUND_INHERITED

V2_A2_portability_status=V2_A2_IS_MODEL_DEPENDENT_RESIDUAL
V2_A2_transferable_material_constant=false

V3_host_status=H-P0_AVAILABLE_H-P1_UNAVAILABLE_H-P2_PROVISIONAL_HOST_ENVELOPE
V3_background_refit_status=PASS_V3_AQ_BACKGROUND_FROZEN_BEFORE_48H
V3_AQ_background_member_count={len(retained)}
V3_AQ_background_fit_quality=RELATIVE_AICc_SELECTION_WITH_POOR_ABSOLUTE_CHI2_WARNING

V3_kappa_upper_6_min_573p15K={min(k6):.15f}
V3_kappa_upper_6_median_573p15K={np.median(k6):.15f}
V3_kappa_upper_6_max_573p15K={max(k6):.15f}

V3_kappa_upper_48_min_573p15K={min(k48):.15f}
V3_kappa_upper_48_median_573p15K={np.median(k48):.15f}
V3_kappa_upper_48_max_573p15K={max(k48):.15f}

V3_endpoint_necessary_condition_status={endpoint_status}
V3_endpoint_passing_member_count={len(passing)}
V3_endpoint_passing_source_literal_count={len(source_literal_passing)}
V3_endpoint_pass_dependency={'H_P2_PROVISIONAL_AND_SINGLE_APT_DIAGNOSTIC_ONLY' if passing and not source_literal_passing else 'HAS_SOURCE_LITERAL_MEMBER' if source_literal_passing else 'NOT_APPLICABLE'}

required_dynamic_decay_D0={percentile_text(by_label['D0']) if by_label['D0'] else 'NOT_APPLICABLE'}
required_dynamic_decay_D2={percentile_text(by_label['D2']) if by_label['D2'] else 'NOT_APPLICABLE'}
required_dynamic_decay_D4={percentile_text(by_label['D4']) if by_label['D4'] else 'NOT_APPLICABLE'}

PF_Sv_ratio_48_6=0.395306
PF_mid_q_ratio_48_6=0.306247
PF_high_q_ratio_48_6=0.458338

required_vs_PF_structure_status={decay_status}

V3_material_envelope_status={material_status}
V3_internal_feasible_region_exists=false
V3_internal_feasible_region_assessment=NOT_ASSESSABLE_WIDE_SOURCED_MATERIAL_ENVELOPE_UNAVAILABLE
V3_provisional_numerical_control_pass_count={len(feasible)}
V3_boundary_only_solution=false
V3_boundary_only_solution_assessment=NOT_ASSESSABLE_NO_QUALIFIED_MATERIAL_REGION

expensive_atomistic_campaign_recommendation={recommendation}
recommended_next_action=RESCOPE_TO_MINIMUM_SOURCE_DEFINED_HIGH_T_AG2TE_TENSOR_DAMPING_OR_GATE_THEN_REPEAT_PRE_48H_FREEZE
final_status={final_status}
"""
    (out / "final_terminal_output.txt").write_text(terminal, encoding="utf-8")
    report = f"""# V3绝对端点可达性与背景可转移性最终报告

## 结论

本审计得到三个必须分开的结论：

1. **数学NO-GO（继承V2背景）**：`NO_GO_V3_IF_V2_BACKGROUND_INHERITED`。冻结V2 M0在573.15 K的48 h上限为`{inherited['upper48']:.6f}`，低于实验`1.03`；任何新增非负散射都不能把它向上推。
2. **必要条件PASS（替代合同）**：`{endpoint_status}`。重新定义host/background并只用AQ重标定后，`{len(passing)}/{len(ensemble)}`个非概率成员的无动态48 h上限达到1.03。但通过者全部属于H-P2温度外推代理，并且全部依赖`SINGLE_APT_COUNT_FRACTION_DIAGNOSTIC` AQ结构案例；`SOURCE_LITERAL_BOUND`通过数为`{len(source_literal_passing)}`。这只是宽诊断包络中的数学可达性，不能称为正式V3预测。
3. **物理材料可行性尚未PASS**：`{material_status}`。V3单颗粒内核的一维标量控制轨迹有`{len(feasible)}`个数值门通过点，但这不是题目要求的宽、多维、有来源材料包络；高温Ag2Te单晶张量、branch damping和界面OR仍无权威包，因此不存在可登记的“内部物理可行区域”，不得写成`PASS_V3_MATERIAL_ENVELOPE_FEASIBILITY`。

## 核心数值

- V1冻结复现：`1.2980916621 -> 1.2948991542 W m^-1 K^-1`。
- V2 M0冻结复现：`0.8733452622 -> 0.8716818275 W m^-1 K^-1`。
- V2 MIS冻结复现：`0.8208832451 -> 0.8490371923 W m^-1 K^-1`。
- 继承V2背景最大恢复：`{inherited['recovery']:.6f} W m^-1 K^-1`，即`{inherited['relative']:.3f}%`，远低于实验`+21.18%`。
- V3替代合同无动态上限（所有保留成员）6 h min/median/max：`{percentile_text(k6)}`。
- V3替代合同无动态上限（所有保留成员）48 h min/median/max：`{percentile_text(k48)}`。

## 背景身份

V2 A2是在Yu单平均声速host、旧点缺陷合同和density-only AQ对象包络下反演的模型残差，不是可转移材料常数。原样继承只保留为负对照。V3候选B0/B1/B2/B4/BU均在各自host下只用AQ七点重新识别；冻结manifest写出后才加载6/48 h权威状态。所有散射率保持非负，未使用Yu `0.1172768/0.119`位错比例，也未调整`A_N=1.5`。

## 科学边界与决策

V3 cannot solve the amplitude problem by being added on top of the frozen V2 background, because the 48 h no-dynamic-scattering upper bound is already below experiment.

Rebuilding the host/background contract removes the mathematical hard cap, but does not prove that physically realistic Ag2Te parameters will produce the required recovery.

按本任务的决策规则，当前证据不足以建议直接继续完整、昂贵的高温Ag2Te DFT/AIMD/TDEP campaign。应先把工作缩减为不读取Sheskin 48 h的最小材料识别门：取得正定的`C11/C12/C44(T)`、branch/direction速度、阻尼及界面OR稳定性；只有这些量能构成有来源的多维包络并通过Born/T-matrix物理门，才重新冻结V3、重复本审计并决定是否进入完整盲预测。不得声称绝对实验晶格热导率复现。

最终状态：`{final_status}`。
"""
    (out / "final_v3_feasibility_report.md").write_text(report, encoding="utf-8")
    return {"endpoint_status": endpoint_status, "material_status": material_status, "final_status": final_status, "recommendation": recommendation}


def provenance(out: Path, root: Path, v3_root: Path, input_paths: list[Path], pf_hashes: dict[str, str], freeze_manifest: dict[str, Any], final: dict[str, Any]) -> None:
    hashes = {str(path): sha256(path) for path in input_paths}
    hashes.update(pf_hashes)
    lines = [
        "# Provenance", "", "This is a read-only transport audit. It launches no PF, DFT, AIMD or TDEP calculation and modifies no V1/V2 frozen directory or PF authority.", "",
        f"- schema: `{SCHEMA}`", f"- primary repository branch: `{git(root, 'branch', '--show-current')}`", f"- primary repository commit: `{git(root, 'rev-parse', 'HEAD')}`",
        f"- V3 worktree branch: `{git(v3_root, 'branch', '--show-current')}`", f"- V3 worktree commit: `{git(v3_root, 'rev-parse', 'HEAD')}`",
        f"- Python: `{sys.version.replace(chr(10), ' ')}`", f"- platform: `{platform.platform()}`", "", "## Frozen identities", "",
        f"- V1 manifest: `{V1_MANIFEST_SHA256}`", f"- V2 manifest: `{V2_MANIFEST_SHA256}`", f"- AQ freeze status: `{freeze_manifest['status']}`",
        "- A_N: `1.5`; S11/S13: `0`; Yu refit scale: `false`; negative rates: `forbidden`", "", "## Input SHA-256", "",
    ]
    lines.extend(f"- `{digest}`  `{path}`" for path, digest in sorted(hashes.items()))
    lines.extend(["", "## Final identities", "", f"- endpoint: `{final['endpoint_status']}`", f"- material: `{final['material_status']}`", f"- final: `{final['final_status']}`", ""])
    (out / "provenance.md").write_text("\n".join(lines), encoding="utf-8")


def output_ledger(out: Path) -> None:
    require(all((out / name).is_file() for name in REQUIRED_OUTPUTS), "required output missing")
    hashed_outputs = tuple(REQUIRED_OUTPUTS) + ("determinism_audit.json",)
    require(all((out / name).is_file() for name in hashed_outputs), "audit output missing")
    ledger = "".join(f"{sha256(out / name)}  {name}\n" for name in sorted(hashed_outputs))
    (out / "output_sha256.txt").write_text(ledger, encoding="utf-8")


def determinism_audit(out: Path, reference: Path | None) -> None:
    if reference is None:
        value = {"schema": "V3_ENDPOINT_AUDIT_DETERMINISM_V1", "status": "PRIMARY_RUN_AWAITING_INDEPENDENT_REPEAT", "compared_file_count": 0, "mismatches": []}
    else:
        require(reference.is_dir(), f"determinism reference missing: {reference}")
        mismatches = [name for name in REQUIRED_OUTPUTS if sha256(out / name) != sha256(reference / name)]
        value = {"schema": "V3_ENDPOINT_AUDIT_DETERMINISM_V1", "status": "PASS_BYTE_IDENTICAL_INDEPENDENT_REGENERATION" if not mismatches else "FAIL_DETERMINISM", "compared_file_count": len(REQUIRED_OUTPUTS), "mismatches": mismatches, "reference_output_ledger_sha256": sha256(reference / "output_sha256.txt")}
        require(not mismatches, f"determinism mismatches: {mismatches}")
    write_json(out / "determinism_audit.json", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--determinism-reference", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    v3_root = args.v3_root.resolve()
    out = args.out.resolve()
    reference = args.determinism_reference.resolve() if args.determinism_reference else None
    require(not out.exists(), f"refusing to overwrite {out}")
    out.mkdir(parents=True)

    transport_path = root / "scripts/pf_full_psd_no_dislocation_transport_v1.py"
    fit_path = root / "scripts/fit_sheskin_aq_background_v1.py"
    config_path = root / "data/qualification/yu2024_transport_v1/yu_48h_parameters.json"
    experimental_path = root / "reports/sheskin_pf_interface_strain_blind_prediction_v1/sheskin_experimental_kappa.csv"
    transport = load_module(transport_path, "v3_endpoint_transport")
    fit_module = load_module(fit_path, "v3_endpoint_aq_cases")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    transport.validate_yu_base_config(config, config_path)
    require(float(config["shared_parameters"]["A_N"]) == 1.5, "A_N is not 1.5")
    evaluator = BranchDebyeEvaluator(config, GAUSS_FIT)
    hosts = host_members()

    # AQ freeze occurs before the function below is allowed to load PF 6 h/48 h.
    aq_inputs = {
        "experimental_AQ_source_csv": sha256(experimental_path), "AQ_case_generator": sha256(fit_path), "Yu_parameter_contract": sha256(config_path),
        "V3_PbTe_material_contract": sha256(v3_root / "reports/transport_v3_polarization_anisotropic_elastic_wave_v1/material_property_contract.csv"),
    }
    observations = load_aq_observations(experimental_path)
    cases = population_cases(fit_module)
    retained, freeze_manifest = fit_aq_backgrounds(out, evaluator, hosts, cases, observations, aq_inputs)
    require((out / "V3_AQ_BACKGROUND_FROZEN_BEFORE_48H_MANIFEST.json").is_file(), "AQ freeze manifest not written")

    # Endpoint data are loaded only after the AQ manifest exists.
    pf, pf_hashes = load_pf_authority(root)
    reproduce_v1_v2(root, v3_root, out, evaluator, pf, transport, config)
    inherited = inherited_v2_audit(out)
    portability_audit(out)
    ensemble, passing, full_rows = endpoint_upper_bounds(out, evaluator, hosts, retained, pf)
    resistance_rows, spectral_rows, decay_status = dynamic_decay(out, evaluator, hosts, passing, pf) if passing else ([], [], "NOT_APPLICABLE_STAGE_E_NO_GO")
    if not passing:
        write_csv(out / "required_dynamic_resistance_decay.csv", [{"status": "SKIPPED_STAGE_E_NO_GO"}])
        write_csv(out / "required_spectral_decay_D0_D2_D4.csv", [{"status": "SKIPPED_STAGE_E_NO_GO"}])
        (out / "required_vs_pf_structural_decay.md").write_text("# Required dynamic decay\n\n`SKIPPED_STAGE_E_NO_GO`\n", encoding="utf-8")
        material_rows = [{"status": "SKIPPED_STAGE_E_NO_GO"}]
        feasible = []
        write_csv(out / "v3_material_envelope_scan.csv", material_rows)
        write_csv(out / "v3_feasible_parameter_region.csv", material_rows)
        material_status, provisional_internal, boundary_only = "SKIPPED_STAGE_E_NO_GO", False, False
        (out / "v3_envelope_leverage_report.md").write_text("# V3 material envelope\n\n`SKIPPED_STAGE_E_NO_GO`\n", encoding="utf-8")
    else:
        material_rows, feasible, material_status, provisional_internal, boundary_only = material_kernel_scan(out, evaluator, hosts, passing, pf, v3_root)
    final = final_reports(out, inherited, retained, ensemble, passing, spectral_rows, decay_status, material_status, material_rows, feasible, provisional_internal, boundary_only)
    make_plots(out, inherited, ensemble, full_rows, spectral_rows, material_rows)

    input_paths = [
        Path(__file__).resolve(), root / "PROJECT_CORE_MEMORY.md", transport_path, fit_path, config_path, experimental_path,
        v3_root / "frozen_contracts/transport_v1_resolved_density_only/V1_FROZEN_CONTRACT.md",
        v3_root / "frozen_contracts/transport_v1_resolved_density_only/V1_FROZEN_MANIFEST.json",
        v3_root / "frozen_contracts/transport_v2_sheskin_interface_scalar_strain/V2_FROZEN_CONTRACT.md",
        v3_root / "frozen_contracts/transport_v2_sheskin_interface_scalar_strain/V2_FROZEN_MANIFEST.json",
        v3_root / "scripts/anisotropic_elastic_wave_scattering_v1.py",
        v3_root / "reports/transport_v3_polarization_anisotropic_elastic_wave_v1/final_transport_v3_report.md",
        v3_root / "reports/transport_v3_polarization_anisotropic_elastic_wave_v1/material_property_contract.csv",
        v3_root / "reports/transport_v3_polarization_anisotropic_elastic_wave_v1/interface_property_contract.csv",
        v3_root / "reports/transport_v3_polarization_anisotropic_elastic_wave_v1/ag2te_scalar_shear_digitization_v1.csv",
        v3_root / "reports/transport_v3_polarization_anisotropic_elastic_wave_v1/ag2te_high_temperature_property_recovery_audit.md",
        v3_root / "reports/transport_v3_polarization_anisotropic_elastic_wave_v1/single_particle_validation/qualification_manifest.json",
        v3_root / "reports/transport_v3_polarization_anisotropic_elastic_wave_v1/single_particle_validation/determinism_audit.json",
        v3_root / "reports/transport_v3_polarization_anisotropic_elastic_wave_v1/pf_particle_transport_descriptors_v3.csv",
        v3_root / "reports/transport_v3_polarization_anisotropic_elastic_wave_v1/tensor_strain_band_integrals_v3.csv",
        v3_root / "reports/transport_v3_polarization_anisotropic_elastic_wave_v1/shape_approximation_audit.md",
        root / "reports/sheskin_pf_interface_strain_blind_prediction_v1/final_sheskin_blind_prediction_report.md",
        root / "reports/sheskin_pf_interface_strain_blind_prediction_v1/dynamic_channel_leverage_audit.md",
        root / "reports/sheskin_pf_interface_strain_comprehensive_summary_v1/comprehensive_report_zh.md",
    ]
    require(all(path.is_file() for path in input_paths), "provenance input missing")
    provenance(out, root, v3_root, input_paths, pf_hashes, freeze_manifest, final)
    determinism_audit(out, reference)
    output_ledger(out)
    print(final["final_status"])


if __name__ == "__main__":
    main()
