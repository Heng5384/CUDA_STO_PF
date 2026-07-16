#!/usr/bin/env python3
"""Fit independent planar interface-response data with an SPD Onsager matrix.

This is an offline data-gate tool. It does not read PF curved-validation data,
does not provide default coefficients, and does not modify the runtime model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares


REQUIRED_COLUMNS = (
    "record_status",
    "temperature_K",
    "phase_driving",
    "diffusion_driving",
    "measured_Vn",
    "measured_JB",
    "uncertainty_Vn",
    "uncertainty_JB",
    "covariance",
    "interface_orientation",
    "coherency_state",
    "elastic_constraint",
    "source_type",
    "source_reference",
)

ALLOWED_SOURCE_TYPES = {
    "standalone_planar_rsmd",
    "standalone_md",
    "standalone_kmc",
    "planar_thin_film_experiment",
    "planar_diffusion_couple_experiment",
    "planar_interface_migration_experiment",
}

FORBIDDEN_SOURCE_MARKERS = (
    "pf_curved",
    "curved_pf",
    "curved velocity matrix",
    "radius-3",
    "radius_3",
    "seed fate",
    "solver speed",
    "desired growth",
)


class DataContractError(ValueError):
    """Raised when raw interface-response data violate the hard data contract."""


@dataclass(frozen=True)
class Observation:
    temperature_K: float
    phase_driving: float
    diffusion_driving: float
    measured_Vn: float
    measured_JB: float
    uncertainty_Vn: float
    uncertainty_JB: float
    covariance: float
    interface_orientation: str
    coherency_state: str
    elastic_constraint: str
    source_type: str
    source_reference: str
    state_id: str = ""
    replicate_id: str = ""

    @property
    def force(self) -> np.ndarray:
        return np.array([self.phase_driving, self.diffusion_driving], dtype=float)

    @property
    def flux(self) -> np.ndarray:
        return np.array([self.measured_Vn, self.measured_JB], dtype=float)

    @property
    def flux_covariance(self) -> np.ndarray:
        return np.array(
            [
                [self.uncertainty_Vn**2, self.covariance],
                [self.covariance, self.uncertainty_JB**2],
            ],
            dtype=float,
        )


def _finite_number(row: Mapping[str, str], key: str, row_number: int) -> float:
    text = row.get(key, "").strip()
    if not text:
        raise DataContractError(f"row {row_number}: DATA field {key} is blank")
    try:
        value = float(text)
    except ValueError as exc:
        raise DataContractError(f"row {row_number}: {key} is not numeric") from exc
    if not math.isfinite(value):
        raise DataContractError(f"row {row_number}: {key} is not finite")
    return value


def validate_csv_header(fieldnames: Sequence[str] | None) -> None:
    if fieldnames is None:
        raise DataContractError("CSV has no header")
    missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
    if missing:
        raise DataContractError("missing required columns: " + ", ".join(missing))


def validate_source(source_type: str, source_reference: str, row_number: int) -> None:
    if source_type not in ALLOWED_SOURCE_TYPES:
        raise DataContractError(f"row {row_number}: source_type is not allowed: {source_type}")
    if not source_reference or source_reference == "REPLACE_WITH_INDEPENDENT_SOURCE":
        raise DataContractError(f"row {row_number}: source_reference is not an independent source")
    lowered = f"{source_type} {source_reference}".lower()
    marker = next((item for item in FORBIDDEN_SOURCE_MARKERS if item in lowered), None)
    if marker:
        raise DataContractError(f"row {row_number}: forbidden fit source marker: {marker}")


def load_observations(path: Path) -> list[Observation]:
    observations: list[Observation] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_csv_header(reader.fieldnames)
        for row_number, row in enumerate(reader, start=2):
            status = row.get("record_status", "").strip()
            if status == "TEMPLATE":
                continue
            if status != "DATA":
                raise DataContractError(
                    f"row {row_number}: record_status must be DATA or TEMPLATE"
                )
            values = {
                key: _finite_number(row, key, row_number)
                for key in (
                    "temperature_K",
                    "phase_driving",
                    "diffusion_driving",
                    "measured_Vn",
                    "measured_JB",
                    "uncertainty_Vn",
                    "uncertainty_JB",
                    "covariance",
                )
            }
            if values["temperature_K"] <= 0:
                raise DataContractError(f"row {row_number}: temperature_K must be positive")
            if values["uncertainty_Vn"] <= 0 or values["uncertainty_JB"] <= 0:
                raise DataContractError(f"row {row_number}: uncertainties must be positive")
            cov_limit = values["uncertainty_Vn"] * values["uncertainty_JB"]
            if abs(values["covariance"]) >= cov_limit:
                raise DataContractError(
                    f"row {row_number}: flux covariance matrix is not positive definite"
                )
            strings = {}
            for key in (
                "interface_orientation",
                "coherency_state",
                "elastic_constraint",
                "source_type",
                "source_reference",
            ):
                strings[key] = row.get(key, "").strip()
                if not strings[key]:
                    raise DataContractError(f"row {row_number}: {key} is blank")
            validate_source(strings["source_type"], strings["source_reference"], row_number)
            observations.append(
                Observation(
                    **values,
                    **strings,
                    state_id=row.get("state_id", "").strip(),
                    replicate_id=row.get("replicate_id", "").strip(),
                )
            )
    if not observations:
        raise DataContractError(
            "no DATA records: independent interface mobility data are required; "
            "template rows are never accepted as measurements"
        )
    return observations


def design_matrix(observations: Sequence[Observation], model: str) -> np.ndarray:
    rows: list[list[float]] = []
    for obs in observations:
        v, j = obs.flux
        if model == "full_spd":
            rows.extend(([v, j, 0.0], [0.0, v, j]))
        elif model == "fixed_A_spd":
            rows.extend(([j, 0.0], [v, j]))
        elif model == "fixed_A_diagonal":
            rows.append([j])
        else:
            raise ValueError(f"unknown model: {model}")
    return np.asarray(rows, dtype=float)


def design_rank(observations: Sequence[Observation], model: str) -> tuple[int, int, np.ndarray]:
    matrix = design_matrix(observations, model)
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = max(matrix.shape) * np.finfo(float).eps * (singular[0] if singular.size else 0.0)
    rank = int(np.sum(singular > tolerance))
    required = {"full_spd": 3, "fixed_A_spd": 2, "fixed_A_diagonal": 1}[model]
    return rank, required, singular


def _linear_initial_matrix(observations: Sequence[Observation]) -> np.ndarray:
    h = design_matrix(observations, "full_spd")
    x = np.concatenate([obs.force for obs in observations])
    coefficients, *_ = np.linalg.lstsq(h, x, rcond=None)
    matrix = np.array(
        [[coefficients[0], coefficients[1]], [coefficients[1], coefficients[2]]],
        dtype=float,
    )
    eig, vec = np.linalg.eigh(matrix)
    floor = max(float(np.max(np.abs(eig))) * 1.0e-9, 1.0e-12)
    return vec @ np.diag(np.maximum(eig, floor)) @ vec.T


def _matrix_from_parameters(parameters: np.ndarray, model: str, fixed_A: float | None) -> np.ndarray:
    if model == "full_spd":
        lower = np.array(
            [[math.exp(parameters[0]), 0.0], [parameters[1], math.exp(parameters[2])]],
            dtype=float,
        )
        return lower @ lower.T
    if fixed_A is None or not math.isfinite(fixed_A) or fixed_A <= 0:
        raise DataContractError(f"{model} requires a positive independently fixed A_I")
    if model == "fixed_A_spd":
        b = parameters[0]
        schur = math.exp(parameters[1])
        return np.array([[fixed_A, b], [b, b * b / fixed_A + schur]], dtype=float)
    if model == "fixed_A_diagonal":
        return np.array([[fixed_A, 0.0], [0.0, math.exp(parameters[0])]], dtype=float)
    raise ValueError(f"unknown model: {model}")


def _initial_parameters(
    observations: Sequence[Observation], model: str, fixed_A: float | None
) -> np.ndarray:
    matrix = _linear_initial_matrix(observations)
    if model == "full_spd":
        lower = np.linalg.cholesky(matrix)
        return np.array([math.log(lower[0, 0]), lower[1, 0], math.log(lower[1, 1])])
    if fixed_A is None or fixed_A <= 0:
        raise DataContractError(f"{model} requires a positive independently fixed A_I")
    h = design_matrix(observations, "fixed_A_spd")
    target: list[float] = []
    for obs in observations:
        v, _j = obs.flux
        target.extend((obs.phase_driving - fixed_A * v, obs.diffusion_driving))
    b, c = np.linalg.lstsq(h, np.asarray(target), rcond=None)[0]
    c_floor = b * b / fixed_A + max(abs(c) * 1.0e-9, 1.0e-12)
    c = max(c, c_floor)
    if model == "fixed_A_spd":
        return np.array([b, math.log(c - b * b / fixed_A)])
    return np.array([math.log(max(c, 1.0e-12))])


def _whitened_residuals(
    parameters: np.ndarray,
    observations: Sequence[Observation],
    model: str,
    fixed_A: float | None,
) -> np.ndarray:
    matrix = _matrix_from_parameters(parameters, model, fixed_A)
    inverse = np.linalg.inv(matrix)
    residuals: list[float] = []
    for obs in observations:
        predicted_flux = inverse @ obs.force
        lower = np.linalg.cholesky(obs.flux_covariance)
        residuals.extend(np.linalg.solve(lower, predicted_flux - obs.flux))
    return np.asarray(residuals)


def _coefficient_vector(matrix: np.ndarray) -> np.ndarray:
    return np.array([matrix[0, 0], matrix[0, 1], matrix[1, 1]], dtype=float)


def _coefficient_jacobian(
    parameters: np.ndarray, model: str, fixed_A: float | None
) -> np.ndarray:
    baseline = _coefficient_vector(_matrix_from_parameters(parameters, model, fixed_A))
    jacobian = np.zeros((3, parameters.size), dtype=float)
    for index in range(parameters.size):
        step = math.sqrt(np.finfo(float).eps) * max(1.0, abs(parameters[index]))
        perturbed = parameters.copy()
        perturbed[index] += step
        jacobian[:, index] = (
            _coefficient_vector(_matrix_from_parameters(perturbed, model, fixed_A)) - baseline
        ) / step
    return jacobian


def fit_temperature_group(
    observations: Sequence[Observation], model: str, fixed_A: float | None = None
) -> dict[str, object]:
    rank, required_rank, singular = design_rank(observations, model)
    if rank < required_rank:
        raise DataContractError(
            f"independent data rank insufficient for {model}: rank={rank}, required={required_rank}"
        )
    initial = _initial_parameters(observations, model, fixed_A)
    result = least_squares(
        _whitened_residuals,
        initial,
        args=(observations, model, fixed_A),
        method="trf",
        x_scale="jac",
        max_nfev=20000,
        ftol=1.0e-13,
        xtol=1.0e-13,
        gtol=1.0e-13,
    )
    if not result.success:
        raise DataContractError(f"SPD fit did not converge: {result.message}")
    matrix = _matrix_from_parameters(result.x, model, fixed_A)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if not np.all(eigenvalues > 0):
        raise DataContractError("internal error: Cholesky fit returned a non-SPD matrix")
    dof = max(0, result.fun.size - result.x.size)
    reduced_chi2 = float(np.dot(result.fun, result.fun) / dof) if dof else math.nan
    # Observation covariances are absolute, not relative weights. Scaling by
    # reduced chi-square would incorrectly collapse uncertainty for exact data.
    parameter_covariance = np.linalg.pinv(result.jac.T @ result.jac)
    coefficient_map = _coefficient_jacobian(result.x, model, fixed_A)
    coefficient_covariance = coefficient_map @ parameter_covariance @ coefficient_map.T
    b_sigma = math.sqrt(max(0.0, coefficient_covariance[1, 1]))
    b_status = (
        "REQUIRED_AT_95_PERCENT"
        if model != "fixed_A_diagonal" and abs(matrix[0, 1]) > 1.96 * b_sigma
        else "NOT_RESOLVED_FROM_ZERO"
    )
    return {
        "model": model,
        "n_observations": len(observations),
        "design_rank": rank,
        "required_rank": required_rank,
        "design_singular_values": singular.tolist(),
        "matrix": matrix.tolist(),
        "A_I": float(matrix[0, 0]),
        "B_I": float(matrix[0, 1]),
        "C_I": float(matrix[1, 1]),
        "eigenvalues": eigenvalues.tolist(),
        "minimum_SPD_margin": float(eigenvalues[0]),
        "determinant": float(np.linalg.det(matrix)),
        "coefficient_covariance_ABC": coefficient_covariance.tolist(),
        "reduced_chi2": reduced_chi2,
        "cross_coupling_status": b_status,
        "optimizer_message": result.message,
    }


def fit_by_temperature(
    observations: Sequence[Observation],
    model: str,
    fixed_A_by_temperature: Mapping[float, float] | None = None,
) -> dict[str, object]:
    grouped: dict[float, list[Observation]] = {}
    for observation in observations:
        grouped.setdefault(observation.temperature_K, []).append(observation)
    output: dict[str, object] = {}
    for temperature, rows in sorted(grouped.items()):
        fixed_A = None
        if model != "full_spd":
            if fixed_A_by_temperature is None or temperature not in fixed_A_by_temperature:
                raise DataContractError(f"missing fixed A_I for temperature {temperature:.12g} K")
            fixed_A = float(fixed_A_by_temperature[temperature])
        output[f"{temperature:.12g}"] = fit_temperature_group(rows, model, fixed_A)
    return output


def _load_fixed_a(path: Path) -> dict[float, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise DataContractError("fixed-A JSON must be an object mapping temperature_K to A_I")
    values = {float(key): float(value) for key, value in raw.items()}
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        raise DataContractError("every fixed A_I must be finite and positive")
    return values


def _write_summary_csv(path: Path, fits: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "temperature_K",
                "model",
                "n_observations",
                "design_rank",
                "required_rank",
                "A_I",
                "B_I",
                "C_I",
                "determinant",
                "minimum_SPD_margin",
                "reduced_chi2",
                "cross_coupling_status",
            ),
        )
        writer.writeheader()
        for temperature, fit in fits.items():
            writer.writerow({"temperature_K": temperature, **fit})


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument(
        "--model",
        choices=("full_spd", "fixed_A_spd", "fixed_A_diagonal"),
        default="fixed_A_spd",
    )
    parser.add_argument(
        "--fixed-a-json",
        type=Path,
        help="Required for fixed_A models; maps temperature_K strings to positive A_I values.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        observations = load_observations(args.input_csv)
        fixed_a = _load_fixed_a(args.fixed_a_json) if args.fixed_a_json else None
        if args.model != "full_spd" and fixed_a is None:
            raise DataContractError("--fixed-a-json is required for fixed_A models")
        fits = fit_by_temperature(observations, args.model, fixed_a)
    except (DataContractError, OSError, json.JSONDecodeError) as exc:
        print(f"interface_matrix_fit_status=REJECTED")
        print(f"reason={exc}")
        return 2
    payload = {
        "fit_contract": "independent_planar_interface_onsager_v1",
        "normal_direction": "beta_to_matrix",
        "positive_Vn": "beta_growth",
        "positive_JB": "matrix_to_beta",
        "dividing_surface": "total_C_equimolar_h_volume",
        "fits": fits,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_summary_csv(args.output_csv, fits)
    print("interface_matrix_fit_status=PASS_SPD_FIT")
    print(f"temperatures_fitted={len(fits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
