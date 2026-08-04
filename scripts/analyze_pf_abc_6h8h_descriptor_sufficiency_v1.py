#!/usr/bin/env python3
"""Compare compact PF microstructure descriptors on real A/B/C 6--8 h data.

The full resolved-particle PSD is the reference.  The five evaluated models
are:

1. Nv + mean R, represented by a monodisperse population;
2. Nv + mean R + CV, represented by a deterministic lognormal closure;
3. Sv, used in the geometric-scattering limit;
4. Sv + M6, represented by the existing two-moment equivalent population;
5. the complete resolved PF PSD.

This is a read-only descriptor-sufficiency diagnostic.  It neither changes PF
fields nor claims an absolute experimental thermal-conductivity reproduction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import pf_full_psd_no_dislocation_transport_v1 as transport  # noqa: E402


SCHEMA = "PF_ABC_6H8H_DESCRIPTOR_SUFFICIENCY_V1"
STATUS = "PASS_PF_ABC_6H8H_DESCRIPTOR_SUFFICIENCY_V1"
SHORT_PASS = "PASS_246CUBE_6H8H_SHORT_SCREENING_V1"
MERGE_PASS = "PASS_246CUBE_RESOLVED_MERGE_AWARE_LINEAGE_V1"
REPLICATES = ("A", "B", "C")
TEMPERATURES_K = (300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0)
MATRIX_MODES = ("fixed_6h_matrix", "pf_time_varying_matrix")
MODELS = (
    "Nv_plus_mean_R",
    "Nv_plus_mean_R_plus_CV_lognormal",
    "Sv_geometric_limit",
    "Sv_plus_M6",
    "full_PSD",
)
LOGNORMAL_ORDER = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), np.finfo(float).tiny)


def fmt(value: float) -> str:
    return f"{value:.17g}"


def numeric_row(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (float, np.floating)):
            result[key] = fmt(float(value))
        elif isinstance(value, (int, np.integer)):
            result[key] = str(int(value))
        else:
            result[key] = value
    return result


def load_inputs(input_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observable_path = input_root / "ensemble_observables_6h_8h.csv"
    observable_rows = read_csv(observable_path)
    expected_steps: list[int] | None = None
    snapshots: list[dict[str, Any]] = []
    max_closure = {"count": 0.0, "mean_R": 0.0, "Sv": 0.0, "M6": 0.0}
    source_files: dict[str, str] = {
        str(observable_path): sha256(observable_path),
    }

    rows_by_replicate: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in observable_rows:
        rows_by_replicate[row["replicate"]].append(row)

    for replicate in REPLICATES:
        label = f"replicate_{replicate}"
        rows = sorted(rows_by_replicate[label], key=lambda item: int(item["step"]))
        steps = [int(row["step"]) for row in rows]
        if expected_steps is None:
            expected_steps = steps
        elif steps != expected_steps:
            raise ValueError(f"A/B/C snapshot steps differ for replicate {replicate}")
        if not rows or int(rows[0]["step"]) != 0 or int(rows[-1]["step"]) != 7266:
            raise ValueError(f"replicate {replicate} does not span registered 6--8 h")

        audit_path = input_root / f"short_6h_8h_{replicate}_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != SHORT_PASS or not all(audit.get("gates", {}).values()):
            raise ValueError(f"replicate {replicate} short screening is not PASS")
        source_files[str(audit_path)] = sha256(audit_path)

        input_hash_path = input_root / f"short_6h_8h_{replicate}_input_hashes.sha256"
        source_files[str(input_hash_path)] = sha256(input_hash_path)

        particle_path = input_root / f"particle_lineage_{replicate}.csv"
        source_files[str(particle_path)] = sha256(particle_path)
        particle_rows = read_csv(particle_path)
        particles_by_step: dict[int, list[dict[str, str]]] = defaultdict(list)
        for particle in particle_rows:
            particles_by_step[int(particle["step"])].append(particle)

        for row in rows:
            step = int(row["step"])
            group = particles_by_step.get(step, [])
            expected_count = int(row["particle_count"])
            if len(group) != expected_count:
                raise ValueError(
                    f"replicate {replicate} step {step} PSD count mismatch: "
                    f"{len(group)} != {expected_count}"
                )
            radii_nm = np.asarray(
                [float(item["equivalent_radius_nm"]) for item in group], dtype=float
            )
            if radii_nm.size == 0 or np.any(~np.isfinite(radii_nm)) or np.any(radii_nm <= 0):
                raise ValueError(f"invalid radius population at {replicate} step {step}")
            box_volume_nm3 = 246.0**3
            moments = transport.moments_from_radii(
                radii_nm * 1.0e-9, box_volume_nm3 * 1.0e-27
            )
            mean_nm = float(np.mean(radii_nm))
            std_nm = float(np.std(radii_nm, ddof=0))
            cv = std_nm / mean_nm
            source_values = {
                "mean_R": float(row["mean_radius_nm"]),
                "Sv": float(row["Sv_nm_inv"]) * 1.0e9,
                "M6": float(row["M6_nm3"]) * 1.0e-27,
            }
            calculated = {
                "mean_R": mean_nm,
                "Sv": moments["Sv_m-1"],
                "M6": moments["M6_m3"],
            }
            for name in source_values:
                max_closure[name] = max(
                    max_closure[name],
                    relative_error(calculated[name], source_values[name]),
                )
            max_closure["count"] = max(
                max_closure["count"], abs(radii_nm.size - expected_count)
            )
            snapshots.append(
                {
                    "replicate": replicate,
                    "step": step,
                    "age_h": float(row["experimental_age_h"]),
                    "elapsed_physical_time_s": float(row["elapsed_physical_time_s"]),
                    "particle_count": expected_count,
                    "box_volume_nm3": box_volume_nm3,
                    "Nv_m-3": moments["Nv_m-3"],
                    "mean_radius_nm": mean_nm,
                    "std_radius_nm": std_nm,
                    "CV": cv,
                    "Sv_nm-1": moments["Sv_m-1"] * 1.0e-9,
                    "M6_nm3": moments["M6_m3"] * 1.0e27,
                    "matrix_xAg": float(row["far_field_matrix_xAg"]),
                    "matrix_xB": float(row["far_field_matrix_xB"]),
                    "radii_nm": radii_nm.tolist(),
                    "full_psd_sha256": transport.canonical_sha256(
                        sorted(float(value) for value in radii_nm)
                    ),
                }
            )

    merge_path = input_root / "merge_aware_C_audit.json"
    merge = json.loads(merge_path.read_text(encoding="utf-8"))
    if merge.get("status") != MERGE_PASS or not all(merge.get("gates", {}).values()):
        raise ValueError("replicate C merge-aware supplemental audit is not PASS")
    source_files[str(merge_path)] = sha256(merge_path)
    merge_provenance = input_root / "merge_aware_C_analysis_provenance.sha256"
    source_files[str(merge_provenance)] = sha256(merge_provenance)

    if expected_steps is None or len(expected_steps) != 17 or len(snapshots) != 51:
        raise ValueError("expected exactly 17 snapshots per replicate and 51 total")
    if max_closure["count"] != 0.0 or any(
        max_closure[name] > 5.0e-12 for name in ("mean_R", "Sv", "M6")
    ):
        raise ValueError(f"source PSD descriptor closure failed: {max_closure}")
    snapshots.sort(key=lambda item: (item["replicate"], item["step"]))
    return snapshots, {
        "source_files": source_files,
        "snapshot_steps": expected_steps,
        "snapshot_count": len(snapshots),
        "max_source_descriptor_closure_relative": max_closure,
        "C_merge_aware_status": merge["status"],
    }


def lognormal_rate(
    omega: np.ndarray,
    radii_m: np.ndarray,
    box_volume_m3: float,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    moments = transport.moments_from_radii(radii_m, box_volume_m3)
    mean_radius = float(np.mean(radii_m))
    cv = float(np.std(radii_m, ddof=0) / mean_radius)
    v = float(config["shared_parameters"]["average_sound_velocity_m_s"])
    if cv <= 1.0e-15:
        cross = transport.precipitate_cross_section(
            omega, np.asarray([mean_radius]), config
        )[:, 0]
        return v * moments["Nv_m-3"] * cross, {
            "quadrature_mean_relative_error": 0.0,
            "quadrature_cv_absolute_error": 0.0,
        }
    sigma2 = math.log1p(cv**2)
    sigma = math.sqrt(sigma2)
    mu = math.log(mean_radius) - 0.5 * sigma2
    nodes, weights = hermgauss(LOGNORMAL_ORDER)
    probabilities = weights / math.sqrt(math.pi)
    quadrature_radii = np.exp(mu + math.sqrt(2.0) * sigma * nodes)
    cross = transport.precipitate_cross_section(omega, quadrature_radii, config)
    # Use an explicit weighted reduction rather than a BLAS dot product.  Some
    # Accelerate builds emit spurious floating-point warnings when Hermite
    # weights span many orders of magnitude even though every operand and the
    # result are finite.
    expected_cross = np.sum(cross * probabilities[np.newaxis, :], axis=1)
    q_mean = float(np.sum(probabilities * quadrature_radii))
    q_variance = float(
        np.sum(probabilities * (quadrature_radii - q_mean) ** 2)
    )
    q_cv = math.sqrt(max(q_variance, 0.0)) / q_mean
    return v * moments["Nv_m-3"] * expected_cross, {
        "quadrature_mean_relative_error": relative_error(q_mean, mean_radius),
        "quadrature_cv_absolute_error": abs(q_cv - cv),
    }


def precipitate_rate(
    omega: np.ndarray,
    radii_m: np.ndarray,
    box_volume_m3: float,
    config: dict[str, Any],
    model: str,
) -> tuple[np.ndarray, dict[str, float]]:
    if model == "full_PSD":
        return (
            transport.descriptor_precipitate_rate(
                omega, radii_m, box_volume_m3, config, "full_psd"
            ),
            {},
        )
    if model == "Nv_plus_mean_R":
        return (
            transport.descriptor_precipitate_rate(
                omega,
                radii_m,
                box_volume_m3,
                config,
                "Nv_plus_mean_R_monodisperse",
            ),
            {},
        )
    if model == "Nv_plus_mean_R_plus_CV_lognormal":
        return lognormal_rate(omega, radii_m, box_volume_m3, config)
    if model == "Sv_geometric_limit":
        return (
            transport.descriptor_precipitate_rate(
                omega, radii_m, box_volume_m3, config, "Sv_geometric_limit"
            ),
            {},
        )
    if model == "Sv_plus_M6":
        return (
            transport.descriptor_precipitate_rate(
                omega,
                radii_m,
                box_volume_m3,
                config,
                "Sv_plus_M6_moment_reconstruction",
            ),
            {},
        )
    raise ValueError(f"unknown descriptor model: {model}")


def integrate_kappa(
    temperature_K: float,
    matrix_xag: float,
    radii_m: np.ndarray,
    box_volume_m3: float,
    config: dict[str, Any],
    model: str,
) -> tuple[float, dict[str, float]]:
    order = int(config["numerics"]["gauss_legendre_order"])
    nodes, weights = leggauss(order)
    x_max = float(config["shared_parameters"]["debye_temperature_K"]) / temperature_K
    x = 0.5 * (nodes + 1.0) * x_max
    constants = config["physical_constants"]
    shared = config["shared_parameters"]
    omega = (
        x
        * float(constants["k_B_J_K"])
        * temperature_K
        / float(constants["hbar_J_s"])
    )
    base = transport.base_scattering_rates(omega, temperature_K, matrix_xag, config)
    precipitate, diagnostics = precipitate_rate(
        omega, radii_m, box_volume_m3, config, model
    )
    total = base["phonon_phonon"] + base["boundary"] + base["point_defect"] + precipitate
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
    values = prefactor * transport.bose_weight(x) / total
    return float(0.5 * x_max * np.sum(weights * values)), diagnostics


def exact_rank(values: dict[str, float]) -> tuple[str, ...]:
    return tuple(sorted(values, key=lambda key: (values[key], key)))


def analyze(
    input_root: Path,
    output_root: Path,
    yu_config_path: Path,
    transport_contract_path: Path,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    config = json.loads(yu_config_path.read_text(encoding="utf-8"))
    contract = json.loads(transport_contract_path.read_text(encoding="utf-8"))
    transport.validate_yu_base_config(config, yu_config_path)
    transport.validate_interface_contract(
        contract, transport_contract_path, yu_config_path
    )
    snapshots, input_audit = load_inputs(input_root)
    source_files = input_audit["source_files"]
    source_files[str(yu_config_path)] = sha256(yu_config_path)
    source_files[str(transport_contract_path)] = sha256(transport_contract_path)
    source_files[str(SCRIPT_DIR / "pf_full_psd_no_dislocation_transport_v1.py")] = sha256(
        SCRIPT_DIR / "pf_full_psd_no_dislocation_transport_v1.py"
    )
    source_files[str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())

    references: dict[str, float] = {}
    for snapshot in snapshots:
        if snapshot["step"] == 0:
            references[snapshot["replicate"]] = float(snapshot["matrix_xAg"])
    if set(references) != set(REPLICATES):
        raise ValueError("missing replicate-specific 6 h matrix references")

    moment_rows = []
    prediction_rows: list[dict[str, Any]] = []
    max_lognormal_mean_error = 0.0
    max_lognormal_cv_error = 0.0
    for snapshot in snapshots:
        radii_m = np.asarray(snapshot["radii_nm"], dtype=float) * 1.0e-9
        box_volume_m3 = float(snapshot["box_volume_nm3"]) * 1.0e-27
        moment_rows.append(
            numeric_row(
                {
                    key: snapshot[key]
                    for key in (
                        "replicate",
                        "step",
                        "age_h",
                        "elapsed_physical_time_s",
                        "particle_count",
                        "Nv_m-3",
                        "mean_radius_nm",
                        "std_radius_nm",
                        "CV",
                        "Sv_nm-1",
                        "M6_nm3",
                        "matrix_xAg",
                        "matrix_xB",
                        "full_psd_sha256",
                    )
                }
            )
        )
        for matrix_mode in MATRIX_MODES:
            matrix_xag = (
                references[snapshot["replicate"]]
                if matrix_mode == "fixed_6h_matrix"
                else float(snapshot["matrix_xAg"])
            )
            for temperature in TEMPERATURES_K:
                values: dict[str, float] = {}
                for model in MODELS:
                    value, diagnostics = integrate_kappa(
                        temperature,
                        matrix_xag,
                        radii_m,
                        box_volume_m3,
                        config,
                        model,
                    )
                    values[model] = value
                    max_lognormal_mean_error = max(
                        max_lognormal_mean_error,
                        diagnostics.get("quadrature_mean_relative_error", 0.0),
                    )
                    max_lognormal_cv_error = max(
                        max_lognormal_cv_error,
                        diagnostics.get("quadrature_cv_absolute_error", 0.0),
                    )
                reference = values["full_PSD"]
                for model in MODELS:
                    prediction_rows.append(
                        {
                            "replicate": snapshot["replicate"],
                            "step": snapshot["step"],
                            "age_h": snapshot["age_h"],
                            "temperature_K": temperature,
                            "matrix_mode": matrix_mode,
                            "matrix_xAg_used": matrix_xag,
                            "descriptor_model": model,
                            "kappa_W_mK": values[model],
                            "full_PSD_kappa_W_mK": reference,
                            "kappa_relative_error": relative_error(values[model], reference),
                        }
                    )

    initial_values: dict[tuple[str, float, str, str], float] = {}
    full_delta_lookup: dict[tuple[str, int, float, str], float] = {}
    for row in prediction_rows:
        key = (
            str(row["replicate"]),
            float(row["temperature_K"]),
            str(row["matrix_mode"]),
            str(row["descriptor_model"]),
        )
        if int(row["step"]) == 0:
            initial_values[key] = float(row["kappa_W_mK"])
    for row in prediction_rows:
        key = (
            str(row["replicate"]),
            float(row["temperature_K"]),
            str(row["matrix_mode"]),
            str(row["descriptor_model"]),
        )
        delta = float(row["kappa_W_mK"]) - initial_values[key]
        row["delta_kappa_from_6h_W_mK"] = delta
        if row["descriptor_model"] == "full_PSD":
            full_delta_lookup[
                (
                    str(row["replicate"]),
                    int(row["step"]),
                    float(row["temperature_K"]),
                    str(row["matrix_mode"]),
                )
            ] = delta
    for row in prediction_rows:
        full_delta = full_delta_lookup[
            (
                str(row["replicate"]),
                int(row["step"]),
                float(row["temperature_K"]),
                str(row["matrix_mode"]),
            )
        ]
        row["full_PSD_delta_kappa_from_6h_W_mK"] = full_delta
        row["delta_kappa_error_W_mK"] = (
            float(row["delta_kappa_from_6h_W_mK"]) - full_delta
        )

    summary_rows: list[dict[str, Any]] = []
    for matrix_mode in MATRIX_MODES:
        for model in MODELS:
            rows = [
                row
                for row in prediction_rows
                if row["matrix_mode"] == matrix_mode
                and row["descriptor_model"] == model
            ]
            kappa_errors = np.asarray(
                [float(row["kappa_relative_error"]) for row in rows], dtype=float
            )
            evolution = [row for row in rows if int(row["step"]) != 0]
            delta_errors = np.asarray(
                [float(row["delta_kappa_error_W_mK"]) for row in evolution],
                dtype=float,
            )
            full_deltas = np.asarray(
                [
                    float(row["full_PSD_delta_kappa_from_6h_W_mK"])
                    for row in evolution
                ],
                dtype=float,
            )
            signal_rms = float(np.sqrt(np.mean(full_deltas**2)))
            delta_nrmse = (
                float(np.sqrt(np.mean(delta_errors**2))) / signal_rms
                if signal_rms > 0.0
                else 0.0
            )
            rank_matches = 0
            rank_total = 0
            for step in input_audit["snapshot_steps"]:
                if step == 0:
                    continue
                for temperature in TEMPERATURES_K:
                    subset = [
                        row
                        for row in rows
                        if int(row["step"]) == step
                        and float(row["temperature_K"]) == temperature
                    ]
                    model_values = {
                        str(row["replicate"]): float(row["delta_kappa_from_6h_W_mK"])
                        for row in subset
                    }
                    full_values = {
                        str(row["replicate"]): float(
                            row["full_PSD_delta_kappa_from_6h_W_mK"]
                        )
                        for row in subset
                    }
                    if set(model_values) != set(REPLICATES):
                        raise ValueError("incomplete A/B/C rank comparison cell")
                    rank_matches += int(exact_rank(model_values) == exact_rank(full_values))
                    rank_total += 1
            summary_rows.append(
                {
                    "matrix_mode": matrix_mode,
                    "descriptor_model": model,
                    "kappa_MAPE_percent": 100.0 * float(np.mean(kappa_errors)),
                    "kappa_RMS_relative_percent": 100.0
                    * float(np.sqrt(np.mean(kappa_errors**2))),
                    "kappa_max_relative_percent": 100.0 * float(np.max(kappa_errors)),
                    "delta_kappa_signal_NRMSE_percent": 100.0 * delta_nrmse,
                    "delta_kappa_max_abs_error_W_mK": float(
                        np.max(np.abs(delta_errors))
                    ),
                    "ABC_delta_order_exact_fraction": rank_matches / rank_total,
                    "comparison_cells": len(rows),
                }
            )

    prediction_csv_rows = [numeric_row(row) for row in prediction_rows]
    summary_csv_rows = [numeric_row(row) for row in summary_rows]
    write_csv(
        output_root / "source_snapshot_descriptors.csv",
        list(moment_rows[0]),
        moment_rows,
    )
    write_csv(
        output_root / "descriptor_predictions.csv",
        list(prediction_csv_rows[0]),
        prediction_csv_rows,
    )
    write_csv(
        output_root / "descriptor_summary.csv",
        list(summary_csv_rows[0]),
        summary_csv_rows,
    )

    def table_rows(matrix_mode: str) -> str:
        lines = []
        for row in summary_rows:
            if row["matrix_mode"] != matrix_mode:
                continue
            lines.append(
                "| {descriptor_model} | {mape:.6f}% | {maximum:.6f}% | "
                "{signal:.3f}% | {delta:.6g} | {rank:.3f} |".format(
                    descriptor_model=row["descriptor_model"],
                    mape=row["kappa_MAPE_percent"],
                    maximum=row["kappa_max_relative_percent"],
                    signal=row["delta_kappa_signal_NRMSE_percent"],
                    delta=row["delta_kappa_max_abs_error_W_mK"],
                    rank=row["ABC_delta_order_exact_fraction"],
                )
            )
        return "\n".join(lines)

    compressed = [
        row for row in summary_rows if row["descriptor_model"] != "full_PSD"
    ]
    best_absolute = min(compressed, key=lambda row: row["kappa_MAPE_percent"])
    best_signal = min(
        compressed, key=lambda row: row["delta_kappa_signal_NRMSE_percent"]
    )
    report = f"""# Real A/B/C 6--8 h descriptor sufficiency comparison

## Scope and data authority

This read-only analysis uses all 51 real short-screening snapshots (17 per
replicate) from `pf_246cube_three_seed_library_handoff_v1`.  A, B, and C share
the same 246^3 box, 380 C physics, elastic contract, `dt_code=0.02`, and
6--8 h time grid.  All three short screenings PASS; the one C diffuse-tail
merge has the frozen merge-aware PASS.  Only snapshot PSDs are used, so no
new independent post-merge identity claim is made.

The source fixture family has a 6 h far-field Ag fraction near 0.006788, not
the later experiment-anchored 0.0062 Method-1 fixture.  Therefore this report
tests descriptor information loss only.  It is not an APT concentration-fit
or absolute experimental thermal-conductivity result.

## Mathematical closures

- `Nv_plus_mean_R`: monodisperse population with the measured `Nv` and mean R.
- `Nv_plus_mean_R_plus_CV_lognormal`: lognormal population matching measured
  `Nv`, mean R, and population CV; {LOGNORMAL_ORDER}-point Gauss-Hermite
  quadrature evaluates its scattering integral.
- `Sv_geometric_limit`: the established high-frequency geometric limit using
  `Sv` alone.
- `Sv_plus_M6`: the established equivalent population matching both moments.
- `full_PSD`: direct sum over every resolved particle; this is the reference.

The transport host is the frozen Yu 48 h non-particle baseline with `A_N=1.5`,
S11/S13 dislocation scattering off, and no refitted scale.  Seven transport
temperatures from 300 to 600 K are evaluated.  `fixed_6h_matrix` isolates PSD
evolution; `pf_time_varying_matrix` also propagates the PF far-field Ag value
through point-defect scattering.

## Fixed 6 h matrix results

| Descriptor | absolute kappa MAPE | maximum kappa error | delta-kappa signal NRMSE | max abs delta error (W/mK) | exact A/B/C delta ordering |
|---|---:|---:|---:|---:|---:|
{table_rows('fixed_6h_matrix')}

## PF time-varying matrix results

| Descriptor | absolute kappa MAPE | maximum kappa error | delta-kappa signal NRMSE | max abs delta error (W/mK) | exact A/B/C delta ordering |
|---|---:|---:|---:|---:|---:|
{table_rows('pf_time_varying_matrix')}

## Interpretation

- Best compressed model by absolute-kappa MAPE: `{best_absolute['descriptor_model']}`.
- Best compressed model for the 6--8 h evolution signal: `{best_signal['descriptor_model']}`.
- Absolute-kappa errors are diluted by the common host scattering background;
  the delta-kappa signal NRMSE is the stricter test of whether a descriptor
  preserves microstructure-evolution information.
- `Nv + mean R + CV` is not a unique PSD without a shape closure.  Its result
  here is conditional on the explicitly registered lognormal assumption.
- This 2 h, narrow-initial-PSD window can rank the candidates, but it cannot
  establish universal sufficiency for the much broader 6--48 h production
  ensemble.  The same comparison must be repeated on the completed Method-1
  A/B/C trajectories before freezing a final minimal descriptor.

```text
status={STATUS}
snapshot_count=51
replicate_count=3
temperatures_K=300,350,400,450,500,550,600
full_PSD_reference=true
dislocation_mode=DISLOCATION_OFF
absolute_experimental_kappa_claim=false
running_PF_jobs_modified=false
```
"""
    (output_root / "descriptor_sufficiency_report.md").write_text(
        report, encoding="utf-8"
    )

    audit = {
        "schema": SCHEMA,
        "status": STATUS,
        "gates": {
            "source_short_screening_pass": True,
            "source_merge_aware_pass": True,
            "source_snapshot_grid_identical": True,
            "source_descriptor_closure": True,
            "transport_contract_frozen": True,
            "full_psd_reference_present": True,
            "lognormal_quadrature_moment_closure": (
                max_lognormal_mean_error <= 5.0e-13
                and max_lognormal_cv_error <= 5.0e-13
            ),
            "all_prediction_cells_present": len(prediction_rows)
            == 3 * 17 * 7 * 2 * 5,
            "running_pf_jobs_untouched": True,
        },
        "input": input_audit,
        "transport": {
            "temperatures_K": TEMPERATURES_K,
            "matrix_modes": MATRIX_MODES,
            "descriptor_models": MODELS,
            "A_N": 1.5,
            "dislocation_mode": "DISLOCATION_OFF",
            "lognormal_quadrature_order": LOGNORMAL_ORDER,
            "max_lognormal_mean_relative_error": max_lognormal_mean_error,
            "max_lognormal_CV_absolute_error": max_lognormal_cv_error,
        },
        "summary": summary_rows,
        "source_sha256": source_files,
    }
    if not all(audit["gates"].values()):
        raise ValueError(f"descriptor analysis gate failed: {audit['gates']}")
    write_json(output_root / "descriptor_sufficiency_audit.json", audit)
    (output_root / "final_status.txt").write_text(
        "\n".join(
            [
                f"status={STATUS}",
                "source_A_status=PASS_246CUBE_6H8H_SHORT_SCREENING_V1",
                "source_B_status=PASS_246CUBE_6H8H_SHORT_SCREENING_V1",
                "source_C_status=PASS_246CUBE_6H8H_SHORT_SCREENING_V1",
                f"source_C_merge_status={MERGE_PASS}",
                "snapshot_count=51",
                "full_PSD_reference=true",
                "dislocation_mode=DISLOCATION_OFF",
                "running_PF_jobs_modified=false",
                "absolute_experimental_kappa_claim=false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_targets = sorted(
        path
        for path in output_root.iterdir()
        if path.is_file() and path.name != "analysis_manifest.sha256"
    )
    (output_root / "analysis_manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in manifest_targets),
        encoding="utf-8",
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("reports/pf_246cube_three_seed_library_handoff_v1"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--yu-config",
        type=Path,
        default=Path("data/qualification/yu2024_transport_v1/yu_48h_parameters.json"),
    )
    parser.add_argument(
        "--transport-contract",
        type=Path,
        default=Path(
            "data/qualification/pf_full_psd_no_dislocation_transport_v1/"
            "transport_parameter_contract.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = analyze(
        args.input_root.resolve(),
        args.output_root.resolve(),
        args.yu_config.resolve(),
        args.transport_contract.resolve(),
    )
    print(audit["status"])


if __name__ == "__main__":
    main()
