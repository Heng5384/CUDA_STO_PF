#!/usr/bin/env python3
"""Read-only no-dislocation transport diagnostic for the blocked Method-1 A run.

This deliberately accepts only the *overall resolved-particle PSD* at each
checkpoint.  It does not promote the source to a production authority, make a
post-merge identity claim, or modify the PF production directory.
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


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_pf_abc_6h8h_descriptor_sufficiency_v1 as descriptors  # noqa: E402
import pf_full_psd_no_dislocation_transport_v1 as transport  # noqa: E402


SCHEMA = "PF_BLOCKED_A_HOURLY_PSD_NO_DISLOCATION_DIAGNOSTIC_V1"
STATUS = "PASS_PF_BLOCKED_A_HOURLY_PSD_NO_DISLOCATION_DIAGNOSTIC_V1"
SOURCE_BLOCKED_STATUS = "BLOCKED_246CUBE_6H48H_PRODUCTION_DRIVER_V1"
SOURCE_LINEAGE_STATUS = "BLOCKED_PERIODIC_OVERLAP_PARTICLE_LINEAGE_V1"
TEMPERATURES_K = descriptors.TEMPERATURES_K
MATRIX_MODES = descriptors.MATRIX_MODES
MODELS = descriptors.MODELS
BOX_VOLUME_NM3 = 246.0**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), np.finfo(float).tiny)


def text_row(row: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in row.items():
        if isinstance(value, (float, np.floating)):
            result[key] = f"{float(value):.17g}"
        elif isinstance(value, (int, np.integer)):
            result[key] = str(int(value))
        else:
            result[key] = str(value)
    return result


def load_snapshots(source_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = {
        "observables": source_root / "ensemble_observables.csv",
        "particles": source_root / "particle_lineage.csv",
        "root_status": source_root / "production_root_status.txt",
        "lineage_summary": source_root / "lineage_summary.txt",
        "campaign": source_root / "campaign_manifest.json",
        "input_hashes": source_root / "input_hashes.sha256",
        "analysis_binary": source_root / "analysis_binary.sha256",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"missing copied A source files: {missing}")
    root_status = paths["root_status"].read_text(encoding="utf-8").splitlines()[0]
    lineage_summary = paths["lineage_summary"].read_text(encoding="utf-8")
    if root_status != SOURCE_BLOCKED_STATUS:
        raise ValueError(f"unexpected A root status: {root_status!r}")
    if f"status={SOURCE_LINEAGE_STATUS}" not in lineage_summary:
        raise ValueError("expected blocked lineage status is absent")

    observable_rows = read_csv(paths["observables"])
    particles_by_step: dict[int, list[dict[str, str]]] = defaultdict(list)
    for item in read_csv(paths["particles"]):
        particles_by_step[int(item["step"])].append(item)
    if len(observable_rows) != 45:
        raise ValueError(f"expected the 45 A raw checkpoints, got {len(observable_rows)}")
    snapshots: list[dict[str, Any]] = []
    closure = {"count": 0.0, "mean_R": 0.0, "Sv": 0.0, "M6": 0.0}
    for row in observable_rows:
        step = int(row["step"])
        group = particles_by_step.get(step, [])
        expected_count = int(row["particle_count"])
        if len(group) != expected_count or expected_count <= 0:
            raise ValueError(f"PSD count mismatch at step {step}: {len(group)} != {expected_count}")
        radii_nm = np.asarray([float(item["equivalent_radius_nm"]) for item in group])
        if np.any(~np.isfinite(radii_nm)) or np.any(radii_nm <= 0.0):
            raise ValueError(f"invalid PSD radius at step {step}")
        moments = transport.moments_from_radii(radii_nm * 1.0e-9, BOX_VOLUME_NM3 * 1.0e-27)
        closure["count"] = max(closure["count"], abs(expected_count - len(radii_nm)))
        closure["mean_R"] = max(closure["mean_R"], relative_error(float(np.mean(radii_nm)), float(row["mean_radius_nm"])))
        closure["Sv"] = max(closure["Sv"], relative_error(moments["Sv_m-1"] * 1.0e-9, float(row["Sv_nm_inv"])))
        closure["M6"] = max(closure["M6"], relative_error(moments["M6_m3"] * 1.0e27, float(row["M6_nm3"])))
        snapshots.append(
            {
                "step": step,
                "age_h": float(row["experimental_age_h"]),
                "elapsed_physical_time_s": float(row["elapsed_physical_time_s"]),
                "particle_count": expected_count,
                "Nv_m-3": moments["Nv_m-3"],
                "mean_radius_nm": float(np.mean(radii_nm)),
                "std_radius_nm": float(np.std(radii_nm, ddof=0)),
                "CV": float(np.std(radii_nm, ddof=0) / np.mean(radii_nm)),
                "Sv_nm-1": moments["Sv_m-1"] * 1.0e-9,
                "M6_nm3": moments["M6_m3"] * 1.0e27,
                "matrix_xAg": float(row["far_field_matrix_xAg"]),
                "matrix_xB": float(row["far_field_matrix_xB"]),
                "radii_nm": sorted(float(value) for value in radii_nm),
                "full_psd_multiset_sha256": transport.canonical_sha256(sorted(float(value) for value in radii_nm)),
            }
        )
    snapshots.sort(key=lambda item: item["step"])
    if snapshots[0]["step"] != 0 or snapshots[-1]["step"] != 152585:
        raise ValueError("A raw checkpoint endpoints do not span 6--48 h")
    if closure["count"] != 0.0 or any(closure[key] > 5.0e-12 for key in ("mean_R", "Sv", "M6")):
        raise ValueError(f"raw-PSD descriptor closure failed: {closure}")
    return snapshots, {
        "source_files": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in paths.items()},
        "source_root_status": root_status,
        "source_lineage_status": SOURCE_LINEAGE_STATUS,
        "source_is_production_authority": False,
        "snapshot_count": len(snapshots),
        "descriptor_closure_max": closure,
    }


def analyze(source_root: Path, output_root: Path, yu_config: Path, contract_path: Path) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True)
    config = json.loads(yu_config.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    transport.validate_yu_base_config(config, yu_config)
    transport.validate_interface_contract(contract, contract_path, yu_config)
    snapshots, source = load_snapshots(source_root)
    fixed_xag = float(snapshots[0]["matrix_xAg"])
    source["source_files"].update(
        {
            "yu_config": {"path": str(yu_config.resolve()), "sha256": sha256(yu_config)},
            "transport_contract": {"path": str(contract_path.resolve()), "sha256": sha256(contract_path)},
            "transport_core": {"path": str((SCRIPT_DIR / "pf_full_psd_no_dislocation_transport_v1.py").resolve()), "sha256": sha256(SCRIPT_DIR / "pf_full_psd_no_dislocation_transport_v1.py")},
            "descriptor_module": {"path": str((SCRIPT_DIR / "analyze_pf_abc_6h8h_descriptor_sufficiency_v1.py").resolve()), "sha256": sha256(SCRIPT_DIR / "analyze_pf_abc_6h8h_descriptor_sufficiency_v1.py")},
            "diagnostic_script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        }
    )

    moments_rows = []
    prediction_rows: list[dict[str, Any]] = []
    max_log_mean = 0.0
    max_log_cv = 0.0
    max_direct_sum = 0.0
    for snapshot in snapshots:
        radii_m = np.asarray(snapshot["radii_nm"], dtype=float) * 1.0e-9
        box_volume_m3 = BOX_VOLUME_NM3 * 1.0e-27
        omega_test = np.array([1.0e11, 1.0e12, 5.0e12])
        direct = transport.direct_scalar_sum_precipitate_rate(omega_test, radii_m, box_volume_m3, config)
        vector = transport.full_psd_precipitate_rate(omega_test, radii_m, box_volume_m3, config)
        max_direct_sum = max(max_direct_sum, float(np.max(np.abs(direct - vector) / np.maximum(np.abs(direct), np.finfo(float).tiny))))
        moments_rows.append(text_row({key: snapshot[key] for key in (
            "step", "age_h", "elapsed_physical_time_s", "particle_count", "Nv_m-3", "mean_radius_nm", "std_radius_nm", "CV", "Sv_nm-1", "M6_nm3", "matrix_xAg", "matrix_xB", "full_psd_multiset_sha256")}))
        for matrix_mode in MATRIX_MODES:
            matrix_xag = fixed_xag if matrix_mode == "fixed_6h_matrix" else float(snapshot["matrix_xAg"])
            for temperature in TEMPERATURES_K:
                values: dict[str, float] = {}
                for model in MODELS:
                    value, diagnostics = descriptors.integrate_kappa(temperature, matrix_xag, radii_m, box_volume_m3, config, model)
                    values[model] = value
                    max_log_mean = max(max_log_mean, diagnostics.get("quadrature_mean_relative_error", 0.0))
                    max_log_cv = max(max_log_cv, diagnostics.get("quadrature_cv_absolute_error", 0.0))
                reference = values["full_PSD"]
                for model in MODELS:
                    prediction_rows.append({"step": snapshot["step"], "age_h": snapshot["age_h"], "temperature_K": temperature, "matrix_mode": matrix_mode, "matrix_xAg_used": matrix_xag, "descriptor_model": model, "kappa_W_mK": values[model], "full_PSD_kappa_W_mK": reference, "kappa_relative_error": relative_error(values[model], reference)})

    initial: dict[tuple[float, str, str], float] = {}
    full_delta: dict[tuple[int, float, str], float] = {}
    for row in prediction_rows:
        key = (float(row["temperature_K"]), str(row["matrix_mode"]), str(row["descriptor_model"]))
        if int(row["step"]) == 0:
            initial[key] = float(row["kappa_W_mK"])
    for row in prediction_rows:
        key = (float(row["temperature_K"]), str(row["matrix_mode"]), str(row["descriptor_model"]))
        row["delta_kappa_from_6h_W_mK"] = float(row["kappa_W_mK"]) - initial[key]
        if row["descriptor_model"] == "full_PSD":
            full_delta[(int(row["step"]), float(row["temperature_K"]), str(row["matrix_mode"]))] = float(row["delta_kappa_from_6h_W_mK"])
    for row in prediction_rows:
        reference_delta = full_delta[(int(row["step"]), float(row["temperature_K"]), str(row["matrix_mode"]))]
        row["full_PSD_delta_kappa_from_6h_W_mK"] = reference_delta
        row["delta_kappa_error_W_mK"] = float(row["delta_kappa_from_6h_W_mK"]) - reference_delta

    summary_rows: list[dict[str, Any]] = []
    for matrix_mode in MATRIX_MODES:
        for model in MODELS:
            rows = [row for row in prediction_rows if row["matrix_mode"] == matrix_mode and row["descriptor_model"] == model]
            kappa_errors = np.asarray([float(row["kappa_relative_error"]) for row in rows])
            evolution = [row for row in rows if int(row["step"]) != 0]
            delta_errors = np.asarray([float(row["delta_kappa_error_W_mK"]) for row in evolution])
            full_deltas = np.asarray([float(row["full_PSD_delta_kappa_from_6h_W_mK"]) for row in evolution])
            signal_rms = float(np.sqrt(np.mean(full_deltas**2)))
            summary_rows.append({"matrix_mode": matrix_mode, "descriptor_model": model, "kappa_MAPE_percent": 100.0 * float(np.mean(kappa_errors)), "kappa_RMS_relative_percent": 100.0 * float(np.sqrt(np.mean(kappa_errors**2))), "kappa_max_relative_percent": 100.0 * float(np.max(kappa_errors)), "delta_kappa_signal_NRMSE_percent": 100.0 * float(np.sqrt(np.mean(delta_errors**2)) / signal_rms) if signal_rms > 0.0 else 0.0, "delta_kappa_max_abs_error_W_mK": float(np.max(np.abs(delta_errors))), "comparison_cells": len(rows)})

    write_csv(output_root / "source_snapshot_descriptors.csv", list(moments_rows[0]), moments_rows)
    output_predictions = [text_row(row) for row in prediction_rows]
    write_csv(output_root / "descriptor_predictions.csv", list(output_predictions[0]), output_predictions)
    output_summary = [text_row(row) for row in summary_rows]
    write_csv(output_root / "descriptor_summary.csv", list(output_summary[0]), output_summary)
    full_psd_rows = [
        row
        for row in prediction_rows
        if row["descriptor_model"] == "full_PSD"
    ]
    fixed_lookup = {
        (int(row["step"]), float(row["temperature_K"])): row
        for row in full_psd_rows
        if row["matrix_mode"] == "fixed_6h_matrix"
    }
    varying_lookup = {
        (int(row["step"]), float(row["temperature_K"])): row
        for row in full_psd_rows
        if row["matrix_mode"] == "pf_time_varying_matrix"
    }
    trajectory_rows = []
    for snapshot in snapshots:
        for temperature in TEMPERATURES_K:
            key = (int(snapshot["step"]), float(temperature))
            fixed = fixed_lookup[key]
            varying = varying_lookup[key]
            trajectory_rows.append(
                text_row(
                    {
                        "step": snapshot["step"],
                        "age_h": snapshot["age_h"],
                        "temperature_K": temperature,
                        "matrix_xAg_6h_fixed": fixed_xag,
                        "matrix_xAg_pf": snapshot["matrix_xAg"],
                        "kappa_L_no_dis_fixed_6h_matrix_W_mK": fixed["kappa_W_mK"],
                        "kappa_L_no_dis_pf_time_varying_matrix_W_mK": varying["kappa_W_mK"],
                        "delta_kappa_PSD_W_mK": fixed["delta_kappa_from_6h_W_mK"],
                        "delta_kappa_PSD_plus_matrix_W_mK": varying["delta_kappa_from_6h_W_mK"],
                    }
                )
            )
    if len(trajectory_rows) != len(snapshots) * len(TEMPERATURES_K):
        raise ValueError("incomplete full-PSD transport trajectory")
    write_csv(
        output_root / "full_psd_no_dislocation_trajectory.csv",
        list(trajectory_rows[0]),
        trajectory_rows,
    )
    lines = []
    for mode in MATRIX_MODES:
        lines.extend([f"## {mode}", "", "| Descriptor | kappa MAPE | kappa maximum | delta-kappa NRMSE | max abs delta error (W/mK) |", "|---|---:|---:|---:|---:|"])
        for row in summary_rows:
            if row["matrix_mode"] == mode:
                lines.append(f"| {row['descriptor_model']} | {row['kappa_MAPE_percent']:.6f}% | {row['kappa_max_relative_percent']:.6f}% | {row['delta_kappa_signal_NRMSE_percent']:.3f}% | {row['delta_kappa_max_abs_error_W_mK']:.6g} |")
        lines.append("")
    report = "\n".join(["# Blocked Method-1 A: hourly full-PSD no-dislocation transport diagnostic", "", "## Scope", "", "This is a read-only diagnostic on the 45 copied overall resolved-particle PSD snapshots from Method-1 A.  The PF solver reached 48 h, but its particle-lineage gate is blocked; hence this is not a production authority, cannot enter the A/B/C ensemble, and makes no absolute experimental-kappa claim.  Particle IDs are not used as persistent identities: only each snapshot's radius multiset is used.", "", "The frozen transport contract uses the Yu 48 h non-precipitate host, A_N=1.5, S11/S13=0, and no Yu refit scale.  `fixed_6h_matrix` isolates PSD evolution; `pf_time_varying_matrix` additionally passes raw A far-field xAg into point-defect scattering.", "", "## Numerical checks", "", f"- PSD/observable descriptor closure maximum: {max(source['descriptor_closure_max'].values()):.3e}", f"- full-PSD vector/direct-sum maximum relative difference: {max_direct_sum:.3e}", f"- lognormal mean closure maximum: {max_log_mean:.3e}", f"- lognormal CV closure maximum: {max_log_cv:.3e}", "", *lines, "## Boundary", "", "These values qualify only the transport-interface calculation applied to raw A snapshot PSDs.  They neither repair the failed particle-lineage audit nor reproduce an absolute experimental thermal conductivity.", "", f"```text\nstatus={STATUS}\nsource_production_authority=false\nsource_root_status={SOURCE_BLOCKED_STATUS}\nsource_lineage_status={SOURCE_LINEAGE_STATUS}\ndislocation_mode=DISLOCATION_OFF\nabsolute_experimental_kappa_claim=false\n```"]) + "\n"
    (output_root / "diagnostic_report.md").write_text(report, encoding="utf-8")
    audit = {"schema": SCHEMA, "status": STATUS, "gates": {"source_is_explicitly_non_authority": True, "raw_psd_snapshot_count": len(snapshots) == 45, "raw_psd_descriptor_closure": max(source["descriptor_closure_max"].values()) <= 5.0e-12, "frozen_transport_contract": True, "full_psd_direct_sum": max_direct_sum <= 5.0e-13, "lognormal_closure": max_log_mean <= 5.0e-13 and max_log_cv <= 5.0e-13, "prediction_grid_complete": len(prediction_rows) == 45 * len(TEMPERATURES_K) * len(MATRIX_MODES) * len(MODELS), "full_psd_trajectory_grid_complete": len(trajectory_rows) == 45 * len(TEMPERATURES_K), "production_pf_untouched": True}, "source": source, "transport": {"A_N": 1.5, "dislocation_mode": "DISLOCATION_OFF", "yu_refit_scale_used": False, "temperatures_K": TEMPERATURES_K, "matrix_modes": MATRIX_MODES, "descriptor_models": MODELS}, "numerics": {"full_psd_vector_direct_sum_max_relative": max_direct_sum, "lognormal_mean_max_relative": max_log_mean, "lognormal_CV_max_absolute": max_log_cv}, "summary": summary_rows}
    if not all(audit["gates"].values()):
        raise ValueError(f"diagnostic gate failed: {audit['gates']}")
    write_json(output_root / "diagnostic_audit.json", audit)
    (output_root / "final_status.txt").write_text(f"status={STATUS}\nsource_production_authority=false\nsource_root_status={SOURCE_BLOCKED_STATUS}\nsource_lineage_status={SOURCE_LINEAGE_STATUS}\n", encoding="utf-8")
    targets = sorted(path for path in output_root.iterdir() if path.is_file() and path.name != "analysis_manifest.sha256")
    (output_root / "analysis_manifest.sha256").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in targets), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--yu-config", type=Path, default=Path("data/qualification/yu2024_transport_v1/yu_48h_parameters.json"))
    parser.add_argument("--transport-contract", type=Path, default=Path("data/qualification/pf_full_psd_no_dislocation_transport_v1/transport_parameter_contract.json"))
    args = parser.parse_args()
    audit = analyze(args.source_root.resolve(), args.output_root.resolve(), args.yu_config.resolve(), args.transport_contract.resolve())
    print(audit["status"])


if __name__ == "__main__":
    main()
