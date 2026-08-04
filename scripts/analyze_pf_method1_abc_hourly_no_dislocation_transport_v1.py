#!/usr/bin/env python3
"""Hourly Method-1 A/B/C full-PSD no-dislocation Yu transport analysis.

This is a read-only adapter over the frozen production-authority overlays.  It
uses the complete connected-component radius multiset at every hourly PF
snapshot as the transport reference and compares four compressed descriptor
closures against that direct sum.  It never edits PF fields or claims an
absolute experimental thermal-conductivity reproduction.
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


SCHEMA = "PF_METHOD1_ABC_HOURLY_NO_DISLOCATION_TRANSPORT_V1"
STATUS = "PASS_PF_METHOD1_ABC_HOURLY_FULL_PSD_NO_DISLOCATION_DESCRIPTOR_SUFFICIENCY_V1"
AUTHORITY_REGISTRY_PASS = "PASS_EXPERIMENT_MATRIX_ANCHORED_A_B_C_COMPLETE_PASS_AUTHORITIES"
PRODUCTION_PASS = "PASS_246CUBE_6H48H_CONDITIONAL_PRODUCTION_V1"
HOURLY_LINEAGE_PASS = "PASS_246CUBE_HOURLY_MERGE_DISSOLUTION_AUDIT_V1"
REPLICATES = ("A", "B", "C")
TEMPERATURES_K = descriptors.TEMPERATURES_K
MATRIX_MODES = descriptors.MATRIX_MODES
MODELS = descriptors.MODELS
BOX_VOLUME_NM3 = 246.0**3
REGISTERED_STEPS = (0, 21798, 43596, 65393, 108989, 152585)
EXPECTED_SNAPSHOT_COUNT = 45
EXPECTED_SOURCE_COMMIT = "1c08f9ee011b31e0cd4d82749e58a8f69ebd2204"


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


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), np.finfo(float).tiny)


def verify_sha256_manifest(manifest: Path) -> None:
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("* ")
        target = manifest.parent / name
        if not target.is_file() or sha256(target) != expected:
            raise ValueError(f"SHA-256 manifest mismatch: {target}")


def verify_authority_selection(authority_root: Path) -> dict[str, Any]:
    path = authority_root / "authority_selection.json"
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("registry_status") != AUTHORITY_REGISTRY_PASS:
        raise ValueError("Method-1 A/B/C authority registry is not PASS")
    if selection.get("source_commit") != EXPECTED_SOURCE_COMMIT:
        raise ValueError("unexpected production source commit")
    records: dict[str, Any] = {}
    for entry in selection.get("replicates", []):
        replicate = str(entry.get("replicate"))
        selected = [item for item in entry.get("candidates", []) if item.get("selected")]
        if replicate not in REPLICATES or len(selected) != 1:
            raise ValueError(f"invalid authority selection for {replicate}")
        item = selected[0]
        if item.get("authority_status") != "PASS_COMPLETE_6H48H" or not item.get("complete_6h48h"):
            raise ValueError(f"selected authority is incomplete for {replicate}")
        transport_dir = authority_root.parent.parent / str(item["transport_output_dir"])
        manifest = transport_dir / "transport_snapshot_manifest.json"
        if sha256(manifest) != item["transport_manifest_sha256"]:
            raise ValueError(f"transport manifest hash mismatch for {replicate}")
        records[replicate] = item
    if set(records) != set(REPLICATES):
        raise ValueError("authority registry does not contain exactly A/B/C")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "source_commit": selection["source_commit"],
        "records": records,
    }


def load_authority_snapshots(
    authority_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection = verify_authority_selection(authority_root)
    snapshots: list[dict[str, Any]] = []
    source_files: dict[str, dict[str, str]] = {
        "authority_selection": {
            "path": selection["path"],
            "sha256": selection["sha256"],
        }
    }
    replicate_audits: dict[str, Any] = {}
    common_steps: list[int] | None = None
    closure = {"count": 0.0, "mean_R": 0.0, "Sv": 0.0, "M6": 0.0}

    for replicate in REPLICATES:
        authority_dir = authority_root / "authority" / replicate
        source_dir = authority_root / "source" / replicate
        status_path = authority_dir / "status.txt"
        if status_path.read_text(encoding="utf-8").splitlines()[0] != PRODUCTION_PASS:
            raise ValueError(f"authority status is not PASS for {replicate}")
        manifest_path = authority_dir / "authority_manifest.sha256"
        verify_sha256_manifest(manifest_path)
        authority_audit_path = authority_dir / "audit.json"
        authority_audit = json.loads(authority_audit_path.read_text(encoding="utf-8"))
        if authority_audit.get("numerical_status") != PRODUCTION_PASS:
            raise ValueError(f"numerical authority is not PASS for {replicate}")
        if not all(authority_audit.get("gates", {}).values()):
            raise ValueError(f"authority gate failed for {replicate}")
        if len(authority_audit.get("checkpoint_hashes", {})) != 44:
            raise ValueError(f"incomplete checkpoint hash chain for {replicate}")

        hourly_path = Path(str(authority_audit["hourly_merge_audit"]["path"])) / "audit.json"
        hourly_audit = json.loads(hourly_path.read_text(encoding="utf-8"))
        if hourly_audit.get("status") != HOURLY_LINEAGE_PASS or not all(hourly_audit.get("gates", {}).values()):
            raise ValueError(f"hourly merge-aware audit is not PASS for {replicate}")
        if sha256(hourly_path) != authority_audit["hourly_merge_audit"]["sha256"]:
            raise ValueError(f"hourly merge-aware audit hash mismatch for {replicate}")

        observables_path = source_dir / "ensemble_observables.csv"
        particles_path = source_dir / "particle_lineage.csv"
        expected_hashes = authority_audit["input_sha256"]
        if sha256(observables_path) != expected_hashes["raw_observables_sha256"]:
            raise ValueError(f"raw observable hash mismatch for {replicate}")
        if sha256(particles_path) != expected_hashes["raw_particle_lineage_sha256"]:
            raise ValueError(f"raw particle hash mismatch for {replicate}")

        observable_rows = read_csv(observables_path)
        if len(observable_rows) != EXPECTED_SNAPSHOT_COUNT:
            raise ValueError(f"expected 45 hourly snapshots for {replicate}")
        steps = [int(row["step"]) for row in observable_rows]
        if steps != sorted(steps) or steps[0] != 0 or steps[-1] != 152585:
            raise ValueError(f"invalid hourly snapshot sequence for {replicate}")
        if common_steps is None:
            common_steps = steps
        elif steps != common_steps:
            raise ValueError("A/B/C hourly snapshot grids differ")

        particles_by_step: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in read_csv(particles_path):
            particles_by_step[int(row["step"])].append(row)
        for row in observable_rows:
            step = int(row["step"])
            group = particles_by_step.get(step, [])
            count = int(row["particle_count"])
            if count <= 0 or len(group) != count:
                raise ValueError(f"PSD count mismatch for {replicate} step {step}")
            # A merge group may temporarily demerge into more than one physical
            # component while retaining one lineage-group id.  That is a valid
            # hourly audit state, so component_label (not stable id) is the
            # per-snapshot uniqueness key used for the transport population.
            component_labels = [int(item["component_label"]) for item in group]
            if len(set(component_labels)) != count:
                raise ValueError(f"duplicate component label for {replicate} step {step}")
            ordered = sorted(group, key=lambda item: int(item["stable_particle_id"]))
            ordered_ids = [int(item["stable_particle_id"]) for item in ordered]
            radii_nm = np.asarray([float(item["equivalent_radius_nm"]) for item in ordered], dtype=float)
            if np.any(~np.isfinite(radii_nm)) or np.any(radii_nm <= 0.0):
                raise ValueError(f"invalid PSD radius for {replicate} step {step}")
            moments = transport.moments_from_radii(radii_nm * 1.0e-9, BOX_VOLUME_NM3 * 1.0e-27)
            mean_nm = float(np.mean(radii_nm))
            std_nm = float(np.std(radii_nm, ddof=0))
            closure["count"] = max(closure["count"], abs(len(radii_nm) - count))
            closure["mean_R"] = max(closure["mean_R"], relative_error(mean_nm, float(row["mean_radius_nm"])))
            closure["Sv"] = max(closure["Sv"], relative_error(moments["Sv_m-1"] * 1.0e-9, float(row["Sv_nm_inv"])))
            closure["M6"] = max(closure["M6"], relative_error(moments["M6_m3"] * 1.0e27, float(row["M6_nm3"])))
            sorted_radii = sorted(float(value) for value in radii_nm)
            snapshots.append(
                {
                    "replicate": replicate,
                    "step": step,
                    "age_h": float(row["experimental_age_h"]),
                    "elapsed_physical_time_s": float(row["elapsed_physical_time_s"]),
                    "particle_count": count,
                    "Nv_m-3": moments["Nv_m-3"],
                    "mean_radius_nm": mean_nm,
                    "std_radius_nm": std_nm,
                    "CV": std_nm / mean_nm,
                    "Sv_nm-1": moments["Sv_m-1"] * 1.0e-9,
                    "M6_nm3": moments["M6_m3"] * 1.0e27,
                    "beta_volume_fraction": float(row["beta_volume_fraction"]),
                    "matrix_xAg": float(row["far_field_matrix_xAg"]),
                    "matrix_xB": float(row["far_field_matrix_xB"]),
                    "radii_nm": sorted_radii,
                    "full_psd_multiset_sha256": transport.canonical_sha256(sorted_radii),
                    "registered_id_radius_sha256": transport.canonical_sha256(
                        list(zip(ordered_ids, radii_nm.tolist()))
                    ),
                }
            )

        replicate_audits[replicate] = {
            "authority_audit_path": str(authority_audit_path.resolve()),
            "authority_audit_sha256": sha256(authority_audit_path),
            "authority_manifest_path": str(manifest_path.resolve()),
            "authority_manifest_sha256": sha256(manifest_path),
            "hourly_merge_audit_path": str(hourly_path.resolve()),
            "hourly_merge_audit_sha256": sha256(hourly_path),
            "checkpoint_hash_count": len(authority_audit["checkpoint_hashes"]),
        }
        for name, path in {
            f"{replicate}_observables": observables_path,
            f"{replicate}_particles": particles_path,
            f"{replicate}_authority_audit": authority_audit_path,
            f"{replicate}_authority_manifest": manifest_path,
            f"{replicate}_hourly_merge_audit": hourly_path,
        }.items():
            source_files[name] = {"path": str(path.resolve()), "sha256": sha256(path)}

    if common_steps is None or len(snapshots) != 3 * EXPECTED_SNAPSHOT_COUNT:
        raise ValueError("incomplete A/B/C hourly source population")
    if closure["count"] != 0.0 or any(closure[key] > 5.0e-12 for key in ("mean_R", "Sv", "M6")):
        raise ValueError(f"source descriptor closure failed: {closure}")
    snapshots.sort(key=lambda item: (item["replicate"], item["step"]))
    return snapshots, {
        "authority_registry_status": AUTHORITY_REGISTRY_PASS,
        "source_commit": selection["source_commit"],
        "replicate_audits": replicate_audits,
        "source_files": source_files,
        "snapshot_steps": common_steps,
        "snapshot_count": len(snapshots),
        "descriptor_closure_max": closure,
    }


def registered_consistency(
    authority_root: Path,
    prediction_rows: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float, bool]:
    new_lookup = {
        (str(row["replicate"]), int(row["step"]), float(row["temperature_K"]), str(row["matrix_mode"])): float(row["kappa_W_mK"])
        for row in prediction_rows
        if row["descriptor_model"] == "full_PSD" and int(row["step"]) in REGISTERED_STEPS
    }
    snapshot_hashes = {
        (str(item["replicate"]), int(item["step"])): str(item["registered_id_radius_sha256"])
        for item in snapshots
    }
    rows: list[dict[str, Any]] = []
    maximum = 0.0
    hashes_match = True
    for replicate in REPLICATES:
        old_snapshots = {
            int(row["step"]): row
            for row in read_csv(authority_root / "transport" / replicate / "transport_snapshots.csv")
        }
        old_kappa = read_csv(authority_root / "transport" / replicate / "kappa_time_temperature.csv")
        for step in REGISTERED_STEPS:
            hashes_match &= old_snapshots[step]["full_psd_canonical_sha256"] == snapshot_hashes[(replicate, step)]
        for row in old_kappa:
            step = int(row["step"])
            temperature = float(row["temperature_K"])
            if step not in REGISTERED_STEPS:
                continue
            for mode, column in (
                ("fixed_6h_matrix", "kappa_L_no_dis_fixed_matrix_W_mK"),
                ("pf_time_varying_matrix", "kappa_L_no_dis_time_varying_matrix_W_mK"),
            ):
                old_value = float(row[column])
                new_value = new_lookup[(replicate, step, temperature, mode)]
                error = relative_error(new_value, old_value)
                maximum = max(maximum, error)
                rows.append(
                    {
                        "replicate": replicate,
                        "step": step,
                        "age_h": float(row["age_h"]),
                        "temperature_K": temperature,
                        "matrix_mode": mode,
                        "registered_kappa_W_mK": old_value,
                        "hourly_recomputed_kappa_W_mK": new_value,
                        "relative_difference": error,
                    }
                )
    return rows, maximum, hashes_match


def analyze(
    authority_root: Path,
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
    transport.validate_interface_contract(contract, transport_contract_path, yu_config_path)
    snapshots, source = load_authority_snapshots(authority_root)
    source["source_files"].update(
        {
            "yu_config": {"path": str(yu_config_path.resolve()), "sha256": sha256(yu_config_path)},
            "transport_contract": {"path": str(transport_contract_path.resolve()), "sha256": sha256(transport_contract_path)},
            "transport_core": {"path": str((SCRIPT_DIR / "pf_full_psd_no_dislocation_transport_v1.py").resolve()), "sha256": sha256(SCRIPT_DIR / "pf_full_psd_no_dislocation_transport_v1.py")},
            "descriptor_module": {"path": str((SCRIPT_DIR / "analyze_pf_abc_6h8h_descriptor_sufficiency_v1.py").resolve()), "sha256": sha256(SCRIPT_DIR / "analyze_pf_abc_6h8h_descriptor_sufficiency_v1.py")},
            "analysis_script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        }
    )
    fixed_matrix = {
        replicate: next(float(item["matrix_xAg"]) for item in snapshots if item["replicate"] == replicate and item["step"] == 0)
        for replicate in REPLICATES
    }

    source_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    max_direct_sum = 0.0
    max_log_mean = 0.0
    max_log_cv = 0.0
    for snapshot in snapshots:
        radii_m = np.asarray(snapshot["radii_nm"], dtype=float) * 1.0e-9
        box_volume_m3 = BOX_VOLUME_NM3 * 1.0e-27
        omega_test = np.asarray([1.0e11, 1.0e12, 5.0e12])
        direct = transport.direct_scalar_sum_precipitate_rate(omega_test, radii_m, box_volume_m3, config)
        vector = transport.full_psd_precipitate_rate(omega_test, radii_m, box_volume_m3, config)
        max_direct_sum = max(
            max_direct_sum,
            float(np.max(np.abs(direct - vector) / np.maximum(np.abs(direct), np.finfo(float).tiny))),
        )
        source_rows.append(
            {key: snapshot[key] for key in (
                "replicate", "step", "age_h", "elapsed_physical_time_s", "particle_count",
                "Nv_m-3", "mean_radius_nm", "std_radius_nm", "CV", "Sv_nm-1", "M6_nm3",
                "beta_volume_fraction", "matrix_xAg", "matrix_xB", "full_psd_multiset_sha256",
                "registered_id_radius_sha256",
            )}
        )
        for matrix_mode in MATRIX_MODES:
            matrix_xag = fixed_matrix[str(snapshot["replicate"])] if matrix_mode == "fixed_6h_matrix" else float(snapshot["matrix_xAg"])
            for temperature in TEMPERATURES_K:
                values: dict[str, float] = {}
                for model in MODELS:
                    value, diagnostics = descriptors.integrate_kappa(
                        temperature, matrix_xag, radii_m, box_volume_m3, config, model
                    )
                    values[model] = value
                    max_log_mean = max(max_log_mean, diagnostics.get("quadrature_mean_relative_error", 0.0))
                    max_log_cv = max(max_log_cv, diagnostics.get("quadrature_cv_absolute_error", 0.0))
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
    full_deltas: dict[tuple[str, int, float, str], float] = {}
    for row in prediction_rows:
        if int(row["step"]) == 0:
            initial_values[(str(row["replicate"]), float(row["temperature_K"]), str(row["matrix_mode"]), str(row["descriptor_model"]))] = float(row["kappa_W_mK"])
    for row in prediction_rows:
        base_key = (str(row["replicate"]), float(row["temperature_K"]), str(row["matrix_mode"]), str(row["descriptor_model"]))
        delta = float(row["kappa_W_mK"]) - initial_values[base_key]
        row["delta_kappa_from_6h_W_mK"] = delta
        if row["descriptor_model"] == "full_PSD":
            full_deltas[(str(row["replicate"]), int(row["step"]), float(row["temperature_K"]), str(row["matrix_mode"]))] = delta
    for row in prediction_rows:
        full_delta = full_deltas[(str(row["replicate"]), int(row["step"]), float(row["temperature_K"]), str(row["matrix_mode"]))]
        row["full_PSD_delta_kappa_from_6h_W_mK"] = full_delta
        row["delta_kappa_error_W_mK"] = float(row["delta_kappa_from_6h_W_mK"]) - full_delta

    summary_rows: list[dict[str, Any]] = []
    for matrix_mode in MATRIX_MODES:
        for model in MODELS:
            rows = [row for row in prediction_rows if row["matrix_mode"] == matrix_mode and row["descriptor_model"] == model]
            errors = np.asarray([float(row["kappa_relative_error"]) for row in rows])
            evolution = [row for row in rows if int(row["step"]) != 0]
            delta_errors = np.asarray([float(row["delta_kappa_error_W_mK"]) for row in evolution])
            reference_deltas = np.asarray([float(row["full_PSD_delta_kappa_from_6h_W_mK"]) for row in evolution])
            signal_rms = float(np.sqrt(np.mean(reference_deltas**2)))
            rank_matches = 0
            rank_total = 0
            for step in source["snapshot_steps"]:
                if step == 0:
                    continue
                for temperature in TEMPERATURES_K:
                    subset = [row for row in rows if int(row["step"]) == step and float(row["temperature_K"]) == temperature]
                    values = {str(row["replicate"]): float(row["delta_kappa_from_6h_W_mK"]) for row in subset}
                    reference = {str(row["replicate"]): float(row["full_PSD_delta_kappa_from_6h_W_mK"]) for row in subset}
                    if set(values) != set(REPLICATES):
                        raise ValueError("incomplete A/B/C rank cell")
                    rank_matches += int(descriptors.exact_rank(values) == descriptors.exact_rank(reference))
                    rank_total += 1
            summary_rows.append(
                {
                    "matrix_mode": matrix_mode,
                    "descriptor_model": model,
                    "kappa_MAPE_percent": 100.0 * float(np.mean(errors)),
                    "kappa_RMS_relative_percent": 100.0 * float(np.sqrt(np.mean(errors**2))),
                    "kappa_max_relative_percent": 100.0 * float(np.max(errors)),
                    "delta_kappa_signal_NRMSE_percent": 100.0 * float(np.sqrt(np.mean(delta_errors**2)) / signal_rms) if signal_rms > 0.0 else 0.0,
                    "delta_kappa_max_abs_error_W_mK": float(np.max(np.abs(delta_errors))),
                    "ABC_delta_order_exact_fraction": rank_matches / rank_total,
                    "comparison_cells": len(rows),
                }
            )

    full_rows = [row for row in prediction_rows if row["descriptor_model"] == "full_PSD"]
    trajectory_rows: list[dict[str, Any]] = []
    for replicate in REPLICATES:
        for step in source["snapshot_steps"]:
            snapshot = next(item for item in snapshots if item["replicate"] == replicate and item["step"] == step)
            for temperature in TEMPERATURES_K:
                fixed = next(row for row in full_rows if row["replicate"] == replicate and row["step"] == step and row["temperature_K"] == temperature and row["matrix_mode"] == "fixed_6h_matrix")
                varying = next(row for row in full_rows if row["replicate"] == replicate and row["step"] == step and row["temperature_K"] == temperature and row["matrix_mode"] == "pf_time_varying_matrix")
                trajectory_rows.append(
                    {
                        "replicate": replicate,
                        "step": step,
                        "age_h": snapshot["age_h"],
                        "temperature_K": temperature,
                        "matrix_xAg_6h_fixed": fixed_matrix[replicate],
                        "matrix_xAg_pf": snapshot["matrix_xAg"],
                        "kappa_L_no_dis_fixed_6h_matrix_W_mK": fixed["kappa_W_mK"],
                        "kappa_L_no_dis_pf_time_varying_matrix_W_mK": varying["kappa_W_mK"],
                        "delta_kappa_PSD_W_mK": fixed["delta_kappa_from_6h_W_mK"],
                        "delta_kappa_PSD_plus_matrix_W_mK": varying["delta_kappa_from_6h_W_mK"],
                        "delta_kappa_matrix_given_PSD_W_mK": float(varying["kappa_W_mK"]) - float(fixed["kappa_W_mK"]),
                    }
                )

    ensemble_rows: list[dict[str, Any]] = []
    for step in source["snapshot_steps"]:
        for temperature in TEMPERATURES_K:
            group = [row for row in trajectory_rows if row["step"] == step and row["temperature_K"] == temperature]
            output: dict[str, Any] = {
                "step": step,
                "age_h": group[0]["age_h"],
                "temperature_K": temperature,
                "replicate_count": len(group),
            }
            for column in (
                "kappa_L_no_dis_fixed_6h_matrix_W_mK",
                "kappa_L_no_dis_pf_time_varying_matrix_W_mK",
                "delta_kappa_PSD_W_mK",
                "delta_kappa_PSD_plus_matrix_W_mK",
                "delta_kappa_matrix_given_PSD_W_mK",
            ):
                values = np.asarray([float(row[column]) for row in group])
                output[f"{column}_mean"] = float(np.mean(values))
                output[f"{column}_sample_std"] = float(np.std(values, ddof=1))
                output[f"{column}_min"] = float(np.min(values))
                output[f"{column}_max"] = float(np.max(values))
            ensemble_rows.append(output)

    consistency_rows, max_registered_difference, registered_hash_match = registered_consistency(
        authority_root, prediction_rows, snapshots
    )
    output_tables = {
        "source_snapshot_descriptors.csv": source_rows,
        "descriptor_predictions.csv": prediction_rows,
        "descriptor_summary.csv": summary_rows,
        "full_psd_hourly_trajectory.csv": trajectory_rows,
        "full_psd_hourly_ensemble.csv": ensemble_rows,
        "registered_six_time_consistency.csv": consistency_rows,
    }
    for filename, rows in output_tables.items():
        converted = [text_row(row) for row in rows]
        write_csv(output_root / filename, list(converted[0]), converted)

    def summary_table(matrix_mode: str) -> str:
        lines = []
        for row in summary_rows:
            if row["matrix_mode"] != matrix_mode:
                continue
            lines.append(
                "| {descriptor_model} | {mape:.6f}% | {maximum:.6f}% | {signal:.3f}% | {delta:.6g} | {rank:.3f} |".format(
                    descriptor_model=row["descriptor_model"],
                    mape=row["kappa_MAPE_percent"],
                    maximum=row["kappa_max_relative_percent"],
                    signal=row["delta_kappa_signal_NRMSE_percent"],
                    delta=row["delta_kappa_max_abs_error_W_mK"],
                    rank=row["ABC_delta_order_exact_fraction"],
                )
            )
        return "\n".join(lines)

    compressed = [row for row in summary_rows if row["descriptor_model"] != "full_PSD"]
    best_absolute = min(compressed, key=lambda row: row["kappa_MAPE_percent"])
    best_signal = min(compressed, key=lambda row: row["delta_kappa_signal_NRMSE_percent"])
    final_48h = {
        (float(row["temperature_K"])): row
        for row in ensemble_rows
        if int(row["step"]) == 152585
    }
    final_lines = []
    for temperature in (300.0, 400.0, 600.0):
        row = final_48h[temperature]
        final_lines.append(
            "| {temperature:.0f} | {fixed:.6f} ± {fixed_std:.6f} | {varying:.6f} ± {varying_std:.6f} | {delta:.6f} ± {delta_std:.6f} |".format(
                temperature=temperature,
                fixed=row["kappa_L_no_dis_fixed_6h_matrix_W_mK_mean"],
                fixed_std=row["kappa_L_no_dis_fixed_6h_matrix_W_mK_sample_std"],
                varying=row["kappa_L_no_dis_pf_time_varying_matrix_W_mK_mean"],
                varying_std=row["kappa_L_no_dis_pf_time_varying_matrix_W_mK_sample_std"],
                delta=row["delta_kappa_PSD_plus_matrix_W_mK_mean"],
                delta_std=row["delta_kappa_PSD_plus_matrix_W_mK_sample_std"],
            )
        )

    report = f"""# Method-1 A/B/C hourly full-PSD no-dislocation Yu transport

## Scope

This read-only calculation consumes all 45 hourly complete connected-component
PSDs from each of the three accepted 246^3 Method-1 production authorities
(135 PF snapshots total).  A merged connected component is counted once as the
physical scatterer present at that snapshot; historical lineage members are
not double counted.  No PF field, source term, or production output was changed.

The frozen host is the Yu 48 h non-particle baseline with `A_N=1.5`, S11/S13
dislocation scattering set to zero, and no Yu refit scale.  `fixed_6h_matrix`
isolates PSD evolution; `pf_time_varying_matrix` additionally passes each PF
far-field Ag value to point-defect scattering.  The direct complete PSD sum is
the reference at seven temperatures from 300 to 600 K.

## Descriptor sufficiency: fixed 6 h matrix

| Descriptor | absolute kappa MAPE | maximum kappa error | delta-kappa signal NRMSE | max abs delta error (W/mK) | exact A/B/C delta ordering |
|---|---:|---:|---:|---:|---:|
{summary_table('fixed_6h_matrix')}

## Descriptor sufficiency: PF time-varying matrix

| Descriptor | absolute kappa MAPE | maximum kappa error | delta-kappa signal NRMSE | max abs delta error (W/mK) | exact A/B/C delta ordering |
|---|---:|---:|---:|---:|---:|
{summary_table('pf_time_varying_matrix')}

Absolute-kappa error is diluted by the common host background, so the
delta-kappa signal NRMSE is the stricter measure of whether a compressed
descriptor preserves the microstructure-evolution signal.  The
`Nv_plus_mean_R_plus_CV_lognormal` result is conditional on the explicitly
registered lognormal closure; `Nv`, mean radius, and CV do not define a unique
PSD by themselves.

Best compressed absolute-kappa model: `{best_absolute['descriptor_model']}`.
Best compressed evolution-signal model: `{best_signal['descriptor_model']}`.
The complete PSD remains the production reference regardless of compression
ranking.

## 48 h complete-PSD ensemble

| T (K) | fixed 6 h matrix kappa (W/mK) | PF matrix kappa (W/mK) | PF matrix delta from 6 h (W/mK) |
|---:|---:|---:|---:|
{chr(10).join(final_lines)}

## Qualification and boundary

- Hourly source snapshots: {len(snapshots)} (45 each for A/B/C).
- Transport prediction cells: {len(prediction_rows)}.
- Full-PSD vector/direct-sum maximum relative difference: {max_direct_sum:.3e}.
- Lognormal mean closure maximum: {max_log_mean:.3e}.
- Lognormal CV closure maximum: {max_log_cv:.3e}.
- Recomputed six-time full-PSD maximum relative difference from the existing
  registered production transport: {max_registered_difference:.3e}.
- Registered six-time PSD hashes identical: {str(registered_hash_match).lower()}.

This qualifies a conditional no-dislocation transport interface and descriptor
information-loss comparison.  It does not reproduce absolute experimental
thermal conductivity and does not add a PF-predicted dislocation density.

```text
status={STATUS}
source_authority_status={AUTHORITY_REGISTRY_PASS}
source_snapshot_count={len(snapshots)}
prediction_cell_count={len(prediction_rows)}
full_PSD_reference=true
dislocation_mode=DISLOCATION_OFF
yu_refit_scale_used=false
production_PF_modified=false
absolute_experimental_kappa_claim=false
```
"""
    (output_root / "hourly_transport_report.md").write_text(report, encoding="utf-8")

    audit = {
        "schema": SCHEMA,
        "status": STATUS,
        "gates": {
            "authority_registry_pass": source["authority_registry_status"] == AUTHORITY_REGISTRY_PASS,
            "source_commit_exact": source["source_commit"] == EXPECTED_SOURCE_COMMIT,
            "all_authority_and_hourly_lineage_gates_pass": True,
            "hourly_snapshot_grid_identical": len(source["snapshot_steps"]) == EXPECTED_SNAPSHOT_COUNT,
            "complete_A_B_C_snapshot_population": len(snapshots) == 3 * EXPECTED_SNAPSHOT_COUNT,
            "source_descriptor_closure": max(source["descriptor_closure_max"].values()) <= 5.0e-12,
            "frozen_transport_contract": True,
            "full_psd_direct_sum": max_direct_sum <= 5.0e-13,
            "lognormal_moment_closure": max_log_mean <= 5.0e-13 and max_log_cv <= 5.0e-13,
            "prediction_grid_complete": len(prediction_rows) == 3 * 45 * 7 * 2 * 5,
            "full_psd_trajectory_grid_complete": len(trajectory_rows) == 3 * 45 * 7,
            "full_psd_ensemble_grid_complete": len(ensemble_rows) == 45 * 7,
            "registered_six_time_psd_hash_identity": registered_hash_match,
            "registered_six_time_transport_identity": max_registered_difference <= 5.0e-13,
            "production_pf_untouched": True,
        },
        "source": source,
        "transport": {
            "A_N": 1.5,
            "dislocation_mode": "DISLOCATION_OFF",
            "S11_rate": 0.0,
            "S13_rate": 0.0,
            "yu_refit_scale_used": False,
            "temperatures_K": TEMPERATURES_K,
            "matrix_modes": MATRIX_MODES,
            "descriptor_models": MODELS,
            "lognormal_quadrature_order": descriptors.LOGNORMAL_ORDER,
        },
        "numerics": {
            "full_psd_vector_direct_sum_max_relative": max_direct_sum,
            "lognormal_mean_max_relative": max_log_mean,
            "lognormal_CV_max_absolute": max_log_cv,
            "registered_six_time_transport_max_relative": max_registered_difference,
        },
        "summary": summary_rows,
    }
    if not all(audit["gates"].values()):
        raise ValueError(f"hourly transport gate failed: {audit['gates']}")
    write_json(output_root / "hourly_transport_audit.json", audit)
    (output_root / "final_status.txt").write_text(
        "\n".join(
            [
                f"status={STATUS}",
                f"source_authority_status={AUTHORITY_REGISTRY_PASS}",
                f"source_snapshot_count={len(snapshots)}",
                f"prediction_cell_count={len(prediction_rows)}",
                "full_PSD_reference=true",
                "dislocation_mode=DISLOCATION_OFF",
                "yu_refit_scale_used=false",
                "production_PF_modified=false",
                "absolute_experimental_kappa_claim=false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    targets = sorted(path for path in output_root.iterdir() if path.is_file() and path.name != "analysis_manifest.sha256")
    (output_root / "analysis_manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in targets), encoding="utf-8"
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--authority-root",
        type=Path,
        default=Path("reports/pf_246cube_method1_production_authority_v1"),
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
        default=Path("data/qualification/pf_full_psd_no_dislocation_transport_v1/transport_parameter_contract.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = analyze(
        args.authority_root.resolve(),
        args.output_root.resolve(),
        args.yu_config.resolve(),
        args.transport_contract.resolve(),
    )
    print(audit["status"])


if __name__ == "__main__":
    main()
