#!/usr/bin/env python3
"""Fail-closed A/B/C authority selection and no-dislocation ensemble assembly."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import pf_full_psd_no_dislocation_transport_v1 as core


AUTHORITY_SCHEMA = "PF_FULL_PSD_TRANSPORT_AUTHORITY_SELECTION_V1"
ENSEMBLE_STATUS = "PASS_PF_FULL_PSD_NO_DISLOCATION_TRANSPORT_ENSEMBLE_V1"
PRODUCTION_PASS = "PASS_246CUBE_6H48H_CONDITIONAL_PRODUCTION_V1"
REQUIRED_REPLICATES = ("A", "B", "C")
REQUIRED_SCIENCE_AGES_H = (6.0, 12.0, 18.0, 24.0, 36.0, 48.0)
PRODUCTION_ADAPTER = Path(__file__).with_name(
    "build_pf_246cube_no_dislocation_transport_authority_v1.py"
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_production_provenance(
    provenance: dict[str, Any], replicate: str
) -> None:
    source_commit = provenance.get("source_commit")
    if not (
        isinstance(source_commit, str)
        and len(source_commit) == 40
        and all(character in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError(f"Replicate {replicate} source commit is not a full Git SHA")
    if provenance.get("production_audit_status") != PRODUCTION_PASS:
        raise ValueError(
            f"Replicate {replicate} does not preserve an exact production PASS"
        )
    gates = provenance.get("production_audit_gates")
    if not isinstance(gates, dict) or not gates or not all(
        value is True for value in gates.values()
    ):
        raise ValueError(f"Replicate {replicate} production gates are not all true")
    if gates.get("merge_aware_particle_lineage") is not True:
        raise ValueError(f"Replicate {replicate} merge-aware gate is not PASS")
    checkpoint_hashes = provenance.get("production_checkpoint_hashes")
    if (
        not isinstance(checkpoint_hashes, dict)
        or len(checkpoint_hashes) != 44
        or not all(_is_sha256(value) for value in checkpoint_hashes.values())
    ):
        raise ValueError(
            f"Replicate {replicate} does not preserve the complete 44-checkpoint hash chain"
        )
    required_hashes = (
        "source_binary_sha256",
        "source_parameter_sha256",
        "analysis_binary_sha256",
        "fixture_manifest_sha256",
        "microstructure_csv_sha256",
        "particle_psd_csv_sha256",
        "upstream_audit_json_sha256",
        "campaign_manifest_sha256",
        "input_hash_ledger_sha256",
        "upstream_analysis_manifest_sha256",
        "production_adapter_script_sha256",
    )
    missing = [
        name for name in required_hashes if not _is_sha256(provenance.get(name))
    ]
    if missing:
        raise ValueError(
            f"Replicate {replicate} production provenance hashes are missing/invalid: {missing}"
        )
    if not PRODUCTION_ADAPTER.is_file():
        raise ValueError("Frozen production transport adapter is missing")
    expected_adapter_hash = core.file_sha256(PRODUCTION_ADAPTER)
    if provenance["production_adapter_script_sha256"] != expected_adapter_hash:
        raise ValueError(f"Replicate {replicate} production adapter hash mismatch")


def _validate_transport_grid(
    *,
    replicate: str,
    snapshots: list[dict[str, str]],
    kappa_rows: list[dict[str, str]],
    descriptor_rows: list[dict[str, str]],
    temperatures_K: Sequence[float],
    required_ages_h: Sequence[float],
) -> None:
    snapshot_ages = [float(row["registered_age_h"]) for row in snapshots]
    if snapshot_ages != list(required_ages_h):
        raise ValueError(
            f"Replicate {replicate} registered snapshot ages mismatch: {snapshot_ages}"
        )
    expected_kappa = {
        (float(age), float(temperature))
        for age in required_ages_h
        for temperature in temperatures_K
    }
    observed_kappa = Counter(
        (float(row["age_h"]), float(row["temperature_K"])) for row in kappa_rows
    )
    if set(observed_kappa) != expected_kappa or any(
        count != 1 for count in observed_kappa.values()
    ):
        raise ValueError(f"Replicate {replicate} kappa time-temperature grid mismatch")
    expected_descriptor = {
        (float(age), float(temperature), matrix_mode, model)
        for age in required_ages_h
        for temperature in temperatures_K
        for matrix_mode in core.MATRIX_MODES
        for model in core.DESCRIPTOR_MODELS
    }
    observed_descriptor = Counter(
        (
            float(row["age_h"]),
            float(row["temperature_K"]),
            row["matrix_mode"],
            row["descriptor_model"],
        )
        for row in descriptor_rows
    )
    if set(observed_descriptor) != expected_descriptor or any(
        count != 1 for count in observed_descriptor.values()
    ):
        raise ValueError(f"Replicate {replicate} descriptor grid mismatch")


def _verify_output_hashes(output_dir: Path, manifest: dict[str, Any]) -> None:
    for name, expected in manifest["outputs"].items():
        path = output_dir / name
        if not path.is_file():
            raise ValueError(f"Missing transport output: {path}")
        actual = core.file_sha256(path)
        if actual != expected:
            raise ValueError(f"Transport output hash mismatch: {path}")


def _select_candidate(replicate_entry: dict[str, Any]) -> dict[str, Any]:
    replicate = replicate_entry["replicate"]
    selected = [item for item in replicate_entry["candidates"] if item.get("selected")]
    if len(selected) != 1:
        raise ValueError(
            f"Replicate {replicate} must have exactly one selected authority, got {len(selected)}"
        )
    candidate = selected[0]
    if candidate.get("authority_status") != "PASS_COMPLETE_6H48H":
        raise ValueError(f"Replicate {replicate} selected authority is not complete PASS")
    if candidate.get("complete_6h48h") is not True:
        raise ValueError(f"Replicate {replicate} is not marked complete_6h48h=true")
    return candidate


def load_authorities(
    authority_path: Path,
    required_ages_h: Sequence[float] = REQUIRED_SCIENCE_AGES_H,
) -> list[dict[str, Any]]:
    authority = core.load_json(authority_path)
    if authority.get("schema") != AUTHORITY_SCHEMA:
        raise ValueError("Unexpected A/B/C authority schema")
    entries = authority.get("replicates", [])
    names = tuple(sorted(item.get("replicate") for item in entries))
    if names != REQUIRED_REPLICATES:
        raise ValueError(f"Authority manifest must contain exactly A/B/C, got {names}")
    loaded: list[dict[str, Any]] = []
    trajectory_identities: set[str] = set()
    for entry in sorted(entries, key=lambda item: item["replicate"]):
        replicate = entry["replicate"]
        candidate = _select_candidate(entry)
        output_dir = Path(candidate["transport_output_dir"])
        manifest_path = output_dir / "transport_snapshot_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Missing transport manifest for replicate {replicate}")
        expected_manifest_hash = candidate.get("transport_manifest_sha256")
        actual_manifest_hash = core.file_sha256(manifest_path)
        if expected_manifest_hash != actual_manifest_hash:
            raise ValueError(f"Replicate {replicate} transport manifest hash mismatch")
        manifest = core.load_json(manifest_path)
        if manifest.get("status") != core.FINAL_STATUS:
            raise ValueError(f"Replicate {replicate} transport interface is not PASS")
        contract = manifest["transport_contract"]
        if not (
            contract["A_N"] == core.FROZEN_A_N
            and contract["S11_rate"] == 0.0
            and contract["S13_rate"] == 0.0
            and contract["dislocation_mode"] == core.DISLOCATION_MODE
            and contract["yu_refit_scale_used"] is False
        ):
            raise ValueError(f"Replicate {replicate} violates the frozen transport contract")
        if manifest["provenance"].get("replicate") != replicate:
            raise ValueError(f"Replicate label mismatch in {manifest_path}")
        provenance = manifest["provenance"]
        trajectory_class = provenance.get("trajectory_class", "")
        if "EXPERIMENT_MATRIX_ANCHORED" not in trajectory_class:
            raise ValueError(
                f"Replicate {replicate} is not experiment-matrix-anchored: {trajectory_class}"
            )
        _validate_production_provenance(provenance, replicate)
        _verify_output_hashes(output_dir, manifest)
        snapshots = core.read_csv(output_dir / "transport_snapshots.csv")
        kappa_rows = core.read_csv(output_dir / "kappa_time_temperature.csv")
        descriptor_rows = core.read_csv(output_dir / "descriptor_predictions.csv")
        temperatures_K = tuple(float(value) for value in contract["temperature_grid_K"])
        _validate_transport_grid(
            replicate=replicate,
            snapshots=snapshots,
            kappa_rows=kappa_rows,
            descriptor_rows=descriptor_rows,
            temperatures_K=temperatures_K,
            required_ages_h=required_ages_h,
        )
        identity = core.canonical_sha256(
            {
                "source_commit": manifest["provenance"].get("source_commit"),
                "source_binary": manifest["provenance"].get("source_binary_sha256"),
                "fixture": manifest["provenance"].get("fixture_manifest_sha256"),
                "particle_psd": manifest["provenance"].get("particle_psd_csv_sha256"),
            }
        )
        if identity in trajectory_identities:
            raise ValueError("The same authoritative trajectory was selected more than once")
        trajectory_identities.add(identity)
        loaded.append(
            {
                "replicate": replicate,
                "candidate_id": candidate["candidate_id"],
                "output_dir": output_dir,
                "manifest_path": manifest_path,
                "manifest_sha256": actual_manifest_hash,
                "manifest": manifest,
                "snapshots": snapshots,
                "kappa_rows": kappa_rows,
                "descriptor_rows": descriptor_rows,
                "temperature_grid_K": temperatures_K,
                "trajectory_identity_sha256": identity,
            }
        )
    return loaded


def assemble_ensemble(
    authorities: list[dict[str, Any]],
    output_dir: Path,
    required_ages_h: Sequence[float] = REQUIRED_SCIENCE_AGES_H,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = (
        "kappa_L_no_dis_fixed_matrix_W_mK",
        "kappa_L_no_dis_time_varying_matrix_W_mK",
        "delta_kappa_PSD_W_mK",
        "delta_kappa_PSD_plus_matrix_W_mK",
        "delta_kappa_matrix_given_PSD_W_mK",
    )
    grouped: dict[tuple[float, float], list[dict[str, str]]] = defaultdict(list)
    for authority in authorities:
        for row in authority["kappa_rows"]:
            age = float(row["age_h"])
            if age in required_ages_h:
                grouped[(age, float(row["temperature_K"]))].append(row)
    ensemble_rows: list[dict[str, Any]] = []
    for (age, temperature), rows in sorted(grouped.items()):
        if len(rows) != len(REQUIRED_REPLICATES):
            raise ValueError(
                f"Ensemble cell age={age}, T={temperature} has {len(rows)} replicates"
            )
        output: dict[str, Any] = {
            "age_h": f"{age:.16g}",
            "temperature_K": f"{temperature:.16g}",
            "replicate_count": len(rows),
        }
        for field in fields:
            values = np.array([float(row[field]) for row in rows], dtype=float)
            output[f"{field}_mean"] = f"{float(np.mean(values)):.16g}"
            output[f"{field}_sample_std"] = f"{float(np.std(values, ddof=1)):.16g}"
            output[f"{field}_min"] = f"{float(np.min(values)):.16g}"
            output[f"{field}_max"] = f"{float(np.max(values)):.16g}"
        ensemble_rows.append(output)
    if not ensemble_rows:
        raise ValueError("No common A/B/C ensemble cells were assembled")
    core.write_csv(
        output_dir / "ensemble_kappa_time_temperature.csv",
        list(ensemble_rows[0].keys()),
        ensemble_rows,
    )

    descriptor_summary_rows: list[dict[str, Any]] = []
    for matrix_mode in core.MATRIX_MODES:
        for model in core.DESCRIPTOR_MODELS:
            selected = []
            for authority in authorities:
                selected.extend(
                    row
                    for row in authority["descriptor_rows"]
                    if row["matrix_mode"] == matrix_mode
                    and row["descriptor_model"] == model
                    and float(row["age_h"]) in required_ages_h
                )
            relative = np.array(
                [float(row["relative_error_vs_full_psd"]) for row in selected]
            )
            absolute = np.array(
                [float(row["absolute_error_vs_full_psd_W_mK"]) for row in selected]
            )
            descriptor_summary_rows.append(
                {
                    "matrix_mode": matrix_mode,
                    "descriptor_model": model,
                    "observations": len(selected),
                    "MAPE_fraction_vs_full_psd": f"{float(np.mean(relative)):.16g}",
                    "max_relative_error_vs_full_psd": f"{float(np.max(relative)):.16g}",
                    "RMSE_W_mK_vs_full_psd": f"{float(np.sqrt(np.mean(absolute**2))):.16g}",
                }
            )
    core.write_csv(
        output_dir / "ensemble_descriptor_error_summary.csv",
        list(descriptor_summary_rows[0].keys()),
        descriptor_summary_rows,
    )
    selection = {
        "schema": "PF_FULL_PSD_NO_DISLOCATION_ENSEMBLE_V1",
        "status": ENSEMBLE_STATUS,
        "scientific_claim": (
            "conditional A/B/C resolved-PF-PSD no-dislocation ensemble; "
            "not absolute experimental kappa reproduction"
        ),
        "required_science_ages_h": list(required_ages_h),
        "replicates": [
            {
                "replicate": item["replicate"],
                "candidate_id": item["candidate_id"],
                "transport_manifest": str(item["manifest_path"].resolve()),
                "transport_manifest_sha256": item["manifest_sha256"],
                "trajectory_identity_sha256": item["trajectory_identity_sha256"],
            }
            for item in authorities
        ],
        "outputs": {
            "ensemble_kappa_time_temperature.csv": core.file_sha256(
                output_dir / "ensemble_kappa_time_temperature.csv"
            ),
            "ensemble_descriptor_error_summary.csv": core.file_sha256(
                output_dir / "ensemble_descriptor_error_summary.csv"
            ),
        },
        "analysis_script_sha256": core.file_sha256(Path(__file__)),
    }
    core.write_json(output_dir / "ensemble_manifest.json", selection)
    return selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authorities = load_authorities(args.authority_manifest)
    assemble_ensemble(authorities, args.output_dir)


if __name__ == "__main__":
    main()
