#!/usr/bin/env python3
"""Build one fail-closed transport authority from a 246^3 production PASS.

The production audit and the historical transport smoke intentionally use
different column names.  This adapter verifies the production authority,
normalizes only those registered observables needed by the transport
interface, and preserves hashes for every raw source used in the conversion.
It never launches, resumes, or modifies a PF calculation.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import pf_full_psd_no_dislocation_transport_v1 as core


PRODUCTION_SCHEMA = "PF_246CUBE_6H48H_PRODUCTION_AUDIT_V1"
PRODUCTION_PASS = "PASS_246CUBE_6H48H_CONDITIONAL_PRODUCTION_V1"
CAMPAIGN_SCHEMA = "PF_246CUBE_6H48H_PRODUCTION_CAMPAIGN_V1"
REGISTERED_STEP_TO_AGE_H = {
    0: 6.0,
    21798: 12.0,
    43596: 18.0,
    65393: 24.0,
    108989: 36.0,
    152585: 48.0,
}
SOURCE_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
HASH_LINE_PATTERN = re.compile(r"^([0-9a-f]{64})\s+(.+?)\s*$")


def _normalize_replicate(value: str) -> str:
    normalized = value.removeprefix("replicate_").upper()
    if normalized not in ("A", "B", "C"):
        raise ValueError(f"Replicate must be A, B, or C, got {value!r}")
    return normalized


def _read_hash_ledger(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        match = HASH_LINE_PATTERN.fullmatch(raw)
        if match is None:
            raise ValueError(f"Malformed SHA-256 ledger line {line_number}: {path}")
        entries.append((match.group(1), match.group(2)))
    if not entries:
        raise ValueError(f"Empty SHA-256 ledger: {path}")
    return entries


def _unique_hash_for_basename(
    entries: Sequence[tuple[str, str]], basename: str, ledger: Path
) -> str:
    matches = {digest for digest, name in entries if Path(name).name == basename}
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {basename!r} hash in {ledger}, got {len(matches)}"
        )
    return next(iter(matches))


def _single_analysis_binary_hash(path: Path) -> str:
    entries = _read_hash_ledger(path)
    hashes = {digest for digest, _ in entries}
    if len(hashes) != 1:
        raise ValueError(
            f"Expected exactly one analysis-binary hash in {path}, got {len(hashes)}"
        )
    return next(iter(hashes))


def _validate_campaign(
    campaign: dict[str, Any], replicate: str, fixture_sha256: str
) -> None:
    if campaign.get("schema") != CAMPAIGN_SCHEMA:
        raise ValueError("Unexpected production campaign schema")
    if _normalize_replicate(str(campaign.get("replicate"))) != replicate:
        raise ValueError("Campaign replicate does not match requested authority")
    if campaign.get("fixture_manifest_sha256") != fixture_sha256:
        raise ValueError("Campaign fixture hash does not match the supplied fixture")
    if int(campaign.get("final_step", -1)) != max(REGISTERED_STEP_TO_AGE_H):
        raise ValueError("Campaign final step is not the frozen 48 h endpoint")
    if campaign.get("registered_science_steps") != list(REGISTERED_STEP_TO_AGE_H):
        raise ValueError("Campaign registered science steps are not frozen")
    disabled = (
        "GP_enabled",
        "GP_birth_enabled",
        "GP_release_enabled",
        "external_source_enabled",
        "new_beta_nucleation_enabled",
    )
    if not all(campaign.get(name) is False for name in disabled):
        raise ValueError("A prohibited GP/source/nucleation path is enabled")


def validate_production_authority(
    *,
    production_audit_dir: Path,
    fixture_manifest: Path,
    campaign_manifest: Path,
    input_hashes: Path,
    analysis_binary_hashes: Path,
    replicate: str,
    trajectory_class: str,
    source_commit: str,
) -> dict[str, Any]:
    replicate = _normalize_replicate(replicate)
    if "EXPERIMENT_MATRIX_ANCHORED" not in trajectory_class:
        raise ValueError("Production transport authority is not experiment-matrix-anchored")
    if SOURCE_COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise ValueError("source_commit must be a full 40-character lowercase Git SHA")

    status_path = production_audit_dir / "status.txt"
    audit_path = production_audit_dir / "audit.json"
    observations_path = production_audit_dir / "registered_observables.csv"
    particles_path = production_audit_dir / "registered_particle_psd.csv"
    required_files = (
        status_path,
        audit_path,
        observations_path,
        particles_path,
        fixture_manifest,
        campaign_manifest,
        input_hashes,
        analysis_binary_hashes,
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise ValueError(f"Missing production authority inputs: {missing}")
    if status_path.read_text(encoding="utf-8").strip() != PRODUCTION_PASS:
        raise ValueError("Production status.txt is not exact PASS")

    audit = core.load_json(audit_path)
    if audit.get("schema") != PRODUCTION_SCHEMA:
        raise ValueError("Unexpected production audit schema")
    if audit.get("numerical_status") != PRODUCTION_PASS:
        raise ValueError("Production audit numerical status is not exact PASS")
    gates = audit.get("gates")
    if not isinstance(gates, dict) or not gates or not all(
        value is True for value in gates.values()
    ):
        raise ValueError("Production audit does not have an all-true gate set")
    if gates.get("merge_aware_particle_lineage") is not True:
        raise ValueError("Production merge-aware particle-lineage gate is not PASS")
    checkpoint_hashes = audit.get("checkpoint_hashes")
    if (
        not isinstance(checkpoint_hashes, dict)
        or len(checkpoint_hashes) != 44
        or not all(
            isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None
            for value in checkpoint_hashes.values()
        )
    ):
        raise ValueError("Production audit does not preserve the complete 44-checkpoint hash chain")
    if _normalize_replicate(str(audit.get("replicate"))) != replicate:
        raise ValueError("Production audit replicate does not match requested authority")

    fixture_sha256 = core.file_sha256(fixture_manifest)
    if audit.get("fixture_manifest_sha256") != fixture_sha256:
        raise ValueError("Production audit fixture hash does not match the supplied fixture")
    campaign = core.load_json(campaign_manifest)
    _validate_campaign(campaign, replicate, fixture_sha256)

    ledger = _read_hash_ledger(input_hashes)
    source_binary_sha256 = _unique_hash_for_basename(
        ledger, "main_cuda", input_hashes
    )
    source_parameter_sha256 = _unique_hash_for_basename(
        ledger, "pf_input_dt0p02.params", input_hashes
    )
    analysis_binary_sha256 = _single_analysis_binary_hash(analysis_binary_hashes)
    return {
        "replicate": replicate,
        "audit": audit,
        "campaign": campaign,
        "fixture_manifest_sha256": fixture_sha256,
        "source_binary_sha256": source_binary_sha256,
        "source_parameter_sha256": source_parameter_sha256,
        "analysis_binary_sha256": analysis_binary_sha256,
        "paths": {
            "status": status_path,
            "audit": audit_path,
            "observations": observations_path,
            "particles": particles_path,
            "fixture": fixture_manifest,
            "campaign": campaign_manifest,
            "input_hashes": input_hashes,
            "analysis_binary_hashes": analysis_binary_hashes,
        },
    }


def load_production_snapshots(
    authority: dict[str, Any],
) -> tuple[list[dict[str, Any]], float, dict[str, float], list[dict[str, Any]]]:
    observations = core.read_csv(authority["paths"]["observations"])
    particle_rows = core.read_csv(authority["paths"]["particles"])
    expected_steps = list(REGISTERED_STEP_TO_AGE_H)
    observed_steps = [int(row["step"]) for row in observations]
    if observed_steps != expected_steps:
        raise ValueError(
            f"Registered production steps mismatch: {observed_steps} != {expected_steps}"
        )
    particles_by_step: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in particle_rows:
        step = int(row["step"])
        if step not in REGISTERED_STEP_TO_AGE_H:
            raise ValueError(f"Unregistered particle-PSD step: {step}")
        particles_by_step[step].append(row)

    snapshots: list[dict[str, Any]] = []
    timing: list[dict[str, Any]] = []
    box_volumes_nm3: list[float] = []
    closure_max = {"Nv": 0.0, "Sv": 0.0, "M6": 0.0, "mean_R": 0.0}
    required_observation_fields = {
        "step",
        "elapsed_physical_time_s",
        "experimental_age_h",
        "particle_count",
        "number_density_m3",
        "mean_radius_nm",
        "Sv_nm_inv",
        "M6_nm3",
        "beta_volume_fraction",
        "far_field_matrix_xAg",
        "far_field_matrix_xB",
    }
    for row in observations:
        missing = required_observation_fields - set(row)
        if missing:
            raise ValueError(f"Production observation fields missing: {sorted(missing)}")
        step = int(row["step"])
        registered_age_h = REGISTERED_STEP_TO_AGE_H[step]
        source_age_h = float(row["experimental_age_h"])
        elapsed_s = float(row["elapsed_physical_time_s"])
        if not math.isfinite(source_age_h) or not math.isfinite(elapsed_s):
            raise ValueError(f"Non-finite production time at step {step}")
        expected_count = int(row["particle_count"])
        group = particles_by_step.get(step, [])
        if expected_count <= 0 or len(group) != expected_count:
            raise ValueError(
                f"Production PSD count mismatch at step {step}: "
                f"{len(group)} != {expected_count}"
            )
        if any("stable_particle_id" not in item for item in group):
            raise ValueError("Production PSD lacks stable_particle_id")
        stable_ids = [int(item["stable_particle_id"]) for item in group]
        if len(set(stable_ids)) != expected_count:
            raise ValueError(f"Duplicate stable_particle_id at step {step}")
        ordered = sorted(group, key=lambda item: int(item["stable_particle_id"]))
        radii_nm = np.array(
            [float(item["equivalent_radius_nm"]) for item in ordered], dtype=float
        )
        nv_m3 = float(row["number_density_m3"])
        if nv_m3 <= 0.0 or not math.isfinite(nv_m3):
            raise ValueError(f"Invalid production number density at step {step}")
        box_volume_nm3 = expected_count / (nv_m3 * 1.0e-27)
        box_volumes_nm3.append(box_volume_nm3)
        moments = core.moments_from_radii(
            radii_nm * 1.0e-9, box_volume_nm3 * 1.0e-27
        )
        source_values = {
            "Nv": nv_m3,
            "Sv": float(row["Sv_nm_inv"]) * 1.0e9,
            "M6": float(row["M6_nm3"]) * 1.0e-27,
            "mean_R": float(row["mean_radius_nm"]) * 1.0e-9,
        }
        calculated_values = {
            "Nv": moments["Nv_m-3"],
            "Sv": moments["Sv_m-1"],
            "M6": moments["M6_m3"],
            "mean_R": moments["mean_radius_m"],
        }
        for name in closure_max:
            closure_max[name] = max(
                closure_max[name],
                core.relative_error(calculated_values[name], source_values[name]),
            )
        snapshot = {
            "snapshot_id": f"{authority['replicate']}_step_{step:06d}",
            "replicate": authority["replicate"],
            "step": step,
            "registered_age_h": registered_age_h,
            "age_h": registered_age_h,
            "elapsed_physical_time_s": elapsed_s,
            "particle_count": expected_count,
            "box_volume_nm3": box_volume_nm3,
            "Nv_nm-3": nv_m3 * 1.0e-27,
            "mean_radius_nm": float(row["mean_radius_nm"]),
            "Sv_nm-1": float(row["Sv_nm_inv"]),
            "M6_nm3": float(row["M6_nm3"]),
            "beta_volume_fraction": float(row["beta_volume_fraction"]),
            "matrix_xAg": float(row["far_field_matrix_xAg"]),
            "matrix_xB": float(row["far_field_matrix_xB"]),
            "matrix_xAg_source_column": "far_field_matrix_xAg",
            "matrix_xB_source_column": "far_field_matrix_xB",
            "matrix_observation_contract": (
                "production registered far-field matrix observation"
            ),
            "radii_nm": radii_nm.tolist(),
            "particle_ids": [int(item["stable_particle_id"]) for item in ordered],
        }
        snapshot["full_psd_canonical_sha256"] = core.canonical_sha256(
            list(zip(snapshot["particle_ids"], snapshot["radii_nm"]))
        )
        snapshots.append(snapshot)
        timing.append(
            {
                "step": step,
                "registered_age_h": registered_age_h,
                "source_experimental_age_h": source_age_h,
                "elapsed_physical_time_s": elapsed_s,
            }
        )

    if set(particles_by_step) != set(expected_steps):
        raise ValueError("Production PSD does not cover exactly the registered steps")
    reference_box_volume = float(np.mean(box_volumes_nm3))
    box_relative_spread = max(
        abs(value - reference_box_volume) / reference_box_volume
        for value in box_volumes_nm3
    )
    if box_relative_spread > 1.0e-12:
        raise ValueError("Production PF box volume is inconsistent across snapshots")
    if any(value > 5.0e-12 for value in closure_max.values()):
        raise ValueError(f"Production PSD descriptor closure failed: {closure_max}")
    return snapshots, reference_box_volume, closure_max, timing


def build_production_provenance(
    *,
    authority: dict[str, Any],
    trajectory_class: str,
    source_commit: str,
    yu_config: Path,
    interface_contract: Path,
    timing: list[dict[str, Any]],
    supplemental_provenance: Sequence[Path] = (),
) -> dict[str, Any]:
    paths = authority["paths"]
    supplemental = list(supplemental_provenance)
    missing_supplemental = [str(path) for path in supplemental if not path.is_file()]
    if missing_supplemental:
        raise ValueError(f"Missing supplemental provenance: {missing_supplemental}")
    provenance = {
        "trajectory_class": trajectory_class,
        "replicate": authority["replicate"],
        "source_commit": source_commit,
        "source_binary_sha256": authority["source_binary_sha256"],
        "source_parameter_sha256": authority["source_parameter_sha256"],
        "analysis_binary_sha256": authority["analysis_binary_sha256"],
        "fixture_manifest": str(paths["fixture"].resolve()),
        "fixture_manifest_sha256": authority["fixture_manifest_sha256"],
        "production_audit_status": authority["audit"]["numerical_status"],
        "production_audit_gates": authority["audit"]["gates"],
        "production_checkpoint_hashes": authority["audit"].get(
            "checkpoint_hashes", {}
        ),
        "identity_use_contract": (
            "overall registered snapshot PSD under exact merge-aware production PASS"
        ),
        "registered_snapshot_timing": timing,
        "microstructure_csv": str(paths["observations"].resolve()),
        "microstructure_csv_sha256": core.file_sha256(paths["observations"]),
        "particle_psd_csv": str(paths["particles"].resolve()),
        "particle_psd_csv_sha256": core.file_sha256(paths["particles"]),
        "upstream_audit_json": str(paths["audit"].resolve()),
        "upstream_audit_json_sha256": core.file_sha256(paths["audit"]),
        "campaign_manifest": str(paths["campaign"].resolve()),
        "campaign_manifest_sha256": core.file_sha256(paths["campaign"]),
        "input_hash_ledger": str(paths["input_hashes"].resolve()),
        "input_hash_ledger_sha256": core.file_sha256(paths["input_hashes"]),
        "upstream_analysis_manifest": str(paths["analysis_binary_hashes"].resolve()),
        "upstream_analysis_manifest_sha256": core.file_sha256(
            paths["analysis_binary_hashes"]
        ),
        "yu_public_parameter_config": str(yu_config.resolve()),
        "yu_public_parameter_config_sha256": core.file_sha256(yu_config),
        "transport_parameter_contract": str(interface_contract.resolve()),
        "transport_parameter_contract_sha256": core.file_sha256(interface_contract),
        "transport_interface_script": str(Path(core.__file__).resolve()),
        "transport_interface_script_sha256": core.file_sha256(Path(core.__file__)),
        "production_adapter_script": str(Path(__file__).resolve()),
        "production_adapter_script_sha256": core.file_sha256(Path(__file__)),
        "supplemental_provenance": [
            {"path": str(path.resolve()), "sha256": core.file_sha256(path)}
            for path in supplemental
        ],
    }
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
        "yu_public_parameter_config_sha256",
        "transport_parameter_contract_sha256",
        "transport_interface_script_sha256",
        "production_adapter_script_sha256",
    )
    missing = [name for name in required_hashes if not provenance.get(name)]
    if missing:
        raise ValueError(f"Required production transport provenance missing: {missing}")
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-audit-dir", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--input-hashes", type=Path, required=True)
    parser.add_argument("--analysis-binary-hashes", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--replicate", choices=("A", "B", "C"), required=True)
    parser.add_argument("--trajectory-class", required=True)
    parser.add_argument(
        "--yu-config",
        type=Path,
        default=Path("data/qualification/yu2024_transport_v1/yu_48h_parameters.json"),
    )
    parser.add_argument(
        "--interface-contract", type=Path, default=core.DEFAULT_INTERFACE_CONTRACT
    )
    parser.add_argument(
        "--supplemental-provenance", type=Path, action="append", default=[]
    )
    parser.add_argument("--fixed-matrix-xag", type=float)
    parser.add_argument(
        "--temperatures-K",
        default=",".join(str(value) for value in core.DEFAULT_TEMPERATURES_K),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output_dir}")
    config = core.load_json(args.yu_config)
    core.validate_yu_base_config(config, args.yu_config)
    core.validate_interface_contract(
        core.load_json(args.interface_contract),
        args.interface_contract,
        args.yu_config,
    )
    authority = validate_production_authority(
        production_audit_dir=args.production_audit_dir,
        fixture_manifest=args.fixture_manifest,
        campaign_manifest=args.campaign_manifest,
        input_hashes=args.input_hashes,
        analysis_binary_hashes=args.analysis_binary_hashes,
        replicate=args.replicate,
        trajectory_class=args.trajectory_class,
        source_commit=args.source_commit,
    )
    snapshots, box_volume_nm3, closure, timing = load_production_snapshots(authority)
    provenance = build_production_provenance(
        authority=authority,
        trajectory_class=args.trajectory_class,
        source_commit=args.source_commit,
        yu_config=args.yu_config,
        interface_contract=args.interface_contract,
        timing=timing,
        supplemental_provenance=args.supplemental_provenance,
    )
    core.export_transport_snapshot(
        output_dir=args.output_dir,
        snapshots=snapshots,
        box_volume_nm3=box_volume_nm3,
        closure=closure,
        provenance=provenance,
        config=config,
        temperatures_K=core.parse_temperatures(args.temperatures_K),
        fixed_matrix_xag=args.fixed_matrix_xag,
    )
    print(core.FINAL_STATUS)


if __name__ == "__main__":
    main()
