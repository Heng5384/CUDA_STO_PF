#!/usr/bin/env python3
"""Qualify the 246^3 production-PASS to transport-authority adapter."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import build_pf_246cube_no_dislocation_transport_authority_v1 as adapter
import pf_full_psd_no_dislocation_transport_v1 as core


STATUS = "PASS_PF_246CUBE_NO_DISLOCATION_TRANSPORT_AUTHORITY_ADAPTER_V1"
BLOCKED = "BLOCKED_PF_246CUBE_NO_DISLOCATION_TRANSPORT_AUTHORITY_ADAPTER_V1"


def _synthetic_digest(token: str) -> str:
    return core.canonical_sha256({"qualification_token": token})


def build_synthetic_production_fixture(
    historical_transport_dir: Path, fixture_root: Path
) -> dict[str, Path]:
    fixture_root.mkdir(parents=True)
    source_snapshots = core.read_csv(
        historical_transport_dir / "transport_snapshots.csv"
    )
    source_particles = core.read_csv(
        historical_transport_dir / "full_psd_particles.csv"
    )
    selected_ages = set(adapter.REGISTERED_STEP_TO_AGE_H.values())
    snapshots_by_age = {
        float(row["age_h"]): row
        for row in source_snapshots
        if float(row["age_h"]) in selected_ages
    }
    if set(snapshots_by_age) != selected_ages:
        raise ValueError("Historical smoke lacks the six qualification ages")
    particles_by_age: dict[float, list[dict[str, str]]] = {}
    for age in selected_ages:
        particles_by_age[age] = [
            row for row in source_particles if float(row["age_h"]) == age
        ]

    fixture_manifest = fixture_root / "fixture_manifest.json"
    core.write_json(
        fixture_manifest,
        {
            "schema": "PF_246CUBE_TRANSPORT_ADAPTER_SYNTHETIC_FIXTURE_V1",
            "replicate_id": "replicate_A",
            "qualification_only": True,
        },
    )
    fixture_sha = core.file_sha256(fixture_manifest)
    campaign_manifest = fixture_root / "campaign_manifest.json"
    core.write_json(
        campaign_manifest,
        {
            "schema": adapter.CAMPAIGN_SCHEMA,
            "replicate": "replicate_A",
            "fixture_manifest_sha256": fixture_sha,
            "final_step": max(adapter.REGISTERED_STEP_TO_AGE_H),
            "registered_science_steps": list(adapter.REGISTERED_STEP_TO_AGE_H),
            "GP_enabled": False,
            "GP_birth_enabled": False,
            "GP_release_enabled": False,
            "external_source_enabled": False,
            "new_beta_nucleation_enabled": False,
            "qualification_only": True,
        },
    )
    source_binary_hash = _synthetic_digest("source_binary")
    source_parameter_hash = _synthetic_digest("source_parameter")
    analysis_binary_hash = _synthetic_digest("analysis_binary")
    input_hashes = fixture_root / "input_hashes.sha256"
    input_hashes.write_text(
        f"{source_binary_hash}  /qualification/main_cuda\n"
        f"{source_parameter_hash}  /qualification/pf_input_dt0p02.params\n",
        encoding="utf-8",
    )
    analysis_hashes = fixture_root / "analysis_binary.sha256"
    analysis_hashes.write_text(
        f"{analysis_binary_hash}  /qualification/analyze_pf_246cube_particle_lineage_v1\n",
        encoding="utf-8",
    )

    audit_dir = fixture_root / "production_audit"
    audit_dir.mkdir()
    observation_rows: list[dict[str, Any]] = []
    particle_rows: list[dict[str, Any]] = []
    for step, age in adapter.REGISTERED_STEP_TO_AGE_H.items():
        source = snapshots_by_age[age]
        group = particles_by_age[age]
        observation_rows.append(
            {
                "replicate": "replicate_A",
                "step": step,
                "elapsed_physical_time_s": float(age - 6.0) * 3600.0,
                "experimental_age_h": age,
                "particle_count": int(source["particle_count"]),
                "number_density_m3": float(source["Nv_nm-3"]) * 1.0e27,
                "mean_radius_nm": source["mean_radius_nm"],
                "Sv_nm_inv": source["Sv_nm-1"],
                "M6_nm3": source["M6_nm3"],
                "beta_volume_fraction": source["beta_volume_fraction"],
                "far_field_matrix_xAg": source["matrix_xAg"],
                "far_field_matrix_xB": source["matrix_xB"],
            }
        )
        if len(group) != int(source["particle_count"]):
            raise ValueError(f"Historical qualification PSD count mismatch at {age} h")
        for stable_id, particle in enumerate(group):
            particle_rows.append(
                {
                    "replicate": "replicate_A",
                    "particle_id": f"P{stable_id:03d}",
                    "lineage_member_particle_ids": f"P{stable_id:03d}",
                    "step": step,
                    "elapsed_physical_time_s": float(age - 6.0) * 3600.0,
                    "experimental_age_h": age,
                    "stable_particle_id": stable_id,
                    "equivalent_radius_nm": particle["equivalent_radius_nm"],
                }
            )
    observations_path = audit_dir / "registered_observables.csv"
    particles_path = audit_dir / "registered_particle_psd.csv"
    core.write_csv(observations_path, list(observation_rows[0]), observation_rows)
    core.write_csv(particles_path, list(particle_rows[0]), particle_rows)
    gates = {
        "fixture_schema": True,
        "original_6h_fixture": True,
        "checkpoint_sequence": True,
        "registered_endpoint": True,
        "finite_and_bounds": True,
        "mass_and_zero_mode": True,
        "stderr_empty": True,
        "checkpoint_provenance": True,
        "prohibited_paths": True,
        "merge_aware_particle_lineage": True,
    }
    core.write_json(
        audit_dir / "audit.json",
        {
            "schema": adapter.PRODUCTION_SCHEMA,
            "numerical_status": adapter.PRODUCTION_PASS,
            "replicate": "replicate_A",
            "fixture_manifest_sha256": fixture_sha,
            "checkpoint_hashes": {
                str(checkpoint): _synthetic_digest(f"checkpoint_{checkpoint}")
                for checkpoint in range(44)
            },
            "gates": gates,
            "qualification_only": True,
        },
    )
    (audit_dir / "status.txt").write_text(
        adapter.PRODUCTION_PASS + "\n", encoding="utf-8"
    )
    return {
        "audit_dir": audit_dir,
        "fixture_manifest": fixture_manifest,
        "campaign_manifest": campaign_manifest,
        "input_hashes": input_hashes,
        "analysis_binary_hashes": analysis_hashes,
    }


def _load_authority(
    paths: dict[str, Path], trajectory_class: str = "QUALIFICATION_EXPERIMENT_MATRIX_ANCHORED"
) -> dict[str, Any]:
    return adapter.validate_production_authority(
        production_audit_dir=paths["audit_dir"],
        fixture_manifest=paths["fixture_manifest"],
        campaign_manifest=paths["campaign_manifest"],
        input_hashes=paths["input_hashes"],
        analysis_binary_hashes=paths["analysis_binary_hashes"],
        replicate="A",
        trajectory_class=trajectory_class,
        source_commit="d" * 40,
    )


def _export_once(
    *,
    paths: dict[str, Path],
    output_dir: Path,
    yu_config: Path,
    interface_contract: Path,
) -> tuple[dict[str, Any], dict[str, float]]:
    authority = _load_authority(paths)
    snapshots, box_volume_nm3, closure, timing = adapter.load_production_snapshots(
        authority
    )
    provenance = adapter.build_production_provenance(
        authority=authority,
        trajectory_class="QUALIFICATION_EXPERIMENT_MATRIX_ANCHORED",
        source_commit="d" * 40,
        yu_config=yu_config,
        interface_contract=interface_contract,
        timing=timing,
    )
    config = core.load_json(yu_config)
    manifest = core.export_transport_snapshot(
        output_dir=output_dir,
        snapshots=snapshots,
        box_volume_nm3=box_volume_nm3,
        closure=closure,
        provenance=provenance,
        config=config,
        temperatures_K=(300.0, 450.0, 600.0),
        fixed_matrix_xag=None,
    )
    return manifest, closure


def _expect_blocked(callable_object: Any) -> bool:
    try:
        callable_object()
    except (KeyError, OSError, TypeError, ValueError):
        return True
    return False


def qualify(
    *,
    historical_transport_dir: Path,
    yu_config: Path,
    interface_contract: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite qualification: {output_dir}")
    output_dir.mkdir(parents=True)
    config = core.load_json(yu_config)
    core.validate_yu_base_config(config, yu_config)
    core.validate_interface_contract(
        core.load_json(interface_contract), interface_contract, yu_config
    )
    paths = build_synthetic_production_fixture(
        historical_transport_dir, output_dir / "synthetic_production_fixture"
    )
    manifest_1, closure_1 = _export_once(
        paths=paths,
        output_dir=output_dir / "transport_run_1",
        yu_config=yu_config,
        interface_contract=interface_contract,
    )
    manifest_2, closure_2 = _export_once(
        paths=paths,
        output_dir=output_dir / "transport_run_2",
        yu_config=yu_config,
        interface_contract=interface_contract,
    )
    hashes_1 = {
        path.name: core.file_sha256(path)
        for path in sorted((output_dir / "transport_run_1").iterdir())
        if path.is_file()
    }
    hashes_2 = {
        path.name: core.file_sha256(path)
        for path in sorted((output_dir / "transport_run_2").iterdir())
        if path.is_file()
    }

    bad_class_blocked = _expect_blocked(
        lambda: _load_authority(paths, trajectory_class="NOT_ANCHORED")
    )

    bad_audit_root = output_dir / "bad_audit_fixture"
    shutil.copytree(paths["audit_dir"], bad_audit_root)
    bad_audit = core.load_json(bad_audit_root / "audit.json")
    bad_audit["gates"]["merge_aware_particle_lineage"] = False
    core.write_json(bad_audit_root / "audit.json", bad_audit)
    bad_audit_paths = dict(paths)
    bad_audit_paths["audit_dir"] = bad_audit_root
    bad_merge_blocked = _expect_blocked(lambda: _load_authority(bad_audit_paths))

    incomplete_chain_root = output_dir / "incomplete_checkpoint_chain_fixture"
    shutil.copytree(paths["audit_dir"], incomplete_chain_root)
    incomplete_chain_audit = core.load_json(incomplete_chain_root / "audit.json")
    incomplete_chain_audit["checkpoint_hashes"].pop("43")
    core.write_json(incomplete_chain_root / "audit.json", incomplete_chain_audit)
    incomplete_chain_paths = dict(paths)
    incomplete_chain_paths["audit_dir"] = incomplete_chain_root
    incomplete_checkpoint_chain_blocked = _expect_blocked(
        lambda: _load_authority(incomplete_chain_paths)
    )

    missing_psd_root = output_dir / "missing_psd_fixture"
    shutil.copytree(paths["audit_dir"], missing_psd_root)
    particle_rows = core.read_csv(missing_psd_root / "registered_particle_psd.csv")
    core.write_csv(
        missing_psd_root / "registered_particle_psd.csv",
        list(particle_rows[0]),
        particle_rows[:-1],
    )
    missing_psd_paths = dict(paths)
    missing_psd_paths["audit_dir"] = missing_psd_root
    missing_psd_blocked = _expect_blocked(
        lambda: adapter.load_production_snapshots(_load_authority(missing_psd_paths))
    )

    bad_fixture = output_dir / "bad_fixture_manifest.json"
    bad_fixture.write_text("{}\n", encoding="utf-8")
    bad_fixture_paths = dict(paths)
    bad_fixture_paths["fixture_manifest"] = bad_fixture
    bad_fixture_blocked = _expect_blocked(lambda: _load_authority(bad_fixture_paths))

    source_timing = manifest_1["provenance"]["registered_snapshot_timing"]
    gates = {
        "frozen_transport_contract": (
            manifest_1["transport_contract"]["A_N"] == 1.5
            and manifest_1["transport_contract"]["S11_rate"] == 0.0
            and manifest_1["transport_contract"]["S13_rate"] == 0.0
            and manifest_1["transport_contract"]["yu_refit_scale_used"] is False
        ),
        "production_schema_positive_path": (
            manifest_1["status"] == core.FINAL_STATUS
            and manifest_1["snapshot_contract"]["count"] == 6
            and manifest_1["snapshot_contract"]["first_age_h"] == 6.0
            and manifest_1["snapshot_contract"]["last_age_h"] == 48.0
        ),
        "raw_descriptor_closure": (
            closure_1 == closure_2
            and max(closure_1.values()) <= 5.0e-12
        ),
        "source_timing_preserved": (
            [item["registered_age_h"] for item in source_timing]
            == list(adapter.REGISTERED_STEP_TO_AGE_H.values())
            and len(source_timing) == 6
        ),
        "deterministic_transport_outputs": hashes_1 == hashes_2,
        "non_experiment_matrix_class_fail_closed": bad_class_blocked,
        "merge_aware_failure_fail_closed": bad_merge_blocked,
        "incomplete_checkpoint_chain_fail_closed": (
            incomplete_checkpoint_chain_blocked
        ),
        "incomplete_psd_fail_closed": missing_psd_blocked,
        "fixture_hash_mismatch_fail_closed": bad_fixture_blocked,
    }
    passed = all(gates.values())
    result = {
        "schema": "PF_246CUBE_NO_DISLOCATION_TRANSPORT_AUTHORITY_ADAPTER_QUALIFICATION_V1",
        "status": STATUS if passed else BLOCKED,
        "all_gates_pass": passed,
        "scientific_claim": (
            "production-ingress and transport-interface qualification only; "
            "not absolute experimental lattice-conductivity reproduction"
        ),
        "production_A_B_C_data_used": False,
        "running_PF_jobs_modified": False,
        "gates": gates,
        "metrics": {
            "descriptor_closure_max_relative": closure_1,
            "determinism_run_1_hashes": hashes_1,
            "determinism_run_2_hashes": hashes_2,
        },
        "provenance": {
            "qualification_script_sha256": core.file_sha256(Path(__file__)),
            "production_adapter_script_sha256": core.file_sha256(
                Path(adapter.__file__)
            ),
            "transport_interface_script_sha256": core.file_sha256(Path(core.__file__)),
            "historical_smoke_manifest_sha256": core.file_sha256(
                historical_transport_dir / "transport_snapshot_manifest.json"
            ),
            "yu_public_parameter_sha256": core.file_sha256(yu_config),
            "transport_parameter_contract_sha256": core.file_sha256(
                interface_contract
            ),
        },
    }
    core.write_json(output_dir / "qualification.json", result)
    gate_rows = [
        {"gate": name, "status": "PASS" if value else "FAIL"}
        for name, value in gates.items()
    ]
    core.write_csv(
        output_dir / "qualification_gates.csv", ("gate", "status"), gate_rows
    )
    report = [
        "# 246³ production PASS to no-dislocation transport qualification",
        "",
        f"Final status: `{result['status']}`",
        "",
        (
            "This is an interface qualification using a synthetic production-schema "
            "fixture derived from the already-qualified historical smoke. No A/B/C "
            "production result is consumed, no running PF job is changed, and no "
            "absolute experimental thermal-conductivity reproduction is claimed."
        ),
        "",
        "| Gate | Status |",
        "|---|---|",
    ]
    report.extend(f"| {row['gate']} | {row['status']} |" for row in gate_rows)
    report.extend(
        [
            "",
            (
                "The adapter accepts only one exact complete production PASS, the "
                "frozen six registered ages, an all-true merge-aware gate set, "
                "experiment-matrix anchoring, descriptor closure, and complete "
                "source/binary/parameter/fixture/analysis hashes."
            ),
        ]
    )
    (output_dir / "qualification_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-transport-dir", type=Path, required=True)
    parser.add_argument(
        "--yu-config",
        type=Path,
        default=Path("data/qualification/yu2024_transport_v1/yu_48h_parameters.json"),
    )
    parser.add_argument(
        "--interface-contract", type=Path, default=core.DEFAULT_INTERFACE_CONTRACT
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = qualify(
        historical_transport_dir=args.historical_transport_dir,
        yu_config=args.yu_config,
        interface_contract=args.interface_contract,
        output_dir=args.output_dir,
    )
    print(result["status"])
    if result["status"] != STATUS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
