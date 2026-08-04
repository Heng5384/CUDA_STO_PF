#!/usr/bin/env python3
"""Qualification suite for PF full-PSD no-dislocation transport V1."""

from __future__ import annotations

import argparse
import copy
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

import assemble_pf_full_psd_no_dislocation_transport_ensemble_v1 as ensemble
import pf_full_psd_no_dislocation_transport_v1 as core


STATUS = "PASS_PF_FULL_PSD_NO_DISLOCATION_TRANSPORT_INTERFACE_V1"


def max_relative_array(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.maximum(np.abs(b), np.finfo(float).tiny)
    return float(np.max(np.abs(a - b) / denominator))


def qualify(
    *,
    yu_config_path: Path,
    interface_contract_path: Path,
    microstructure_csv: Path,
    particle_psd_csv: Path,
    historical_output_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = core.load_json(yu_config_path)
    core.validate_yu_base_config(config, yu_config_path)
    core.validate_interface_contract(
        core.load_json(interface_contract_path),
        interface_contract_path,
        yu_config_path,
    )
    gates: dict[str, dict[str, Any]] = {}

    omega = np.geomspace(1.0e9, 3.0e13, 257)
    base = core.base_scattering_rates(omega, 300.0, 0.0062, config)
    no_dis_max = float(
        max(np.max(np.abs(base["dislocation_core"])), np.max(np.abs(base["dislocation_strain"])))
    )
    gates["frozen_contract"] = {
        "pass": bool(
            float(config["shared_parameters"]["A_N"]) == core.FROZEN_A_N
            and no_dis_max == 0.0
        ),
        "A_N": float(config["shared_parameters"]["A_N"]),
        "S11_S13_max_rate_s-1": no_dis_max,
        "refit_scale_used": False,
    }

    mono_radius = 30.0e-9
    mono_nv = 1.9e21
    mono_count = 64
    mono_volume = mono_count / mono_nv
    mono_radii = np.full(mono_count, mono_radius)
    full_rate = core.full_psd_precipitate_rate(
        omega, mono_radii, mono_volume, config
    )
    sigma = core.precipitate_cross_section(
        omega, np.array([mono_radius]), config
    )[:, 0]
    single_radius_rate = (
        float(config["shared_parameters"]["average_sound_velocity_m_s"])
        * mono_nv
        * sigma
    )
    mono_rate_relative = max_relative_array(full_rate, single_radius_rate)
    mono_full_kappa = core.integrate_kappa_gauss(
        300.0, 0.0062, mono_radii, mono_volume, config, "full_psd"
    )
    mono_nv_r_kappa = core.integrate_kappa_gauss(
        300.0,
        0.0062,
        mono_radii,
        mono_volume,
        config,
        "Nv_plus_mean_R_monodisperse",
    )
    mono_moment_kappa = core.integrate_kappa_gauss(
        300.0,
        0.0062,
        mono_radii,
        mono_volume,
        config,
        "Sv_plus_M6_moment_reconstruction",
    )
    mono_kappa_relative = max(
        core.relative_error(mono_nv_r_kappa, mono_full_kappa),
        core.relative_error(mono_moment_kappa, mono_full_kappa),
    )
    gates["monodisperse_degeneracy"] = {
        "pass": mono_rate_relative <= 2.0e-14 and mono_kappa_relative <= 2.0e-13,
        "max_rate_relative_difference": mono_rate_relative,
        "max_kappa_relative_difference": mono_kappa_relative,
    }

    test_radii = np.array([2.0, 3.5, 5.0, 8.0, 13.0, 21.0, 34.0, 50.0]) * 1.0e-9
    test_volume = 2.0e-20
    vectorized = core.full_psd_precipitate_rate(
        omega, test_radii, test_volume, config
    )
    scalar = core.direct_scalar_sum_precipitate_rate(
        omega, test_radii, test_volume, config
    )
    direct_relative = max_relative_array(vectorized, scalar)
    gates["direct_particle_sum"] = {
        "pass": direct_relative <= 5.0e-15,
        "max_relative_difference": direct_relative,
    }

    broad_radii = np.geomspace(2.0e-9, 50.0e-9, 4096)
    broad_volume = 1.0e-18
    direct_kappa = core.integrate_kappa_gauss(
        300.0, 0.0062, broad_radii, broad_volume, config, "full_psd"
    )
    widths_nm = (8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125)
    bin_rows = []
    for width_nm in widths_nm:
        binned = core.binned_center_radii(broad_radii, width_nm * 1.0e-9)
        value = core.integrate_kappa_gauss(
            300.0, 0.0062, binned, broad_volume, config, "full_psd"
        )
        bin_rows.append(
            {
                "bin_width_nm": width_nm,
                "kappa_W_mK": value,
                "relative_error_vs_direct": core.relative_error(value, direct_kappa),
            }
        )
    gates["psd_bin_refinement"] = {
        "pass": bool(
            bin_rows[-1]["relative_error_vs_direct"]
            < bin_rows[0]["relative_error_vs_direct"]
            and bin_rows[-1]["relative_error_vs_direct"] <= 5.0e-6
        ),
        "direct_kappa_W_mK": direct_kappa,
        "coarsest_relative_error": bin_rows[0]["relative_error_vs_direct"],
        "finest_relative_error": bin_rows[-1]["relative_error_vs_direct"],
        "rows": bin_rows,
    }

    broad_moments = core.moments_from_radii(broad_radii, broad_volume)
    reconstructed_radius = (
        4.0 * math.pi * broad_moments["M6_m3"] / broad_moments["Sv_m-1"]
    ) ** 0.25
    reconstructed_nv = broad_moments["Sv_m-1"] / (
        4.0 * math.pi * reconstructed_radius**2
    )
    reconstructed_sv = 4.0 * math.pi * reconstructed_nv * reconstructed_radius**2
    reconstructed_m6 = reconstructed_nv * reconstructed_radius**6
    moment_relative = max(
        core.relative_error(reconstructed_sv, broad_moments["Sv_m-1"]),
        core.relative_error(reconstructed_m6, broad_moments["M6_m3"]),
    )
    gates["moment_reconstruction"] = {
        "pass": moment_relative <= 5.0e-15,
        "max_Sv_M6_relative_closure": moment_relative,
        "equivalent_radius_nm": reconstructed_radius * 1.0e9,
        "equivalent_number_density_m-3": reconstructed_nv,
    }

    snapshots, box_volume_nm3, source_closure = core.load_transport_snapshots(
        microstructure_csv,
        particle_psd_csv,
        "historical_elastic_72848",
        "matrix_xAg",
        "matrix_xB",
        "historical h(phi)<0.005 matrix observation; smoke only",
    )
    convergence_rows = []
    for snapshot in (snapshots[0], snapshots[-1]):
        radii = np.asarray(snapshot["radii_nm"]) * 1.0e-9
        for temperature in (300.0, 450.0, 600.0):
            kwargs = (
                temperature,
                float(snapshot["matrix_xAg"]),
                radii,
                box_volume_nm3 * 1.0e-27,
                config,
                "full_psd",
            )
            gauss_256 = core.integrate_kappa_gauss(*kwargs, order=256)
            gauss_512 = core.integrate_kappa_gauss(*kwargs, order=512)
            adaptive, adaptive_error = core.integrate_kappa_adaptive(*kwargs)
            convergence_rows.append(
                {
                    "age_h": snapshot["age_h"],
                    "temperature_K": temperature,
                    "gauss_256_W_mK": gauss_256,
                    "gauss_512_W_mK": gauss_512,
                    "adaptive_W_mK": adaptive,
                    "adaptive_reported_error": adaptive_error,
                    "gauss_refinement_relative": core.relative_error(
                        gauss_256, gauss_512
                    ),
                    "gauss_vs_adaptive_relative": core.relative_error(
                        gauss_256, adaptive
                    ),
                }
            )
    max_refinement = max(row["gauss_refinement_relative"] for row in convergence_rows)
    max_adaptive = max(row["gauss_vs_adaptive_relative"] for row in convergence_rows)
    gates["debye_integration"] = {
        "pass": max_refinement <= 1.0e-11 and max_adaptive <= 1.0e-11,
        "max_256_512_relative_difference": max_refinement,
        "max_gauss_adaptive_relative_difference": max_adaptive,
        "rows": convergence_rows,
    }

    historical_manifest = core.load_json(
        historical_output_dir / "transport_snapshot_manifest.json"
    )
    historical_rows = core.read_csv(
        historical_output_dir / "kappa_time_temperature.csv"
    )
    first_rows = [row for row in historical_rows if float(row["age_h"]) == 6.0]
    initial_mode_difference = max(
        abs(
            float(row["kappa_L_no_dis_fixed_matrix_W_mK"])
            - float(row["kappa_L_no_dis_time_varying_matrix_W_mK"])
        )
        for row in first_rows
    )
    historical_pass = bool(
        historical_manifest["status"] == STATUS
        and historical_manifest["snapshot_contract"]["count"] == 43
        and historical_manifest["snapshot_contract"]["first_age_h"] == 6.0
        and historical_manifest["snapshot_contract"]["last_age_h"] == 48.0
        and historical_manifest["transport_contract"]["A_N"] == 1.5
        and historical_manifest["transport_contract"]["S11_rate"] == 0.0
        and historical_manifest["transport_contract"]["S13_rate"] == 0.0
        and not historical_manifest["transport_contract"]["yu_refit_scale_used"]
        and initial_mode_difference <= 1.0e-14
    )
    gates["historical_6h48h_interface_smoke"] = {
        "pass": historical_pass,
        "snapshot_count": historical_manifest["snapshot_contract"]["count"],
        "first_age_h": historical_manifest["snapshot_contract"]["first_age_h"],
        "last_age_h": historical_manifest["snapshot_contract"]["last_age_h"],
        "source_descriptor_closure_max_relative": source_closure,
        "initial_fixed_vs_coupled_absolute_difference_W_mK": initial_mode_difference,
        "trajectory_class": historical_manifest["provenance"]["trajectory_class"],
    }

    deterministic_dirs = (
        output_dir / "determinism_run_1",
        output_dir / "determinism_run_2",
    )
    subset = [snapshots[0], snapshots[-1]]
    deterministic_provenance = {
        "qualification_fixture": "historical first/last snapshot subset",
        "source_microstructure_sha256": core.file_sha256(microstructure_csv),
        "source_particles_sha256": core.file_sha256(particle_psd_csv),
        "yu_public_parameter_sha256": core.file_sha256(yu_config_path),
    }
    deterministic_hashes = []
    for directory in deterministic_dirs:
        core.export_transport_snapshot(
            output_dir=directory,
            snapshots=subset,
            box_volume_nm3=box_volume_nm3,
            closure=source_closure,
            provenance=deterministic_provenance,
            config=config,
            temperatures_K=(300.0, 450.0, 600.0),
            fixed_matrix_xag=None,
        )
        deterministic_hashes.append(
            {
                path.name: core.file_sha256(path)
                for path in sorted(directory.iterdir())
                if path.is_file()
            }
        )
    gates["determinism"] = {
        "pass": deterministic_hashes[0] == deterministic_hashes[1],
        "run_1_hashes": deterministic_hashes[0],
        "run_2_hashes": deterministic_hashes[1],
    }

    science_ages = set(ensemble.REQUIRED_SCIENCE_AGES_H)
    science_snapshots = [
        snapshot for snapshot in snapshots if float(snapshot["age_h"]) in science_ages
    ]
    synthetic_root = output_dir / "synthetic_abc_authority"
    authority_entries = []
    for index, replicate in enumerate(ensemble.REQUIRED_REPLICATES):
        replicate_snapshots = []
        for source in science_snapshots:
            item = dict(source)
            item["replicate"] = replicate
            item["snapshot_id"] = f"{replicate}_step_{int(item['step']):06d}"
            replicate_snapshots.append(item)
        token = core.canonical_sha256({"replicate": replicate, "index": index})
        provenance = {
            "trajectory_class": "QUALIFICATION_SYNTHETIC_EXPERIMENT_MATRIX_ANCHORED",
            "replicate": replicate,
            "source_commit": token[:40],
            "source_binary_sha256": core.canonical_sha256([token, "binary"]),
            "source_parameter_sha256": core.canonical_sha256([token, "parameter"]),
            "analysis_binary_sha256": core.canonical_sha256([token, "analysis_binary"]),
            "fixture_manifest_sha256": core.canonical_sha256([token, "fixture"]),
            "microstructure_csv_sha256": core.canonical_sha256(
                [token, "microstructure"]
            ),
            "particle_psd_csv_sha256": core.canonical_sha256([token, "particles"]),
            "upstream_audit_json_sha256": core.canonical_sha256([token, "audit"]),
            "campaign_manifest_sha256": core.canonical_sha256([token, "campaign"]),
            "input_hash_ledger_sha256": core.canonical_sha256(
                [token, "input_hashes"]
            ),
            "upstream_analysis_manifest_sha256": core.canonical_sha256(
                [token, "analysis_manifest"]
            ),
            "production_adapter_script_sha256": core.file_sha256(
                ensemble.PRODUCTION_ADAPTER
            ),
            "production_audit_status": ensemble.PRODUCTION_PASS,
            "production_audit_gates": {
                "fixture_schema": True,
                "checkpoint_sequence": True,
                "registered_endpoint": True,
                "mass_and_zero_mode": True,
                "checkpoint_provenance": True,
                "prohibited_paths": True,
                "merge_aware_particle_lineage": True,
            },
            "production_checkpoint_hashes": {
                str(checkpoint): core.canonical_sha256(
                    [token, "checkpoint", checkpoint]
                )
                for checkpoint in range(44)
            },
        }
        replicate_dir = synthetic_root / replicate
        core.export_transport_snapshot(
            output_dir=replicate_dir,
            snapshots=replicate_snapshots,
            box_volume_nm3=box_volume_nm3,
            closure=source_closure,
            provenance=provenance,
            config=config,
            temperatures_K=(300.0, 450.0, 600.0),
            fixed_matrix_xag=None,
        )
        manifest_path = replicate_dir / "transport_snapshot_manifest.json"
        authority_entries.append(
            {
                "replicate": replicate,
                "candidates": [
                    {
                        "candidate_id": f"QUALIFICATION_{replicate}",
                        "selected": True,
                        "authority_status": "PASS_COMPLETE_6H48H",
                        "complete_6h48h": True,
                        "transport_output_dir": str(replicate_dir.resolve()),
                        "transport_manifest_sha256": core.file_sha256(manifest_path),
                    }
                ],
            }
        )
    authority_path = synthetic_root / "authority_manifest.json"
    core.write_json(
        authority_path,
        {"schema": ensemble.AUTHORITY_SCHEMA, "replicates": authority_entries},
    )
    loaded_authorities = ensemble.load_authorities(authority_path)
    ensemble_result = ensemble.assemble_ensemble(
        loaded_authorities, synthetic_root / "ensemble"
    )
    zero_selected_blocked = False
    two_selected_blocked = False
    try:
        ensemble._select_candidate({"replicate": "A", "candidates": []})
    except ValueError:
        zero_selected_blocked = True
    try:
        ensemble._select_candidate(
            {
                "replicate": "A",
                "candidates": [{"selected": True}, {"selected": True}],
            }
        )
    except ValueError:
        two_selected_blocked = True
    missing_production_pass_blocked = False
    missing_descriptor_cell_blocked = False

    def negative_authority(
        *, candidate_output_dir: Path, manifest_sha256: str, filename: str
    ) -> Path:
        negative_entries = copy.deepcopy(authority_entries)
        negative_entries[0]["candidates"][0]["transport_output_dir"] = str(
            candidate_output_dir.resolve()
        )
        negative_entries[0]["candidates"][0]["transport_manifest_sha256"] = (
            manifest_sha256
        )
        path = synthetic_root / filename
        core.write_json(
            path,
            {"schema": ensemble.AUTHORITY_SCHEMA, "replicates": negative_entries},
        )
        return path

    bad_provenance_dir = synthetic_root / "negative_missing_production_pass"
    shutil.copytree(synthetic_root / "A", bad_provenance_dir)
    bad_provenance_manifest_path = (
        bad_provenance_dir / "transport_snapshot_manifest.json"
    )
    bad_provenance_manifest = core.load_json(bad_provenance_manifest_path)
    bad_provenance_manifest["provenance"].pop("production_audit_status")
    core.write_json(bad_provenance_manifest_path, bad_provenance_manifest)
    bad_provenance_authority = negative_authority(
        candidate_output_dir=bad_provenance_dir,
        manifest_sha256=core.file_sha256(bad_provenance_manifest_path),
        filename="negative_missing_production_pass_authority.json",
    )
    try:
        ensemble.load_authorities(bad_provenance_authority)
    except ValueError:
        missing_production_pass_blocked = True

    bad_grid_dir = synthetic_root / "negative_missing_descriptor_cell"
    shutil.copytree(synthetic_root / "A", bad_grid_dir)
    descriptor_path = bad_grid_dir / "descriptor_predictions.csv"
    descriptor_rows = core.read_csv(descriptor_path)
    core.write_csv(
        descriptor_path,
        list(descriptor_rows[0]),
        descriptor_rows[:-1],
    )
    bad_grid_manifest_path = bad_grid_dir / "transport_snapshot_manifest.json"
    bad_grid_manifest = core.load_json(bad_grid_manifest_path)
    bad_grid_manifest["outputs"]["descriptor_predictions.csv"] = core.file_sha256(
        descriptor_path
    )
    core.write_json(bad_grid_manifest_path, bad_grid_manifest)
    bad_grid_authority = negative_authority(
        candidate_output_dir=bad_grid_dir,
        manifest_sha256=core.file_sha256(bad_grid_manifest_path),
        filename="negative_missing_descriptor_cell_authority.json",
    )
    try:
        ensemble.load_authorities(bad_grid_authority)
    except ValueError:
        missing_descriptor_cell_blocked = True
    gates["abc_authority_selector"] = {
        "pass": bool(
            ensemble_result["status"] == ensemble.ENSEMBLE_STATUS
            and len(loaded_authorities) == 3
            and zero_selected_blocked
            and two_selected_blocked
            and missing_production_pass_blocked
            and missing_descriptor_cell_blocked
        ),
        "positive_synthetic_status": ensemble_result["status"],
        "selected_replicates": [item["replicate"] for item in loaded_authorities],
        "zero_selected_fail_closed": zero_selected_blocked,
        "two_selected_fail_closed": two_selected_blocked,
        "missing_production_pass_fail_closed": missing_production_pass_blocked,
        "missing_descriptor_cell_fail_closed": missing_descriptor_cell_blocked,
        "production_A_B_C_data_used": False,
    }

    all_pass = all(bool(item["pass"]) for item in gates.values())
    status = STATUS if all_pass else "BLOCKED_PF_FULL_PSD_NO_DISLOCATION_TRANSPORT_QUALIFICATION"
    result = {
        "schema": "PF_FULL_PSD_NO_DISLOCATION_TRANSPORT_QUALIFICATION_V1",
        "status": status,
        "all_gates_pass": all_pass,
        "scientific_claim": (
            "interface and numerical qualification only; no absolute experimental "
            "lattice-conductivity reproduction claim"
        ),
        "gates": gates,
        "provenance": {
            "qualification_script_sha256": core.file_sha256(Path(__file__)),
            "transport_interface_script_sha256": core.file_sha256(
                Path(core.__file__)
            ),
            "yu_public_parameter_sha256": core.file_sha256(yu_config_path),
            "transport_parameter_contract_sha256": core.file_sha256(
                interface_contract_path
            ),
            "historical_transport_manifest_sha256": core.file_sha256(
                historical_output_dir / "transport_snapshot_manifest.json"
            ),
        },
    }
    core.write_json(output_dir / "qualification.json", result)
    gate_rows = [
        {"gate": name, "status": "PASS" if value["pass"] else "FAIL"}
        for name, value in gates.items()
    ]
    core.write_csv(
        output_dir / "qualification_gates.csv", ("gate", "status"), gate_rows
    )
    report_lines = [
        "# PF full-PSD no-dislocation transport V1 qualification",
        "",
        f"Final status: `{status}`",
        "",
        (
            "This qualifies a conditional resolved-PF-PSD transport interface. It "
            "does not claim absolute experimental lattice-thermal-conductivity "
            "reproduction."
        ),
        "",
        "| Gate | Status |",
        "|---|---|",
    ]
    report_lines.extend(
        f"| {row['gate']} | {row['status']} |" for row in gate_rows
    )
    report_lines.extend(
        [
            "",
            "Frozen physics: `A_N=1.5`, `S11=0`, `S13=0`, no Yu refit scale.",
            "",
            (
                "The historical 6--48 h trajectory is used only as an interface "
                "smoke. Its overall PSD observations are admissible; no post-merge "
                "independent-particle lineage claim is made."
            ),
            "",
            (
                "A/B/C ensemble production remains pending until exactly one "
                "complete PASS authority is selected for each "
                "experiment-matrix-anchored replicate."
            ),
        ]
    )
    (output_dir / "qualification_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yu-config",
        type=Path,
        default=Path("data/qualification/yu2024_transport_v1/yu_48h_parameters.json"),
    )
    parser.add_argument(
        "--interface-contract", type=Path, default=core.DEFAULT_INTERFACE_CONTRACT
    )
    parser.add_argument("--microstructure-csv", type=Path, required=True)
    parser.add_argument("--particle-psd-csv", type=Path, required=True)
    parser.add_argument("--historical-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = qualify(
        yu_config_path=args.yu_config,
        interface_contract_path=args.interface_contract,
        microstructure_csv=args.microstructure_csv,
        particle_psd_csv=args.particle_psd_csv,
        historical_output_dir=args.historical_output_dir,
        output_dir=args.output_dir,
    )
    if result["status"] != STATUS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
