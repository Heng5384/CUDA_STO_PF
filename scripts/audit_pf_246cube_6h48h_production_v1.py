#!/usr/bin/env python3
"""Fail-closed audit for one 246^3 conditional 6--48 h production path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List


PASS = "PASS_246CUBE_6H48H_CONDITIONAL_PRODUCTION_V1"
FAIL = "BLOCKED_246CUBE_6H48H_CONDITIONAL_PRODUCTION_V1"
MERGE_PASS = "PASS_246CUBE_RESOLVED_MERGE_AWARE_LINEAGE_V1"
TRACKER_PASS = "PASS_PERIODIC_OVERLAP_PARTICLE_LINEAGE_V1"
FINAL_STEP = 152585
PHYSICAL_DT_S = 0.9909260953431841
CHECKPOINT_CADENCE = 3633
REGISTERED_STEPS = [0, 21798, 43596, 65393, 108989, 152585]
EXPECTED_ENDPOINTS = sorted(
    set(range(CHECKPOINT_CADENCE, FINAL_STEP, CHECKPOINT_CADENCE))
    | set(REGISTERED_STEPS[1:])
)
EXPERIMENT_XAG_LOW = 0.0058
EXPERIMENT_XAG_HIGH = 0.0066
DEFAULT_LIBRARY_SHA256 = (
    "58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def key_values(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = raw.partition("=")
        if separator:
            result[key] = value
    return result


def final_row(path: Path) -> Dict[str, str]:
    data = rows(path)
    if not data:
        raise ValueError(f"empty CSV: {path}")
    return data[-1]


def wall_seconds(text: str) -> float:
    hits = re.findall(r"wall_time_s\s*:\s*([0-9.eE+-]+)", text)
    if not hits:
        raise ValueError("wall time not found")
    return float(hits[-1])


def gpu_metrics(path: Path) -> Dict[str, float]:
    utilization: List[float] = []
    memory: List[float] = []
    if path.is_file():
        for raw in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            fields = [value.strip() for value in raw.split(",")]
            if len(fields) < 2:
                continue
            try:
                utilization.append(float(fields[-2]))
                memory.append(float(fields[-1]))
            except ValueError:
                continue
    return {
        "gpu_sample_count": len(memory),
        "gpu_utilization_mean_percent": (
            sum(utilization) / len(utilization)
            if utilization
            else math.nan
        ),
        "peak_VRAM_MiB": max(memory) if memory else math.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--library-sha256", default=DEFAULT_LIBRARY_SHA256)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--lineage-root", type=Path, required=True)
    parser.add_argument("--merge-aware-audit", type=Path, required=True)
    parser.add_argument("--gpu-samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)
    try:
        manifest = json.loads(
            args.fixture_manifest.read_text(encoding="utf-8")
        )
        fixture_sha = sha256(args.fixture_manifest)
        physical = manifest["physical_contract"]
        merge = json.loads(
            args.merge_aware_audit.read_text(encoding="utf-8")
        )
        if (
            merge.get("status") != MERGE_PASS
            or not all(merge.get("gates", {}).values())
        ):
            raise ValueError("merge-aware audit is not exact PASS")

        summary = key_values(args.lineage_root / "lineage_summary.txt")
        observations = rows(
            args.lineage_root / "ensemble_observables.csv"
        )
        observed_steps = [int(row["step"]) for row in observations]
        expected_steps = [0] + EXPECTED_ENDPOINTS
        if observed_steps != expected_steps:
            raise ValueError("checkpoint/snapshot sequence mismatch")

        stderr_empty = True
        zero_mode_ok = True
        provenance_ok = True
        prohibited_ok = True
        total_wall = 0.0
        max_lambda = 0.0
        elastic_by_step: Dict[int, Dict[str, str]] = {}
        checkpoint_hashes: Dict[str, str] = {}
        required_tokens = (
            "initial_state_class="
            "MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1",
            f"fixture_manifest_sha256={fixture_sha}",
            "profile_library_manifest_sha256="
            f"{args.library_sha256}",
            "pf_sm_explicit_context=SM_EXPLICIT_CONTEXT_N_V1",
            "pf_reaction_discretization=SM_TANGENT_N_V1",
            "gp_enabled=false",
        )
        for index, step in enumerate(EXPECTED_ENDPOINTS):
            segment = args.run_root / "segments" / f"step_{step}"
            checkpoint = args.run_root / "checkpoints" / f"step_{step}.chk"
            if (
                (segment / "status.txt").read_text(encoding="utf-8").strip()
                != "PASS_246CUBE_6H48H_SEGMENT_V1"
            ):
                raise ValueError(f"segment status is not PASS: {step}")
            checkpoint_hashes[str(step)] = sha256(checkpoint)
            stdout = (segment / "stdout.log").read_text(
                encoding="utf-8", errors="replace"
            )
            stderr_empty = (
                stderr_empty and not (segment / "stderr.log").read_bytes()
            )
            total_wall += wall_seconds(stdout)
            zero_mode_ok = (
                zero_mode_ok
                and "PF_ZERO_MODE_FINAL_AUDIT status=PASS" in stdout
            )
            provenance_ok = provenance_ok and all(
                token in stdout for token in required_tokens
            )
            if index:
                provenance_ok = (
                    provenance_ok
                    and "checkpoint_restart_provenance="
                    "RESTORED_AND_VALIDATED" in stdout
                )
            prohibited_ok = prohibited_ok and not re.search(
                r"GP_EVENT|GP_BIRTH|BETA_NUCLEATION_EVENT|source_event",
                stdout,
            )
            lambda_hits = re.findall(
                r"PF_ZERO_MODE_FINAL_AUDIT status=PASS .*?"
                r"last_lambda=([0-9.eE+-]+)",
                stdout,
            )
            if not lambda_hits:
                raise ValueError(f"missing zero-mode lambda: {step}")
            max_lambda = max(max_lambda, abs(float(lambda_hits[-1])))
            mass = final_row(
                segment / "dynamics_mass_diagnostics.csv"
            )
            if int(float(mass["step"])) != step:
                raise ValueError(f"mass endpoint mismatch: {step}")
            elastic_by_step[step] = mass

        initial_inventory = float(
            observations[0]["canonical_inventory_code"]
        )
        max_mass_drift = max(
            abs(float(row["canonical_inventory_code"]) - initial_inventory)
            / max(abs(initial_inventory), 1.0)
            for row in observations
        )
        finite_bounds = all(
            row["finite"] == "1" and row["bounds"] == "1"
            for row in observations
        )
        final_age = float(observations[-1]["experimental_age_h"])
        expected_age = 6.0 + FINAL_STEP * PHYSICAL_DT_S / 3600.0
        lineage_ok = (
            summary.get("status") == TRACKER_PASS
            and int(summary.get("initial_particle_count", -1)) == 96
            and int(summary.get("split_count", -1)) == 0
            and int(summary.get("new_component_count", -1)) == 0
            and int(summary.get("unqualified_dissolution_count", -1)) == 0
            and int(summary.get("unqualified_merge_count", -1)) == 0
            and int(summary.get("qualified_merge_count", -1))
            == int(summary.get("merge_count", -2))
        )
        paths_off = all(
            physical.get(key) is False
            for key in (
                "GP_enabled",
                "GP_birth_enabled",
                "GP_release_enabled",
                "external_source_enabled",
                "new_beta_nucleation_enabled",
            )
        )
        gates = {
            "fixture_schema": (
                manifest.get("schema")
                == "PF_246CUBE_LIBRARY_HANDOFF_MANIFEST_V1"
            ),
            "original_6h_fixture": observed_steps[0] == 0,
            "checkpoint_sequence": observed_steps == expected_steps,
            "registered_endpoint": (
                int(observations[-1]["step"]) == FINAL_STEP
                and abs(final_age - expected_age) <= 1.0e-12
            ),
            "finite_and_bounds": finite_bounds,
            "mass_and_zero_mode": (
                max_mass_drift <= 1.0e-10 and zero_mode_ok
            ),
            "stderr_empty": stderr_empty,
            "checkpoint_provenance": provenance_ok,
            "prohibited_paths": prohibited_ok and paths_off,
            "merge_aware_particle_lineage": lineage_ok,
        }
        if not all(gates.values()):
            failed = [key for key, value in gates.items() if not value]
            raise ValueError(f"failed production gates: {failed}")

        registered = {
            int(row["step"]): dict(row)
            for row in observations
            if int(row["step"]) in REGISTERED_STEPS
        }
        enriched: List[Dict[str, Any]] = []
        box_volume_m3 = (246.0e-9) ** 3
        for step in REGISTERED_STEPS:
            row: Dict[str, Any] = registered[step]
            row["replicate"] = manifest["replicate_id"]
            row["number_density_m3"] = (
                int(row["particle_count"]) / box_volume_m3
            )
            mass = elastic_by_step.get(step)
            for name in (
                "mean_elastic_energy",
                "max_elastic_energy",
                "stress_hydro_min",
                "stress_hydro_max",
            ):
                row[name] = float(mass[name]) if mass else ""
            enriched.append(row)
        fieldnames = ["replicate"] + [
            name for name in enriched[0] if name != "replicate"
        ]
        with (args.out / "registered_observables.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(enriched)

        lineage = rows(args.lineage_root / "particle_lineage.csv")
        registered_lineage = [
            {"replicate": manifest["replicate_id"], **row}
            for row in lineage
            if int(row["step"]) in REGISTERED_STEPS
        ]
        with (args.out / "registered_particle_psd.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            fields = list(registered_lineage[0])
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(registered_lineage)

        first = enriched[0]
        final = enriched[-1]
        coarsening = (
            int(final["particle_count"]) < int(first["particle_count"])
            and float(final["mean_radius_nm"])
            > float(first["mean_radius_nm"])
        )
        final_xag = float(final["far_field_matrix_xAg"])
        concentration_in_band = (
            EXPERIMENT_XAG_LOW <= final_xag <= EXPERIMENT_XAG_HIGH
        )
        if coarsening and concentration_in_band:
            scientific_status = "EXPERIMENT_ALLOWED_CONDITIONAL_PATH"
        elif coarsening:
            scientific_status = (
                "SCIENTIFICALLY_EXCLUDED_TOTAL_INVENTORY_PSD_FAMILY"
            )
        else:
            scientific_status = "CONDITIONAL_PATH_WITHOUT_COARSENING_TREND"

        audit = {
            "schema": "PF_246CUBE_6H48H_PRODUCTION_AUDIT_V1",
            "numerical_status": PASS,
            "scientific_status": scientific_status,
            "replicate": manifest["replicate_id"],
            "fixture_manifest_sha256": fixture_sha,
            "checkpoint_hashes": checkpoint_hashes,
            "gates": gates,
            "metrics": {
                "initial_particle_count": int(first["particle_count"]),
                "final_particle_count": int(final["particle_count"]),
                "initial_mean_radius_nm": float(first["mean_radius_nm"]),
                "final_mean_radius_nm": float(final["mean_radius_nm"]),
                "final_far_field_matrix_xAg": final_xag,
                "coarsening_direction": coarsening,
                "final_concentration_in_experimental_band": (
                    concentration_in_band
                ),
                "dissolution_count": int(summary["dissolution_count"]),
                "merge_count": int(summary["merge_count"]),
                "split_count": 0,
                "max_system_mass_drift": max_mass_drift,
                "max_zero_mode_lambda": max_lambda,
                "wall_seconds": total_wall,
                "wall_seconds_per_step": total_wall / FINAL_STEP,
                "actual_endpoint_age_h": final_age,
                **gpu_metrics(args.gpu_samples),
            },
        }
        (args.out / "audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.out / "status.txt").write_text(PASS + "\n", encoding="utf-8")
        (args.out / "final_terminal_output.txt").write_text(
            "\n".join(
                (
                    f"numerical_status={PASS}",
                    f"scientific_status={scientific_status}",
                    f"replicate={manifest['replicate_id']}",
                    f"final_step={FINAL_STEP}",
                    f"actual_endpoint_age_h={final_age}",
                    f"final_particle_count={final['particle_count']}",
                    f"final_mean_radius_nm={final['mean_radius_nm']}",
                    f"final_far_field_matrix_xAg={final_xag}",
                    f"max_system_mass_drift={max_mass_drift}",
                    f"max_zero_mode_lambda={max_lambda}",
                    "GP_enabled=false",
                    "external_source_enabled=false",
                    "new_beta_nucleation_enabled=false",
                    "physical_parameter_retuning=false",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        print(PASS)
    except (KeyError, OSError, ValueError) as exc:
        (args.out / "status.txt").write_text(
            f"{FAIL}\n{exc}\n", encoding="utf-8"
        )
        raise SystemExit(f"[fatal] {exc}") from exc


if __name__ == "__main__":
    main()
