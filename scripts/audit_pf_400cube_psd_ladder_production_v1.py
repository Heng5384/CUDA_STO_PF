#!/usr/bin/env python3
"""Read-only completion audit for one 400^3 PSD/spatial/density ladder case."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


PASS = "PASS_400CUBE_PSD_LADDER_PRODUCTION_V1"
BLOCKED = "BLOCKED_400CUBE_PSD_LADDER_PRODUCTION_V1"
SEGMENT_PASS = "PASS_400CUBE_PSD_LADDER_SEGMENT_V1"
FINAL_STEP = 152585
DT_PHYSICAL_S = 0.9909260953431841
CHECKPOINT_CADENCE = 3633
MATRIX_XAG_INTERVAL = (0.0058, 0.0066)
F_BETA_REL_TOL = 1.0e-2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def kv(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, marker, value = raw.partition("=")
        if marker:
            result[key] = value
    return result


def gpu_metrics(path: Path) -> dict[str, float | int | None]:
    util: list[float] = []
    memory: list[float] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines() if path.is_file() else []:
        fields = [value.strip() for value in raw.split(",")]
        try:
            util.append(float(fields[-2])); memory.append(float(fields[-1]))
        except (ValueError, IndexError):
            continue
    return {"gpu_sample_count": len(memory), "gpu_utilization_mean_percent": sum(util) / len(util) if util else None, "peak_VRAM_MiB": max(memory) if memory else None}


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    if not values:
        raise ValueError(f"refusing empty CSV: {path}")
    fields = list(values[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-manifest", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--lineage-root", required=True, type=Path)
    parser.add_argument("--merge-aware-audit", required=True, type=Path)
    parser.add_argument("--gpu-samples", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)
    try:
        fixture = json.loads(args.fixture_manifest.read_text(encoding="utf-8"))
        campaign = json.loads((args.run_root / "provenance/campaign_manifest.json").read_text(encoding="utf-8"))
        if fixture.get("schema") != "PF_400CUBE_PSD_SPATIAL_DENSITY_INTEGER_GATE_FIXTURE_V2":
            raise ValueError("fixture schema mismatch")
        if campaign.get("grid") != [400, 400, 400] or campaign.get("final_step") != FINAL_STEP:
            raise ValueError("campaign grid/final step mismatch")
        if abs(float(campaign.get("dt_physical_s", math.nan)) - DT_PHYSICAL_S) > 1.0e-12:
            raise ValueError("physical timestep mismatch")
        if sha256(args.fixture_manifest) != campaign.get("fixture_manifest_sha256"):
            raise ValueError("campaign fixture hash mismatch")
        physical = fixture.get("physical_contract", {})
        if physical.get("elasticity_enabled") is not True:
            raise ValueError("elasticity was not enabled")
        if any(physical.get(key) is not False for key in ("GP_enabled", "GP_birth_enabled", "GP_release_enabled", "external_source_enabled", "new_beta_nucleation_enabled")):
            raise ValueError("a prohibited source or nucleation path is enabled")
        endpoints = [int(line) for line in (args.run_root / "provenance/checkpoint_steps.txt").read_text().split()]
        # The campaign retains every one-hour checkpoint and also preserves
        # the exact registered 24 h/36 h science endpoints.  Those two
        # physical-time mappings fall one macro step before the corresponding
        # hourly integer multiples, so both entries are intentionally present.
        registered_science = {int(step) for step in campaign.get("registered_science_steps", []) if int(step) > 0}
        expected = sorted(set(range(CHECKPOINT_CADENCE, FINAL_STEP, CHECKPOINT_CADENCE)) | {FINAL_STEP} | registered_science)
        if endpoints != expected or len(endpoints) != 44:
            raise ValueError("checkpoint schedule is not the registered hourly/science chain")
        lineage = rows(args.lineage_root / "particle_lineage.csv")
        observations = rows(args.lineage_root / "ensemble_observables.csv")
        if [int(row["step"]) for row in observations] != [0] + endpoints:
            raise ValueError("hourly observations do not cover the complete checkpoint chain")
        initial_count = int(fixture["component_contract"]["expected_count"])
        if int(observations[0]["particle_count"]) != initial_count:
            raise ValueError("initial PSD registration mismatch")
        initial_inventory = float(observations[0]["canonical_inventory_code"])
        mass_drift = max(abs(float(row["canonical_inventory_code"]) - initial_inventory) / max(abs(initial_inventory), 1.0) for row in observations)
        finite_bounds = all(row["finite"] == "1" and row["bounds"] == "1" for row in observations)
        checkpoint_hashes: dict[str, str] = {}
        zero_mode_ok = stderr_empty = provenance_ok = True
        wall_seconds = 0.0
        max_lambda = 0.0
        tokens = ("pf_zero_mode=PF_CONSERVED_Y_ZERO_MODE_V1", "gp_enabled=false", f"fixture_manifest_sha256={sha256(args.fixture_manifest)}")
        for index, step in enumerate(endpoints):
            segment = args.run_root / "segments" / f"step_{step}"
            checkpoint = args.run_root / "checkpoints" / f"step_{step}.chk"
            if not checkpoint.is_file() or not (segment / "dynamics_mass_diagnostics.csv").is_file():
                raise ValueError(f"missing segment artifact at step {step}")
            if (segment / "status.txt").read_text().strip() != SEGMENT_PASS:
                raise ValueError(f"segment is not an exact PASS at step {step}")
            checkpoint_hashes[str(step)] = sha256(checkpoint)
            stdout = (segment / "stdout.log").read_text(encoding="utf-8", errors="replace")
            stderr_empty = stderr_empty and not (segment / "stderr.log").read_bytes()
            zero_mode_ok = zero_mode_ok and "PF_ZERO_MODE_FINAL_AUDIT status=PASS" in stdout
            provenance_ok = provenance_ok and all(token in stdout for token in tokens)
            if index:
                provenance_ok = provenance_ok and "checkpoint_restart_provenance=RESTORED_AND_VALIDATED" in stdout
            hits = re.findall(r"last_lambda=([0-9.eE+-]+)", stdout)
            if hits:
                max_lambda = max(max_lambda, abs(float(hits[-1])))
            wall_hits = re.findall(r"wall_time_s\\s*:\\s*([0-9.eE+-]+)", stdout)
            if wall_hits:
                wall_seconds += float(wall_hits[-1])
            mass_row = rows(segment / "dynamics_mass_diagnostics.csv")[-1]
            if int(float(mass_row["step"])) != step:
                raise ValueError(f"mass diagnostic endpoint mismatch at {step}")
        chain = (args.run_root / "provenance/checkpoint_chain.sha256").read_text(encoding="utf-8")
        if any(f"{value}  {args.run_root}/checkpoints/step_{step}.chk" not in chain for step, value in checkpoint_hashes.items()):
            raise ValueError("checkpoint SHA-256 chain mismatch")
        merge = json.loads(args.merge_aware_audit.read_text(encoding="utf-8"))
        merge_ok = str(merge.get("status", "")).startswith("PASS_") and all(merge.get("gates", {}).values())
        lineage_summary = kv(args.lineage_root / "lineage_summary.txt")
        tracker_ok = lineage_summary.get("status") == "PASS_PERIODIC_OVERLAP_PARTICLE_LINEAGE_V1"
        particle_rows_by_step: dict[int, list[dict[str, str]]] = {}
        for row in lineage:
            particle_rows_by_step.setdefault(int(row["step"]), []).append(row)
        hourly: list[dict[str, Any]] = []
        volume_m3 = (400.0e-9) ** 3
        for row in observations:
            step = int(row["step"])
            group = particle_rows_by_step.get(step, [])
            if len(group) != int(row["particle_count"]):
                raise ValueError(f"PSD count mismatch at step {step}")
            hourly.append({**row, "number_density_m3": int(row["particle_count"]) / volume_m3})
        write_csv(args.out / "hourly_observables.csv", hourly)
        write_csv(args.out / "hourly_particle_psd.csv", lineage)
        final = hourly[-1]
        expected_age = 6.0 + FINAL_STEP * DT_PHYSICAL_S / 3600.0
        matrix_xag_values = [float(row["far_field_matrix_xAg"]) for row in hourly]
        matrix_platform = all(MATRIX_XAG_INTERVAL[0] <= value <= MATRIX_XAG_INTERVAL[1] for value in matrix_xag_values)
        f_beta_0 = float(hourly[0]["beta_volume_fraction"])
        f_beta_rel = abs(float(final["beta_volume_fraction"]) - f_beta_0) / max(abs(f_beta_0), 1.0e-300)
        f_beta_conserved = f_beta_rel <= F_BETA_REL_TOL
        gates = {
            "fixture_and_campaign_contract": True,
            "complete_hourly_checkpoint_chain": True,
            "final_endpoint": int(final["step"]) == FINAL_STEP and abs(float(final["experimental_age_h"]) - expected_age) <= 1.0e-9,
            "finite_bounds": finite_bounds,
            "mass_conservation": mass_drift <= 1.0e-10,
            "matrix_xAg_platform": matrix_platform,
            "f_beta_conservation": f_beta_conserved,
            "zero_mode": zero_mode_ok,
            "segment_stderr_empty": stderr_empty,
            "checkpoint_provenance": provenance_ok,
            "merge_aware_identity": merge_ok and tracker_ok,
        }
        scientific = "COARSENING_DIRECTION_PRESENT" if int(final["particle_count"]) < initial_count and float(final["mean_radius_nm"]) > float(hourly[0]["mean_radius_nm"]) else "NO_REGISTERED_COARSENING_DIRECTION"
        audit = {"schema": "PF_400CUBE_PSD_LADDER_PRODUCTION_AUDIT_V1", "status": PASS if all(gates.values()) else BLOCKED, "gates": gates, "scientific_status": scientific, "checkpoint_hashes": checkpoint_hashes, "metrics": {"initial_particle_count": initial_count, "final_particle_count": int(final["particle_count"]), "initial_mean_radius_nm": float(hourly[0]["mean_radius_nm"]), "final_mean_radius_nm": float(final["mean_radius_nm"]), "final_far_field_matrix_xAg": float(final["far_field_matrix_xAg"]), "final_beta_volume_fraction": float(final["beta_volume_fraction"]), "beta_volume_fraction_relative_change": f_beta_rel, "max_system_mass_drift": mass_drift, "max_zero_mode_lambda": max_lambda, "wall_seconds": wall_seconds, "wall_seconds_per_step": wall_seconds / FINAL_STEP if wall_seconds else None, **gpu_metrics(args.gpu_samples)}, "qualification_scope": "256-step continuous/restart, first-hour checkpoint and 6-8 h stage passed before production; full hourly/science checkpoint chain and merge-aware lineage audited after production"}
        (args.out / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (args.out / "status.txt").write_text(audit["status"] + "\n", encoding="utf-8")
        (args.out / "final_terminal_output.txt").write_text("\n".join(f"{key}={value}" for key, value in {"final_status": audit["status"], "identity_status": "PASS" if gates["merge_aware_identity"] else "FAIL_CLOSED", "max_system_mass_drift": mass_drift, "max_zero_mode_lambda": max_lambda, "final_particle_count": final["particle_count"], "final_mean_radius_nm": final["mean_radius_nm"], "final_far_field_xAg": final["far_field_matrix_xAg"], "beta_volume_fraction_relative_change": f_beta_rel}.items()) + "\n", encoding="utf-8")
        print(audit["status"])
    except Exception as exc:
        detail = str(exc).replace("\n", " ")
        (args.out / "first_failure.csv").write_text("stage,gate,detail\ncompletion_audit,fail_closed," + detail.replace(",", ";") + "\n", encoding="utf-8")
        (args.out / "status.txt").write_text(BLOCKED + "\n", encoding="utf-8")
        (args.out / "final_terminal_output.txt").write_text(f"final_status={BLOCKED}\nrecommended_next_action=inspect_first_failure\n", encoding="utf-8")
        raise SystemExit(f"[fatal] {detail}") from exc


if __name__ == "__main__":
    main()
