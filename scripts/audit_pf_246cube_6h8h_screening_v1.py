#!/usr/bin/env python3
"""Fail-closed audit of one chained 246^3 6--8 h short screening."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List


PASS = "PASS_246CUBE_6H8H_SHORT_SCREENING_V1"
FAIL = "BLOCKED_246CUBE_6H8H_SHORT_SCREENING_V1"
MERGE_AWARE_PASS = "PASS_246CUBE_RESOLVED_MERGE_AWARE_LINEAGE_V1"
EXPECTED_STEPS = [
    0,
    256,
    512,
    1024,
    1536,
    2048,
    2560,
    3072,
    3584,
    4096,
    4608,
    5120,
    5632,
    6144,
    6656,
    7168,
    7266,
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def key_values(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = raw.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    return values


def csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def final_csv(path: Path) -> Dict[str, str]:
    rows = csv_rows(path)
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows[-1]


def wall_seconds(text: str) -> float:
    hits = re.findall(r"wall_time_s\s*:\s*([0-9.eE+-]+)", text)
    if not hits:
        raise ValueError("wall time not found")
    return float(hits[-1])


def gpu_metrics(path: Path) -> Dict[str, float]:
    memory: List[float] = []
    utilization: List[float] = []
    if path.is_file():
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
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
            sum(utilization) / len(utilization) if utilization else math.nan
        ),
        "peak_VRAM_MiB": max(memory) if memory else math.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--stage7-checkpoint", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--lineage-root", type=Path, required=True)
    parser.add_argument("--gpu-samples", type=Path, required=True)
    parser.add_argument("--merge-aware-audit", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)
    try:
        manifest = json.loads(
            args.fixture_manifest.read_text(encoding="utf-8")
        )
        merge_aware = None
        if args.merge_aware_audit is not None:
            merge_aware = json.loads(
                args.merge_aware_audit.read_text(encoding="utf-8")
            )
            if (
                merge_aware.get("status") != MERGE_AWARE_PASS
                or not all(merge_aware.get("gates", {}).values())
            ):
                raise ValueError("merge-aware audit is not exact PASS")
        lineage_summary = key_values(
            args.lineage_root / "lineage_summary.txt"
        )
        observations = csv_rows(
            args.lineage_root / "ensemble_observables.csv"
        )
        observed_steps = [int(row["step"]) for row in observations]
        if observed_steps != EXPECTED_STEPS:
            raise ValueError(
                f"snapshot step contract mismatch: {observed_steps}"
            )
        segment_steps = EXPECTED_STEPS[2:]
        checkpoint_hashes = {
            "256": sha256(args.stage7_checkpoint),
        }
        mass_rows: Dict[int, Dict[str, str]] = {}
        logs: Dict[int, str] = {}
        stderr_empty = True
        total_wall = 0.0
        max_abs_lambda = 0.0
        provenance_ok = True
        prohibited_ok = True
        zero_mode_ok = True
        fixture_hash = sha256(args.fixture_manifest)
        tokens = (
            "initial_state_class="
            "MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1",
            f"fixture_manifest_sha256={fixture_hash}",
            "profile_library_manifest_sha256="
            "58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe",
            "checkpoint_restart_provenance=RESTORED_AND_VALIDATED",
            "pf_sm_explicit_context=SM_EXPLICIT_CONTEXT_N_V1",
            "pf_reaction_discretization=SM_TANGENT_N_V1",
            "gp_enabled=false",
        )
        for step in segment_steps:
            root = args.run_root / "segments" / f"step_{step}"
            checkpoint = args.run_root / "checkpoints" / f"step_{step}.chk"
            checkpoint_hashes[str(step)] = sha256(checkpoint)
            text = (root / "stdout.log").read_text(
                encoding="utf-8", errors="replace"
            )
            logs[step] = text
            stderr_empty = stderr_empty and not (root / "stderr.log").read_bytes()
            total_wall += wall_seconds(text)
            zero_mode_ok = (
                zero_mode_ok
                and "PF_ZERO_MODE_FINAL_AUDIT status=PASS" in text
            )
            provenance_ok = provenance_ok and all(
                token in text for token in tokens
            )
            prohibited_ok = prohibited_ok and not re.search(
                r"GP_EVENT|GP_BIRTH|BETA_NUCLEATION_EVENT|source_event",
                text,
            )
            lambda_hits = re.findall(
                r"PF_ZERO_MODE_FINAL_AUDIT status=PASS .*?"
                r"last_lambda=([0-9.eE+-]+)",
                text,
            )
            if not lambda_hits:
                raise ValueError(f"missing zero-mode lambda at step {step}")
            max_abs_lambda = max(
                max_abs_lambda, abs(float(lambda_hits[-1]))
            )
            mass_rows[step] = final_csv(
                root / "dynamics_mass_diagnostics.csv"
            )
            if int(float(mass_rows[step]["step"])) != step:
                raise ValueError(f"mass diagnostic endpoint mismatch: {step}")

        enriched: List[Dict[str, Any]] = []
        for row in observations:
            step = int(row["step"])
            copy: Dict[str, Any] = dict(row)
            if step in mass_rows:
                mass = mass_rows[step]
                for name in (
                    "mean_elastic_energy",
                    "max_elastic_energy",
                    "stress_hydro_min",
                    "stress_hydro_max",
                ):
                    copy[name] = float(mass[name])
            else:
                for name in (
                    "mean_elastic_energy",
                    "max_elastic_energy",
                    "stress_hydro_min",
                    "stress_hydro_max",
                ):
                    copy[name] = ""
            enriched.append(copy)

        mass_drift = max(
            abs(float(row["canonical_inventory_code"])
                - float(observations[0]["canonical_inventory_code"]))
            / max(
                abs(float(observations[0]["canonical_inventory_code"])),
                1.0,
            )
            for row in observations
        )
        finite_bounds = all(
            int(row["finite"]) == 1 and int(row["bounds"]) == 1
            for row in observations
        )
        final_age = float(observations[-1]["experimental_age_h"])
        merge_count = int(lineage_summary.get("merge_count", -1))
        unqualified_merge_count = int(
            lineage_summary.get("unqualified_merge_count", merge_count)
        )
        merge_lineage_ok = (
            merge_count == 0
            or (
                merge_aware is not None
                and unqualified_merge_count == 0
                and int(
                    lineage_summary.get("qualified_merge_count", -1)
                )
                == merge_count
            )
        )
        gates = {
            "fixture_schema": (
                manifest.get("schema")
                == "PF_246CUBE_LIBRARY_HANDOFF_MANIFEST_V1"
            ),
            "snapshot_contract": observed_steps == EXPECTED_STEPS,
            "endpoint": (
                int(observations[-1]["step"]) == 7266
                and abs(final_age - 8.000019169100993) <= 1.0e-12
            ),
            "finite_and_bounds": finite_bounds,
            "mass_and_zero_mode": (
                mass_drift <= 1.0e-10 and zero_mode_ok
            ),
            "stderr_empty": stderr_empty,
            "checkpoint_provenance": provenance_ok,
            "prohibited_paths": prohibited_ok,
            "particle_lineage": (
                lineage_summary.get("status")
                == "PASS_PERIODIC_OVERLAP_PARTICLE_LINEAGE_V1"
                and merge_lineage_ok
                and int(lineage_summary.get("split_count", -1)) == 0
                and int(lineage_summary.get("new_component_count", -1)) == 0
                and int(
                    lineage_summary.get(
                        "unqualified_dissolution_count", -1
                    )
                )
                == 0
            ),
        }
        if not all(gates.values()):
            failed = [key for key, value in gates.items() if not value]
            raise ValueError(f"failed 6--8 h gates: {failed}")

        with (args.out / "ensemble_observables_6h_8h.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(enriched[0]))
            writer.writeheader()
            writer.writerows(enriched)
        audit = {
            "schema": "PF_246CUBE_6H8H_SHORT_SCREENING_AUDIT_V1",
            "status": PASS,
            "replicate": manifest["replicate_id"],
            "fixture_manifest_sha256": fixture_hash,
            "checkpoint_hashes": checkpoint_hashes,
            "gates": gates,
            "metrics": {
                "initial_particle_count": int(
                    lineage_summary["initial_particle_count"]
                ),
                "final_particle_count": int(
                    lineage_summary["final_particle_count"]
                ),
                "dissolution_count": int(
                    lineage_summary["dissolution_count"]
                ),
                "merge_count": merge_count,
                "qualified_merge_count": int(
                    lineage_summary.get("qualified_merge_count", 0)
                ),
                "unqualified_merge_count": unqualified_merge_count,
                "merge_aware_status": (
                    merge_aware["status"]
                    if merge_aware is not None
                    else "NOT_REQUIRED_NO_MERGE"
                ),
                "split_count": 0,
                "max_system_mass_drift": mass_drift,
                "max_zero_mode_lambda": max_abs_lambda,
                "segment_wall_seconds": total_wall,
                "wall_seconds_per_step": total_wall / (7266 - 256),
                "actual_endpoint_age_h": final_age,
                "endpoint_time_error_s": 0.0690087636,
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
                    f"status={PASS}",
                    f"replicate={audit['replicate']}",
                    f"final_particle_count={audit['metrics']['final_particle_count']}",
                    f"dissolution_count={audit['metrics']['dissolution_count']}",
                    "merge_split_status="
                    + (
                        "PASS_RESOLVED_MERGE_GROUPS"
                        if merge_count
                        else "PASS_NONE"
                    ),
                    f"max_system_mass_drift={mass_drift}",
                    f"max_zero_mode_lambda={max_abs_lambda}",
                    "GP_enabled=false",
                    "new_beta_nucleation_enabled=false",
                )
            )
            + "\n",
            encoding="utf-8",
        )
    except ValueError as exc:
        (args.out / "status.txt").write_text(
            f"{FAIL}\n{exc}\n", encoding="utf-8"
        )
        raise SystemExit(f"[fatal] {exc}") from exc
    print(PASS)


if __name__ == "__main__":
    main()
