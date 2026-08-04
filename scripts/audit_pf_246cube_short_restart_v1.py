#!/usr/bin/env python3
"""Assemble one 246^3/256-step restart and observable qualification."""

from __future__ import annotations

import argparse
import csv
import filecmp
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict


PASS = "PASS_246CUBE_SHORT_RESTART_AND_OBSERVABLES_V1"
FAIL = "BLOCKED_246CUBE_SHORT_RESTART_AND_OBSERVABLES_V1"
FIXTURE_SCHEMA = "PF_246CUBE_LIBRARY_HANDOFF_MANIFEST_V1"
DEFAULT_LIBRARY_SHA256 = (
    "58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe"
)
INITIAL_STATE_CLASS = (
    "MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_key_values(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = raw.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    return values


def last_csv_row(path: Path) -> Dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows[-1]


def wall_seconds(text: str) -> float:
    hits = re.findall(r"wall_time_s\s*:\s*([0-9.eE+-]+)", text)
    if not hits:
        raise ValueError("wall time not found")
    return float(hits[-1])


def zero_mode_metrics(text: str) -> Dict[str, float]:
    hits = re.findall(
        r"PF_ZERO_MODE_FINAL_AUDIT status=PASS .*?"
        r"final_mass_code=([0-9.eE+-]+) .*?"
        r"mean_mass_error=([0-9.eE+-]+) .*?"
        r"last_lambda=([0-9.eE+-]+)",
        text,
    )
    if not hits:
        raise ValueError("zero-mode final PASS audit not found")
    final_mass, mean_error, lambda_value = hits[-1]
    return {
        "final_mass_code": float(final_mass),
        "mean_mass_error": float(mean_error),
        "last_lambda": float(lambda_value),
    }


def gpu_metrics(path: Path) -> Dict[str, float]:
    memory = []
    utilization = []
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
    parser.add_argument("--library-sha256", default=DEFAULT_LIBRARY_SHA256)
    parser.add_argument("--continuous-checkpoint", type=Path, required=True)
    parser.add_argument("--restart-checkpoint", type=Path, required=True)
    parser.add_argument("--continuous-stdout", type=Path, required=True)
    parser.add_argument("--restart-stdout", type=Path, required=True)
    parser.add_argument("--continuous-stderr", type=Path, required=True)
    parser.add_argument("--restart-stderr", type=Path, required=True)
    parser.add_argument("--continuous-mass-csv", type=Path, required=True)
    parser.add_argument("--restart-mass-csv", type=Path, required=True)
    parser.add_argument("--lineage-root", type=Path, required=True)
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
        continuous_text = args.continuous_stdout.read_text(
            encoding="utf-8", errors="replace"
        )
        restart_text = args.restart_stdout.read_text(
            encoding="utf-8", errors="replace"
        )
        joined = continuous_text + "\n" + restart_text
        fixture_hash = sha256(args.fixture_manifest)
        provenance_tokens = (
            f"initial_state_class={INITIAL_STATE_CLASS}",
            f"fixture_manifest_sha256={fixture_hash}",
            f"profile_library_manifest_sha256={args.library_sha256}",
            "pf_sm_explicit_context=SM_EXPLICIT_CONTEXT_N_V1",
            "pf_reaction_discretization=SM_TANGENT_N_V1",
            "gp_enabled=false",
        )
        lineage = read_key_values(args.lineage_root / "lineage_summary.txt")
        observations = last_csv_row(
            args.lineage_root / "ensemble_observables.csv"
        )
        continuous_mass = last_csv_row(args.continuous_mass_csv)
        restart_mass = last_csv_row(args.restart_mass_csv)
        continuous_zero = zero_mode_metrics(continuous_text)
        restart_zero = zero_mode_metrics(restart_text)
        bytewise = filecmp.cmp(
            args.continuous_checkpoint,
            args.restart_checkpoint,
            shallow=False,
        )
        elastic_names = (
            "mean_elastic_energy",
            "max_elastic_energy",
            "stress_hydro_min",
            "stress_hydro_max",
        )
        elastic = {
            key: float(continuous_mass[key]) for key in elastic_names
        }
        gates = {
            "fixture_schema": manifest.get("schema") == FIXTURE_SCHEMA,
            "initial_state_class": (
                manifest.get("initial_state_class") == INITIAL_STATE_CLASS
            ),
            "library_hash": (
                manifest.get("profile_library_manifest_sha256")
                == args.library_sha256
            ),
            "prohibited_paths_off": all(
                manifest.get("physical_contract", {}).get(key) is False
                for key in (
                    "GP_enabled",
                    "GP_birth_enabled",
                    "GP_release_enabled",
                    "external_source_enabled",
                    "new_beta_nucleation_enabled",
                )
            ),
            "stderr_empty": (
                not args.continuous_stderr.read_bytes()
                and not args.restart_stderr.read_bytes()
            ),
            "zero_mode": (
                "PF_ZERO_MODE_FINAL_AUDIT status=PASS" in continuous_text
                and "PF_ZERO_MODE_FINAL_AUDIT status=PASS" in restart_text
            ),
            "checkpoint_bytewise": bytewise,
            "checkpoint_provenance": all(
                token in continuous_text and token in restart_text
                for token in provenance_tokens
            )
            and "checkpoint_restart_provenance=RESTORED_AND_VALIDATED"
            in restart_text,
            "particle_identity": (
                lineage.get("status")
                == "PASS_PERIODIC_OVERLAP_PARTICLE_LINEAGE_V1"
                and int(lineage.get("initial_particle_count", -1)) == 96
                and int(lineage.get("final_particle_count", -1)) == 96
                and int(lineage.get("merge_count", -1)) == 0
                and int(lineage.get("split_count", -1)) == 0
                and int(lineage.get("new_component_count", -1)) == 0
                and int(lineage.get("dissolution_count", -1)) == 0
            ),
            "endpoint": (
                int(observations["step"]) == 256
                and abs(float(observations["experimental_age_h"])
                        - 6.070465855668848)
                <= 1.0e-12
            ),
            "finite_and_bounds": (
                int(observations["finite"]) == 1
                and int(observations["bounds"]) == 1
            ),
            "mass": (
                float(observations["mass_relative_error"]) <= 1.0e-10
                and abs(float(continuous_mass["mean_xBtot_end_step"]) - 0.03)
                <= 1.0e-12
                and abs(float(restart_mass["mean_xBtot_end_step"]) - 0.03)
                <= 1.0e-12
            ),
            "elastic_finite": all(
                math.isfinite(value) for value in elastic.values()
            ),
            "no_illegal_event_marker": not re.search(
                r"GP_EVENT|GP_BIRTH|BETA_NUCLEATION_EVENT|source_event",
                joined,
            ),
        }
        if not all(gates.values()):
            failed = [key for key, value in gates.items() if not value]
            raise ValueError(f"failed short-restart gates: {failed}")
        metrics: Dict[str, Any] = {
            "replicate": manifest["replicate_id"],
            "fixture_manifest_sha256": fixture_hash,
            "continuous_checkpoint_sha256": sha256(
                args.continuous_checkpoint
            ),
            "restart_checkpoint_sha256": sha256(args.restart_checkpoint),
            "restart_bytewise_equal": bytewise,
            "physical_time_s": 253.67708040785513,
            "endpoint_age_h": float(observations["experimental_age_h"]),
            "particle_count": int(observations["particle_count"]),
            "beta_volume_fraction": float(
                observations["beta_volume_fraction"]
            ),
            "mean_radius_nm": float(observations["mean_radius_nm"]),
            "Sv_nm_inv": float(observations["Sv_nm_inv"]),
            "M6_nm3": float(observations["M6_nm3"]),
            "far_field_matrix_xAg": float(
                observations["far_field_matrix_xAg"]
            ),
            "initial_reconstruction_phi_normalized_L1": float(
                observations["phi_normalized_l1_from_initial"]
            ),
            "initial_reconstruction_xB_MAE": float(
                observations["xB_mean_absolute_from_initial"]
            ),
            "mass_relative_error": float(
                observations["mass_relative_error"]
            ),
            "continuous_zero_mode": continuous_zero,
            "restart_zero_mode": restart_zero,
            "continuous_wall_seconds": wall_seconds(continuous_text),
            "restart_second_leg_wall_seconds": wall_seconds(restart_text),
            **elastic,
            **gpu_metrics(args.gpu_samples),
        }
        audit = {
            "schema": "PF_246CUBE_SHORT_RESTART_AUDIT_V1",
            "status": PASS,
            "gates": gates,
            "metrics": metrics,
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
                    f"replicate={metrics['replicate']}",
                    "restart_status=PASS_BYTEWISE",
                    "particle_identity_status=PASS",
                    f"particle_count={metrics['particle_count']}",
                    f"mass_relative_error={metrics['mass_relative_error']}",
                    "zero_mode_status=PASS",
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
