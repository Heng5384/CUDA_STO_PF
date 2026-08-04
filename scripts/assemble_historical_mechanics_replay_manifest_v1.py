#!/usr/bin/env python3
"""Audit and summarize immutable historical accepted-field mechanics replays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


AGES_TO_STEPS = {6: 0, 12: 21798, 18: 43596, 24: 65393, 36: 108989, 48: 152585}
REPLICATES = ("A", "B", "C")
COMPONENTS = ("xx", "yy", "zz", "xy", "xz", "yz")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def composite_sha(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_summary(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            raise ValueError(f"malformed replay summary line: {line!r}")
        values[key] = value
    return values


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--parameter-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)

    source_files = [
        args.source_root / "main_cuda",
        args.source_root / "main_cuda.cu",
        args.source_root / "cuda_kernels.cu",
        args.source_root / "pf_zero_mode_checkpoint.cpp",
        args.source_root / "jobs/run_historical_mechanics_accepted_field_replay_v1.sh",
    ]
    for path in source_files + [args.parameter_file]:
        if not path.is_file():
            raise FileNotFoundError(path)

    field_rows: list[dict] = []
    residual_rows: list[dict] = []
    authority_rows: list[dict] = []
    for replicate in REPLICATES:
        run_root = args.replay_root / replicate
        if (run_root / "status.txt").read_text(encoding="utf-8").strip() != "PASS_HISTORICAL_MECHANICS_ACCEPTED_FIELD_REPLAY_V1":
            raise ValueError(f"historical replay did not pass: {replicate}")
        audit_path = args.authority_root / replicate / "audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        fixture_path = args.input_root / replicate / "fixture_manifest.json"
        fixture_sha = sha256(fixture_path)
        if fixture_sha != audit["fixture_manifest_sha256"]:
            raise ValueError(f"fixture manifest mismatch: {replicate}")
        authority_rows.append(
            {
                "replicate": replicate,
                "authority_audit_sha256": sha256(audit_path),
                "fixture_manifest_sha256": fixture_sha,
                "authority_numerical_status": audit["numerical_status"],
                "hourly_merge_audit_status": audit["hourly_merge_audit"]["status"],
            }
        )
        for age_h, step in AGES_TO_STEPS.items():
            fields = run_root / f"age_{age_h}h" / "fields"
            summary_path = fields / "replay_summary.txt"
            summary = parse_summary(summary_path)
            required_contract = {
                "schema": "MECHANICS_ONLY_ACCEPTED_FIELD_REPLAY_V1",
                "time_advanced": "false",
                "phi_advanced": "false",
                "xB_advanced": "false",
                "checkpoint_written": "false",
            }
            for key, expected in required_contract.items():
                if summary.get(key) != expected:
                    raise ValueError(f"{replicate} {age_h} h: {key}={summary.get(key)!r}")
            residual = float(summary["solver_relative_residual"])
            tolerance = float(summary["solver_residual_tolerance"])
            if residual > tolerance or tolerance > 1.0e-6 * (1.0 + 1.0e-12):
                raise ValueError(f"solver residual gate failed: {replicate} {age_h} h")
            checkpoint_sha = "INITIAL_FIXTURE_NO_CHECKPOINT"
            if step:
                checkpoint_path = args.input_root / replicate / "checkpoints" / f"step_{step}.chk"
                checkpoint_sha = sha256(checkpoint_path)
                if checkpoint_sha != audit["checkpoint_hashes"][str(step)]:
                    raise ValueError(f"checkpoint mismatch: {replicate} step {step}")
            displacement_paths = [fields / f"displacement_{axis}.raw.f32" for axis in "xyz"]
            strain_paths = [fields / f"strain_{component}.raw.f32" for component in COMPONENTS]
            stress_paths = [fields / f"stress_{component}.raw.f32" for component in COMPONENTS]
            required_fields = [
                fields / "accepted_phi.raw.f64",
                fields / "accepted_xB.raw.f64",
                fields / "displacement_k_packed.raw.f32",
                fields / "elastic_energy_density.raw.f64",
                *displacement_paths,
                *strain_paths,
                *stress_paths,
            ]
            for path in required_fields:
                if not path.is_file():
                    raise FileNotFoundError(path)
            field_rows.append(
                {
                    "replicate": replicate,
                    "experimental_age_h": age_h,
                    "accepted_step": step,
                    "checkpoint_sha256": checkpoint_sha,
                    "accepted_phi_sha256": sha256(fields / "accepted_phi.raw.f64"),
                    "accepted_xB_sha256": sha256(fields / "accepted_xB.raw.f64"),
                    "displacement_real_composite_sha256": composite_sha(displacement_paths),
                    "displacement_k_packed_sha256": sha256(fields / "displacement_k_packed.raw.f32"),
                    "strain_composite_sha256": composite_sha(strain_paths),
                    "stress_composite_sha256": composite_sha(stress_paths),
                    "elastic_energy_density_sha256": sha256(fields / "elastic_energy_density.raw.f64"),
                    "replay_summary_sha256": sha256(summary_path),
                    "bundle_role": summary["bundle_role"],
                    "warm_start_used": summary["warm_start_used"],
                }
            )
            residual_rows.append(
                {
                    "replicate": replicate,
                    "experimental_age_h": age_h,
                    "accepted_step": step,
                    "solver_iterations": int(summary["solver_iterations"]),
                    "solver_relative_residual": residual,
                    "solver_residual_tolerance": tolerance,
                    "residual_gate_pass": True,
                    "elastic_energy_density_mean_hat": float(summary["elastic_energy_density_mean_hat"]),
                    "elastic_energy_density_mean_J_m3": float(summary["elastic_energy_density_mean_J_m3"]),
                    "total_elastic_energy_J": float(summary["total_elastic_energy_J"]),
                    "time_advanced": summary["time_advanced"],
                    "phi_advanced": summary["phi_advanced"],
                    "xB_advanced": summary["xB_advanced"],
                    "checkpoint_written": summary["checkpoint_written"],
                }
            )

    write_csv(args.out / "replay_field_manifest.csv", field_rows)
    write_csv(args.out / "replay_solver_residuals.csv", residual_rows)
    write_csv(args.out / "replay_authority_manifest.csv", authority_rows)
    source_hashes = {path.name: sha256(path) for path in source_files}
    provenance = {
        "schema": "HISTORICAL_MECHANICS_ACCEPTED_FIELD_REPLAY_AUDIT_V1",
        "status": "PASS_HISTORICAL_MECHANICS_ACCEPTED_FIELD_REPLAY_AUDIT_V1",
        "replicates": list(REPLICATES),
        "ages_h": list(AGES_TO_STEPS),
        "accepted_steps": list(AGES_TO_STEPS.values()),
        "parameter_file_sha256": sha256(args.parameter_file),
        "source_hashes": source_hashes,
        "writeback_to_authority": False,
        "pf_time_advance": False,
        "replay_initialization_contract": "6 h synchronized fixture solve; 12-48 h checkpoint_warm accepted-field replay",
    }
    (args.out / "replay_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Historical accepted-field mechanics replay provenance",
        "",
        f"Status: `{provenance['status']}`.",
        "",
        "A, B, and C each use one frozen Method-1 production authority. The 6 h field is the immutable initial fixture; 12-48 h fields are read from the exact audited checkpoints. Every checkpoint is verified against its authority audit before replay.",
        "",
        "The replay advances neither PF fields nor simulation time and writes no checkpoint. Historical authorities are read-only. The only qualified historical initialization is `checkpoint_warm`; the 6 h fixture uses the synchronized pre-update mechanics diagnostic.",
        "",
        f"Parameter SHA-256: `{provenance['parameter_file_sha256']}`.",
        "",
        "## Source hashes",
        "",
    ]
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in sorted(source_hashes.items()))
    (args.out / "replay_provenance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_paths = sorted(path for path in args.out.iterdir() if path.is_file())
    with (args.out / "replay_audit_outputs.sha256").open("w", encoding="utf-8") as stream:
        for path in output_paths:
            stream.write(f"{sha256(path)}  {path.name}\n")
    (args.out / "status.txt").write_text(provenance["status"] + "\n", encoding="utf-8")
    print(provenance["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
