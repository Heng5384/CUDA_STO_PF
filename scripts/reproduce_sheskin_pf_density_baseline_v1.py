#!/usr/bin/env python3
"""Reproduce the frozen Sheskin/PF density-only endpoint baseline.

This is deliberately narrower than the historical global radius/PSD audit. It
only validates the registered Method-1 A/B/C authority, evaluates the exact
full resolved PSD at 573.15 K, and records provenance.  It never scans or
modifies a PF population and never reads a dislocation refit contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA = "SHESKIN_PF_DENSITY_BASELINE_REPRODUCTION_V1"
REPLICATES = ("A", "B", "C")
AGES_H = (6.0, 48.0)
TEMPERATURE_K = 573.15
BOX_VOLUME_M3 = 246.0**3 * 1.0e-27
GAUSS_ORDER = 512
EXPECTED = {6.0: 1.2980916621210676, 48.0: 1.2948991542326225}
EXPECTED_DELTA = -0.003192507888445162
TOLERANCE_W_MK = 1.0e-12
PRODUCTION_STATUS = "PASS_246CUBE_6H48H_CONDITIONAL_PRODUCTION_V1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    require(bool(values), f"refusing to write empty CSV: {path}")
    fieldnames = list(values[0])
    require(
        all(list(row) == fieldnames for row in values),
        f"inconsistent CSV schema: {path}",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("sheskin_pf_transport", path)
    require(spec is not None and spec.loader is not None, "cannot load transport module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def commit_tree_digest(root: Path) -> str:
    return hashlib.sha256(
        subprocess.check_output(["git", "ls-tree", "-r", "HEAD"], cwd=root)
    ).hexdigest()


def parse_hash_ledger(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, source = line.split(maxsplit=1)
        require(len(digest) == 64, f"invalid SHA-256 ledger entry: {line}")
        entries[source.strip()] = digest
    return entries


def collect_authority(
    authority_root: Path,
) -> tuple[dict[str, dict[float, dict[str, Any]]], dict[str, Any], str]:
    populations: dict[str, dict[float, dict[str, Any]]] = {}
    provenance: dict[str, Any] = {}
    runtime_binary_hashes: set[str] = set()

    for replicate in REPLICATES:
        overlay = authority_root / "authority" / replicate
        source = authority_root / "source" / replicate
        transport = authority_root / "transport" / replicate
        audit_path = overlay / "audit.json"
        status_path = overlay / "status.txt"
        fixture_path = source / "fixture_manifest.json"
        ledger_path = source / "input_hashes.sha256"
        snapshots_path = transport / "transport_snapshots.csv"
        particles_path = transport / "full_psd_particles.csv"
        assembly_path = overlay / "assembly_provenance.json"
        required = (
            audit_path,
            status_path,
            fixture_path,
            ledger_path,
            snapshots_path,
            particles_path,
            assembly_path,
        )
        require(all(path.is_file() for path in required), f"missing {replicate} authority input")

        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        status = status_path.read_text(encoding="utf-8").strip()
        require(status == PRODUCTION_STATUS, f"{replicate} root status is not production PASS")
        require(audit.get("numerical_status") == status, f"{replicate} audit status mismatch")
        require(all(audit.get("gates", {}).values()), f"{replicate} authority gate failed")
        checkpoint_hashes = audit.get("checkpoint_hashes", {})
        require(len(checkpoint_hashes) == 44, f"{replicate} checkpoint chain is not 44 entries")
        require(
            audit["gates"].get("merge_aware_particle_lineage") is True
            and audit["gates"].get("hourly_merge_aware_particle_lineage") is True,
            f"{replicate} merge-aware gate failed",
        )

        ledger = parse_hash_ledger(ledger_path)
        binary_matches = {
            digest for source_path, digest in ledger.items() if source_path.endswith("/main_cuda")
        }
        require(len(binary_matches) == 1, f"{replicate} binary identity is ambiguous")
        runtime_binary_hashes.update(binary_matches)

        snapshot_rows = read_csv(snapshots_path)
        particle_rows = read_csv(particles_path)
        particles_by_snapshot: dict[str, list[float]] = defaultdict(list)
        for row in particle_rows:
            particles_by_snapshot[row["snapshot_id"]].append(
                float(row["equivalent_radius_nm"])
            )
        populations[replicate] = {}
        for row in snapshot_rows:
            age_h = float(row["age_h"])
            if age_h not in AGES_H:
                continue
            radii_nm = np.asarray(
                sorted(particles_by_snapshot[row["snapshot_id"]]), dtype=float
            )
            require(
                radii_nm.size == int(row["particle_count"]),
                f"{replicate} PSD count closure failed at {age_h:g} h",
            )
            populations[replicate][age_h] = {
                "radii_nm": radii_nm,
                "matrix_xAg": float(row["matrix_xAg"]),
                "snapshot_id": row["snapshot_id"],
                "step": int(row["step"]),
            }
        require(set(populations[replicate]) == set(AGES_H), f"{replicate} endpoints missing")

        assembly = json.loads(assembly_path.read_text(encoding="utf-8"))
        merge_path = Path(assembly["hourly_audit_dir"])
        if not merge_path.is_absolute():
            merge_path = authority_root.parents[1] / merge_path
        merge_audit = merge_path / "audit.json"
        require(merge_audit.is_file(), f"{replicate} merge-aware audit is missing")
        require(
            sha256(merge_audit) == assembly["hourly_audit_sha256"],
            f"{replicate} merge-aware audit hash mismatch",
        )
        require(
            sha256(fixture_path) == audit["fixture_manifest_sha256"],
            f"{replicate} fixture hash mismatch",
        )
        provenance[replicate] = {
            "authority_status_path": str(status_path.resolve()),
            "authority_status_sha256": sha256(status_path),
            "authority_audit_path": str(audit_path.resolve()),
            "authority_audit_sha256": sha256(audit_path),
            "fixture_manifest_path": str(fixture_path.resolve()),
            "fixture_manifest_sha256": sha256(fixture_path),
            "input_hash_ledger_path": str(ledger_path.resolve()),
            "input_hash_ledger_sha256": sha256(ledger_path),
            "checkpoint_hash_count": len(checkpoint_hashes),
            "checkpoint_hash_chain_sha256": canonical_sha256(checkpoint_hashes),
            "merge_aware_audit_path": str(merge_audit.resolve()),
            "merge_aware_audit_sha256": sha256(merge_audit),
            "transport_snapshots_sha256": sha256(snapshots_path),
            "full_psd_particles_sha256": sha256(particles_path),
            "remote_production_root": audit["remote_production_root"],
        }

    require(len(runtime_binary_hashes) == 1, "A/B/C production binaries are not identical")
    return populations, provenance, next(iter(runtime_binary_hashes))


def reproduce(
    transport: Any,
    config: dict[str, Any],
    populations: dict[str, dict[float, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[float, float], float]:
    rows: list[dict[str, Any]] = []
    for replicate in REPLICATES:
        for age_h in AGES_H:
            snapshot = populations[replicate][age_h]
            radii_nm = snapshot["radii_nm"]
            kappa = transport.integrate_kappa_gauss(
                TEMPERATURE_K,
                snapshot["matrix_xAg"],
                radii_nm * 1.0e-9,
                BOX_VOLUME_M3,
                config,
                "full_psd",
                GAUSS_ORDER,
            )
            rows.append(
                {
                    "replicate": replicate,
                    "age_h": age_h,
                    "temperature_K": TEMPERATURE_K,
                    "matrix_xAg": snapshot["matrix_xAg"],
                    "particle_count": radii_nm.size,
                    "full_PSD_sha256": canonical_sha256(radii_nm.tolist()),
                    "kappa_W_mK": kappa,
                    "expected_ensemble_kappa_W_mK": "",
                    "absolute_difference_W_mK": "",
                    "status": "REPLICATE_DIAGNOSTIC",
                }
            )

    means = {
        age_h: float(
            np.mean(
                [
                    float(row["kappa_W_mK"])
                    for row in rows
                    if row["age_h"] == age_h
                ]
            )
        )
        for age_h in AGES_H
    }
    for age_h in AGES_H:
        difference = abs(means[age_h] - EXPECTED[age_h])
        require(
            difference <= TOLERANCE_W_MK,
            f"BLOCKED_BASELINE_REPRODUCTION: {age_h:g} h difference={difference:.17g}",
        )
        rows.append(
            {
                "replicate": "ensemble_mean",
                "age_h": age_h,
                "temperature_K": TEMPERATURE_K,
                "matrix_xAg": "PF authority per replicate",
                "particle_count": "",
                "full_PSD_sha256": "",
                "kappa_W_mK": means[age_h],
                "expected_ensemble_kappa_W_mK": EXPECTED[age_h],
                "absolute_difference_W_mK": difference,
                "status": "PASS_BASELINE_REPRODUCTION",
            }
        )
    delta = means[48.0] - means[6.0]
    delta_difference = abs(delta - EXPECTED_DELTA)
    require(
        delta_difference <= TOLERANCE_W_MK,
        f"BLOCKED_BASELINE_REPRODUCTION: delta difference={delta_difference:.17g}",
    )
    rows.append(
        {
            "replicate": "ensemble_delta",
            "age_h": "6_to_48",
            "temperature_K": TEMPERATURE_K,
            "matrix_xAg": "PF authority per replicate",
            "particle_count": "",
            "full_PSD_sha256": "",
            "kappa_W_mK": delta,
            "expected_ensemble_kappa_W_mK": EXPECTED_DELTA,
            "absolute_difference_W_mK": delta_difference,
            "status": "PASS_BASELINE_REPRODUCTION",
        }
    )
    return rows, means, delta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/sheskin_pf_interface_strain_blind_prediction_v1"),
    )
    parser.add_argument("--sheskin-main-pdf", type=Path, required=True)
    parser.add_argument("--sheskin-si-pdf", type=Path)
    parser.add_argument("--yu-main-pdf", type=Path, required=True)
    parser.add_argument("--yu-si-pdf", type=Path, required=True)
    parser.add_argument("--digitization", type=Path, action="append", default=[])
    args = parser.parse_args()

    root = args.root.resolve()
    out = args.out if args.out.is_absolute() else root / args.out
    out = out.resolve()
    require(not out.exists(), f"refusing to overwrite output root: {out}")
    out.mkdir(parents=True)

    script_path = Path(__file__).resolve()
    authority_root = root / "reports/pf_246cube_method1_production_authority_v1"
    transport_path = root / "scripts/pf_full_psd_no_dislocation_transport_v1.py"
    contract_path = (
        root
        / "data/qualification/pf_full_psd_no_dislocation_transport_v1/transport_parameter_contract.json"
    )
    yu_config_path = root / "data/qualification/yu2024_transport_v1/yu_48h_parameters.json"

    transport = load_module(transport_path)
    config = transport.load_json(yu_config_path)
    contract = transport.load_json(contract_path)
    transport.validate_yu_base_config(config, yu_config_path)
    transport.validate_interface_contract(contract, contract_path, yu_config_path)
    require(float(config["shared_parameters"]["A_N"]) == 1.5, "A_N is not frozen")

    populations, authority_provenance, runtime_binary_sha256 = collect_authority(
        authority_root
    )
    rows, means, delta = reproduce(transport, config, populations)
    write_csv(out / "baseline_reproduction.csv", rows)
    (out / "baseline_reproduction_report.md").write_text(
        "# Baseline reproduction\n\n"
        "`PASS_BASELINE_REPRODUCTION`\n\n"
        f"- T = {TEMPERATURE_K:.2f} K\n"
        f"- kappa(6 h) = {means[6.0]:.16g} W m^-1 K^-1\n"
        f"- kappa(48 h) = {means[48.0]:.16g} W m^-1 K^-1\n"
        f"- delta kappa = {delta:.16g} W m^-1 K^-1\n"
        f"- absolute tolerance = {TOLERANCE_W_MK:.1e} W m^-1 K^-1\n\n"
        "The calculation uses the registered per-replicate PF matrix xAg and the "
        "full resolved PSD direct sum with A_N=1.5, S11=S13=0, and no Yu refit "
        "scale. No radius/count/PSD scan was executed.\n",
        encoding="utf-8",
    )

    source_paths = {
        "analysis_script": script_path,
        "transport_script": transport_path,
        "transport_parameter_contract": contract_path,
        "yu_parameter_contract": yu_config_path,
        "yu_main_pdf": args.yu_main_pdf.resolve(),
        "yu_supporting_information_pdf": args.yu_si_pdf.resolve(),
        "sheskin_main_pdf": args.sheskin_main_pdf.resolve(),
    }
    if args.sheskin_si_pdf is not None:
        source_paths["sheskin_supporting_information_pdf"] = args.sheskin_si_pdf.resolve()
    for index, path in enumerate(args.digitization, start=1):
        source_paths[f"digitization_{index}"] = path.resolve()
    require(all(path.is_file() for path in source_paths.values()), "missing provenance input")
    input_hashes = {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in source_paths.items()
    }
    provenance = {
        "schema": SCHEMA,
        "status": "PASS_BASELINE_REPRODUCTION",
        "git_branch": git(root, "branch", "--show-current"),
        "git_commit": git(root, "rev-parse", "HEAD"),
        "git_commit_tree_sha256": commit_tree_digest(root),
        "working_tree_porcelain": git(root, "status", "--porcelain"),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "production_cuda_binary_sha256": runtime_binary_sha256,
        "analysis_inputs": input_hashes,
        "authority": authority_provenance,
        "frozen_contract": {
            "temperature_K": TEMPERATURE_K,
            "gauss_legendre_order": GAUSS_ORDER,
            "A_N": 1.5,
            "S11_rate": 0.0,
            "S13_rate": 0.0,
            "yu_refit_scale_used": False,
            "population": "Method-1 A/B/C full resolved PF PSD direct sum",
            "matrix_xAg": "registered PF far-field observation per replicate",
        },
    }
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    provenance_lines = [
        "# Provenance",
        "",
        "This stage is read-only and does not start PF, modify a checkpoint, or run a PSD scan.",
        "",
        f"- branch: `{provenance['git_branch']}`",
        f"- commit: `{provenance['git_commit']}`",
        f"- committed source-tree digest: `{provenance['git_commit_tree_sha256']}`",
        f"- production CUDA binary SHA-256: `{runtime_binary_sha256}`",
        "",
        "## Inputs",
        "",
    ]
    provenance_lines.extend(
        f"- {name}: `{entry['sha256']}` (`{entry['path']}`)"
        for name, entry in sorted(input_hashes.items())
    )
    provenance_lines.extend(["", "## A/B/C authorities", ""])
    for replicate in REPLICATES:
        item = authority_provenance[replicate]
        provenance_lines.append(
            f"- {replicate}: audit `{item['authority_audit_sha256']}`, fixture "
            f"`{item['fixture_manifest_sha256']}`, merge-aware audit "
            f"`{item['merge_aware_audit_sha256']}`, checkpoint-chain "
            f"`{item['checkpoint_hash_chain_sha256']}` ({item['checkpoint_hash_count']} entries)"
        )
    (out / "provenance.md").write_text(
        "\n".join(provenance_lines) + "\n", encoding="utf-8"
    )

    manifest = {
        path.name: sha256(path)
        for path in sorted(out.iterdir())
        if path.is_file()
    }
    (out / "stage0_manifest.json").write_text(
        json.dumps({"schema": SCHEMA, "outputs": manifest}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print("PASS_BASELINE_REPRODUCTION")


if __name__ == "__main__":
    main()
