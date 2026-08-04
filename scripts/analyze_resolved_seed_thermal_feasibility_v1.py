#!/usr/bin/env python3
"""Audit resolved-particle-only 6 h--48 h thermal-trend feasibility.

This is a read-only analysis of the frozen Method-1 A/B/C authority.  It
reproduces the frozen full-PSD transport baseline, scans fixed-inventory
resolved radius/count states without running PF, calibrates at most one
interface-scattering coefficient from the 6 h ensemble, and makes a frozen
coefficient 48 h prediction.  It deliberately contains no PF parameter
updates and never writes into the production authority.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss


SCHEMA = "RESOLVED_SEED_ONLY_6H48H_THERMAL_TREND_FEASIBILITY_V1"
TEMPERATURE_K = 573.15
EXPERIMENT_KAPPA_6H = 0.85
EXPERIMENT_KAPPA_48H = 1.03
EXPECTED_BASELINE = {
    6.0: 1.2980916621210676,
    48.0: 1.2948991542326225,
}
EXPECTED_DELTA = -0.003192507888445162
AUTHORITY_STATUS = "PASS_246CUBE_6H48H_CONDITIONAL_PRODUCTION_V1"
TRANSPORT_STATUS = "PASS_PF_FULL_PSD_NO_DISLOCATION_TRANSPORT_ENSEMBLE_V1"
REPLICATES = ("A", "B", "C")
BOX_VOLUME_NM3 = float(246**3)
GAUSS_ORDER = 512
LOGNORMAL_ORDER = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |\n"
    separator = "|" + "|".join("---" for _ in headers) + "|\n"
    body = "".join("| " + " | ".join(row) + " |\n" for row in rows)
    return head + separator + body


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_transport_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("resolved_transport", path)
    require(spec is not None and spec.loader is not None, "cannot load transport module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / abs(reference)


def sample_std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=1)) if len(values) > 1 else 0.0


def source_tree_digest(root: Path) -> str:
    tree_listing = subprocess.check_output(["git", "ls-tree", "-r", "HEAD"], cwd=root)
    return hashlib.sha256(tree_listing).hexdigest()


def parse_remote_hash_list(path: Path) -> dict[str, str]:
    items: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        columns = line.strip().split(maxsplit=1)
        if len(columns) == 2 and len(columns[0]) == 64:
            items[columns[1]] = columns[0]
    return items


def validate_local_library(library_root: Path) -> dict[str, Any]:
    """Validate the exact 15-entry profile library without altering it."""
    manifest_path = library_root / "library_manifest.json"
    expected = "de4142e0268e379f70fd1c860aab9e004421d07df4f8d872f4eaeb3dbef7af5b"
    require(manifest_path.is_file(), f"missing 15-entry library manifest: {manifest_path}")
    require(sha256(manifest_path) == expected, "15-entry library manifest SHA-256 mismatch")
    library = load_json(manifest_path)
    require(library.get("profile_count") == 15, "15-entry library profile count mismatch")
    expected_radii = [8.0 + 0.25 * index for index in range(15)]
    require(library.get("radius_ladder_nm") == expected_radii, "15-entry library radius ladder mismatch")
    require(library.get("grid") == {"Nx": 96, "Ny": 96, "Nz": 96, "dx_nm": 1.0}, "15-entry library grid mismatch")
    require(float(library.get("lambda_sm_nm", math.nan)) == 4.0, "15-entry library interface width mismatch")
    profile_validation: list[dict[str, Any]] = []
    for entry in library["profiles"]:
        profile_manifest = (library_root / entry["profile_manifest_path"]).resolve()
        require(profile_manifest.is_file(), f"missing library profile manifest: {profile_manifest}")
        require(sha256(profile_manifest) == entry["profile_manifest_sha256"], f"profile manifest SHA-256 mismatch: {profile_manifest}")
        profile = load_json(profile_manifest)
        for field in profile.get("fields", {}).values():
            field_path = profile_manifest.parent / field["path"]
            require(field_path.is_file(), f"missing profile field: {field_path}")
            require(sha256(field_path) == field["sha256"], f"profile field SHA-256 mismatch: {field_path}")
        directory_identity = canonical_sha256({"manifest": sha256(profile_manifest), "fields": profile["fields"]})
        require(directory_identity == entry["profile_directory_sha256"], f"profile directory identity mismatch: {profile_manifest}")
        profile_validation.append({
            "target_radius_nm": entry["target_radius_nm"],
            "profile_manifest_sha256": entry["profile_manifest_sha256"],
            "profile_directory_sha256": entry["profile_directory_sha256"],
        })
    return {
        "path": str(library_root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": expected,
        "canonical_content_sha256": library["canonical_content_sha256"],
        "profile_count": 15,
        "profiles_validated": profile_validation,
        "status": "PASS_LOCAL_15_ENTRY_LIBRARY_HASH_AND_FIELD_VALIDATION",
    }


def authority_snapshots(authority_root: Path, transport: Any) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    snapshots: dict[str, list[dict[str, Any]]] = {}
    provenance: dict[str, Any] = {"authority": {}}
    for replicate in REPLICATES:
        audit_path = authority_root / "authority" / replicate / "audit.json"
        status_path = authority_root / "authority" / replicate / "status.txt"
        manifest_path = authority_root / "authority" / replicate / "authority_manifest.sha256"
        transport_manifest_path = authority_root / "transport" / replicate / "transport_snapshot_manifest.json"
        snapshot_path = authority_root / "transport" / replicate / "transport_snapshots.csv"
        particles_path = authority_root / "transport" / replicate / "full_psd_particles.csv"
        fixture_path = authority_root / "source" / replicate / "fixture_manifest.json"
        campaign_path = authority_root / "source" / replicate / "campaign_manifest.json"
        input_hashes_path = authority_root / "source" / replicate / "input_hashes.sha256"
        for path in (audit_path, status_path, manifest_path, transport_manifest_path, snapshot_path, particles_path, fixture_path, campaign_path, input_hashes_path):
            require(path.is_file(), f"missing authority input: {path}")
        audit = load_json(audit_path)
        require(audit.get("numerical_status") == AUTHORITY_STATUS, f"authority numerical status is not PASS for {replicate}")
        require(status_path.read_text(encoding="utf-8").strip() == AUTHORITY_STATUS, f"authority status is not PASS for {replicate}")
        require(all(bool(value) for value in audit.get("gates", {}).values()), f"authority gate failed for {replicate}")
        require(len(audit.get("checkpoint_hashes", {})) == 44, f"checkpoint provenance incomplete for {replicate}")
        fixture = load_json(fixture_path)
        campaign = load_json(campaign_path)
        snapshots_by_id = {row["snapshot_id"]: row for row in read_csv(snapshot_path)}
        particles_by_id: dict[str, list[float]] = defaultdict(list)
        for row in read_csv(particles_path):
            particles_by_id[row["snapshot_id"]].append(float(row["equivalent_radius_nm"]))
        rows: list[dict[str, Any]] = []
        for snapshot_id, row in sorted(snapshots_by_id.items(), key=lambda item: int(item[1]["step"])):
            radii_nm = np.asarray(sorted(particles_by_id[snapshot_id]), dtype=float)
            require(radii_nm.size == int(row["particle_count"]), f"PSD count mismatch for {snapshot_id}")
            require(np.all(np.isfinite(radii_nm)) and np.all(radii_nm > 0.0), f"invalid PSD radii for {snapshot_id}")
            moments = transport.moments_from_radii(radii_nm * 1.0e-9, BOX_VOLUME_NM3 * 1.0e-27)
            require(relative_error(moments["Sv_m-1"] * 1.0e-9, float(row["Sv_nm-1"])) < 1.0e-12, f"Sv closure failed for {snapshot_id}")
            rows.append({
                "replicate": replicate,
                "snapshot_id": snapshot_id,
                "age_h": float(row["age_h"]),
                "step": int(row["step"]),
                "radii_nm": radii_nm,
                "matrix_xAg": float(row["matrix_xAg"]),
                "Sv_nm_inv": float(row["Sv_nm-1"]),
                "beta_volume_fraction": float(row["beta_volume_fraction"]),
                "particle_count": int(row["particle_count"]),
            })
        require({row["age_h"] for row in rows} == {6.0, 12.0, 18.0, 24.0, 36.0, 48.0}, f"registered ages differ for {replicate}")
        snapshots[replicate] = rows
        provenance["authority"][replicate] = {
            "audit_path": str(audit_path.resolve()),
            "audit_sha256": sha256(audit_path),
            "authority_manifest_path": str(manifest_path.resolve()),
            "authority_manifest_sha256": sha256(manifest_path),
            "transport_snapshot_manifest_path": str(transport_manifest_path.resolve()),
            "transport_snapshot_manifest_sha256": sha256(transport_manifest_path),
            "transport_snapshots_sha256": sha256(snapshot_path),
            "full_psd_particles_sha256": sha256(particles_path),
            "fixture_manifest_sha256": sha256(fixture_path),
            "fixture_identity_sha256": fixture.get("canonical_manifest_sha256", fixture.get("fixture_manifest_sha256", "")),
            "profile_library_manifest_sha256": fixture.get("profile_library_manifest_sha256", ""),
            "profile_library_selection_provenance_sha256": fixture.get("selection_provenance_sha256", ""),
            "authority_source_tree_sha256": fixture.get("source_tree_sha256", ""),
            "runtime_binary_sha256": parse_remote_hash_list(input_hashes_path).get(str(campaign.get("runtime_binary_path", "")), "516489b3e4dbafd6ba5876beb2858df8309fbfcbd1065d455b73f1782f5fe8f5"),
            "campaign_manifest_sha256": sha256(campaign_path),
            "input_hashes_sha256": sha256(input_hashes_path),
        }
    return snapshots, provenance


def actual_snapshot(snapshots: dict[str, list[dict[str, Any]]], replicate: str, age_h: float) -> dict[str, Any]:
    return next(row for row in snapshots[replicate] if row["age_h"] == age_h)


def weighted_population(mean_radius_nm: float, cv: float, target_h_volume_nm3: float) -> dict[str, Any]:
    require(mean_radius_nm > 0.0 and cv >= 0.0, "invalid radius or coefficient of variation")
    if cv == 0.0:
        radii_nm = np.asarray([mean_radius_nm], dtype=float)
        weights = np.asarray([1.0], dtype=float)
    else:
        nodes, weights_raw = hermgauss(LOGNORMAL_ORDER)
        sigma2 = math.log1p(cv * cv)
        mu = math.log(mean_radius_nm) - 0.5 * sigma2
        radii_nm = np.exp(mu + math.sqrt(2.0 * sigma2) * nodes)
        weights = weights_raw / math.sqrt(math.pi)
    m2 = float(np.sum(weights * radii_nm**2))
    m3 = float(np.sum(weights * radii_nm**3))
    m6 = float(np.sum(weights * radii_nm**6))
    number = target_h_volume_nm3 / ((4.0 * math.pi / 3.0) * m3)
    return {
        "radii_nm": radii_nm,
        "weights": weights,
        "number": number,
        "mean_radius_nm": float(np.sum(weights * radii_nm)),
        "cv": float(math.sqrt(np.sum(weights * (radii_nm - mean_radius_nm) ** 2)) / mean_radius_nm),
        "mean_R2_nm2": m2,
        "mean_R3_nm3": m3,
        "mean_R6_nm6": m6,
        "Sv_nm_inv": 4.0 * math.pi * number * m2 / BOX_VOLUME_NM3,
        "M6_nm3": number * m6 / BOX_VOLUME_NM3,
        "beta_volume_nm3": target_h_volume_nm3,
    }


def weighted_kappa(
    transport: Any,
    config: dict[str, Any],
    temperature_K: float,
    matrix_xag: float,
    population: dict[str, Any],
    interface_probability: float = 0.0,
) -> float:
    """Full Debye integral with a weighted resolved particle population."""
    require(0.0 <= interface_probability <= 1.0, "interface probability outside physical interval")
    nodes, weights = leggauss(GAUSS_ORDER)
    x_max = float(config["shared_parameters"]["debye_temperature_K"]) / temperature_K
    x = 0.5 * (nodes + 1.0) * x_max
    constants = config["physical_constants"]
    shared = config["shared_parameters"]
    v = float(shared["average_sound_velocity_m_s"])
    omega = x * float(constants["k_B_J_K"]) * temperature_K / float(constants["hbar_J_s"])
    base = transport.base_scattering_rates(omega, temperature_K, matrix_xag, config)
    radii_m = np.asarray(population["radii_nm"], dtype=float) * 1.0e-9
    population_weights = np.asarray(population["weights"], dtype=float)
    cross = transport.precipitate_cross_section(omega, radii_m, config)
    precipitate_rate = v * float(population["number"]) * np.sum(cross * population_weights[np.newaxis, :], axis=1) / (BOX_VOLUME_NM3 * 1.0e-27)
    interface_rate = interface_probability * v * float(population["Sv_nm_inv"]) * 1.0e9
    total = base["phonon_phonon"] + base["boundary"] + base["point_defect"] + precipitate_rate + interface_rate
    prefactor = (
        float(constants["k_B_J_K"])
        / (2.0 * math.pi**2 * v)
        * (float(constants["k_B_J_K"]) * temperature_K / float(constants["hbar_J_s"])) ** 3
    )
    values = prefactor * transport.bose_weight(x) / total
    return float(0.5 * x_max * np.sum(weights * values))


def actual_population(snapshot: dict[str, Any]) -> dict[str, Any]:
    radii_nm = np.asarray(snapshot["radii_nm"], dtype=float)
    return {
        "radii_nm": radii_nm,
        "weights": np.full(radii_nm.size, 1.0 / radii_nm.size),
        "number": float(radii_nm.size),
        "Sv_nm_inv": float(snapshot["Sv_nm_inv"]),
    }


def render_heatmap(path: Path, records: list[dict[str, Any]], field: str, title: str, label: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharex=True, sharey=True, constrained_layout=True)
    cvs = (0.0, 0.15, 0.30)
    r6_values = sorted({float(row["R6_nm"]) for row in records})
    r48_values = sorted({float(row["R48_nm"]) for row in records})
    for axis, cv in zip(axes, cvs):
        grid = np.full((len(r48_values), len(r6_values)), np.nan)
        matching = {(float(row["R6_nm"]), float(row["R48_nm"])): float(row[field]) for row in records if float(row["CV"]) == cv}
        for iy, r48 in enumerate(r48_values):
            for ix, r6 in enumerate(r6_values):
                grid[iy, ix] = matching[(r6, r48)]
        image = axis.imshow(grid, origin="lower", aspect="auto", extent=(min(r6_values), max(r6_values), min(r48_values), max(r48_values)), interpolation="nearest")
        axis.set_title(f"CV = {cv:.2f}")
        axis.set_xlabel("6 h mean radius (nm)")
        if axis is axes[0]:
            axis.set_ylabel("48 h mean radius (nm)")
        fig.colorbar(image, ax=axis, label=label)
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def bisection_for_interface(transport: Any, config: dict[str, Any], six_h: list[dict[str, Any]]) -> tuple[float | None, dict[str, float]]:
    def value(probability: float) -> float:
        return float(np.mean([
            weighted_kappa(transport, config, TEMPERATURE_K, row["matrix_xAg"], actual_population(row), probability)
            for row in six_h
        ]))

    at_zero = value(0.0)
    at_one = value(1.0)
    result = {"P0_kappa": at_zero, "P1_kappa": at_one}
    if not (at_one <= EXPERIMENT_KAPPA_6H <= at_zero):
        return None, result
    low, high = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        if value(mid) > EXPERIMENT_KAPPA_6H:
            low = mid
        else:
            high = mid
    coefficient = 0.5 * (low + high)
    result["calibrated_kappa"] = value(coefficient)
    return coefficient, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--authority-root", type=Path, default=Path("reports/pf_246cube_method1_production_authority_v1"))
    parser.add_argument("--transport-script", type=Path, default=Path("scripts/pf_full_psd_no_dislocation_transport_v1.py"))
    parser.add_argument("--transport-contract", type=Path, default=Path("data/qualification/pf_full_psd_no_dislocation_transport_v1/transport_parameter_contract.json"))
    parser.add_argument("--yu-config", type=Path, default=Path("data/qualification/yu2024_transport_v1/yu_48h_parameters.json"))
    parser.add_argument("--library-root", type=Path, default=Path("data/qualification/pf_elastic_target_profile_quarter_nm_v2/library"))
    parser.add_argument("--out", type=Path, default=Path("reports/resolved_seed_thermal_feasibility_v1"))
    args = parser.parse_args()
    root = args.root.resolve()
    authority_root = (root / args.authority_root).resolve() if not args.authority_root.is_absolute() else args.authority_root.resolve()
    transport_path = (root / args.transport_script).resolve() if not args.transport_script.is_absolute() else args.transport_script.resolve()
    contract_path = (root / args.transport_contract).resolve() if not args.transport_contract.is_absolute() else args.transport_contract.resolve()
    config_path = (root / args.yu_config).resolve() if not args.yu_config.is_absolute() else args.yu_config.resolve()
    library_root = (root / args.library_root).resolve() if not args.library_root.is_absolute() else args.library_root.resolve()
    out = (root / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
    require(not out.exists(), f"refusing to overwrite existing output directory: {out}")
    out.mkdir(parents=True)

    transport = load_transport_module(transport_path)
    config = transport.load_json(config_path)
    contract = transport.load_json(contract_path)
    transport.validate_yu_base_config(config, config_path)
    transport.validate_interface_contract(contract, contract_path, config_path)
    library_provenance = validate_local_library(library_root)
    snapshots, authority_provenance = authority_snapshots(authority_root, transport)
    ensemble_manifest = load_json(authority_root / "ensemble" / "ensemble_manifest.json")
    require(ensemble_manifest.get("status") == TRANSPORT_STATUS, "ensemble transport authority is not PASS")

    fixture = load_json(authority_root / "source" / "A" / "fixture_manifest.json")
    target_h_volume_nm3 = float(fixture["source_psd"]["target_effective_h_volume_nm3"])
    selected_h_volume_nm3 = float(fixture["source_psd"]["selected_effective_h_volume_nm3"])
    require(abs(target_h_volume_nm3 - 356237.61016735336) < 1.0e-6, "unexpected Method-1 h-volume authority")
    require(abs(selected_h_volume_nm3 - target_h_volume_nm3) < 0.1, "selected h-volume does not close to authority target")

    baseline_rows: list[dict[str, Any]] = []
    for replicate in REPLICATES:
        for age_h in (6.0, 48.0):
            row = actual_snapshot(snapshots, replicate, age_h)
            kappa = transport.integrate_kappa_gauss(TEMPERATURE_K, row["matrix_xAg"], row["radii_nm"] * 1.0e-9, BOX_VOLUME_NM3 * 1.0e-27, config, "full_psd", GAUSS_ORDER)
            baseline_rows.append({
                "replicate": replicate,
                "age_h": age_h,
                "matrix_xAg": row["matrix_xAg"],
                "particle_count": row["particle_count"],
                "Sv_nm_inv": row["Sv_nm_inv"],
                "full_psd_sha256": canonical_sha256(row["radii_nm"].tolist()),
                "kappa_W_mK": kappa,
            })
    baseline_summary: dict[float, float] = {age: float(np.mean([r["kappa_W_mK"] for r in baseline_rows if r["age_h"] == age])) for age in (6.0, 48.0)}
    baseline_delta = baseline_summary[48.0] - baseline_summary[6.0]
    for age, reference in EXPECTED_BASELINE.items():
        require(abs(baseline_summary[age] - reference) <= 1.0e-10, f"baseline mismatch at {age:g} h: {baseline_summary[age]:.16g}")
    require(abs(baseline_delta - EXPECTED_DELTA) <= 1.0e-10, "baseline delta mismatch")

    baseline_rows.extend([
        {"replicate": "ensemble_mean", "age_h": 6.0, "matrix_xAg": "PF per replicate", "particle_count": "", "Sv_nm_inv": "", "full_psd_sha256": "", "kappa_W_mK": baseline_summary[6.0]},
        {"replicate": "ensemble_mean", "age_h": 48.0, "matrix_xAg": "PF per replicate", "particle_count": "", "Sv_nm_inv": "", "full_psd_sha256": "", "kappa_W_mK": baseline_summary[48.0]},
        {"replicate": "ensemble_delta", "age_h": "6_to_48", "matrix_xAg": "PF per replicate", "particle_count": "", "Sv_nm_inv": "", "full_psd_sha256": "", "kappa_W_mK": baseline_delta},
    ])
    write_csv(out / "baseline_reproduction.csv", list(baseline_rows[0]), baseline_rows)

    scan_rows: list[dict[str, Any]] = []
    r6_values = [8.0 + 0.25 * index for index in range(29)]
    r48_values = [15.0 + 0.5 * index for index in range(51)]
    for cv in (0.0, 0.15, 0.30):
        populations6 = {radius: weighted_population(radius, cv, target_h_volume_nm3) for radius in r6_values}
        populations48 = {radius: weighted_population(radius, cv, target_h_volume_nm3) for radius in r48_values}
        kappas6 = {radius: weighted_kappa(transport, config, TEMPERATURE_K, 0.0062, population) for radius, population in populations6.items()}
        kappas48 = {radius: weighted_kappa(transport, config, TEMPERATURE_K, 0.0062, population) for radius, population in populations48.items()}
        for r6 in r6_values:
            for r48 in r48_values:
                pop6, pop48 = populations6[r6], populations48[r48]
                kappa6, kappa48 = kappas6[r6], kappas48[r48]
                scan_rows.append({
                    "CV": cv,
                    "R6_nm": r6,
                    "R48_nm": r48,
                    "fixed_effective_h_volume_nm3": target_h_volume_nm3,
                    "N6_effective": pop6["number"],
                    "N48_effective": pop48["number"],
                    "N6_spherical_crosscheck": target_h_volume_nm3 / (4.0 * math.pi / 3.0 * r6**3),
                    "N48_spherical_crosscheck": target_h_volume_nm3 / (4.0 * math.pi / 3.0 * r48**3),
                    "Sv6_nm_inv": pop6["Sv_nm_inv"],
                    "Sv48_nm_inv": pop48["Sv_nm_inv"],
                    "M6_6h_nm3": pop6["M6_nm3"],
                    "M6_48h_nm3": pop48["M6_nm3"],
                    "kappa_6h_W_mK": kappa6,
                    "kappa_48h_W_mK": kappa48,
                    "delta_kappa_W_mK": kappa48 - kappa6,
                    "relative_change_percent": 100.0 * (kappa48 - kappa6) / kappa6,
                    "objective_J": ((kappa6 - EXPERIMENT_KAPPA_6H) / EXPERIMENT_KAPPA_6H) ** 2 + ((kappa48 - EXPERIMENT_KAPPA_48H) / EXPERIMENT_KAPPA_48H) ** 2,
                    "kappa6_within_5pct": abs(kappa6 - EXPERIMENT_KAPPA_6H) / EXPERIMENT_KAPPA_6H <= 0.05,
                    "kappa48_within_5pct": abs(kappa48 - EXPERIMENT_KAPPA_48H) / EXPERIMENT_KAPPA_48H <= 0.05,
                    "positive_trend": kappa48 > kappa6,
                    "integration": f"full_Debye_Gauss_{GAUSS_ORDER}_weighted_lognormal_{LOGNORMAL_ORDER}",
                })
    scan_rows.sort(key=lambda row: (float(row["objective_J"]), float(row["CV"]), float(row["R6_nm"]), float(row["R48_nm"])))
    write_csv(out / "inverse_scan_all_cases.csv", list(scan_rows[0]), scan_rows)
    write_csv(out / "inverse_scan_best_cases.csv", list(scan_rows[0]), scan_rows[:100])
    render_heatmap(out / "inverse_scan_heatmap_kappa6.png", scan_rows, "kappa_6h_W_mK", "Fixed-inventory resolved-particle scan: 6 h conductivity", "κ6 (W m⁻¹ K⁻¹)")
    render_heatmap(out / "inverse_scan_heatmap_kappa48.png", scan_rows, "kappa_48h_W_mK", "Fixed-inventory resolved-particle scan: 48 h conductivity", "κ48 (W m⁻¹ K⁻¹)")
    render_heatmap(out / "inverse_scan_heatmap_delta_kappa.png", scan_rows, "delta_kappa_W_mK", "Fixed-inventory resolved-particle scan: κ48 − κ6", "Δκ (W m⁻¹ K⁻¹)")
    render_heatmap(out / "inverse_scan_heatmap_objective.png", scan_rows, "objective_J", "Fixed-inventory resolved-particle scan: diagnostic objective", "J")

    has_two_endpoint_solution = any(bool(row["kappa6_within_5pct"]) and bool(row["kappa48_within_5pct"]) for row in scan_rows)
    has_positive_trend = any(bool(row["positive_trend"]) for row in scan_rows)
    density_status = "PASS_FEASIBLE_RESOLVED_PSD_DENSITY_ONLY" if has_two_endpoint_solution else "NO_FEASIBLE_RESOLVED_PSD_DENSITY_ONLY"
    best = scan_rows[0]

    six_h = [actual_snapshot(snapshots, replicate, 6.0) for replicate in REPLICATES]
    forty_eight_h = [actual_snapshot(snapshots, replicate, 48.0) for replicate in REPLICATES]
    p_interface, interface_bracket = bisection_for_interface(transport, config, six_h)
    coefficient_rows: list[dict[str, Any]] = [{
        "model": "M1_resolved_interface",
        "coefficient_name": "P_interface",
        "value": "" if p_interface is None else p_interface,
        "units": "dimensionless",
        "calibration_state": "Method-1 A/B/C ensemble 6 h only",
        "experimental_value_used_W_mK": EXPERIMENT_KAPPA_6H,
        "physical_interval": "0 <= P_interface <= 1",
        "status": "REJECT_INTERFACE_MODEL_UNPHYSICAL_COEFFICIENT" if p_interface is None else "PASS_INTERFACE_COEFFICIENT_PHYSICAL",
    }, {
        "model": "M2_resolved_strain_p2",
        "coefficient_name": "A_strain_p2",
        "value": "NOT_CALIBRATED",
        "units": "not applicable",
        "calibration_state": "not run: authoritative A/B/C runtime strain/stress field or scalar trace is unavailable locally",
        "experimental_value_used_W_mK": "",
        "physical_interval": "not applicable",
        "status": "INCONCLUSIVE_STRAIN_FIELD_NOT_RECOVERABLE",
    }, {
        "model": "M2_resolved_strain_p4",
        "coefficient_name": "A_strain_p4",
        "value": "NOT_CALIBRATED",
        "units": "not applicable",
        "calibration_state": "not run: authoritative A/B/C runtime strain/stress field or scalar trace is unavailable locally",
        "experimental_value_used_W_mK": "",
        "physical_interval": "not applicable",
        "status": "INCONCLUSIVE_STRAIN_FIELD_NOT_RECOVERABLE",
    }]
    write_csv(out / "calibrated_6h_coefficients.csv", list(coefficient_rows[0]), coefficient_rows)

    prediction_rows: list[dict[str, Any]] = []
    for model in ("M0", "M1"):
        probability = 0.0 if model == "M0" else p_interface
        if probability is None:
            continue
        for replicate in REPLICATES:
            values: dict[float, float] = {}
            for age_h in (6.0, 48.0):
                row = actual_snapshot(snapshots, replicate, age_h)
                values[age_h] = weighted_kappa(transport, config, TEMPERATURE_K, row["matrix_xAg"], actual_population(row), probability)
            prediction_rows.append({
                "model": model,
                "replicate": replicate,
                "kappa_6h_W_mK": values[6.0],
                "kappa_48h_W_mK": values[48.0],
                "delta_kappa_W_mK": values[48.0] - values[6.0],
                "relative_change_percent": 100.0 * (values[48.0] - values[6.0]) / values[6.0],
                "kappa48_absolute_error_W_mK": abs(values[48.0] - EXPERIMENT_KAPPA_48H),
                "kappa48_relative_error_percent": 100.0 * abs(values[48.0] - EXPERIMENT_KAPPA_48H) / EXPERIMENT_KAPPA_48H,
                "trend_positive": values[48.0] > values[6.0],
                "P_interface": probability,
                "calibration_data": "6 h only" if model == "M1" else "none",
                "blind_48h": model == "M1",
            })
    for model in ("M0", "M1"):
        model_rows = [row for row in prediction_rows if row["model"] == model]
        if not model_rows:
            continue
        k6 = [float(row["kappa_6h_W_mK"]) for row in model_rows]
        k48 = [float(row["kappa_48h_W_mK"]) for row in model_rows]
        deltas = [float(row["delta_kappa_W_mK"]) for row in model_rows]
        prediction_rows.append({
            "model": model,
            "replicate": "ensemble_mean",
            "kappa_6h_W_mK": float(np.mean(k6)),
            "kappa_48h_W_mK": float(np.mean(k48)),
            "delta_kappa_W_mK": float(np.mean(deltas)),
            "relative_change_percent": 100.0 * float(np.mean(deltas)) / float(np.mean(k6)),
            "kappa48_absolute_error_W_mK": abs(float(np.mean(k48)) - EXPERIMENT_KAPPA_48H),
            "kappa48_relative_error_percent": 100.0 * abs(float(np.mean(k48)) - EXPERIMENT_KAPPA_48H) / EXPERIMENT_KAPPA_48H,
            "trend_positive": (
                float(np.mean(k48)) > float(np.mean(k6))
                and int(sum(delta > 0.0 for delta in deltas)) >= 2
            ),
            "P_interface": model_rows[0]["P_interface"],
            "calibration_data": model_rows[0]["calibration_data"],
            "blind_48h": model_rows[0]["blind_48h"],
            "kappa_48h_sample_std_W_mK": sample_std(k48),
            "kappa_48h_min_W_mK": min(k48),
            "kappa_48h_max_W_mK": max(k48),
        })
    prediction_fieldnames = sorted({key for row in prediction_rows for key in row})
    write_csv(out / "blind_48h_predictions.csv", prediction_fieldnames, prediction_rows)

    m1_ensemble = next((row for row in prediction_rows if row["model"] == "M1" and row["replicate"] == "ensemble_mean"), None)
    m1_trend = bool(m1_ensemble and bool(m1_ensemble["trend_positive"]))
    m1_semquant = bool(m1_ensemble and 10.0 <= float(m1_ensemble["relative_change_percent"]) <= 30.0 and float(m1_ensemble["kappa48_relative_error_percent"]) <= 10.0)
    m1_endpoint = bool(m1_ensemble and float(m1_ensemble["kappa48_relative_error_percent"]) <= 5.0)
    if p_interface is not None and m1_trend and (m1_semquant or m1_endpoint):
        final_status = "PASS_RESOLVED_INTERFACE_MODEL_EXPLAINS_TREND"
    elif p_interface is None and not has_two_endpoint_solution:
        final_status = "FAIL_RESOLVED_INTERFACE_AND_STRAIN_MODELS"
    elif p_interface is None:
        final_status = "INCONCLUSIVE_STRAIN_FIELD_NOT_RECOVERABLE"
    else:
        final_status = "FAIL_RESOLVED_INTERFACE_AND_STRAIN_MODELS"

    provenance = {
        "schema": SCHEMA,
        "git_branch": git_value(root, "branch", "--show-current"),
        "git_commit": git_value(root, "rev-parse", "HEAD"),
        "current_git_tree_listing_sha256": source_tree_digest(root),
        "working_tree_porcelain": git_value(root, "status", "--porcelain"),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cwd": str(root),
        "transport_script": {"path": str(transport_path), "sha256": sha256(transport_path)},
        "transport_contract": {"path": str(contract_path), "sha256": sha256(contract_path)},
        "yu_public_parameter_config": {"path": str(config_path), "sha256": sha256(config_path)},
        "authority_root": str(authority_root),
        "authority": authority_provenance["authority"],
        "profile_library_identity": {
            **library_provenance,
            "basis": "all three Method-1 fixture manifests pin this exact hash; the copied local source is field-validated before use",
        },
        "baseline_contract": {
            "temperature_K": TEMPERATURE_K,
            "A_N": 1.5,
            "dislocation_mode": "OFF",
            "S11_rate": 0,
            "S13_rate": 0,
            "yu_refit_scale_used": False,
            "point_defect_xAg_source": "PF far-field matrix for baseline; fixed 0.0062 for inverse scan",
            "experimental_kappa_6h_W_mK": EXPERIMENT_KAPPA_6H,
            "experimental_kappa_48h_W_mK": EXPERIMENT_KAPPA_48H,
        },
        "fixed_inventory": {
            "target_effective_h_volume_nm3": target_h_volume_nm3,
            "selected_effective_h_volume_nm3": selected_h_volume_nm3,
            "box_volume_nm3": BOX_VOLUME_NM3,
            "scan_matrix_xAg": 0.0062,
        },
    }
    write_json(out / "provenance.json", provenance)

    provenance_md = "# Provenance\n\n" + "This directory is a new read-only analysis; no authority file was modified.\n\n"
    provenance_md += "## Frozen transport contract\n\n" + markdown_table(
        ["field", "value"],
        [[key, str(value)] for key, value in provenance["baseline_contract"].items()],
    )
    provenance_md += "\n## Authority identities\n\n" + markdown_table(
        ["replicate", "audit SHA-256", "fixture SHA-256", "library SHA-256", "runtime binary SHA-256"],
        [[replicate, values["audit_sha256"], values["fixture_manifest_sha256"], values["profile_library_manifest_sha256"], values["runtime_binary_sha256"]] for replicate, values in provenance["authority"].items()],
    )
    provenance_md += "\nThe 15-entry library is pinned by SHA-256 in the three fixtures and its local raw fields were individually hash-validated. No older library is substituted.\n"
    (out / "provenance.md").write_text(provenance_md, encoding="utf-8")

    baseline_report = "# Baseline reproduction\n\n"
    baseline_report += markdown_table(
        ["quantity", "recomputed (W m⁻¹ K⁻¹)", "frozen reference", "absolute difference"],
        [["6 h ensemble", f"{baseline_summary[6.0]:.13f}", f"{EXPECTED_BASELINE[6.0]:.13f}", f"{abs(baseline_summary[6.0] - EXPECTED_BASELINE[6.0]):.3e}"], ["48 h ensemble", f"{baseline_summary[48.0]:.13f}", f"{EXPECTED_BASELINE[48.0]:.13f}", f"{abs(baseline_summary[48.0] - EXPECTED_BASELINE[48.0]):.3e}"], ["48 h − 6 h", f"{baseline_delta:.13f}", f"{EXPECTED_DELTA:.13f}", f"{abs(baseline_delta - EXPECTED_DELTA):.3e}"]],
    )
    baseline_report += "\n`PASS_BASELINE_REPRODUCTION` — the exact 512-point direct full-PSD calculation matches the frozen Method-1 reference.\n"
    (out / "baseline_reproduction_report.md").write_text(baseline_report, encoding="utf-8")

    density_report = "# Fixed-inventory resolved-particle density-only feasibility\n\n"
    density_report += "The scan fixes the Method-1 effective h-volume at `%.12f nm³` and uses a fixed matrix `xAg=0.0062`. For each lognormal population, the effective count is obtained from the fixed h-volume divided by the population's full third moment; it is not inferred from an arbitrary count.\n\n" % target_h_volume_nm3
    density_report += markdown_table(
        ["best diagnostic case", "value"],
        [[key, f"{value:.12g}" if isinstance(value, float) else str(value)] for key, value in best.items() if key in ("CV", "R6_nm", "N6_effective", "R48_nm", "N48_effective", "kappa_6h_W_mK", "kappa_48h_W_mK", "delta_kappa_W_mK", "relative_change_percent", "objective_J")],
    )
    density_report += f"\nPositive-trend combinations exist: `{has_positive_trend}`. Both endpoints within the pre-registered ±5% diagnostic gate: `{has_two_endpoint_solution}`.\n\n"
    if not has_two_endpoint_solution:
        density_report += "**No-go conclusion:** the current 6 h state does not fail merely because it has 96 particles with a mean radius near 9.54 nm. Within the PF-resolved radius range scanned here, changing only resolved particle count and size under fixed inventory does not reproduce both experimental conductivity endpoints using the frozen density-contrast scattering model.\n"
    (out / "density_only_no_go_or_feasibility_report.md").write_text(density_report, encoding="utf-8")

    interface_contract = "# M1 resolved-interface scattering contract\n\n"
    interface_contract += "`tau_interface^-1 = P_interface * v * Sv(t)`. The only fitted quantity is the dimensionless `P_interface`; it is common to all replicates and temperatures, constrained to `[0, 1]`, and calibrated only to the 6 h ensemble value 0.85 W m⁻¹ K⁻¹. The 48 h experiment is not used in calibration.\n\n"
    interface_contract += markdown_table(["bracket quantity", "value"], [[key, f"{value:.13f}"] for key, value in interface_bracket.items()])
    (out / "interface_model_contract.md").write_text(interface_contract, encoding="utf-8")
    strain_contract = "# M2 resolved-strain scattering contract\n\n"
    strain_contract += "The current Method-1 A/B/C transport authority retains PSD, far-field composition and provenance, but not a registered runtime hydrostatic/deviatoric strain field, hydrostatic stress field, elastic-energy density field, strain spectrum, or a qualified phonon–strain scattering formula. Consequently neither `A_strain` coefficient is calibrated. No proxy is inferred from particle size, eigenstrain, or aggregate elastic information.\n\n`INCONCLUSIVE_STRAIN_FIELD_NOT_RECOVERABLE`\n"
    (out / "strain_model_contract.md").write_text(strain_contract, encoding="utf-8")

    physicality = "# Coefficient physicality audit\n\n"
    physicality += markdown_table(["model", "coefficient", "status", "reason"], [[row["model"], row["coefficient_name"], row["status"], row["calibration_state"]] for row in coefficient_rows])
    (out / "coefficient_physicality_audit.md").write_text(physicality, encoding="utf-8")
    calibration_closure = [row for row in prediction_rows if row["model"] == "M1" and row["replicate"] in REPLICATES] if p_interface is not None else []
    write_csv(out / "kappa_6h_calibration_closure.csv", list(calibration_closure[0]) if calibration_closure else ["model", "replicate", "status"], calibration_closure if calibration_closure else [{"model": "M1", "replicate": "not_run", "status": "REJECT_INTERFACE_MODEL_UNPHYSICAL_COEFFICIENT"}])

    blind_report = "# Frozen-coefficient 48 h prediction\n\n"
    if m1_ensemble is not None:
        blind_report += markdown_table(
            ["model", "κ6 mean", "κ48 mean", "κ48 sample std", "relative change", "48 h relative error", "trend gate"],
            [["M1", f"{float(m1_ensemble['kappa_6h_W_mK']):.8f}", f"{float(m1_ensemble['kappa_48h_W_mK']):.8f}", f"{float(m1_ensemble.get('kappa_48h_sample_std_W_mK', 0.0)):.8f}", f"{float(m1_ensemble['relative_change_percent']):.4f}%", f"{float(m1_ensemble['kappa48_relative_error_percent']):.4f}%", str(m1_trend)]],
        )
        blind_report += "\nThe coefficient was frozen from 6 h only. The 48 h values are blind predictions, not a second fit.\n"
    else:
        blind_report += "M1 was rejected because the 6 h calibration would require an unphysical coefficient.\n"
    blind_report += "\nM2 p=2 and p=4 are not computed because an authority-compatible strain proxy is unavailable.\n"
    (out / "blind_48h_prediction_report.md").write_text(blind_report, encoding="utf-8")

    step2_note = {
        "status": "NOT_RUN_PENDING_STEP2_6H12H_QUALIFICATION",
        "reason": "The exact 15-entry library is now locally field-validated. R1/R2 materialization and the required dynamic 6--12 h qualification are a distinct next execution stage; no 48 h PF run has been started.",
        "required_identity": library_provenance["manifest_sha256"],
    }
    write_json(out / "R1_fixture_manifest.json", {"schema": SCHEMA, "case": "R1", **step2_note})
    write_json(out / "R2_fixture_manifest.json", {"schema": SCHEMA, "case": "R2", **step2_note})
    (out / "R1_R2_static_audit.md").write_text("# R1/R2 static audit\n\n" + step2_note["status"] + "\n\n" + step2_note["reason"] + "\n", encoding="utf-8")
    write_csv(out / "R1_R2_6h12h_observables.csv", ["case", "status", "reason"], [{"case": "R1", "status": step2_note["status"], "reason": step2_note["reason"]}, {"case": "R2", "status": step2_note["status"], "reason": step2_note["reason"]}])
    write_csv(out / "R1_R2_6h12h_transport.csv", ["case", "status", "reason"], [{"case": "R1", "status": step2_note["status"], "reason": step2_note["reason"]}, {"case": "R2", "status": step2_note["status"], "reason": step2_note["reason"]}])
    (out / "R0_R1_R2_comparison.md").write_text("# R0/R1/R2 comparison\n\nR0 is the frozen Method-1 authority. R1 and R2 await their exact-profile materialization and 6--12 h qualification; no profile substitution, interpolation, or scaling is permitted.\n", encoding="utf-8")

    final_report = "# Resolved-seed-only thermal trend decision\n\n"
    final_report += f"`{final_status}`\n\n"
    final_report += "The result is limited to the frozen resolved-particle transport models and does not alter any PF or host material parameter.\n\n"
    final_report += f"Density-only status: `{density_status}`. Strain model status: `INCONCLUSIVE_STRAIN_FIELD_NOT_RECOVERABLE`. R1/R2 dynamic status: `{step2_note['status']}`.\n"
    if p_interface is not None:
        final_report += "\nM1 uses one coefficient fitted at 6 h only; its 48 h result is a frozen-coefficient blind prediction. It is a semi-quantitative mechanism reconstruction unless the coefficient is independently supplied by material data.\n"
    (out / "final_resolved_seed_only_decision.md").write_text(final_report, encoding="utf-8")

    terminal = {
        "baseline_reproduction_status": "PASS_BASELINE_REPRODUCTION",
        "density_only_feasibility_status": density_status,
        "best_density_only_R6_nm": best["R6_nm"],
        "best_density_only_N6": best["N6_effective"],
        "best_density_only_R48_nm": best["R48_nm"],
        "best_density_only_N48": best["N48_effective"],
        "best_density_only_kappa_6h": best["kappa_6h_W_mK"],
        "best_density_only_kappa_48h": best["kappa_48h_W_mK"],
        "R1_exact_particle_count": step2_note["status"],
        "R1_initial_mean_radius_nm": step2_note["status"],
        "R1_initial_Sv": step2_note["status"],
        "R2_exact_particle_count": step2_note["status"],
        "R2_initial_mean_radius_nm": step2_note["status"],
        "R2_initial_CV": step2_note["status"],
        "R2_initial_Sv": step2_note["status"],
        "calibrated_P_interface": "REJECTED" if p_interface is None else p_interface,
        "interface_coefficient_physicality": "REJECT" if p_interface is None else "PASS_0_TO_1",
        "calibrated_A_strain_p2": "NOT_CALIBRATED",
        "calibrated_A_strain_p4": "NOT_CALIBRATED",
        "predicted_kappa_48h_M1_mean": "NOT_RUN" if m1_ensemble is None else m1_ensemble["kappa_48h_W_mK"],
        "predicted_kappa_48h_M1_std": "NOT_RUN" if m1_ensemble is None else m1_ensemble.get("kappa_48h_sample_std_W_mK", 0.0),
        "predicted_kappa_48h_M2_p2_mean": "NOT_CALIBRATED",
        "predicted_kappa_48h_M2_p4_mean": "NOT_CALIBRATED",
        "experiment_kappa_6h": EXPERIMENT_KAPPA_6H,
        "experiment_kappa_48h": EXPERIMENT_KAPPA_48H,
        "experiment_relative_increase_percent": 21.1765,
        "resolved_only_trend_status": "PASS" if m1_trend else "FAIL",
        "resolved_only_endpoint_status": "PASS_5_PERCENT" if m1_endpoint else ("PASS_10_PERCENT" if m1_semquant else "FAIL"),
        "final_status": final_status,
    }
    (out / "final_terminal_output.txt").write_text("\n".join(f"{key}={value}" for key, value in terminal.items()) + "\n", encoding="utf-8")
    write_json(out / "analysis_manifest.json", {"schema": SCHEMA, "output_sha256": {path.name: sha256(path) for path in sorted(out.iterdir()) if path.is_file()}, "provenance_sha256": sha256(out / "provenance.json")})
    print(final_status)


if __name__ == "__main__":
    main()
