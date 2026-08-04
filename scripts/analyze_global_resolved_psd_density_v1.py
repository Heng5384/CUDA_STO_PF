#!/usr/bin/env python3
"""Read-only global resolved-PSD density-contrast feasibility audit.

This script deliberately leaves all PF trajectories untouched.  It uses the
frozen Method-1 A/B/C authority only to reproduce the 573.15 K full-PSD
baseline and to obtain the exact effective beta h-volume.  Radius populations
in the scans are continuous mathematical diagnostics: their count is an
equivalent count at fixed h-volume, not a claim that an unregistered radius is
already a constructible PF target profile.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
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


SCHEMA = "GLOBAL_RESOLVED_PSD_NO_GO_DENSITY_AUDIT_V1"
TEMPERATURE_K = 573.15
BOX_EDGE_NM = 246.0
BOX_VOLUME_NM3 = BOX_EDGE_NM**3
BOX_VOLUME_M3 = BOX_VOLUME_NM3 * 1.0e-27
TARGET_H_VOLUME_NM3 = 356237.61016735336
MATRIX_XAG = 0.0062
EXPERIMENT_K6 = 0.85
EXPERIMENT_K48 = 1.03
EXPERIMENT_DELTA = 0.18
EXPERIMENT_RELATIVE = 0.211765
FROZEN_BASELINE = {6.0: 1.2980916621210676, 48.0: 1.2948991542326225}
FROZEN_DELTA = -0.003192507888445162
REPLICATES = ("A", "B", "C")
GAUSS_ORDER = 512
HERMITE_ORDER = 96
CVS = (0.0, 0.15, 0.30, 0.50)


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
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    require(values, f"refusing to write empty CSV: {path}")
    keys = sorted({key for row in values for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def load_transport(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("global_resolved_transport", path)
    require(spec is not None and spec.loader is not None, "cannot load transport module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def source_tree_digest(root: Path) -> str:
    return hashlib.sha256(
        subprocess.check_output(["git", "ls-tree", "-r", "HEAD"], cwd=root)
    ).hexdigest()


def population_class(radius_nm: float) -> str:
    if radius_nm < 8.0:
        return "NON_PF_RESOLVED_MATHEMATICAL_DIAGNOSTIC"
    if radius_nm <= 11.5:
        return "CURRENT_PROFILE_LIBRARY_REALIZABLE"
    return "RESOLVED_CONTINUUM_TRANSPORT_DIAGNOSTIC_NOT_CURRENTLY_PROFILE_LIBRARY_QUALIFIED"


def piecewise_radii() -> list[float]:
    groups = (
        np.arange(0.50, 8.00, 0.05),
        np.arange(8.00, 30.00 + 1.0e-12, 0.10),
        np.arange(30.25, 60.00 + 1.0e-12, 0.25),
        np.arange(61.0, 120.00 + 1.0e-12, 1.0),
        np.arange(122.0, 200.00 + 1.0e-12, 2.0),
    )
    return sorted({round(float(item), 10) for group in groups for item in group})


def lognormal_population(mean_radius_nm: float, cv: float) -> dict[str, Any]:
    require(mean_radius_nm > 0.0 and cv >= 0.0, "invalid mean radius or CV")
    if cv == 0.0:
        radii = np.asarray([mean_radius_nm])
        weights = np.asarray([1.0])
    else:
        node, raw_weight = hermgauss(HERMITE_ORDER)
        sigma2 = math.log1p(cv * cv)
        mu = math.log(mean_radius_nm) - 0.5 * sigma2
        radii = np.exp(mu + math.sqrt(2.0 * sigma2) * node)
        weights = raw_weight / math.sqrt(math.pi)
    mean_r = float(np.sum(weights * radii))
    variance = float(np.sum(weights * (radii - mean_r) ** 2))
    mean_r2 = float(np.sum(weights * radii**2))
    mean_r3 = float(np.sum(weights * radii**3))
    mean_r6 = float(np.sum(weights * radii**6))
    number = TARGET_H_VOLUME_NM3 / ((4.0 * math.pi / 3.0) * mean_r3)
    below = float(np.sum(weights[radii < 0.5]))
    above = float(np.sum(weights[radii > 200.0]))
    return {
        "kind": "lognormal",
        "radii_nm": radii,
        "weights": weights,
        "mean_radius_nm": mean_r,
        "cv": math.sqrt(variance) / mean_r,
        "mean_r2_nm2": mean_r2,
        "mean_r3_nm3": mean_r3,
        "mean_r6_nm6": mean_r6,
        "equivalent_N": number,
        "Sv_nm_inv": 4.0 * math.pi * number * mean_r2 / BOX_VOLUME_NM3,
        "M0_per_nm3": number / BOX_VOLUME_NM3,
        "M1_nm_per_nm3": number * mean_r / BOX_VOLUME_NM3,
        "M2_nm2_per_nm3": number * mean_r2 / BOX_VOLUME_NM3,
        "M3_nm3_per_nm3": number * mean_r3 / BOX_VOLUME_NM3,
        "M4_nm4_per_nm3": number * float(np.sum(weights * radii**4)) / BOX_VOLUME_NM3,
        "M5_nm5_per_nm3": number * float(np.sum(weights * radii**5)) / BOX_VOLUME_NM3,
        "M6_nm6_per_nm3": number * mean_r6 / BOX_VOLUME_NM3,
        "tail_probability_below_0p5_nm": below,
        "tail_probability_above_200_nm": above,
        "tail_probability_outside_global_scan": below + above,
        "beta_inventory_h_volume_nm3": TARGET_H_VOLUME_NM3,
        "canonical_beta_inventory_contract": "Method-1 fixed canonical inventory / exact h-volume authority",
    }


def bimodal_population(small_nm: float, large_nm: float, small_fraction: float) -> dict[str, Any]:
    require(0.0 < small_fraction < 1.0 and small_nm < large_nm, "invalid bimodal definition")
    radii = np.asarray([small_nm, large_nm], dtype=float)
    # Weights are number weights.  The requested inventory fraction is in beta
    # h-volume (the continuous spherical mapping used only for this diagnostic).
    raw_n = np.asarray(
        [small_fraction / small_nm**3, (1.0 - small_fraction) / large_nm**3],
        dtype=float,
    )
    weights = raw_n / np.sum(raw_n)
    base = population_from_nodes(radii, weights, kind="bimodal")
    base.update({
        "small_mode_R_nm": small_nm,
        "large_mode_R_nm": large_nm,
        "small_mode_beta_inventory_fraction": small_fraction,
    })
    return base


def population_from_nodes(radii: np.ndarray, weights: np.ndarray, *, kind: str) -> dict[str, Any]:
    mean_r = float(np.sum(weights * radii))
    mean_r2 = float(np.sum(weights * radii**2))
    mean_r3 = float(np.sum(weights * radii**3))
    mean_r6 = float(np.sum(weights * radii**6))
    number = TARGET_H_VOLUME_NM3 / ((4.0 * math.pi / 3.0) * mean_r3)
    variance = float(np.sum(weights * (radii - mean_r) ** 2))
    return {
        "kind": kind,
        "radii_nm": radii,
        "weights": weights,
        "mean_radius_nm": mean_r,
        "cv": math.sqrt(variance) / mean_r,
        "mean_r2_nm2": mean_r2,
        "mean_r3_nm3": mean_r3,
        "mean_r6_nm6": mean_r6,
        "equivalent_N": number,
        "Sv_nm_inv": 4.0 * math.pi * number * mean_r2 / BOX_VOLUME_NM3,
        "M0_per_nm3": number / BOX_VOLUME_NM3,
        "M1_nm_per_nm3": number * mean_r / BOX_VOLUME_NM3,
        "M2_nm2_per_nm3": number * mean_r2 / BOX_VOLUME_NM3,
        "M3_nm3_per_nm3": number * mean_r3 / BOX_VOLUME_NM3,
        "M4_nm4_per_nm3": number * float(np.sum(weights * radii**4)) / BOX_VOLUME_NM3,
        "M5_nm5_per_nm3": number * float(np.sum(weights * radii**5)) / BOX_VOLUME_NM3,
        "M6_nm6_per_nm3": number * mean_r6 / BOX_VOLUME_NM3,
        "tail_probability_below_0p5_nm": 0.0,
        "tail_probability_above_200_nm": 0.0,
        "tail_probability_outside_global_scan": 0.0,
        "beta_inventory_h_volume_nm3": TARGET_H_VOLUME_NM3,
        "canonical_beta_inventory_contract": "Method-1 fixed canonical inventory / exact h-volume authority",
    }


def kappa_and_spectrum(transport: Any, config: dict[str, Any], population: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    nodes, gauss_weights = leggauss(GAUSS_ORDER)
    x_max = float(config["shared_parameters"]["debye_temperature_K"]) / TEMPERATURE_K
    x = 0.5 * (nodes + 1.0) * x_max
    constants = config["physical_constants"]
    v = float(config["shared_parameters"]["average_sound_velocity_m_s"])
    omega = x * float(constants["k_B_J_K"]) * TEMPERATURE_K / float(constants["hbar_J_s"])
    base = transport.base_scattering_rates(omega, TEMPERATURE_K, MATRIX_XAG, config)
    cross = transport.precipitate_cross_section(omega, population["radii_nm"] * 1.0e-9, config)
    rate = v * float(population["equivalent_N"]) * np.sum(cross * population["weights"][None, :], axis=1) / BOX_VOLUME_M3
    host = base["phonon_phonon"] + base["boundary"] + base["point_defect"]
    total = host + rate
    prefactor = float(constants["k_B_J_K"]) / (2.0 * math.pi**2 * v) * (float(constants["k_B_J_K"]) * TEMPERATURE_K / float(constants["hbar_J_s"]))**3
    kappa = float(0.5 * x_max * np.sum(gauss_weights * prefactor * transport.bose_weight(x) / total))
    frequencies = (0.25e12, 0.50e12, 1.00e12, 2.00e12, 4.00e12)
    spectra: dict[str, Any] = {}
    for frequency in frequencies:
        w = 2.0 * math.pi * frequency
        base_one = transport.base_scattering_rates(np.asarray([w]), TEMPERATURE_K, MATRIX_XAG, config)
        cross_one = transport.precipitate_cross_section(np.asarray([w]), population["radii_nm"] * 1.0e-9, config)
        pre = float(v * population["equivalent_N"] * np.sum(cross_one[0] * population["weights"]) / BOX_VOLUME_M3)
        host_one = float(base_one["phonon_phonon"][0] + base_one["boundary"][0] + base_one["point_defect"][0])
        label = f"{frequency / 1.0e12:g}THz".replace(".", "p")
        spectra.update({
            f"tau_pre_inverse_{label}_s_inv": pre,
            f"tau_host_inverse_{label}_s_inv": host_one,
            f"tau_total_inverse_{label}_s_inv": host_one + pre,
            f"tau_pre_fraction_total_{label}": pre / (host_one + pre),
        })
    return kappa, spectra


def actual_authority(transport: Any, authority_root: Path) -> tuple[dict[str, dict[float, dict[str, Any]]], dict[str, Any]]:
    output: dict[str, dict[float, dict[str, Any]]] = {}
    provenance: dict[str, Any] = {}
    for rep in REPLICATES:
        audit_path = authority_root / "authority" / rep / "audit.json"
        status_path = authority_root / "authority" / rep / "status.txt"
        snapshots_path = authority_root / "transport" / rep / "transport_snapshots.csv"
        particle_path = authority_root / "transport" / rep / "full_psd_particles.csv"
        for path in (audit_path, status_path, snapshots_path, particle_path):
            require(path.is_file(), f"missing authority input: {path}")
        audit = json.loads(audit_path.read_text())
        require(audit["numerical_status"] == "PASS_246CUBE_6H48H_CONDITIONAL_PRODUCTION_V1", f"{rep} not authority PASS")
        require(status_path.read_text().strip() == audit["numerical_status"], f"{rep} status mismatch")
        require(all(audit["gates"].values()), f"{rep} authority gate failed")
        require(len(audit["checkpoint_hashes"]) == 44, f"{rep} must retain 44 checkpoint hashes")
        snapshots = {row["snapshot_id"]: row for row in read_csv(snapshots_path)}
        radii: dict[str, list[float]] = defaultdict(list)
        for row in read_csv(particle_path):
            radii[row["snapshot_id"]].append(float(row["equivalent_radius_nm"]))
        output[rep] = {}
        for snapshot_id, row in snapshots.items():
            age = float(row["age_h"])
            if age not in (6.0, 12.0, 18.0, 24.0, 36.0, 48.0):
                continue
            rs = np.asarray(sorted(radii[snapshot_id]), dtype=float)
            require(rs.size == int(row["particle_count"]), f"{rep} PSD count closure failure at {age}")
            output[rep][age] = {"radii_nm": rs, "matrix_xAg": float(row["matrix_xAg"]), "snapshot_id": snapshot_id, "step": int(row["step"])}
        require(set(output[rep]) == {6.0, 12.0, 18.0, 24.0, 36.0, 48.0}, f"{rep} registered ages missing")
        provenance[rep] = {
            "authority_audit_path": str(audit_path.resolve()),
            "authority_audit_sha256": sha256(audit_path),
            "checkpoint_hashes_sha256": canonical_sha256(audit["checkpoint_hashes"]),
            "checkpoint_hash_count": len(audit["checkpoint_hashes"]),
            "remote_production_root": audit["remote_production_root"],
            "transport_snapshots_sha256": sha256(snapshots_path),
            "full_psd_particles_sha256": sha256(particle_path),
        }
    return output, provenance


def baseline(transport: Any, config: dict[str, Any], authority: dict[str, dict[float, dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[float, float]]:
    rows: list[dict[str, Any]] = []
    for rep in REPLICATES:
        for age in (6.0, 48.0):
            snap = authority[rep][age]
            kappa = transport.integrate_kappa_gauss(TEMPERATURE_K, snap["matrix_xAg"], snap["radii_nm"] * 1.0e-9, BOX_VOLUME_M3, config, "full_psd", GAUSS_ORDER)
            rows.append({"replicate": rep, "age_h": age, "matrix_xAg": snap["matrix_xAg"], "kappa_W_mK": kappa, "full_PSD_sha256": canonical_sha256(snap["radii_nm"].tolist())})
    mean = {age: float(np.mean([float(r["kappa_W_mK"]) for r in rows if r["age_h"] == age])) for age in (6.0, 48.0)}
    delta = mean[48.0] - mean[6.0]
    require(abs(mean[6.0] - FROZEN_BASELINE[6.0]) <= 1.0e-12, "6 h frozen baseline mismatch")
    require(abs(mean[48.0] - FROZEN_BASELINE[48.0]) <= 1.0e-12, "48 h frozen baseline mismatch")
    require(abs(delta - FROZEN_DELTA) <= 1.0e-12, "frozen baseline delta mismatch")
    rows.extend([
        {"replicate": "ensemble_mean", "age_h": 6.0, "matrix_xAg": "PF authority", "kappa_W_mK": mean[6.0], "full_PSD_sha256": ""},
        {"replicate": "ensemble_mean", "age_h": 48.0, "matrix_xAg": "PF authority", "kappa_W_mK": mean[48.0], "full_PSD_sha256": ""},
        {"replicate": "ensemble_delta", "age_h": "6_to_48", "matrix_xAg": "PF authority", "kappa_W_mK": delta, "full_PSD_sha256": ""},
    ])
    return rows, mean


def annotate_population(pop: dict[str, Any], radius_key: str, radius_nm: float, cv: float, kappa: float, spectrum: dict[str, Any]) -> dict[str, Any]:
    equivalent = float(pop["equivalent_N"])
    integer_lower = math.floor(equivalent)
    integer_upper = math.ceil(equivalent)
    per_particle = TARGET_H_VOLUME_NM3 / equivalent
    lower_error = (integer_lower * per_particle - TARGET_H_VOLUME_NM3) / TARGET_H_VOLUME_NM3
    upper_error = (integer_upper * per_particle - TARGET_H_VOLUME_NM3) / TARGET_H_VOLUME_NM3
    return {
        radius_key: radius_nm,
        "population_class": population_class(radius_nm),
        "CV_requested": cv,
        "mean_radius_closure_nm": pop["mean_radius_nm"],
        "CV_closure": pop["cv"],
        "equivalent_continuous_particle_count": equivalent,
        "nearest_integer_particle_count_lower": integer_lower,
        "nearest_integer_particle_count_upper": integer_upper,
        "integer_rounding_inventory_error_lower": lower_error,
        "integer_rounding_inventory_error_upper": upper_error,
        "beta_inventory_h_volume_nm3": TARGET_H_VOLUME_NM3,
        "beta_volume_fraction": TARGET_H_VOLUME_NM3 / BOX_VOLUME_NM3,
        "Nv_m_inv3": pop["M0_per_nm3"] * 1.0e27,
        "Sv_spherical_nm_inv": pop["Sv_nm_inv"],
        "M0_per_nm3": pop["M0_per_nm3"],
        "M1_nm_per_nm3": pop["M1_nm_per_nm3"],
        "M2_nm2_per_nm3": pop["M2_nm2_per_nm3"],
        "M3_nm3_per_nm3": pop["M3_nm3_per_nm3"],
        "M4_nm4_per_nm3": pop["M4_nm4_per_nm3"],
        "M5_nm5_per_nm3": pop["M5_nm5_per_nm3"],
        "M6_nm6_per_nm3": pop["M6_nm6_per_nm3"],
        "kappa_573p15K_density_only_W_mK": kappa,
        "tail_probability_below_0p5_nm": pop["tail_probability_below_0p5_nm"],
        "tail_probability_above_200_nm": pop["tail_probability_above_200_nm"],
        "tail_probability_outside_global_scan": pop["tail_probability_outside_global_scan"],
        "integration": f"full_Debye_Gauss_{GAUSS_ORDER}_lognormal_Hermite_{HERMITE_ORDER}",
        **spectrum,
    }


def plot_1d(path: Path, rows: list[dict[str, Any]], y: str, ylabel: str, title: str) -> None:
    fig, axis = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    for cv in CVS:
        matching = [r for r in rows if float(r["CV_requested"]) == cv]
        axis.plot([r["R_nm"] for r in matching], [r[y] for r in matching], label=f"CV={cv:g}")
    axis.axvspan(0.5, 8.0, color="0.9", label="non-PF mathematical range")
    axis.axvspan(8.0, 11.5, color="#b7e4c7", alpha=0.55, label="current 15-entry library")
    axis.set_xlabel("Arithmetic mean radius (nm)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def heatmap(path: Path, rows: list[dict[str, Any]], value: str, title: str, label: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.4), constrained_layout=True, sharex=True, sharey=True)
    r6 = sorted({float(r["R6_nm"]) for r in rows if r["scan_grid"] == "coarse"})
    r48 = sorted({float(r["R48_nm"]) for r in rows if r["scan_grid"] == "coarse"})
    for axis, cv6, cv48 in zip(axes.flat, (0.0, 0.15, 0.30, 0.50), (0.0, 0.15, 0.30, 0.50)):
        matching = {(float(r["R6_nm"]), float(r["R48_nm"])): float(r[value]) for r in rows if r["scan_grid"] == "coarse" and float(r["CV6"]) == cv6 and float(r["CV48"]) == cv48}
        grid = np.full((len(r48), len(r6)), np.nan)
        for iy, b in enumerate(r48):
            for ix, a in enumerate(r6):
                if (a, b) in matching:
                    grid[iy, ix] = matching[(a, b)]
        image = axis.imshow(grid, origin="lower", aspect="auto", extent=(min(r6), max(r6), min(r48), max(r48)), interpolation="nearest")
        axis.set_title(f"CV6=CV48={cv6:g}")
        axis.set_xlabel("R6 (nm)")
        axis.set_ylabel("R48 (nm)")
        fig.colorbar(image, ax=axis, label=label)
    fig.suptitle(title)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def sort_best(rows: list[dict[str, Any]], key: str, reverse: bool = False) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row[key]), reverse=reverse)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=Path("reports/global_resolved_psd_strain_audit_v1"))
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve() if args.out.is_absolute() else (root / args.out).resolve()
    require(not out.exists(), f"refusing to overwrite output root: {out}")
    out.mkdir(parents=True)
    script = Path(__file__).resolve()
    authority_root = root / "reports/pf_246cube_method1_production_authority_v1"
    transport_path = root / "scripts/pf_full_psd_no_dislocation_transport_v1.py"
    contract_path = root / "data/qualification/pf_full_psd_no_dislocation_transport_v1/transport_parameter_contract.json"
    config_path = root / "data/qualification/yu2024_transport_v1/yu_48h_parameters.json"
    library_path = root / "data/qualification/pf_elastic_target_profile_quarter_nm_v2/library/library_manifest.json"
    transport = load_transport(transport_path)
    config = transport.load_json(config_path)
    contract = transport.load_json(contract_path)
    transport.validate_yu_base_config(config, config_path)
    transport.validate_interface_contract(contract, contract_path, config_path)
    library = json.loads(library_path.read_text())
    require(sha256(library_path) == "de4142e0268e379f70fd1c860aab9e004421d07df4f8d872f4eaeb3dbef7af5b", "wrong 15-entry library")
    require(library.get("profile_count") == 15, "expected 15 profile entries")
    authority, authority_provenance = actual_authority(transport, authority_root)
    baseline_rows, baseline_mean = baseline(transport, config, authority)
    write_csv(out / "baseline_reproduction.csv", baseline_rows)
    (out / "baseline_reproduction_report.md").write_text(
        "# Baseline reproduction\n\n"
        f"`PASS_BASELINE_REPRODUCTION`: κ(6 h)={baseline_mean[6.0]:.16g}, "
        f"κ(48 h)={baseline_mean[48.0]:.16g}; both match the frozen 512-point "
        "full-PSD reference to <=1e-12 W m^-1 K^-1.\n",
        encoding="utf-8",
    )

    provenance = {
        "schema": SCHEMA,
        "git_branch": git(root, "branch", "--show-current"),
        "git_commit": git(root, "rev-parse", "HEAD"),
        "source_tree_sha256": source_tree_digest(root),
        "working_tree_porcelain": git(root, "status", "--porcelain"),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "inputs": {
            "analysis_script": {"path": str(script), "sha256": sha256(script)},
            "transport_script": {"path": str(transport_path), "sha256": sha256(transport_path)},
            "transport_contract": {"path": str(contract_path), "sha256": sha256(contract_path)},
            "yu_parameter_config": {"path": str(config_path), "sha256": sha256(config_path)},
            "profile_library_manifest": {"path": str(library_path), "sha256": sha256(library_path)},
            "authority": authority_provenance,
        },
        "frozen_contract": {
            "temperature_K": TEMPERATURE_K, "A_N": 1.5, "dislocation_mode": "DISLOCATION_OFF",
            "S11_rate": 0, "S13_rate": 0, "yu_refit_scale_used": False,
            "point_defect_xAg_source": "PF_FAR_FIELD_MATRIX_OBSERVATION",
            "scan_matrix_xAg": MATRIX_XAG, "fixed_h_volume_nm3": TARGET_H_VOLUME_NM3,
        },
        "elastic_contract_from_fixture": json.loads((authority_root / "source/A/fixture_manifest.json").read_text()).get("physical_contract", {}),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "provenance.md").write_text(
        "# Provenance\n\nThis is a read-only density-contrast analysis. It does not start PF, modify checkpoints, or alter any frozen transport/PF parameter.\n\n"
        f"- branch: `{provenance['git_branch']}`\n- commit: `{provenance['git_commit']}`\n"
        f"- 15-entry library SHA-256: `{sha256(library_path)}`\n"
        f"- fixed h-volume: `{TARGET_H_VOLUME_NM3:.14f} nm^3`\n"
        f"- fixed matrix xAg: `{MATRIX_XAG}`\n",
        encoding="utf-8",
    )

    # STAGE 1: one-dimensional global radius and width scan.
    one_d: list[dict[str, Any]] = []
    state_cache: dict[tuple[float, float], tuple[dict[str, Any], float, dict[str, Any]]] = {}
    for cv in CVS:
        for radius in piecewise_radii():
            pop = lognormal_population(radius, cv)
            kappa, spectrum = kappa_and_spectrum(transport, config, pop)
            state_cache[(radius, cv)] = (pop, kappa, spectrum)
            one_d.append({"R_nm": radius, **annotate_population(pop, "R_nm", radius, cv, kappa, spectrum)})
    write_csv(out / "global_1d_radius_scan.csv", one_d)
    summary_rows: list[dict[str, Any]] = []
    for cv in CVS:
        select = [row for row in one_d if float(row["CV_requested"]) == cv]
        for name, where in (
            ("global", lambda r: True),
            ("resolved", lambda r: float(r["R_nm"]) >= 8.0),
            ("library", lambda r: 8.0 <= float(r["R_nm"]) <= 11.5),
        ):
            best = min((row for row in select if where(row)), key=lambda r: float(r["kappa_573p15K_density_only_W_mK"]))
            summary_rows.append({"CV_requested": cv, "range": name, "minimum_R_nm": best["R_nm"], "minimum_kappa_W_mK": best["kappa_573p15K_density_only_W_mK"], "minimum_equivalent_N": best["equivalent_continuous_particle_count"], "minimum_class": best["population_class"]})
    write_csv(out / "global_1d_radius_scan_summary.csv", summary_rows)
    plot_1d(out / "kappa_vs_radius", one_d, "kappa_573p15K_density_only_W_mK", "κ (W m⁻¹ K⁻¹)", "Fixed-inventory density-only transport")
    plot_1d(out / "precipitate_rate_vs_radius", one_d, "tau_pre_inverse_1THz_s_inv", "τ⁻¹_precipitate at 1 THz (s⁻¹)", "Precipitate rate at representative frequency")
    plot_1d(out / "scattering_fraction_vs_radius", one_d, "tau_pre_fraction_total_1THz", "precipitate fraction of total τ⁻¹ at 1 THz", "Relative precipitate scattering")
    resolved_min = min((r for r in one_d if float(r["R_nm"]) >= 8.0), key=lambda r: float(r["kappa_573p15K_density_only_W_mK"]))
    global_min = min(one_d, key=lambda r: float(r["kappa_573p15K_density_only_W_mK"]))
    library_min = min((r for r in one_d if 8.0 <= float(r["R_nm"]) <= 11.5), key=lambda r: float(r["kappa_573p15K_density_only_W_mK"]))
    original_15_boundary = float(resolved_min["R_nm"]) != 15.0
    one_d_status = "RESOLVED_6H_ENDPOINT_UNREACHABLE_DENSITY_ONLY" if float(resolved_min["kappa_573p15K_density_only_W_mK"]) > 0.8925 else "RESOLVED_6H_ENDPOINT_DIAGNOSTICALLY_REACHABLE"
    (out / "one_dimensional_global_minimum_report.md").write_text(
        "# Global one-dimensional radius scan\n\n"
        f"- global minimum: R={global_min['R_nm']} nm, κ={global_min['kappa_573p15K_density_only_W_mK']:.12g}; class `{global_min['population_class']}`.\n"
        f"- resolved minimum (R≥8 nm): R={resolved_min['R_nm']} nm, κ={resolved_min['kappa_573p15K_density_only_W_mK']:.12g}.\n"
        f"- current-library minimum (8–11.5 nm): R={library_min['R_nm']} nm, κ={library_min['kappa_573p15K_density_only_W_mK']:.12g}.\n"
        f"- 15 nm boundary artifact: `{original_15_boundary}`.\n"
        f"- 6 h endpoint gate: `{one_d_status}`.\n\n"
        "R<8 nm entries are explicitly mathematical diagnostics and are not PF seed recommendations.\n",
        encoding="utf-8",
    )

    # STAGE 2: all coarsening-compatible pairs.  Kappa depends on a state only,
    # so states are calculated once and paired exactly without a grey model.
    r6_coarse = [round(8.0 + 0.5 * i, 10) for i in range(145)]
    r48_coarse = [float(i) for i in range(8, 201)]
    for cv in CVS:
        for radius in sorted(set(r6_coarse + r48_coarse)):
            if (radius, cv) not in state_cache:
                pop = lognormal_population(radius, cv)
                kappa, spectrum = kappa_and_spectrum(transport, config, pop)
                state_cache[(radius, cv)] = (pop, kappa, spectrum)
    two_d: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float, float, str]] = set()

    def append_pair(r6: float, r48: float, cv6: float, cv48: float, grid: str, reason: str) -> None:
        key = (r6, r48, cv6, cv48, grid)
        if key in seen or r48 < r6:
            return
        seen.add(key)
        pop6, k6, _ = state_cache[(r6, cv6)]
        pop48, k48, _ = state_cache[(r48, cv48)]
        ratio = r48 / r6
        two_d.append({
            "scan_grid": grid, "refinement_reason": reason, "CV6": cv6, "CV48": cv48, "R6_nm": r6, "R48_nm": r48,
            "R48_over_R6": ratio, "kappa6_W_mK": k6, "kappa48_W_mK": k48,
            "delta_kappa_W_mK": k48 - k6, "relative_change": (k48 - k6) / k6,
            "J_abs": ((k6 - EXPERIMENT_K6) / EXPERIMENT_K6)**2 + ((k48 - EXPERIMENT_K48) / EXPERIMENT_K48)**2,
            "J_delta": ((k48 - k6) - EXPERIMENT_DELTA)**2,
            "J_relative": ((k48 - k6) / k6 - EXPERIMENT_RELATIVE)**2,
            "N6_equivalent": pop6["equivalent_N"], "N48_equivalent": pop48["equivalent_N"],
            "Sv6_nm_inv": pop6["Sv_nm_inv"], "Sv48_nm_inv": pop48["Sv_nm_inv"],
            "M6_6h_nm6_per_nm3": pop6["M6_nm6_per_nm3"], "M6_48h_nm6_per_nm3": pop48["M6_nm6_per_nm3"],
            "mathematical_combination": True, "PF_coarsening_compatible": ratio >= 1.2 and ratio <= 5.0 and pop48["equivalent_N"] < pop6["equivalent_N"],
            "both_endpoints_within_5percent": abs(k6 - EXPERIMENT_K6) / EXPERIMENT_K6 <= .05 and abs(k48 - EXPERIMENT_K48) / EXPERIMENT_K48 <= .05,
            "beta_inventory_h_volume_nm3": TARGET_H_VOLUME_NM3, "matrix_xAg": MATRIX_XAG,
        })

    for cv6 in CVS:
        for cv48 in CVS:
            for r6 in r6_coarse:
                for r48 in r48_coarse:
                    append_pair(r6, r48, cv6, cv48, "coarse", "registered_coarse_grid")
    coarse_pf = [r for r in two_d if r["PF_coarsening_compatible"]]
    require(coarse_pf, "no coarse PF-compatible pairs")
    refinement_centres = [
        (min(coarse_pf, key=lambda r: r["J_abs"]), "minimum_J_abs"),
        (max(coarse_pf, key=lambda r: r["delta_kappa_W_mK"]), "maximum_positive_delta"),
        (min(coarse_pf, key=lambda r: r["J_relative"]), "closest_relative_change"),
        (min(coarse_pf, key=lambda r: (r["kappa6_W_mK"] + r["kappa48_W_mK"])), "one_dimensional_extrema_neighborhood"),
    ]
    for centre, reason in refinement_centres:
        for cv6 in CVS:
            for cv48 in CVS:
                r6_values = np.arange(max(8.0, float(centre["R6_nm"]) - .5), min(80.0, float(centre["R6_nm"]) + .5) + 1.0e-9, .05)
                r48_values = np.arange(max(8.0, float(centre["R48_nm"]) - 1.0), min(200.0, float(centre["R48_nm"]) + 1.0) + 1.0e-9, .10)
                for r6_raw in r6_values:
                    r6 = round(float(r6_raw), 10)
                    if (r6, cv6) not in state_cache:
                        pop = lognormal_population(r6, cv6); k, spec = kappa_and_spectrum(transport, config, pop); state_cache[(r6, cv6)] = (pop, k, spec)
                    for r48_raw in r48_values:
                        r48 = round(float(r48_raw), 10)
                        if r48 < r6:
                            continue
                        if (r48, cv48) not in state_cache:
                            pop = lognormal_population(r48, cv48); k, spec = kappa_and_spectrum(transport, config, pop); state_cache[(r48, cv48)] = (pop, k, spec)
                        append_pair(r6, r48, cv6, cv48, "local_refine", reason)
    write_csv(out / "global_2d_scan_all_cases.csv", two_d)
    resolved_all = [r for r in two_d if r["PF_coarsening_compatible"]]
    write_csv(out / "global_2d_scan_best_absolute.csv", sort_best(resolved_all, "J_abs")[:500])
    write_csv(out / "global_2d_scan_best_delta.csv", sort_best(resolved_all, "delta_kappa_W_mK", reverse=True)[:500])
    write_csv(out / "global_2d_scan_best_relative.csv", sort_best(resolved_all, "J_relative")[:500])
    heatmap(out / "heatmap_kappa6", two_d, "kappa6_W_mK", "κ6: resolved coarsening-compatible radius pairs", "κ6 (W m⁻¹ K⁻¹)")
    heatmap(out / "heatmap_kappa48", two_d, "kappa48_W_mK", "κ48: resolved coarsening-compatible radius pairs", "κ48 (W m⁻¹ K⁻¹)")
    heatmap(out / "heatmap_delta_kappa", two_d, "delta_kappa_W_mK", "Δκ: resolved coarsening-compatible radius pairs", "Δκ (W m⁻¹ K⁻¹)")
    heatmap(out / "heatmap_relative_change", two_d, "relative_change", "relative conductivity change", "(κ48−κ6)/κ6")
    heatmap(out / "heatmap_J_abs", two_d, "J_abs", "Absolute-endpoint diagnostic objective", "J_abs")

    # Bimodal diagnostics: all states retain identical h-volume; pairing uses
    # the mass-equivalent R3 mean as a registered coarsening metric.
    bimodal_states: list[dict[str, Any]] = []
    for small in (8.0, 10.0, 12.0, 15.0):
        for large in (20.0, 30.0, 40.0, 60.0):
            if large <= small:
                continue
            for fraction in (.25, .50, .75):
                pop = bimodal_population(small, large, fraction)
                kappa, spectrum = kappa_and_spectrum(transport, config, pop)
                bimodal_states.append({"state": f"s{small:g}_l{large:g}_f{fraction:g}", "R3_equivalent_nm": pop["mean_r3_nm3"]**(1.0 / 3.0), "kappa_W_mK": kappa, **{key: value for key, value in pop.items() if key not in ("radii_nm", "weights")}, **spectrum})
    bimodal: list[dict[str, Any]] = []
    for state6 in bimodal_states:
        for state48 in bimodal_states:
            r6, r48 = float(state6["R3_equivalent_nm"]), float(state48["R3_equivalent_nm"])
            if r48 < r6:
                continue
            k6, k48 = float(state6["kappa_W_mK"]), float(state48["kappa_W_mK"])
            bimodal.append({
                "six_h_state": state6["state"], "forty_eight_h_state": state48["state"], "R6_R3_equivalent_nm": r6, "R48_R3_equivalent_nm": r48,
                "kappa6_W_mK": k6, "kappa48_W_mK": k48, "delta_kappa_W_mK": k48-k6, "relative_change": (k48-k6)/k6,
                "J_abs": ((k6-EXPERIMENT_K6)/EXPERIMENT_K6)**2 + ((k48-EXPERIMENT_K48)/EXPERIMENT_K48)**2,
                "J_delta": ((k48-k6)-EXPERIMENT_DELTA)**2, "J_relative": ((k48-k6)/k6-EXPERIMENT_RELATIVE)**2,
                "mathematical_combination": True, "PF_coarsening_compatible": 1.2 <= r48/r6 <= 5.0 and float(state48["equivalent_N"]) < float(state6["equivalent_N"]),
                "both_endpoints_within_5percent": abs(k6-EXPERIMENT_K6)/EXPERIMENT_K6 <= .05 and abs(k48-EXPERIMENT_K48)/EXPERIMENT_K48 <= .05,
                "beta_inventory_h_volume_nm3": TARGET_H_VOLUME_NM3,
            })
    write_csv(out / "global_bimodal_scan.csv", bimodal)

    all_pairs = resolved_all + [row for row in bimodal if row["PF_coarsening_compatible"]]
    best_abs = min(all_pairs, key=lambda row: float(row["J_abs"]))
    max_delta = max(all_pairs, key=lambda row: float(row["delta_kappa_W_mK"]))
    best_relative = min(all_pairs, key=lambda row: float(row["J_relative"]))
    any_endpoint = any(bool(row["both_endpoints_within_5percent"]) for row in all_pairs)
    max_delta_value = float(max_delta["delta_kappa_W_mK"])
    if (not any_endpoint and float(resolved_min["kappa_573p15K_density_only_W_mK"]) > .8925 and max_delta_value < .5 * EXPERIMENT_DELTA):
        density_status = "NO_GO_RESOLVED_DENSITY_CONTRAST_GLOBAL_RANGE"
    elif not any_endpoint and max_delta_value >= .5 * EXPERIMENT_DELTA:
        density_status = "ENDPOINT_NO_GO_BUT_TREND_LEVERAGE_EXISTS"
    elif any_endpoint:
        density_status = "PASS_GLOBAL_RESOLVED_DENSITY_ONLY_FEASIBLE"
    else:
        density_status = "ONLY_NON_PF_RADIUS_RANGE_FEASIBLE"
    (out / "resolved_density_contrast_global_decision.md").write_text(
        "# Global resolved density-contrast decision\n\n"
        f"`{density_status}`\n\n"
        f"Best absolute diagnostic pair: `{best_abs}`.\n\n"
        f"Maximum positive Δκ: `{max_delta_value:.12g} W m^-1 K^-1` ({100*max_delta_value/EXPERIMENT_DELTA:.4g}% of experiment); pair: `{max_delta}`.\n\n"
        f"Closest relative-change pair: `{best_relative}`.\n\n"
        f"Any PF-coarsening-compatible endpoint pair within the pre-registered 5% diagnostic gate: `{any_endpoint}`.\n\n"
        "All results keep the same target h-volume, canonical-inventory contract and xAg=0.0062.  Radius/count populations outside 8–11.5 nm are not claims of current profile-library constructibility.\n",
        encoding="utf-8",
    )
    terminal = {
        "baseline_reproduction_status": "PASS_BASELINE_REPRODUCTION",
        "global_radius_scan_min_R_nm": global_min["R_nm"], "global_radius_scan_min_kappa": global_min["kappa_573p15K_density_only_W_mK"],
        "resolved_radius_scan_min_R_nm": resolved_min["R_nm"], "resolved_radius_scan_min_kappa": resolved_min["kappa_573p15K_density_only_W_mK"],
        "library_range_min_R_nm": library_min["R_nm"], "library_range_min_kappa": library_min["kappa_573p15K_density_only_W_mK"],
        "original_15nm_boundary_artifact": original_15_boundary,
        "best_absolute_R6_nm": best_abs.get("R6_nm", best_abs.get("R6_R3_equivalent_nm")), "best_absolute_R48_nm": best_abs.get("R48_nm", best_abs.get("R48_R3_equivalent_nm")),
        "best_absolute_kappa6": best_abs["kappa6_W_mK"], "best_absolute_kappa48": best_abs["kappa48_W_mK"], "best_absolute_J": best_abs["J_abs"],
        "maximum_positive_delta_kappa": max_delta_value, "maximum_relative_increase_percent": 100.0 * max(float(row["relative_change"]) for row in all_pairs),
        "delta_kappa_fraction_of_experiment": max_delta_value / EXPERIMENT_DELTA,
        "any_resolved_endpoint_pair_within_5percent": any_endpoint, "density_only_global_status": density_status,
        "R1_R2_dynamic_PF_status": "DEFERRED", "recommended_next_action": "Run the read-only checkpoint elastic-field reconstruction before proposing a coherent-strain phonon-scattering model.",
        "final_status": density_status,
    }
    (out / "final_terminal_density_stage.txt").write_text("\n".join(f"{key}={value}" for key, value in terminal.items()) + "\n", encoding="utf-8")
    (out / "analysis_manifest_density.json").write_text(json.dumps({"schema": SCHEMA, "outputs": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(density_status)


if __name__ == "__main__":
    main()
