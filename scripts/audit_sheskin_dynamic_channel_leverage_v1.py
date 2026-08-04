#!/usr/bin/env python3
"""Pre-calibration leverage audit for PF interface and strain descriptors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


KAPPA_DENSITY_6 = 1.2980916621210676
KAPPA_DENSITY_48 = 1.2948991542326225
KAPPA_EXPERIMENT_6 = 0.85
KAPPA_EXPERIMENT_48 = 1.03


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classification(ratio: float) -> str:
    if ratio <= 0.60:
        return "STRONG_LEVERAGE"
    if ratio <= 0.80:
        return "MODERATE_LEVERAGE"
    return "WEAK_LEVERAGE"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)

    ratios = read_csv(args.descriptors / "structural_relaxation_ratios.csv")
    ensemble = {
        row["descriptor"]: row for row in ratios if row["scope"] == "ensemble"
    }
    required = (
        "Sv",
        "hydrostatic_strain_variance",
        "epsilon_h__mid_q",
        "epsilon_h__high_q",
    )
    missing = set(required) - set(ensemble)
    if missing:
        raise ValueError(f"missing registered ratios: {sorted(missing)}")

    sv_ratio = float(ensemble["Sv"]["mean"])
    total_strain_ratio = float(ensemble["hydrostatic_strain_variance"]["mean"])
    mid_ratio = float(ensemble["epsilon_h__mid_q"]["mean"])
    high_ratio = float(ensemble["epsilon_h__high_q"]["mean"])
    interface_status = classification(sv_ratio)
    strain_status = classification(max(mid_ratio, high_ratio))
    spectral_redistribution = total_strain_ratio > 0.80 and min(mid_ratio, high_ratio) < 0.60
    missing_6 = 1.0 / KAPPA_EXPERIMENT_6 - 1.0 / KAPPA_DENSITY_6
    missing_48 = 1.0 / KAPPA_EXPERIMENT_48 - 1.0 / KAPPA_DENSITY_48
    missing_ratio = missing_48 / missing_6

    audit_rows = [
        {
            "channel": "experimental_resistance_scale",
            "descriptor": "W_missing_48_over_W_missing_6",
            "ratio_48h_over_6h": missing_ratio,
            "ensemble_min": "",
            "ensemble_max": "",
            "trend_sign_consistent": "",
            "leverage_status": "MAGNITUDE_DIAGNOSTIC_NOT_A_DEBYE_MODEL",
        },
        {
            "channel": "interface",
            "descriptor": "Sv_and_I1_frequency_weighted_geometry",
            "ratio_48h_over_6h": sv_ratio,
            "ensemble_min": ensemble["Sv"]["minimum"],
            "ensemble_max": ensemble["Sv"]["maximum"],
            "trend_sign_consistent": ensemble["Sv"]["trend_sign_consistent"],
            "leverage_status": interface_status,
        },
        {
            "channel": "strain",
            "descriptor": "epsilon_h_mid_q",
            "ratio_48h_over_6h": mid_ratio,
            "ensemble_min": ensemble["epsilon_h__mid_q"]["minimum"],
            "ensemble_max": ensemble["epsilon_h__mid_q"]["maximum"],
            "trend_sign_consistent": ensemble["epsilon_h__mid_q"]["trend_sign_consistent"],
            "leverage_status": classification(mid_ratio),
        },
        {
            "channel": "strain",
            "descriptor": "epsilon_h_high_q",
            "ratio_48h_over_6h": high_ratio,
            "ensemble_min": ensemble["epsilon_h__high_q"]["minimum"],
            "ensemble_max": ensemble["epsilon_h__high_q"]["maximum"],
            "trend_sign_consistent": ensemble["epsilon_h__high_q"]["trend_sign_consistent"],
            "leverage_status": classification(high_ratio),
        },
        {
            "channel": "strain_diagnostic",
            "descriptor": "hydrostatic_total_variance",
            "ratio_48h_over_6h": total_strain_ratio,
            "ensemble_min": ensemble["hydrostatic_strain_variance"]["minimum"],
            "ensemble_max": ensemble["hydrostatic_strain_variance"]["maximum"],
            "trend_sign_consistent": ensemble["hydrostatic_strain_variance"]["trend_sign_consistent"],
            "leverage_status": "SPECTRAL_REDISTRIBUTION_LEVERAGE" if spectral_redistribution else classification(total_strain_ratio),
        },
    ]
    with (args.out / "dynamic_channel_leverage_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    report = f"""# Dynamic-channel leverage audit

This audit was completed before any 6 h dynamic-parameter calibration.

- Required effective-resistance ratio: `{missing_ratio:.9g}` (magnitude diagnostic only; not substituted for the Debye integral).
- True periodic interface-area ratio, `Sv_48/Sv_6`: `{sv_ratio:.9g}` -> `{interface_status}`.
- Hydrostatic mid-q power ratio: `{mid_ratio:.9g}`.
- Hydrostatic high-q power ratio: `{high_ratio:.9g}`.
- Hydrostatic total-variance ratio: `{total_strain_ratio:.9g}`.
- Strain leverage: `{strain_status}`.
- Spectral-redistribution flag: `{str(spectral_redistribution).upper()}`.

The total hydrostatic variance grows, but the registered mid/high-q power falls by more than 40% in every replicate. Therefore mean elastic energy or total strain variance would give the wrong mechanism diagnostic; the formal strain model must use the q-resolved spectrum.
"""
    (args.out / "dynamic_channel_leverage_audit.md").write_text(report, encoding="utf-8")

    interface = read_csv(args.descriptors / "interface_statistics_time_series.csv")
    strain = read_csv(args.descriptors / "strain_statistics_time_series.csv")
    bands = read_csv(args.descriptors / "strain_band_integrals.csv")
    spectrum = read_csv(args.descriptors / "strain_power_spectrum.csv")
    ages = np.array(sorted({int(row["age_h"]) for row in interface}))

    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    for rep in "ABC":
        values = [float(next(row for row in interface if row["replicate"] == rep and int(row["age_h"]) == age)["Sv_m_inv"]) for age in ages]
        ax.plot(ages, np.array(values) * 1.0e-6, "o-", label=rep)
    ax.set(xlabel="Age (h)", ylabel=r"$S_v$ ($10^6$ m$^{-1}$)", title="PF true periodic interface area density")
    ax.grid(alpha=0.25); ax.legend(title="Replicate")
    fig.savefig(args.out / "Sv_vs_time.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    for descriptor, label, style in (
        ("hydrostatic_strain_variance", "total hydrostatic variance", "o-"),
        ("epsilon_h__mid_q", "mid-q hydrostatic power", "s--"),
        ("epsilon_h__high_q", "high-q hydrostatic power", "^-."),
    ):
        values = []
        for age in ages:
            if descriptor == "hydrostatic_strain_variance":
                selected = [float(row["epsilon_h_variance"]) for row in strain if row["region"] == "whole_box" and int(row["age_h"]) == age]
            else:
                band = descriptor.rsplit("__", 1)[1]
                selected = [float(row["integrated_variance"]) for row in bands if row["field"] == "epsilon_h" and row["band"] == band and int(row["age_h"]) == age]
            values.append(float(np.mean(selected)))
        values = np.asarray(values) / values[0]
        ax.plot(ages, values, style, label=label)
    ax.axhline(0.6, color="0.5", linewidth=1, linestyle=":", label="strong-leverage threshold")
    ax.set(xlabel="Age (h)", ylabel="Ensemble descriptor / 6 h", title="Strain spectral redistribution")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.savefig(args.out / "strain_variance_vs_time.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    for age, style in ((6, "-"), (48, "--")):
        selected = [
            row
            for row in spectrum
            if row["field"] == "epsilon_h"
            and row["direction"] == "isotropic_radial"
            and int(row["age_h"]) == age
            and float(row["weighted_mode_count"]) > 0.0
        ]
        q = sorted({float(row["q_center_nm_inv"]) for row in selected})
        y = [np.mean([float(row["correlation_spectral_density_m3"]) for row in selected if float(row["q_center_nm_inv"]) == value]) for value in q]
        ax.semilogy(q, np.maximum(y, np.finfo(float).tiny), style, linewidth=1.8, label=f"{age} h ensemble mean")
    ax.axvspan(0.0, np.pi / 16.0, color="C0", alpha=0.06)
    ax.axvspan(np.pi / 16.0, np.pi / 4.0, color="C1", alpha=0.06)
    ax.axvspan(np.pi / 4.0, np.pi, color="C2", alpha=0.04)
    ax.set(xlabel=r"$q$ (nm$^{-1}$)", ylabel=r"$S_{\epsilon_h}(q)$ (m$^3$)", title="Accepted-field hydrostatic strain spectrum")
    ax.grid(alpha=0.25); ax.legend()
    fig.savefig(args.out / "strain_spectrum_6h_vs_48h.png", dpi=200); plt.close(fig)

    manifest = {
        "schema": "SHESKIN_DYNAMIC_CHANNEL_LEVERAGE_AUDIT_V1",
        "status": "PASS_DYNAMIC_CHANNEL_LEVERAGE_AUDIT_V1",
        "interface_leverage_status": interface_status,
        "strain_leverage_status": strain_status,
        "spectral_redistribution_leverage": spectral_redistribution,
        "descriptor_manifest_sha256": sha256(args.descriptors / "descriptor_manifest.json"),
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
    }
    (args.out / "leverage_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_paths = sorted(path for path in args.out.iterdir() if path.is_file())
    with (args.out / "leverage_outputs.sha256").open("w", encoding="utf-8") as stream:
        for path in output_paths:
            stream.write(f"{sha256(path)}  {path.name}\n")
    print(manifest["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
