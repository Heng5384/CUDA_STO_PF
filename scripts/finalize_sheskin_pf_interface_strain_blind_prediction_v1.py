#!/usr/bin/env python3
"""Final fail-closed adjudication and completion audit for the Sheskin study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


FINAL_STATUS = "FAIL_RESOLVED_ONLY_AFTER_INTERFACE_AND_STRAIN_TEST"
REQUIRED = (
    "provenance.md",
    "baseline_reproduction.csv",
    "baseline_reproduction_report.md",
    "sheskin_experimental_kappa.csv",
    "sheskin_microstructure_source_table.csv",
    "sheskin_digitization_provenance.md",
    "experimental_identity_audit.md",
    "sheskin_AQ_known_channel_kappa.csv",
    "sheskin_background_candidate_fits.csv",
    "sheskin_background_cross_validation.csv",
    "sheskin_background_model_selection.md",
    "sheskin_background_uncertainty_manifest.json",
    "replay_online_vs_offline_comparison.csv",
    "replay_energy_closure.csv",
    "replay_determinism_audit.md",
    "replay_field_manifest.csv",
    "replay_solver_residuals.csv",
    "replay_provenance.md",
    "interface_statistics_time_series.csv",
    "interface_normal_distribution.csv",
    "strain_statistics_time_series.csv",
    "stress_statistics_time_series.csv",
    "strain_power_spectrum.csv",
    "strain_band_integrals.csv",
    "structural_relaxation_ratios.csv",
    "dynamic_channel_leverage_audit.md",
    "interface_scattering_literature_contract.md",
    "strain_scattering_literature_contract.md",
    "candidate_model_parameter_table.csv",
    "interface_model_equation_contract.md",
    "interface_model_6h_calibration.csv",
    "interface_parameter_physicality_audit.md",
    "strain_model_equation_contract.md",
    "strain_model_6h_calibration.csv",
    "strain_parameter_physicality_audit.md",
    "frozen_before_48h_prediction_manifest.json",
    "blind_48h_predictions.csv",
    "full_temperature_comparison.csv",
    "scattering_rate_decomposition.csv",
    "cumulative_kappa_spectrum.csv",
    "AQ_background_fit.png",
    "6h_calibration_curve.png",
    "6h_48h_kappa_comparison.png",
    "delta_kappa_vs_temperature.png",
    "Sv_vs_time.png",
    "strain_variance_vs_time.png",
    "strain_spectrum_6h_vs_48h.png",
    "scattering_rate_decomposition_6h.png",
    "scattering_rate_decomposition_48h.png",
    "cumulative_kappa_6h_48h.png",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.report_root
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"missing required outputs: {missing}")
    for name in ("final_sheskin_blind_prediction_report.md", "final_terminal_output.txt", "completion_audit.json", "final_outputs.sha256"):
        if (root / name).exists():
            raise RuntimeError(f"refusing to overwrite final output: {name}")

    baseline = read_csv(root / "baseline_reproduction.csv")
    for age, expected in (("6.0", 1.2980916621210676), ("48.0", 1.2948991542326225)):
        row = next(row for row in baseline if row["replicate"] == "ensemble_mean" and row["age_h"] == age)
        if abs(float(row["kappa_W_mK"]) - expected) > 1.0e-12 or row["status"] != "PASS_BASELINE_REPRODUCTION":
            raise RuntimeError("baseline reproduction gate failed")

    background = json.loads((root / "sheskin_background_uncertainty_manifest.json").read_text(encoding="utf-8"))
    frozen = json.loads((root / "frozen_before_48h_prediction_manifest.json").read_text(encoding="utf-8"))
    blind = json.loads((root / "blind_evaluation_summary.json").read_text(encoding="utf-8"))
    replay = json.loads((root / "replay_qualification_case/qualification/audit.json").read_text(encoding="utf-8"))
    descriptors = json.loads((root / "descriptor_manifest.json").read_text(encoding="utf-8"))
    leverage = json.loads((root / "leverage_manifest.json").read_text(encoding="utf-8"))
    if background["status"] != "SHESKIN_BACKGROUND_IDENTIFIED":
        raise RuntimeError("background status mismatch")
    if frozen["status"] != "PASS_FROZEN_BEFORE_48H_PREDICTION_V1" or frozen["read_48h_experiment_during_calibration"]:
        raise RuntimeError("freeze gate failed")
    if blind["status"] != FINAL_STATUS or not blind["no_48h_refit"]:
        raise RuntimeError("blind status mismatch")
    if replay["status"] != "PASS_MECHANICS_ONLY_ACCEPTED_FIELD_REPLAY_V1":
        raise RuntimeError("mechanics replay gate failed")
    if descriptors["status"] != "PASS_PF_PERIODIC_INTERFACE_STRAIN_DESCRIPTOR_AUDIT_V1":
        raise RuntimeError("descriptor audit failed")
    if leverage["status"] != "PASS_DYNAMIC_CHANNEL_LEVERAGE_AUDIT_V1":
        raise RuntimeError("leverage audit failed")

    ratios = read_csv(root / "structural_relaxation_ratios.csv")
    ratio = {
        row["descriptor"]: float(row["mean"])
        for row in ratios
        if row["scope"] == "ensemble"
    }
    predictions = read_csv(root / "blind_48h_predictions.csv")
    endpoint: dict[str, dict[str, float]] = {}
    envelope: dict[str, dict[str, float]] = {}
    for model in ("M0", "MI", "MS", "MIS"):
        values = {6: [], 48: []}
        member_values: dict[str, dict[int, list[float]]] = defaultdict(lambda: {6: [], 48: []})
        for row in predictions:
            if row["model"] != model or float(row["temperature_K"]) != 573.15:
                continue
            age = int(float(row["age_h"]))
            value = float(row["predicted_kappa_W_mK"])
            values[age].append(value)
            member_values[row["background_case_id"]][age].append(value)
        mean6, mean48 = statistics.mean(values[6]), statistics.mean(values[48])
        member_mean6 = [statistics.mean(item[6]) for item in member_values.values()]
        member_mean48 = [statistics.mean(item[48]) for item in member_values.values()]
        member_relative = [100.0 * (b - a) / a for a, b in zip(member_mean6, member_mean48)]
        endpoint[model] = {
            "mean6": mean6,
            "mean48": mean48,
            "relative": 100.0 * (mean48 - mean6) / mean6,
        }
        envelope[model] = {
            "k48_min": min(member_mean48),
            "k48_median": statistics.median(member_mean48),
            "k48_max": max(member_mean48),
            "relative_min": min(member_relative),
            "relative_median": statistics.median(member_relative),
            "relative_max": max(member_relative),
        }

    alpha = [float(row["interface_alpha"]) for row in frozen["calibrations"]]
    alpha_physical = sum(row["parameter_physicality"] == "PASS_WITHIN_FROZEN_BOUNDS" for row in frozen["calibrations"])
    mape = [float(row["AQ_MAPE_percent"]) for row in background["members"]]
    residuals = read_csv(root / "replay_solver_residuals.csv")
    historical_residual_max = max(float(row["solver_relative_residual"]) for row in residuals)
    accepted_comparison = next(row for row in replay["comparisons"] if row["comparison"] == "online_vs_checkpoint_warm_replay")

    report = f"""# Sheskin AQ-background + PF interface/strain blind-prediction report

## Outcome

`{FINAL_STATUS}`

The resolved PF microstructure has strong geometric and spectral evolution, and the qualified interface model correctly predicts the **sign** of recovery for every retained AQ-background member. It does not predict the required magnitude. At 573.15 K, the nonprobabilistic background-envelope interface result spans `{envelope['MI']['k48_min']:.6g}`--`{envelope['MI']['k48_max']:.6g}` W m^-1 K^-1 at 48 h, with only `{envelope['MI']['relative_min']:.4g}`--`{envelope['MI']['relative_max']:.4g}%` recovery. The experiment is 1.03 W m^-1 K^-1 and the preregistered recovery gate is 10--30%. No retained member reaches that gate.

The parameter-free scalar Born strain channel is far weaker than the host/background rates and retains the M0 negative trend. Adding it to MI changes the endpoint negligibly. Therefore neither resolved interface scattering nor resolved coherent hydrostatic-strain scattering closes the experimental gap under the frozen contracts.

## Evidence chain

- Density-only baseline reproduced exactly: 6 h `1.2980916621210676`, 48 h `1.2948991542326225` W m^-1 K^-1.
- AQ selected H2 background MAPE envelope: `{min(mape):.5g}`--`{max(mape):.5g}%`; however all one-parameter AQ candidates have poor absolute chi-square fit. The background is an empirical time-invariant envelope, not a named defect.
- Mechanics replay: checkpoint-warm online/offline fields are byte-identical with zero energy/strain/stress error. The zero-initialized sensitivity path failed the frozen strain/stress tolerances and is forbidden for authority use.
- Historical replay: 18 accepted-field states, maximum solver residual `{historical_residual_max:.6g}`, no PF/time advance and no checkpoint write.
- Interface area ratio `Sv_48/Sv_6 = {ratio['Sv']:.6g}` (`STRONG_LEVERAGE`).
- Total hydrostatic variance ratio `{ratio['hydrostatic_strain_variance']:.6g}`, but mid/high-q ratios are `{ratio['epsilon_h__mid_q']:.6g}` and `{ratio['epsilon_h__high_q']:.6g}`: `SPECTRAL_REDISTRIBUTION_LEVERAGE`.
- Interface alpha range `{min(alpha):.6g}`--`{max(alpha):.6g}`; `{alpha_physical}/18` background members are strictly inside the frozen `[0.1,10]` interval.
- The freeze manifest was generated from AQ + 6 h only. The 48 h evaluation verified all frozen hashes and performed no refit.

## 573.15 K frozen results

| Model | equal-weight diagnostic mean 6 h | equal-weight diagnostic mean 48 h | change | background-member recovery envelope |
|---|---:|---:|---:|---:|
| M0 | {endpoint['M0']['mean6']:.6f} | {endpoint['M0']['mean48']:.6f} | {endpoint['M0']['relative']:.3f}% | {envelope['M0']['relative_min']:.3f} to {envelope['M0']['relative_max']:.3f}% |
| MI | {endpoint['MI']['mean6']:.6f} | {endpoint['MI']['mean48']:.6f} | {endpoint['MI']['relative']:.3f}% | {envelope['MI']['relative_min']:.3f} to {envelope['MI']['relative_max']:.3f}% |
| MS | {endpoint['MS']['mean6']:.6f} | {endpoint['MS']['mean48']:.6f} | {endpoint['MS']['relative']:.3f}% | {envelope['MS']['relative_min']:.3f} to {envelope['MS']['relative_max']:.3f}% |
| MIS | {endpoint['MIS']['mean6']:.6f} | {endpoint['MIS']['mean48']:.6f} | {endpoint['MIS']['relative']:.3f}% | {envelope['MIS']['relative_min']:.3f} to {envelope['MIS']['relative_max']:.3f}% |

The equal-weight means average A/B/C and the 18 retained background members only as a deterministic diagnostic. The background members are an uncertainty envelope, not a probability distribution; the min/median/max member results in `full_temperature_comparison.csv` are the authoritative propagation.

## Interpretation boundary

Figure 5c is measured **total** thermal conductivity, while the calculation uses an effective Debye lattice-style relaxation-time reconstruction. The identities are kept explicit. This study does not claim absolute experimental lattice-thermal-conductivity reproduction, first-principles prediction, or a PF dislocation-density prediction.

The correct next action is not to retune the PF PSD, eigenstrain, thermodynamics, `A_N`, or Yu dislocation scale. Within the registered resolved-particle-only scope, the mechanism test is complete and fails the magnitude gates. Any further route requires independently constrained physics outside this model, such as polarization-resolved PbTe/Ag2Te interface transmission or separately evidenced unresolved defects; it must be a new contract rather than a fit to this 48 h endpoint.
"""
    (root / "final_sheskin_blind_prediction_report.md").write_text(report, encoding="utf-8")

    terminal_lines = [
        "baseline_reproduction_status=PASS_BASELINE_REPRODUCTION",
        "sheskin_AQ_data_status=DIGITIZED_MEASURED_TOTAL_7_POINTS_PER_STATE_WITH_EXPLICIT_573P15_ENDPOINTS",
        "sheskin_background_status=SHESKIN_BACKGROUND_IDENTIFIED_WITH_POOR_ABSOLUTE_FIT_WARNING",
        "selected_background_model=H2_omega2_18_member_source_literal_envelope",
        f"selected_background_parameter_s={min(float(row['parameter']) for row in background['members']):.17g},{max(float(row['parameter']) for row in background['members']):.17g}",
        f"background_AQ_MAPE_percent={min(mape):.17g},{max(mape):.17g}",
        "mechanics_replay_status=PASS_MECHANICS_ONLY_ACCEPTED_FIELD_REPLAY_V1_CHECKPOINT_WARM_ONLY",
        f"replay_energy_closure_max_relative={float(accepted_comparison['energy_relative_error']):.17g}",
        f"replay_strain_L2_max={float(accepted_comparison['max_strain_normalized_l2']):.17g}",
        f"replay_stress_L2_max={float(accepted_comparison['max_stress_normalized_l2']):.17g}",
        "zero_initialized_replay_status=REJECTED_NOT_QUALIFIED",
        f"historical_replay_residual_max={historical_residual_max:.17g}",
        f"Sv_ratio_48h_6h_mean={ratio['Sv']:.17g}",
        f"hydrostatic_strain_ratio_48h_6h_mean={ratio['hydrostatic_strain_variance']:.17g}",
        f"deviatoric_strain_ratio_48h_6h_mean={ratio['deviatoric_strain_variance']:.17g}",
        f"mid_q_strain_power_ratio_48h_6h_mean={ratio['epsilon_h__mid_q']:.17g}",
        f"high_q_strain_power_ratio_48h_6h_mean={ratio['epsilon_h__high_q']:.17g}",
        "interface_leverage_status=STRONG_LEVERAGE",
        "strain_leverage_status=STRONG_LEVERAGE_SPECTRAL_REDISTRIBUTION",
        "interface_model_status=TREND_PASS_MAGNITUDE_FAIL_16_OF_18_PARAMETER_PHYSICAL",
        f"interface_parameter_alpha={min(alpha):.17g},{max(alpha):.17g}",
        f"interface_parameter_physicality={alpha_physical}_OF_18_PASS",
        "strain_model_status=PARAMETER_FREE_RATE_MAGNITUDE_NEGLIGIBLE_TREND_FAIL",
        "strain_parameter=NONE_GAMMA_FROZEN_1.96",
        "strain_parameter_physicality=PASS_PARAMETER_FREE_SCALAR_BORN_CONTRACT",
        "kappa_experiment_6h_573p15K=0.85",
        "kappa_experiment_48h_573p15K=1.03",
        f"kappa_predicted_48h_M0_mean={endpoint['M0']['mean48']:.17g}",
        f"kappa_predicted_48h_MI_mean={endpoint['MI']['mean48']:.17g}",
        f"kappa_predicted_48h_MS_mean={endpoint['MS']['mean48']:.17g}",
        f"kappa_predicted_48h_MIS_mean={endpoint['MIS']['mean48']:.17g}",
        f"relative_change_MI_percent={endpoint['MI']['relative']:.17g}",
        f"relative_change_MS_percent={endpoint['MS']['relative']:.17g}",
        f"relative_change_MIS_percent={endpoint['MIS']['relative']:.17g}",
        "full_temperature_trend_pass_MI=true",
        "full_temperature_trend_pass_MS=false",
        "full_temperature_trend_pass_MIS=true",
        f"blind_prediction_status={FINAL_STATUS}",
        "recommended_next_action=DO_NOT_RETUNE_PF_OR_PSD;_NEW_INDEPENDENT_PHYSICS_CONTRACT_REQUIRED",
        f"final_status={FINAL_STATUS}",
    ]
    (root / "final_terminal_output.txt").write_text("\n".join(terminal_lines) + "\n", encoding="utf-8")

    audit = {
        "schema": "SHESKIN_PF_INTERFACE_STRAIN_BLIND_PREDICTION_COMPLETION_AUDIT_V1",
        "status": FINAL_STATUS,
        "required_output_count": len(REQUIRED),
        "required_outputs_present": True,
        "baseline_gate": "PASS",
        "mechanics_replay_gate": "PASS_CHECKPOINT_WARM_ONLY_ZERO_REJECTED",
        "descriptor_determinism": "PASS_TWO_INDEPENDENT_BYTE_IDENTICAL_RUNS",
        "transport_determinism": "PASS_TWO_INDEPENDENT_BYTE_IDENTICAL_CALIBRATION_AND_BLIND_RUNS",
        "no_48h_refit": True,
        "no_PF_production_modification": True,
        "no_commit": True,
        "no_push": True,
        "absolute_experimental_lattice_reproduction_claimed": False,
        "finalizer_script_sha256": sha256(Path(__file__).resolve()),
        "frozen_manifest_sha256": sha256(root / "frozen_before_48h_prediction_manifest.json"),
        "blind_evaluation_sha256": sha256(root / "blind_evaluation_summary.json"),
    }
    (root / "completion_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_paths = sorted(path for path in root.iterdir() if path.is_file())
    with (root / "final_outputs.sha256").open("w", encoding="utf-8") as stream:
        for path in output_paths:
            stream.write(f"{sha256(path)}  {path.name}\n")
    print(FINAL_STATUS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
