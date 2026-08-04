#!/usr/bin/env python3
"""Select the largest stable production dt from converged PF observables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import qualify_pf_elastic_multi_particle_6h_dynamics_v1 as base


SCHEMA = "PF_PRODUCTION_DT_OBSERVABLE_CONVERGENCE_V1"
DEFAULT_FIXTURE_SCHEMA = "PF_MASS_CONSERVING_LIBRARY_HANDOFF_MANIFEST_V1"
PASS = "PASS_PRODUCTION_DT_OBSERVABLE_CONVERGENCE_V1"
FAIL = "BLOCKED_PRODUCTION_DT_OBSERVABLE_CONVERGENCE_V1"

# These thresholds are registered engineering/observable criteria, not fitted
# physical parameters.  Full-field xB MAE is intentionally diagnostic-only
# under the conditional-handoff policy.
HARD = {
    "phi_normalized_L1": 2.0e-2,
    "beta_volume_fraction_relative": 1.0e-2,
    "mean_radius_relative": 1.0e-2,
    "Sv_relative": 2.0e-2,
    "M6_relative": 5.0e-2,
    "matrix_xAg_absolute": 4.0e-4,
    "mean_elastic_energy_relative": 5.0e-2,
    "mass_relative": 1.0e-10,
}
PREFERRED = {
    "beta_volume_fraction_relative": 5.0e-3,
    "mean_radius_relative": 5.0e-3,
    "Sv_relative": 1.0e-2,
    "M6_relative": 2.0e-2,
    "matrix_xAg_absolute": 1.0e-4,
    "mean_elastic_energy_relative": 2.0e-2,
}
PHI_NUMERICAL_MIN = -1.0e-6
PHI_NUMERICAL_MAX = 1.0 + 1.0e-6


def final_elastic_row(path: Path) -> Dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty dynamics mass diagnostics: {path}")
    row = rows[-1]
    required = (
        "mean_elastic_energy",
        "max_elastic_energy",
        "stress_hydro_min",
        "stress_hydro_max",
    )
    values = {key: float(row[key]) for key in required}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"non-finite elastic diagnostic: {path}")
    return values


def wall_seconds(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    hits = re.findall(r"wall_time_s\s*:\s*([0-9.eE+-]+)", text)
    if not hits:
        raise ValueError(f"wall time not found: {path}")
    return float(hits[-1])


def relative(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-30)


def xag(xb: float) -> float:
    return 2.0 * xb / (2.0 + xb)


def state_metrics(
    manifest: Dict[str, Any],
    initial_phi: np.ndarray,
    initial_labels: np.ndarray,
    checkpoint: Dict[str, Any],
    elastic_csv: Path,
    stdout: Path,
) -> Dict[str, Any]:
    threshold = float(manifest["component_contract"]["h_threshold"])
    expected_count = int(manifest["component_contract"]["expected_count"])
    dx_nm = float(manifest["grid"]["dx_nm"])
    labels, rows = base.tracked_state(checkpoint["phi"], threshold, dx_nm)
    identity, mapping, overlap = base.match_components(
        initial_labels, labels, expected_count
    )
    identity = identity and len(rows) == expected_count
    radii = np.asarray(
        [float(row["equivalent_radius_nm"]) for row in rows],
        dtype=np.float64,
    )
    h = base.h_of_phi(checkpoint["phi"])
    box_volume_nm3 = float(np.prod(checkpoint["phi"].shape)) * dx_nm**3
    matrix_mask = h < 0.005
    if not np.any(matrix_mask):
        raise ValueError("no matrix cells remain under h<0.005")
    matrix_xB = float(
        np.mean(checkpoint["xB"][matrix_mask], dtype=np.float64)
    )
    canonical = h + (1.0 - h) * checkpoint["xB"]
    elastic = final_elastic_row(elastic_csv)
    finite = all(
        np.all(np.isfinite(checkpoint[name]))
        for name in ("phi", "xB", "Y", "dY")
    ) and all(math.isfinite(value) for value in elastic.values())
    phi_min = float(np.min(checkpoint["phi"]))
    phi_max = float(np.max(checkpoint["phi"]))
    xB_min = float(np.min(checkpoint["xB"]))
    xB_max = float(np.max(checkpoint["xB"]))
    bounds = (
        phi_min >= PHI_NUMERICAL_MIN
        and phi_max <= PHI_NUMERICAL_MAX
        and xB_min > 0.0
        and xB_max < 1.0
    )
    return {
        "dt_code": float(checkpoint["dt"]),
        "step": int(checkpoint["step"]),
        "code_time": float(checkpoint["dt"] * checkpoint["step"]),
        "particle_count": len(rows),
        "identity_exact": bool(identity),
        "identity_mapping": mapping,
        "minimum_initial_support_overlap": min(
            overlap.values(), default=0.0
        ),
        "beta_volume_fraction": float(np.mean(h, dtype=np.float64)),
        "mean_radius_nm": float(np.mean(radii)),
        "Sv_spherical_equivalent_nm^-1": float(
            4.0 * math.pi * np.sum(radii**2, dtype=np.float64)
            / box_volume_nm3
        ),
        "M6_integral_nm^3": float(
            np.sum(radii**6, dtype=np.float64) / box_volume_nm3
        ),
        "matrix_xB_h_lt_0p005": matrix_xB,
        "matrix_xAg_h_lt_0p005": xag(matrix_xB),
        "canonical_inventory_code": float(
            np.sum(canonical, dtype=np.float64)
        ),
        "mean_elastic_energy": elastic["mean_elastic_energy"],
        "max_elastic_energy": elastic["max_elastic_energy"],
        "stress_hydro_min": elastic["stress_hydro_min"],
        "stress_hydro_max": elastic["stress_hydro_max"],
        "finite": finite,
        "bounds": bounds,
        "phi_min": phi_min,
        "phi_max": phi_max,
        "xB_min": xB_min,
        "xB_max": xB_max,
        "wall_seconds": wall_seconds(stdout),
        "phi_normalized_L1_from_initial": float(
            np.sum(
                np.abs(checkpoint["phi"] - initial_phi),
                dtype=np.float64,
            )
            / max(
                float(
                    np.sum(base.h_of_phi(initial_phi), dtype=np.float64)
                ),
                1.0,
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--fixture-schema", default=DEFAULT_FIXTURE_SCHEMA)
    parser.add_argument("--case", action="append", nargs=5, metavar=(
        "LABEL", "CHECKPOINT", "STDOUT", "STDERR", "MASS_CSV"
    ), required=True)
    parser.add_argument("--t-real-unit-s", type=float, required=True)
    parser.add_argument("--restart-checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)
    manifest = json.loads(
        args.fixture_manifest.read_text(encoding="utf-8")
    )
    if manifest.get("schema") != args.fixture_schema:
        raise SystemExit("wrong conditional-handoff fixture schema")
    if manifest.get("full_field_xB_dt_MAE_blocking") is not False:
        raise SystemExit("fixture has the wrong full-field xB dt policy")
    root = args.fixture_manifest.parent
    shape = tuple(
        int(manifest["grid"][axis]) for axis in ("Nx", "Ny", "Nz")
    )
    initial_phi = base.field(manifest, root, "phi")
    initial_c = base.field(manifest, root, "C_B_tot")
    initial_labels, initial_rows = base.tracked_state(
        initial_phi,
        float(manifest["component_contract"]["h_threshold"]),
        float(manifest["grid"]["dx_nm"]),
    )
    expected_count = int(manifest["component_contract"]["expected_count"])
    if len(initial_rows) != expected_count:
        raise SystemExit("initial component count mismatch")
    target_inventory = float(np.sum(initial_c, dtype=np.float64))

    cases: Dict[str, Dict[str, Any]] = {}
    logs: Dict[str, str] = {}
    stderr_empty: Dict[str, bool] = {}
    for label, checkpoint_path, stdout_path, stderr_path, mass_csv in args.case:
        checkpoint = base.checkpoint(Path(checkpoint_path), shape)
        stdout = Path(stdout_path)
        stderr = Path(stderr_path)
        cases[label] = state_metrics(
            manifest,
            initial_phi,
            initial_labels,
            checkpoint,
            Path(mass_csv),
            stdout,
        )
        cases[label]["checkpoint_sha256"] = base.sha256(Path(checkpoint_path))
        logs[label] = stdout.read_text(
            encoding="utf-8", errors="replace"
        )
        stderr_empty[label] = not stderr.read_bytes()
    ordered = sorted(
        cases, key=lambda label: cases[label]["dt_code"], reverse=True
    )
    reference_label = min(
        cases, key=lambda label: cases[label]["dt_code"]
    )
    reference = cases[reference_label]
    endpoint_match = max(
        abs(row["code_time"] - reference["code_time"])
        for row in cases.values()
    ) <= 1.0e-12
    physical_endpoint_s = reference["code_time"] * args.t_real_unit_s

    comparisons: Dict[str, Dict[str, Any]] = {}
    candidates: Dict[str, Dict[str, bool]] = {}
    observable_names = (
        "beta_volume_fraction",
        "mean_radius_nm",
        "Sv_spherical_equivalent_nm^-1",
        "M6_integral_nm^3",
        "mean_elastic_energy",
    )
    for label in ordered:
        row = cases[label]
        phi_mae = float(
            np.mean(
                np.abs(
                    base.checkpoint(
                        Path(next(
                            item[1] for item in args.case
                            if item[0] == label
                        )),
                        shape,
                    )["phi"]
                    - base.checkpoint(
                        Path(next(
                            item[1] for item in args.case
                            if item[0] == reference_label
                        )),
                        shape,
                    )["phi"]
                ),
                dtype=np.float64,
            )
        )
        xB_mae = float(
            np.mean(
                np.abs(
                    base.checkpoint(
                        Path(next(
                            item[1] for item in args.case
                            if item[0] == label
                        )),
                        shape,
                    )["xB"]
                    - base.checkpoint(
                        Path(next(
                            item[1] for item in args.case
                            if item[0] == reference_label
                        )),
                        shape,
                    )["xB"]
                ),
                dtype=np.float64,
            )
        )
        errors = {
            "phi_full_field_MAE": phi_mae,
            "xB_full_field_MAE_diagnostic": xB_mae,
            "beta_volume_fraction_relative": relative(
                row["beta_volume_fraction"],
                reference["beta_volume_fraction"],
            ),
            "mean_radius_relative": relative(
                row["mean_radius_nm"], reference["mean_radius_nm"]
            ),
            "Sv_relative": relative(
                row["Sv_spherical_equivalent_nm^-1"],
                reference["Sv_spherical_equivalent_nm^-1"],
            ),
            "M6_relative": relative(
                row["M6_integral_nm^3"],
                reference["M6_integral_nm^3"],
            ),
            "matrix_xAg_absolute": abs(
                row["matrix_xAg_h_lt_0p005"]
                - reference["matrix_xAg_h_lt_0p005"]
            ),
            "mean_elastic_energy_relative": relative(
                row["mean_elastic_energy"],
                reference["mean_elastic_energy"],
            ),
            "mass_relative": abs(
                row["canonical_inventory_code"] - target_inventory
            )
            / max(abs(target_inventory), 1.0),
        }
        gates = {
            "endpoint": endpoint_match,
            "finite": row["finite"],
            "bounds": row["bounds"],
            "stderr_empty": stderr_empty[label],
            "zero_mode": "PF_ZERO_MODE_FINAL_AUDIT status=PASS" in logs[label],
            "prohibited_paths": all(
                token not in logs[label]
                for token in (
                    "GP_EVENT",
                    "GP_BIRTH",
                    "BETA_NUCLEATION_EVENT",
                    "source_event",
                )
            ),
            "particle_count": row["particle_count"] == expected_count,
            "particle_identity": row["identity_exact"],
            "mass": errors["mass_relative"] <= HARD["mass_relative"],
            "phi": (
                errors["phi_full_field_MAE"]
                / max(reference["beta_volume_fraction"], 1.0e-30)
                <= HARD["phi_normalized_L1"]
            ),
            "beta_volume_fraction": (
                errors["beta_volume_fraction_relative"]
                <= HARD["beta_volume_fraction_relative"]
            ),
            "mean_radius": (
                errors["mean_radius_relative"]
                <= HARD["mean_radius_relative"]
            ),
            "Sv": errors["Sv_relative"] <= HARD["Sv_relative"],
            "M6": errors["M6_relative"] <= HARD["M6_relative"],
            "matrix_xAg": (
                errors["matrix_xAg_absolute"]
                <= HARD["matrix_xAg_absolute"]
            ),
            "mean_elastic_energy": (
                errors["mean_elastic_energy_relative"]
                <= HARD["mean_elastic_energy_relative"]
            ),
        }
        comparisons[label] = {
            "reference_label": reference_label,
            "errors": errors,
            "gates": gates,
            "preferred": {
                key: errors[key] <= limit
                for key, limit in PREFERRED.items()
            },
        }
        candidates[label] = gates

    # Refinement must not make any primary observable farther from the finest
    # reference.  This is evaluated in dt order, excluding the reference.
    non_reference = [
        label for label in ordered if label != reference_label
    ]
    monotone = True
    if len(non_reference) >= 2:
        coarse, finer = non_reference[0], non_reference[1]
        coarse_errors = comparisons[coarse]["errors"]
        finer_errors = comparisons[finer]["errors"]
        monotone = all(
            finer_errors[key]
            <= coarse_errors[key] + 1.0e-12 * max(1.0, coarse_errors[key])
            for key in (
                "beta_volume_fraction_relative",
                "mean_radius_relative",
                "Sv_relative",
                "M6_relative",
                "matrix_xAg_absolute",
                "mean_elastic_energy_relative",
            )
        )
    for label in candidates:
        candidates[label]["refinement_monotone"] = monotone

    qualified = [
        label for label in ordered if all(candidates[label].values())
    ]
    selected_label = qualified[0] if qualified else None
    fallback_label = qualified[1] if len(qualified) > 1 else None
    restart_equal = None
    restart_sha256 = None
    if args.restart_checkpoint is not None:
        if selected_label is None:
            restart_equal = False
        else:
            selected_path = Path(
                next(
                    item[1]
                    for item in args.case
                    if item[0] == selected_label
                )
            )
            restart_equal = (
                selected_path.read_bytes()
                == args.restart_checkpoint.read_bytes()
            )
            restart_sha256 = base.sha256(args.restart_checkpoint)

    status = (
        PASS
        if selected_label is not None
        and (restart_equal is not False)
        else FAIL
    )
    result = {
        "schema": SCHEMA,
        "status": status,
        "analysis_script_sha256": base.sha256(Path(__file__)),
        "fixture_manifest_sha256": base.sha256(args.fixture_manifest),
        "t_real_unit_s": args.t_real_unit_s,
        "common_code_time": reference["code_time"],
        "common_physical_time_s": physical_endpoint_s,
        "common_endpoint_age_h": 6.0 + physical_endpoint_s / 3600.0,
        "thresholds": {
            "hard": HARD,
            "preferred": PREFERRED,
            "phi_numerical_bounds": [
                PHI_NUMERICAL_MIN,
                PHI_NUMERICAL_MAX,
            ],
        },
        "reference_label": reference_label,
        "cases": cases,
        "comparisons": comparisons,
        "refinement_monotone": monotone,
        "selected_label": selected_label,
        "selected_production_dt_code": (
            cases[selected_label]["dt_code"]
            if selected_label is not None
            else None
        ),
        "selected_production_dt_physical_s": (
            cases[selected_label]["dt_code"] * args.t_real_unit_s
            if selected_label is not None
            else None
        ),
        "fallback_label": fallback_label,
        "fallback_dt_code": (
            cases[fallback_label]["dt_code"]
            if fallback_label is not None
            else None
        ),
        "restart_bytewise_equal": restart_equal,
        "restart_checkpoint_sha256": restart_sha256,
        "full_field_xB_dt_MAE_blocking": False,
    }
    (args.out / "production_dt_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    terminal = {
        "production_dt_observable_convergence_status": status,
        "selected_production_dt_code": result[
            "selected_production_dt_code"
        ],
        "selected_production_dt_physical_s": result[
            "selected_production_dt_physical_s"
        ],
        "fallback_dt_code": result["fallback_dt_code"],
        "common_physical_time_s": physical_endpoint_s,
        "common_endpoint_age_h": result["common_endpoint_age_h"],
        "refinement_monotone": monotone,
        "restart_bytewise_equal": restart_equal,
        "full_field_xB_dt_MAE_blocking": False,
        "final_status": status,
    }
    (args.out / "final_terminal_output.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in terminal.items())
        + "\n",
        encoding="utf-8",
    )
    print(status)


if __name__ == "__main__":
    main()
