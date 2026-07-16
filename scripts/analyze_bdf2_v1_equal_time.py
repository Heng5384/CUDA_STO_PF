#!/usr/bin/env python3
"""Compare the dt/8 BDF2 candidate with a dt/16 equal-time reference."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def only(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} below {root}, got {matches}")
    return matches[0]


def h(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def l2(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def right_interface_position(phi: np.ndarray) -> float:
    center = int(np.argmax(phi))
    for offset in range(1, phi.size // 2 + 1):
        left = (center + offset - 1) % phi.size
        right = (center + offset) % phi.size
        a = float(phi[left])
        b = float(phi[right])
        if a >= 0.5 and b < 0.5:
            return center + offset - 1 + (a - 0.5) / (a - b)
    return math.nan


def load_state(root: Path, step: int) -> dict[str, np.ndarray]:
    return {
        "Ctot": np.fromfile(only(root, f"ctot_checkpoint_step{step:06d}_Ctot.raw"), np.float64),
        "phi": np.fromfile(only(root, f"ctot_checkpoint_step{step:06d}_phi.raw"), np.float64),
        "xB": np.fromfile(only(root, f"ctot_checkpoint_step{step:06d}_xB_alpha.raw"), np.float64),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    candidate = load_state(
        repo / "runs/bdf2_v1/fixed_dt_qualification/dt_div8/run", 1000
    )
    reference = load_state(
        repo / "runs/bdf2_v1/fixed_dt_equal_time_reference/dt_div16_2000/run", 2000
    )
    initial = {
        "Ctot": np.fromfile(
            repo / "runs/frozen_input/ctot_checkpoint_step000054_Ctot.raw", np.float64
        ),
        "phi": np.fromfile(
            repo / "runs/frozen_input/ctot_checkpoint_step000054_phi.raw", np.float64
        ),
        "xB": np.fromfile(
            repo / "runs/frozen_input/ctot_checkpoint_step000054_xB_alpha.raw", np.float64
        ),
    }
    h0, hc, hr = h(initial["phi"]), h(candidate["phi"]), h(reference["phi"])
    alpha_reference = 1.0 - hr
    transfer_candidate = 0.5 * float(np.sum(np.abs(candidate["Ctot"] - initial["Ctot"])))
    transfer_reference = 0.5 * float(np.sum(np.abs(reference["Ctot"] - initial["Ctot"])))
    candidate_interface = right_interface_position(candidate["phi"])
    reference_interface = right_interface_position(reference["phi"])
    initial_interface = right_interface_position(initial["phi"])
    matrix_l1 = float(np.mean(np.abs(candidate["xB"] - reference["xB"])))
    matrix_profile_scale = float(np.mean(np.abs(reference["xB"])))
    weighted_matrix_error = float(
        np.sum(alpha_reference * np.abs(candidate["xB"] - reference["xB"]))
    )
    weighted_matrix_scale = float(
        np.sum(alpha_reference * np.abs(reference["xB"]))
    )
    row = {
        "candidate": "dt_div8_1000",
        "reference": "dt_div16_2000",
        "equal_time_code": 0.390625,
        "equal_time_physical_s": 16.066244306466707,
        "Ctot_field_L2_error": l2(candidate["Ctot"] - reference["Ctot"]),
        "Ctot_field_Linf_error": float(
            np.max(np.abs(candidate["Ctot"] - reference["Ctot"]))
        ),
        "Ctot_increment_relative_L2_error": l2(
            candidate["Ctot"] - reference["Ctot"]
        ) / max(l2(reference["Ctot"] - initial["Ctot"]), 1.0e-30),
        "phi_increment_relative_L2_error": l2(
            candidate["phi"] - reference["phi"]
        ) / max(l2(reference["phi"] - initial["phi"]), 1.0e-30),
        "hvolume_candidate": float(np.sum(hc)),
        "hvolume_reference": float(np.sum(hr)),
        "hvolume_relative_error": abs(float(np.sum(hc - hr)))
        / max(abs(float(np.sum(hr))), 1.0e-30),
        "hvolume_increment_relative_error": abs(
            float(np.sum((hc - h0) - (hr - h0)))
        ) / max(abs(float(np.sum(hr - h0))), 1.0e-30),
        "interface_candidate": candidate_interface,
        "interface_reference": reference_interface,
        "interface_error_dx": abs(candidate_interface - reference_interface),
        "interface_direction_same": (
            math.copysign(1.0, candidate_interface - initial_interface)
            == math.copysign(1.0, reference_interface - initial_interface)
        ),
        "cumulative_transfer_candidate": transfer_candidate,
        "cumulative_transfer_reference": transfer_reference,
        "cumulative_transfer_relative_error": abs(
            transfer_candidate - transfer_reference
        ) / max(abs(transfer_reference), 1.0e-30),
        "matrix_profile_unweighted_L1_error": matrix_l1,
        "matrix_profile_unweighted_relative_L1_error": matrix_l1
        / max(matrix_profile_scale, 1.0e-30),
        "matrix_profile_capacity_weighted_relative_L1_error": weighted_matrix_error
        / max(weighted_matrix_scale, 1.0e-30),
        "matrix_profile_unweighted_Linf_error": float(
            np.max(np.abs(candidate["xB"] - reference["xB"]))
        ),
        "mass_difference_abs": abs(
            float(np.sum(candidate["Ctot"]) - np.sum(reference["Ctot"]))
        ),
    }
    row["medium_equal_time_pass"] = (
        row["cumulative_transfer_relative_error"] <= 0.05
        and row["hvolume_increment_relative_error"] <= 0.05
        and row["interface_error_dx"] <= 0.5
        and row["matrix_profile_capacity_weighted_relative_L1_error"] <= 0.05
        and row["interface_direction_same"]
    )
    root = repo / "runs/bdf2_v1/fixed_dt_equal_time_analysis"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "equal_time_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (root / "summary.json").write_text(
        json.dumps(row, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(row, indent=2))
    return 0 if row["medium_equal_time_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
