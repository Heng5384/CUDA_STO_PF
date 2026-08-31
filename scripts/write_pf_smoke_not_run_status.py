#!/usr/bin/env python3
"""Write the required A--E smoke table without inventing unmeasured PF data.

This is a status artifact, not a PF trajectory generator.  It is used only
when the configured CUDA compiler and PF binary are unavailable, so every
numerical measurement cell stays empty and every row is labelled accordingly.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "kwn_pf_state_closure_v1" / "pf_smoke_trajectories.csv"
CONTRACT = ROOT / "contracts" / "pf_kwn_validation_contract_v1.json"

CASES = (
    "A_BASELINE_LEGACY_ZERO_AUX",
    "B_IDENTITY_ADAPTER_ZERO_AUX",
    "C_NONZERO_FROZEN_AUX_STORAGE",
    "D_CONSERVATIVE_MATRIX_TO_GP_CONTROL",
    "E_FIXTURE_CONDITIONED_KWN_HANDOFF_V2",
)
TIMES_H = (0, 6, 48)
MEASUREMENT_COLUMNS = (
    "Q_B_matrix_mol",
    "Q_B_GP_mol",
    "Q_B_beta_subgrid_mol",
    "Q_B_beta_resolved_mol",
    "Q_B_total_mol",
    "relative_residual",
    "xB_matrix_mean",
    "xB_matrix_min",
    "xB_matrix_max",
    "beta_volume_fraction",
    "particle_count",
    "mean_radius_m",
    "S_v_m_minus_1",
    "free_energy_components",
    "clipping_count",
    "nan_inf_count",
    "restart_hash",
    "gp_subgrid_psd_checksum",
    "maximum_matrix_composition_pulse",
    "beta_volume_transient",
    "particle_loss",
    "energy_jump",
    "profile_relaxation_time_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # The JSON deliberately does not store its own hash.  Reuse the loader
    # instead of replicating its canonicalization rules here.
    sys.path.insert(0, str(ROOT / "src"))
    from kwn_mvp.contract import load_validation_contract  # noqa: PLC0415

    contract_hash = load_validation_contract(CONTRACT).sha256

    nvcc_path = shutil.which("nvcc")
    binary_path = ROOT / "main_cuda"
    if nvcc_path is not None or binary_path.is_file():
        raise SystemExit(
            "This status-only artifact is reserved for an unavailable CUDA/PF runtime; "
            "run the actual smoke matrix instead."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "case",
        "time_h",
        "execution_status",
        "evidence_level",
        "status_reason",
        "runtime_adapter_status",
        "contract_hash",
        *MEASUREMENT_COLUMNS,
    )
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=header,
            lineterminator="\n",
        )
        writer.writeheader()
        for case in CASES:
            for time_h in TIMES_H:
                row = {
                    "case": case,
                    "time_h": time_h,
                    "execution_status": "NOT_RUN_NO_CUDA_OR_PF_BINARY",
                    "evidence_level": "NOT_A_PF_TRAJECTORY",
                    "status_reason": "NVCC_NOT_FOUND;MAIN_CUDA_BINARY_NOT_FOUND",
                    "runtime_adapter_status": "SOURCE_INTEGRATED_UNCOMPILED_NOT_RUN",
                    "contract_hash": contract_hash,
                }
                row.update({column: "" for column in MEASUREMENT_COLUMNS})
                writer.writerow(row)
    print(
        json.dumps(
            {
                "status": "NOT_RUN_NO_CUDA_OR_PF_BINARY",
                "rows": len(CASES) * len(TIMES_H),
                "contract_hash": contract_hash,
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
