#!/usr/bin/env python3
"""Prepare the workstation-only fixed-step BDF2 startup smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPLACEMENTS = {
    "ctot_numerics_contract": "ctot_jichen_imex_bdf2_v1",
    "ctot_split_defect_policy": "IMEX_BDF2_NO_POST_PHASE_POLISH",
    "ctot_max_coupling_correctors": "0",
    "ctot_outer_acceleration": "OFF",
    "ctot_automatic_dt_growth": "0",
    "ctot_step_max_retries": "0",
    "ctot_finite_interface_antitrapping_enabled": "0",
    "coarse_interface_mobility_a_M": "0.00000000000000000e+00",
    "elastic_enabled": "0",
    "diagnostic_rsmd_enabled": "0",
    "enable_gp_assisted_beta_nucleation": "0",
    "gp_growth_enabled": "0",
}


def rewrite_params(
    source: Path,
    destination: Path,
    case_tag: str,
    dt: float,
    extra_replacements: dict[str, str] | None = None,
) -> None:
    replacements = dict(REPLACEMENTS)
    replacements["init_case_tag"] = case_tag
    replacements["dt"] = f"{dt:.17e}"
    if extra_replacements:
        replacements.update(extra_replacements)
    seen: set[str] = set()
    output: list[str] = []
    for raw in source.read_text(encoding="utf-8").splitlines():
        if "=" in raw and not raw.lstrip().startswith("#"):
            key = raw.split("=", 1)[0].strip()
            if key in replacements:
                output.append(f"{key}={replacements[key]}")
                seen.add(key)
                continue
        output.append(raw)
    for key, value in replacements.items():
        if key not in seen:
            output.append(f"{key}={value}")
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")


def rewrite_meta(source: Path, destination: Path) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    data.update(
        {
            "ctot_numerics_contract": "ctot_jichen_imex_bdf2_v1",
            "ctot_split_defect_policy": "IMEX_BDF2_NO_POST_PHASE_POLISH",
            "ctot_max_coupling_correctors": 0,
            "time_integrator": "FIXED_STEP_IMEX_BDF2_V1",
            "history_contract_version": "FIXED_STEP_BDF2_ACCEPTED_HISTORY_V1",
            "phase_context_version": "PHI_EXTRAPOLATION_2N_MINUS_NM1_ULP64_V1",
            "BDF2_energy_contract_version": "BDF2_ENDPOINT_DISCRETE_WORK_V1",
            "bdf2_history_valid": 0,
            "bdf2_restart_fallback_pending": 0,
            "bdf2_last_fallback_reason": "startup_history_unavailable",
        }
    )
    destination.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--dt", type=float, default=0.003125)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    repo = args.repo.resolve()
    base = repo / "runs/preparation_v2/cases/T400_step655_lie_be_v2_dt_div1_full"
    frozen = repo / "runs/frozen_input"
    output = repo / "runs/bdf2_v1/startup_smoke"
    output.mkdir(parents=True, exist_ok=True)
    params = output / "runtime.params"
    meta = output / "init_meta.json"
    rewrite_params(base / "runtime.params", params, "T400_bdf2_v1_startup_smoke", args.dt)
    rewrite_meta(base / "init_meta.json", meta)
    command = [
        str(repo / "main_cuda"), "512", "1", "1", f"{args.dt:.17e}",
        str(args.steps), str(args.steps), "1", "0", "--mode", "dynamics",
        "--pf-param-file", str(params), "--init-mode", "raw_fields",
        "--init-phi-raw", str(frozen / "ctot_checkpoint_step000054_phi.raw"),
        "--init-xB-raw", str(frozen / "ctot_checkpoint_step000054_xB_alpha.raw"),
        "--init-Ctot-raw", str(frozen / "ctot_checkpoint_step000054_Ctot.raw"),
        "--init-meta", str(meta), "--init-case-tag", "T400_bdf2_v1_startup_smoke",
    ]
    (output / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
