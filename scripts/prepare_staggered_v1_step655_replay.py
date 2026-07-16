#!/usr/bin/env python3
"""Prepare audited step655 reference/staggered replay contracts.

The accepted raw fields are immutable inputs.  This tool creates per-numerics
parameter files and metadata views that version the physics/numerics contract;
it never rewrites or copies the raw arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PHYSICS_MODEL = "pbte_ag2te_gp_coarse4_stoich_rd_v2"
REFERENCE = "ctot_fully_coupled_M3_polish_BE_v1"
STAGGERED = "ctot_jichen_staggered_defect1_BE_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_params(path: Path) -> tuple[list[str], dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    values: dict[str, str] = {}
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return lines, values


def write_params(source: Path, target: Path, overrides: dict[str, str]) -> None:
    lines, _ = parse_params(source)
    seen: set[str] = set()
    output: list[str] = []
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if "=" not in line:
            output.append(raw)
            continue
        key = line.split("=", 1)[0].strip()
        if key in overrides:
            output.append(f"{key}={overrides[key]}")
            seen.add(key)
        else:
            output.append(raw)
    output.append("")
    output.append("# Staggered-v1 contract view; physical parameters unchanged.")
    for key, value in overrides.items():
        if key not in seen:
            output.append(f"{key}={value}")
    target.write_text("\n".join(output) + "\n", encoding="utf-8")


def validate_frozen_physics(base: dict[str, str]) -> None:
    required = {
        "composition_evolution_mode": "ctot_mimetic_be",
        "D_compound": "0.00000000000000000e+00",
        "D_beta_for_calibration": "0.00000000000000000e+00",
        "v_B": "1.00000000000000000e+00",
        "coarse_interface_mobility_mode": "off",
        "coarse_interface_mobility_a_M": "0.00000000000000000e+00",
        "ctot_finite_interface_antitrapping_enabled": "0",
        "GP_population_mode": "OFF",
        "gp_growth_enabled": "0",
        "enable_legacy_gp_storage_coupling": "0",
        "diagnostic_rsmd_enabled": "0",
    }
    mismatch = {
        key: (base.get(key), expected)
        for key, expected in required.items()
        if base.get(key) != expected
    }
    if mismatch:
        raise ValueError(f"frozen physics contract mismatch: {mismatch}")


def build_case(
    case_root: Path,
    base_params: Path,
    base_meta: dict[str, object],
    case_id: str,
    numerics: str,
    policy: str,
    acceleration: str,
    skip_threshold: float,
    hard_cap: float,
    dt_override: float | None = None,
    replay_steps: int = 1,
    debug_transport_floor_audit: bool = False,
    diagnostic_only: bool = False,
) -> dict[str, object]:
    case_dir = case_root / case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    overrides = {
        "PF_RESEARCH_MODEL": PHYSICS_MODEL,
        "coarse_model_name": PHYSICS_MODEL,
        "coarse_model_version": "2",
        "ctot_numerics_contract": numerics,
        "ctot_split_defect_policy": policy,
        "ctot_max_coupling_correctors": "1",
        "ctot_split_defect_skip_threshold": f"{skip_threshold:.17e}",
        "ctot_split_defect_hard_cap": f"{hard_cap:.17e}",
        "ctot_split_defect_scale": "1.00000000000000008e-30",
        "ctot_outer_acceleration": acceleration,
        "ctot_debug_transport_floor_audit": (
            "1" if debug_transport_floor_audit else "0"
        ),
        "init_case_tag": case_id,
    }
    if dt_override is not None:
        overrides["dt"] = f"{dt_override:.17e}"
    params_path = case_dir / "runtime.params"
    write_params(base_params, params_path, overrides)

    meta = dict(base_meta)
    meta.update(
        {
            "PF_RESEARCH_MODEL": PHYSICS_MODEL,
            "ctot_numerics_contract": numerics,
            "ctot_split_defect_policy": policy,
            "ctot_max_coupling_correctors": 1,
            "coarse_model_name": PHYSICS_MODEL,
            "coarse_model_version": "2",
            "contract_view_migration": {
                "type": "metadata_only_same_accepted_raw_state",
                "source_checkpoint_schema": base_meta.get("schema"),
                "raw_state_modified": False,
            },
        }
    )
    meta_path = case_dir / "init_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {
        "case_id": case_id,
        "numerics_contract": numerics,
        "defect_policy": policy,
        "outer_acceleration": acceleration,
        "dt": dt_override,
        "replay_steps": replay_steps,
        "debug_transport_floor_audit": debug_transport_floor_audit,
        "diagnostic_only": diagnostic_only,
        "params": str(params_path),
        "params_sha256": sha256(params_path),
        "meta": str(meta_path),
        "meta_sha256": sha256(meta_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-params", type=Path, required=True)
    parser.add_argument("--base-meta", type=Path, required=True)
    parser.add_argument("--raw-prefix", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    _, base_values = parse_params(args.base_params)
    validate_frozen_physics(base_values)
    base_meta = json.loads(args.base_meta.read_text(encoding="utf-8"))
    expected_raw = {
        "phi": Path(str(args.raw_prefix) + "_phi.raw"),
        "xB_alpha": Path(str(args.raw_prefix) + "_xB_alpha.raw"),
        "Ctot": Path(str(args.raw_prefix) + "_Ctot.raw"),
    }
    for path in expected_raw.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite replay preparation: {args.output_root}"
        )
    cases_root = args.output_root / "cases"
    cases_root.mkdir(parents=True)

    cases = [
        build_case(
            cases_root, args.base_params, base_meta,
            "T400_step655_reference_M3_polish", REFERENCE,
            "ALWAYS_ONE_POLISH", "ANDERSON_M3_V1", -1.0, -1.0,
        ),
        build_case(
            cases_root, args.base_params, base_meta,
            "T400_step655_staggered_always", STAGGERED,
            "ALWAYS_ONE_POLISH", "OFF", -1.0, -1.0,
        ),
        build_case(
            cases_root, args.base_params, base_meta,
            "T400_step655_staggered_optional_precalibration", STAGGERED,
            "OPTIONAL_ONE_POLISH", "OFF", 1.0e-16, 1.0e-16,
        ),
        build_case(
            cases_root, args.base_params, base_meta,
            "T400_step655_staggered_always_dt2_equal_time", STAGGERED,
            "ALWAYS_ONE_POLISH", "OFF", -1.0, -1.0,
            dt_override=0.0015625, replay_steps=2,
        ),
        build_case(
            cases_root, args.base_params, base_meta,
            "T400_step655_staggered_always_dt4_equal_time", STAGGERED,
            "ALWAYS_ONE_POLISH", "OFF", -1.0, -1.0,
            dt_override=0.00078125, replay_steps=4,
        ),
        build_case(
            cases_root, args.base_params, base_meta,
            "T400_step655_staggered_always_dt8_probe", STAGGERED,
            "ALWAYS_ONE_POLISH", "OFF", -1.0, -1.0,
            dt_override=0.000390625, replay_steps=1,
        ),
        build_case(
            cases_root, args.base_params, base_meta,
            "T400_step655_staggered_always_dt16_probe", STAGGERED,
            "ALWAYS_ONE_POLISH", "OFF", -1.0, -1.0,
            dt_override=0.0001953125, replay_steps=1,
        ),
        build_case(
            cases_root, args.base_params, base_meta,
            "T400_step655_staggered_always_dt2_cold_kkt_audit", STAGGERED,
            "ALWAYS_ONE_POLISH", "OFF", -1.0, -1.0,
            dt_override=0.0015625, replay_steps=1,
            debug_transport_floor_audit=True,
        ),
        build_case(
            cases_root, args.base_params, base_meta,
            "T400_step655_staggered_always_dt32_closure_probe", STAGGERED,
            "ALWAYS_ONE_POLISH", "OFF", -1.0, -1.0,
            dt_override=0.00009765625, replay_steps=1,
        ),
        build_case(
            cases_root, args.base_params, base_meta,
            "T400_step655_staggered_always_dt64_closure_probe", STAGGERED,
            "ALWAYS_ONE_POLISH", "OFF", -1.0, -1.0,
            dt_override=0.000048828125, replay_steps=1,
        ),
        build_case(
            cases_root, args.base_params, base_meta,
            "T400_step655_staggered_always_dt128_closure_probe", STAGGERED,
            "ALWAYS_ONE_POLISH", "OFF", -1.0, -1.0,
            dt_override=0.0000244140625, replay_steps=1,
        ),
        build_case(
            cases_root, args.base_params, base_meta,
            "T400_step655_staggered_always_dt128_equal_time", STAGGERED,
            "ALWAYS_ONE_POLISH", "OFF", -1.0, -1.0,
            dt_override=0.0000244140625, replay_steps=128,
        ),
        build_case(
            cases_root, args.base_params, base_meta,
            "T400_step655_optional_skip_bracket_dt", STAGGERED,
            "OPTIONAL_ONE_POLISH", "OFF", 0.3, 0.3,
            dt_override=0.003125, replay_steps=1,
            diagnostic_only=True,
        ),
        build_case(
            cases_root, args.base_params, base_meta,
            "T400_step655_optional_skip_bracket_dt2", STAGGERED,
            "OPTIONAL_ONE_POLISH", "OFF", 0.3, 0.3,
            dt_override=0.0015625, replay_steps=2,
            diagnostic_only=True,
        ),
        build_case(
            cases_root, args.base_params, base_meta,
            "T400_step655_optional_skip_bracket_dt4", STAGGERED,
            "OPTIONAL_ONE_POLISH", "OFF", 0.3, 0.3,
            dt_override=0.00078125, replay_steps=4,
            diagnostic_only=True,
        ),
        build_case(
            cases_root, args.base_params, base_meta,
            "T400_step655_optional_skip_bracket_dt8", STAGGERED,
            "OPTIONAL_ONE_POLISH", "OFF", 0.3, 0.3,
            dt_override=0.000390625, replay_steps=8,
            diagnostic_only=True,
        ),
        build_case(
            cases_root, args.base_params, base_meta,
            "T400_step655_optional_skip_bracket_dt16", STAGGERED,
            "OPTIONAL_ONE_POLISH", "OFF", 0.3, 0.3,
            dt_override=0.0001953125, replay_steps=16,
            diagnostic_only=True,
        ),
        build_case(
            cases_root, args.base_params, base_meta,
            "T400_step655_optional_skip_bracket_dt32", STAGGERED,
            "OPTIONAL_ONE_POLISH", "OFF", 0.3, 0.3,
            dt_override=0.00009765625, replay_steps=32,
            diagnostic_only=True,
        ),
        build_case(
            cases_root, args.base_params, base_meta,
            "T400_step655_optional_skip_bracket_dt64", STAGGERED,
            "OPTIONAL_ONE_POLISH", "OFF", 0.3, 0.3,
            dt_override=0.000048828125, replay_steps=64,
            diagnostic_only=True,
        ),
        build_case(
            cases_root, args.base_params, base_meta,
            "T400_step655_optional_skip_bracket_dt128", STAGGERED,
            "OPTIONAL_ONE_POLISH", "OFF", 0.3, 0.3,
            dt_override=0.0000244140625, replay_steps=128,
            diagnostic_only=True,
        ),
    ]
    manifest = {
        "physics_contract": PHYSICS_MODEL,
        "frozen_state": {
            "raw_prefix": str(args.raw_prefix),
            "raw_sha256": {
                key: sha256(path) for key, path in expected_raw.items()
            },
            "source_meta": str(args.base_meta),
            "source_meta_sha256": sha256(args.base_meta),
            "raw_state_modified": False,
        },
        "cases": cases,
        "optional_precalibration_note": (
            "Threshold 1e-16 intentionally forces the one-polish path; it is "
            "not a qualified skip threshold and cannot support production skipping."
        ),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"prepared_cases={len(cases)}")
    print("raw_state_modified=false")
    print(f"manifest={args.output_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
