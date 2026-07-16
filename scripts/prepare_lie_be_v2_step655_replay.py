#!/usr/bin/env python3
"""Prepare immutable-input Lie-BE v2 step655 full/half-step replays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PHYSICS_MODEL = "pbte_ag2te_gp_coarse4_stoich_rd_v2"
LIE_BE_V2 = "ctot_jichen_lie_be_v2"
LIE_POLICY = "LIE_NO_POST_PHASE_POLISH"
BASE_DT = 0.003125
DIVISORS = (1, 2, 4, 8, 16, 32, 64, 128)


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
    output.extend(("", "# Lie-BE v2 numerical view; physical inputs unchanged."))
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
        "elastic_enabled": "0",
    }
    mismatch = {
        key: (base.get(key), expected)
        for key, expected in required.items()
        if base.get(key) != expected
    }
    if mismatch:
        raise ValueError(f"frozen physics contract mismatch: {mismatch}")


def build_case(
    root: Path,
    base_params: Path,
    base_meta: dict[str, object],
    divisor: int,
    half_reference: bool,
) -> dict[str, object]:
    dt = BASE_DT / divisor
    run_dt = dt / 2.0 if half_reference else dt
    steps = 2 if half_reference else 1
    suffix = "two_half" if half_reference else "full"
    case_id = f"T400_step655_lie_be_v2_dt_div{divisor}_{suffix}"
    case_dir = root / case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    overrides = {
        "PF_RESEARCH_MODEL": PHYSICS_MODEL,
        "coarse_model_name": PHYSICS_MODEL,
        "coarse_model_version": "2",
        "ctot_numerics_contract": LIE_BE_V2,
        "ctot_split_defect_policy": LIE_POLICY,
        "ctot_max_coupling_correctors": "0",
        "ctot_split_defect_skip_threshold": "-1.00000000000000000e+00",
        "ctot_split_defect_hard_cap": "-1.00000000000000000e+00",
        "ctot_split_defect_scale": "1.00000000000000008e-30",
        "ctot_outer_acceleration": "OFF",
        "ctot_debug_transport_floor_audit": "0",
        "ctot_automatic_dt_growth": "0",
        "ctot_step_max_retries": "0",
        "dt": f"{run_dt:.17e}",
        "init_case_tag": case_id,
    }
    params = case_dir / "runtime.params"
    write_params(base_params, params, overrides)

    meta = dict(base_meta)
    meta.update(
        {
            "PF_RESEARCH_MODEL": PHYSICS_MODEL,
            "ctot_numerics_contract": LIE_BE_V2,
            "ctot_split_defect_policy": LIE_POLICY,
            "ctot_max_coupling_correctors": 0,
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
        "divisor": divisor,
        "comparison_dt": dt,
        "run_dt": run_dt,
        "steps": steps,
        "half_reference": half_reference,
        "equal_physical_time": dt,
        "params": str(params),
        "params_sha256": sha256(params),
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

    _, base = parse_params(args.base_params)
    validate_frozen_physics(base)
    base_meta = json.loads(args.base_meta.read_text(encoding="utf-8"))
    raw = {
        "phi": Path(str(args.raw_prefix) + "_phi.raw"),
        "xB_alpha": Path(str(args.raw_prefix) + "_xB_alpha.raw"),
        "Ctot": Path(str(args.raw_prefix) + "_Ctot.raw"),
    }
    for path in raw.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    cases_root = args.output_root / "cases"
    cases_root.mkdir(parents=True)
    cases = [
        build_case(cases_root, args.base_params, base_meta, divisor, half)
        for divisor in DIVISORS
        for half in (False, True)
    ]
    manifest = {
        "method": LIE_BE_V2,
        "policy": LIE_POLICY,
        "physics_contract": PHYSICS_MODEL,
        "base_dt": BASE_DT,
        "divisors": list(DIVISORS),
        "reference_definition": "two half steps from the same frozen state",
        "frozen_state": {
            "raw_prefix": str(args.raw_prefix),
            "raw_sha256": {key: sha256(path) for key, path in raw.items()},
            "source_meta": str(args.base_meta),
            "source_meta_sha256": sha256(args.base_meta),
            "raw_state_modified": False,
        },
        "cases": cases,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"prepared_cases={len(cases)}")
    print("raw_state_modified=false")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
