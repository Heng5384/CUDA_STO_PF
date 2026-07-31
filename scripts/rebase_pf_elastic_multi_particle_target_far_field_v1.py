#!/usr/bin/env python3
"""Rebase a conserved V2 multi-particle fixture to a pinned matrix far field.

The source V2 fixture already owns the translated, hash-pinned single-particle
elastic shapes and the phase-consistent local matrix correction

    delta_x_alpha_total = sum_j(xB_alpha,j - xB_far,j).

This tool keeps that geometry and correction byte-identical and replaces only
the uniform target-box matrix baseline:

    xB_alpha = target_matrix_xB + delta_x_alpha_total
    C_B_tot  = h(phi) + (1-h(phi))*xB_alpha.

Consequently the global inventory is *derived* from the requested matrix
baseline and the frozen resolved-beta population.  It is not independently
forced to 0.03.  This is a validation-only sensitivity fixture, not a
production 6 h initial state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np


SOURCE_SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_6H_FIXTURE_V2"
COMPOSITION_CONTRACT = "PHASE_CONSISTENT_DELTA_X_ALPHA_V2"
REBASE_SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_TARGET_FAR_FIELD_REBASE_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verified_field(manifest: Dict[str, Any], root: Path, name: str, count: int) -> Path:
    row = manifest.get("fields", {}).get(name)
    if not isinstance(row, dict):
        raise ValueError(f"source fixture is missing field {name}")
    path = root / str(row.get("path", ""))
    if not path.is_file() or path.stat().st_size != count * 8:
        raise ValueError(f"source field {name} is missing or has the wrong size")
    if sha256(path) != row.get("sha256"):
        raise ValueError(f"source field {name} SHA-256 mismatch")
    return path


def raw(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    value = np.fromfile(path, dtype="<f8")
    if value.size != math.prod(shape):
        raise ValueError(f"{path}: wrong raw element count")
    return value.reshape(shape, order="C")


def write_raw(path: Path, value: np.ndarray) -> None:
    np.asarray(value, dtype="<f8").ravel(order="C").tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-fixture", type=Path, required=True)
    parser.add_argument("--source-fixture-sha256", required=True)
    parser.add_argument("--target-matrix-xB", type=float, required=True)
    parser.add_argument("--target-matrix-xAg", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source_manifest_path = args.source_fixture.resolve()
    if not source_manifest_path.is_file():
        raise SystemExit("[fatal] source fixture manifest is missing")
    if sha256(source_manifest_path) != args.source_fixture_sha256:
        raise SystemExit("[fatal] source fixture manifest SHA-256 mismatch")
    source_root = source_manifest_path.parent
    source = load_json(source_manifest_path)
    if source.get("schema") != SOURCE_SCHEMA:
        raise SystemExit("[fatal] source fixture is not V2")
    if source.get("fixture_kind") != "E2" or source.get("validation_only") is not True:
        raise SystemExit("[fatal] only a validation-only V2 E2 fixture may be rebased")
    if source.get("combination", {}).get("composition_contract") != COMPOSITION_CONTRACT:
        raise SystemExit("[fatal] source fixture has the wrong composition contract")
    if any(
        source.get("physical_contract", {}).get(key)
        for key in (
            "GP_enabled", "GP_birth_enabled", "GP_release_enabled",
            "external_source_enabled", "new_beta_nucleation_enabled",
        )
    ):
        raise SystemExit("[fatal] source fixture enables a prohibited path")

    target_xb = float(args.target_matrix_xB)
    expected_xag = 2.0 * target_xb / (2.0 + target_xb)
    if not math.isfinite(target_xb) or not 0.0 < target_xb < 0.5:
        raise SystemExit("[fatal] target matrix xB is outside the admissible range")
    if abs(expected_xag - float(args.target_matrix_xAg)) > 1.0e-15:
        raise SystemExit("[fatal] target xB/xAg conversion is inconsistent")
    if args.out.exists():
        raise SystemExit(f"[fatal] refusing to overwrite output root: {args.out}")
    args.out.mkdir(parents=True)

    grid = source["grid"]
    shape = tuple(int(grid[axis]) for axis in ("Nx", "Ny", "Nz"))
    count = math.prod(shape)
    paths = {
        name: verified_field(source, source_root, name, count)
        for name in (
            "phi", "h_phi", "delta_x_alpha_total",
            "delta_x_alpha_source_total", "delta_C_relaxation_total",
        )
    }
    phi = raw(paths["phi"], shape)
    h = raw(paths["h_phi"], shape)
    delta_x = raw(paths["delta_x_alpha_total"], shape)
    if not all(np.all(np.isfinite(value)) for value in (phi, h, delta_x)):
        raise SystemExit("[fatal] source fixture contains non-finite values")
    x_b = target_xb + delta_x
    x_b_min_safe = float(source.get("phase_storage_contract", {}).get("xB_min_safe", 1.0e-12))
    x_b_max_safe = float(source.get("phase_storage_contract", {}).get("xB_max_safe", 0.499999))
    # Older V2 manifests record the bounds in init_meta/inventory rather than
    # phase_storage_contract.  Preserve their registered defaults.
    x_b_min_safe = float(source.get("inventory", {}).get("xB_min_safe", x_b_min_safe))
    x_b_max_safe = float(source.get("inventory", {}).get("xB_max_safe", x_b_max_safe))
    if float(np.min(x_b)) < x_b_min_safe or float(np.max(x_b)) > x_b_max_safe:
        raise SystemExit("[fatal] target far-field rebase would require xB clipping")
    c_b = h + (1.0 - h) * x_b
    y = np.log(x_b / (1.0 - x_b))
    d_y = np.zeros(shape, dtype=np.float64)
    if not all(np.all(np.isfinite(value)) for value in (c_b, y)):
        raise SystemExit("[fatal] rebased canonical fields are non-finite")

    copied_names = (
        "phi", "h_phi", "delta_x_alpha_total",
        "delta_x_alpha_source_total", "delta_C_relaxation_total",
    )
    fields: Dict[str, Dict[str, Any]] = {}
    for name in copied_names:
        destination = args.out / paths[name].name
        shutil.copyfile(paths[name], destination)
        fields[name] = {
            "path": destination.name, "sha256": sha256(destination),
            "dtype": "float64-le", "order": "C",
        }
    generated = {
        "xB_alpha": x_b,
        "C_B_tot": c_b,
        "Y": y,
        "dY_dt_prev": d_y,
    }
    for name, value in generated.items():
        destination = args.out / f"{name}.raw.f64"
        write_raw(destination, value)
        fields[name] = {
            "path": destination.name, "sha256": sha256(destination),
            "dtype": "float64-le", "order": "C",
        }

    for name in ("initial_particles.csv", "initial_components.csv"):
        source_path = source_root / name
        if not source_path.is_file():
            raise SystemExit(f"[fatal] source fixture is missing {name}")
        shutil.copyfile(source_path, args.out / name)

    mean_c = float(np.mean(c_b, dtype=np.float64))
    total_c = float(np.sum(c_b, dtype=np.float64))
    alpha = 1.0 - h
    far_mask = h < float(source["component_contract"]["h_threshold"])
    far_mean = float(np.mean(x_b[far_mask], dtype=np.float64)) if np.any(far_mask) else math.nan
    contract = {
        "schema": REBASE_SCHEMA,
        "source_fixture_manifest_path": str(source_manifest_path),
        "source_fixture_manifest_sha256": sha256(source_manifest_path),
        "target_matrix_xB": target_xb,
        "target_matrix_xAg": expected_xag,
        "global_inventory_mode": "DERIVED_FROM_TARGET_MATRIX_XB_AND_FROZEN_RESOLVED_BETA_POPULATION",
        "derived_mean_C_B_tot": mean_c,
        "source_mean_C_B_tot": float(source["inventory"]["actual_mean_C_B_tot"]),
        "mean_C_B_tot_change": mean_c - float(source["inventory"]["actual_mean_C_B_tot"]),
        "phi_bytewise_preserved": True,
        "delta_x_alpha_total_bytewise_preserved": True,
        "clipping_used": False,
        "normalization_used": False,
    }
    write_json(args.out / "target_far_field_contract.json", contract)

    meta = load_json(source_root / "init_meta.json")
    meta.update({
        "schema": "PF_ELASTIC_MULTI_PARTICLE_TARGET_FAR_FIELD_RAW_INIT_META_V1",
        "mean_xBtot": mean_c,
        "xB_min_safe": x_b_min_safe,
        "xB_max_safe": x_b_max_safe,
        "phi_path": "phi.raw.f64",
        "xB_path": "xB_alpha.raw.f64",
        "target_matrix_xB": target_xb,
        "target_matrix_xAg": expected_xag,
        "global_inventory_mode": contract["global_inventory_mode"],
    })
    write_json(args.out / "init_meta.json", meta)

    manifest = dict(source)
    manifest.update({
        "fixture_id": "pf_elastic_multi_particle_six_particle_96cube_target_far_field_v1",
        "scientific_status": "ENGINEERING_TARGET_FAR_FIELD_SENSITIVITY_ONLY_NOT_AN_EXPERIMENTAL_6H_PSD_CANDIDATE",
        "source_fixture": {
            "path": str(source_manifest_path),
            "sha256": sha256(source_manifest_path),
            "fixture_id": source.get("fixture_id"),
        },
        "target_far_field_rebase": {
            **contract,
            "contract_path": "target_far_field_contract.json",
            "contract_sha256": sha256(args.out / "target_far_field_contract.json"),
            "far_matrix_h_threshold": float(source["component_contract"]["h_threshold"]),
            "initial_far_matrix_xB_mean": far_mean,
        },
        "fields": fields,
        "inventory": {
            **source["inventory"],
            "target_mean_C_B_tot": mean_c,
            "target_total_C_B_tot": total_c,
            "actual_mean_C_B_tot": mean_c,
            "actual_total_C_B_tot": total_c,
            "relative_error": 0.0,
            "matrix_xB": target_xb,
            "matrix_xAg": expected_xag,
            "reference_matrix_xB": target_xb,
            "source_mean_C_B_tot": float(source["inventory"]["actual_mean_C_B_tot"]),
            "mean_C_B_tot_change": mean_c - float(source["inventory"]["actual_mean_C_B_tot"]),
        },
        "time_level_contract": {
            "Y": "logit(xB_alpha)",
            "dY_dt_prev": "zero_for_fresh_dynamic_start",
            "dY_dt_prev_nonzero": False,
        },
        "init_meta": {
            "path": "init_meta.json",
            "sha256": sha256(args.out / "init_meta.json"),
        },
        "initial_particles": {
            "path": "initial_particles.csv",
            "sha256": sha256(args.out / "initial_particles.csv"),
        },
        "initial_components": {
            "path": "initial_components.csv",
            "sha256": sha256(args.out / "initial_components.csv"),
        },
    })
    manifest["combination"] = {
        **source["combination"],
        "target_box_matrix_baseline": "PINNED_TARGET_MATRIX_XB_V1",
        "global_inventory": "DERIVED_NOT_INDEPENDENTLY_FORCED",
        "clipping_used": False,
        "normalization_used": False,
    }
    write_json(args.out / "fixture_manifest.json", manifest)

    # Final self-audit after every output hash is frozen.
    loaded = load_json(args.out / "fixture_manifest.json")
    for name, row in loaded["fields"].items():
        if sha256(args.out / row["path"]) != row["sha256"]:
            raise SystemExit(f"[fatal] output field hash mismatch: {name}")
    if not np.array_equal(raw(args.out / "phi.raw.f64", shape), phi):
        raise SystemExit("[fatal] phi changed during far-field rebase")
    if not np.array_equal(raw(args.out / "delta_x_alpha_total.raw.f64", shape), delta_x):
        raise SystemExit("[fatal] local matrix correction changed during far-field rebase")

    print(json.dumps({
        "status": "PASS_PF_ELASTIC_MULTI_PARTICLE_TARGET_FAR_FIELD_REBASE_V1",
        "fixture_manifest_sha256": sha256(args.out / "fixture_manifest.json"),
        "target_matrix_xB": target_xb,
        "target_matrix_xAg": expected_xag,
        "initial_far_matrix_xB_mean": far_mean,
        "derived_mean_C_B_tot": mean_c,
        "source_mean_C_B_tot": float(source["inventory"]["actual_mean_C_B_tot"]),
        "mean_C_B_tot_change": mean_c - float(source["inventory"]["actual_mean_C_B_tot"]),
        "xB_min": float(np.min(x_b)),
        "xB_max": float(np.max(x_b)),
        "alpha_mean": float(np.mean(alpha, dtype=np.float64)),
        "component_count": int(source["component_contract"]["actual_count"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
