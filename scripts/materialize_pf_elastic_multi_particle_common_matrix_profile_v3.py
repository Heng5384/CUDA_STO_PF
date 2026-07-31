#!/usr/bin/env python3
"""Freeze the converged V3 common-matrix target profile for raw handoff."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np

import materialize_pf_elastic_multi_particle_6h_fixture_v1 as base


SEED_SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_COMMON_MATRIX_SEED_V3"
SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_COMMON_MATRIX_PROFILE_V3"
META_SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_COMMON_MATRIX_PROFILE_RAW_META_V3"
V4_SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_FIXED_PHI_COMMON_MATRIX_PROFILE_V4"
V4_META_SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_FIXED_PHI_COMMON_MATRIX_PROFILE_RAW_META_V4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_raw(path: Path, shape: tuple[int, int, int], label: str) -> np.ndarray:
    data = np.fromfile(path, dtype="<f8")
    if data.size != math.prod(shape):
        raise ValueError(f"{label} raw size mismatch")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{label} contains non-finite values")
    return data.reshape(shape, order="C")


def write_raw(path: Path, values: np.ndarray) -> None:
    np.asarray(values, dtype="<f8").ravel(order="C").tofile(path)


def require_stdout_contract(path: Path, require_fixed_phi: bool) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if "MINIMIZE_TARGET_PROFILE_CONVERGENCE_FINAL_AUDIT status=PASS" not in text:
        raise ValueError("constrained full-model minimization did not converge")
    if "MINIMIZE_MASS_CONSTRAINT_FINAL_AUDIT status=PASS" not in text:
        raise ValueError("constrained full-model minimization did not close mass")
    if "raw init required clamping" in text:
        raise ValueError("preparatory raw seed required clamping")
    fixed_phi_marker = (
        "MINIMIZE_FIXED_PHI_COMPOSITION_FINAL_AUDIT status=PASS "
        "mode=FIXED_PHI_CONSERVED_COMPOSITION_TARGET_PROFILE_V4" in text
    )
    if require_fixed_phi and not fixed_phi_marker:
        raise ValueError("fixed-phi composition minimizer audit is absent")
    match = re.search(r"MINIMIZE_MASS_CONSTRAINT_FINAL_AUDIT status=PASS .*?residual_relative=([0-9.eE+-]+)", text)
    if not match:
        raise ValueError("cannot parse final mass-constraint residual")
    return {
        "minimizer_stdout_sha256": sha256(path),
        "mass_constraint_residual_relative": float(match.group(1)),
        "fixed_phi_composition_audit": fixed_phi_marker,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--minimizer-stdout", type=Path, required=True)
    parser.add_argument("--phi-raw", type=Path, required=True)
    parser.add_argument("--xB-raw", type=Path, required=True)
    parser.add_argument("--Y-raw", type=Path, required=True)
    parser.add_argument("--dY-dt-prev-raw", type=Path, required=True)
    parser.add_argument("--require-fixed-phi", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"[fatal] refusing to overwrite: {args.out}")
    seed_path = args.seed / "seed_manifest.json"
    if not seed_path.is_file():
        raise SystemExit("[fatal] seed manifest missing")
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    if seed.get("schema") != SEED_SCHEMA:
        raise SystemExit("[fatal] wrong V3 seed schema")
    min_contract = require_stdout_contract(args.minimizer_stdout, args.require_fixed_phi)
    for path in (args.phi_raw, args.xB_raw, args.Y_raw, args.dY_dt_prev_raw):
        if not path.is_file():
            raise SystemExit(f"[fatal] minimizer artifact missing: {path}")
    grid = seed["grid"]
    shape = tuple(int(grid[key]) for key in ("Nx", "Ny", "Nz"))
    phi = read_raw(args.phi_raw, shape, "phi")
    xb = read_raw(args.xB_raw, shape, "xB")
    y = read_raw(args.Y_raw, shape, "Y")
    dY = read_raw(args.dY_dt_prev_raw, shape, "dY_dt_prev")
    phi_identity = None
    if args.require_fixed_phi:
        seed_phi_path = args.seed / "phi.raw.f64"
        if not seed_phi_path.is_file():
            raise SystemExit("[fatal] fixed-phi seed raw field missing")
        if seed_phi_path.read_bytes() != args.phi_raw.read_bytes():
            raise SystemExit("[fatal] fixed-phi output differs bytewise from seed phi")
        phi_identity = True
    h = base.h_of_phi(phi)
    if float(np.min(phi)) < 0.0 or float(np.max(phi)) > 1.0:
        raise SystemExit("[fatal] constrained phi violates bounds")
    seed_meta = json.loads((args.seed / "init_meta.json").read_text(encoding="utf-8"))
    lower, upper = float(seed_meta["xB_min_safe"]), float(seed_meta["xB_max_safe"])
    if float(np.min(xb)) < lower or float(np.max(xb)) > upper:
        raise SystemExit("[fatal] constrained xB violates raw storage bounds")
    logit_error = float(np.max(np.abs(y - base.logit(xb))))
    if logit_error > 5.0e-12:
        raise SystemExit(f"[fatal] minimizer Y/xB identity mismatch: {logit_error:.3e}")
    c_total = h + (1.0 - h) * xb
    target_total = float(seed["inventory"]["target_total_C_B_tot"])
    relative_error = abs(float(np.sum(c_total, dtype=np.float64)) - target_total) / max(abs(target_total), 1.0)
    if relative_error > 1.0e-12:
        raise SystemExit(f"[fatal] frozen profile mass mismatch: {relative_error:.3e}")
    threshold = float(seed["component_contract"]["h_threshold"])
    labels, components = base.components(h, threshold, float(grid["dx_nm"]))
    del labels
    expected_count = int(seed["component_contract"]["expected_count"])
    if len(components) != expected_count:
        raise SystemExit("[fatal] constrained profile has unexpected merge/split")
    args.out.mkdir(parents=True, exist_ok=False)
    arrays = {"phi": phi, "h_phi": h, "C_B_tot": c_total, "xB_alpha": xb, "Y": y, "dY_dt_prev": dY}
    fields: Dict[str, Dict[str, str]] = {}
    for name, values in arrays.items():
        path = args.out / f"{name}.raw.f64"
        write_raw(path, values)
        fields[name] = {"path": path.name, "sha256": sha256(path), "dtype": "float64-le", "order": "C"}
    schema = V4_SCHEMA if args.require_fixed_phi else SCHEMA
    meta_schema = V4_META_SCHEMA if args.require_fixed_phi else META_SCHEMA
    dY_provenance = (
        "fixed_phi_conserved_composition_minimizer_output_v4"
        if args.require_fixed_phi
        else "constrained_full_model_minimizer_output_v3"
    )
    meta = {
        "schema": meta_schema,
        "Nx": shape[0], "Ny": shape[1], "Nz": shape[2], "dx_nm": float(grid["dx_nm"]),
        "interface_width_nm": float(seed["physical_contract"]["lambda_sm_nm"]),
        "dt_recommended": 0.02, "mean_xBtot": float(np.mean(c_total, dtype=np.float64)),
        "xB_min_safe": lower, "xB_max_safe": upper, "dtype": "float64", "order": "C",
        "phi_path": fields["phi"]["path"], "xB_path": fields["xB_alpha"]["path"],
        "dY_dt_prev_path": fields["dY_dt_prev"]["path"],
        "dY_dt_prev_initialization": dY_provenance,
        "dynamic_raw_history_required": True,
    }
    (args.out / "init_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.out / "initial_components.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = list(components[0]) if components else ["component_label"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(components)
    fixture = {
        "schema": schema,
        "fixture_kind": (
            "E2_FIXED_PHI_COMMON_MATRIX_CONSTRAINED_PROFILE"
            if args.require_fixed_phi else "E2_COMMON_MATRIX_CONSTRAINED_PROFILE"
        ),
        "validation_only": True,
        "seed_manifest_sha256": sha256(seed_path),
        "selected_library": seed["selected_library"],
        "physical_contract": seed["physical_contract"],
        "grid": grid,
        "particles": seed["particles"],
        "component_contract": {"h_threshold": threshold, "expected_count": expected_count, "actual_count": len(components)},
        "composition_contract": {
            "common_matrix_seed": "UNIFORM_COMMON_MATRIX_CONSERVED_MULTI_E2_SEED_V3",
            "final_profile": (
                "FIXED_PHI_FULL_MODEL_CONSERVED_COMPOSITION_MINIMIZATION_V4"
                if args.require_fixed_phi
                else "FULL_MODEL_ELASTIC_MASS_CONSTRAINED_MINIMIZATION_V3"
            ),
            "phi_evolution": (
                "FROZEN_BYTE_IDENTICAL_TO_COMMON_MATRIX_SEED_V4"
                if args.require_fixed_phi else "FULL_MODEL_MINIMIZATION_V3"
            ),
            "isolated_profile_delta_x_reused": False,
            "isolated_absolute_xB_reused": False,
            "normalization_used": False,
            "raw_init_clipping_used": False,
        },
        "inventory": {
            "target_total_C_B_tot": target_total,
            "actual_total_C_B_tot": float(np.sum(c_total, dtype=np.float64)),
            "relative_error": relative_error,
            "mean_xBtot": float(np.mean(c_total, dtype=np.float64)),
            "beta_volume_fraction": float(np.mean(h, dtype=np.float64)),
        },
        "time_level_contract": {
            "Y": "minimizer_raw_output",
            "dY_dt_prev": dY_provenance,
            "dynamic_raw_history_required": True,
            "dY_dt_prev_nonzero_count": int(np.count_nonzero(dY)),
            "dY_dt_prev_max_abs": float(np.max(np.abs(dY))),
        },
        "minimization": {
            **min_contract,
            "input_raw_sha256": {"phi": sha256(args.phi_raw), "xB": sha256(args.xB_raw), "Y": sha256(args.Y_raw), "dY_dt_prev": sha256(args.dY_dt_prev_raw)},
            "Y_logit_xB_max_abs": logit_error,
            "seed_to_final_phi_bytewise_equal": phi_identity,
        },
        "fields": fields,
        "init_meta": {"path": "init_meta.json", "sha256": sha256(args.out / "init_meta.json")},
        "initial_components": {"path": "initial_components.csv", "sha256": sha256(args.out / "initial_components.csv")},
    }
    (args.out / "fixture_manifest.json").write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": (
            "PASS_PF_ELASTIC_MULTI_PARTICLE_FIXED_PHI_COMMON_MATRIX_PROFILE_V4"
            if args.require_fixed_phi
            else "PASS_PF_ELASTIC_MULTI_PARTICLE_COMMON_MATRIX_PROFILE_V3"
        ),
        "fixture_manifest_sha256": sha256(args.out / "fixture_manifest.json"),
        "mass_relative": relative_error,
        "component_count": len(components),
        "dY_dt_prev_max_abs": fixture["time_level_contract"]["dY_dt_prev_max_abs"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
