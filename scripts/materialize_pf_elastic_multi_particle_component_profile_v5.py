#!/usr/bin/env python3
"""Freeze a V5 component-volume-constrained elastic target profile."""

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
CONSTRAINT_SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_COMPONENT_CONSTRAINT_V5"
SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_COMPONENT_VOLUME_PROFILE_V5"
META_SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_COMPONENT_VOLUME_PROFILE_RAW_META_V5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_raw(path: Path, shape: tuple[int, int, int], label: str) -> np.ndarray:
    data = np.fromfile(path, dtype="<f8")
    if data.size != math.prod(shape) or not np.all(np.isfinite(data)):
        raise ValueError(f"invalid {label} raw field")
    return data.reshape(shape, order="C")


def write_raw(path: Path, values: np.ndarray) -> None:
    np.asarray(values, dtype="<f8").ravel(order="C").tofile(path)


def require_stdout_contract(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    required = (
        "MINIMIZE_TARGET_PROFILE_CONVERGENCE_FINAL_AUDIT status=PASS "
        "mode=COMPONENT_CONSERVED_H_VOLUME_TARGET_PROFILE_V5",
        "MINIMIZE_MASS_CONSTRAINT_FINAL_AUDIT status=PASS",
        "MINIMIZE_COMPONENT_VOLUME_FINAL_AUDIT status=PASS "
        "mode=COMPONENT_CONSERVED_H_VOLUME_TARGET_PROFILE_V5",
    )
    if any(marker not in text for marker in required):
        raise ValueError("V5 minimizer terminal contract is incomplete")
    if "raw init required clamping" in text:
        raise ValueError("V5 raw seed required clamping")
    match = re.search(
        r"MINIMIZE_COMPONENT_VOLUME_FINAL_AUDIT status=PASS .*?"
        r"max_component_h_volume_relative_error=([0-9.eE+-]+).*?"
        r"component_kkt_residual=([0-9.eE+-]+).*?"
        r"component_kkt_threshold=([0-9.eE+-]+)", text)
    if not match:
        raise ValueError("cannot parse V5 component volume/KKT audit")
    component_kkt_residual = float(match.group(2))
    component_kkt_threshold = float(match.group(3))
    if not math.isfinite(component_kkt_residual) or not math.isfinite(component_kkt_threshold):
        raise ValueError("V5 component KKT audit is non-finite")
    if component_kkt_residual > component_kkt_threshold:
        raise ValueError("V5 component KKT residual exceeds its registered threshold")
    return {
        "minimizer_stdout_sha256": sha256(path),
        "max_component_h_volume_relative_error": float(match.group(1)),
        "component_kkt_residual": component_kkt_residual,
        "component_kkt_threshold": component_kkt_threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--constraint", type=Path, required=True)
    parser.add_argument("--minimizer-stdout", type=Path, required=True)
    parser.add_argument("--phi-raw", type=Path, required=True)
    parser.add_argument("--xB-raw", type=Path, required=True)
    parser.add_argument("--Y-raw", type=Path, required=True)
    parser.add_argument("--dY-dt-prev-raw", type=Path, required=True)
    parser.add_argument(
        "--dynamic-time-level", choices=("fresh_zero", "minimizer_output"),
        default="fresh_zero",
        help=("dynamic handoff history; a nonphysical minimizer derivative "
              "must not be reused as a physical previous-time-level by default"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"[fatal] refusing to overwrite: {args.out}")
    seed_path = args.seed / "seed_manifest.json"
    constraint_path = args.constraint / "component_constraint_manifest.json"
    if not seed_path.is_file() or not constraint_path.is_file():
        raise SystemExit("[fatal] V5 seed or component constraint manifest missing")
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    constraint = json.loads(constraint_path.read_text(encoding="utf-8"))
    if seed.get("schema") != SEED_SCHEMA or constraint.get("schema") != CONSTRAINT_SCHEMA:
        raise SystemExit("[fatal] incompatible V5 seed/constraint schemas")
    if constraint.get("seed_manifest_sha256") != sha256(seed_path):
        raise SystemExit("[fatal] component constraint is not pinned to this seed")
    grid = seed["grid"]
    shape = tuple(int(grid[key]) for key in ("Nx", "Ny", "Nz"))
    raw_row = constraint["ownership_contract"]["label_raw"]
    labels_path = args.constraint / str(raw_row["path"])
    if not labels_path.is_file() or sha256(labels_path) != raw_row["sha256"]:
        raise SystemExit("[fatal] component ownership raw hash mismatch")
    labels = np.fromfile(labels_path, dtype="<i4")
    if labels.size != math.prod(shape):
        raise SystemExit("[fatal] component ownership raw size mismatch")
    labels = labels.reshape(shape, order="C")
    count = int(constraint["ownership_contract"]["component_count"])
    if int(np.min(labels)) < 0 or int(np.max(labels)) >= count:
        raise SystemExit("[fatal] invalid component ownership labels")
    min_contract = require_stdout_contract(args.minimizer_stdout)
    phi, xb, y, preparatory_dy = (read_raw(path, shape, label) for path, label in (
        (args.phi_raw, "phi"), (args.xB_raw, "xB"),
        (args.Y_raw, "Y"), (args.dY_dt_prev_raw, "dY_dt_prev")))
    dy = (np.zeros(shape, dtype=np.float64)
          if args.dynamic_time_level == "fresh_zero" else preparatory_dy.copy())
    if float(np.min(phi)) < 0.0 or float(np.max(phi)) > 1.0:
        raise SystemExit("[fatal] V5 phi violates bounds")
    seed_meta = json.loads((args.seed / "init_meta.json").read_text(encoding="utf-8"))
    lower, upper = float(seed_meta["xB_min_safe"]), float(seed_meta["xB_max_safe"])
    if float(np.min(xb)) < lower or float(np.max(xb)) > upper:
        raise SystemExit("[fatal] V5 xB violates raw bounds")
    logit_error = float(np.max(np.abs(y - base.logit(xb))))
    if logit_error > 5.0e-12:
        raise SystemExit(f"[fatal] V5 Y/xB identity mismatch: {logit_error:.3e}")
    h = base.h_of_phi(phi)
    components = constraint["components"]
    target_h_sums = [float(item["target_h_sum"]) for item in components]
    observed_h_sums = [float(np.sum(h[labels == index], dtype=np.float64)) for index in range(count)]
    component_errors = [
        abs(observed - target) / max(abs(target), 1.0e-30)
        for observed, target in zip(observed_h_sums, target_h_sums)
    ]
    if max(component_errors, default=math.inf) > 1.0e-4:
        raise SystemExit("[fatal] V5 materialized component h-volume does not close")
    total = h + (1.0 - h) * xb
    target_total = float(seed["inventory"]["target_total_C_B_tot"])
    mass_relative = abs(float(np.sum(total, dtype=np.float64)) - target_total) / max(abs(target_total), 1.0)
    if mass_relative > 1.0e-12:
        raise SystemExit(f"[fatal] V5 profile mass does not close: {mass_relative:.3e}")
    threshold = float(seed["component_contract"]["h_threshold"])
    _, tracked = base.components(h, threshold, float(grid["dx_nm"]))
    expected_count = int(seed["component_contract"]["expected_count"])
    if len(tracked) != expected_count:
        raise SystemExit("[fatal] V5 profile has unexpected merge/split")
    args.out.mkdir(parents=True, exist_ok=False)
    arrays = {"phi": phi, "h_phi": h, "C_B_tot": total, "xB_alpha": xb, "Y": y, "dY_dt_prev": dy}
    fields: Dict[str, Dict[str, str]] = {}
    for name, values in arrays.items():
        path = args.out / f"{name}.raw.f64"
        write_raw(path, values)
        fields[name] = {"path": path.name, "sha256": sha256(path), "dtype": "float64-le", "order": "C"}
    # Copy the pinned ownership map into the handoff fixture so later
    # qualification is self-contained rather than dependent on a parent dir.
    out_labels = args.out / labels_path.name
    out_labels.write_bytes(labels_path.read_bytes())
    if sha256(out_labels) != raw_row["sha256"]:
        raise SystemExit("[fatal] copied V5 ownership map hash mismatch")
    meta = {
        "schema": META_SCHEMA, "Nx": shape[0], "Ny": shape[1], "Nz": shape[2],
        "dx_nm": float(grid["dx_nm"]),
        "interface_width_nm": float(seed["physical_contract"]["lambda_sm_nm"]),
        "dt_recommended": 0.02, "mean_xBtot": float(np.mean(total, dtype=np.float64)),
        "xB_min_safe": lower, "xB_max_safe": upper, "dtype": "float64", "order": "C",
        "phi_path": fields["phi"]["path"], "xB_path": fields["xB_alpha"]["path"],
        "dY_dt_prev_path": fields["dY_dt_prev"]["path"],
        "dY_dt_prev_initialization": (
            "fresh_zero_after_nonphysical_component_minimization_v5"
            if args.dynamic_time_level == "fresh_zero"
            else "component_volume_constrained_minimizer_output_v5"),
        "dynamic_raw_history_required": args.dynamic_time_level == "minimizer_output",
    }
    (args.out / "init_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.out / "initial_components.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tracked[0]) if tracked else ["component_label"])
        writer.writeheader(); writer.writerows(tracked)
    fixture = {
        "schema": SCHEMA,
        "fixture_kind": "E2_COMPONENT_VOLUME_CONSTRAINED_PROFILE",
        "validation_only": True,
        "seed_manifest_sha256": sha256(seed_path),
        "component_constraint_manifest_sha256": sha256(constraint_path),
        "selected_library": seed["selected_library"], "physical_contract": seed["physical_contract"],
        "grid": grid, "particles": seed["particles"],
        "component_contract": {"h_threshold": threshold, "expected_count": expected_count, "actual_count": len(tracked),
                               "per_particle_h_volume_constrained": True,
                               "max_relative_error": max(component_errors)},
        "composition_contract": {
            "common_matrix_seed": "UNIFORM_COMMON_MATRIX_CONSERVED_MULTI_E2_SEED_V3",
            "final_profile": "JOINT_PHI_Y_COMPONENT_H_VOLUME_CONSTRAINED_MINIMIZATION_V5",
            "phi_evolution": "JOINT_RELAXATION_WITH_FIXED_PER_COMPONENT_H_VOLUME_V5",
            "inter_component_volume_exchange": False,
            "raw_init_clipping_used": False,
        },
        "inventory": {"target_total_C_B_tot": target_total, "actual_total_C_B_tot": float(np.sum(total, dtype=np.float64)),
                      "relative_error": mass_relative, "mean_xBtot": float(np.mean(total, dtype=np.float64)),
                      "beta_volume_fraction": float(np.mean(h, dtype=np.float64))},
        "time_level_contract": {
            "Y": "minimizer_raw_output",
            "dY_dt_prev": meta["dY_dt_prev_initialization"],
            "dynamic_raw_history_required": args.dynamic_time_level == "minimizer_output",
            "dynamic_time_level_kind": args.dynamic_time_level,
            "dY_dt_prev_nonzero_count": int(np.count_nonzero(dy)),
            "dY_dt_prev_max_abs": float(np.max(np.abs(dy))),
            "preparatory_minimizer_dY_dt_prev_nonzero_count": int(np.count_nonzero(preparatory_dy)),
            "preparatory_minimizer_dY_dt_prev_max_abs": float(np.max(np.abs(preparatory_dy))),
            "preparatory_minimizer_dY_dt_prev_sha256": sha256(args.dY_dt_prev_raw),
        },
        "minimization": {**min_contract, "input_raw_sha256": {"phi": sha256(args.phi_raw), "xB": sha256(args.xB_raw), "Y": sha256(args.Y_raw), "dY_dt_prev": sha256(args.dY_dt_prev_raw)},
                         "Y_logit_xB_max_abs": logit_error, "component_target_h_sums": target_h_sums,
                         "component_observed_h_sums": observed_h_sums, "component_relative_errors": component_errors},
        "fields": fields,
        "component_ownership_labels": {"path": out_labels.name, "sha256": sha256(out_labels), "dtype": "int32-le", "order": "C"},
        "init_meta": {"path": "init_meta.json", "sha256": sha256(args.out / "init_meta.json")},
        "initial_components": {"path": "initial_components.csv", "sha256": sha256(args.out / "initial_components.csv")},
    }
    manifest_path = args.out / "fixture_manifest.json"
    manifest_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_PF_ELASTIC_MULTI_PARTICLE_COMPONENT_VOLUME_PROFILE_V5",
                      "fixture_manifest_sha256": sha256(manifest_path), "mass_relative": mass_relative,
                      "component_count": expected_count, "max_component_relative_error": max(component_errors)}, sort_keys=True))


if __name__ == "__main__":
    main()
