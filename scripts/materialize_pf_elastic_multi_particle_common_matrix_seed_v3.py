#!/usr/bin/env python3
"""Create a common-matrix, conserved seed for V3 multi-particle relaxation.

V2 correctly established a canonical inventory, but its composition field
contains the finite deviations from *several different isolated far-field
states*.  Those states are not a common multi-particle equilibrium.  V3
therefore uses only V2's exact elastic phase union and frozen total inventory
to produce a uniform-matrix conservative seed.  The full constrained PF
minimizer subsequently generates the multi-particle composition profile.

This is a preparatory target-profile calculation, not an initial dynamic
relaxation and not a physical-parameter change.  V2 remains immutable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np

import materialize_pf_elastic_multi_particle_6h_fixture_v1 as base


INPUT_SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_6H_FIXTURE_V2"
SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_COMMON_MATRIX_SEED_V3"
META_SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_COMMON_MATRIX_SEED_RAW_META_V3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_raw(path: Path, values: np.ndarray) -> None:
    np.asarray(values, dtype="<f8").ravel(order="C").tofile(path)


def load_field(manifest: Dict[str, Any], root: Path, name: str, shape: tuple[int, int, int]) -> np.ndarray:
    row = manifest["fields"].get(name)
    if not isinstance(row, dict):
        raise ValueError(f"missing field {name}")
    path = root / str(row.get("path", ""))
    if not path.is_file() or sha256(path) != row.get("sha256"):
        raise ValueError(f"hash mismatch for V2 field {name}")
    values = np.fromfile(path, dtype="<f8")
    if values.size != math.prod(shape):
        raise ValueError(f"size mismatch for V2 field {name}")
    return values.reshape(shape, order="C")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fixture_manifest_path = args.v2_fixture / "fixture_manifest.json"
    if not fixture_manifest_path.is_file():
        raise SystemExit("[fatal] V2 fixture manifest is missing")
    if args.out.exists():
        raise SystemExit(f"[fatal] refusing to overwrite: {args.out}")
    manifest = json.loads(fixture_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != INPUT_SCHEMA or manifest.get("fixture_kind") != "E2":
        raise SystemExit("[fatal] V3 seed accepts only the frozen V2 E2 fixture")
    if manifest.get("combination", {}).get("composition_contract") != "PHASE_CONSISTENT_DELTA_X_ALPHA_V2":
        raise SystemExit("[fatal] wrong V2 composition contract")
    grid_doc = manifest.get("grid", {})
    shape = tuple(int(grid_doc[key]) for key in ("Nx", "Ny", "Nz"))
    if any(value <= 0 for value in shape):
        raise SystemExit("[fatal] invalid V2 grid")
    phi = load_field(manifest, args.v2_fixture, "phi", shape)
    h = load_field(manifest, args.v2_fixture, "h_phi", shape)
    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(h)):
        raise SystemExit("[fatal] non-finite V2 phase field")
    if float(np.max(np.abs(h - base.h_of_phi(phi)))) > 5.0e-14:
        raise SystemExit("[fatal] V2 h(phi) identity does not hold")
    alpha = 1.0 - h
    target_total = float(manifest["inventory"]["target_total_C_B_tot"])
    alpha_sum = float(np.sum(alpha, dtype=np.float64))
    h_sum = float(np.sum(h, dtype=np.float64))
    if not math.isfinite(target_total) or not math.isfinite(alpha_sum) or alpha_sum <= 0.0:
        raise SystemExit("[fatal] invalid conserved inventory input")
    xB_matrix = (target_total - h_sum) / alpha_sum
    xB_min = float(manifest["inventory"].get("xB_min_safe", manifest.get("phase_storage_contract", {}).get("alpha_floor", 1.0e-12)))
    xB_max = float(manifest.get("fields", {}).get("xB_alpha", {}).get("xB_max_safe", 0.499999))
    # V2 stores the explicit safe bound in its raw meta, not the field row.
    v2_meta = json.loads((args.v2_fixture / "init_meta.json").read_text(encoding="utf-8"))
    xB_min = float(v2_meta.get("xB_min_safe", 1.0e-12))
    xB_max = float(v2_meta.get("xB_max_safe", 0.499999))
    if not (xB_min <= xB_matrix <= xB_max):
        raise SystemExit("[fatal] common-matrix seed would violate xB storage bounds")
    xB = np.full(shape, xB_matrix, dtype=np.float64)
    c_total = h + alpha * xB
    relative_error = abs(float(np.sum(c_total, dtype=np.float64)) - target_total) / max(abs(target_total), 1.0)
    if relative_error > 1.0e-12:
        raise SystemExit(f"[fatal] common-matrix inventory does not close: {relative_error:.3e}")
    y = base.logit(xB)
    dY = np.zeros(shape, dtype=np.float64)
    threshold = float(manifest["component_contract"]["h_threshold"])
    labels, rows = base.components(h, threshold, float(grid_doc["dx_nm"]))
    if len(rows) != int(manifest["component_contract"]["expected_count"]):
        raise SystemExit("[fatal] V3 seed changed the frozen phase-component count")
    args.out.mkdir(parents=True, exist_ok=False)
    fields = {
        "phi": phi, "h_phi": h, "C_B_tot": c_total, "xB_alpha": xB,
        "Y": y, "dY_dt_prev": dY,
    }
    field_rows: Dict[str, Dict[str, str]] = {}
    for name, values in fields.items():
        path = args.out / f"{name}.raw.f64"
        write_raw(path, values)
        field_rows[name] = {"path": path.name, "sha256": sha256(path), "dtype": "float64-le", "order": "C"}
    meta = {
        "schema": META_SCHEMA,
        "Nx": shape[0], "Ny": shape[1], "Nz": shape[2],
        "dx_nm": float(grid_doc["dx_nm"]),
        "interface_width_nm": float(manifest["physical_contract"]["lambda_sm_nm"]),
        "dt_recommended": 0.1,
        "mean_xBtot": float(np.mean(c_total, dtype=np.float64)),
        "xB_min_safe": xB_min, "xB_max_safe": xB_max,
        "dtype": "float64", "order": "C",
        "phi_path": field_rows["phi"]["path"], "xB_path": field_rows["xB_alpha"]["path"],
        "dY_dt_prev_path": field_rows["dY_dt_prev"]["path"],
        "dY_dt_prev_initialization": "zero_for_preparatory_constrained_minimization",
    }
    (args.out / "init_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.out / "initial_components.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = list(rows[0]) if rows else ["component_label"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    seed_manifest = {
        "schema": SCHEMA,
        "seed_type": "UNIFORM_COMMON_MATRIX_CONSERVED_MULTI_E2_SEED_V3",
        "validation_only": True,
        "source_v2_fixture_manifest_sha256": sha256(fixture_manifest_path),
        "source_v2_fixture_schema": manifest["schema"],
        "selected_library": manifest["selected_library"],
        "physical_contract": manifest["physical_contract"],
        "grid": grid_doc,
        "particles": manifest["particles"],
        "component_contract": {
            "h_threshold": threshold,
            "expected_count": int(manifest["component_contract"]["expected_count"]),
            "actual_count": len(rows),
        },
        "composition_contract": {
            "seed_phase": "exact_frozen_V2_E2_phi_union",
            "seed_xB_alpha": "uniform_common_matrix_solved_from_frozen_total_inventory",
            "isolated_profile_delta_x_reused": False,
            "isolated_absolute_xB_reused": False,
            "normalization_used": False,
            "clipping_used": False,
            "next_required_step": "FULL_MODEL_ELASTIC_MASS_CONSTRAINED_MINIMIZATION_V3",
        },
        "inventory": {
            "target_total_C_B_tot": target_total,
            "actual_total_C_B_tot": float(np.sum(c_total, dtype=np.float64)),
            "relative_error": relative_error,
            "common_matrix_xB": xB_matrix,
            "common_matrix_xAg": 2.0 * xB_matrix / (2.0 + xB_matrix),
            "h_total": h_sum,
            "alpha_total": alpha_sum,
            "beta_volume_fraction": float(np.mean(h, dtype=np.float64)),
        },
        "time_level_contract": {
            "dY_dt_prev": "zero_for_preparatory_constrained_minimization_only",
            "dynamic_handoff_history_must_come_from_minimizer_output": True,
        },
        "fields": field_rows,
        "init_meta": {"path": "init_meta.json", "sha256": sha256(args.out / "init_meta.json")},
        "initial_components": {"path": "initial_components.csv", "sha256": sha256(args.out / "initial_components.csv")},
    }
    (args.out / "seed_manifest.json").write_text(json.dumps(seed_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS_PF_ELASTIC_MULTI_PARTICLE_COMMON_MATRIX_SEED_V3",
        "seed_manifest_sha256": sha256(args.out / "seed_manifest.json"),
        "common_matrix_xB": xB_matrix,
        "relative_inventory_error": relative_error,
        "component_count": len(rows),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
