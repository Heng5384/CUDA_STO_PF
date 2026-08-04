#!/usr/bin/env python3
"""Read-only static audit for the 400-cube fixed-xB03 elastic PSD pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def h_of_phi(phi: np.ndarray) -> np.ndarray:
    direct = phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)
    q = 1.0 - phi
    complement = 1.0 - q**3 * (1.0 + 3.0 * phi + 6.0 * phi**2)
    return np.where(phi <= 0.5, direct, complement)


def xag_from_xb(xb: float) -> float:
    return 2.0 * xb / (2.0 + xb)


def periodic_distance(a: list[int], b: list[int], n: int) -> float:
    delta = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return float(np.linalg.norm(np.minimum(delta, float(n) - delta)))


def chunked_field_audit(phi: np.memmap, xb: np.memmap, ctot: np.memmap, threshold: float) -> dict[str, float]:
    total = 0.0
    count = 0
    xag_sum = 0.0
    matrix_count = 0
    max_identity_error = 0.0
    phi_min, phi_max, xb_min, xb_max = math.inf, -math.inf, math.inf, -math.inf
    h_sum = 0.0
    for begin in range(0, phi.shape[0], 8):
        end = min(phi.shape[0], begin + 8)
        p = np.asarray(phi[begin:end], dtype=np.float64)
        x = np.asarray(xb[begin:end], dtype=np.float64)
        c = np.asarray(ctot[begin:end], dtype=np.float64)
        h = h_of_phi(p)
        total += float(np.sum(c, dtype=np.float64)); count += c.size
        h_sum += float(np.sum(h, dtype=np.float64))
        max_identity_error = max(max_identity_error, float(np.max(np.abs(c - (h + (1.0 - h) * x)))))
        mask = h < threshold
        xag_sum += float(np.sum(2.0 * x[mask] / (2.0 + x[mask]), dtype=np.float64))
        matrix_count += int(np.count_nonzero(mask))
        phi_min, phi_max = min(phi_min, float(np.min(p))), max(phi_max, float(np.max(p)))
        xb_min, xb_max = min(xb_min, float(np.min(x))), max(xb_max, float(np.max(x)))
    return {
        "mean_C_Btot": total / count,
        "beta_h_volume_nm3": h_sum,
        "far_field_xAg_h_lt_threshold": xag_sum / matrix_count,
        "far_field_voxel_fraction": matrix_count / count,
        "field_identity_max_abs_error": max_identity_error,
        "phi_min": phi_min, "phi_max": phi_max, "xB_min": xb_min, "xB_max": xb_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"[fatal] refusing to overwrite audit root: {args.out}")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    manifest_path = args.fixture / "fixture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    grid = manifest.get("grid", {})
    shape = tuple(int(grid.get(key, -1)) for key in ("Nx", "Ny", "Nz"))
    if shape != (400, 400, 400) or float(grid.get("dx_nm", 0.0)) != 1.0:
        failures.append("wrong_grid")
    if manifest.get("schema") != "PF_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_FIXTURE_V1":
        failures.append("wrong_schema")
    expected_library = args.library_root / "library_manifest.json"
    if not expected_library.is_file() or manifest.get("profile_library_manifest_sha256") != sha256(expected_library):
        failures.append("profile_library_hash_mismatch")
    fields = manifest.get("fields", {})
    required_fields = ("phi", "h_phi", "xB_alpha", "Y", "dY_dt_prev", "C_B_tot", "delta_C_relaxation_total")
    expected_bytes = math.prod(shape) * 8
    for name in required_fields:
        row = fields.get(name, {})
        path = args.fixture / str(row.get("path", ""))
        if not path.is_file() or path.stat().st_size != expected_bytes or sha256(path) != row.get("sha256"):
            failures.append(f"field_hash_or_size_{name}")
    init_meta = args.fixture / "init_meta.json"
    if not init_meta.is_file():
        failures.append("missing_init_meta")
    else:
        meta = json.loads(init_meta.read_text(encoding="utf-8"))
        if (meta.get("Nx"), meta.get("Ny"), meta.get("Nz"), meta.get("dx_nm")) != (400, 400, 400, 1.0):
            failures.append("init_meta_grid_mismatch")
    if failures:
        raise SystemExit("[fatal] " + ",".join(failures))
    phi = np.memmap(args.fixture / fields["phi"]["path"], dtype="<f8", mode="r", shape=shape, order="C")
    xb = np.memmap(args.fixture / fields["xB_alpha"]["path"], dtype="<f8", mode="r", shape=shape, order="C")
    ctot = np.memmap(args.fixture / fields["C_B_tot"]["path"], dtype="<f8", mode="r", shape=shape, order="C")
    threshold = float(manifest["component_contract"]["h_threshold"])
    fields_audit = chunked_field_audit(phi, xb, ctot, threshold)
    target_mean = float(spec["inventory_contract"]["mean_C_B_tot"])
    target_beta = float(spec["inventory_contract"]["target_beta_inventory_nm3"])
    lo, hi = (float(x) for x in spec["inventory_contract"]["matrix_xAg_allowed_interval"])
    if abs(fields_audit["mean_C_Btot"] - target_mean) > 1.0e-12:
        failures.append("mean_C_Btot_not_exact")
    if abs(fields_audit["beta_h_volume_nm3"] - target_beta) / target_beta > 1.0e-4:
        failures.append("beta_inventory_outside_discrete_contract")
    if not lo <= fields_audit["far_field_xAg_h_lt_threshold"] <= hi:
        failures.append("far_field_xAg_outside_experimental_band")
    if fields_audit["field_identity_max_abs_error"] > 5.0e-13:
        failures.append("canonical_field_identity_mismatch")
    if not (-1.0e-12 <= fields_audit["phi_min"] <= fields_audit["phi_max"] <= 1.0 + 1.0e-12):
        failures.append("phi_bounds")
    if not (0.0 < fields_audit["xB_min"] <= fields_audit["xB_max"] < 0.499999):
        failures.append("xB_bounds")
    physical = manifest.get("physical_contract", {})
    for key in ("GP_enabled", "GP_birth_enabled", "GP_release_enabled", "external_source_enabled", "new_beta_nucleation_enabled"):
        if physical.get(key) is not False:
            failures.append(f"forbidden_path_{key}")
    if physical.get("elasticity_enabled") is not True:
        failures.append("elasticity_not_enabled")
    rows = manifest.get("particle_library_mappings", [])
    if len(rows) != int(spec["discrete_psd"]["target_particle_count"]):
        failures.append("particle_count")
    histogram: dict[str, int] = {}
    for row in rows:
        key = f"{float(row['registered_radius_nm']):.1f}"
        histogram[key] = histogram.get(key, 0) + 1
    if histogram != {key: int(value) for key, value in spec["discrete_psd"]["registered_radius_histogram"].items() if int(value)}:
        failures.append("frozen_psd_histogram")
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            distance = periodic_distance(left["center_grid"], right["center_grid"], 400)
            required = float(left["registered_radius_nm"]) + float(right["registered_radius_nm"]) + 16.0
            if distance <= required:
                failures.append("periodic_separation")
                break
        if "periodic_separation" in failures:
            break
    result = {
        "status": "PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_FIXTURE_STATIC_V1" if not failures else "BLOCKED_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_FIXTURE_STATIC_V1",
        "fixture_manifest_sha256": sha256(manifest_path),
        "spec_sha256": sha256(args.spec),
        "profile_library_manifest_sha256": sha256(expected_library),
        "particle_count": len(rows), "registered_radius_histogram": histogram,
        "field_audit": fields_audit, "failures": failures,
        "structural_binary_readability_contract": "main_cuda raw_fields requires matching init_meta grid, binary64 C-order phi/xB fields; both are verified here without executing a numerical macrostep",
        "one_step_or_restart_qualification_run": False,
    }
    args.out.mkdir(parents=True)
    (args.out / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "status.txt").write_text(result["status"] + "\n", encoding="utf-8")
    (args.out / "final_terminal_output.txt").write_text("\n".join(f"{key}={value}" for key, value in (("status", result["status"]), ("fixture_manifest_sha256", result["fixture_manifest_sha256"]), ("mean_C_Btot", fields_audit["mean_C_Btot"]), ("far_field_xAg", fields_audit["far_field_xAg_h_lt_threshold"]), ("beta_h_volume_nm3", fields_audit["beta_h_volume_nm3"]))) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if failures:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
