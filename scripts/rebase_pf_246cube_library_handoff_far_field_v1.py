#!/usr/bin/env python3
"""Rebase a qualified 246^3 library handoff to an absolute matrix xAg.

The resolved-beta geometry and the frozen local relaxation field are preserved
byte-for-byte.  Only the uniform alpha-matrix baseline is changed.  The global
canonical inventory is then derived from that baseline plus the frozen beta
population; it is never projected back to 0.03.
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


SOURCE_SCHEMA = "PF_246CUBE_LIBRARY_HANDOFF_MANIFEST_V1"
REBASE_SCHEMA = "PF_246CUBE_EXPERIMENT_MATRIX_ANCHORED_HANDOFF_V1"
GRID = (246, 246, 246)
COUNT = math.prod(GRID)
FIELD_NAMES = (
    "phi",
    "h_phi",
    "delta_C_relaxation_total",
    "C_B_tot",
    "xB_alpha",
    "Y",
    "dY_dt_prev",
)
PRESERVED_FIELDS = (
    "phi",
    "h_phi",
    "delta_C_relaxation_total",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verified_field(
    manifest: Dict[str, Any], root: Path, name: str
) -> Path:
    row = manifest.get("fields", {}).get(name)
    if not isinstance(row, dict):
        raise ValueError(f"source fixture is missing field {name}")
    path = root / str(row.get("path", ""))
    if not path.is_file() or path.stat().st_size != COUNT * 8:
        raise ValueError(f"source field {name} is missing or has wrong size")
    if sha256(path) != row.get("sha256"):
        raise ValueError(f"source field {name} SHA-256 mismatch")
    return path


def read_raw(path: Path) -> np.ndarray:
    value = np.fromfile(path, dtype="<f8")
    if value.size != COUNT:
        raise ValueError(f"{path}: wrong element count")
    return value.reshape(GRID, order="C")


def write_raw(path: Path, value: np.ndarray) -> None:
    np.asarray(value, dtype="<f8").ravel(order="C").tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-fixture", type=Path, required=True)
    parser.add_argument("--source-fixture-sha256", required=True)
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
        raise SystemExit("[fatal] source fixture schema mismatch")
    if source.get("component_contract", {}).get("actual_count") != 96:
        raise SystemExit("[fatal] source fixture does not contain 96 particles")
    if any(
        source.get("physical_contract", {}).get(key) is not False
        for key in (
            "GP_enabled",
            "GP_birth_enabled",
            "GP_release_enabled",
            "external_source_enabled",
            "new_beta_nucleation_enabled",
        )
    ):
        raise SystemExit("[fatal] source fixture enables a prohibited path")
    if args.out.exists():
        raise SystemExit(f"[fatal] refusing to overwrite output: {args.out}")

    target_xag = float(args.target_matrix_xAg)
    if not math.isfinite(target_xag) or not 0.0 < target_xag < 1.0:
        raise SystemExit("[fatal] target matrix xAg is outside (0,1)")
    target_xb = 2.0 * target_xag / (2.0 - target_xag)
    if abs(2.0 * target_xb / (2.0 + target_xb) - target_xag) > 1.0e-15:
        raise SystemExit("[fatal] xB/xAg conversion did not close")

    source_paths = {
        name: verified_field(source, source_root, name)
        for name in FIELD_NAMES
    }
    phi = read_raw(source_paths["phi"])
    h = read_raw(source_paths["h_phi"])
    delta_c = read_raw(source_paths["delta_C_relaxation_total"])
    if not all(np.all(np.isfinite(value)) for value in (phi, h, delta_c)):
        raise SystemExit("[fatal] source fields contain non-finite values")
    alpha = 1.0 - h
    if float(np.min(alpha)) <= 0.0:
        raise SystemExit("[fatal] source fixture has zero alpha storage")
    delta_x = delta_c / alpha
    x_b = target_xb + delta_x
    x_b_max_safe = float(
        load_json(source_root / source["init_meta"]["path"]).get(
            "xB_max_safe", 0.499999
        )
    )
    if (
        not np.all(np.isfinite(x_b))
        or float(np.min(x_b)) <= 0.0
        or float(np.max(x_b)) >= x_b_max_safe
    ):
        raise SystemExit("[fatal] matrix rebase would require clipping")

    c_total = h + alpha * x_b
    y = np.log(x_b / (1.0 - x_b))
    d_y = np.zeros(GRID, dtype=np.float64)
    actual_total = float(np.sum(c_total, dtype=np.float64))
    actual_mean = actual_total / COUNT
    beta_inventory = float(np.sum(h, dtype=np.float64))
    relaxation_inventory = float(np.sum(delta_c, dtype=np.float64))
    effective_matrix_volume = float(np.sum(alpha, dtype=np.float64))
    matrix_inventory = target_xb * effective_matrix_volume
    particle_inventory = beta_inventory + relaxation_inventory
    decomposition_error = abs(
        matrix_inventory + particle_inventory - actual_total
    ) / max(abs(actual_total), 1.0)
    if decomposition_error > 1.0e-14:
        raise SystemExit("[fatal] rebased inventory decomposition did not close")

    threshold = float(source["component_contract"]["h_threshold"])
    far_mask = h < threshold
    observed_xb = float(np.mean(x_b[far_mask], dtype=np.float64))
    observed_xag = 2.0 * observed_xb / (2.0 + observed_xb)

    args.out.mkdir(parents=True)
    fields: Dict[str, Dict[str, Any]] = {}
    for name in PRESERVED_FIELDS:
        destination = args.out / f"{name}.raw.f64"
        shutil.copyfile(source_paths[name], destination)
        fields[name] = {
            "path": destination.name,
            "sha256": sha256(destination),
            "dtype": "float64-le",
            "order": "C",
        }
    for name, value in (
        ("C_B_tot", c_total),
        ("xB_alpha", x_b),
        ("Y", y),
        ("dY_dt_prev", d_y),
    ):
        destination = args.out / f"{name}.raw.f64"
        write_raw(destination, value)
        fields[name] = {
            "path": destination.name,
            "sha256": sha256(destination),
            "dtype": "float64-le",
            "order": "C",
        }

    for key in ("initial_particles", "initial_components"):
        row = source[key]
        source_path = source_root / str(row["path"])
        if not source_path.is_file() or sha256(source_path) != row["sha256"]:
            raise SystemExit(f"[fatal] source auxiliary mismatch: {key}")
        shutil.copyfile(source_path, args.out / source_path.name)

    meta = load_json(source_root / source["init_meta"]["path"])
    meta.update(
        {
            "schema": "PF_246CUBE_EXPERIMENT_MATRIX_ANCHORED_RAW_INIT_META_V1",
            "mean_xBtot": actual_mean,
            "phi_path": "phi.raw.f64",
            "xB_path": "xB_alpha.raw.f64",
            "matrix_anchor_xB": target_xb,
            "matrix_anchor_xAg": target_xag,
            "global_inventory_mode": (
                "DERIVED_FROM_EXPERIMENT_MATRIX_AND_FROZEN_RESOLVED_BETA"
            ),
        }
    )
    write_json(args.out / "init_meta.json", meta)

    manifest = dict(source)
    manifest.update(
        {
            "fixture_id": (
                f"pf_246cube_experiment_matrix_xAg0p0062_"
                f"{source['replicate_id']}_v1"
            ),
            "scientific_semantics": (
                "experiment-matrix-anchored, mass-conserving, exact-profile-"
                "library-assembled conditional 6 h handoff state"
            ),
            "source_fixture": {
                "path": str(source_manifest_path),
                "sha256": sha256(source_manifest_path),
                "fixture_id": source.get("fixture_id"),
            },
            "matrix_anchor_contract": {
                "schema": REBASE_SCHEMA,
                "target_matrix_xAg": target_xag,
                "target_matrix_xB": target_xb,
                "global_inventory_mode": (
                    "DERIVED_FROM_EXPERIMENT_MATRIX_AND_FROZEN_RESOLVED_BETA"
                ),
                "source_mean_C_B_tot": float(
                    source["target_global_inventory"]["mean_C_B_tot"]
                ),
                "derived_mean_C_B_tot": actual_mean,
                "phi_bytewise_preserved": True,
                "delta_C_relaxation_bytewise_preserved": True,
                "particle_population_preserved": True,
                "clipping_used": False,
                "normalization_used": False,
                "global_mass_projection_used": False,
            },
            "target_global_inventory": {
                "mean_C_B_tot": actual_mean,
                "total_C_B_tot_code": actual_total,
                "source": (
                    "derived_from_experiment_matrix_and_hash_pinned_"
                    "resolved_beta_population"
                ),
            },
            "derived_matrix_baseline": {
                "xB_alpha": target_xb,
                "xAg": target_xag,
                "observed_matrix_xB_h_lt_1e4": observed_xb,
                "observed_matrix_xAg_h_lt_1e4": observed_xag,
                "effective_matrix_volume_code": effective_matrix_volume,
                "matrix_baseline_inventory_code": matrix_inventory,
                "clipping_used": False,
                "normalization_used": False,
            },
            "initial_canonical_inventory": {
                "target_total_code": actual_total,
                "actual_total_code": actual_total,
                "matrix_baseline_inventory_code": matrix_inventory,
                "particle_inventory_sum_code": particle_inventory,
                "assembled_beta_phase_inventory_code": beta_inventory,
                "local_relaxation_inventory_code": relaxation_inventory,
                "field_relative_error": 0.0,
                "decomposition_relative_error": decomposition_error,
                "status": "PASS_MACHINE_PRECISION",
            },
            "field_bounds": {
                **source.get("field_bounds", {}),
                "xB_min": float(np.min(x_b)),
                "xB_max": float(np.max(x_b)),
            },
            "fields": fields,
        }
    )
    manifest["assembly_contract"] = {
        **source.get("assembly_contract", {}),
        "absolute_matrix_anchor_applied": True,
        "global_inventory_reprojected_to_0p03": False,
    }
    manifest["init_meta"] = {
        "path": "init_meta.json",
        "sha256": sha256(args.out / "init_meta.json"),
    }
    for key in ("initial_particles", "initial_components"):
        filename = Path(str(source[key]["path"])).name
        manifest[key] = {
            "path": filename,
            "sha256": sha256(args.out / filename),
        }
    write_json(args.out / "fixture_manifest.json", manifest)

    audit = {
        "schema": "PF_246CUBE_EXPERIMENT_MATRIX_REBASE_AUDIT_V1",
        "status": "PASS_EXPERIMENT_MATRIX_ANCHORED_FIXTURE_V1",
        "replicate": source["replicate_id"],
        "source_fixture_manifest_sha256": sha256(source_manifest_path),
        "fixture_manifest_sha256": sha256(args.out / "fixture_manifest.json"),
        "target_matrix_xAg": target_xag,
        "target_matrix_xB": target_xb,
        "observed_matrix_xAg_h_lt_1e4": observed_xag,
        "derived_mean_C_B_tot": actual_mean,
        "derived_total_C_B_tot_code": actual_total,
        "decomposition_relative_error": decomposition_error,
        "particle_count": 96,
        "phi_bytewise_preserved": (
            fields["phi"]["sha256"]
            == source["fields"]["phi"]["sha256"]
        ),
        "delta_C_relaxation_bytewise_preserved": (
            fields["delta_C_relaxation_total"]["sha256"]
            == source["fields"]["delta_C_relaxation_total"]["sha256"]
        ),
        "GP_enabled": False,
        "external_source_enabled": False,
        "new_beta_nucleation_enabled": False,
    }
    write_json(args.out / "rebase_audit.json", audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
