#!/usr/bin/env python3
"""Materialize the v2 fixture-conditioned handoff as PF raw-init fields.

The compact v2 handoff intentionally stores only the matrix baseline plus
frozen auxiliary populations.  This script reconstructs the two local PF
fields required by ``main_cuda --init-mode raw_fields`` from the hash-checked
host fixture and that compact baseline:

* ``phi_init.raw.f64`` preserves the existing diffuse resolved-beta geometry;
* ``xB_init.raw.f64`` uses the exact inverse storage mapping
  ``baseline + delta_C_relaxation / (1-h(phi))``.

GP and sub-grid-beta remain compact frozen sidecar state.  They are not
materialized as dense fields, released, converted, or evolved here.  This is
raw-input materialization only; it neither compiles nor runs CUDA/PF.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coupling.fixture_conditioned_handoff_v2 import (  # noqa: E402
    ARRAYS_FILENAME,
    LEDGER_FILENAME,
    METADATA_FILENAME,
    VALIDATION_REPORT_FILENAME,
    FixtureConditionedHandoffError,
    ValidationContract,
    field_matrix_inventory_mol,
    field_resolved_inventory_mol,
    h_of_phi,
    load_host_96cube_fixture,
    load_validation_contract,
    read_fixture_conditioned_handoff_v2,
    reconstruct_matrix_xb_alpha,
    sha256_file,
)


RAW_INIT_META_SCHEMA = "PF_RAW_INIT_META_V1"
PROVENANCE_SCHEMA = "PF_FIXTURE_CONDITIONED_KWN_HANDOFF_V2_RAW_INIT_PROVENANCE_V1"
PHI_FILENAME = "phi_init.raw.f64"
XB_FILENAME = "xB_init.raw.f64"
META_FILENAME = "init_meta.json"
PROVENANCE_FILENAME = "raw_init_provenance_manifest.json"
DEFAULT_HOST_PROFILE_ROOT = (
    ROOT.parent
    / "CUDA_STO_PF"
    / "data"
    / "qualification"
    / "pf_elastic_target_profile_quarter_nm_v2"
    / "profiles"
)
DEFAULT_CONTRACT = ROOT / "contracts" / "pf_kwn_validation_contract_v1.json"
DEFAULT_FIXTURE_SPEC = (
    ROOT
    / "data"
    / "qualification"
    / "pf_mass_conserving_library_handoff_v1"
    / "six_particle_96cube_spec.json"
)


class RawInitMaterializationError(ValueError):
    """Raised when a v2 package cannot honestly become a raw PF initializer."""


def _require_finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise RawInitMaterializationError(f"{label} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RawInitMaterializationError(f"{label} must be a finite positive number")
    return result


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_raw_f64(path: Path, values: np.ndarray) -> None:
    field = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    field.ravel(order="C").tofile(path)


def _source_fixture_xb_max_safe(fixture_spec_path: Path) -> float:
    try:
        document = json.loads(Path(fixture_spec_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RawInitMaterializationError(
            f"cannot read fixture specification: {fixture_spec_path}"
        ) from error
    if not isinstance(document, Mapping):
        raise RawInitMaterializationError("fixture specification must be an object")
    target = document.get("target")
    if not isinstance(target, Mapping):
        raise RawInitMaterializationError("fixture specification has no target object")
    return _require_finite_positive(target.get("xB_max_safe"), "fixture xB_max_safe")


def _contract_scalar(contract: ValidationContract, path: str) -> float:
    return _require_finite_positive(contract.canonical.value(path), path)


def _file_hashes(handoff_dir: Path, auxiliary_filename: str) -> Dict[str, str]:
    required = {
        "metadata": METADATA_FILENAME,
        "arrays": ARRAYS_FILENAME,
        "ledger": LEDGER_FILENAME,
        "validation_report": VALIDATION_REPORT_FILENAME,
        "auxiliary_sidecar": auxiliary_filename,
    }
    result: Dict[str, str] = {}
    for label, filename in required.items():
        path = handoff_dir / filename
        if not path.is_file():
            raise RawInitMaterializationError(
                f"handoff package is missing {label}: {path}"
            )
        result[label] = sha256_file(path)
    return result


def _raw_field_mean_xb_total(
    phi: np.ndarray, x_b: np.ndarray, contract: ValidationContract
) -> float:
    """Return the field-only storage that main_cuda will reconstruct from phi/xB.

    The auxiliary populations deliberately do not appear in a dense raw field;
    their absolute inventories remain in the separately hash-bound sidecar.
    """

    h = h_of_phi(phi)
    return float(np.mean((1.0 - h) * x_b + h * contract.v_B, dtype=np.float64))


def materialize_fixture_conditioned_handoff_v2_raw_init(
    *,
    handoff_dir: Path,
    out: Path,
    contract_path: Path = DEFAULT_CONTRACT,
    fixture_spec_path: Path = DEFAULT_FIXTURE_SPEC,
    profile_root: Path = DEFAULT_HOST_PROFILE_ROOT,
) -> Dict[str, Path]:
    """Write deterministic raw PF fields from one feasible, real-host v2 package.

    ``out`` must not exist.  The source fixture is always reconstructed from
    the frozen host raw profiles; a synthetic control is intentionally not an
    accepted input for this raw-init path.
    """

    output = Path(out)
    if output.exists():
        raise RawInitMaterializationError(f"refusing to overwrite output: {output}")

    try:
        contract = load_validation_contract(Path(contract_path))
        fixture = load_host_96cube_fixture(
            Path(fixture_spec_path), Path(profile_root), contract
        )
        metadata, arrays, report = read_fixture_conditioned_handoff_v2(
            Path(handoff_dir), fixture, contract
        )
    except FixtureConditionedHandoffError as error:
        raise RawInitMaterializationError(str(error)) from error

    if fixture.source_kind != "HOST_RAW_PROFILE_FIELDS_HASH_VALIDATED":
        raise RawInitMaterializationError(
            "raw-init materialization requires hash-validated host profile fields"
        )
    if metadata.get("fixture_source_kind") != fixture.source_kind:
        raise RawInitMaterializationError("handoff fixture source kind does not bind the host fixture")
    if metadata.get("pf_raw_initialization_allowed") is not True:
        raise RawInitMaterializationError(
            "handoff package did not pass the prerequisite raw-initialization gate"
        )
    if report.get("status") != "PASS_FIXTURE_CONDITIONED_HANDOFF_V2":
        raise RawInitMaterializationError("handoff package did not pass v2 validation")
    if metadata.get("contract_hash") != contract.contract_hash:
        raise RawInitMaterializationError("handoff contract hash mismatch")
    if metadata.get("fixture_hash") != fixture.fixture_hash:
        raise RawInitMaterializationError("handoff fixture hash mismatch")

    auxiliary = metadata.get("auxiliary_sidecar")
    if not isinstance(auxiliary, Mapping):
        raise RawInitMaterializationError("handoff has no auxiliary sidecar declaration")
    auxiliary_filename = auxiliary.get("filename")
    if not isinstance(auxiliary_filename, str) or not auxiliary_filename:
        raise RawInitMaterializationError("handoff auxiliary sidecar filename is invalid")
    handoff_hashes = _file_hashes(Path(handoff_dir), auxiliary_filename)

    phi = np.ascontiguousarray(np.asarray(fixture.phi, dtype=np.float64))
    x_b = np.ascontiguousarray(
        np.asarray(reconstruct_matrix_xb_alpha(fixture, arrays), dtype=np.float64)
    )
    if phi.shape != contract.grid_shape or x_b.shape != contract.grid_shape:
        raise RawInitMaterializationError("materialized raw fields do not match the validation grid")
    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(x_b)):
        raise RawInitMaterializationError("materialized raw fields contain NaN or Inf")
    if float(np.min(phi)) < 0.0 or float(np.max(phi)) > 1.0:
        raise RawInitMaterializationError("fixture phi is outside [0, 1]")

    x_b_max_safe = _source_fixture_xb_max_safe(Path(fixture_spec_path))
    if float(np.min(x_b)) <= 0.0 or float(np.max(x_b)) > x_b_max_safe:
        raise RawInitMaterializationError(
            "reconstructed matrix xB violates the fixture's declared safe interval"
        )

    matrix_mapping = metadata.get("matrix_field_mapping")
    if not isinstance(matrix_mapping, Mapping):
        raise RawInitMaterializationError("handoff has no matrix field mapping")
    baseline_value = arrays.get("matrix_baseline_xB")
    if baseline_value is None:
        raise RawInitMaterializationError("handoff has no matrix baseline array")
    baseline = np.asarray(baseline_value, dtype=np.float64)
    if baseline.shape != (1,) or not np.all(np.isfinite(baseline)):
        raise RawInitMaterializationError("handoff matrix baseline is invalid")
    recovered_delta = fixture.alpha * (x_b - float(baseline[0]))
    delta_error = float(
        np.max(np.abs(recovered_delta - fixture.delta_C_relaxation))
    )
    if delta_error > 5.0e-14:
        raise RawInitMaterializationError(
            "matrix inverse did not preserve delta_C_relaxation"
        )
    phi_h_error = float(np.max(np.abs(h_of_phi(phi) - fixture.h_phi)))
    if phi_h_error > 5.0e-14:
        raise RawInitMaterializationError(
            "fixture phi no longer satisfies its h(phi) storage relation"
        )

    q_matrix = field_matrix_inventory_mol(
        fixture.alpha, x_b, fixture.voxel_volume_m3, contract.vm_alpha_m3_mol
    )
    q_resolved = field_resolved_inventory_mol(
        fixture.h_phi,
        contract.v_B,
        fixture.voxel_volume_m3,
        contract.vm_beta_m3_mol,
    )
    ledger = metadata.get("ledger")
    if not isinstance(ledger, Mapping):
        raise RawInitMaterializationError("handoff ledger is invalid")
    for key, actual in (
        ("Q_B_matrix_mol", q_matrix),
        ("Q_B_beta_resolved_fixed_mol", q_resolved),
    ):
        expected = _require_finite_positive(ledger.get(key), f"ledger {key}")
        if abs(actual - expected) > 1.0e-12 * max(abs(expected), 1.0e-300):
            raise RawInitMaterializationError(f"raw field inventory does not bind {key}")

    dt_recommended = _contract_scalar(contract, "numerics.dt_code")
    interface_width_nm = _contract_scalar(contract, "interface.lambda_sm_m") * 1.0e9
    raw_mean_xb_total = _raw_field_mean_xb_total(phi, x_b, contract)
    if not math.isfinite(raw_mean_xb_total):
        raise RawInitMaterializationError("raw field storage is not finite")

    output.mkdir(parents=True)
    phi_path = output / PHI_FILENAME
    x_b_path = output / XB_FILENAME
    _write_raw_f64(phi_path, phi)
    _write_raw_f64(x_b_path, x_b)
    phi_hash = sha256_file(phi_path)
    x_b_hash = sha256_file(x_b_path)
    if phi_hash != fixture.field_hashes["phi"]:
        raise RawInitMaterializationError("written phi raw hash does not preserve fixture phi")

    init_meta: Dict[str, Any] = {
        "schema": RAW_INIT_META_SCHEMA,
        "Nx": int(contract.grid_shape[0]),
        "Ny": int(contract.grid_shape[1]),
        "Nz": int(contract.grid_shape[2]),
        "dx_nm": contract.dx_m * 1.0e9,
        "interface_width_nm": interface_width_nm,
        "dt_recommended": dt_recommended,
        "mean_xBtot": raw_mean_xb_total,
        "xB_max_safe": x_b_max_safe,
        "dtype": "float64",
        "order": "C",
        "phi_path": PHI_FILENAME,
        "xB_path": XB_FILENAME,
        "phi_sha256": phi_hash,
        "xB_sha256": x_b_hash,
        "source_handoff_hash": metadata["source_handoff_hash"],
        "package_hash": metadata["package_hash"],
        "fixture_hash": metadata["fixture_hash"],
        "auxiliary_sidecar_sha256": handoff_hashes["auxiliary_sidecar"],
        "fresh_dY_dt_prev_contract": "zero_for_fresh_dynamic_start",
        "raw_field_scope": "MATRIX_PLUS_FIXED_RESOLVED_BETA_ONLY",
        "frozen_auxiliary_scope": "COMPACT_SIDECAR_ONLY_NO_DENSE_GP_OR_SUBGRID_FIELD",
        "validation_contract_hash": contract.contract_hash,
    }
    meta_path = output / META_FILENAME
    _json_write(meta_path, init_meta)

    source_x_b = np.asarray(fixture.xB_alpha, dtype=np.float64)
    source_shift = x_b - source_x_b
    provenance: Dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA,
        "validation_only": True,
        "pf_execution": "NOT_RUN_RAW_INITIALIZATION_MATERIALIZATION_ONLY",
        "contract": {
            "path": str(Path(contract_path)),
            "sha256": sha256_file(Path(contract_path)),
            "contract_hash": contract.contract_hash,
        },
        "handoff": {
            "directory": str(Path(handoff_dir)),
            "package_hash": metadata.get("package_hash"),
            "source_handoff_hash": metadata.get("source_handoff_hash"),
            "file_sha256": handoff_hashes,
        },
        "fixture": {
            "fixture_id": fixture.fixture_id,
            "fixture_hash": fixture.fixture_hash,
            "source_kind": fixture.source_kind,
            "source_details": dict(fixture.source_details),
            "field_sha256": fixture.field_hashes,
        },
        "raw_fields": {
            "dtype": "float64-le",
            "order": "C",
            "shape": list(contract.grid_shape),
            "phi": {"path": PHI_FILENAME, "sha256": phi_hash},
            "xB_alpha": {"path": XB_FILENAME, "sha256": x_b_hash},
            "phi_min": float(np.min(phi)),
            "phi_max": float(np.max(phi)),
            "xB_min": float(np.min(x_b)),
            "xB_max": float(np.max(x_b)),
            "field_only_mean_xBtot": raw_mean_xb_total,
        },
        "preservation": {
            "phi_fixture_hash_matches_raw": phi_hash == fixture.field_hashes["phi"],
            "h_phi_reconstruction_max_abs_error": phi_h_error,
            "delta_C_relaxation_sha256": fixture.field_hashes["delta_C_relaxation"],
            "delta_C_relaxation_reconstruction_max_abs_error": delta_error,
            "matrix_mapping": matrix_mapping,
            "source_to_target_xB_baseline_shift_min": float(np.min(source_shift)),
            "source_to_target_xB_baseline_shift_max": float(np.max(source_shift)),
        },
        "inventory": {
            "raw_field_matrix_mol": q_matrix,
            "raw_field_resolved_beta_mol": q_resolved,
            "raw_field_matrix_plus_resolved_mol": q_matrix + q_resolved,
            "four_bucket_ledger": dict(ledger),
            "auxiliary_inventory_is_not_embedded_in_raw_fields": True,
            "auxiliary_sidecar": {
                "filename": auxiliary_filename,
                "sha256": handoff_hashes["auxiliary_sidecar"],
                "semantics": auxiliary.get("semantics"),
            },
        },
        "main_cuda_raw_meta": {
            "path": META_FILENAME,
            "sha256": sha256_file(meta_path),
            "required_fields": [
                "Nx",
                "Ny",
                "Nz",
                "dx_nm",
                "interface_width_nm",
                "dtype",
                "order",
            "phi_path",
            "xB_path",
            "source_handoff_hash",
            "package_hash",
            "fixture_hash",
            "auxiliary_sidecar_sha256",
        ],
        },
    }
    provenance_path = output / PROVENANCE_FILENAME
    _json_write(provenance_path, provenance)
    return {
        "directory": output,
        "phi": phi_path,
        "xB": x_b_path,
        "init_meta": meta_path,
        "provenance": provenance_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--handoff-dir",
        type=Path,
        required=True,
        help="existing feasible kwn_pf_handoff_fixture_conditioned_v2 package",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="new caller-owned directory for raw fields; it must not already exist",
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--fixture-spec", type=Path, default=DEFAULT_FIXTURE_SPEC)
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=DEFAULT_HOST_PROFILE_ROOT,
        help="read-only frozen host profile directory; synthetic fallback is never allowed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = materialize_fixture_conditioned_handoff_v2_raw_init(
            handoff_dir=args.handoff_dir,
            out=args.out,
            contract_path=args.contract,
            fixture_spec_path=args.fixture_spec,
            profile_root=args.profile_root,
        )
    except RawInitMaterializationError as error:
        raise SystemExit(f"raw-init materialization failed: {error}") from error
    print(
        json.dumps(
            {
                "status": "RAW_INIT_MATERIALIZED_NO_CUDA_EXECUTION",
                "pf_execution": "NOT_RUN_RAW_INITIALIZATION_MATERIALIZATION_ONLY",
                "paths": {name: str(path) for name, path in paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
