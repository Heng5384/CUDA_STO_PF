#!/usr/bin/env python3
"""Prepare the frozen 96-cube CUDA A--E validation inputs without CUDA.

The five cases are deliberately materialized as input evidence only.  This
script never invokes a PF executable, mutates the frozen state-closure-v1
handoff package, changes a physical parameter, or creates a GP release/
conversion/birth path.

Case A is the original six-particle raw field.  Case B carries a zero-auxiliary
v2 sidecar while preserving Case A's raw bytes.  Case C carries the exact
nonzero *storage-only* S2 GP/sub-grid populations while retaining B's raw
bytes, so its full four-bucket total intentionally differs from B.  Case D
uses the exact frozen S3 matrix-to-GP transfer.  Case E is materialized from
the frozen fixture-conditioned v2 package, not rebuilt from v1 or a new
source allocation.

All new files live below outputs/kwn_pf_cuda_runtime_closure_v1/cuda_ae_assets.
The generated index is intended to be the only handoff needed by the remote
CUDA submission script.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from coupling.fixture_conditioned_handoff_v2 import (  # noqa: E402
    AUXILIARY_SIDECAR_FILENAME,
    FIXTURE_CONDITIONED_DISCLAIMER,
    FixtureConditionedHandoffError,
    FrozenPopulation,
    build_fixture_conditioned_handoff_v2,
    field_matrix_inventory_mol,
    field_resolved_inventory_mol,
    load_host_96cube_fixture,
    load_validation_contract,
    map_matrix_inventory_preserving_fixture,
    prescribed_population_from_inventory,
    read_fixture_conditioned_handoff_v2,
    reconstruct_matrix_xb_alpha,
    sha256_file,
    validate_fixture_conditioned_handoff_v2,
    write_fixture_conditioned_handoff_v2,
)
from scripts.materialize_fixture_conditioned_kwn_pf_handoff_v2_raw_init import (  # noqa: E402
    RawInitMaterializationError,
    materialize_fixture_conditioned_handoff_v2_raw_init,
)


OUTPUT_ROOT = ROOT / "outputs" / "kwn_pf_cuda_runtime_closure_v1"
DEFAULT_OUTPUT = OUTPUT_ROOT / "cuda_ae_assets"
DEFAULT_CONTRACT = ROOT / "contracts" / "pf_kwn_validation_contract_v1.json"
DEFAULT_FIXTURE_SPEC = (
    ROOT
    / "data"
    / "qualification"
    / "pf_mass_conserving_library_handoff_v1"
    / "six_particle_96cube_spec.json"
)
DEFAULT_PROFILE_ROOT = (
    ROOT.parent
    / "CUDA_STO_PF"
    / "data"
    / "qualification"
    / "pf_elastic_target_profile_quarter_nm_v2"
    / "profiles"
)
DEFAULT_FROZEN_HANDOFF = (
    ROOT
    / "outputs"
    / "kwn_pf_state_closure_v1"
    / "kwn_pf_handoff_fixture_conditioned_v2"
)
DEFAULT_STORAGE_SUMMARY = (
    ROOT
    / "outputs"
    / "kwn_pf_state_closure_v1"
    / "four_bucket_storage_control_summary.json"
)

ASSET_SCHEMA = "PF_CUDA_AE_VALIDATION_ASSETS_V1"
RAW_META_SCHEMA = "PF_CUDA_AE_RAW_INIT_META_V1"
INDEX_FILENAME = "cuda_ae_asset_index.json"
RELATIVE_TOLERANCE = 1.0e-10


class CudaAeAssetError(RuntimeError):
    """Raised when an A--E input cannot be prepared without changing scope."""


def _inside_output_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(OUTPUT_ROOT.resolve())
        return True
    except ValueError:
        return False


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise CudaAeAssetError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CudaAeAssetError(f"{label} must be finite")
    return result


def _relative_error(lhs: float, rhs: float) -> float:
    return abs(float(lhs) - float(rhs)) / max(abs(float(rhs)), 1.0e-300)


def _require_close(label: str, lhs: float, rhs: float, *, tolerance: float = RELATIVE_TOLERANCE) -> None:
    relative = _relative_error(lhs, rhs)
    if relative > tolerance:
        raise CudaAeAssetError(
            f"{label} differs by {relative:.17g}, above {tolerance:.17g}"
        )


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CudaAeAssetError(f"{label} must be an object")
    return value


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CudaAeAssetError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise CudaAeAssetError(f"{label} must be a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_raw_f64(path: Path, field: np.ndarray) -> None:
    np.ascontiguousarray(np.asarray(field, dtype="<f8")).ravel(order="C").tofile(path)


def _array_manifest(arrays: Mapping[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
    """Match the v2 content manifest without depending on its private helper."""

    result: Dict[str, Dict[str, Any]] = {}
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
        result[name] = {
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            "shape": list(array.shape),
            "dtype": "float64",
        }
    return result


def _four_bucket_ledger(
    *,
    q_matrix: float,
    q_gp: float,
    q_subgrid: float,
    q_resolved: float,
    box_volume_m3: float,
    q_total: float | None = None,
) -> Dict[str, float]:
    bucket_sum = q_matrix + q_gp + q_subgrid + q_resolved
    total = bucket_sum if q_total is None else float(q_total)
    residual = total - bucket_sum
    return {
        "box_volume_m3": float(box_volume_m3),
        "Q_B_total_mol": total,
        "Q_B_matrix_mol": float(q_matrix),
        "Q_B_GP_mol": float(q_gp),
        "Q_B_beta_subgrid_mol": float(q_subgrid),
        "Q_B_beta_resolved_fixed_mol": float(q_resolved),
        "Q_B_bucket_sum_mol": bucket_sum,
        "residual_mol": residual,
        "relative_residual": abs(residual) / max(abs(total), 1.0e-300),
        "unit_Q_B": "mol_B",
        "unit_C_B": "mol_B_m-3",
        "C_B_total_mol_m3": total / box_volume_m3,
        "C_B_matrix_mol_m3": q_matrix / box_volume_m3,
        "C_B_GP_mol_m3": q_gp / box_volume_m3,
        "C_B_beta_subgrid_mol_m3": q_subgrid / box_volume_m3,
        "C_B_beta_resolved_fixed_mol_m3": q_resolved / box_volume_m3,
    }


def _storage_population(
    *,
    name: str,
    inventory_mol: float,
    fixture: Any,
    contract: Any,
) -> FrozenPopulation:
    if name == "GP":
        edges = (0.75e-9, 1.25e-9, 1.75e-9)
        weights = (0.45, 0.55)
        x_b = 0.03
        vm = contract.vm_alpha_m3_mol
    elif name == "beta_subgrid":
        edges = (1.75e-9, 2.25e-9, 2.75e-9)
        weights = (0.5, 0.5)
        x_b = 1.0
        vm = contract.vm_beta_m3_mol
    else:
        raise CudaAeAssetError(f"unsupported frozen population: {name}")
    return prescribed_population_from_inventory(
        name=name,
        target_inventory_mol=inventory_mol,
        box_volume_m3=fixture.box_volume_m3,
        radius_bin_edges_m=edges,
        volume_weights=weights,
        x_b=x_b,
        vm_m3_mol=vm,
        provenance={
            "kind": "PRESCRIBED_NON_PREDICTIVE_STORAGE_CONTROL",
            "backend": "compact_host_population_state_v2",
            "disclaimer": FIXTURE_CONDITIONED_DISCLAIMER,
        },
    )


def _population_arrays(prefix: str, population: FrozenPopulation, fixture: Any) -> Dict[str, np.ndarray]:
    return {
        f"{prefix}_radius_bin_edges_m": np.asarray(
            population.radius_bin_edges_m, dtype=np.float64
        ),
        f"{prefix}_number_density_per_m4": np.asarray(
            population.number_density_per_m4, dtype=np.float64
        ),
        f"{prefix}_expected_count": np.asarray(
            population.expected_count(fixture.box_volume_m3), dtype=np.float64
        ),
    }


def _population_metadata(population: FrozenPopulation, inventory_mol: float) -> Dict[str, Any]:
    if population.name == "GP":
        return {
            "xB_g": population.x_b,
            "Vm_m3_mol": population.vm_m3_mol,
            "inventory_mol": inventory_mol,
            "volume_fraction": population.volume_fraction(),
            "provenance": dict(population.provenance),
        }
    return {
        "xB_beta": population.x_b,
        "Vm_m3_mol": population.vm_m3_mol,
        "inventory_mol": inventory_mol,
        "volume_fraction": population.volume_fraction(),
        "provenance": dict(population.provenance),
    }


def _build_custom_auxiliary_package(
    *,
    fixture: Any,
    contract: Any,
    q_gp_requested: float,
    q_subgrid_requested: float,
    q_matrix_target: float,
    time_state_label: str,
    conditioning_mode: str,
    extra_context: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, Any]]:
    """Build an honest v2 sidecar around one fixed matrix target.

    The v2 writer/reader remains the authority for schema and sidecar syntax.
    This narrow adapter is needed because Case C intentionally preserves the
    original matrix field while adding frozen storage, whereas the stock v2
    builder always rebalances the matrix to keep the source total fixed.
    """

    template_metadata, template_arrays, _ = build_fixture_conditioned_handoff_v2(
        fixture,
        contract,
        gp_fraction_of_source_matrix=0.0,
        beta_subgrid_fraction_of_source_matrix=0.0,
    )
    gp = _storage_population(
        name="GP",
        inventory_mol=q_gp_requested,
        fixture=fixture,
        contract=contract,
    )
    subgrid = _storage_population(
        name="beta_subgrid",
        inventory_mol=q_subgrid_requested,
        fixture=fixture,
        contract=contract,
    )
    q_gp = gp.inventory_mol(fixture.box_volume_m3)
    q_subgrid = subgrid.inventory_mol(fixture.box_volume_m3)
    target_x_b, matrix_audit = map_matrix_inventory_preserving_fixture(
        fixture,
        contract,
        q_matrix_target,
    )
    arrays = {
        name: np.array(value, dtype=np.float64, copy=True)
        for name, value in template_arrays.items()
    }
    arrays["matrix_baseline_xB"] = np.asarray(
        [matrix_audit["baseline_xB"]], dtype=np.float64
    )
    arrays.update(_population_arrays("gp", gp, fixture))
    arrays.update(_population_arrays("beta_subgrid", subgrid, fixture))
    reconstructed = reconstruct_matrix_xb_alpha(fixture, arrays)
    if float(np.max(np.abs(reconstructed - target_x_b))) != 0.0:
        raise CudaAeAssetError("custom package baseline does not reproduce inverse-mapped matrix")
    q_matrix = field_matrix_inventory_mol(
        fixture.alpha,
        reconstructed,
        fixture.voxel_volume_m3,
        contract.vm_alpha_m3_mol,
    )
    q_resolved = field_resolved_inventory_mol(
        fixture.h_phi,
        contract.v_B,
        fixture.voxel_volume_m3,
        contract.vm_beta_m3_mol,
    )
    ledger = _four_bucket_ledger(
        q_matrix=q_matrix,
        q_gp=q_gp,
        q_subgrid=q_subgrid,
        q_resolved=q_resolved,
        box_volume_m3=fixture.box_volume_m3,
    )
    metadata: Dict[str, Any] = copy.deepcopy(template_metadata)
    metadata["time_state_label"] = time_state_label
    metadata["conditioning_mode"] = conditioning_mode
    metadata["auxiliary_state"]["frozen"] = True
    metadata["auxiliary_state"]["GP"] = _population_metadata(gp, q_gp)
    metadata["auxiliary_state"]["beta_subgrid"] = _population_metadata(subgrid, q_subgrid)
    metadata["matrix_field_mapping"]["method"] = (
        "baseline_only_inverse_storage_preserve_phi_and_delta_C_relaxation"
    )
    metadata["matrix_field_mapping"]["audit"] = matrix_audit
    metadata["matrix_field_mapping"]["clipping_used"] = False
    metadata["matrix_field_mapping"]["phi_reset"] = False
    metadata["matrix_field_mapping"]["delta_C_relaxation_reset"] = False
    metadata["ledger"] = ledger
    metadata["pf_raw_initialization_emitted"] = False
    metadata["pf_raw_initialization_allowed"] = False
    metadata["cuda_ae_case_context"] = dict(extra_context)
    metadata["array_manifest"] = _array_manifest(arrays)
    report = validate_fixture_conditioned_handoff_v2(metadata, arrays, fixture, contract)
    return metadata, arrays, report


def _read_frozen_storage_values(summary_path: Path, fixture: Any, contract: Any) -> Dict[str, float]:
    """Read, rather than re-choose, the S2 and S3 inventories frozen last round."""

    summary = _read_json(summary_path, "frozen four-bucket storage summary")
    if summary.get("status") != "PASS_FOUR_BUCKET_HOST_STORAGE_CONTROL_NOT_CUDA":
        raise CudaAeAssetError("frozen four-bucket storage summary is not a passing host control")
    if summary.get("contract_hash") != contract.contract_hash:
        raise CudaAeAssetError("frozen four-bucket storage summary contract hash mismatch")
    fixture_summary = _require_mapping(summary.get("fixture"), "frozen summary fixture")
    if fixture_summary.get("fixture_hash") != fixture.fixture_hash:
        raise CudaAeAssetError("frozen four-bucket storage summary fixture hash mismatch")
    s2 = _require_mapping(
        summary.get("S2_nonzero_frozen_aux_storage_only"), "frozen S2 storage control"
    )
    s3 = _require_mapping(
        summary.get("S3_exact_matrix_to_GP_inverse_transfer_and_reverse"),
        "frozen S3 transfer control",
    )
    if s2.get("status") != "PASS" or s3.get("status") != "PASS":
        raise CudaAeAssetError("frozen S2/S3 control was not passing")
    s2_ledger = _require_mapping(s2.get("ledger"), "frozen S2 ledger")
    s3_ledger = _require_mapping(s3.get("ledger_forward"), "frozen S3 ledger")
    q_gp_c = _finite(s2_ledger.get("Q_B_GP_mol"), "S2 GP inventory")
    q_subgrid_c = _finite(
        s2_ledger.get("Q_B_beta_subgrid_mol"), "S2 beta-subgrid inventory"
    )
    q_transfer_d = _finite(s3.get("transfer_delta_Q_mol"), "S3 transfer Delta Q")
    _require_close(
        "S2 source matrix inventory",
        _finite(s2_ledger.get("Q_B_matrix_mol"), "S2 matrix inventory"),
        fixture.source_matrix_inventory_mol,
    )
    _require_close(
        "S3 source total inventory",
        _finite(s3_ledger.get("Q_B_total_mol"), "S3 total inventory"),
        fixture.source_total_inventory_mol,
    )
    _require_close(
        "S3 exact GP transfer inventory",
        _finite(s3_ledger.get("Q_B_GP_mol"), "S3 GP inventory"),
        q_transfer_d,
    )
    if q_gp_c <= 0.0 or q_subgrid_c <= 0.0 or q_transfer_d <= 0.0:
        raise CudaAeAssetError("frozen S2/S3 controls do not contain the required nonzero inventory")
    return {
        "case_C_Q_B_GP_mol": q_gp_c,
        "case_C_Q_B_beta_subgrid_mol": q_subgrid_c,
        "case_D_transfer_delta_Q_mol": q_transfer_d,
        "summary_sha256": sha256_file(summary_path),
    }


def _fixture_xb_max_safe(spec_path: Path) -> float:
    spec = _read_json(spec_path, "fixture specification")
    target = _require_mapping(spec.get("target"), "fixture target")
    value = _finite(target.get("xB_max_safe"), "fixture xB_max_safe")
    if not 0.0 < value <= 1.0:
        raise CudaAeAssetError("fixture xB_max_safe is outside (0, 1]")
    return value


def _raw_field_ledger(phi: np.ndarray, x_b: np.ndarray, fixture: Any, contract: Any) -> Dict[str, float]:
    q_matrix = field_matrix_inventory_mol(
        fixture.alpha, x_b, fixture.voxel_volume_m3, contract.vm_alpha_m3_mol
    )
    q_resolved = field_resolved_inventory_mol(
        fixture.h_phi, contract.v_B, fixture.voxel_volume_m3, contract.vm_beta_m3_mol
    )
    h = fixture.h_phi
    field_mean = float(
        np.mean((1.0 - h) * x_b + h * contract.v_B, dtype=np.float64)
    )
    return {
        "Q_B_matrix_mol": q_matrix,
        "Q_B_beta_resolved_mol": q_resolved,
        "Q_B_field_total_mol": q_matrix + q_resolved,
        "mean_C_B_code": field_mean,
        "phi_min": float(np.min(phi)),
        "phi_max": float(np.max(phi)),
        "xB_alpha_min": float(np.min(x_b)),
        "xB_alpha_max": float(np.max(x_b)),
    }


def _write_raw_case(
    *,
    case_dir: Path,
    case_id: str,
    phi: np.ndarray,
    x_b: np.ndarray,
    fixture: Any,
    contract: Any,
    x_b_max_safe: float,
    raw_identity: Mapping[str, str] | None,
    extra_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    if case_dir.exists():
        raise CudaAeAssetError(f"refusing to overwrite case directory: {case_dir}")
    if phi.shape != contract.grid_shape or x_b.shape != contract.grid_shape:
        raise CudaAeAssetError(f"{case_id} raw fields do not match the frozen 96-cube")
    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(x_b)):
        raise CudaAeAssetError(f"{case_id} raw fields contain NaN or Inf")
    if float(np.min(phi)) < 0.0 or float(np.max(phi)) > 1.0:
        raise CudaAeAssetError(f"{case_id} phi is outside [0,1]")
    if float(np.min(x_b)) <= 0.0 or float(np.max(x_b)) > x_b_max_safe:
        raise CudaAeAssetError(f"{case_id} matrix composition would require a clamp")

    case_dir.mkdir(parents=True)
    phi_path = case_dir / "phi_init.raw.f64"
    x_b_path = case_dir / "xB_init.raw.f64"
    _write_raw_f64(phi_path, phi)
    _write_raw_f64(x_b_path, x_b)
    ledger = _raw_field_ledger(phi, x_b, fixture, contract)
    meta: Dict[str, Any] = {
        "schema": RAW_META_SCHEMA,
        "case_id": case_id,
        "Nx": int(contract.grid_shape[0]),
        "Ny": int(contract.grid_shape[1]),
        "Nz": int(contract.grid_shape[2]),
        "dx_nm": contract.dx_m * 1.0e9,
        "interface_width_nm": float(contract.canonical.value("interface.lambda_sm_m")) * 1.0e9,
        "dt_recommended": _finite(contract.canonical.value("numerics.dt_code"), "contract dt"),
        "mean_xBtot": ledger["mean_C_B_code"],
        "xB_max_safe": x_b_max_safe,
        "dtype": "float64",
        "order": "C",
        "phi_path": phi_path.name,
        "xB_path": x_b_path.name,
        "phi_sha256": sha256_file(phi_path),
        "xB_sha256": sha256_file(x_b_path),
        "validation_contract_hash": contract.contract_hash,
        "fresh_dY_dt_prev_contract": "zero_for_fresh_dynamic_start",
        "raw_field_scope": "MATRIX_PLUS_FIXED_RESOLVED_BETA_ONLY",
        "physical_parameter_changes": "NONE",
        "cuda_execution": "NOT_RUN_ASSET_PREPARATION_ONLY",
    }
    if raw_identity is not None:
        required = (
            "source_handoff_hash",
            "package_hash",
            "fixture_hash",
            "auxiliary_sidecar_sha256",
        )
        for key in required:
            value = raw_identity.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise CudaAeAssetError(f"{case_id} raw identity lacks {key}")
            meta[key] = value
    meta.update(dict(extra_meta))
    meta_path = case_dir / "init_meta.json"
    _write_json(meta_path, meta)
    _write_json(case_dir / "raw_field_ledger.json", ledger)
    return {
        "directory": str(case_dir),
        "phi_path": str(phi_path),
        "xB_path": str(x_b_path),
        "init_meta_path": str(meta_path),
        "raw_field_ledger_path": str(case_dir / "raw_field_ledger.json"),
        "phi_sha256": meta["phi_sha256"],
        "xB_sha256": meta["xB_sha256"],
        "init_meta_sha256": sha256_file(meta_path),
        "ledger": ledger,
    }


def _write_adapter_package(
    *, directory: Path,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    fixture: Any,
    contract: Any,
) -> Dict[str, Any]:
    paths = write_fixture_conditioned_handoff_v2(
        directory, metadata, arrays, fixture, contract
    )
    restored, restored_arrays, report = read_fixture_conditioned_handoff_v2(
        directory, fixture, contract
    )
    if report.get("status") != "PASS_FIXTURE_CONDITIONED_HANDOFF_V2":
        raise CudaAeAssetError("written v2 package did not pass its reader validation")
    return {
        "metadata": restored,
        "arrays": restored_arrays,
        "report": report,
        "paths": {name: str(path) for name, path in paths.items()},
        "sidecar_sha256": sha256_file(paths["auxiliary_sidecar"]),
    }


def _sidecar_identity(metadata: Mapping[str, Any], sidecar_sha256: str) -> Dict[str, str]:
    return {
        "source_handoff_hash": str(metadata["source_handoff_hash"]),
        "package_hash": str(metadata["package_hash"]),
        "fixture_hash": str(metadata["fixture_hash"]),
        "auxiliary_sidecar_sha256": sidecar_sha256,
    }


def _case_entry(
    *,
    case_id: str,
    semantics: str,
    raw: Mapping[str, Any],
    sidecar: Mapping[str, Any] | None,
    full_ledger: Mapping[str, Any],
    extra: Mapping[str, Any],
) -> Dict[str, Any]:
    raw_directory = Path(str(raw["directory"]))
    case_directory = raw_directory.parent if raw_directory.name == "raw_init" else raw_directory
    result: Dict[str, Any] = {
        "case_id": case_id,
        "case_directory": str(case_directory),
        "semantics": semantics,
        "raw_fields": dict(raw),
        "full_four_bucket_ledger": dict(full_ledger),
        "prohibited_dynamics": {
            "GP_release": False,
            "GP_to_beta_conversion": False,
            "beta_birth": False,
            "online_KWN_callback": False,
            "matrix_global_reset": False,
            "composition_clamp": False,
        },
        "cuda_execution": "NOT_RUN_ASSET_PREPARATION_ONLY",
    }
    if sidecar is None:
        result["requires_auxiliary_sidecar"] = False
        result["auxiliary_sidecar"] = {
            "present": False,
            "Q_B_GP_mol": 0.0,
            "Q_B_beta_subgrid_mol": 0.0,
        }
    else:
        metadata = _require_mapping(sidecar["metadata"], "sidecar metadata")
        result["requires_auxiliary_sidecar"] = True
        result["auxiliary_sidecar"] = {
            "present": True,
            "path": sidecar["paths"]["auxiliary_sidecar"],
            "filename": Path(str(sidecar["paths"]["auxiliary_sidecar"])).name,
            "sha256": sidecar["sidecar_sha256"],
            "schema_version": metadata["auxiliary_sidecar"]["schema_version"],
            "identity": {
                "validation_contract_hash": metadata["contract_hash"],
                **_sidecar_identity(metadata, str(sidecar["sidecar_sha256"])),
            },
            "package_directory": sidecar["paths"]["directory"],
            "package_metadata_path": sidecar["paths"]["metadata"],
            "package_hash": metadata["package_hash"],
            "source_handoff_hash": metadata["source_handoff_hash"],
            "fixture_hash": metadata["fixture_hash"],
            "validation_contract_hash": metadata["contract_hash"],
        }
    result.update(dict(extra))
    return result


def _copy_frozen_sidecar(source: Path, destination: Path) -> str:
    if destination.exists():
        raise CudaAeAssetError(f"refusing to overwrite copied frozen sidecar: {destination}")
    shutil.copyfile(source, destination)
    source_hash = sha256_file(source)
    copied_hash = sha256_file(destination)
    if copied_hash != source_hash:
        raise CudaAeAssetError("copied frozen sidecar is not byte-identical")
    return copied_hash


def prepare_assets(
    *,
    out: Path,
    contract_path: Path,
    fixture_spec_path: Path,
    profile_root: Path,
    frozen_handoff_dir: Path,
    storage_summary_path: Path,
) -> Dict[str, Any]:
    output = Path(out)
    if not _inside_output_root(output):
        raise CudaAeAssetError(f"refusing output outside task result root: {output}")
    if output.exists():
        raise CudaAeAssetError(f"refusing to overwrite existing asset root: {output}")
    try:
        contract = load_validation_contract(contract_path)
        fixture = load_host_96cube_fixture(fixture_spec_path, profile_root, contract)
        frozen_metadata, frozen_arrays, frozen_report = read_fixture_conditioned_handoff_v2(
            frozen_handoff_dir, fixture, contract
        )
    except FixtureConditionedHandoffError as error:
        raise CudaAeAssetError(str(error)) from error
    if frozen_report.get("status") != "PASS_FIXTURE_CONDITIONED_HANDOFF_V2":
        raise CudaAeAssetError("frozen fixture-conditioned v2 package is not passing")
    frozen_sidecar = frozen_handoff_dir / AUXILIARY_SIDECAR_FILENAME
    if not frozen_sidecar.is_file():
        raise CudaAeAssetError("frozen fixture-conditioned v2 package has no sidecar")
    storage_values = _read_frozen_storage_values(storage_summary_path, fixture, contract)
    x_b_max_safe = _fixture_xb_max_safe(fixture_spec_path)
    original_raw_ledger = _raw_field_ledger(fixture.phi, fixture.xB_alpha, fixture, contract)
    _require_close(
        "original fixture matrix inventory",
        original_raw_ledger["Q_B_matrix_mol"],
        fixture.source_matrix_inventory_mol,
    )
    _require_close(
        "original fixture resolved inventory",
        original_raw_ledger["Q_B_beta_resolved_mol"],
        fixture.source_resolved_inventory_mol,
    )

    output.mkdir(parents=True)

    # A: direct frozen source fields; no auxiliary adapter or sidecar.
    raw_a = _write_raw_case(
        case_dir=output / "case_A_baseline_legacy_zero_aux",
        case_id="A_BASELINE_LEGACY_ZERO_AUX",
        phi=fixture.phi,
        x_b=fixture.xB_alpha,
        fixture=fixture,
        contract=contract,
        x_b_max_safe=x_b_max_safe,
        raw_identity=None,
        extra_meta={
            "initial_state_class": "MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1",
            "fixture_hash": fixture.fixture_hash,
            "adapter": "NONE_ORIGINAL_FROZEN_SIX_PARTICLE_FIXTURE",
        },
    )
    ledger_a = _four_bucket_ledger(
        q_matrix=raw_a["ledger"]["Q_B_matrix_mol"],
        q_gp=0.0,
        q_subgrid=0.0,
        q_resolved=raw_a["ledger"]["Q_B_beta_resolved_mol"],
        box_volume_m3=fixture.box_volume_m3,
    )

    # B: a valid zero-auxiliary v2 sidecar, but direct raw pass-through is
    # mandatory to make the A/B local input bytes literally identical.
    metadata_b, arrays_b, _ = build_fixture_conditioned_handoff_v2(
        fixture,
        contract,
        gp_fraction_of_source_matrix=0.0,
        beta_subgrid_fraction_of_source_matrix=0.0,
    )
    case_b_dir = output / "case_B_identity_adapter_zero_aux"
    case_b_dir.mkdir()
    package_b = _write_adapter_package(
        directory=case_b_dir / "adapter_package",
        metadata=metadata_b,
        arrays=arrays_b,
        fixture=fixture,
        contract=contract,
    )
    raw_b = _write_raw_case(
        case_dir=case_b_dir / "raw_init",
        case_id="B_IDENTITY_ADAPTER_ZERO_AUX",
        phi=fixture.phi,
        x_b=fixture.xB_alpha,
        fixture=fixture,
        contract=contract,
        x_b_max_safe=x_b_max_safe,
        raw_identity=_sidecar_identity(package_b["metadata"], package_b["sidecar_sha256"]),
        extra_meta={
            "initial_state_class": "MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1",
            "adapter": "ZERO_AUXILIARY_STORAGE_ONLY_RAW_FIELD_PASSTHROUGH",
            "auxiliary_inventory_embedded_in_raw_fields": False,
        },
    )
    b_adapter_reconstruction = reconstruct_matrix_xb_alpha(fixture, package_b["arrays"])
    b_adapter_difference = float(np.max(np.abs(b_adapter_reconstruction - fixture.xB_alpha)))

    # C: exact frozen S2 populations, intentionally *not* matrix-rebalanced.
    # The custom v2 package is still written/read through the common schema.
    metadata_c, arrays_c, _ = _build_custom_auxiliary_package(
        fixture=fixture,
        contract=contract,
        q_gp_requested=storage_values["case_C_Q_B_GP_mol"],
        q_subgrid_requested=storage_values["case_C_Q_B_beta_subgrid_mol"],
        q_matrix_target=fixture.source_matrix_inventory_mol,
        time_state_label="cuda_ae_case_C_nonzero_frozen_aux_storage_t0",
        conditioning_mode="NONZERO_FROZEN_AUX_STORAGE_ONLY_NO_MATRIX_REBALANCE",
        extra_context={
            "case": "C_NONZERO_FROZEN_AUX_STORAGE",
            "source_summary_sha256": storage_values["summary_sha256"],
            "matrix_rebalanced": False,
            "dynamics": "STORAGE_ONLY_NO_PF_FIELD_MUTATION",
        },
    )
    case_c_dir = output / "case_C_nonzero_frozen_aux_storage"
    case_c_dir.mkdir()
    package_c = _write_adapter_package(
        directory=case_c_dir / "adapter_package",
        metadata=metadata_c,
        arrays=arrays_c,
        fixture=fixture,
        contract=contract,
    )
    raw_c = _write_raw_case(
        case_dir=case_c_dir / "raw_init",
        case_id="C_NONZERO_FROZEN_AUX_STORAGE",
        phi=fixture.phi,
        x_b=fixture.xB_alpha,
        fixture=fixture,
        contract=contract,
        x_b_max_safe=x_b_max_safe,
        raw_identity=_sidecar_identity(package_c["metadata"], package_c["sidecar_sha256"]),
        extra_meta={
            "initial_state_class": "MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1",
            "adapter": "NONZERO_FROZEN_AUX_STORAGE_ONLY_RAW_FIELD_PASSTHROUGH",
            "auxiliary_inventory_embedded_in_raw_fields": False,
            "matrix_rebalanced": False,
        },
    )

    # D: use the exact Delta Q recorded by S3, rather than a newly selected
    # fraction.  The inverse map writes the only changed local field.
    q_d_matrix_target = fixture.source_matrix_inventory_mol - storage_values[
        "case_D_transfer_delta_Q_mol"
    ]
    if q_d_matrix_target <= 0.0:
        raise CudaAeAssetError("frozen S3 transfer exhausts the matrix inventory")
    metadata_d, arrays_d, _ = _build_custom_auxiliary_package(
        fixture=fixture,
        contract=contract,
        q_gp_requested=storage_values["case_D_transfer_delta_Q_mol"],
        q_subgrid_requested=0.0,
        q_matrix_target=q_d_matrix_target,
        time_state_label="cuda_ae_case_D_exact_matrix_to_GP_control_t0",
        conditioning_mode="EXACT_MATRIX_TO_GP_INVERSE_TRANSFER_FROM_FROZEN_S3_LEDGER",
        extra_context={
            "case": "D_CONSERVATIVE_MATRIX_TO_GP_CONTROL",
            "source_summary_sha256": storage_values["summary_sha256"],
            "transfer_delta_Q_mol": storage_values["case_D_transfer_delta_Q_mol"],
            "clipping_used": False,
            "matrix_global_reset": False,
        },
    )
    case_d_dir = output / "case_D_conservative_matrix_to_GP_control"
    case_d_dir.mkdir()
    package_d = _write_adapter_package(
        directory=case_d_dir / "adapter_package",
        metadata=metadata_d,
        arrays=arrays_d,
        fixture=fixture,
        contract=contract,
    )
    d_x_b = reconstruct_matrix_xb_alpha(fixture, package_d["arrays"])
    raw_d = _write_raw_case(
        case_dir=case_d_dir / "raw_init",
        case_id="D_CONSERVATIVE_MATRIX_TO_GP_CONTROL",
        phi=fixture.phi,
        x_b=d_x_b,
        fixture=fixture,
        contract=contract,
        x_b_max_safe=x_b_max_safe,
        raw_identity=_sidecar_identity(package_d["metadata"], package_d["sidecar_sha256"]),
        extra_meta={
            "initial_state_class": "MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1",
            "adapter": "EXACT_FROZEN_S3_MATRIX_TO_GP_INVERSE_MAP",
            "transfer_delta_Q_mol": storage_values["case_D_transfer_delta_Q_mol"],
            "matrix_global_reset": False,
            "clipping_used": False,
        },
    )

    # E: materialize precisely the frozen v2 package.  That helper refuses a
    # synthetic profile source and verifies the package before writing raw
    # fields.  A byte-identical sidecar copy keeps all remote inputs together.
    case_e_dir = output / "case_E_fixture_conditioned_kwn_handoff_v2"
    try:
        materialized_e = materialize_fixture_conditioned_handoff_v2_raw_init(
            handoff_dir=frozen_handoff_dir,
            out=case_e_dir,
            contract_path=contract_path,
            fixture_spec_path=fixture_spec_path,
            profile_root=profile_root,
        )
    except RawInitMaterializationError as error:
        raise CudaAeAssetError(f"cannot materialize frozen Case E raw fields: {error}") from error
    copied_e_sidecar = case_e_dir / AUXILIARY_SIDECAR_FILENAME
    copied_e_sidecar_hash = _copy_frozen_sidecar(frozen_sidecar, copied_e_sidecar)
    e_meta = _read_json(materialized_e["init_meta"], "Case E raw init metadata")
    d_e_phi = np.fromfile(materialized_e["phi"], dtype="<f8").reshape(contract.grid_shape)
    d_e_x_b = np.fromfile(materialized_e["xB"], dtype="<f8").reshape(contract.grid_shape)
    raw_e_ledger = _raw_field_ledger(d_e_phi, d_e_x_b, fixture, contract)
    _write_json(case_e_dir / "raw_field_ledger.json", raw_e_ledger)

    # Narrow checks whose outcomes decide whether the forthcoming CUDA launch
    # receives a usable A--E matrix or must stop before submitting anything.
    bytes_a_b_phi = raw_a["phi_sha256"] == raw_b["phi_sha256"]
    bytes_a_b_x_b = raw_a["xB_sha256"] == raw_b["xB_sha256"]
    bytes_b_c_phi = raw_b["phi_sha256"] == raw_c["phi_sha256"]
    bytes_b_c_x_b = raw_b["xB_sha256"] == raw_c["xB_sha256"]
    if not all((bytes_a_b_phi, bytes_a_b_x_b, bytes_b_c_phi, bytes_b_c_x_b)):
        raise CudaAeAssetError("A/B or B/C raw-field byte identity was not preserved")
    _require_close(
        "Case D matrix depletion",
        raw_a["ledger"]["Q_B_matrix_mol"] - raw_d["ledger"]["Q_B_matrix_mol"],
        storage_values["case_D_transfer_delta_Q_mol"],
    )
    _require_close(
        "Case D matrix-plus-GP conservation",
        raw_d["ledger"]["Q_B_matrix_mol"]
        + package_d["report"]["ledger"]["Q_B_GP_mol"],
        raw_a["ledger"]["Q_B_matrix_mol"],
    )
    d_baseline = float(package_d["arrays"]["matrix_baseline_xB"][0])
    d_delta_error = float(
        np.max(
            np.abs(
                fixture.alpha * (d_x_b - d_baseline)
                - fixture.delta_C_relaxation
            )
        )
    )
    if d_delta_error > 5.0e-14:
        raise CudaAeAssetError("Case D inverse map failed to preserve delta_C_relaxation")
    if copied_e_sidecar_hash != frozen_metadata.get("auxiliary_sidecar_sha256"):
        raise CudaAeAssetError("Case E copied sidecar hash does not bind frozen metadata")
    for key in ("validation_contract_hash", "source_handoff_hash", "package_hash", "fixture_hash"):
        expected = {
            "validation_contract_hash": frozen_metadata["contract_hash"],
            "source_handoff_hash": frozen_metadata["source_handoff_hash"],
            "package_hash": frozen_metadata["package_hash"],
            "fixture_hash": frozen_metadata["fixture_hash"],
        }[key]
        if e_meta.get(key) != expected:
            raise CudaAeAssetError(f"Case E raw metadata lost frozen {key}")

    entries = {
        "A": _case_entry(
            case_id="A_BASELINE_LEGACY_ZERO_AUX",
            semantics="ORIGINAL_SIX_PARTICLE_FIXTURE_NO_NEW_ADAPTER_GP_ZERO_SUBGRID_ZERO",
            raw=raw_a,
            sidecar=None,
            full_ledger=ledger_a,
            extra={"matrix_rebalanced": False, "raw_fields_direct_from_fixture": True},
        ),
        "B": _case_entry(
            case_id="B_IDENTITY_ADAPTER_ZERO_AUX",
            semantics="ZERO_AUXILIARY_V2_STORAGE_ADAPTER_WITH_BYTE_IDENTICAL_A_RAW_FIELDS",
            raw=raw_b,
            sidecar=package_b,
            full_ledger=package_b["report"]["ledger"],
            extra={
                "matrix_rebalanced": False,
                "adapter_reconstruction_max_abs_xB_difference_from_source": b_adapter_difference,
                "raw_fields_direct_passthrough": True,
            },
        ),
        "C": _case_entry(
            case_id="C_NONZERO_FROZEN_AUX_STORAGE",
            semantics="FROZEN_S2_GP_AND_SUBGRID_STORAGE_ONLY_B_RAW_FIELDS_RETAINED_FULL_TOTAL_INTENTIONALLY_INCREASED",
            raw=raw_c,
            sidecar=package_c,
            full_ledger=package_c["report"]["ledger"],
            extra={
                "matrix_rebalanced": False,
                "raw_fields_direct_passthrough": True,
                "frozen_storage_summary_sha256": storage_values["summary_sha256"],
            },
        ),
        "D": _case_entry(
            case_id="D_CONSERVATIVE_MATRIX_TO_GP_CONTROL",
            semantics="EXACT_FROZEN_S3_DELTA_Q_MATRIX_TO_GP_INVERSE_MAP_PHI_AND_RESOLVED_BETA_UNCHANGED",
            raw=raw_d,
            sidecar=package_d,
            full_ledger=package_d["report"]["ledger"],
            extra={
                "matrix_rebalanced": True,
                "exact_transfer_delta_Q_mol": storage_values["case_D_transfer_delta_Q_mol"],
                "delta_C_relaxation_reconstruction_max_abs_error": d_delta_error,
                "clipping_used": False,
            },
        ),
        "E": {
            "case_id": "E_FIXTURE_CONDITIONED_KWN_HANDOFF_V2",
            "case_directory": str(case_e_dir),
            "semantics": "RAW_FIELDS_MATERIALIZED_FROM_FROZEN_FIXTURE_CONDITIONED_KWN_HANDOFF_V2_ONLY",
            "raw_fields": {
                "directory": str(case_e_dir),
                "phi_path": str(materialized_e["phi"]),
                "xB_path": str(materialized_e["xB"]),
                "init_meta_path": str(materialized_e["init_meta"]),
                "raw_field_ledger_path": str(case_e_dir / "raw_field_ledger.json"),
                "phi_sha256": sha256_file(materialized_e["phi"]),
                "xB_sha256": sha256_file(materialized_e["xB"]),
                "init_meta_sha256": sha256_file(materialized_e["init_meta"]),
                "ledger": raw_e_ledger,
            },
            "auxiliary_sidecar": {
                "present": True,
                "path": str(copied_e_sidecar),
                "filename": copied_e_sidecar.name,
                "sha256": copied_e_sidecar_hash,
                "source_frozen_path": str(frozen_sidecar),
                "schema_version": frozen_metadata["auxiliary_sidecar"]["schema_version"],
                "identity": {
                    "validation_contract_hash": frozen_metadata["contract_hash"],
                    "source_handoff_hash": frozen_metadata["source_handoff_hash"],
                    "package_hash": frozen_metadata["package_hash"],
                    "fixture_hash": frozen_metadata["fixture_hash"],
                    "auxiliary_sidecar_sha256": copied_e_sidecar_hash,
                },
                "package_hash": frozen_metadata["package_hash"],
                "source_handoff_hash": frozen_metadata["source_handoff_hash"],
                "fixture_hash": frozen_metadata["fixture_hash"],
                "validation_contract_hash": frozen_metadata["contract_hash"],
            },
            "full_four_bucket_ledger": frozen_report["ledger"],
            "requires_auxiliary_sidecar": True,
            "prohibited_dynamics": {
                "GP_release": False,
                "GP_to_beta_conversion": False,
                "beta_birth": False,
                "online_KWN_callback": False,
                "matrix_global_reset": False,
                "composition_clamp": False,
            },
            "frozen_package": {
                "directory": str(frozen_handoff_dir),
                "metadata_sha256": sha256_file(frozen_handoff_dir / "metadata.json"),
                "arrays_sha256": sha256_file(frozen_handoff_dir / "arrays.npz"),
                "ledger_sha256": sha256_file(frozen_handoff_dir / "ledger.csv"),
                "validation_report_sha256": sha256_file(frozen_handoff_dir / "validation_report.json"),
            },
            "cuda_execution": "NOT_RUN_ASSET_PREPARATION_ONLY",
        },
    }
    index: Dict[str, Any] = {
        "schema_version": ASSET_SCHEMA,
        "status": "PASS_CUDA_AE_INPUT_ASSET_PREPARATION_NO_CUDA_EXECUTION",
        "validation_only": True,
        "historical_as_run_claim": False,
        "cuda_execution": "NOT_RUN_ASSET_PREPARATION_ONLY",
        "contract": {
            "path": str(contract_path),
            "hash": contract.contract_hash,
            "file_sha256": sha256_file(contract_path),
        },
        "fixture": {
            "id": fixture.fixture_id,
            "hash": fixture.fixture_hash,
            "source_kind": fixture.source_kind,
            "source_total_inventory_mol": fixture.source_total_inventory_mol,
            "source_matrix_inventory_mol": fixture.source_matrix_inventory_mol,
            "source_resolved_inventory_mol": fixture.source_resolved_inventory_mol,
            "phi_sha256": fixture.field_hashes["phi"],
            "xB_alpha_sha256": fixture.field_hashes["xB_alpha"],
            "delta_C_relaxation_sha256": fixture.field_hashes["delta_C_relaxation"],
        },
        "frozen_source_evidence": {
            "four_bucket_storage_summary_path": str(storage_summary_path),
            "four_bucket_storage_summary_sha256": storage_values["summary_sha256"],
            "frozen_handoff_directory": str(frozen_handoff_dir),
            "frozen_handoff_package_hash": frozen_metadata["package_hash"],
        },
        "assertions": {
            "A_vs_B_phi_raw_byte_identical": bytes_a_b_phi,
            "A_vs_B_xB_raw_byte_identical": bytes_a_b_x_b,
            "B_vs_C_phi_raw_byte_identical": bytes_b_c_phi,
            "B_vs_C_xB_raw_byte_identical": bytes_b_c_x_b,
            "Case_C_full_total_exceeds_B_by_frozen_auxiliary_inventory": True,
            "Case_D_uses_exact_frozen_S3_delta_Q": True,
            "Case_D_phi_is_original_fixture": raw_d["phi_sha256"] == raw_a["phi_sha256"],
            "Case_D_delta_C_relaxation_preserved_without_clipping": d_delta_error <= 5.0e-14,
            "Case_E_sidecar_copy_byte_identical_to_frozen_source": True,
            "all_nonzero_sidecars_pass_v2_reader_validation": True,
        },
        "cases": entries,
        "submission_contract": {
            "only_allowed_grid": [96, 96, 96],
            "GP_release": "OFF",
            "GP_to_beta_conversion": "OFF",
            "beta_birth": "OFF",
            "online_KWN_callback": "OFF",
            "matrix_global_reset": "OFF",
            "composition_clamp": "FORBIDDEN",
            "case_C_note": "storage test only; it is intentionally not the same total-inventory experiment as B",
        },
    }
    _write_json(output / INDEX_FILENAME, index)
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--fixture-spec", type=Path, default=DEFAULT_FIXTURE_SPEC)
    parser.add_argument("--profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
    parser.add_argument("--frozen-handoff-dir", type=Path, default=DEFAULT_FROZEN_HANDOFF)
    parser.add_argument("--storage-summary", type=Path, default=DEFAULT_STORAGE_SUMMARY)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        index = prepare_assets(
            out=args.out,
            contract_path=args.contract,
            fixture_spec_path=args.fixture_spec,
            profile_root=args.profile_root,
            frozen_handoff_dir=args.frozen_handoff_dir,
            storage_summary_path=args.storage_summary,
        )
    except CudaAeAssetError as error:
        raise SystemExit(f"CUDA A--E asset preparation failed: {error}") from error
    print(
        json.dumps(
            {
                "status": index["status"],
                "asset_index": str(Path(args.out) / INDEX_FILENAME),
                "contract_hash": index["contract"]["hash"],
                "fixture_hash": index["fixture"]["hash"],
                "A_B_raw_byte_identity": {
                    "phi": index["assertions"]["A_vs_B_phi_raw_byte_identical"],
                    "xB": index["assertions"]["A_vs_B_xB_raw_byte_identical"],
                },
                "B_C_raw_byte_identity": {
                    "phi": index["assertions"]["B_vs_C_phi_raw_byte_identical"],
                    "xB": index["assertions"]["B_vs_C_xB_raw_byte_identical"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
