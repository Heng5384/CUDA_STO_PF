#!/usr/bin/env python3
"""Audit the controlled 96-cube CUDA A--E validation run without inventing data.

This reader is deliberately independent of ``main_cuda``.  It validates the
frozen input index, extracts only the V6 checkpoint facts that are actually
present on disk, and writes compact machine-readable summaries below the
caller-owned run root.  It is not a PF, KWN, GP, or microstructure analysis
path: absent CUDA diagnostics remain explicitly ``NOT_CAPTURED``.

The companion Slurm runner uses ``--emit-run-manifest`` before launching any
CUDA work.  That gives the shell a checked, tab-separated view of the A--E
assets without evaluating paths or hashes as shell code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coupling.fixture_conditioned_handoff_v2 import (  # noqa: E402
    field_matrix_inventory_mol,
    field_resolved_inventory_mol,
    h_of_phi,
    load_validation_contract,
)


ASSET_SCHEMA = "PF_CUDA_AE_VALIDATION_ASSETS_V1"
EXPECTED_ASSET_STATUS = "PASS_CUDA_AE_INPUT_ASSET_PREPARATION_NO_CUDA_EXECUTION"
CASES: tuple[str, ...] = ("A", "B", "C", "D", "E")
STAGE_STEPS: Mapping[str, tuple[int, ...]] = {
    "R0": (0,),
    "R1": (1, 10),
    "R2": (36, 363),
    "R3": (3633, 10899, 21798),
    "R4": (43596, 87191, 174382),
}
STAGES: tuple[str, ...] = ("ASSETS", "R0", "R1", "R2", "R3", "R4")
DEFAULT_PROFILE_LIBRARY_SHA256 = (
    "58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe"
)
INITIAL_STATE_CLASS = "MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1"
ZERO_MODE = "PF_CONSERVED_Y_ZERO_MODE_V1"
ZERO_BACKEND = "HOST_NEWTON_BISECTION_V1"
EXPLICIT_CONTEXT = "SM_EXPLICIT_CONTEXT_N_V1"
REACTION_DISCRETIZATION = "SM_TANGENT_N_V1"
REL_TOL = 1.0e-10
FIELD_EQ_TOL = 1.0e-14


class AuditError(RuntimeError):
    """An input or observed runtime artifact violates this validation contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise AuditError(f"{label} must be a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AuditError(f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AuditError(f"{label} must be a nonempty string")
    return value


def _sha(value: Any, label: str) -> str:
    text = _text(value, label)
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise AuditError(f"{label} must be a lowercase SHA-256")
    return text


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise AuditError(f"{label} must be finite")
    return result


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1.0e-300)


def _path(value: Any, label: str) -> Path:
    return Path(_text(value, label))


def _read_meta(path: Path, label: str) -> Mapping[str, Any]:
    return _mapping(_read_json(path, label), label)


def _sidecar_checkpoint_bins(path: Path) -> dict[str, list[dict[str, float]]]:
    """Parse the exact sidecar bins after the same m^-4 -> m^-3 conversion as C++."""

    scalar: dict[str, str] = {}
    source_bins: dict[str, list[tuple[float, float, float, float]]] = {
        "GP": [], "beta_subgrid": []
    }
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw:
            raise AuditError(f"malformed staged auxiliary sidecar: {path}")
        key, value = raw.split("=", 1)
        if key == "gp_bin":
            source_bins["GP"].append(tuple(float(item) for item in value.split(",")))
        elif key == "beta_subgrid_bin":
            source_bins["beta_subgrid"].append(tuple(float(item) for item in value.split(",")))
        else:
            scalar[key] = value
    result: dict[str, list[dict[str, float]]] = {}
    for name, prefix in (("GP", "gp"), ("beta_subgrid", "beta_subgrid")):
        x_b = float(scalar[f"{prefix}_xB"])
        vm = float(scalar[f"{prefix}_Vm_m3_mol"])
        expected_count = int(scalar[f"{prefix}_bin_count"])
        if len(source_bins[name]) != expected_count:
            raise AuditError(f"staged sidecar bin count is inconsistent: {path}")
        bins: list[dict[str, float]] = []
        for lower, upper, density_per_m4, inventory in source_bins[name]:
            bins.append(
                {
                    "radius_lower_m": lower,
                    "radius_upper_m": upper,
                    "number_density_m3": density_per_m4 * (upper - lower),
                    "count": 0.0,
                    "xB": x_b,
                    "Vm_m3_mol": vm,
                    "inventory_mol": inventory,
                }
            )
        result[name] = bins
    return result


def _case_asset(index: Mapping[str, Any], case: str) -> dict[str, Any]:
    cases = _mapping(index.get("cases"), "asset index cases")
    raw_case = _mapping(cases.get(case), f"asset case {case}")
    raw_fields = _mapping(raw_case.get("raw_fields"), f"asset case {case} raw_fields")
    phi = _path(raw_fields.get("phi_path"), f"asset case {case} phi_path")
    xb = _path(raw_fields.get("xB_path"), f"asset case {case} xB_path")
    meta = _path(raw_fields.get("init_meta_path"), f"asset case {case} init_meta_path")
    for item, name in ((phi, "phi"), (xb, "xB"), (meta, "init_meta")):
        if not item.is_file():
            raise AuditError(f"asset case {case} missing {name}: {item}")
    if _sha(raw_fields.get("phi_sha256"), f"asset case {case} phi_sha256") != _sha256(phi):
        raise AuditError(f"asset case {case} phi raw SHA-256 mismatch")
    if _sha(raw_fields.get("xB_sha256"), f"asset case {case} xB_sha256") != _sha256(xb):
        raise AuditError(f"asset case {case} xB raw SHA-256 mismatch")
    raw_meta = _read_meta(meta, f"asset case {case} init meta")
    sidecar_required = raw_case.get("requires_auxiliary_sidecar") is True
    sidecar_info = _mapping(raw_case.get("auxiliary_sidecar"), f"asset case {case} auxiliary_sidecar")
    sidecar: Path | None = None
    identities: dict[str, str] = {}
    if sidecar_required:
        sidecar = _path(sidecar_info.get("path"), f"asset case {case} sidecar path")
        if not sidecar.is_file():
            raise AuditError(f"asset case {case} missing sidecar: {sidecar}")
        if _sha(sidecar_info.get("sha256"), f"asset case {case} sidecar SHA-256") != _sha256(sidecar):
            raise AuditError(f"asset case {case} sidecar SHA-256 mismatch")
        identities = {
            "validation_contract_hash": _sha(
                sidecar_info.get("validation_contract_hash"),
                f"asset case {case} sidecar contract hash",
            ),
            "source_handoff_hash": _sha(
                sidecar_info.get("source_handoff_hash"),
                f"asset case {case} sidecar source hash",
            ),
            "package_hash": _sha(
                sidecar_info.get("package_hash"),
                f"asset case {case} sidecar package hash",
            ),
            "fixture_hash": _sha(
                sidecar_info.get("fixture_hash"),
                f"asset case {case} sidecar fixture hash",
            ),
        }
        for key in ("source_handoff_hash", "package_hash", "fixture_hash"):
            if raw_meta.get(key) != identities[key]:
                raise AuditError(f"asset case {case} raw meta does not bind sidecar {key}")
        if raw_meta.get("auxiliary_sidecar_sha256") != _sha256(sidecar):
            raise AuditError(f"asset case {case} raw meta does not bind its sidecar bytes")
        expected_sidecar_bins = _sidecar_checkpoint_bins(sidecar)
    elif sidecar_info.get("present") is not False:
        raise AuditError(f"asset case {case} unexpectedly declares an auxiliary sidecar")
    else:
        expected_sidecar_bins = {"GP": [], "beta_subgrid": []}

    ledger = _mapping(raw_case.get("full_four_bucket_ledger"), f"asset case {case} four-bucket ledger")
    required_ledger = (
        "Q_B_total_mol",
        "Q_B_matrix_mol",
        "Q_B_GP_mol",
        "Q_B_beta_subgrid_mol",
        "Q_B_beta_resolved_fixed_mol",
    )
    parsed_ledger = {key: _number(ledger.get(key), f"asset case {case} {key}") for key in required_ledger}
    bucket_sum = sum(parsed_ledger[key] for key in required_ledger[1:])
    if _relative_error(bucket_sum, parsed_ledger["Q_B_total_mol"]) > REL_TOL:
        raise AuditError(f"asset case {case} four-bucket ledger does not close")
    forbidden = _mapping(raw_case.get("prohibited_dynamics"), f"asset case {case} prohibited_dynamics")
    for key in (
        "GP_release",
        "GP_to_beta_conversion",
        "beta_birth",
        "online_KWN_callback",
        "matrix_global_reset",
        "composition_clamp",
    ):
        if forbidden.get(key) is not False:
            raise AuditError(f"asset case {case} permits prohibited dynamics: {key}")
    return {
        "case": case,
        "case_id": _text(raw_case.get("case_id"), f"asset case {case} case_id"),
        "phi": phi,
        "xb": xb,
        "meta": meta,
        "raw_meta": raw_meta,
        "sidecar_required": sidecar_required,
        "sidecar": sidecar,
        "identities": identities,
        "expected_sidecar_bins": expected_sidecar_bins,
        "ledger": parsed_ledger,
        "exact_transfer_delta_Q_mol": raw_case.get("exact_transfer_delta_Q_mol"),
    }


def validate_assets(index_path: Path, *, profile_library_sha256: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return a checked asset index and normalized A--E case records."""

    index = _read_json(index_path, "CUDA A--E asset index")
    if index.get("schema_version") != ASSET_SCHEMA:
        raise AuditError("unexpected CUDA A--E asset-index schema")
    if index.get("status") != EXPECTED_ASSET_STATUS:
        raise AuditError("asset index does not claim a passed input-only preparation")
    if index.get("validation_only") is not True or index.get("historical_as_run_claim") is not False:
        raise AuditError("asset index scope is not validation-only/nonhistorical")
    contract = _mapping(index.get("contract"), "asset index contract")
    fixture = _mapping(index.get("fixture"), "asset index fixture")
    contract_hash = _sha(contract.get("hash"), "asset index contract hash")
    fixture_hash = _sha(fixture.get("hash"), "asset index fixture hash")
    submission = _mapping(index.get("submission_contract"), "asset index submission contract")
    if submission.get("only_allowed_grid") != [96, 96, 96]:
        raise AuditError("asset index is not restricted to the 96-cube")
    for key in (
        "GP_release",
        "GP_to_beta_conversion",
        "beta_birth",
        "online_KWN_callback",
        "matrix_global_reset",
    ):
        if str(submission.get(key, "")).upper() != "OFF":
            raise AuditError(f"asset submission contract does not disable {key}")
    if str(submission.get("composition_clamp", "")).upper() != "FORBIDDEN":
        raise AuditError("asset submission contract does not forbid composition clamps")
    if not re.fullmatch(r"[0-9a-f]{64}", profile_library_sha256):
        raise AuditError("profile-library hash must be a lowercase SHA-256")

    cases = {case: _case_asset(index, case) for case in CASES}
    for case, record in cases.items():
        meta = record["raw_meta"]
        if meta.get("validation_contract_hash") != contract_hash:
            raise AuditError(f"asset case {case} raw meta contract hash mismatch")
        if record["sidecar_required"]:
            identities = record["identities"]
            if identities["validation_contract_hash"] != contract_hash:
                raise AuditError(f"asset case {case} sidecar contract hash mismatch")
            if identities["fixture_hash"] != fixture_hash:
                raise AuditError(f"asset case {case} sidecar fixture hash mismatch")
    return index, cases


def emit_run_manifest(path: Path, index_path: Path, cases: Mapping[str, Mapping[str, Any]]) -> None:
    """Write a checked TSV consumed by the runner, without shell expansion."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "case",
                "phi_path",
                "xb_path",
                "init_meta_path",
                "sidecar_path_or_dash",
                "source_handoff_hash_or_dash",
                "package_hash_or_dash",
                "fixture_hash_or_dash",
                "asset_index",
            )
        )
        for case in CASES:
            record = cases[case]
            identity = record["identities"]
            writer.writerow(
                (
                    case,
                    record["phi"],
                    record["xb"],
                    record["meta"],
                    record["sidecar"] if record["sidecar"] is not None else "-",
                    identity.get("source_handoff_hash", "-"),
                    identity.get("package_hash", "-"),
                    identity.get("fixture_hash", "-"),
                    index_path,
                )
            )


# Native C++ uses the ordinary x86_64 ABI; every field is explicitly aligned
# here and the on-disk ``header_bytes`` is required to equal this value.  A
# different build layout is therefore a hard incompatibility, not a guessed
# parse.  The runner targets the documented gpu_uvip x86_64 environment.
HEADER_STRUCT = struct.Struct(
    "<8sIIQQQiiiiIIII"  # prefix, counts, grid, elastic/aux flags
    "dddddd"  # dt, temperature, zero-mode scalars
    "QQQQQQ"  # iteration/fingerprints/elastic counters
    "dQQdd"  # elastic residual, bin counts, auxiliary inventories
    + "64s" * 9
    + "96s" * 8
    + "Q"
)
BIN_STRUCT = struct.Struct("<dddQddd")
if HEADER_STRUCT.size != 1560 or BIN_STRUCT.size != 56:
    raise RuntimeError("internal V6 checkpoint layout assertion failed")


def _c_text(value: bytes) -> str:
    return value.split(b"\0", 1)[0].decode("ascii", errors="strict")


def _header_from(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = handle.read(HEADER_STRUCT.size)
    if len(data) != HEADER_STRUCT.size:
        raise AuditError(f"truncated V6 checkpoint header: {path}")
    value = list(HEADER_STRUCT.unpack(data))
    index = 0

    def take() -> Any:
        nonlocal index
        result = value[index]
        index += 1
        return result

    header: dict[str, Any] = {
        "magic": take(),
        "version": take(),
        "header_bytes": take(),
        "element_count": take(),
        "k_element_count": take(),
        "accepted_step": take(),
        "nx": take(),
        "ny": take(),
        "nz": take(),
        "elastic_state_present": take(),
        "aux_state_present": take(),
        "aux_schema_version": take(),
        "aux_frozen": take(),
        "reserved": take(),
        "dt_code": take(),
        "temperature_K": take(),
        "target_mass_code": take(),
        "last_lambda": take(),
        "last_residual_code": take(),
        "last_derivative_code": take(),
        "last_iterations": take(),
        "accepted_zero_mode_steps": take(),
        "parameter_fingerprint": take(),
        "elastic_solver_fingerprint": take(),
        "elastic_source_field_step": take(),
        "elastic_last_iterations": take(),
        "elastic_last_relative_residual": take(),
        "gp_bin_count": take(),
        "beta_subgrid_bin_count": take(),
        "Q_B_GP_mol": take(),
        "Q_B_beta_subgrid_mol": take(),
    }
    selector_names = (
        "zero_mode",
        "backend",
        "composition_mode",
        "y_update_mode",
        "explicit_context",
        "reaction_discretization",
        "elastic_solver_mode",
        "auxiliary_state",
        "auxiliary_units",
    )
    for name in selector_names:
        header[name] = _c_text(take())
    identity_names = (
        "initial_state_class",
        "fixture_manifest_sha256",
        "profile_library_manifest_sha256",
        "validation_contract_hash",
        "source_handoff_hash",
        "package_handoff_hash",
        "gp_population_provenance",
        "beta_subgrid_population_provenance",
    )
    for name in identity_names:
        header[name] = _c_text(take())
    header["payload_checksum"] = take()
    if index != len(value):
        raise RuntimeError("internal V6 header indexing assertion failed")
    return header


def _checkpoint_payload_layout(header: Mapping[str, Any]) -> dict[str, int]:
    count = int(header["element_count"])
    k_count = int(header["k_element_count"])
    offset = int(header["header_bytes"])
    fields: dict[str, int] = {}
    for name in ("phi", "Y", "xB", "dY_dt_prev"):
        fields[name] = offset
        offset += 8 * count
    if int(header["elastic_state_present"]):
        fields["elastic"] = offset
        offset += 24 * k_count
    fields["gp_bins"] = offset
    offset += BIN_STRUCT.size * int(header["gp_bin_count"])
    fields["beta_subgrid_bins"] = offset
    offset += BIN_STRUCT.size * int(header["beta_subgrid_bin_count"])
    fields["file_size"] = offset
    return fields


def _array_at(path: Path, offset: int, count: int) -> np.memmap:
    return np.memmap(path, mode="r", dtype="<f8", offset=offset, shape=(count,))


def _bin_inventory(path: Path, offset: int, count: int) -> tuple[float, list[dict[str, float]]]:
    if count == 0:
        return 0.0, []
    bins: list[dict[str, float]] = []
    with path.open("rb") as handle:
        handle.seek(offset)
        for _ in range(count):
            raw = handle.read(BIN_STRUCT.size)
            if len(raw) != BIN_STRUCT.size:
                raise AuditError(f"truncated auxiliary bin payload: {path}")
            lower, upper, density, number, xb, vm, inventory = BIN_STRUCT.unpack(raw)
            if not all(math.isfinite(float(item)) for item in (lower, upper, density, xb, vm, inventory)):
                raise AuditError(f"non-finite auxiliary bin payload: {path}")
            bins.append(
                {
                    "radius_lower_m": lower,
                    "radius_upper_m": upper,
                    "number_density_m3": density,
                    "count": float(number),
                    "xB": xb,
                    "Vm_m3_mol": vm,
                    "inventory_mol": inventory,
                }
            )
    return float(sum(item["inventory_mol"] for item in bins)), bins


def _checkpoint_observation(
    path: Path,
    *,
    case: Mapping[str, Any],
    contract: Any,
    contract_hash: str,
    profile_library_sha256: str,
    expected_step: int,
) -> dict[str, Any]:
    header = _header_from(path)
    if header["magic"] != b"PFZMCHK6" or header["version"] != 6 or header["header_bytes"] != HEADER_STRUCT.size:
        raise AuditError(f"checkpoint is not the expected V6 layout: {path}")
    if (header["nx"], header["ny"], header["nz"]) != (96, 96, 96):
        raise AuditError(f"checkpoint grid is not 96^3: {path}")
    count = 96 * 96 * 96
    if header["element_count"] != count:
        raise AuditError(f"checkpoint element count is not 96^3: {path}")
    if header["accepted_step"] != expected_step:
        raise AuditError(f"checkpoint accepted step does not match its staged path: {path}")
    if not math.isclose(float(header["dt_code"]), float(contract.canonical.value("numerics.dt_code")), rel_tol=0.0, abs_tol=1.0e-15):
        raise AuditError(f"checkpoint dt differs from validation contract: {path}")
    for key, expected in (
        ("zero_mode", ZERO_MODE),
        ("backend", ZERO_BACKEND),
        ("composition_mode", "legacy"),
        ("y_update_mode", "lagged_rhs"),
        ("explicit_context", EXPLICIT_CONTEXT),
        ("reaction_discretization", REACTION_DISCRETIZATION),
        ("initial_state_class", INITIAL_STATE_CLASS),
        ("fixture_manifest_sha256", _text(case["raw_meta"].get("fixture_hash", case.get("fixture_hash", "")) if case["sidecar_required"] else case.get("fixture_hash", ""), "fixture hash")),
        ("profile_library_manifest_sha256", profile_library_sha256),
        ("validation_contract_hash", contract_hash),
    ):
        # Case A meta intentionally has no package identity, but the fixture
        # identity is still supplied by the checked top-level asset index below.
        if key == "fixture_manifest_sha256" and not expected:
            continue
        if header[key] != expected:
            raise AuditError(f"checkpoint {key} mismatch: {path}")
    # The caller installs this immutable top-level fixture hash so Case A is
    # checked too, without pretending it had a sidecar package.
    expected_fixture = _text(case["fixture_hash"], "case fixture hash")
    if header["fixture_manifest_sha256"] != expected_fixture:
        raise AuditError(f"checkpoint fixture hash mismatch: {path}")
    sidecar_required = bool(case["sidecar_required"])
    if sidecar_required:
        identities = _mapping(case["identities"], "case identities")
        if (
            header["aux_state_present"] != 1
            or header["aux_frozen"] != 1
            or header["auxiliary_state"] != "AUXILIARY_POPULATION_STORAGE_ONLY_V1"
            or header["source_handoff_hash"] != identities["source_handoff_hash"]
            or header["package_handoff_hash"] != identities["package_hash"]
        ):
            raise AuditError(f"checkpoint auxiliary provenance mismatch: {path}")
    elif (
        header["aux_state_present"] != 0
        or header["gp_bin_count"] != 0
        or header["beta_subgrid_bin_count"] != 0
        or header["Q_B_GP_mol"] != 0.0
        or header["Q_B_beta_subgrid_mol"] != 0.0
    ):
        raise AuditError(f"Case A checkpoint unexpectedly carries auxiliary state: {path}")
    layout = _checkpoint_payload_layout(header)
    actual_size = path.stat().st_size
    if actual_size != layout["file_size"]:
        raise AuditError(f"checkpoint payload size is inconsistent/truncated: {path}")
    fields = {name: _array_at(path, layout[name], count) for name in ("phi", "Y", "xB", "dY_dt_prev")}
    for name, values in fields.items():
        if not np.all(np.isfinite(values)):
            raise AuditError(f"checkpoint {name} contains NaN/Inf: {path}")
    phi = fields["phi"]
    xb = fields["xB"]
    if float(np.min(phi)) < -FIELD_EQ_TOL or float(np.max(phi)) > 1.0 + FIELD_EQ_TOL:
        raise AuditError(f"checkpoint phi leaves [0,1]: {path}")
    if float(np.min(xb)) < -FIELD_EQ_TOL or float(np.max(xb)) > 1.0 + FIELD_EQ_TOL:
        raise AuditError(f"checkpoint xB leaves [0,1]: {path}")
    h = h_of_phi(phi)
    alpha = 1.0 - h
    q_matrix = field_matrix_inventory_mol(alpha, xb, contract.dx_m**3, contract.vm_alpha_m3_mol)
    q_resolved = field_resolved_inventory_mol(h, contract.v_B, contract.dx_m**3, contract.vm_beta_m3_mol)
    q_gp_bins, gp_bins = _bin_inventory(path, layout["gp_bins"], int(header["gp_bin_count"]))
    q_subgrid_bins, subgrid_bins = _bin_inventory(path, layout["beta_subgrid_bins"], int(header["beta_subgrid_bin_count"]))
    observed_populations = {"GP": gp_bins, "beta_subgrid": subgrid_bins}
    expected_populations = _mapping(case["expected_sidecar_bins"], "expected staged sidecar bins")
    for population_name in ("GP", "beta_subgrid"):
        expected_bins = expected_populations[population_name]
        observed_bins = observed_populations[population_name]
        if len(observed_bins) != len(expected_bins):
            raise AuditError(
                f"checkpoint {population_name} bin count differs from staged sidecar: {path}"
            )
        for index, (observed, expected) in enumerate(zip(observed_bins, expected_bins)):
            for key in (
                "radius_lower_m", "radius_upper_m", "number_density_m3",
                "count", "xB", "Vm_m3_mol", "inventory_mol",
            ):
                actual_value = float(observed[key])
                expected_value = float(expected[key])
                if _relative_error(actual_value, expected_value) > REL_TOL:
                    raise AuditError(
                        f"checkpoint {population_name} bin {index} {key} differs from staged sidecar: {path}"
                    )
    if _relative_error(q_gp_bins, float(header["Q_B_GP_mol"])) > REL_TOL:
        raise AuditError(f"checkpoint GP bin ledger does not match its header: {path}")
    if _relative_error(q_subgrid_bins, float(header["Q_B_beta_subgrid_mol"])) > REL_TOL:
        raise AuditError(f"checkpoint subgrid bin ledger does not match its header: {path}")
    q_total = q_matrix + q_resolved + q_gp_bins + q_subgrid_bins
    expected_total = float(case["ledger"]["Q_B_total_mol"])
    mass_relative = _relative_error(q_total, expected_total)
    if mass_relative > REL_TOL:
        raise AuditError(f"checkpoint four-bucket physical inventory drifted: {path}")
    for name, observed, expected in (
        ("GP", q_gp_bins, float(case["ledger"]["Q_B_GP_mol"])),
        ("beta-subgrid", q_subgrid_bins, float(case["ledger"]["Q_B_beta_subgrid_mol"])),
    ):
        if _relative_error(observed, expected) > REL_TOL:
            raise AuditError(
                f"checkpoint frozen {name} inventory differs from its case input: {path}"
            )
    initial_zero = bool(int(header["reserved"]) & 1)
    if expected_step == 0:
        if not initial_zero or header["accepted_zero_mode_steps"] != 0 or header["elastic_state_present"] != 0:
            raise AuditError(f"R0 checkpoint lacks the required zero-step semantic state: {path}")
    elif initial_zero:
        raise AuditError(f"post-R0 checkpoint incorrectly remains an R0 snapshot: {path}")
    observation = {
        "checkpoint": str(path),
        "checkpoint_sha256": _sha256(path),
        "accepted_step": int(header["accepted_step"]),
        "initial_zero_step": initial_zero,
        "header": {
            key: header[key]
            for key in (
                "version", "header_bytes", "element_count", "k_element_count", "dt_code",
                "temperature_K", "target_mass_code", "last_lambda", "last_residual_code",
                "last_derivative_code", "last_iterations", "accepted_zero_mode_steps",
                "parameter_fingerprint", "elastic_solver_fingerprint", "elastic_state_present",
                "elastic_source_field_step", "elastic_last_iterations", "elastic_last_relative_residual",
                "aux_state_present", "aux_schema_version", "aux_frozen", "gp_bin_count",
                "beta_subgrid_bin_count", "zero_mode", "backend", "composition_mode",
                "y_update_mode", "explicit_context", "reaction_discretization", "elastic_solver_mode",
                "initial_state_class", "fixture_manifest_sha256", "profile_library_manifest_sha256",
                "validation_contract_hash", "source_handoff_hash", "package_handoff_hash",
            )
        },
        "inventory_mol": {
            "Q_B_total_mol": q_total,
            "Q_B_matrix_mol": q_matrix,
            "Q_B_GP_mol": q_gp_bins,
            "Q_B_beta_subgrid_mol": q_subgrid_bins,
            "Q_B_beta_resolved_fixed_mol": q_resolved,
            "relative_error_vs_case_t0": mass_relative,
        },
        "field_summary": {
            "phi_min": float(np.min(phi)),
            "phi_max": float(np.max(phi)),
            "xB_min": float(np.min(xb)),
            "xB_max": float(np.max(xb)),
            "phi_sha256": hashlib.sha256(np.ascontiguousarray(phi, dtype="<f8").tobytes()).hexdigest(),
            "xB_sha256": hashlib.sha256(np.ascontiguousarray(xb, dtype="<f8").tobytes()).hexdigest(),
        },
        "auxiliary_bins": {"GP": gp_bins, "beta_subgrid": subgrid_bins},
        "diagnostics_not_captured": [
            "component_identity_lineage",
            "specific_interfacial_area",
            "chemical_potential_field",
            "full_free_energy_decomposition",
        ],
    }
    return observation


def _checkpoint_path(run_root: Path, case: str, step: int) -> Path:
    return run_root / "checkpoints" / case / f"step_{step}.pfzck"


def _stdout_path(run_root: Path, case: str, step: int) -> Path:
    return run_root / "logs" / case / f"step_{step}.stdout.log"


def _stderr_path(run_root: Path, case: str, step: int) -> Path:
    return run_root / "logs" / case / f"step_{step}.stderr.log"


def _log_observation(path: Path, *, step: int) -> dict[str, Any]:
    if not path.is_file():
        raise AuditError(f"missing stdout log for checkpoint step {step}: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    lower = text.lower()
    if "[fatal]" in lower or "pf_zero_mode_final_audit status=fail" in lower:
        raise AuditError(f"CUDA runtime log reports a fatal/failing zero-mode audit: {path}")
    if re.search(r"\b(?:nan|inf)\b", lower):
        raise AuditError(f"CUDA runtime log reports NaN/Inf: {path}")
    required = "PF_ZERO_MODE_R0_INITIAL_CHECKPOINT_PASS" if step == 0 else "PF_ZERO_MODE_FINAL_AUDIT status=PASS"
    if required not in text:
        raise AuditError(f"CUDA runtime log lacks its required completion marker: {path}")
    for forbidden in (
        "gp_release",
        "gp_to_beta",
        "scheduled_nucleation_test : enabled",
        "online_kwn",
        "composition_clamp",
    ):
        if forbidden in lower:
            raise AuditError(f"CUDA runtime log contains prohibited mode marker '{forbidden}': {path}")
    return {"stdout": str(path), "stderr": str(_stderr_path(path.parents[3], path.parent.name, step)) if False else None}


def _load_raw_field(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype="<f8")
    if values.size != 96 * 96 * 96:
        raise AuditError(f"raw field does not contain 96^3 float64 values: {path}")
    return values


def _compare_field_checkpoints(
    left_path: Path, right_path: Path, *, label: str
) -> dict[str, Any]:
    left_header = _header_from(left_path)
    right_header = _header_from(right_path)
    if left_header["element_count"] != right_header["element_count"]:
        raise AuditError(f"restart comparison count mismatch: {label}")
    count = int(left_header["element_count"])
    left_layout = _checkpoint_payload_layout(left_header)
    right_layout = _checkpoint_payload_layout(right_header)
    summary: dict[str, Any] = {"label": label, "left": str(left_path), "right": str(right_path), "fields": {}}
    for field in ("phi", "Y", "xB", "dY_dt_prev"):
        left = _array_at(left_path, left_layout[field], count)
        right = _array_at(right_path, right_layout[field], count)
        maximum = float(np.max(np.abs(left - right)))
        summary["fields"][field] = {"max_abs_difference": maximum, "within_1e-14": maximum <= FIELD_EQ_TOL}
        if maximum > FIELD_EQ_TOL:
            raise AuditError(f"restart comparison exceeds 1e-14 for {field}: {label}")
    if left_header["aux_state_present"] != right_header["aux_state_present"]:
        raise AuditError(f"restart auxiliary presence mismatch: {label}")
    for key in ("Q_B_GP_mol", "Q_B_beta_subgrid_mol", "source_handoff_hash", "package_handoff_hash"):
        if left_header[key] != right_header[key]:
            raise AuditError(f"restart auxiliary state mismatch for {key}: {label}")
    return summary


def _available_steps(run_root: Path, case: str) -> list[int]:
    directory = run_root / "checkpoints" / case
    if not directory.is_dir():
        return []
    result: list[int] = []
    for candidate in directory.glob("step_*.pfzck"):
        match = re.fullmatch(r"step_(\d+)\.pfzck", candidate.name)
        if match:
            result.append(int(match.group(1)))
    return sorted(result)


def _compare_local_case_fields(
    run_root: Path, step: int, left_case: str, right_case: str
) -> dict[str, Any]:
    """Compare observed local PF state while deliberately excluding auxiliary PSDs."""

    left_path = _checkpoint_path(run_root, left_case, step)
    right_path = _checkpoint_path(run_root, right_case, step)
    left_header = _header_from(left_path)
    right_header = _header_from(right_path)
    if left_header["element_count"] != right_header["element_count"]:
        raise AuditError(f"A--C local-field count mismatch at step {step}")
    count = int(left_header["element_count"])
    left_layout = _checkpoint_payload_layout(left_header)
    right_layout = _checkpoint_payload_layout(right_header)
    fields: dict[str, Any] = {}
    for name in ("phi", "xB", "Y", "dY_dt_prev"):
        lhs = _array_at(left_path, left_layout[name], count)
        rhs = _array_at(right_path, right_layout[name], count)
        maximum = float(np.max(np.abs(lhs - rhs)))
        fields[name] = {"max_abs_difference": maximum, "within_1e-14": maximum <= FIELD_EQ_TOL}
        if maximum > FIELD_EQ_TOL:
            raise AuditError(
                f"{left_case}/{right_case} local {name} differs by more than 1e-14 at step {step}"
            )
    return {
        "step": step,
        "left_case": left_case,
        "right_case": right_case,
        "fields": fields,
    }


def audit_runtime(
    *,
    index_path: Path,
    run_root: Path,
    out_dir: Path,
    contract_path: Path,
    profile_library_sha256: str,
    required_stage: str | None,
) -> tuple[dict[str, Any], int]:
    index, cases = validate_assets(index_path, profile_library_sha256=profile_library_sha256)
    fixture_hash = _sha(_mapping(index["fixture"], "fixture").get("hash"), "fixture hash")
    for record in cases.values():
        record["fixture_hash"] = fixture_hash
    contract = load_validation_contract(contract_path)
    index_contract = _sha(_mapping(index["contract"], "contract").get("hash"), "contract hash")
    if contract.contract_hash != index_contract:
        raise AuditError("runtime contract path does not match the frozen asset index")
    if required_stage is not None and required_stage not in STAGES:
        raise AuditError(f"unsupported required stage: {required_stage}")
    maximum_stage_index = STAGES.index(required_stage) if required_stage else len(STAGES) - 1
    needed_steps: set[int] = set()
    for stage in STAGES[1 : maximum_stage_index + 1]:
        needed_steps.update(STAGE_STEPS[stage])

    observations: list[dict[str, Any]] = []
    runtime_failures: list[str] = []
    per_case_available: dict[str, list[int]] = {}
    for case in CASES:
        available = _available_steps(run_root, case)
        per_case_available[case] = available
        for step in available:
            try:
                log = _log_observation(_stdout_path(run_root, case, step), step=step)
                checkpoint = _checkpoint_observation(
                    _checkpoint_path(run_root, case, step),
                    case=cases[case],
                    contract=contract,
                    contract_hash=contract.contract_hash,
                    profile_library_sha256=profile_library_sha256,
                    expected_step=step,
                )
                checkpoint["log"] = log
                checkpoint["case"] = case
                observations.append(checkpoint)
            except AuditError as error:
                runtime_failures.append(str(error))
    missing: list[str] = []
    for case in CASES:
        for step in sorted(needed_steps):
            if step not in per_case_available[case]:
                missing.append(f"{case}: step {step}")

    # R0 must still bind byte-identically materialized local fields.  This is
    # an observed input-state comparison, not a claim about later dynamics.
    r0_records = {(record["case"], record["accepted_step"]): record for record in observations}
    raw_comparisons: list[dict[str, Any]] = []
    for case in CASES:
        r0 = r0_records.get((case, 0))
        if r0 is None:
            continue
        header = _header_from(_checkpoint_path(run_root, case, 0))
        layout = _checkpoint_payload_layout(header)
        phi = _array_at(_checkpoint_path(run_root, case, 0), layout["phi"], 96**3)
        xb = _array_at(_checkpoint_path(run_root, case, 0), layout["xB"], 96**3)
        phi_error = float(np.max(np.abs(phi - _load_raw_field(cases[case]["phi"]))))
        xb_error = float(np.max(np.abs(xb - _load_raw_field(cases[case]["xb"]))))
        raw_comparisons.append({"case": case, "phi_max_abs_difference": phi_error, "xB_max_abs_difference": xb_error})
        if phi_error > FIELD_EQ_TOL or xb_error > FIELD_EQ_TOL:
            runtime_failures.append(f"R0 raw field mismatch exceeds 1e-14 for case {case}")

    local_control_comparisons: list[dict[str, Any]] = []
    common_abc_steps = set(per_case_available["A"])
    common_abc_steps.intersection_update(per_case_available["B"])
    common_abc_steps.intersection_update(per_case_available["C"])
    for step in sorted(common_abc_steps):
        for left_case, right_case in (("A", "B"), ("B", "C")):
            try:
                local_control_comparisons.append(
                    _compare_local_case_fields(run_root, step, left_case, right_case)
                )
            except AuditError as error:
                runtime_failures.append(str(error))

    d_r0_control: dict[str, Any] | None = None
    a0 = r0_records.get(("A", 0))
    d0 = r0_records.get(("D", 0))
    if a0 is not None and d0 is not None:
        try:
            a0_path = _checkpoint_path(run_root, "A", 0)
            d0_path = _checkpoint_path(run_root, "D", 0)
            a_header = _header_from(a0_path)
            d_header = _header_from(d0_path)
            a_phi = _array_at(a0_path, _checkpoint_payload_layout(a_header)["phi"], 96**3)
            d_phi = _array_at(d0_path, _checkpoint_payload_layout(d_header)["phi"], 96**3)
            phi_difference = float(np.max(np.abs(a_phi - d_phi)))
            a_inventory = _mapping(a0["inventory_mol"], "A R0 inventory")
            d_inventory = _mapping(d0["inventory_mol"], "D R0 inventory")
            expected_transfer = _number(
                cases["D"]["exact_transfer_delta_Q_mol"],
                "Case D exact transfer delta",
            )
            matrix_delta = float(a_inventory["Q_B_matrix_mol"]) - float(d_inventory["Q_B_matrix_mol"])
            gp_delta = float(d_inventory["Q_B_GP_mol"]) - float(a_inventory["Q_B_GP_mol"])
            resolved_difference = float(d_inventory["Q_B_beta_resolved_fixed_mol"]) - float(a_inventory["Q_B_beta_resolved_fixed_mol"])
            d_r0_control = {
                "phi_max_abs_difference": phi_difference,
                "resolved_inventory_difference_mol": resolved_difference,
                "matrix_inventory_delta_mol": matrix_delta,
                "GP_inventory_delta_mol": gp_delta,
                "expected_exact_transfer_delta_mol": expected_transfer,
            }
            if (
                phi_difference > FIELD_EQ_TOL
                or abs(resolved_difference) > REL_TOL * max(
                    abs(float(a_inventory["Q_B_beta_resolved_fixed_mol"])),
                    abs(float(d_inventory["Q_B_beta_resolved_fixed_mol"])),
                    1.0e-300,
                )
                or _relative_error(matrix_delta, expected_transfer) > REL_TOL
                or _relative_error(gp_delta, expected_transfer) > REL_TOL
                or _relative_error(matrix_delta, gp_delta) > REL_TOL
            ):
                raise AuditError("R0 Case D matrix-to-GP control does not preserve its declared transfer")
        except AuditError as error:
            runtime_failures.append(str(error))

    matrix_inventory_pulse_summary: list[dict[str, Any]] = []
    initial_matrix = {
        case: float(_mapping(record["inventory_mol"], "R0 inventory")["Q_B_matrix_mol"])
        for (case, step), record in r0_records.items() if step == 0
    }
    for record in observations:
        case = str(record["case"])
        if case not in initial_matrix:
            continue
        inventory = _mapping(record["inventory_mol"], "checkpoint inventory")
        matrix_inventory_pulse_summary.append(
            {
                "case": case,
                "step": record["accepted_step"],
                "Q_B_matrix_mol": inventory["Q_B_matrix_mol"],
                "delta_from_R0_mol": float(inventory["Q_B_matrix_mol"]) - initial_matrix[case],
            }
        )
    # The requested pulse is an observed R0--R2 diagnostic only.  Do not
    # extend it to a 48 h interpretation when R3/R4 happen to be present.
    pulse_window = [row for row in matrix_inventory_pulse_summary if int(row["step"]) <= 363]
    pulse_by_case: dict[str, dict[str, Any]] = {}
    seconds_per_step = 0.99092609534318414
    for case in CASES:
        rows = [row for row in pulse_window if row["case"] == case]
        if not rows:
            continue
        peak = max(rows, key=lambda row: abs(float(row["delta_from_R0_mol"])))
        initial = initial_matrix[case]
        pulse_by_case[case] = {
            "window": "OBSERVED_R0_TO_R2_ONLY",
            "own_R0_matrix_inventory_mol": initial,
            "peak_minus_own_R0_mol": peak["delta_from_R0_mol"],
            "peak_abs_minus_own_R0_mol": abs(float(peak["delta_from_R0_mol"])),
            "peak_step": peak["step"],
            "peak_time_s_from_R0": float(peak["step"]) * seconds_per_step,
            "normalized_peak_minus_own_R0": float(peak["delta_from_R0_mol"]) / max(abs(initial), 1.0e-300),
        }
    a_b_pulse_comparison: dict[str, Any] | str
    if "A" in pulse_by_case and "B" in pulse_by_case:
        a_peak = pulse_by_case["A"]
        b_peak = pulse_by_case["B"]
        a_b_pulse_comparison = {
            "window": "OBSERVED_R0_TO_R2_ONLY",
            "A_minus_B_peak_delta_mol": float(a_peak["peak_minus_own_R0_mol"]) - float(b_peak["peak_minus_own_R0_mol"]),
            "A_peak_step": a_peak["peak_step"],
            "B_peak_step": b_peak["peak_step"],
            "same_peak_step": a_peak["peak_step"] == b_peak["peak_step"],
        }
    else:
        a_b_pulse_comparison = "NOT_CAPTURED_UNTIL_A_AND_B_HAVE_OBSERVED_R0_TO_R2_CHECKPOINTS"

    restart_comparisons: list[dict[str, Any]] = []
    qualification_pairs = (
        ("A", "restart_qualification/A/continuous_6h.pfzck", "restart_qualification/A/restart_6h.pfzck"),
        ("B", "restart_qualification/B/continuous_6h.pfzck", "restart_qualification/B/restart_6h.pfzck"),
        ("E", "restart_qualification/E/continuous_6h.pfzck", "restart_qualification/E/restart_6h.pfzck"),
    )
    if maximum_stage_index >= STAGES.index("R3"):
        for case, lhs, rhs in qualification_pairs:
            left = run_root / lhs
            right = run_root / rhs
            if not left.is_file() or not right.is_file():
                missing.append(f"{case}: 6 h continuous-vs-restart qualification")
                continue
            try:
                restart_comparisons.append(_compare_field_checkpoints(left, right, label=f"{case} 6h continuous/restart"))
            except AuditError as error:
                runtime_failures.append(str(error))
    if maximum_stage_index >= STAGES.index("R4"):
        left = run_root / "restart_qualification/E/continuous_48h.pfzck"
        right = run_root / "restart_qualification/E/restart_48h.pfzck"
        if not left.is_file() or not right.is_file():
            missing.append("E: 48 h continuous-vs-restart qualification")
        else:
            try:
                restart_comparisons.append(_compare_field_checkpoints(left, right, label="E 48h continuous/restart"))
            except AuditError as error:
                runtime_failures.append(str(error))

    if runtime_failures:
        status = "FAIL_CUDA_AE_RUNTIME_AUDIT"
        exit_code = 1
    elif missing:
        status = "INCOMPLETE_CUDA_AE_RUNTIME_AUDIT"
        exit_code = 1 if required_stage else 0
    elif required_stage in ("R3", "R4"):
        status = f"PASS_{required_stage}_CUDA_AE_RUNTIME_GATE_DIAGNOSTICS_PENDING"
        exit_code = 0
    elif required_stage in ("R0", "R1", "R2"):
        status = f"PASS_{required_stage}_CUDA_AE_RUNTIME_GATE"
        exit_code = 0
    elif required_stage == "ASSETS":
        status = "PASS_CUDA_AE_ASSET_GATE"
        exit_code = 0
    else:
        status = "PARTIAL_CUDA_AE_RUNTIME_AUDIT"
        exit_code = 0

    payload: dict[str, Any] = {
        "schema_version": "PF_CUDA_AE_RUNTIME_AUDIT_V1",
        "status": status,
        "validation_only": True,
        "historical_as_run_claim": False,
        "asset_index": str(index_path),
        "asset_index_sha256": _sha256(index_path),
        "run_root": str(run_root),
        "required_stage": required_stage,
        "contract_hash": contract.contract_hash,
        "fixture_hash": fixture_hash,
        "profile_library_manifest_sha256": profile_library_sha256,
        "observed_checkpoints": observations,
        "available_steps_by_case": per_case_available,
        "missing_required_artifacts": missing,
        "runtime_failures": runtime_failures,
        "r0_raw_field_comparisons": raw_comparisons,
        "A_B_C_local_field_comparisons": local_control_comparisons,
        "D_vs_A_R0_matrix_to_GP_control": d_r0_control,
        "matrix_inventory_pulse_summary": matrix_inventory_pulse_summary,
        "matrix_inventory_pulse_R0_to_R2": {
            "seconds_per_step": seconds_per_step,
            "per_case": pulse_by_case,
            "A_vs_B": a_b_pulse_comparison,
        },
        "restart_comparisons": restart_comparisons,
        "not_captured": {
            "component_identity_lineage": "NOT_CAPTURED_BY_CURRENT_CUDA_AE_RUNNER",
            "specific_interfacial_area": "NOT_CAPTURED_BY_CURRENT_CUDA_AE_RUNNER",
            "chemical_potential_field": "NOT_CAPTURED_BY_CURRENT_CUDA_AE_RUNNER",
            "full_free_energy_decomposition": "NOT_CAPTURED_BY_CURRENT_CUDA_AE_RUNNER",
        },
    }
    _write_json(out_dir / "cuda_ae_runtime_audit.json", payload)
    _write_csv_summaries(out_dir, observations, restart_comparisons)
    return payload, exit_code


def _write_csv_summaries(out_dir: Path, observations: Sequence[Mapping[str, Any]], restart_comparisons: Sequence[Mapping[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = out_dir / "cuda_ae_inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("case", "step", "Q_B_total_mol", "Q_B_matrix_mol", "Q_B_GP_mol", "Q_B_beta_subgrid_mol", "Q_B_beta_resolved_fixed_mol", "relative_error_vs_case_t0"))
        for record in observations:
            inventory = _mapping(record["inventory_mol"], "observation inventory")
            writer.writerow((record["case"], record["accepted_step"], *(inventory[name] for name in ("Q_B_total_mol", "Q_B_matrix_mol", "Q_B_GP_mol", "Q_B_beta_subgrid_mol", "Q_B_beta_resolved_fixed_mol", "relative_error_vs_case_t0"))))
    restart_path = out_dir / "cuda_ae_restart_comparison.csv"
    with restart_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("label", "field", "max_abs_difference", "within_1e-14"))
        for record in restart_comparisons:
            for field, value in _mapping(record["fields"], "restart fields").items():
                detail = _mapping(value, "restart field detail")
                writer.writerow((record["label"], field, detail["max_abs_difference"], detail["within_1e-14"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-index", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--contract", type=Path, default=ROOT / "contracts" / "pf_kwn_validation_contract_v1.json")
    parser.add_argument("--profile-library-sha256", default=DEFAULT_PROFILE_LIBRARY_SHA256)
    parser.add_argument("--require-stage", choices=STAGES)
    parser.add_argument("--emit-run-manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        index, cases = validate_assets(args.asset_index, profile_library_sha256=args.profile_library_sha256)
        if args.emit_run_manifest:
            emit_run_manifest(args.emit_run_manifest, args.asset_index, cases)
        if args.run_root is None:
            if args.out_dir is not None or args.require_stage is not None:
                raise AuditError("--run-root is required for a runtime audit")
            print(json.dumps({"status": "PASS_CUDA_AE_ASSET_GATE", "asset_index": str(args.asset_index), "cases": list(index["cases"])}))
            return 0
        out_dir = args.out_dir or args.run_root / "compact_audit"
        payload, exit_code = audit_runtime(
            index_path=args.asset_index,
            run_root=args.run_root,
            out_dir=out_dir,
            contract_path=args.contract,
            profile_library_sha256=args.profile_library_sha256,
            required_stage=args.require_stage,
        )
    except AuditError as error:
        print(f"cuda A--E runtime audit failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": payload["status"], "audit": str(out_dir / "cuda_ae_runtime_audit.json")}, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
