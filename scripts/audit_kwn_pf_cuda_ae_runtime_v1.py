#!/usr/bin/env python3
"""Audit the controlled 96-cube CUDA A--E validation run.

The reader is deliberately independent of ``main_cuda``.  It validates the
frozen input index and V6 checkpoints, then derives compact accepted-state
diagnostics from the checkpoint fields plus the CUDA accepted-field mechanics
bundle.  R0's bundle is an online synchronized reference; later bundles are
deterministic mechanics replays from the exact saved checkpoint and explicitly
prove that time, phi, and xB were not advanced.  The resulting chemical
potential, energy, and periodic-component values are therefore labelled as
checkpoint/replay-derived CUDA accepted-state diagnostics rather than invented
or solver-native scalar output.

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
# The CUDA PF state is stored after the solver's established phase-field
# representation projection, not after a separate physical-field rewrite.
# PROJECT_CORE_MEMORY.md §13.11 freezes this as the cross-language analysis
# contract.  Physical phase-storage evaluations use clamp01(phi), exactly as
# the CUDA kernels do; xB remains subject to its strict physical [0, 1]
# bound.  This is distinct from (and must not enable) a composition clamp.
PHI_REPRESENTATION_LOWER = -1.0e-6
PHI_REPRESENTATION_UPPER = 1.0 + 1.0e-6
DIAGNOSTIC_PROVENANCE = "CHECKPOINT_REPLAY_DERIVED_CUDA_ACCEPTED_STATE_V1"
MECHANICS_REPLAY_SCHEMA = "MECHANICS_ONLY_ACCEPTED_FIELD_REPLAY_V1"
PARTICLE_H_THRESHOLD = 1.0e-4
INITIAL_FIXTURE_COMPONENT_COUNT = 6
NATIVE_DIAGNOSTIC_TAG_BY_STEP: Mapping[int, str] = {
    1: "R1a",
    10: "R1b",
    36: "R2a",
    363: "R2b",
    3633: "R3a",
    10899: "R3b",
    21798: "R3c",
    43596: "R4a",
    87191: "R4b",
    174382: "R4c",
}
NATIVE_SEGMENT_START_BY_STEP: Mapping[int, int] = {
    1: 0,
    10: 1,
    36: 10,
    363: 36,
    3633: 363,
    10899: 3633,
    21798: 10899,
    43596: 21798,
    87191: 43596,
    174382: 87191,
}
ALL_STEP_CLIP_LEDGER_SCHEMA = "DYNAMICS_ALL_STEP_CLIP_LEDGER_V1"


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


def _phase_storage_values(phi: np.ndarray, *, path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate the frozen CUDA representation interval and return clamp01(phi).

    The returned array is used only for phase-storage/inventory evaluation.
    The raw checkpoint field remains the source of identity hashes and all
    reported extrema, so the audit cannot conceal a representation excursion.
    """

    values = np.asarray(phi, dtype=np.float64)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if (
        minimum < PHI_REPRESENTATION_LOWER - FIELD_EQ_TOL
        or maximum > PHI_REPRESENTATION_UPPER + FIELD_EQ_TOL
    ):
        raise AuditError(
            "checkpoint phi leaves the frozen CUDA representation interval "
            f"[{PHI_REPRESENTATION_LOWER}, {PHI_REPRESENTATION_UPPER}]: {path}"
        )
    negative = int(np.count_nonzero(values < 0.0))
    above_one = int(np.count_nonzero(values > 1.0))
    floor = int(
        np.count_nonzero(
            np.isclose(values, PHI_REPRESENTATION_LOWER, rtol=0.0, atol=FIELD_EQ_TOL)
        )
    )
    ceiling = int(
        np.count_nonzero(
            np.isclose(values, PHI_REPRESENTATION_UPPER, rtol=0.0, atol=FIELD_EQ_TOL)
        )
    )
    return np.clip(values, 0.0, 1.0), {
        "frozen_interval": [PHI_REPRESENTATION_LOWER, PHI_REPRESENTATION_UPPER],
        "raw_phi_min": minimum,
        "raw_phi_max": maximum,
        "negative_raw_cell_count": negative,
        "above_one_raw_cell_count": above_one,
        "lower_floor_cell_count": floor,
        "upper_ceiling_cell_count": ceiling,
        "storage_evaluation": "h_of_phi(clamp01(phi))",
    }


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


def _population_checksum(bins: Sequence[Mapping[str, float]]) -> str:
    """Stable compact checksum of the decoded frozen PSD bins."""

    payload = json.dumps(list(bins), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    phase_storage_phi, phase_representation = _phase_storage_values(phi, path=path)
    if float(np.min(xb)) < -FIELD_EQ_TOL or float(np.max(xb)) > 1.0 + FIELD_EQ_TOL:
        raise AuditError(f"checkpoint xB leaves [0,1]: {path}")
    h = h_of_phi(phase_storage_phi)
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
        "phase_representation": phase_representation,
        "auxiliary_bins": {"GP": gp_bins, "beta_subgrid": subgrid_bins},
        "auxiliary_population_checksums": {
            "GP": _population_checksum(gp_bins),
            "beta_subgrid": _population_checksum(subgrid_bins),
        },
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


def _read_key_value_text(path: Path, *, label: str) -> dict[str, str]:
    if not path.is_file():
        raise AuditError(f"missing {label}: {path}")
    result: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        if "=" not in text:
            raise AuditError(f"malformed {label} line {line_number}: {path}")
        key, value = (part.strip() for part in text.split("=", 1))
        if not key or not value or key in result:
            raise AuditError(f"malformed or duplicate {label} key at line {line_number}: {path}")
        result[key] = value
    return result


def _pf_parameter_values(path: Path, *, contract: Any) -> dict[str, float]:
    raw = _read_key_value_text(path, label="frozen PF parameter file")
    required = (
        "dt", "dx", "dy", "dz", "t_real_unit", "temperature_C",
        "mu_reference_scale", "W", "kappa_phi", "v_A", "v_B",
        "Vm_compound", "Vm_alpha_0", "dVm_alpha_dxB", "eps_iso_over_vB",
    )
    result: dict[str, float] = {}
    for key in required:
        try:
            value = float(raw[key])
        except (KeyError, ValueError) as error:
            raise AuditError(f"frozen PF parameter file lacks finite {key}: {path}") from error
        if not math.isfinite(value):
            raise AuditError(f"frozen PF parameter value is non-finite: {key}")
        result[key] = value
    # These two parameters are defaults in main_cuda unless explicitly set in
    # the frozen file.  Reading them here keeps the offline reconstruction on
    # exactly the same logit branch as compute_mu_x_kernel.
    for key, default in (("Y_clip", 20.0), ("xB_eps", 1.0e-8)):
        try:
            value = float(raw.get(key, default))
        except ValueError as error:
            raise AuditError(f"invalid optional PF parameter {key}: {path}") from error
        if not math.isfinite(value) or value <= 0.0:
            raise AuditError(f"invalid optional PF parameter {key}: {path}")
        result[key] = value
    if not math.isclose(
        result["temperature_C"] + 273.15,
        float(contract.temperature_K),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise AuditError("frozen PF parameter temperature differs from validation contract")
    if not math.isclose(result["dt"], float(contract.canonical.value("numerics.dt_code")), rel_tol=0.0, abs_tol=1.0e-15):
        raise AuditError("frozen PF parameter dt differs from validation contract")
    return result


def _contract_thermodynamics(document: Mapping[str, Any], *, temperature_K: float, scale: float) -> tuple[Any, Any, float]:
    """Port the generated contract helpers without duplicating numeric inputs.

    The formulas and all coefficients are read from the frozen JSON, which is
    the same source used to generate the CUDA header.  ``mu_a`` and ``mu_b``
    return the dimensionless quantities used by CUDA's ``compute_mu_x_kernel``.
    """

    thermo = _mapping(document.get("thermodynamics"), "contract thermodynamics")
    standard = _mapping(_mapping(thermo.get("standard_state_coefficients"), "standard state coefficients").get("value"), "standard state coefficient values")
    gas = _number(_mapping(thermo.get("gas_constant_j_mol_k"), "gas constant").get("value"), "gas constant")
    delta_h = _number(_mapping(thermo.get("delta_H_J_mol"), "delta H").get("value"), "delta H")
    delta_s = _number(_mapping(thermo.get("delta_S_J_mol_K"), "delta S").get("value"), "delta S")

    def standard_state(name: str) -> float:
        entry = _mapping(standard.get(name), f"standard state {name}")
        transition = _number(entry.get("transition_K"), f"{name} transition")
        coefficients = entry.get("low" if temperature_K < transition else "high")
        if not isinstance(coefficients, Sequence) or len(coefficients) != 7:
            raise AuditError(f"unexpected standard-state coefficients for {name}")
        a0, a1, alog, a2, a3, ainv, pinv = (
            _number(value, f"{name} coefficient") for value in coefficients
        )
        tail = 0.0 if ainv == 0.0 else ainv * temperature_K**pinv
        return a0 + a1 * temperature_K + alog * temperature_K * math.log(temperature_K) + a2 * temperature_K**2 + a3 * temperature_K**3 + tail

    pbte = _mapping(standard.get("G_PbTe"), "G_PbTe")
    pbte_base = pbte.get("base")
    if not isinstance(pbte_base, Sequence) or len(pbte_base) != 2:
        raise AuditError("unexpected G_PbTe base coefficients")
    g_pbte = _number(pbte_base[0], "G_PbTe base") + _number(pbte_base[1], "G_PbTe slope") * temperature_K + standard_state("GHSER_Pb") + standard_state("GHSER_Te")

    ag2te = _mapping(standard.get("G_Ag2Te"), "G_Ag2Te")
    ag_base = ag2te.get("base_per_atom")
    weights = ag2te.get("atom_weights")
    if not isinstance(ag_base, Sequence) or len(ag_base) != 2 or not isinstance(weights, Sequence) or len(weights) != 2:
        raise AuditError("unexpected G_Ag2Te coefficients")
    atom = (
        _number(ag_base[0], "G_Ag2Te base")
        + _number(ag_base[1], "G_Ag2Te slope") * temperature_K
        + _number(weights[0], "G_Ag2Te Ag weight") * standard_state("GHSER_Ag")
        + _number(weights[1], "G_Ag2Te Te weight") * standard_state("GHSER_Te")
    )
    g_ag2te = _number(ag2te.get("molecular_multiplier"), "G_Ag2Te multiplier") * atom
    interaction = delta_h - temperature_K * delta_s

    def clamp_fraction(values: np.ndarray | float) -> np.ndarray | float:
        return np.clip(values, 1.0e-12, 1.0 - 1.0e-12)

    def mu_a(values: np.ndarray | float) -> np.ndarray | float:
        x = clamp_fraction(values)
        return (g_pbte + gas * temperature_K * np.log(1.0 - x) + interaction * x * x) / scale

    def mu_b(values: np.ndarray | float) -> np.ndarray | float:
        x = clamp_fraction(values)
        return (g_ag2te + gas * temperature_K * np.log(x) + interaction * (1.0 - x) * (1.0 - x)) / scale

    lo, hi = 1.0e-12, 0.5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        value = gas * temperature_K * math.log(mid) + interaction * (1.0 - mid) ** 2
        if value > 0.0:
            hi = mid
        else:
            lo = mid
    xeq = 0.5 * (lo + hi)
    return mu_a, mu_b, xeq


def _sigmoid_from_logit(values: np.ndarray, *, y_clip: float, xB_eps: float) -> np.ndarray:
    y = np.clip(np.asarray(values, dtype=np.float64), -y_clip, y_clip)
    positive = y >= 0.0
    out = np.empty_like(y)
    out[positive] = 1.0 / (1.0 + np.exp(-y[positive]))
    exp_y = np.exp(y[~positive])
    out[~positive] = exp_y / (1.0 + exp_y)
    return np.clip(out, xB_eps, 1.0 - xB_eps)


def _mechanics_bundle_path(run_root: Path, case: str, step: int) -> Path:
    if step == 0:
        return run_root / "mechanics_sync" / case / "step_0"
    return run_root / "mechanics_replay" / case / f"step_{step}"


def _read_exact_array(path: Path, *, dtype: str, count: int, label: str) -> np.ndarray:
    expected_bytes = np.dtype(dtype).itemsize * count
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise AuditError(f"missing or incorrectly sized {label}: {path}")
    values = np.fromfile(path, dtype=dtype)
    if values.size != count or not np.all(np.isfinite(values)):
        raise AuditError(f"non-finite or truncated {label}: {path}")
    return values


def _single_native_result_file(root: Path, filename: str, *, label: str) -> Path:
    """Find the one case-tagged CUDA result under this runner invocation."""

    matches = sorted(root.rglob(filename)) if root.is_dir() else []
    if len(matches) != 1:
        rendered = ", ".join(str(path) for path in matches) if matches else "<none>"
        raise AuditError(f"expected exactly one {label} below {root}, found {rendered}")
    return matches[0]


def _native_step_mass_diagnostic(run_root: Path, *, case: str, step: int) -> dict[str, Any]:
    if step == 0:
        return {
            "source": "R0_ZERO_STEP_NO_ACCEPTED_UPDATE",
            "coverage": "zero_step_only",
            "phi_clip_count_low": 0.0,
            "phi_clip_count_high": 0.0,
            "Y_clip_count_low": 0.0,
            "Y_clip_count_high": 0.0,
            "xB_clip_count_low": 0.0,
            "xB_clip_count_high": 0.0,
            "segment_total_phi_clip_count": 0.0,
            "segment_total_xB_clip_count": 0.0,
            "segment_total_Y_clip_count": 0.0,
            "all_step_ledger": "R0_ZERO_STEP_NOT_APPLICABLE",
        }
    try:
        tag = NATIVE_DIAGNOSTIC_TAG_BY_STEP[step]
        segment_start = NATIVE_SEGMENT_START_BY_STEP[step]
    except KeyError as error:
        raise AuditError(f"no native mass-diagnostic tag is registered for step {step}") from error
    result_root = run_root / "results" / case / tag
    path = _single_native_result_file(
        result_root,
        "dynamics_mass_diagnostics.csv",
        label="native endpoint mass diagnostics",
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row.get("step", "-1")) == step]
    if len(rows) != 1:
        raise AuditError(f"native endpoint mass diagnostics lack a unique step {step}: {path}")
    row = rows[0]
    keys = (
        "phi_clip_count_low", "phi_clip_count_high", "Y_clip_count_low",
        "Y_clip_count_high", "xB_clip_count_low", "xB_clip_count_high",
    )
    result: dict[str, Any] = {
        "source": str(path),
        "source_sha256": _sha256(path),
        # The runner requests exactly one row at each required accepted
        # endpoint.  The native summary consequently covers that sampled row,
        # rather than every intervening PF step in the restart segment.
        "coverage": "one_native_accepted_endpoint_row",
    }
    for key in keys:
        try:
            value = float(row[key])
        except (KeyError, ValueError) as error:
            raise AuditError(f"native endpoint diagnostics have invalid {key}: {path}") from error
        if not math.isfinite(value) or value < 0.0:
            raise AuditError(f"native endpoint diagnostics have invalid {key}: {path}")
        result[key] = value
    summary_path = _single_native_result_file(
        result_root,
        "mass_drift_summary.csv",
        label="native mass-diagnostic summary",
    )
    if summary_path.parent != path.parent:
        raise AuditError(f"native mass diagnostics and summary are not colocated: {result_root}")
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    if len(summary_rows) != 1:
        raise AuditError(f"native mass-diagnostic summary lacks one row: {summary_path}")
    summary = summary_rows[0]
    sampled_totals = (
        ("sampled_endpoint_summary_phi_clip_count", "total_clip_count_phi"),
        ("sampled_endpoint_summary_xB_clip_count", "total_clip_count_xB"),
        ("sampled_endpoint_summary_Y_clip_count", "total_clip_count_Y"),
    )
    for output_key, source_key in sampled_totals:
        try:
            value = float(summary[source_key])
        except (KeyError, ValueError) as error:
            raise AuditError(f"native mass-diagnostic summary has invalid {source_key}: {summary_path}") from error
        if not math.isfinite(value) or value < 0.0:
            raise AuditError(f"native mass-diagnostic summary has invalid {source_key}: {summary_path}")
        result[output_key] = value
    endpoint_totals = (
        ("sampled_endpoint_summary_phi_clip_count", "phi_clip_count_low", "phi_clip_count_high"),
        ("sampled_endpoint_summary_xB_clip_count", "xB_clip_count_low", "xB_clip_count_high"),
        ("sampled_endpoint_summary_Y_clip_count", "Y_clip_count_low", "Y_clip_count_high"),
    )
    for summary_key, low_key, high_key in endpoint_totals:
        endpoint_value = float(result[low_key]) + float(result[high_key])
        if not math.isclose(float(result[summary_key]), endpoint_value, rel_tol=0.0, abs_tol=0.0):
            raise AuditError(
                f"native endpoint mass-diagnostic summary does not match its sole sampled row: {summary_path}"
            )
    result["summary_source"] = str(summary_path)
    result["summary_source_sha256"] = _sha256(summary_path)

    ledger_path = _single_native_result_file(
        result_root,
        "dynamics_clip_ledger.json",
        label="native all-step clipping ledger",
    )
    if ledger_path.parent != path.parent:
        raise AuditError(f"native mass diagnostics and clipping ledger are not colocated: {result_root}")
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AuditError(f"invalid native all-step clipping ledger JSON: {ledger_path}") from error
    if not isinstance(ledger, Mapping) or ledger.get("schema") != ALL_STEP_CLIP_LEDGER_SCHEMA:
        raise AuditError(f"unexpected native all-step clipping ledger schema: {ledger_path}")
    try:
        ledger_start = int(ledger["segment_start_accepted_step"])
        ledger_end = int(ledger["segment_end_accepted_step"])
        ledger_count = int(ledger["accepted_step_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise AuditError(f"native all-step clipping ledger lacks segment bounds: {ledger_path}") from error
    if (ledger_start, ledger_end, ledger_count) != (segment_start, step, step - segment_start):
        raise AuditError(
            f"native all-step clipping ledger has wrong accepted-step interval: {ledger_path}"
        )
    ledger_keys = (
        "phi_projection_lower_count",
        "phi_projection_upper_count",
        "mu_x_logit_Y_projection_lower_count",
        "mu_x_logit_Y_projection_upper_count",
        "mu_x_logit_xB_projection_lower_count",
        "mu_x_logit_xB_projection_upper_count",
    )
    parsed_ledger: dict[str, Any] = {
        "path": str(ledger_path),
        "sha256": _sha256(ledger_path),
        "schema": ALL_STEP_CLIP_LEDGER_SCHEMA,
        "segment_start_accepted_step": ledger_start,
        "segment_end_accepted_step": ledger_end,
        "accepted_step_count": ledger_count,
    }
    for key in ledger_keys:
        value = ledger.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise AuditError(f"native all-step clipping ledger has invalid {key}: {ledger_path}")
        parsed_ledger[key] = value
    result["all_step_ledger"] = parsed_ledger
    result["segment_total_phi_clip_count"] = float(
        parsed_ledger["phi_projection_lower_count"] + parsed_ledger["phi_projection_upper_count"]
    )
    result["segment_total_xB_clip_count"] = float(
        parsed_ledger["mu_x_logit_xB_projection_lower_count"] + parsed_ledger["mu_x_logit_xB_projection_upper_count"]
    )
    result["segment_total_Y_clip_count"] = float(
        parsed_ledger["mu_x_logit_Y_projection_lower_count"] + parsed_ledger["mu_x_logit_Y_projection_upper_count"]
    )
    return result


def _shared_energy_reference(
    *, run_root: Path, contract: Any, params: Mapping[str, float]
) -> dict[str, Any]:
    """Use one fixed Case-A R0 matrix reference for every energy snapshot."""

    checkpoint = _checkpoint_path(run_root, "A", 0)
    header = _header_from(checkpoint)
    layout = _checkpoint_payload_layout(header)
    count = int(header["element_count"])
    phi = _array_at(checkpoint, layout["phi"], count)
    xB = _array_at(checkpoint, layout["xB"], count)
    storage_h = h_of_phi(np.clip(phi, 0.0, 1.0))
    matrix = xB[storage_h < 5.0e-3]
    if matrix.size == 0:
        raise AuditError(f"Case A R0 has no matrix support for fixed energy reference: {checkpoint}")
    xB_reference = float(np.mean(matrix, dtype=np.float64))
    mu_a, mu_b, _ = _contract_thermodynamics(
        _mapping(contract.document, "validation contract document"),
        temperature_K=float(params["temperature_C"]) + 273.15,
        scale=float(params["mu_reference_scale"]),
    )
    g_bulk0 = (1.0 - xB_reference) * float(mu_a(xB_reference)) + xB_reference * float(mu_b(xB_reference))
    return {
        "definition": "fixed Case-A R0 matrix h<0.005 reference used for every checkpoint/replay energy",
        "case": "A",
        "step": 0,
        "xB_reference": xB_reference,
        "g_bulk0_hat": g_bulk0,
    }


def _accepted_field_mechanics_diagnostics(
    *,
    run_root: Path,
    record: Mapping[str, Any],
    contract: Any,
    params: Mapping[str, float],
    energy_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a CUDA mechanics bundle to one V6 accepted checkpoint."""

    case = _text(record.get("case"), "diagnostic case")
    step = int(record["accepted_step"])
    checkpoint_path = Path(_text(record.get("checkpoint"), "diagnostic checkpoint"))
    header = _header_from(checkpoint_path)
    layout = _checkpoint_payload_layout(header)
    count = int(header["element_count"])
    phi = _array_at(checkpoint_path, layout["phi"], count)
    y = _array_at(checkpoint_path, layout["Y"], count)
    xB_checkpoint = _array_at(checkpoint_path, layout["xB"], count)
    bundle = _mechanics_bundle_path(run_root, case, step)
    summary_path = bundle / "replay_summary.txt"
    summary = _read_key_value_text(summary_path, label="mechanics accepted-field summary")
    expected_role = "online_synchronized_reference" if step == 0 else "offline_accepted_field_replay"
    expected = {
        "schema": MECHANICS_REPLAY_SCHEMA,
        "bundle_role": expected_role,
        "accepted_field_step": str(step),
        "solver_step": str(step + 1),
        "grid": "96,96,96",
        "dtype_real_fields": "float32-le",
        "dtype_source_fields": "float64-le",
        "time_advanced": "false",
        "phi_advanced": "false",
        "xB_advanced": "false",
        "checkpoint_written": "false",
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise AuditError(f"mechanics bundle {key} is not {value}: {summary_path}")
    try:
        residual = float(summary["solver_relative_residual"])
        residual_tolerance = float(summary["solver_residual_tolerance"])
        summary_elastic_mean = float(summary["elastic_energy_density_mean_hat"])
    except (KeyError, ValueError) as error:
        raise AuditError(f"mechanics bundle lacks a finite solver/energy summary: {summary_path}") from error
    if not all(math.isfinite(value) for value in (residual, residual_tolerance, summary_elastic_mean)):
        raise AuditError(f"mechanics bundle summary is non-finite: {summary_path}")
    if residual > residual_tolerance * (1.0 + 1.0e-8):
        raise AuditError(f"mechanics replay did not meet its residual tolerance: {summary_path}")

    phi_bundle_path = bundle / "accepted_phi.raw.f64"
    xB_bundle_path = bundle / "accepted_xB.raw.f64"
    phi_checkpoint_hash = hashlib.sha256(np.ascontiguousarray(phi, dtype="<f8").tobytes()).hexdigest()
    xB_checkpoint_hash = hashlib.sha256(np.ascontiguousarray(xB_checkpoint, dtype="<f8").tobytes()).hexdigest()
    if _sha256(phi_bundle_path) != phi_checkpoint_hash:
        raise AuditError(f"mechanics accepted phi bytes differ from V6 checkpoint: {bundle}")
    if _sha256(xB_bundle_path) != xB_checkpoint_hash:
        raise AuditError(f"mechanics accepted xB bytes differ from V6 checkpoint: {bundle}")
    phi_bundle = _read_exact_array(phi_bundle_path, dtype="<f8", count=count, label="accepted phi bundle")
    xB_bundle = _read_exact_array(xB_bundle_path, dtype="<f8", count=count, label="accepted xB bundle")
    stresses = [
        _read_exact_array(bundle / f"stress_{axis}.raw.f32", dtype="<f4", count=count, label=f"stress_{axis} bundle")
        for axis in ("xx", "yy", "zz")
    ]
    elastic = _read_exact_array(bundle / "elastic_energy_density.raw.f64", dtype="<f8", count=count, label="elastic-energy bundle")
    elastic_mean = float(np.mean(elastic, dtype=np.float64))
    if _relative_error(elastic_mean, summary_elastic_mean) > 1.0e-10:
        raise AuditError(f"mechanics elastic-energy field disagrees with its summary: {bundle}")

    xB_from_y = _sigmoid_from_logit(y, y_clip=float(params["Y_clip"]), xB_eps=float(params["xB_eps"]))
    xB_from_y_difference = float(np.max(np.abs(xB_from_y - xB_bundle)))
    if xB_from_y_difference > FIELD_EQ_TOL:
        raise AuditError(f"checkpoint Y does not reconstruct accepted xB at 1e-14: {checkpoint_path}")

    mu_a, mu_b, xeq = _contract_thermodynamics(
        _mapping(contract.document, "validation contract document"),
        temperature_K=float(params["temperature_C"]) + 273.15,
        scale=float(params["mu_reference_scale"]),
    )
    mu_a_eq = float(mu_a(xeq))
    mu_b_eq = float(mu_b(xeq))
    stoich = float(params["v_A"]) + float(params["v_B"])
    if abs(stoich) <= 1.0e-30:
        raise AuditError("invalid v_A + v_B in frozen PF parameters")
    mu0 = (float(params["v_A"]) * mu_a_eq + float(params["v_B"]) * mu_b_eq) / stoich
    h_raw = h_of_phi(np.asarray(phi_bundle, dtype=np.float64))
    mu_a_values = np.asarray(mu_a(xB_from_y), dtype=np.float64)
    mu_b_values = np.asarray(mu_b(xB_from_y), dtype=np.float64)
    vm_alpha = float(params["Vm_alpha_0"]) + float(params["dVm_alpha_dxB"]) * xB_from_y
    denominator = vm_alpha * (1.0 - h_raw) + float(params["Vm_compound"]) * h_raw
    denominator = np.where(np.abs(denominator) < 1.0e-12, np.copysign(1.0e-12, denominator), denominator)
    c_bulk = 1.0 / denominator
    mu_mix_for_mu = (1.0 - xB_from_y) * mu_a_values + xB_from_y * mu_b_values
    mu_total = (1.0 - h_raw) * mu_mix_for_mu + h_raw * mu0
    mu_x = c_bulk * (mu_b_values - mu_a_values - c_bulk * mu_total * float(params["dVm_alpha_dxB"]))
    stress_hydro = (
        stresses[0].astype(np.float64)
        + stresses[1].astype(np.float64)
        + stresses[2].astype(np.float64)
    )
    mu_x -= float(params["eps_iso_over_vB"]) * stress_hydro
    if not np.all(np.isfinite(mu_x)):
        raise AuditError(f"checkpoint/replay chemical potential is non-finite: {checkpoint_path}")

    # These are the CUDA functional pieces evaluated on the accepted state.
    # The chemical term follows compute_gbulk_excess_hat_kernel's clamp01(phi)
    # convention and reports an excess against one shared Case-A R0 reference;
    # gradient and double-well terms use raw phi, exactly as their CUDA kernels
    # do.  The solver did not emit these dynamics-mode scalars natively.
    h_chemical = h_of_phi(np.clip(phi_bundle, 0.0, 1.0))
    mu_a_energy = np.asarray(mu_a(xB_bundle), dtype=np.float64)
    mu_b_energy = np.asarray(mu_b(xB_bundle), dtype=np.float64)
    mu_mix_energy = (1.0 - xB_bundle) * mu_a_energy + xB_bundle * mu_b_energy
    chemical_absolute_hat = float(np.mean((1.0 - h_chemical) * mu_mix_energy + h_chemical * mu0, dtype=np.float64))
    chemical_excess_hat = chemical_absolute_hat - _number(
        energy_reference.get("g_bulk0_hat"), "shared energy reference g_bulk0_hat"
    )
    phi_3d = np.asarray(phi_bundle, dtype=np.float64).reshape((96, 96, 96))
    dx, dy, dz = (float(params[name]) for name in ("dx", "dy", "dz"))
    dphi_dx = (np.roll(phi_3d, -1, axis=0) - np.roll(phi_3d, 1, axis=0)) / (2.0 * dx)
    dphi_dy = (np.roll(phi_3d, -1, axis=1) - np.roll(phi_3d, 1, axis=1)) / (2.0 * dy)
    dphi_dz = (np.roll(phi_3d, -1, axis=2) - np.roll(phi_3d, 1, axis=2)) / (2.0 * dz)
    gradient_hat = float(np.mean(0.5 * float(params["kappa_phi"]) * (dphi_dx * dphi_dx + dphi_dy * dphi_dy + dphi_dz * dphi_dz), dtype=np.float64))
    barrier_hat = float(np.mean(float(params["W"]) * phi_bundle * phi_bundle * (1.0 - phi_bundle) * (1.0 - phi_bundle), dtype=np.float64))
    total_excess_hat = chemical_excess_hat + gradient_hat + barrier_hat + elastic_mean
    if not all(math.isfinite(value) for value in (chemical_absolute_hat, chemical_excess_hat, gradient_hat, barrier_hat, elastic_mean, total_excess_hat)):
        raise AuditError(f"checkpoint/replay energy decomposition is non-finite: {checkpoint_path}")

    storage_h, representation = _phase_storage_values(phi_bundle, path=checkpoint_path)
    matrix_mask = h_of_phi(storage_h) < 5.0e-3
    matrix_values = xB_bundle[matrix_mask]
    if matrix_values.size == 0:
        raise AuditError(f"accepted field has no matrix support for matrix diagnostics: {checkpoint_path}")
    native_clip = _native_step_mass_diagnostic(run_root, case=case, step=step)
    if native_clip["segment_total_xB_clip_count"] > 0.0 or native_clip["segment_total_Y_clip_count"] > 0.0:
        raise AuditError(
            f"composition clipping occurred in native CUDA segment ending at {case} step {step}; composition clamp is forbidden"
        )
    return {
        "provenance": DIAGNOSTIC_PROVENANCE,
        "time_h": step * float(params["dt"]) * float(params["t_real_unit"]) / 3600.0,
        "mechanics_bundle": {
            "path": str(bundle),
            "summary_sha256": _sha256(summary_path),
            "bundle_role": expected_role,
            "accepted_phi_sha256": phi_checkpoint_hash,
            "accepted_xB_sha256": xB_checkpoint_hash,
            "solver_relative_residual": residual,
            "solver_residual_tolerance": residual_tolerance,
            "elastic_energy_density_mean_summary_hat": summary_elastic_mean,
            "xB_from_Y_max_abs_difference": xB_from_y_difference,
        },
        "local_fields": {
            "xB_alpha_mean": float(np.mean(xB_bundle, dtype=np.float64)),
            "xB_alpha_min": float(np.min(xB_bundle)),
            "xB_alpha_max": float(np.max(xB_bundle)),
            "Y_min": float(np.min(y)),
            "Y_max": float(np.max(y)),
            "phi_min": float(np.min(phi_bundle)),
            "phi_max": float(np.max(phi_bundle)),
            "matrix_xB_h_lt_0p005_mean": float(np.mean(matrix_values, dtype=np.float64)),
            "matrix_xB_h_lt_0p005_min": float(np.min(matrix_values)),
            "matrix_xB_h_lt_0p005_max": float(np.max(matrix_values)),
            "nan_count": int(np.count_nonzero(np.isnan(phi_bundle)) + np.count_nonzero(np.isnan(y)) + np.count_nonzero(np.isnan(xB_bundle))),
            "inf_count": int(np.count_nonzero(np.isinf(phi_bundle)) + np.count_nonzero(np.isinf(y)) + np.count_nonzero(np.isinf(xB_bundle))),
            "phase_representation": representation,
        },
        "chemical_potential": {
            "definition": "CUDA_compute_mu_x_kernel_reconstructed_from_checkpoint_Y_and_accepted_field_stress",
            "mu0_compound_hat": mu0,
            "planar_solvus_xB": xeq,
            "mean": float(np.mean(mu_x, dtype=np.float64)),
            "min": float(np.min(mu_x)),
            "max": float(np.max(mu_x)),
        },
        "clipping": {
            "definition": "native endpoint mass-diagnostic counters plus a persistent all-step CUDA projection ledger; R0 has no accepted update",
            **native_clip,
            "checkpoint_phase_representation_boundary_counts": {
                "lower_floor": representation["lower_floor_cell_count"],
                "upper_ceiling": representation["upper_ceiling_cell_count"],
                "not_runtime_clip_event_counts": True,
            },
        },
        "auxiliary": {
            "GP_inventory_mol": _number(_mapping(record.get("inventory_mol"), "diagnostic inventory").get("Q_B_GP_mol"), "GP inventory"),
            "beta_subgrid_inventory_mol": _number(_mapping(record.get("inventory_mol"), "diagnostic inventory").get("Q_B_beta_subgrid_mol"), "beta-subgrid inventory"),
            "GP_PSD_checksum": _sha(_mapping(record.get("auxiliary_population_checksums"), "auxiliary checksums").get("GP"), "GP PSD checksum"),
            "beta_subgrid_PSD_checksum": _sha(_mapping(record.get("auxiliary_population_checksums"), "auxiliary checksums").get("beta_subgrid"), "beta-subgrid PSD checksum"),
            "frozen": int(header["aux_frozen"]) == 1 if int(header["aux_state_present"]) else True,
        },
        "energy": {
            "definition": "checkpoint/replay-derived CUDA accepted-state excess functional components in code/hat units",
            "reference": dict(energy_reference),
            "chemical_absolute_hat": chemical_absolute_hat,
            "chemical_excess_hat": chemical_excess_hat,
            "gradient_hat": gradient_hat,
            "barrier_hat": barrier_hat,
            "elastic_hat": elastic_mean,
            "total_excess_hat": total_excess_hat,
        },
    }


def _periodic_components(mask: np.ndarray) -> np.ndarray:
    """Return dense periodic six-neighbour labels without a SciPy dependency."""

    if mask.shape != (96, 96, 96):
        raise AuditError(f"periodic component mask is not 96^3: {mask.shape}")
    total = mask.size
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.int32)
    parent = np.arange(total, dtype=np.int32)
    rank = np.zeros(total, dtype=np.uint8)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        root_left = find(int(left))
        root_right = find(int(right))
        if root_left == root_right:
            return
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1

    indices = np.arange(total, dtype=np.int32).reshape(mask.shape)
    for axis in range(3):
        linked = np.flatnonzero((mask & np.roll(mask, -1, axis=axis)).ravel())
        neighbours = np.roll(indices, -1, axis=axis).ravel()[linked]
        for left, right in zip(linked, neighbours):
            union(int(left), int(right))
    active = np.flatnonzero(mask.ravel())
    roots = np.fromiter((find(int(value)) for value in active), dtype=np.int32, count=active.size)
    _, inverse = np.unique(roots, return_inverse=True)
    labels = np.zeros(total, dtype=np.int32)
    labels[active] = inverse + 1
    return labels.reshape(mask.shape)


def _periodic_centroid(flat_indices: np.ndarray, shape: tuple[int, int, int]) -> tuple[float, float, float]:
    coordinates = np.unravel_index(flat_indices, shape)
    output: list[float] = []
    for values, size in zip(coordinates, shape):
        angles = 2.0 * math.pi * (values.astype(np.float64) + 0.5) / size
        angle = math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
        output.append((angle % (2.0 * math.pi)) * size / (2.0 * math.pi))
    return output[0], output[1], output[2]


def _component_snapshot(phi: np.ndarray, *, dx_nm: float) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    storage_phi = np.clip(np.asarray(phi, dtype=np.float64), 0.0, 1.0)
    h = h_of_phi(storage_phi).reshape((96, 96, 96))
    mask = h > PARTICLE_H_THRESHOLD
    labels = _periodic_components(mask)
    cell_volume_nm3 = dx_nm**3
    box_volume_nm3 = (96.0 * dx_nm) ** 3
    rows: list[dict[str, Any]] = []
    radii: list[float] = []
    for component in range(1, int(labels.max()) + 1):
        flat_indices = np.flatnonzero(labels.ravel() == component)
        if flat_indices.size == 0:
            continue
        h_volume_nm3 = float(np.sum(h.ravel()[flat_indices], dtype=np.float64) * cell_volume_nm3)
        radius_nm = (3.0 * h_volume_nm3 / (4.0 * math.pi)) ** (1.0 / 3.0)
        coords = np.unravel_index(flat_indices, labels.shape)
        wraps = tuple(
            bool(np.any(axis_values == 0) and np.any(axis_values == size - 1))
            for axis_values, size in zip(coords, labels.shape)
        )
        centroid = _periodic_centroid(flat_indices, labels.shape)
        radii.append(radius_nm)
        rows.append(
            {
                "dense_component_label": component,
                "component_cell_count": int(flat_indices.size),
                "h_volume_nm3": h_volume_nm3,
                "equivalent_radius_nm": radius_nm,
                "centroid_x_nm": centroid[0] * dx_nm,
                "centroid_y_nm": centroid[1] * dx_nm,
                "centroid_z_nm": centroid[2] * dx_nm,
                "wraps_periodic_x": wraps[0],
                "wraps_periodic_y": wraps[1],
                "wraps_periodic_z": wraps[2],
                "_flat_indices": flat_indices,
            }
        )
    threshold_faces = int(sum(np.count_nonzero(mask != np.roll(mask, -1, axis=axis)) for axis in range(3)))
    sv_equivalent = 4.0 * math.pi * sum(radius * radius for radius in radii) / box_volume_nm3
    sv_threshold_faces = threshold_faces * dx_nm**2 / box_volume_nm3
    summary: dict[str, Any] = {
        "definition": "h(clamp01(phi)) > 1e-4; periodic six-neighbour connected components",
        "particle_h_threshold": PARTICLE_H_THRESHOLD,
        "beta_volume_fraction": float(np.mean(h, dtype=np.float64)),
        "total_resolved_beta_h_volume_nm3": float(np.sum(h, dtype=np.float64) * cell_volume_nm3),
        "thresholded_component_h_volume_nm3": float(np.sum(h[mask], dtype=np.float64) * cell_volume_nm3),
        "connected_particle_count": len(rows),
        "equivalent_radius_min_nm": min(radii) if radii else None,
        "equivalent_radius_mean_nm": float(np.mean(radii, dtype=np.float64)) if radii else None,
        "S_v_equivalent_sphere_nm_inverse": sv_equivalent,
        "S_v_threshold_faces_nm_inverse": sv_threshold_faces,
        "threshold_interface_face_count": threshold_faces,
        "periodic_component_status": "PERIODIC_6_NEIGHBOR_CCL",
    }
    return summary, rows, labels.ravel()


def _attach_component_lineage(
    observations: Sequence[dict[str, Any]], *, contract: Any, params: Mapping[str, float]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Add component observables and fail closed on unresolved identity events."""

    dx_nm = float(contract.dx_m) * 1.0e9
    seconds_per_step = float(params["dt"]) * float(params["t_real_unit"])
    history: list[dict[str, Any]] = []
    failures: list[str] = []
    for case in CASES:
        snapshots = sorted(
            (record for record in observations if record.get("case") == case),
            key=lambda record: int(record["accepted_step"]),
        )
        previous_ids: np.ndarray | None = None
        previous_component_ids: set[int] = set()
        previous_step: int | None = None
        next_particle_id = 1
        initial_count: int | None = None
        for record in snapshots:
            step = int(record["accepted_step"])
            checkpoint_path = Path(_text(record.get("checkpoint"), "component checkpoint"))
            header = _header_from(checkpoint_path)
            layout = _checkpoint_payload_layout(header)
            phi = _array_at(checkpoint_path, layout["phi"], int(header["element_count"]))
            summary, components, current_dense = _component_snapshot(phi, dx_nm=dx_nm)
            current_ids = np.zeros_like(current_dense, dtype=np.int32)
            parent_to_children: dict[int, set[int]] = {}
            component_parent_ids: dict[int, list[int]] = {}
            if previous_ids is not None:
                for component in components:
                    old = previous_ids[component["_flat_indices"]]
                    parents = sorted(int(value) for value in np.unique(old[old > 0]))
                    component_parent_ids[int(component["dense_component_label"])] = parents
                    for parent_id in parents:
                        parent_to_children.setdefault(parent_id, set()).add(int(component["dense_component_label"]))
            split_parent_ids = {parent_id for parent_id, children in parent_to_children.items() if len(children) > 1}
            current_component_ids: set[int] = set()
            merge_count = split_count = new_count = 0
            for component in components:
                dense = int(component["dense_component_label"])
                parents = component_parent_ids.get(dense, [])
                merge = len(parents) > 1
                split = any(parent_id in split_parent_ids for parent_id in parents)
                if previous_ids is None:
                    particle_id = next_particle_id
                    next_particle_id += 1
                    event = "initial_component"
                elif len(parents) == 1 and not merge and not split:
                    particle_id = parents[0]
                    event = "continuous_identity"
                elif not parents:
                    particle_id = next_particle_id
                    next_particle_id += 1
                    event = "new_unmatched_component"
                    new_count += 1
                else:
                    particle_id = next_particle_id
                    next_particle_id += 1
                    event = "unresolved_merge_split"
                if merge:
                    merge_count += 1
                if split:
                    split_count += 1
                current_component_ids.add(particle_id)
                current_ids[component["_flat_indices"]] = particle_id
                history_row = {
                    key: value for key, value in component.items() if key != "_flat_indices"
                }
                history_row.update(
                    {
                        "case": case,
                        "step": step,
                        "time_h": step * seconds_per_step / 3600.0,
                        "particle_id": particle_id,
                        "previous_step": previous_step,
                        "overlap_parent_ids": ";".join(str(value) for value in parents),
                        "overlap_parent_count": len(parents),
                        "identity_event": event,
                        "unresolved_merge": merge,
                        "unresolved_split": split,
                        "unresolved_new_component": event == "new_unmatched_component",
                    }
                )
                history.append(history_row)
            disappeared = sorted(previous_component_ids - set(parent_to_children)) if previous_ids is not None else []
            for particle_id in disappeared:
                if step == 1:
                    loss_category = "first_step_profile_loss_candidate"
                elif step <= 363:
                    loss_category = "early_physical_dissolution_candidate"
                else:
                    loss_category = "later_coarsening_or_dissolution_candidate"
                history.append(
                    {
                        "case": case,
                        "step": step,
                        "time_h": step * seconds_per_step / 3600.0,
                        "particle_id": particle_id,
                        "previous_step": previous_step,
                        "dense_component_label": None,
                        "component_cell_count": 0,
                        "h_volume_nm3": 0.0,
                        "equivalent_radius_nm": 0.0,
                        "centroid_x_nm": None,
                        "centroid_y_nm": None,
                        "centroid_z_nm": None,
                        "wraps_periodic_x": False,
                        "wraps_periodic_y": False,
                        "wraps_periodic_z": False,
                        "overlap_parent_ids": str(particle_id),
                        "overlap_parent_count": 1,
                        "identity_event": loss_category,
                        "unresolved_merge": False,
                        "unresolved_split": False,
                        "unresolved_new_component": False,
                    }
                )
            if initial_count is None:
                initial_count = int(summary["connected_particle_count"])
                if initial_count != INITIAL_FIXTURE_COMPONENT_COUNT:
                    failures.append(
                        f"FAIL_INITIALIZATION_GEOMETRY: {case} R0 initializes {initial_count} periodic components, expected {INITIAL_FIXTURE_COMPONENT_COUNT}"
                    )
            unresolved = merge_count > 0 or split_count > 0 or new_count > 0
            if unresolved:
                failures.append(
                    f"{case} step {step} has unresolved periodic component identity event(s): "
                    f"merge={merge_count} split={split_count} new={new_count}"
                )
            summary.update(
                {
                    "initial_component_count": initial_count,
                    "immediate_component_loss_from_R0": (
                        initial_count - int(summary["connected_particle_count"])
                        if step == 1 and initial_count is not None
                        else 0
                    ),
                    "component_disappearance_candidate_count": len(disappeared),
                    "unresolved_merge_count": merge_count,
                    "unresolved_split_count": split_count,
                    "unresolved_new_component_count": new_count,
                    "identity_resolution_status": (
                        "FAIL_CLOSED_UNRESOLVED_COMPONENT_EVENT"
                        if unresolved
                        else "PASS_PERIODIC_COMPONENT_IDENTITY"
                    ),
                }
            )
            runtime = record.get("runtime_diagnostics")
            if not isinstance(runtime, dict):
                raise AuditError(f"missing accepted-state diagnostics before component analysis: {checkpoint_path}")
            runtime["microstructure"] = summary
            previous_ids = current_ids
            previous_component_ids = current_component_ids
            previous_step = step
    return history, failures


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
    for path, layout in ((left_path, left_layout), (right_path, right_layout)):
        if path.stat().st_size != layout["file_size"]:
            raise AuditError(f"restart comparison finds an invalid V6 payload size: {label}: {path}")
    header_mismatches = [
        key for key in left_header
        if left_header[key] != right_header[key]
    ]
    if header_mismatches:
        raise AuditError(
            "restart comparison V6 header/provenance mismatch for "
            f"{label}: {', '.join(header_mismatches)}"
        )
    summary: dict[str, Any] = {
        "label": label,
        "left": str(left_path),
        "right": str(right_path),
        "v6_header_and_provenance": {
            "all_fields_exact": True,
            "compared_field_count": len(left_header),
            "mismatched_fields": [],
        },
        "fields": {},
    }
    for field in ("phi", "Y", "xB", "dY_dt_prev"):
        left = _array_at(left_path, left_layout[field], count)
        right = _array_at(right_path, right_layout[field], count)
        maximum = float(np.max(np.abs(left - right)))
        summary["fields"][field] = {"max_abs_difference": maximum, "within_1e-14": maximum <= FIELD_EQ_TOL}
        if maximum > FIELD_EQ_TOL:
            raise AuditError(f"restart comparison exceeds 1e-14 for {field}: {label}")
    elastic_bytes = 24 * int(left_header["k_element_count"])
    if bool(left_header["elastic_state_present"]):
        with left_path.open("rb") as handle:
            handle.seek(left_layout["elastic"])
            left_elastic = handle.read(elastic_bytes)
        with right_path.open("rb") as handle:
            handle.seek(right_layout["elastic"])
            right_elastic = handle.read(elastic_bytes)
        if len(left_elastic) != elastic_bytes or len(right_elastic) != elastic_bytes:
            raise AuditError(f"restart comparison truncated elastic warm state: {label}")
        if left_elastic != right_elastic:
            raise AuditError(f"restart comparison elastic warm state differs: {label}")
    summary["elastic_warm_state"] = {
        "present": bool(left_header["elastic_state_present"]),
        "byte_count": elastic_bytes,
        "byte_identical": True,
    }
    auxiliary_summary: dict[str, Any] = {}
    for population, header_key, layout_key in (
        ("GP", "gp_bin_count", "gp_bins"),
        ("beta_subgrid", "beta_subgrid_bin_count", "beta_subgrid_bins"),
    ):
        bin_count = int(left_header[header_key])
        left_inventory, left_bins = _bin_inventory(left_path, left_layout[layout_key], bin_count)
        right_inventory, right_bins = _bin_inventory(right_path, right_layout[layout_key], bin_count)
        if left_bins != right_bins:
            raise AuditError(f"restart comparison decoded {population} PSD bins differ: {label}")
        auxiliary_summary[population] = {
            "bin_count": bin_count,
            "decoded_bins_exact": True,
            "inventory_mol": left_inventory,
            "inventory_exact": left_inventory == right_inventory,
        }
    summary["auxiliary_populations"] = auxiliary_summary
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


def _live_component_signature(rows: Sequence[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
    signature: list[tuple[Any, ...]] = []
    for row in rows:
        if row.get("dense_component_label") is None:
            continue
        signature.append(
            (
                int(row["particle_id"]),
                int(row["dense_component_label"]),
                int(row["component_cell_count"]),
                float(row["h_volume_nm3"]),
                float(row["equivalent_radius_nm"]),
                bool(row["wraps_periodic_x"]),
                bool(row["wraps_periodic_y"]),
                bool(row["wraps_periodic_z"]),
            )
        )
    return sorted(signature)


def _compare_local_case_runtime_diagnostics(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    left_components: Sequence[Mapping[str, Any]],
    right_components: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare control observables which are not raw checkpoint fields."""

    left_runtime = _mapping(left.get("runtime_diagnostics"), "left runtime diagnostics")
    right_runtime = _mapping(right.get("runtime_diagnostics"), "right runtime diagnostics")
    compared: dict[str, dict[str, Any]] = {}

    def compare_scalar(group: str, key: str) -> None:
        lhs = _number(_mapping(left_runtime.get(group), f"left {group}").get(key), f"left {group}.{key}")
        rhs = _number(_mapping(right_runtime.get(group), f"right {group}").get(key), f"right {group}.{key}")
        maximum = abs(lhs - rhs)
        compared[f"{group}.{key}"] = {"max_abs_difference": maximum, "within_1e-14": maximum <= FIELD_EQ_TOL}
        if maximum > FIELD_EQ_TOL:
            raise AuditError(
                f"{left['case']}/{right['case']} {group}.{key} differs by more than 1e-14 at step {left['accepted_step']}"
            )

    for key in (
        "xB_alpha_mean", "xB_alpha_min", "xB_alpha_max", "phi_min", "phi_max",
        "matrix_xB_h_lt_0p005_mean", "matrix_xB_h_lt_0p005_min", "matrix_xB_h_lt_0p005_max",
    ):
        compare_scalar("local_fields", key)
    for key in ("mean", "min", "max"):
        compare_scalar("chemical_potential", key)
    for key in ("chemical_excess_hat", "gradient_hat", "barrier_hat", "elastic_hat", "total_excess_hat"):
        compare_scalar("energy", key)
    for key in (
        "phi_clip_count_low", "phi_clip_count_high", "Y_clip_count_low", "Y_clip_count_high",
        "xB_clip_count_low", "xB_clip_count_high", "segment_total_phi_clip_count",
        "segment_total_xB_clip_count", "segment_total_Y_clip_count",
    ):
        compare_scalar("clipping", key)
    for key in (
        "beta_volume_fraction", "total_resolved_beta_h_volume_nm3", "thresholded_component_h_volume_nm3", "S_v_equivalent_sphere_nm_inverse",
        "S_v_threshold_faces_nm_inverse",
    ):
        compare_scalar("microstructure", key)
    left_micro = _mapping(left_runtime.get("microstructure"), "left microstructure")
    right_micro = _mapping(right_runtime.get("microstructure"), "right microstructure")
    discrete_keys = (
        "connected_particle_count", "threshold_interface_face_count", "identity_resolution_status",
        "unresolved_merge_count", "unresolved_split_count", "unresolved_new_component_count",
        "immediate_component_loss_from_R0",
    )
    discrete: dict[str, Any] = {}
    for key in discrete_keys:
        equal = left_micro.get(key) == right_micro.get(key)
        discrete[key] = {"equal": equal, "left": left_micro.get(key), "right": right_micro.get(key)}
        if not equal:
            if key == "immediate_component_loss_from_R0" and left["case"] == "A" and right["case"] == "B":
                raise AuditError("FAIL_IDENTITY_ADAPTER_RUNTIME: Case B has an adapter-specific first-step component loss")
            raise AuditError(f"{left['case']}/{right['case']} microstructure {key} differs at step {left['accepted_step']}")
    component_equal = _live_component_signature(left_components) == _live_component_signature(right_components)
    if not component_equal:
        raise AuditError(f"{left['case']}/{right['case']} particle labels/radii differ at step {left['accepted_step']}")
    for key in ("Q_B_matrix_mol", "Q_B_beta_resolved_fixed_mol"):
        lhs = _number(_mapping(left.get("inventory_mol"), "left inventory").get(key), f"left {key}")
        rhs = _number(_mapping(right.get("inventory_mol"), "right inventory").get(key), f"right {key}")
        maximum = abs(lhs - rhs)
        compared[f"inventory.{key}"] = {"max_abs_difference": maximum, "within_1e-14": maximum <= FIELD_EQ_TOL}
        if maximum > FIELD_EQ_TOL:
            raise AuditError(f"{left['case']}/{right['case']} {key} differs by more than 1e-14 at step {left['accepted_step']}")
    return {
        "checkpoint_replay_derived": DIAGNOSTIC_PROVENANCE,
        "scalars": compared,
        "discrete": discrete,
        "particle_labels_and_components_identical": component_equal,
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
    params = _pf_parameter_values(run_root / "pf_input.params", contract=contract)
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

    # Checkpoint fields establish the accepted PF state; the bundle adds a
    # deterministic CUDA mechanics solve for that exact state.  This is done
    # before control comparisons so A/B and B/C include every requested local
    # observable, not only phi/xB/Y history.
    component_history: list[dict[str, Any]] = []
    energy_reference: dict[str, Any] | None = None
    try:
        energy_reference = _shared_energy_reference(run_root=run_root, contract=contract, params=params)
    except AuditError as error:
        runtime_failures.append(str(error))
    if energy_reference is not None:
        for record in observations:
            try:
                record["runtime_diagnostics"] = _accepted_field_mechanics_diagnostics(
                    run_root=run_root,
                    record=record,
                    contract=contract,
                    params=params,
                    energy_reference=energy_reference,
                )
            except AuditError as error:
                runtime_failures.append(str(error))
    if not runtime_failures:
        try:
            component_history, component_failures = _attach_component_lineage(
                observations, contract=contract, params=params
            )
            runtime_failures.extend(component_failures)
        except AuditError as error:
            runtime_failures.append(str(error))

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
    observation_by_case_step = {
        (str(record["case"]), int(record["accepted_step"])): record for record in observations
    }
    component_history_by_case_step: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in component_history:
        component_history_by_case_step.setdefault((str(row["case"]), int(row["step"])), []).append(row)
    common_abc_steps = set(per_case_available["A"])
    common_abc_steps.intersection_update(per_case_available["B"])
    common_abc_steps.intersection_update(per_case_available["C"])
    for step in sorted(common_abc_steps):
        for left_case, right_case in (("A", "B"), ("B", "C")):
            try:
                comparison = _compare_local_case_fields(run_root, step, left_case, right_case)
                left_record = observation_by_case_step.get((left_case, step))
                right_record = observation_by_case_step.get((right_case, step))
                if left_record is None or right_record is None:
                    raise AuditError(f"missing accepted-state observation for {left_case}/{right_case} step {step}")
                comparison["checkpoint_replay_diagnostics"] = _compare_local_case_runtime_diagnostics(
                    left_record,
                    right_record,
                    left_components=component_history_by_case_step.get((left_case, step), []),
                    right_components=component_history_by_case_step.get((right_case, step), []),
                )
                local_control_comparisons.append(comparison)
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
                "time_h_from_R0": float(record["accepted_step"]) * float(params["dt"]) * float(params["t_real_unit"]) / 3600.0,
                "Q_B_matrix_mol": inventory["Q_B_matrix_mol"],
                "delta_from_R0_mol": float(inventory["Q_B_matrix_mol"]) - initial_matrix[case],
            }
        )
    pulse_by_case: dict[str, dict[str, Any]] = {}
    seconds_per_step = float(params["dt"]) * float(params["t_real_unit"])
    for case in CASES:
        rows = [row for row in matrix_inventory_pulse_summary if row["case"] == case]
        if not rows:
            continue
        peak = max(rows, key=lambda row: abs(float(row["delta_from_R0_mol"])))
        initial = initial_matrix[case]
        pulse_by_case[case] = {
            "window": "ALL_OBSERVED_ACCEPTED_CHECKPOINTS",
            "own_R0_matrix_inventory_mol": initial,
            "peak_minus_own_R0_mol": peak["delta_from_R0_mol"],
            "peak_abs_minus_own_R0_mol": abs(float(peak["delta_from_R0_mol"])),
            "peak_step": peak["step"],
            "peak_time_s_from_R0": float(peak["step"]) * seconds_per_step,
            "peak_time_h_from_R0": float(peak["step"]) * seconds_per_step / 3600.0,
            "normalized_peak_minus_own_R0": float(peak["delta_from_R0_mol"]) / max(abs(initial), 1.0e-300),
        }
    a_b_pulse_comparison: dict[str, Any] | str
    if "A" in pulse_by_case and "B" in pulse_by_case:
        a_peak = pulse_by_case["A"]
        b_peak = pulse_by_case["B"]
        a_b_pulse_comparison = {
            "window": "ALL_OBSERVED_ACCEPTED_CHECKPOINTS",
            "A_minus_B_peak_delta_mol": float(a_peak["peak_minus_own_R0_mol"]) - float(b_peak["peak_minus_own_R0_mol"]),
            "A_peak_step": a_peak["peak_step"],
            "B_peak_step": b_peak["peak_step"],
            "same_peak_step": a_peak["peak_step"] == b_peak["peak_step"],
        }
    else:
        a_b_pulse_comparison = "NOT_CAPTURED_UNTIL_A_AND_B_HAVE_ACCEPTED_CHECKPOINTS"

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
        r4_pairs = (
            (
                _checkpoint_path(run_root, "E", 87191),
                run_root / "restart_qualification/E/checkpoint_24h_from_6h.pfzck",
                "E 24h primary/6h-start checkpoint",
            ),
            (
                _checkpoint_path(run_root, "E", 174382),
                run_root / "restart_qualification/E/continuous_6to48h.pfzck",
                "E 48h primary/6h-continuous",
            ),
            (
                run_root / "restart_qualification/E/continuous_6to48h.pfzck",
                run_root / "restart_qualification/E/restart_48h_from_24h.pfzck",
                "E 48h 6h-continuous/24h-restart",
            ),
        )
        for left, right, label in r4_pairs:
            if not left.is_file() or not right.is_file():
                missing.append(f"{label}: restart qualification")
                continue
            try:
                restart_comparisons.append(_compare_field_checkpoints(left, right, label=label))
            except AuditError as error:
                runtime_failures.append(str(error))

    case_e_runtime_closure: dict[str, Any] = {}
    e_records = sorted(
        (record for record in observations if record.get("case") == "E" and isinstance(record.get("runtime_diagnostics"), Mapping)),
        key=lambda record: int(record["accepted_step"]),
    )
    if e_records:
        initial_aux = _mapping(e_records[0]["runtime_diagnostics"], "E runtime diagnostics").get("auxiliary")
        initial_aux = _mapping(initial_aux, "E initial auxiliary diagnostics")
        frozen_auxiliary = True
        for record in e_records:
            auxiliary = _mapping(_mapping(record["runtime_diagnostics"], "E runtime diagnostics").get("auxiliary"), "E auxiliary diagnostics")
            frozen_auxiliary = frozen_auxiliary and bool(auxiliary.get("frozen"))
            for key in ("GP_inventory_mol", "beta_subgrid_inventory_mol"):
                frozen_auxiliary = frozen_auxiliary and _relative_error(
                    _number(auxiliary.get(key), f"E {key}"), _number(initial_aux.get(key), f"E initial {key}")
                ) <= REL_TOL
            for key in ("GP_PSD_checksum", "beta_subgrid_PSD_checksum"):
                frozen_auxiliary = frozen_auxiliary and auxiliary.get(key) == initial_aux.get(key)
        if not frozen_auxiliary:
            runtime_failures.append("Case E frozen auxiliary storage changed across accepted CUDA checkpoints")
        max_residual = max(
            float(_mapping(record["inventory_mol"], "E inventory")["relative_error_vs_case_t0"])
            for record in e_records
        )
        e_restart_labels = [str(record["label"]) for record in restart_comparisons if str(record["label"]).startswith("E ")]
        case_e_runtime_closure = {
            "frozen_auxiliary_inventory_and_PSD_persistent": frozen_auxiliary,
            "max_four_bucket_relative_residual": max_residual,
            "restart_qualifications": e_restart_labels,
            "matrix_pulse": pulse_by_case.get("E"),
            "matrix_pulse_interpretation": (
                "OBSERVED_RELATIVE_TO_OWN_R0; auxiliary PSD/inventory is frozen, so any recorded matrix evolution has no GP/subgrid handoff transfer route"
            ),
            "double_count_status": "PASS_FOUR_BUCKET_SUM_AND_FIXED_RESOLVED_INVENTORY_CHECKED_AT_EVERY_CHECKPOINT",
        }

    clipping_coverage: dict[str, Any] = {}
    for case in CASES:
        records = [
            record for record in observations
            if record.get("case") == case and isinstance(record.get("runtime_diagnostics"), Mapping)
        ]
        if not records:
            continue
        total_phi = total_xB = total_y = 0.0
        for record in records:
            clipping = _mapping(_mapping(record["runtime_diagnostics"], "runtime diagnostics").get("clipping"), "runtime clipping")
            total_phi += _number(clipping.get("segment_total_phi_clip_count"), "segment phi clipping")
            total_xB += _number(clipping.get("segment_total_xB_clip_count"), "segment xB clipping")
            total_y += _number(clipping.get("segment_total_Y_clip_count"), "segment Y clipping")
        clipping_coverage[case] = {
            "native_coverage": "persistent native CUDA ledger across every primary accepted PF segment; R0 is zero-step",
            "all_step_phi_projection_count": total_phi,
            "all_step_xB_projection_count": total_xB,
            "all_step_Y_projection_count": total_y,
            "composition_clipping_status": "PASS_NO_COMPOSITION_PROJECTION_IN_ALL_ACCEPTED_STEPS" if total_xB == 0.0 and total_y == 0.0 else "FAIL_COMPOSITION_PROJECTION_IN_ACCEPTED_STEPS",
            "all_accepted_step_bound_status": "PF_CONSERVED_Y_ZERO_MODE_V1 validates Y bounds before each accepted update and rejects invalid states; no composition-clamp path is enabled",
            "phi_projection_status": "ACCOUNTED_NATIVE_PHASE_REPRESENTATION_PROJECTION_COUNTER",
        }

    if runtime_failures:
        status = "FAIL_CUDA_AE_RUNTIME_AUDIT"
        exit_code = 1
    elif missing:
        status = "INCOMPLETE_CUDA_AE_RUNTIME_AUDIT"
        exit_code = 1 if required_stage else 0
    elif required_stage == "R4":
        status = "PASS_CUDA_AE_SMOKE"
        exit_code = 0
    elif required_stage == "R3":
        status = "PASS_R3_CUDA_AE_RUNTIME_GATE"
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
        "schema_version": "PF_CUDA_AE_RUNTIME_AUDIT_V2",
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
        "checkpoint_replay_diagnostic_provenance": DIAGNOSTIC_PROVENANCE,
        "shared_energy_reference": energy_reference,
        "observed_checkpoints": observations,
        "available_steps_by_case": per_case_available,
        "missing_required_artifacts": missing,
        "runtime_failures": runtime_failures,
        "r0_raw_field_comparisons": raw_comparisons,
        "A_B_C_local_field_comparisons": local_control_comparisons,
        "D_vs_A_R0_matrix_to_GP_control": d_r0_control,
        "matrix_inventory_pulse_summary": matrix_inventory_pulse_summary,
        "matrix_inventory_pulse": {
            "seconds_per_step": seconds_per_step,
            "per_case": pulse_by_case,
            "A_vs_B": a_b_pulse_comparison,
        },
        "restart_comparisons": restart_comparisons,
        "case_E_runtime_closure": case_e_runtime_closure,
        "clipping_coverage": clipping_coverage,
        "component_history_row_count": len(component_history),
    }
    _write_json(out_dir / "cuda_ae_runtime_audit.json", payload)
    _write_csv_summaries(out_dir, observations, restart_comparisons, component_history)
    return payload, exit_code


def _write_csv_summaries(
    out_dir: Path,
    observations: Sequence[Mapping[str, Any]],
    restart_comparisons: Sequence[Mapping[str, Any]],
    component_history: Sequence[Mapping[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = out_dir / "cuda_ae_inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("case", "step", "Q_B_total_mol", "Q_B_matrix_mol", "Q_B_GP_mol", "Q_B_beta_subgrid_mol", "Q_B_beta_resolved_fixed_mol", "relative_error_vs_case_t0"))
        for record in observations:
            inventory = _mapping(record["inventory_mol"], "observation inventory")
            writer.writerow((record["case"], record["accepted_step"], *(inventory[name] for name in ("Q_B_total_mol", "Q_B_matrix_mol", "Q_B_GP_mol", "Q_B_beta_subgrid_mol", "Q_B_beta_resolved_fixed_mol", "relative_error_vs_case_t0"))))

    trajectory_path = out_dir / "cuda_ae_trajectories.csv"
    trajectory_columns = (
        "case", "step", "time_h", "diagnostic_provenance", "checkpoint_sha256",
        "validation_contract_hash", "fixture_manifest_sha256", "package_handoff_hash",
        "Q_B_total_mol", "Q_B_matrix_mol", "Q_B_GP_mol", "Q_B_beta_subgrid_mol",
        "Q_B_beta_resolved_fixed_mol", "four_bucket_relative_residual",
        "xB_alpha_mean", "xB_alpha_min", "xB_alpha_max", "phi_min", "phi_max",
        "matrix_xB_h_lt_0p005_mean", "matrix_xB_h_lt_0p005_min", "matrix_xB_h_lt_0p005_max",
        "chemical_potential_mean", "chemical_potential_min", "chemical_potential_max",
        "phi_clip_count_low", "phi_clip_count_high", "Y_clip_count_low", "Y_clip_count_high",
        "xB_clip_count_low", "xB_clip_count_high", "segment_total_phi_clip_count",
        "segment_total_xB_clip_count", "segment_total_Y_clip_count", "nan_count", "inf_count",
        "beta_volume_fraction", "connected_particle_count", "equivalent_radius_min_nm",
        "equivalent_radius_mean_nm", "S_v_equivalent_sphere_nm_inverse",
        "S_v_threshold_faces_nm_inverse", "immediate_component_loss_from_R0",
        "component_disappearance_candidate_count", "periodic_component_status",
        "chemical_excess_hat", "gradient_hat", "barrier_hat", "elastic_hat", "total_excess_hat",
        "GP_inventory_mol", "beta_subgrid_inventory_mol", "GP_PSD_checksum",
        "beta_subgrid_PSD_checksum", "auxiliary_frozen",
    )
    with trajectory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trajectory_columns, lineterminator="\n")
        writer.writeheader()
        for record in sorted(observations, key=lambda value: (str(value["case"]), int(value["accepted_step"]))):
            runtime = _mapping(record.get("runtime_diagnostics"), "trajectory runtime diagnostics")
            fields = _mapping(runtime.get("local_fields"), "trajectory local fields")
            chemical = _mapping(runtime.get("chemical_potential"), "trajectory chemical potential")
            clipping = _mapping(runtime.get("clipping"), "trajectory clipping")
            micro = _mapping(runtime.get("microstructure"), "trajectory microstructure")
            energy = _mapping(runtime.get("energy"), "trajectory energy")
            auxiliary = _mapping(runtime.get("auxiliary"), "trajectory auxiliary")
            inventory = _mapping(record["inventory_mol"], "trajectory inventory")
            header = _mapping(record["header"], "trajectory checkpoint header")
            writer.writerow(
                {
                    "case": record["case"],
                    "step": record["accepted_step"],
                    "time_h": runtime["time_h"],
                    "diagnostic_provenance": runtime["provenance"],
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "validation_contract_hash": header["validation_contract_hash"],
                    "fixture_manifest_sha256": header["fixture_manifest_sha256"],
                    "package_handoff_hash": header["package_handoff_hash"],
                    "Q_B_total_mol": inventory["Q_B_total_mol"],
                    "Q_B_matrix_mol": inventory["Q_B_matrix_mol"],
                    "Q_B_GP_mol": inventory["Q_B_GP_mol"],
                    "Q_B_beta_subgrid_mol": inventory["Q_B_beta_subgrid_mol"],
                    "Q_B_beta_resolved_fixed_mol": inventory["Q_B_beta_resolved_fixed_mol"],
                    "four_bucket_relative_residual": inventory["relative_error_vs_case_t0"],
                    "xB_alpha_mean": fields["xB_alpha_mean"],
                    "xB_alpha_min": fields["xB_alpha_min"],
                    "xB_alpha_max": fields["xB_alpha_max"],
                    "phi_min": fields["phi_min"],
                    "phi_max": fields["phi_max"],
                    "matrix_xB_h_lt_0p005_mean": fields["matrix_xB_h_lt_0p005_mean"],
                    "matrix_xB_h_lt_0p005_min": fields["matrix_xB_h_lt_0p005_min"],
                    "matrix_xB_h_lt_0p005_max": fields["matrix_xB_h_lt_0p005_max"],
                    "chemical_potential_mean": chemical["mean"],
                    "chemical_potential_min": chemical["min"],
                    "chemical_potential_max": chemical["max"],
                    "phi_clip_count_low": clipping["phi_clip_count_low"],
                    "phi_clip_count_high": clipping["phi_clip_count_high"],
                    "Y_clip_count_low": clipping["Y_clip_count_low"],
                    "Y_clip_count_high": clipping["Y_clip_count_high"],
                    "xB_clip_count_low": clipping["xB_clip_count_low"],
                    "xB_clip_count_high": clipping["xB_clip_count_high"],
                    "segment_total_phi_clip_count": clipping["segment_total_phi_clip_count"],
                    "segment_total_xB_clip_count": clipping["segment_total_xB_clip_count"],
                    "segment_total_Y_clip_count": clipping["segment_total_Y_clip_count"],
                    "nan_count": fields["nan_count"],
                    "inf_count": fields["inf_count"],
                    "beta_volume_fraction": micro["beta_volume_fraction"],
                    "connected_particle_count": micro["connected_particle_count"],
                    "equivalent_radius_min_nm": micro["equivalent_radius_min_nm"],
                    "equivalent_radius_mean_nm": micro["equivalent_radius_mean_nm"],
                    "S_v_equivalent_sphere_nm_inverse": micro["S_v_equivalent_sphere_nm_inverse"],
                    "S_v_threshold_faces_nm_inverse": micro["S_v_threshold_faces_nm_inverse"],
                    "immediate_component_loss_from_R0": micro["immediate_component_loss_from_R0"],
                    "component_disappearance_candidate_count": micro["component_disappearance_candidate_count"],
                    "periodic_component_status": micro["periodic_component_status"],
                    "chemical_excess_hat": energy["chemical_excess_hat"],
                    "gradient_hat": energy["gradient_hat"],
                    "barrier_hat": energy["barrier_hat"],
                    "elastic_hat": energy["elastic_hat"],
                    "total_excess_hat": energy["total_excess_hat"],
                    "GP_inventory_mol": auxiliary["GP_inventory_mol"],
                    "beta_subgrid_inventory_mol": auxiliary["beta_subgrid_inventory_mol"],
                    "GP_PSD_checksum": auxiliary["GP_PSD_checksum"],
                    "beta_subgrid_PSD_checksum": auxiliary["beta_subgrid_PSD_checksum"],
                    "auxiliary_frozen": auxiliary["frozen"],
                }
            )

    component_path = out_dir / "cuda_component_history.csv"
    component_columns = (
        "case", "step", "time_h", "particle_id", "previous_step", "dense_component_label",
        "component_cell_count", "h_volume_nm3", "equivalent_radius_nm", "centroid_x_nm",
        "centroid_y_nm", "centroid_z_nm", "wraps_periodic_x", "wraps_periodic_y",
        "wraps_periodic_z", "overlap_parent_ids", "overlap_parent_count", "identity_event",
        "unresolved_merge", "unresolved_split", "unresolved_new_component",
    )
    with component_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=component_columns, lineterminator="\n")
        writer.writeheader()
        for row in sorted(component_history, key=lambda value: (str(value["case"]), int(value["step"]), int(value["particle_id"]))):
            writer.writerow({key: row.get(key) for key in component_columns})

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
