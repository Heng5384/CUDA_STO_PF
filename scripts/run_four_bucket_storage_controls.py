#!/usr/bin/env python3
"""Run host-only four-bucket storage controls on the frozen 96^3 fixture.

The controls deliberately do *not* invoke CUDA or change a PF field.  They
reconstruct the qualified six-particle fixture from SHA-validated host profile
fields, exercise the v2 storage adapter in memory, and write a compact ledger
under ``outputs/kwn_pf_state_closure_v1``.  GP and sub-grid beta are frozen
bookkeeping populations here: they do not enter chemical potentials, eta,
seeding, release, or an online KWN call.

The optional C++ checkpoint invocation is also host-only.  It detects a
failure to serialize/recover V5 auxiliary storage provenance (or to keep
V2--V4 backward reads), rather than proving a CUDA PF restart.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coupling.fixture_conditioned_handoff_v2 import (  # noqa: E402
    FixtureConditionedHandoffError,
    FrozenPopulation,
    build_fixture_conditioned_handoff_v2,
    field_matrix_inventory_mol,
    field_resolved_inventory_mol,
    load_host_96cube_fixture,
    load_validation_contract,
    map_matrix_inventory_preserving_fixture,
    prescribed_population_from_inventory,
    reconstruct_matrix_xb_alpha,
    validate_fixture_conditioned_handoff_v2,
)


OUTPUT_ROOT = ROOT / "outputs" / "kwn_pf_state_closure_v1"
DEFAULT_HOST_PROFILE_ROOT = (
    ROOT.parent
    / "CUDA_STO_PF"
    / "data"
    / "qualification"
    / "pf_elastic_target_profile_quarter_nm_v2"
    / "profiles"
)
LEDGER_FILENAME = "four_bucket_test_ledger.csv"
SUMMARY_FILENAME = "four_bucket_storage_control_summary.json"
HOST_SCOPE = "HOST_STORAGE_CONTROL_NOT_CUDA"
RELATIVE_TOLERANCE = 1.0e-10


class StorageControlError(RuntimeError):
    """Raised if a claimed host-only storage control does not close."""


def _inside_output_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(OUTPUT_ROOT.resolve())
        return True
    except ValueError:
        return False


def _relative_error(lhs: float, rhs: float) -> float:
    return abs(float(lhs) - float(rhs)) / max(abs(float(rhs)), 1.0e-300)


def _require_close(label: str, lhs: float, rhs: float, *, tolerance: float = RELATIVE_TOLERANCE) -> float:
    residual = _relative_error(lhs, rhs)
    if residual > tolerance:
        raise StorageControlError(
            f"{label} does not close: relative residual {residual:.17g} > {tolerance:.17g}"
        )
    return residual


def _max_abs_difference(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(lhs, dtype=np.float64) - np.asarray(rhs, dtype=np.float64))))


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _field_snapshot(fixture: Any, contract: Any) -> Dict[str, Any]:
    return {
        "phi": np.array(fixture.phi, dtype=np.float64, copy=True),
        "xB_alpha": np.array(fixture.xB_alpha, dtype=np.float64, copy=True),
        "resolved_inventory_mol": field_resolved_inventory_mol(
            fixture.h_phi,
            contract.v_B,
            fixture.voxel_volume_m3,
            contract.vm_beta_m3_mol,
        ),
    }


def _field_preservation_report(fixture: Any, contract: Any, before: Mapping[str, Any]) -> Dict[str, Any]:
    phi_max = _max_abs_difference(before["phi"], fixture.phi)
    x_b_max = _max_abs_difference(before["xB_alpha"], fixture.xB_alpha)
    resolved_after = field_resolved_inventory_mol(
        fixture.h_phi,
        contract.v_B,
        fixture.voxel_volume_m3,
        contract.vm_beta_m3_mol,
    )
    resolved_residual = _require_close(
        "resolved beta inventory after storage control",
        resolved_after,
        float(before["resolved_inventory_mol"]),
    )
    if phi_max != 0.0 or x_b_max != 0.0:
        raise StorageControlError(
            "host storage control mutated a frozen source field in memory"
        )
    return {
        "source_phi_max_abs_difference": phi_max,
        "source_xB_alpha_max_abs_difference": x_b_max,
        "source_phi_sha256_before": _array_sha256(before["phi"]),
        "source_phi_sha256_after": _array_sha256(fixture.phi),
        "source_xB_alpha_sha256_before": _array_sha256(before["xB_alpha"]),
        "source_xB_alpha_sha256_after": _array_sha256(fixture.xB_alpha),
        "resolved_inventory_relative_residual": resolved_residual,
    }


def _ledger_value(ledger: Mapping[str, Any], bucket: str) -> float:
    key = {
        "matrix": "Q_B_matrix_mol",
        "GP": "Q_B_GP_mol",
        "beta_subgrid": "Q_B_beta_subgrid_mol",
        "beta_resolved_fixed": "Q_B_beta_resolved_fixed_mol",
        "total": "Q_B_total_mol",
    }[bucket]
    return float(ledger[key])


def _ledger_rows(
    scenario: str,
    ledger: Mapping[str, Any],
    *,
    source_preservation: Mapping[str, Any],
    mapped_matrix_max_delta_xb: float,
    frozen: bool,
    status: str,
    extra: Mapping[str, Any] | None = None,
) -> Iterable[Dict[str, Any]]:
    box_volume = float(ledger["box_volume_m3"])
    residual = float(ledger["relative_residual"])
    extras = dict(extra or {})
    for bucket in ("matrix", "GP", "beta_subgrid", "beta_resolved_fixed", "total"):
        quantity = _ledger_value(ledger, bucket)
        yield {
            "scenario": scenario,
            "bucket": bucket,
            "Q_B_mol": quantity,
            "C_B_mol_m3": quantity / box_volume,
            "ledger_relative_residual": residual,
            "source_phi_max_abs_difference": source_preservation["source_phi_max_abs_difference"],
            "source_xB_alpha_max_abs_difference": source_preservation["source_xB_alpha_max_abs_difference"],
            "mapped_matrix_max_abs_delta_xB": mapped_matrix_max_delta_xb,
            "frozen_auxiliary": frozen,
            "execution_scope": HOST_SCOPE,
            "status": status,
            "transfer_delta_Q_mol": extras.get("transfer_delta_Q_mol", ""),
            "no_clipping": extras.get("no_clipping", ""),
            "max_xB_roundtrip_error": extras.get("max_xB_roundtrip_error", ""),
        }


def _four_bucket_ledger(
    *,
    q_total: float,
    q_matrix: float,
    q_gp: float,
    q_subgrid: float,
    q_resolved: float,
    box_volume_m3: float,
) -> Dict[str, float]:
    bucket_sum = q_matrix + q_gp + q_subgrid + q_resolved
    residual = q_total - bucket_sum
    return {
        "box_volume_m3": box_volume_m3,
        "Q_B_total_mol": q_total,
        "Q_B_matrix_mol": q_matrix,
        "Q_B_GP_mol": q_gp,
        "Q_B_beta_subgrid_mol": q_subgrid,
        "Q_B_beta_resolved_fixed_mol": q_resolved,
        "Q_B_bucket_sum_mol": bucket_sum,
        "residual_mol": residual,
        "relative_residual": abs(residual) / max(abs(q_total), 1.0e-300),
    }


def _matrix_inventory_bounds(fixture: Any, contract: Any) -> Dict[str, float]:
    """Return exact feasible inventory limits for the unclipped inverse map."""

    alpha = fixture.alpha
    delta_over_alpha = fixture.delta_C_relaxation / alpha
    support = float(np.sum(alpha, dtype=np.float64))
    relaxation = float(np.sum(fixture.delta_C_relaxation, dtype=np.float64))
    baseline_lower = float(np.max(-delta_over_alpha))
    baseline_upper = float(np.min(1.0 - delta_over_alpha))
    scale = fixture.voxel_volume_m3 / contract.vm_alpha_m3_mol
    q_min = (baseline_lower * support + relaxation) * scale
    q_max = (baseline_upper * support + relaxation) * scale
    source = fixture.source_matrix_inventory_mol
    if not q_min <= source <= q_max:
        raise StorageControlError("source matrix inventory lies outside inverse-map bounds")
    return {
        "xB_lower_bound": 0.0,
        "xB_upper_bound": 1.0,
        "baseline_lower": baseline_lower,
        "baseline_upper": baseline_upper,
        "Q_B_matrix_min_feasible_mol": q_min,
        "Q_B_matrix_max_feasible_mol": q_max,
        "Q_B_transfer_capacity_to_GP_mol": source - q_min,
        "Q_B_transfer_capacity_from_GP_mol": q_max - source,
        "matrix_support_storage": support,
    }


def _run_cpp_v5_checkpoint_test() -> Dict[str, Any]:
    """Run the host C++ persistence target; it is explicitly not a CUDA run."""

    command: Sequence[str] = ("make", "test_pf_zero_mode_checkpoint")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    marker = "PASS_PF_ZERO_MODE_CHECKPOINT_PROVENANCE_V2_TO_V5_AUX"
    output = completed.stdout + completed.stderr
    if completed.returncode != 0 or marker not in output:
        raise StorageControlError(
            "C++ V5 checkpoint host test failed; do not claim auxiliary persistence"
        )
    return {
        "status": marker,
        "execution_scope": "HOST_CHECKPOINT_PERSISTENCE_NOT_CUDA",
        "command": list(command),
        "return_code": completed.returncode,
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "cuda_pf_dynamics": "NOT_RUN",
        "meaning": (
            "V5 host checkpoint serialization/recovery and V2-V4 backward reads; "
            "this is not a CUDA PF restart smoke"
        ),
    }


def run_controls(
    *,
    contract_path: Path,
    fixture_spec_path: Path,
    profile_root: Path,
    gp_fraction: float,
    subgrid_fraction: float,
    transfer_fraction: float,
    run_cpp_v5_test: bool,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    """Execute all static controls without changing the source fixture arrays."""

    if not 0.0 < gp_fraction < 1.0:
        raise StorageControlError("gp fraction must be in (0, 1)")
    if not 0.0 < subgrid_fraction < 1.0 or gp_fraction + subgrid_fraction >= 1.0:
        raise StorageControlError("subgrid fraction must be positive and leave matrix inventory")
    if not 0.0 < transfer_fraction < 1.0:
        raise StorageControlError("transfer fraction must be in (0, 1)")

    contract = load_validation_contract(contract_path)
    fixture = load_host_96cube_fixture(fixture_spec_path, profile_root, contract)
    source_before = _field_snapshot(fixture, contract)
    records: list[Dict[str, Any]] = []

    # S1: zero auxiliary inventory is the identity adapter.  Both storage and
    # source fields must agree at machine precision.
    zero_metadata, zero_arrays, _ = build_fixture_conditioned_handoff_v2(
        fixture,
        contract,
        gp_fraction_of_source_matrix=0.0,
        beta_subgrid_fraction_of_source_matrix=0.0,
    )
    zero_report = validate_fixture_conditioned_handoff_v2(
        zero_metadata, zero_arrays, fixture, contract
    )
    zero_mapped_xb = reconstruct_matrix_xb_alpha(fixture, zero_arrays)
    zero_xb_difference = _max_abs_difference(zero_mapped_xb, fixture.xB_alpha)
    if zero_xb_difference > 2.0e-12:
        raise StorageControlError(
            f"S1 zero-aux identity map differs by {zero_xb_difference:.17g}"
        )
    source_after_s1 = _field_preservation_report(fixture, contract, source_before)
    records.extend(
        _ledger_rows(
            "S1_ZERO_AUX_IDENTITY",
            zero_report["ledger"],
            source_preservation=source_after_s1,
            mapped_matrix_max_delta_xb=zero_xb_difference,
            frozen=True,
            status="PASS_S1_ZERO_AUX_IDENTITY_HOST_STORAGE_CONTROL_NOT_CUDA",
        )
    )

    # S2: nonzero frozen GP + sub-grid beta can be represented alongside the
    # intact fixture without changing a local PF field.  This is the direct
    # storage-only control: its full total increases by the auxiliary
    # inventory.  It is distinct from the following fixture-conditioned map,
    # which deliberately rebalances matrix composition to keep the original
    # source total fixed.
    storage_only_gp: FrozenPopulation = prescribed_population_from_inventory(
        name="GP_storage_only_control",
        target_inventory_mol=fixture.source_matrix_inventory_mol * gp_fraction,
        box_volume_m3=fixture.box_volume_m3,
        radius_bin_edges_m=(0.75e-9, 1.25e-9, 1.75e-9),
        volume_weights=(0.45, 0.55),
        x_b=0.03,
        vm_m3_mol=contract.vm_alpha_m3_mol,
        provenance={
            "kind": "FROZEN_STORAGE_ONLY_HOST_CONTROL",
            "execution_scope": HOST_SCOPE,
        },
    )
    storage_only_subgrid: FrozenPopulation = prescribed_population_from_inventory(
        name="beta_subgrid_storage_only_control",
        target_inventory_mol=fixture.source_matrix_inventory_mol * subgrid_fraction,
        box_volume_m3=fixture.box_volume_m3,
        radius_bin_edges_m=(1.75e-9, 2.25e-9, 2.75e-9),
        volume_weights=(0.5, 0.5),
        x_b=1.0,
        vm_m3_mol=contract.vm_beta_m3_mol,
        provenance={
            "kind": "FROZEN_STORAGE_ONLY_HOST_CONTROL",
            "execution_scope": HOST_SCOPE,
        },
    )
    q_storage_gp = storage_only_gp.inventory_mol(fixture.box_volume_m3)
    q_storage_subgrid = storage_only_subgrid.inventory_mol(fixture.box_volume_m3)
    q_storage_total = fixture.source_total_inventory_mol + q_storage_gp + q_storage_subgrid
    s2_storage_ledger = _four_bucket_ledger(
        q_total=q_storage_total,
        q_matrix=fixture.source_matrix_inventory_mol,
        q_gp=q_storage_gp,
        q_subgrid=q_storage_subgrid,
        q_resolved=fixture.source_resolved_inventory_mol,
        box_volume_m3=fixture.box_volume_m3,
    )
    if float(s2_storage_ledger["relative_residual"]) > RELATIVE_TOLERANCE:
        raise StorageControlError("S2 storage-only four-bucket ledger does not close")
    total_increase = q_storage_total - fixture.source_total_inventory_mol
    total_increase_residual = _require_close(
        "S2 storage-only total increase", total_increase, q_storage_gp + q_storage_subgrid
    )
    local_matrix_residual = _require_close(
        "S2 unchanged local matrix inventory",
        field_matrix_inventory_mol(
            fixture.alpha,
            fixture.xB_alpha,
            fixture.voxel_volume_m3,
            contract.vm_alpha_m3_mol,
        ),
        fixture.source_matrix_inventory_mol,
    )
    local_resolved_residual = _require_close(
        "S2 unchanged local resolved inventory",
        field_resolved_inventory_mol(
            fixture.h_phi,
            contract.v_B,
            fixture.voxel_volume_m3,
            contract.vm_beta_m3_mol,
        ),
        fixture.source_resolved_inventory_mol,
    )
    source_after_s2_storage = _field_preservation_report(fixture, contract, source_before)
    records.extend(
        _ledger_rows(
            "S2_NONZERO_FROZEN_AUX_STORAGE_ONLY",
            s2_storage_ledger,
            source_preservation=source_after_s2_storage,
            mapped_matrix_max_delta_xb=0.0,
            frozen=True,
            status="PASS_S2_STORAGE_ONLY_AUX_TOTAL_INCREASE_HOST_STORAGE_CONTROL_NOT_CUDA",
        )
    )

    # S2E: this is the existing fixture-conditioned package mode.  It is not
    # the local-field identity control above: it deliberately returns a new
    # *compact target matrix baseline* so that its package total stays equal to
    # the frozen source total.  Source arrays remain immutable either way.
    nonzero_metadata, nonzero_arrays, _ = build_fixture_conditioned_handoff_v2(
        fixture,
        contract,
        gp_fraction_of_source_matrix=gp_fraction,
        beta_subgrid_fraction_of_source_matrix=subgrid_fraction,
        gp_x_b=0.03,
        beta_subgrid_x_b=1.0,
    )
    nonzero_report = validate_fixture_conditioned_handoff_v2(
        nonzero_metadata, nonzero_arrays, fixture, contract
    )
    nonzero_ledger = nonzero_report["ledger"]
    if float(nonzero_ledger["Q_B_GP_mol"]) <= 0.0 or float(nonzero_ledger["Q_B_beta_subgrid_mol"]) <= 0.0:
        raise StorageControlError("S2 requested nonzero GP and sub-grid beta did not survive")
    if nonzero_metadata["auxiliary_state"]["frozen"] is not True:
        raise StorageControlError("S2 auxiliary state is not frozen")
    if nonzero_metadata["pf_raw_initialization_emitted"] is not False:
        raise StorageControlError("S2 unexpectedly emitted a PF raw initialization")
    source_after_s2 = _field_preservation_report(fixture, contract, source_before)
    nonzero_mapped_xb = reconstruct_matrix_xb_alpha(fixture, nonzero_arrays)
    nonzero_map_difference = _max_abs_difference(nonzero_mapped_xb, fixture.xB_alpha)
    records.extend(
        _ledger_rows(
            "S2E_FIXTURE_CONDITIONED_REBALANCED_AUX",
            nonzero_ledger,
            source_preservation=source_after_s2,
            mapped_matrix_max_delta_xb=nonzero_map_difference,
            frozen=True,
            status="PASS_S2E_REBALANCED_FIXTURE_CONDITIONED_HOST_STORAGE_CONTROL_NOT_CUDA",
        )
    )

    # S3: use the same exact inverse map to move inventory from matrix to a
    # compact GP storage PSD, then re-map it back to the original matrix.  No
    # clip or PF update is permitted on either leg.
    q_matrix_source = fixture.source_matrix_inventory_mol
    q_transfer = q_matrix_source * transfer_fraction
    bounds = _matrix_inventory_bounds(fixture, contract)
    if q_transfer > bounds["Q_B_transfer_capacity_to_GP_mol"]:
        raise StorageControlError("requested S3 transfer exceeds matrix-to-GP capacity")
    x_b_forward, forward_audit = map_matrix_inventory_preserving_fixture(
        fixture, contract, q_matrix_source - q_transfer
    )
    transfer_gp: FrozenPopulation = prescribed_population_from_inventory(
        name="GP_matrix_inverse_transfer_control",
        target_inventory_mol=q_transfer,
        box_volume_m3=fixture.box_volume_m3,
        radius_bin_edges_m=(0.75e-9, 1.25e-9, 1.75e-9),
        volume_weights=(0.45, 0.55),
        x_b=0.03,
        vm_m3_mol=contract.vm_alpha_m3_mol,
        provenance={
            "kind": "EXACT_MATRIX_TO_GP_INVERSE_TRANSFER_HOST_CONTROL",
            "execution_scope": HOST_SCOPE,
        },
    )
    q_gp_transfer = transfer_gp.inventory_mol(fixture.box_volume_m3)
    q_matrix_forward = field_matrix_inventory_mol(
        fixture.alpha,
        x_b_forward,
        fixture.voxel_volume_m3,
        contract.vm_alpha_m3_mol,
    )
    transfer_residual = _require_close(
        "S3 matrix-to-GP transfer", q_matrix_source - q_matrix_forward, q_gp_transfer
    )
    x_b_reverse, reverse_audit = map_matrix_inventory_preserving_fixture(
        fixture, contract, q_matrix_source
    )
    q_matrix_reverse = field_matrix_inventory_mol(
        fixture.alpha,
        x_b_reverse,
        fixture.voxel_volume_m3,
        contract.vm_alpha_m3_mol,
    )
    reverse_residual = _require_close(
        "S3 reverse matrix inventory", q_matrix_reverse, q_matrix_source
    )
    roundtrip_error = _max_abs_difference(x_b_reverse, fixture.xB_alpha)
    if roundtrip_error > 2.0e-12:
        raise StorageControlError(
            f"S3 matrix inverse round trip differs by {roundtrip_error:.17g}"
        )
    if forward_audit["clipping_used"] != 0.0 or reverse_audit["clipping_used"] != 0.0:
        raise StorageControlError("S3 inverse map reported clipping")
    source_after_s3 = _field_preservation_report(fixture, contract, source_before)
    s3_ledger = _four_bucket_ledger(
        q_total=fixture.source_total_inventory_mol,
        q_matrix=q_matrix_forward,
        q_gp=q_gp_transfer,
        q_subgrid=0.0,
        q_resolved=fixture.source_resolved_inventory_mol,
        box_volume_m3=fixture.box_volume_m3,
    )
    if float(s3_ledger["relative_residual"]) > RELATIVE_TOLERANCE:
        raise StorageControlError("S3 four-bucket transfer ledger does not close")
    records.extend(
        _ledger_rows(
            "S3_MATRIX_TO_GP_AND_REVERSE",
            s3_ledger,
            source_preservation=source_after_s3,
            mapped_matrix_max_delta_xb=_max_abs_difference(x_b_forward, fixture.xB_alpha),
            frozen=True,
            status="PASS_S3_EXACT_INVERSE_TRANSFER_HOST_STORAGE_CONTROL_NOT_CUDA",
            extra={
                "transfer_delta_Q_mol": q_transfer,
                "no_clipping": True,
                "max_xB_roundtrip_error": roundtrip_error,
            },
        )
    )

    s4 = (
        _run_cpp_v5_checkpoint_test()
        if run_cpp_v5_test
        else {
            "status": "NOT_RUN_CPP_V5_HOST_TEST",
            "execution_scope": "HOST_CHECKPOINT_PERSISTENCE_NOT_CUDA",
            "cuda_pf_dynamics": "NOT_RUN",
            "meaning": "persistence claim remains dependent on make test_pf_zero_mode_checkpoint",
        }
    )
    records.append(
        {
            "scenario": "S4_CPP_V5_CHECKPOINT_DEPENDENCY",
            "bucket": "checkpoint_host_persistence",
            "Q_B_mol": "",
            "C_B_mol_m3": "",
            "ledger_relative_residual": "",
            "source_phi_max_abs_difference": source_after_s3["source_phi_max_abs_difference"],
            "source_xB_alpha_max_abs_difference": source_after_s3["source_xB_alpha_max_abs_difference"],
            "mapped_matrix_max_abs_delta_xB": "",
            "frozen_auxiliary": True,
            "execution_scope": HOST_SCOPE,
            "status": s4["status"],
            "transfer_delta_Q_mol": "",
            "no_clipping": "",
            "max_xB_roundtrip_error": "",
        }
    )

    summary: Dict[str, Any] = {
        "schema_version": "PF_KWN_FOUR_BUCKET_HOST_STORAGE_CONTROL_V1",
        "status": "PASS_FOUR_BUCKET_HOST_STORAGE_CONTROL_NOT_CUDA",
        "execution_scope": HOST_SCOPE,
        "cuda_pf_dynamics": "NOT_RUN",
        "historical_as_run_claim": False,
        "contract_hash": contract.contract_hash,
        "fixture": {
            "fixture_id": fixture.fixture_id,
            "fixture_hash": fixture.fixture_hash,
            "source_kind": fixture.source_kind,
            "source_total_inventory_mol": fixture.source_total_inventory_mol,
            "source_matrix_inventory_mol": fixture.source_matrix_inventory_mol,
            "source_resolved_inventory_mol": fixture.source_resolved_inventory_mol,
            "box_volume_m3": fixture.box_volume_m3,
        },
        "S1_zero_aux_identity": {
            "status": "PASS",
            "ledger": zero_report["ledger"],
            "max_mapped_matrix_xB_difference_from_source": zero_xb_difference,
            "source_field_preservation": source_after_s1,
        },
        "S2_nonzero_frozen_aux_storage_only": {
            "status": "PASS",
            "ledger": s2_storage_ledger,
            "gp_fraction_of_source_matrix": gp_fraction,
            "subgrid_fraction_of_source_matrix": subgrid_fraction,
            "total_increase_mol": total_increase,
            "total_increase_relative_residual": total_increase_residual,
            "local_matrix_inventory_mol": fixture.source_matrix_inventory_mol,
            "local_resolved_inventory_mol": fixture.source_resolved_inventory_mol,
            "local_matrix_inventory_relative_residual": local_matrix_residual,
            "local_resolved_inventory_relative_residual": local_resolved_residual,
            "mapped_matrix_max_xB_difference_from_source": 0.0,
            "source_field_preservation": source_after_s2_storage,
            "dynamics_excluded": [
                "chemical_potential",
                "phi",
                "GP_release",
                "GP_to_beta_conversion",
                "new_beta_seed",
                "online_KWN_call",
            ],
            "pf_raw_initialization_emitted": False,
            "matrix_rebalanced": False,
        },
        "S2E_fixture_conditioned_rebalanced_aux": {
            "status": "PASS",
            "ledger": nonzero_ledger,
            "gp_fraction_of_source_matrix": gp_fraction,
            "subgrid_fraction_of_source_matrix": subgrid_fraction,
            "mapped_matrix_max_xB_difference_from_source": nonzero_map_difference,
            "source_field_preservation": source_after_s2,
            "dynamics_excluded": nonzero_metadata["auxiliary_state"]["dynamics_excluded"],
            "pf_raw_initialization_emitted": nonzero_metadata["pf_raw_initialization_emitted"],
            "matrix_rebalanced": True,
        },
        "S3_exact_matrix_to_GP_inverse_transfer_and_reverse": {
            "status": "PASS",
            "ledger_forward": s3_ledger,
            "transfer_delta_Q_mol": q_transfer,
            "transfer_relative_residual": transfer_residual,
            "reverse_relative_residual": reverse_residual,
            "max_xB_roundtrip_error": roundtrip_error,
            "forward_matrix_inverse_audit": forward_audit,
            "reverse_matrix_inverse_audit": reverse_audit,
            "matrix_capacity_and_bounds": bounds,
            "source_field_preservation": source_after_s3,
        },
        "S4_cpp_V5_checkpoint_dependency": s4,
        "interpretation": (
            "This proves only hash-validated host storage closure and the separately invoked "
            "host C++ checkpoint contract.  It does not run CUDA, PF dynamics, GP release, "
            "GP-to-beta conversion, or new beta nucleation."
        ),
    }
    return summary, records


def _write_outputs(output_directory: Path, summary: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> Dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    ledger_path = output_directory / LEDGER_FILENAME
    summary_path = output_directory / SUMMARY_FILENAME
    if ledger_path.exists() or summary_path.exists():
        raise StorageControlError(
            f"refusing to overwrite existing storage-control result: {ledger_path} or {summary_path}"
        )
    columns = (
        "scenario",
        "bucket",
        "Q_B_mol",
        "C_B_mol_m3",
        "ledger_relative_residual",
        "source_phi_max_abs_difference",
        "source_xB_alpha_max_abs_difference",
        "mapped_matrix_max_abs_delta_xB",
        "frozen_auxiliary",
        "execution_scope",
        "status",
        "transfer_delta_Q_mol",
        "no_clipping",
        "max_xB_roundtrip_error",
    )
    with ledger_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(records)
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {"ledger": ledger_path, "summary": summary_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "contracts" / "pf_kwn_validation_contract_v1.json",
    )
    parser.add_argument(
        "--fixture-spec",
        type=Path,
        default=ROOT
        / "data"
        / "qualification"
        / "pf_mass_conserving_library_handoff_v1"
        / "six_particle_96cube_spec.json",
    )
    parser.add_argument("--profile-root", type=Path, default=DEFAULT_HOST_PROFILE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--gp-fraction", type=float, default=0.01)
    parser.add_argument("--subgrid-fraction", type=float, default=0.005)
    parser.add_argument("--transfer-fraction", type=float, default=0.01)
    parser.add_argument(
        "--skip-cpp-v5-test",
        action="store_true",
        help="record the host C++ persistence dependency without executing its target",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not _inside_output_root(args.out_dir):
        raise SystemExit(f"refusing output outside validation result root: {args.out_dir}")
    try:
        summary, records = run_controls(
            contract_path=args.contract,
            fixture_spec_path=args.fixture_spec,
            profile_root=args.profile_root,
            gp_fraction=args.gp_fraction,
            subgrid_fraction=args.subgrid_fraction,
            transfer_fraction=args.transfer_fraction,
            run_cpp_v5_test=not args.skip_cpp_v5_test,
        )
        paths = _write_outputs(args.out_dir, summary, records)
    except (FixtureConditionedHandoffError, StorageControlError) as error:
        raise SystemExit(f"four-bucket storage control failed: {error}") from error
    print(
        json.dumps(
            {
                "status": summary["status"],
                "execution_scope": summary["execution_scope"],
                "contract_hash": summary["contract_hash"],
                "S1_identity_xB_max_difference": summary["S1_zero_aux_identity"]["max_mapped_matrix_xB_difference_from_source"],
                "S2_storage_only_four_bucket_relative_residual": summary["S2_nonzero_frozen_aux_storage_only"]["ledger"]["relative_residual"],
                "S2_storage_only_total_increase_mol": summary["S2_nonzero_frozen_aux_storage_only"]["total_increase_mol"],
                "S2E_rebalanced_four_bucket_relative_residual": summary["S2E_fixture_conditioned_rebalanced_aux"]["ledger"]["relative_residual"],
                "S3_roundtrip_xB_max_error": summary["S3_exact_matrix_to_GP_inverse_transfer_and_reverse"]["max_xB_roundtrip_error"],
                "S3_transfer_relative_residual": summary["S3_exact_matrix_to_GP_inverse_transfer_and_reverse"]["transfer_relative_residual"],
                "S4_checkpoint_host_status": summary["S4_cpp_V5_checkpoint_dependency"]["status"],
                "outputs": {name: str(path) for name, path in paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
