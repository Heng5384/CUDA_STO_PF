"""Audit the accounting boundary between a KWN package and a PF handoff plan."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .kwn_pf_schema import (
    DEFAULT_MASS_RELATIVE_TOLERANCE,
    HandoffPackage,
    ValidationReport,
    validate_handoff_package,
)
from .kwn_to_pf import (
    FAIL_PF_COUPLING_CONTRACT,
    PARTIAL_PF_STATE_NOT_CLOSED,
    READY_FOR_PF_HANDOFF,
    PfInitializationPlan,
)


PASS_HANDOFF_LEDGER_CLOSED = "PASS_HANDOFF_LEDGER_CLOSED"


@dataclass(frozen=True)
class HandoffAudit:
    """Machine-readable evidence of inventory destinations at one handoff."""

    status: str
    source_validation: ValidationReport
    source_ledger: Dict[str, float]
    pf_materialized_inventory: Dict[str, float]
    package_only_inventory: Dict[str, float]
    destination_by_bucket: Dict[str, str]
    accounted_total: float
    accounting_residual: float
    relative_accounting_residual: float
    double_counting_detected: bool
    pf_raw_initialization_emitted: bool
    pf_raw_initialization_allowed: bool
    notes: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "source_validation": self.source_validation.as_dict(),
            "source_ledger": dict(self.source_ledger),
            "pf_materialized_inventory": dict(self.pf_materialized_inventory),
            "package_only_inventory": dict(self.package_only_inventory),
            "destination_by_bucket": dict(self.destination_by_bucket),
            "accounted_total": self.accounted_total,
            "accounting_residual": self.accounting_residual,
            "relative_accounting_residual": self.relative_accounting_residual,
            "double_counting_detected": self.double_counting_detected,
            "pf_raw_initialization_emitted": self.pf_raw_initialization_emitted,
            "pf_raw_initialization_allowed": self.pf_raw_initialization_allowed,
            "notes": list(self.notes),
        }


def _is_close(lhs: float, rhs: float, total: float, tolerance: float) -> bool:
    return abs(lhs - rhs) <= tolerance * max(abs(total), 1.0e-300)


def audit_handoff(
    package: HandoffPackage,
    plan: PfInitializationPlan,
    mass_relative_tolerance: float = DEFAULT_MASS_RELATIVE_TOLERANCE,
) -> HandoffAudit:
    """Verify that a plan neither loses nor double-counts any KWN inventory.

    A partial result is intentionally not upgraded to a pass: its mass ledger
    can close because GP/sub-grid inventory remains in the package, but the PF
    state itself is not closed.
    """

    validation = validate_handoff_package(package, mass_relative_tolerance)
    if not isinstance(plan, PfInitializationPlan):
        raise TypeError("plan must be a PfInitializationPlan")
    source = dict(plan.source_ledger)
    ledger = package.metadata["ledger"]
    for key in (
        "C_B_total",
        "C_B_matrix",
        "C_B_GP",
        "C_B_beta_subgrid",
        "C_B_beta_resolved",
        "residual",
    ):
        if key not in source or not _is_close(
            float(source[key]), float(ledger[key]), float(ledger["C_B_total"]), mass_relative_tolerance
        ):
            raise ValueError("handoff plan source_ledger does not match package ledger")

    total = float(source["C_B_total"])
    matrix_payload = plan.materialized_state.get("matrix", {})
    resolved_payload = plan.materialized_state.get("resolved_beta", {})
    materialized = {
        "C_B_matrix": float(matrix_payload.get("B_inventory", 0.0)),
        "C_B_GP": float(
            plan.materialized_state.get("gp_population", {}).get("B_inventory", 0.0)
        ),
        "C_B_beta_subgrid": float(
            plan.materialized_state.get("beta_subgrid_population", {}).get("B_inventory", 0.0)
        ),
        "C_B_beta_resolved": float(resolved_payload.get("B_inventory", 0.0)),
    }
    package_only = {
        "C_B_GP": float(plan.unmapped_inventory.get("C_B_GP", 0.0)),
        "C_B_beta_subgrid": float(plan.unmapped_inventory.get("C_B_beta_subgrid", 0.0)),
    }
    destinations: Dict[str, str] = {
        "C_B_matrix": "PF matrix xB_alpha" if plan.capabilities.supports_matrix_composition else "unmapped",
        "C_B_GP": (
            "PF GP inventory state"
            if plan.capabilities.supports_gp_inventory
            else "package-only retained inventory"
        ),
        "C_B_beta_subgrid": (
            "PF sub-grid beta inventory state"
            if plan.capabilities.supports_beta_subgrid_inventory
            else "package-only retained inventory"
        ),
        "C_B_beta_resolved": (
            "PF resolved-beta geometry"
            if plan.capabilities.supports_resolved_beta_geometry
            else "unmapped"
        ),
    }

    expected_materialized = {
        "C_B_matrix": float(source["C_B_matrix"])
        if plan.capabilities.supports_matrix_composition
        else 0.0,
        "C_B_GP": float(source["C_B_GP"])
        if plan.capabilities.supports_gp_inventory
        else 0.0,
        "C_B_beta_subgrid": float(source["C_B_beta_subgrid"])
        if plan.capabilities.supports_beta_subgrid_inventory
        else 0.0,
        "C_B_beta_resolved": float(source["C_B_beta_resolved"])
        if plan.capabilities.supports_resolved_beta_geometry
        else 0.0,
    }
    double_counting = any(
        not _is_close(materialized[key], expected_materialized[key], total, mass_relative_tolerance)
        for key in materialized
    )
    # GP/sub-grid inventory must have a single, explicit destination when PF
    # cannot represent it.  This catches a future accidental matrix transfer.
    supports_bucket = {
        "C_B_GP": plan.capabilities.supports_gp_inventory,
        "C_B_beta_subgrid": plan.capabilities.supports_beta_subgrid_inventory,
    }
    for key, supported in supports_bucket.items():
        if not supported and not _is_close(
            package_only[key], float(source[key]), total, mass_relative_tolerance
        ):
            double_counting = True

    accounted_total = sum(materialized.values()) + sum(package_only.values())
    accounting_residual = total - accounted_total
    relative_residual = abs(accounting_residual) / max(abs(total), 1.0e-300)
    if double_counting or relative_residual > mass_relative_tolerance:
        status = FAIL_PF_COUPLING_CONTRACT
    elif plan.status == PARTIAL_PF_STATE_NOT_CLOSED:
        status = PARTIAL_PF_STATE_NOT_CLOSED
    elif plan.status == READY_FOR_PF_HANDOFF:
        status = PASS_HANDOFF_LEDGER_CLOSED
    else:
        status = plan.status

    notes = list(plan.notes)
    if status == PARTIAL_PF_STATE_NOT_CLOSED:
        notes.append(
            "Ledger closure is package-level only; no PF raw initialization is emitted while state remains partial."
        )
    if double_counting:
        notes.append("Audit detected a mismatch between source buckets and PF/package destinations.")
    return HandoffAudit(
        status=status,
        source_validation=validation,
        source_ledger=source,
        pf_materialized_inventory=materialized,
        package_only_inventory=package_only,
        destination_by_bucket=destinations,
        accounted_total=accounted_total,
        accounting_residual=accounting_residual,
        relative_accounting_residual=relative_residual,
        double_counting_detected=double_counting,
        # This package deliberately emits a plan only.  A separate qualified
        # PF materializer may use ``pf_raw_initialization_allowed`` later.
        pf_raw_initialization_emitted=False,
        pf_raw_initialization_allowed=plan.pf_state_closed,
        notes=notes,
    )


def write_handoff_audit(audit: HandoffAudit, path: Path) -> Path:
    """Write one deterministic JSON audit record supplied by the caller."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
