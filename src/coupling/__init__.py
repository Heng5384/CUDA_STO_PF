"""Inventory-conserving KWN--PF snapshot coupling primitives.

The v1 package is intentionally one-way and offline.  It records KWN's four
solute buckets exactly, validates them before any PF plan is built, and refuses
to hide GP or sub-grid beta inventory in the PF matrix field.
"""

from .handoff_audit import (
    PASS_HANDOFF_LEDGER_CLOSED,
    HandoffAudit,
    audit_handoff,
    write_handoff_audit,
)
from .kwn_pf_schema import (
    ARRAYS_FILENAME,
    DEFAULT_MASS_RELATIVE_TOLERANCE,
    METADATA_FILENAME,
    SCHEMA_VERSION,
    HandoffPackage,
    HandoffPaths,
    HandoffSchemaError,
    SchemaValidationError,
    ValidationReport,
    assert_roundtrip_equivalent,
    build_handoff_package,
    make_handoff_package,
    read_handoff_package,
    validate_handoff_package,
    write_handoff_package,
)
from .kwn_to_pf import (
    CURRENT_PF_CAPABILITIES,
    FAIL_PF_COUPLING_CONTRACT,
    PARTIAL_PF_STATE_NOT_CLOSED,
    READY_FOR_PF_HANDOFF,
    KwnHandoffExportError,
    PFStateNotClosedError,
    PfCapabilities,
    PfInitializationPlan,
    assess_current_pf_state,
    adapt_kwn_to_pf,
    build_handoff_from_kwn_solver,
    build_pf_initialization_plan,
)
from .pf_to_kwn import (
    PfResolvedSnapshot,
    build_handoff_from_pf_snapshot,
    pf_snapshot_to_handoff,
)
from .spatial_sampler import (
    PF_RESOLUTION_MISMATCH,
    SampledResolvedBetaGeometry,
    SpatialSamplingError,
    integerize_expected_counts,
    periodic_distance_m,
    sample_resolved_beta_geometry,
    validate_periodic_nonoverlap,
)

__all__ = [
    "ARRAYS_FILENAME",
    "DEFAULT_MASS_RELATIVE_TOLERANCE",
    "METADATA_FILENAME",
    "SCHEMA_VERSION",
    "HandoffPackage",
    "HandoffPaths",
    "HandoffSchemaError",
    "SchemaValidationError",
    "ValidationReport",
    "assert_roundtrip_equivalent",
    "build_handoff_package",
    "make_handoff_package",
    "read_handoff_package",
    "validate_handoff_package",
    "write_handoff_package",
    "CURRENT_PF_CAPABILITIES",
    "FAIL_PF_COUPLING_CONTRACT",
    "PARTIAL_PF_STATE_NOT_CLOSED",
    "READY_FOR_PF_HANDOFF",
    "KwnHandoffExportError",
    "PFStateNotClosedError",
    "PfCapabilities",
    "PfInitializationPlan",
    "assess_current_pf_state",
    "adapt_kwn_to_pf",
    "build_handoff_from_kwn_solver",
    "build_pf_initialization_plan",
    "PfResolvedSnapshot",
    "build_handoff_from_pf_snapshot",
    "pf_snapshot_to_handoff",
    "PASS_HANDOFF_LEDGER_CLOSED",
    "HandoffAudit",
    "audit_handoff",
    "write_handoff_audit",
    "PF_RESOLUTION_MISMATCH",
    "SampledResolvedBetaGeometry",
    "SpatialSamplingError",
    "integerize_expected_counts",
    "periodic_distance_m",
    "sample_resolved_beta_geometry",
    "validate_periodic_nonoverlap",
]
