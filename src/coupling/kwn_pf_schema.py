"""Versioned, inventory-conserving KWN-to-PF snapshot handoff schema.

The package deliberately stores KWN state as a JSON metadata document plus a
NumPy ``.npz`` array bundle.  The JSON carries provenance and the four
inventory buckets (all on a ``mol_B_per_m3`` basis); the NPZ keeps every radius
distribution free of unit or delimiter ambiguity.  This module does not alter
PF fields or infer a missing PF state variable.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = "kwn_pf_handoff_v1"
METADATA_FILENAME = "metadata.json"
ARRAYS_FILENAME = "arrays.npz"
DEFAULT_MASS_RELATIVE_TOLERANCE = 1.0e-10

LEDGER_KEYS = (
    "C_B_total",
    "C_B_matrix",
    "C_B_GP",
    "C_B_beta_subgrid",
    "C_B_beta_resolved",
    "residual",
)
INVENTORY_BUCKET_KEYS = (
    "C_B_matrix",
    "C_B_GP",
    "C_B_beta_subgrid",
    "C_B_beta_resolved",
)
REQUIRED_ARRAY_KEYS = (
    "gp_radius_bin_edges_m",
    "gp_number_density_per_m4",
    "beta_subgrid_radius_bin_edges_m",
    "beta_subgrid_number_density_per_m4",
    "beta_resolved_radius_bin_edges_m",
    "beta_resolved_expected_count",
)
OPTIONAL_ARRAY_KEYS = (
    "beta_resolved_sampled_radii_m",
    "beta_resolved_centers_m",
)


class SchemaValidationError(ValueError):
    """Raised when a handoff package violates the v1 data contract."""


# The longer historical spelling is kept as a public alias for callers that
# name the error after the artifact rather than the validation operation.
HandoffSchemaError = SchemaValidationError

@dataclass(frozen=True)
class HandoffPaths:
    """The two on-disk files that make up one handoff package."""

    metadata_path: Path
    arrays_path: Path


@dataclass(frozen=True)
class ValidationReport:
    """Small, serializable result of a successful strict schema validation."""

    schema_version: str
    C_B_total: float
    C_B_bucket_sum: float
    residual: float
    relative_residual: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "C_B_total": self.C_B_total,
            "C_B_bucket_sum": self.C_B_bucket_sum,
            "residual": self.residual,
            "relative_residual": self.relative_residual,
        }


@dataclass
class HandoffPackage:
    """In-memory representation of a ``kwn_pf_handoff_v1`` package.

    ``metadata`` must be JSON serializable and ``arrays`` must be numeric NumPy
    arrays.  Validation is explicit so tests and calling code can construct a
    malformed candidate and receive the contract error at a controlled point.
    """

    metadata: Dict[str, Any]
    arrays: Dict[str, np.ndarray]

    def validate(
        self,
        mass_relative_tolerance: float = DEFAULT_MASS_RELATIVE_TOLERANCE,
    ) -> ValidationReport:
        """Validate this package against the exact v1 schema."""

        return validate_handoff_package(self, mass_relative_tolerance)

    def write(
        self,
        directory: Path,
        mass_relative_tolerance: float = DEFAULT_MASS_RELATIVE_TOLERANCE,
    ) -> HandoffPaths:
        """Validate and write ``metadata.json`` plus ``arrays.npz``."""

        return write_handoff_package(self, directory, mass_relative_tolerance)

    @classmethod
    def read(
        cls,
        directory: Path,
        mass_relative_tolerance: float = DEFAULT_MASS_RELATIVE_TOLERANCE,
    ) -> "HandoffPackage":
        """Read and strictly validate an on-disk handoff package."""

        return read_handoff_package(directory, mass_relative_tolerance)


def _fail(message: str) -> None:
    raise SchemaValidationError(message)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    )


def _finite_number(value: Any, field: str) -> float:
    if not _is_number(value):
        _fail(f"{field} must be a finite numeric value")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _nonnegative(value: Any, field: str) -> float:
    result = _finite_number(value, field)
    if result < 0.0:
        _fail(f"{field} must be non-negative")
    return result


def _fraction(value: Any, field: str) -> float:
    result = _finite_number(value, field)
    if not 0.0 <= result <= 1.0:
        _fail(f"{field} must be in [0, 1]")
    return result


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be an object")
    return value


def _require_keys(
    value: Mapping[str, Any],
    required: Iterable[str],
    field: str,
    optional: Iterable[str] = (),
) -> None:
    required_set = set(required)
    allowed_set = required_set | set(optional)
    missing = sorted(required_set - set(value))
    unknown = sorted(set(value) - allowed_set)
    if missing:
        _fail(f"{field} is missing required field(s): {', '.join(missing)}")
    if unknown:
        _fail(f"{field} contains unknown field(s): {', '.join(unknown)}")


def _validate_source(source: Any) -> None:
    source_mapping = _mapping(source, "source")
    _require_keys(
        source_mapping,
        ("git_commit", "config_hash", "kwn_backend", "kwn_backend_version"),
        "source",
        optional=("binary_hash", "fixture_hash", "analysis_hash", "additional_provenance"),
    )
    for key in ("git_commit", "config_hash", "kwn_backend", "kwn_backend_version"):
        if not isinstance(source_mapping[key], str) or not source_mapping[key].strip():
            _fail(f"source.{key} must be a non-empty string")
    for key in ("binary_hash", "fixture_hash", "analysis_hash"):
        if key in source_mapping and (
            not isinstance(source_mapping[key], str) or not source_mapping[key].strip()
        ):
            _fail(f"source.{key} must be a non-empty string when supplied")
    if "additional_provenance" in source_mapping:
        _mapping(source_mapping["additional_provenance"], "source.additional_provenance")


def _validate_pf_box(pf_box: Any) -> Tuple[float, float, float]:
    pf_box_mapping = _mapping(pf_box, "pf_box")
    _require_keys(
        pf_box_mapping,
        ("lengths_m", "periodic"),
        "pf_box",
        optional=("grid_shape",),
    )
    lengths = pf_box_mapping["lengths_m"]
    if not isinstance(lengths, (list, tuple)) or len(lengths) != 3:
        _fail("pf_box.lengths_m must contain exactly three lengths")
    parsed_lengths = tuple(_finite_number(item, "pf_box.lengths_m") for item in lengths)
    if any(item <= 0.0 for item in parsed_lengths):
        _fail("pf_box.lengths_m values must be positive")
    if pf_box_mapping["periodic"] is not True:
        _fail("pf_box.periodic must be true for a periodic PF handoff")
    if "grid_shape" in pf_box_mapping:
        grid_shape = pf_box_mapping["grid_shape"]
        if (
            not isinstance(grid_shape, (list, tuple))
            or len(grid_shape) != 3
            or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in grid_shape)
        ):
            _fail("pf_box.grid_shape must contain exactly three positive integers")
    return parsed_lengths


def _validate_population(
    value: Any,
    field: str,
    composition_key: str,
    composition_exactly_one: bool = False,
) -> Mapping[str, Any]:
    population = _mapping(value, field)
    _require_keys(
        population,
        (composition_key, "Vm_m3_per_mol", "volume_fraction", "B_inventory"),
        field,
        optional=("shape_orientation_metadata",),
    )
    composition = _fraction(population[composition_key], f"{field}.{composition_key}")
    if composition_exactly_one and not math.isclose(composition, 1.0, abs_tol=1.0e-12):
        _fail(f"{field}.{composition_key} must equal 1 for the beta line-compound v1 contract")
    vm = _finite_number(population["Vm_m3_per_mol"], f"{field}.Vm_m3_per_mol")
    if vm <= 0.0:
        _fail(f"{field}.Vm_m3_per_mol must be positive")
    _fraction(population["volume_fraction"], f"{field}.volume_fraction")
    _nonnegative(population["B_inventory"], f"{field}.B_inventory")
    if "shape_orientation_metadata" in population:
        _mapping(population["shape_orientation_metadata"], f"{field}.shape_orientation_metadata")
    return population


def _validate_matrix(value: Any) -> Mapping[str, Any]:
    matrix = _mapping(value, "matrix")
    _require_keys(matrix, ("xB_alpha", "Vm_m3_per_mol", "B_inventory"), "matrix")
    _fraction(matrix["xB_alpha"], "matrix.xB_alpha")
    vm = _finite_number(matrix["Vm_m3_per_mol"], "matrix.Vm_m3_per_mol")
    if vm <= 0.0:
        _fail("matrix.Vm_m3_per_mol must be positive")
    _nonnegative(matrix["B_inventory"], "matrix.B_inventory")
    return matrix


def _validate_ledger(
    value: Any,
    mass_relative_tolerance: float,
) -> Tuple[float, float, float, float, float, float, float]:
    ledger = _mapping(value, "ledger")
    _require_keys(ledger, LEDGER_KEYS, "ledger")
    if mass_relative_tolerance < 0.0 or not math.isfinite(mass_relative_tolerance):
        _fail("mass_relative_tolerance must be finite and non-negative")
    total = _nonnegative(ledger["C_B_total"], "ledger.C_B_total")
    buckets = tuple(_nonnegative(ledger[key], f"ledger.{key}") for key in INVENTORY_BUCKET_KEYS)
    residual = _finite_number(ledger["residual"], "ledger.residual")
    bucket_sum = sum(buckets)
    computed_residual = total - bucket_sum
    scale = max(abs(total), 1.0e-300)
    if abs(residual - computed_residual) > mass_relative_tolerance * scale:
        _fail(
            "ledger.residual does not equal C_B_total minus the four inventory buckets "
            "within the requested mass tolerance"
        )
    if abs(computed_residual) / scale > mass_relative_tolerance:
        _fail(
            "four inventory buckets do not close C_B_total within the requested mass tolerance"
        )
    return (total, buckets[0], buckets[1], buckets[2], buckets[3], residual, bucket_sum)


def _numeric_array(value: Any, field: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        _fail(f"array {field!r} must be a numpy.ndarray")
    if value.dtype.kind not in "fiu":
        _fail(f"array {field!r} must have a real numeric dtype")
    if not np.all(np.isfinite(value)):
        _fail(f"array {field!r} must contain only finite values")
    return value


def _validate_bins(edges: np.ndarray, values: np.ndarray, name: str) -> None:
    if edges.ndim != 1 or edges.size < 2:
        _fail(f"array {name}_radius_bin_edges_m must be one-dimensional with at least two edges")
    if np.any(edges <= 0.0) or np.any(np.diff(edges) <= 0.0):
        _fail(f"array {name}_radius_bin_edges_m must be strictly increasing and positive")
    if values.ndim != 1 or values.size != edges.size - 1:
        _fail(f"array {name} population values must have exactly one entry per radius bin")
    if np.any(values < 0.0):
        _fail(f"array {name} population values must be non-negative")


def _validate_arrays(arrays: Mapping[str, np.ndarray], box_lengths_m: Tuple[float, float, float]) -> None:
    if not isinstance(arrays, Mapping):
        _fail("arrays must be a mapping")
    keys = set(arrays)
    missing = sorted(set(REQUIRED_ARRAY_KEYS) - keys)
    unknown = sorted(keys - set(REQUIRED_ARRAY_KEYS) - set(OPTIONAL_ARRAY_KEYS))
    if missing:
        _fail(f"arrays is missing required array(s): {', '.join(missing)}")
    if unknown:
        _fail(f"arrays contains unknown array(s): {', '.join(unknown)}")

    checked = {key: _numeric_array(arrays[key], key) for key in keys}
    _validate_bins(
        checked["gp_radius_bin_edges_m"],
        checked["gp_number_density_per_m4"],
        "gp",
    )
    _validate_bins(
        checked["beta_subgrid_radius_bin_edges_m"],
        checked["beta_subgrid_number_density_per_m4"],
        "beta_subgrid",
    )
    _validate_bins(
        checked["beta_resolved_radius_bin_edges_m"],
        checked["beta_resolved_expected_count"],
        "beta_resolved",
    )

    sampled_radii = checked.get("beta_resolved_sampled_radii_m")
    centers = checked.get("beta_resolved_centers_m")
    if sampled_radii is not None:
        if sampled_radii.ndim != 1 or np.any(sampled_radii <= 0.0):
            _fail("beta_resolved_sampled_radii_m must be a one-dimensional positive array")
        edges = checked["beta_resolved_radius_bin_edges_m"]
        if np.any(sampled_radii < edges[0]) or np.any(sampled_radii > edges[-1]):
            _fail("beta_resolved_sampled_radii_m values must lie within the declared resolved bins")
    if centers is not None:
        if sampled_radii is None:
            _fail("beta_resolved_centers_m requires beta_resolved_sampled_radii_m")
        if centers.ndim != 2 or centers.shape != (sampled_radii.size, 3):
            _fail("beta_resolved_centers_m must have shape (number of sampled radii, 3)")
        for axis, length in enumerate(box_lengths_m):
            if np.any(centers[:, axis] < 0.0) or np.any(centers[:, axis] >= length):
                _fail("beta_resolved_centers_m must lie inside the periodic PF box")


def _canonical_json_value(value: Any) -> Any:
    """Convert a metadata mapping to the exact JSON value used on disk."""

    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        _fail(f"metadata must be JSON serializable without NaN values: {error}")
    raise AssertionError("_fail always raises")


def validate_handoff_package(
    package: HandoffPackage,
    mass_relative_tolerance: float = DEFAULT_MASS_RELATIVE_TOLERANCE,
) -> ValidationReport:
    """Perform strict version, unit, array, and four-bucket ledger validation."""

    if not isinstance(package, HandoffPackage):
        _fail("package must be a HandoffPackage")
    metadata = _mapping(package.metadata, "metadata")
    _canonical_json_value(metadata)
    _require_keys(
        metadata,
        (
            "schema_version",
            "source",
            "temperature_K",
            "handoff_time_s",
            "pf_box",
            "unit_system",
            "inventory_basis",
            "total_pseudo_binary_inventory",
            "assumptions",
            "observation_dataset_role",
            "matrix",
            "gp_population",
            "beta_subgrid_population",
            "beta_resolved_population",
            "ledger",
        ),
        "metadata",
        optional=("extensions",),
    )
    if metadata["schema_version"] != SCHEMA_VERSION:
        _fail(
            f"metadata.schema_version must be {SCHEMA_VERSION!r}, got {metadata['schema_version']!r}"
        )
    if metadata["unit_system"] != "SI":
        _fail("metadata.unit_system must be exactly 'SI'")
    if metadata["inventory_basis"] != "mol_B_per_m3":
        _fail("metadata.inventory_basis must be exactly 'mol_B_per_m3'")
    _validate_source(metadata["source"])
    _finite_number(metadata["temperature_K"], "metadata.temperature_K")
    if float(metadata["temperature_K"]) <= 0.0:
        _fail("metadata.temperature_K must be positive")
    _nonnegative(metadata["handoff_time_s"], "metadata.handoff_time_s")
    box_lengths_m = _validate_pf_box(metadata["pf_box"])
    total_inventory = _nonnegative(
        metadata["total_pseudo_binary_inventory"],
        "metadata.total_pseudo_binary_inventory",
    )
    if not isinstance(metadata["assumptions"], list) or not all(
        isinstance(item, str) and item.strip() for item in metadata["assumptions"]
    ):
        _fail("metadata.assumptions must be a list of non-empty strings")
    if (
        not isinstance(metadata["observation_dataset_role"], str)
        or not metadata["observation_dataset_role"].strip()
    ):
        _fail("metadata.observation_dataset_role must be a non-empty string")
    if "extensions" in metadata:
        _mapping(metadata["extensions"], "metadata.extensions")

    matrix = _validate_matrix(metadata["matrix"])
    gp = _validate_population(metadata["gp_population"], "gp_population", "xB_g")
    beta_subgrid = _validate_population(
        metadata["beta_subgrid_population"],
        "beta_subgrid_population",
        "xB_beta",
        composition_exactly_one=True,
    )
    beta_resolved = _validate_population(
        metadata["beta_resolved_population"],
        "beta_resolved_population",
        "xB_beta",
        composition_exactly_one=True,
    )
    total_volume_fraction = sum(
        float(population["volume_fraction"])
        for population in (gp, beta_subgrid, beta_resolved)
    )
    if total_volume_fraction > 1.0 + mass_relative_tolerance:
        _fail("GP plus beta-subgrid plus beta-resolved volume fractions exceed one")

    total, matrix_inventory, gp_inventory, subgrid_inventory, resolved_inventory, residual, bucket_sum = _validate_ledger(
        metadata["ledger"], mass_relative_tolerance
    )
    linked = (
        (matrix["B_inventory"], matrix_inventory, "matrix"),
        (gp["B_inventory"], gp_inventory, "gp_population"),
        (beta_subgrid["B_inventory"], subgrid_inventory, "beta_subgrid_population"),
        (beta_resolved["B_inventory"], resolved_inventory, "beta_resolved_population"),
    )
    scale = max(abs(total), 1.0e-300)
    for population_inventory, ledger_inventory, name in linked:
        if abs(float(population_inventory) - ledger_inventory) > mass_relative_tolerance * scale:
            _fail(f"{name}.B_inventory must equal its matching ledger bucket")
    if abs(total_inventory - total) > mass_relative_tolerance * scale:
        _fail("total_pseudo_binary_inventory must equal ledger.C_B_total")
    _validate_arrays(package.arrays, box_lengths_m)

    return ValidationReport(
        schema_version=SCHEMA_VERSION,
        C_B_total=total,
        C_B_bucket_sum=bucket_sum,
        residual=residual,
        relative_residual=abs(total - bucket_sum) / scale,
    )


def write_handoff_package(
    package: HandoffPackage,
    directory: Path,
    mass_relative_tolerance: float = DEFAULT_MASS_RELATIVE_TOLERANCE,
) -> HandoffPaths:
    """Write a validated handoff package to ``directory``.

    The caller owns the output directory.  No source/PF files are modified.
    """

    validate_handoff_package(package, mass_relative_tolerance)
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    metadata_path = target / METADATA_FILENAME
    arrays_path = target / ARRAYS_FILENAME
    canonical_metadata = _canonical_json_value(package.metadata)
    metadata_path.write_text(
        json.dumps(canonical_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez(arrays_path, **{key: np.asarray(value) for key, value in package.arrays.items()})
    return HandoffPaths(metadata_path=metadata_path, arrays_path=arrays_path)


def read_handoff_package(
    directory: Path,
    mass_relative_tolerance: float = DEFAULT_MASS_RELATIVE_TOLERANCE,
) -> HandoffPackage:
    """Read and strictly validate a v1 JSON-plus-NPZ handoff package."""

    source = Path(directory)
    metadata_path = source / METADATA_FILENAME
    arrays_path = source / ARRAYS_FILENAME
    if not metadata_path.is_file():
        _fail(f"missing handoff metadata file: {metadata_path}")
    if not arrays_path.is_file():
        _fail(f"missing handoff arrays file: {arrays_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail(f"cannot read valid JSON metadata from {metadata_path}: {error}")
    try:
        with np.load(arrays_path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key].copy() for key in loaded.files}
    except (OSError, ValueError) as error:
        _fail(f"cannot read valid NPZ arrays from {arrays_path}: {error}")
    package = HandoffPackage(metadata=metadata, arrays=arrays)
    validate_handoff_package(package, mass_relative_tolerance)
    return package


def assert_roundtrip_equivalent(
    original: HandoffPackage,
    restored: HandoffPackage,
) -> None:
    """Raise ``SchemaValidationError`` unless two valid packages are identical."""

    original.validate()
    restored.validate()
    if _canonical_json_value(original.metadata) != _canonical_json_value(restored.metadata):
        _fail("metadata changed during handoff package roundtrip")
    if set(original.arrays) != set(restored.arrays):
        _fail("array keys changed during handoff package roundtrip")
    for key in original.arrays:
        if not np.array_equal(np.asarray(original.arrays[key]), np.asarray(restored.arrays[key])):
            _fail(f"array {key!r} changed during handoff package roundtrip")


def make_handoff_package(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    mass_relative_tolerance: float = DEFAULT_MASS_RELATIVE_TOLERANCE,
) -> HandoffPackage:
    """Copy, validate, and return a handoff package from mapping inputs."""

    package = HandoffPackage(
        metadata=copy.deepcopy(dict(metadata)),
        arrays={key: np.asarray(value).copy() for key, value in arrays.items()},
    )
    validate_handoff_package(package, mass_relative_tolerance)
    return package


# Small aliases make the package discoverable without weakening the strict API.
validate_package = validate_handoff_package
write_package = write_handoff_package
read_package = read_handoff_package
build_handoff_package = make_handoff_package
