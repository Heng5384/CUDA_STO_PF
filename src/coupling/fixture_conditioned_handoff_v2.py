"""Validation-only, fixture-conditioned KWN--PF handoff v2.

This module deliberately sits beside the historical v1 handoff rather than
changing it.  V1 assumed that a KWN resolved-beta population could be sampled
into PF.  The qualified 96-cube fixture already has a resolved-beta geometry,
so treating that geometry as a second KWN population would double count its
solute.  V2 therefore fixes the resolved field inventory first and allocates
only the remaining source inventory among matrix, frozen GP and frozen
sub-grid-beta buckets.

The implementation is a storage adapter, not a GP model.  It never changes
``phi``, never releases auxiliary solute, never creates a beta seed, and never
calls a KWN solver.  It reads the canonical validation contract directly and
stores all four buckets in absolute mol B and in mol B m^-3.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from kwn_mvp.contract import (
    PFKWNValidationContract,
    ValidationContractError as CanonicalValidationContractError,
    canonical_contract_json,
    load_validation_contract as _load_hash_bound_validation_contract,
)


SCHEMA_VERSION = "kwn_pf_handoff_v2"
METADATA_FILENAME = "metadata.json"
ARRAYS_FILENAME = "arrays.npz"
LEDGER_FILENAME = "ledger.csv"
VALIDATION_REPORT_FILENAME = "validation_report.json"
DEFAULT_RELATIVE_TOLERANCE = 1.0e-10
FIXTURE_CONDITIONED_DISCLAIMER = (
    "FIXTURE_CONDITIONED_PRESCRIBED_SOURCE_IS_NOT_A_GP_NUCLEATION_PREDICTION"
)


class FixtureConditionedHandoffError(ValueError):
    """Raised when a v2 handoff violates its storage or provenance contract."""


class InfeasibleFixtureConditionedHandoff(FixtureConditionedHandoffError):
    """Raised when a requested matrix inventory cannot be represented honestly."""

    def __init__(self, message: str, *, minimum_inventory_change_mol: float) -> None:
        super().__init__(message)
        self.minimum_inventory_change_mol = float(minimum_inventory_change_mol)


def canonical_json_bytes(value: Any) -> bytes:
    """Return canonical bytes through the shared validation-contract encoder."""

    try:
        if not isinstance(value, Mapping):
            raise TypeError("canonical payload must be a mapping")
        text = canonical_contract_json(value)
    except (TypeError, ValueError, CanonicalValidationContractError) as error:
        raise FixtureConditionedHandoffError(
            "value cannot be represented by canonical JSON"
        ) from error
    return text.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise FixtureConditionedHandoffError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise FixtureConditionedHandoffError(f"{label} must be finite")
    return result


def _nonnegative(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise FixtureConditionedHandoffError(f"{label} must be non-negative")
    return result


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FixtureConditionedHandoffError(f"{label} must be an object")
    return value


@dataclass(frozen=True)
class ValidationContract:
    """Parsed canonical JSON contract with its exact content hash."""

    path: Path
    document: Mapping[str, Any]
    contract_hash: str
    canonical: PFKWNValidationContract

    @property
    def temperature_K(self) -> float:
        return self.canonical.temperature_k

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        raw = self.canonical.value("numerics.grid_shape")
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise FixtureConditionedHandoffError("contract grid shape must have three entries")
        shape = tuple(int(item) for item in raw)
        if any(item <= 0 for item in shape):
            raise FixtureConditionedHandoffError("contract grid shape must be positive")
        return shape  # type: ignore[return-value]

    @property
    def dx_m(self) -> float:
        value = _finite(self.canonical.value("numerics.dx_m"), "dx_m")
        if value <= 0.0:
            raise FixtureConditionedHandoffError("contract dx_m must be positive")
        return value

    @property
    def v_B(self) -> float:
        return self.canonical.beta_xb

    @property
    def vm_alpha_m3_mol(self) -> float:
        value = self.canonical.vm_alpha_m3_mol
        if value <= 0.0:
            raise FixtureConditionedHandoffError("Vm_alpha must be positive")
        return value

    @property
    def vm_beta_m3_mol(self) -> float:
        value = self.canonical.vm_beta_m3_mol
        if value <= 0.0:
            raise FixtureConditionedHandoffError("Vm_beta must be positive")
        return value


def load_validation_contract(path: Path) -> ValidationContract:
    """Load contract data through the shared KWN/PF hash-bound loader."""

    source = Path(path)
    try:
        canonical = _load_hash_bound_validation_contract(source)
    except (OSError, CanonicalValidationContractError) as error:
        raise FixtureConditionedHandoffError(
            f"cannot read validation contract: {source}"
        ) from error
    mapping = _require_mapping(canonical.data, "validation contract")
    if mapping.get("schema_version") != "PF_KWN_VALIDATION_CONTRACT_V1":
        raise FixtureConditionedHandoffError("unexpected validation contract schema")
    if mapping.get("historical_as_run_claim") is not False:
        raise FixtureConditionedHandoffError(
            "validation contract must not claim historical as-run authority"
        )
    purpose = str(mapping.get("purpose", ""))
    if "VALIDATION_CONTROL_ONLY" not in purpose:
        raise FixtureConditionedHandoffError(
            "contract purpose must declare validation-control-only scope"
        )
    contract = ValidationContract(
        path=canonical.path,
        document=mapping,
        contract_hash=canonical.sha256,
        canonical=canonical,
    )
    # Exercise every scalar required below at load time.  This makes a missing
    # provenance-wrapped input a hard error before a package is emitted.
    _ = (
        contract.temperature_K,
        contract.grid_shape,
        contract.dx_m,
        contract.v_B,
        contract.vm_alpha_m3_mol,
        contract.vm_beta_m3_mol,
    )
    return contract


def h_of_phi(phi: np.ndarray) -> np.ndarray:
    """The contract's smooth phase-storage interpolation, without clipping."""

    value = np.asarray(phi, dtype=np.float64)
    return value * value * value * (10.0 + value * (-15.0 + 6.0 * value))


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class FixtureState:
    """A resolved-beta fixture represented by immutable source arrays in RAM."""

    fixture_id: str
    fixture_hash: str
    source_kind: str
    source_details: Mapping[str, Any]
    shape: Tuple[int, int, int]
    dx_m: float
    phi: np.ndarray
    h_phi: np.ndarray
    delta_C_relaxation: np.ndarray
    xB_alpha: np.ndarray
    resolved_equivalent_radii_m: np.ndarray
    source_total_inventory_mol: float
    source_matrix_inventory_mol: float
    source_resolved_inventory_mol: float

    @property
    def voxel_volume_m3(self) -> float:
        return self.dx_m**3

    @property
    def box_volume_m3(self) -> float:
        return float(math.prod(self.shape)) * self.voxel_volume_m3

    @property
    def alpha(self) -> np.ndarray:
        return 1.0 - self.h_phi

    @property
    def field_hashes(self) -> Dict[str, str]:
        return {
            "phi": _array_sha256(self.phi),
            "h_phi": _array_sha256(self.h_phi),
            "delta_C_relaxation": _array_sha256(self.delta_C_relaxation),
            "xB_alpha": _array_sha256(self.xB_alpha),
        }


def _read_raw_field(path: Path, shape: Tuple[int, int, int], label: str) -> np.ndarray:
    try:
        values = np.fromfile(path, dtype="<f8")
    except OSError as error:
        raise FixtureConditionedHandoffError(f"cannot read {label}: {path}") from error
    expected = int(math.prod(shape))
    if values.size != expected:
        raise FixtureConditionedHandoffError(
            f"{label} has {values.size} values; expected {expected}"
        )
    result = values.reshape(shape, order="C")
    if not np.all(np.isfinite(result)):
        raise FixtureConditionedHandoffError(f"{label} contains NaN or Inf")
    return result


def _profile_directory(profile_root: Path, radius_nm: float) -> Path:
    return Path(profile_root) / f"R{radius_nm:.1f}".replace(".", "p")


def _require_raw_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise FixtureConditionedHandoffError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )


def _assemble_source_fixture(
    spec: Mapping[str, Any],
    contract: ValidationContract,
    *,
    source_kind: str,
    source_details: Mapping[str, Any],
    profile_fields: Iterable[Tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]],
) -> FixtureState:
    """Assemble one fixed geometry from already validated profile fields."""

    target = _require_mapping(spec.get("target"), "fixture target")
    raw_shape = target.get("grid")
    if not isinstance(raw_shape, list) or tuple(raw_shape) != contract.grid_shape:
        raise FixtureConditionedHandoffError("fixture grid does not match validation contract")
    if not math.isclose(float(target.get("temperature_C", math.nan)) + 273.15, contract.temperature_K, abs_tol=1.0e-12):
        raise FixtureConditionedHandoffError("fixture temperature does not match validation contract")
    if not math.isclose(float(target.get("lambda_sm_nm", math.nan)) * 1.0e-9, float(contract.canonical.value("interface.lambda_sm_m")), abs_tol=1.0e-20):
        raise FixtureConditionedHandoffError("fixture interface width does not match validation contract")
    shape = contract.grid_shape
    phi_complement = np.ones(shape, dtype=np.float64)
    delta_x_total = np.zeros(shape, dtype=np.float64)
    radii: list[float] = []
    for particle, phi_profile, h_profile, delta_profile in profile_fields:
        if phi_profile.shape != shape or h_profile.shape != shape or delta_profile.shape != shape:
            raise FixtureConditionedHandoffError("profile field shape mismatch")
        h_recomputed = h_of_phi(phi_profile)
        if float(np.max(np.abs(h_profile - h_recomputed))) > 5.0e-14:
            raise FixtureConditionedHandoffError("source h(phi) does not satisfy storage contract")
        alpha_source = 1.0 - h_profile
        if float(np.min(alpha_source)) <= 0.0:
            raise FixtureConditionedHandoffError(
                "source profile has a zero matrix-support cell; portable delta_C is undefined"
            )
        delta_x = delta_profile / alpha_source
        if not np.all(np.isfinite(delta_x)):
            raise FixtureConditionedHandoffError("source delta_C/alpha is non-finite")
        phi_complement *= 1.0 - phi_profile
        delta_x_total += delta_x
        radii.append(float(particle["registered_radius_nm"]) * 1.0e-9)

    phi = 1.0 - phi_complement
    if not np.all(np.isfinite(phi)) or float(np.min(phi)) < 0.0 or float(np.max(phi)) > 1.0:
        raise FixtureConditionedHandoffError("assembled phi is not bounded")
    h_phi = h_of_phi(phi)
    alpha = 1.0 - h_phi
    if float(np.min(alpha)) <= 0.0:
        raise FixtureConditionedHandoffError(
            "assembled fixture has zero matrix support; baseline mapping would divide by zero"
        )
    delta_c = alpha * delta_x_total
    if not np.all(np.isfinite(delta_c)):
        raise FixtureConditionedHandoffError("assembled delta_C_relaxation is non-finite")

    target_mean = _finite(target.get("mean_C_B_tot"), "fixture target mean_C_B_tot")
    target_storage = target_mean * float(math.prod(shape))
    beta_storage = float(np.sum(h_phi * contract.v_B, dtype=np.float64))
    relaxation_storage = float(np.sum(delta_c, dtype=np.float64))
    matrix_capacity = float(np.sum(alpha, dtype=np.float64))
    baseline = (target_storage - beta_storage - relaxation_storage) / matrix_capacity
    upper = _finite(target.get("xB_max_safe", 1.0), "fixture xB_max_safe")
    if not 0.0 < baseline < upper:
        raise FixtureConditionedHandoffError("fixture-derived matrix baseline is not physical")
    x_b = baseline + delta_c / alpha
    if not np.all(np.isfinite(x_b)) or float(np.min(x_b)) <= 0.0 or float(np.max(x_b)) >= upper:
        raise FixtureConditionedHandoffError("fixture-derived matrix xB is out of bounds")

    voxel_volume = contract.dx_m**3
    q_matrix = field_matrix_inventory_mol(alpha, x_b, voxel_volume, contract.vm_alpha_m3_mol)
    q_resolved = field_resolved_inventory_mol(h_phi, contract.v_B, voxel_volume, contract.vm_beta_m3_mol)
    total = q_matrix + q_resolved
    expected_total = target_storage * voxel_volume / contract.vm_alpha_m3_mol
    if abs(total - expected_total) / max(abs(expected_total), 1.0e-300) > 2.0e-13:
        raise FixtureConditionedHandoffError("assembled fixture physical inventory does not close")
    return FixtureState(
        fixture_id=str(spec.get("fixture_id", "unknown_fixture")),
        fixture_hash=canonical_sha256(spec),
        source_kind=source_kind,
        source_details=dict(source_details),
        shape=shape,
        dx_m=contract.dx_m,
        phi=phi,
        h_phi=h_phi,
        delta_C_relaxation=delta_c,
        xB_alpha=x_b,
        resolved_equivalent_radii_m=np.asarray(radii, dtype=np.float64),
        source_total_inventory_mol=total,
        source_matrix_inventory_mol=q_matrix,
        source_resolved_inventory_mol=q_resolved,
    )


def load_host_96cube_fixture(
    fixture_spec_path: Path,
    profile_root: Path,
    contract: ValidationContract,
) -> FixtureState:
    """Read the frozen six-particle source profiles with field SHA validation.

    This is a read-only host-side reconstruction of the existing fixture.  It
    intentionally does not copy its raw fields into the repository.
    """

    spec_path = Path(fixture_spec_path)
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FixtureConditionedHandoffError(f"cannot read fixture spec {spec_path}") from error
    spec_mapping = _require_mapping(spec, "fixture spec")
    if spec_mapping.get("schema") != "PF_MASS_CONSERVING_LIBRARY_HANDOFF_SPEC_V1":
        raise FixtureConditionedHandoffError("unexpected source fixture schema")
    if spec_mapping.get("validation_only") is not True:
        raise FixtureConditionedHandoffError("source fixture must remain validation-only")
    particles = spec_mapping.get("particles")
    if not isinstance(particles, list) or len(particles) != 6:
        raise FixtureConditionedHandoffError("expected exactly six frozen resolved-beta particles")

    entries = []
    provenance: Dict[str, Any] = {
        "fixture_spec_path": str(spec_path),
        "fixture_spec_sha256": sha256_file(spec_path),
        "profile_root": str(profile_root),
        "profiles": [],
    }
    profile_source_commits = set()
    for particle_value in particles:
        particle = _require_mapping(particle_value, "fixture particle")
        radius_nm = _finite(particle.get("registered_radius_nm"), "registered_radius_nm")
        center = particle.get("center_grid")
        if not isinstance(center, list) or len(center) != 3:
            raise FixtureConditionedHandoffError("fixture particle center_grid is invalid")
        directory = _profile_directory(profile_root, radius_nm)
        manifest_path = directory / "profile_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FixtureConditionedHandoffError(
                f"cannot read profile manifest {manifest_path}"
            ) from error
        expected_manifest = str(particle.get("source_profile_manifest_sha256", ""))
        if expected_manifest:
            _require_raw_hash(manifest_path, expected_manifest, f"{particle.get('particle_id')} manifest")
        fields = _require_mapping(manifest.get("fields"), "profile fields")
        profile_source_commit = str(manifest.get("source_commit", ""))
        if profile_source_commit:
            profile_source_commits.add(profile_source_commit)
        field_values: Dict[str, np.ndarray] = {}
        for name, expected_from_spec in (
            ("phi", particle.get("source_phi_sha256")),
            ("delta_C_relaxation", particle.get("source_delta_C_relaxation_sha256")),
            ("h_phi", None),
        ):
            field_info = _require_mapping(fields.get(name), f"{name} field")
            raw_path = directory / str(field_info.get("path", ""))
            expected = str(expected_from_spec or field_info.get("sha256", ""))
            if not expected:
                raise FixtureConditionedHandoffError(f"{name} field has no SHA-256")
            _require_raw_hash(raw_path, expected, f"{particle.get('particle_id')} {name}")
            field_values[name] = _read_raw_field(raw_path, contract.grid_shape, name)
        shift = tuple(int(value) - (size // 2) for value, size in zip(center, contract.grid_shape))
        entries.append(
            (
                particle,
                np.roll(field_values["phi"], shift, axis=(0, 1, 2)),
                np.roll(field_values["h_phi"], shift, axis=(0, 1, 2)),
                np.roll(field_values["delta_C_relaxation"], shift, axis=(0, 1, 2)),
            )
        )
        provenance["profiles"].append(
            {
                "particle_id": particle.get("particle_id"),
                "registered_radius_nm": radius_nm,
                "profile_manifest_sha256": sha256_file(manifest_path),
                "profile_source_commit": profile_source_commit,
                "center_grid": [int(value) for value in center],
            }
        )
    provenance["profile_source_commits"] = sorted(profile_source_commits)
    return _assemble_source_fixture(
        spec_mapping,
        contract,
        source_kind="HOST_RAW_PROFILE_FIELDS_HASH_VALIDATED",
        source_details=provenance,
        profile_fields=entries,
    )


def make_synthetic_fixture_control(
    contract: ValidationContract,
    fixture_spec_path: Path,
) -> FixtureState:
    """Make a deterministic fallback control, explicitly not a historical field.

    It preserves the six-particle layout but uses analytic diffuse spheres.  It
    exists only so the v2 storage contract has a local test fixture when raw
    frozen fields are unavailable.  Its provenance status prevents it from
    being mistaken for the qualified host fixture.
    """

    try:
        spec = json.loads(Path(fixture_spec_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FixtureConditionedHandoffError("cannot read synthetic fixture spec") from error
    spec_mapping = _require_mapping(spec, "fixture spec")
    shape = contract.grid_shape
    indices = np.indices(shape, dtype=np.float64)
    entries = []
    for index, item in enumerate(spec_mapping.get("particles", [])):
        particle = _require_mapping(item, "fixture particle")
        center = tuple(float(value) for value in particle["center_grid"])
        radius_nm = _finite(particle["registered_radius_nm"], "registered radius")
        squared = np.zeros(shape, dtype=np.float64)
        for axis, size in enumerate(shape):
            distance = np.abs(indices[axis] - center[axis])
            distance = np.minimum(distance, float(size) - distance)
            squared += distance * distance
        radial = np.sqrt(squared)
        # A fixed analytic profile is permitted only in this fallback control.
        phi_profile = 0.5 * (1.0 - np.tanh((radial - radius_nm) / 2.0))
        h_profile = h_of_phi(phi_profile)
        alpha = 1.0 - h_profile
        # A deterministic, small matrix-side relaxation pattern makes the
        # preservation test meaningful without asserting a physical profile.
        delta_profile = alpha * 1.0e-6 * np.exp(-((radial - radius_nm) / 4.0) ** 2)
        entries.append((particle, phi_profile, h_profile, delta_profile))
    if len(entries) != 6:
        raise FixtureConditionedHandoffError("synthetic control requires the six-particle layout")
    details = {
        "fixture_spec_path": str(fixture_spec_path),
        "fixture_spec_sha256": sha256_file(Path(fixture_spec_path)),
        "reason": "HOST_RAW_PROFILE_FIELDS_UNAVAILABLE",
        "scientific_status": "SYNTHETIC_STORAGE_CONTROL_NOT_HISTORICAL_NOT_PRODUCTION",
        "profile_kind": "deterministic_analytic_diffuse_sphere_control",
    }
    return _assemble_source_fixture(
        spec_mapping,
        contract,
        source_kind="SYNTHETIC_STORAGE_CONTROL_NOT_HISTORICAL_NOT_PRODUCTION",
        source_details=details,
        profile_fields=entries,
    )


def field_matrix_inventory_mol(
    alpha: np.ndarray,
    x_b_alpha: np.ndarray,
    voxel_volume_m3: float,
    vm_alpha_m3_mol: float,
) -> float:
    """Integrate the exact matrix term of the declared phase-storage equation."""

    return float(np.sum(np.asarray(alpha) * np.asarray(x_b_alpha), dtype=np.float64)) * voxel_volume_m3 / vm_alpha_m3_mol


def field_resolved_inventory_mol(
    h_phi: np.ndarray,
    v_b: float,
    voxel_volume_m3: float,
    vm_beta_m3_mol: float,
) -> float:
    """Integrate the exact resolved-beta term, not a volume-fraction proxy."""

    return float(np.sum(np.asarray(h_phi) * float(v_b), dtype=np.float64)) * voxel_volume_m3 / vm_beta_m3_mol


@dataclass(frozen=True)
class FrozenPopulation:
    """Compact host-side PSD state; it does not contribute to PF dynamics."""

    name: str
    radius_bin_edges_m: np.ndarray
    number_density_per_m4: np.ndarray
    x_b: float
    vm_m3_mol: float
    frozen: bool
    provenance: Mapping[str, Any]

    def validate(self) -> None:
        edges = np.asarray(self.radius_bin_edges_m, dtype=np.float64)
        density = np.asarray(self.number_density_per_m4, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2 or np.any(~np.isfinite(edges)):
            raise FixtureConditionedHandoffError(f"{self.name} bin edges are invalid")
        if np.any(edges <= 0.0) or np.any(np.diff(edges) <= 0.0):
            raise FixtureConditionedHandoffError(f"{self.name} bin edges must increase")
        if density.ndim != 1 or density.size != edges.size - 1 or np.any(~np.isfinite(density)):
            raise FixtureConditionedHandoffError(f"{self.name} number density is invalid")
        if np.any(density < 0.0):
            raise FixtureConditionedHandoffError(f"{self.name} number density is negative")
        if not 0.0 <= _finite(self.x_b, f"{self.name}.x_b") <= 1.0:
            raise FixtureConditionedHandoffError(f"{self.name}.x_b is out of bounds")
        if _finite(self.vm_m3_mol, f"{self.name}.Vm") <= 0.0:
            raise FixtureConditionedHandoffError(f"{self.name}.Vm must be positive")
        if self.frozen is not True:
            raise FixtureConditionedHandoffError(f"{self.name} must be frozen in v2")

    @property
    def widths_m(self) -> np.ndarray:
        return np.diff(np.asarray(self.radius_bin_edges_m, dtype=np.float64))

    @property
    def centres_m(self) -> np.ndarray:
        edges = np.asarray(self.radius_bin_edges_m, dtype=np.float64)
        return 0.5 * (edges[:-1] + edges[1:])

    def volume_fraction(self) -> float:
        self.validate()
        particle_volume = 4.0 * math.pi / 3.0 * self.centres_m**3
        return float(np.sum(self.number_density_per_m4 * self.widths_m * particle_volume, dtype=np.float64))

    def inventory_mol(self, box_volume_m3: float) -> float:
        return self.volume_fraction() * float(box_volume_m3) * self.x_b / self.vm_m3_mol

    def expected_count(self, box_volume_m3: float) -> np.ndarray:
        return np.asarray(self.number_density_per_m4, dtype=np.float64) * self.widths_m * float(box_volume_m3)


def prescribed_population_from_inventory(
    *,
    name: str,
    target_inventory_mol: float,
    box_volume_m3: float,
    radius_bin_edges_m: Sequence[float],
    volume_weights: Sequence[float],
    x_b: float,
    vm_m3_mol: float,
    provenance: Mapping[str, Any],
) -> FrozenPopulation:
    """Build a deterministic compact PSD whose integrated inventory is exact.

    The weights partition physical population volume, not counts.  No
    nucleation law is implied by them.
    """

    target = _nonnegative(target_inventory_mol, "target_inventory_mol")
    edges = np.asarray(radius_bin_edges_m, dtype=np.float64)
    weights = np.asarray(volume_weights, dtype=np.float64)
    if edges.ndim != 1 or weights.ndim != 1 or weights.size != edges.size - 1:
        raise FixtureConditionedHandoffError("prescribed PSD bins and weights mismatch")
    if np.any(weights < 0.0) or not math.isclose(float(np.sum(weights)), 1.0, abs_tol=1.0e-14):
        raise FixtureConditionedHandoffError("prescribed PSD volume weights must sum to one")
    x_value = _finite(x_b, "prescribed xB")
    vm_value = _finite(vm_m3_mol, "prescribed Vm")
    if target > 0.0 and (x_value <= 0.0 or vm_value <= 0.0 or box_volume_m3 <= 0.0):
        raise FixtureConditionedHandoffError("nonzero auxiliary inventory has invalid composition/volume")
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    particle_volume = 4.0 * math.pi / 3.0 * centres**3
    population_volume_fraction = 0.0 if target == 0.0 else target * vm_value / (x_value * box_volume_m3)
    density = np.zeros_like(weights)
    if target > 0.0:
        density = population_volume_fraction * weights / (widths * particle_volume)
    population = FrozenPopulation(
        name=name,
        radius_bin_edges_m=edges,
        number_density_per_m4=density,
        x_b=x_value,
        vm_m3_mol=vm_value,
        frozen=True,
        provenance=dict(provenance),
    )
    population.validate()
    return population


def map_matrix_inventory_preserving_fixture(
    fixture: FixtureState,
    contract: ValidationContract,
    target_matrix_inventory_mol: float,
    *,
    x_b_lower_bound: float = 0.0,
    x_b_upper_bound: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Invert the exact matrix storage equation by changing only its baseline.

    ``phi`` and ``delta_C_relaxation`` are neither copied nor reset.  No clamp
    is used: an out-of-range target is reported as infeasible with the smallest
    inventory change needed to reach a composition bound.
    """

    target = _nonnegative(target_matrix_inventory_mol, "target_matrix_inventory_mol")
    lower = _finite(x_b_lower_bound, "xB lower bound")
    upper = _finite(x_b_upper_bound, "xB upper bound")
    if not 0.0 <= lower < upper <= 1.0:
        raise FixtureConditionedHandoffError("matrix xB bounds must lie inside [0, 1]")
    alpha = fixture.alpha
    if float(np.min(alpha)) <= 0.0:
        raise FixtureConditionedHandoffError("fixture has zero matrix-support cells")
    storage_target = target * contract.vm_alpha_m3_mol / fixture.voxel_volume_m3
    support = float(np.sum(alpha, dtype=np.float64))
    relaxation = float(np.sum(fixture.delta_C_relaxation, dtype=np.float64))
    baseline = (storage_target - relaxation) / support
    delta_over_alpha = fixture.delta_C_relaxation / alpha
    feasible_lower = float(np.max(lower - delta_over_alpha))
    feasible_upper = float(np.min(upper - delta_over_alpha))
    if baseline < feasible_lower or baseline > feasible_upper:
        nearest = min(max(baseline, feasible_lower), feasible_upper)
        required_storage_change = abs(nearest - baseline) * support
        required_inventory_change = required_storage_change * fixture.voxel_volume_m3 / contract.vm_alpha_m3_mol
        raise InfeasibleFixtureConditionedHandoff(
            "target matrix inventory cannot preserve the fixture delta_C_relaxation within bounds",
            minimum_inventory_change_mol=required_inventory_change,
        )
    x_b = baseline + delta_over_alpha
    if not np.all(np.isfinite(x_b)):
        raise FixtureConditionedHandoffError("matrix inverse produced NaN or Inf")
    # The comparisons are deliberately direct.  They are a feasibility check,
    # not a request to clip numerical tails back into range.
    if float(np.min(x_b)) < lower or float(np.max(x_b)) > upper:
        raise FixtureConditionedHandoffError("matrix inverse violates its declared bounds")
    actual = field_matrix_inventory_mol(alpha, x_b, fixture.voxel_volume_m3, contract.vm_alpha_m3_mol)
    relative = abs(actual - target) / max(abs(target), 1.0e-300)
    if relative > 2.0e-13:
        raise FixtureConditionedHandoffError("matrix inverse does not close its target inventory")
    return x_b, {
        "target_matrix_inventory_mol": target,
        "actual_matrix_inventory_mol": actual,
        "relative_residual": relative,
        "baseline_xB": baseline,
        "matrix_support_storage": support,
        "relaxation_storage": relaxation,
        "xB_min": float(np.min(x_b)),
        "xB_max": float(np.max(x_b)),
        "lower_bound_margin": float(np.min(x_b) - lower),
        "upper_bound_margin": float(upper - np.max(x_b)),
        "clipping_used": 0.0,
    }


def _ledger_row(name: str, q_mol: float, box_volume_m3: float) -> Dict[str, Any]:
    return {
        "bucket": name,
        "Q_B_mol": float(q_mol),
        "C_B_mol_m3": float(q_mol) / float(box_volume_m3),
        "unit_Q_B": "mol_B",
        "unit_C_B": "mol_B_m-3",
    }


def _ledger_mapping(
    *,
    q_total: float,
    q_matrix: float,
    q_gp: float,
    q_subgrid: float,
    q_resolved: float,
    box_volume_m3: float,
) -> Dict[str, Any]:
    bucket_sum = q_matrix + q_gp + q_subgrid + q_resolved
    residual = q_total - bucket_sum
    return {
        "unit_Q_B": "mol_B",
        "unit_C_B": "mol_B_m-3",
        "box_volume_m3": box_volume_m3,
        "Q_B_total_mol": q_total,
        "Q_B_matrix_mol": q_matrix,
        "Q_B_GP_mol": q_gp,
        "Q_B_beta_subgrid_mol": q_subgrid,
        "Q_B_beta_resolved_fixed_mol": q_resolved,
        "Q_B_bucket_sum_mol": bucket_sum,
        "residual_mol": residual,
        "relative_residual": abs(residual) / max(abs(q_total), 1.0e-300),
        "C_B_total_mol_m3": q_total / box_volume_m3,
        "C_B_matrix_mol_m3": q_matrix / box_volume_m3,
        "C_B_GP_mol_m3": q_gp / box_volume_m3,
        "C_B_beta_subgrid_mol_m3": q_subgrid / box_volume_m3,
        "C_B_beta_resolved_fixed_mol_m3": q_resolved / box_volume_m3,
    }


def _array_manifest(arrays: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    return {
        name: {
            "sha256": _array_sha256(value),
            "shape": list(np.asarray(value).shape),
            "dtype": "float64",
        }
        for name, value in sorted(arrays.items())
    }


def _source_handoff_hash(fixture: FixtureState, contract: ValidationContract) -> str:
    return canonical_sha256(
        {
            "schema": "FIXTURE_CONDITIONED_PRESCRIBED_SOURCE_V2",
            "contract_hash": contract.contract_hash,
            "fixture_hash": fixture.fixture_hash,
            "source_total_inventory_mol": fixture.source_total_inventory_mol,
            "source_matrix_inventory_mol": fixture.source_matrix_inventory_mol,
            "source_resolved_inventory_mol": fixture.source_resolved_inventory_mol,
            "fixture_field_hashes": fixture.field_hashes,
        }
    )


def build_fixture_conditioned_handoff_v2(
    fixture: FixtureState,
    contract: ValidationContract,
    *,
    gp_fraction_of_source_matrix: float = 0.01,
    beta_subgrid_fraction_of_source_matrix: float = 0.0,
    gp_x_b: float = 0.03,
    beta_subgrid_x_b: float = 1.0,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, Any]]:
    """Build a feasible v2 adapter package without mutating fixture fields."""

    if fixture.shape != contract.grid_shape:
        raise FixtureConditionedHandoffError("fixture shape does not match contract")
    if not math.isclose(fixture.dx_m, contract.dx_m, rel_tol=0.0, abs_tol=1.0e-24):
        raise FixtureConditionedHandoffError("fixture dx does not match contract")
    gp_fraction = _nonnegative(gp_fraction_of_source_matrix, "gp fraction")
    subgrid_fraction = _nonnegative(beta_subgrid_fraction_of_source_matrix, "subgrid fraction")
    if gp_fraction + subgrid_fraction >= 1.0:
        raise FixtureConditionedHandoffError("auxiliary fractions must leave matrix inventory positive")
    requested_gp = fixture.source_matrix_inventory_mol * gp_fraction
    requested_subgrid = fixture.source_matrix_inventory_mol * subgrid_fraction
    gp = prescribed_population_from_inventory(
        name="GP",
        target_inventory_mol=requested_gp,
        box_volume_m3=fixture.box_volume_m3,
        radius_bin_edges_m=(0.75e-9, 1.25e-9, 1.75e-9),
        volume_weights=(0.45, 0.55),
        x_b=gp_x_b,
        vm_m3_mol=contract.vm_alpha_m3_mol,
        provenance={
            "kind": "PRESCRIBED_NON_PREDICTIVE_STORAGE_CONTROL",
            "disclaimer": FIXTURE_CONDITIONED_DISCLAIMER,
            "backend": "compact_host_population_state_v2",
        },
    )
    subgrid = prescribed_population_from_inventory(
        name="beta_subgrid",
        target_inventory_mol=requested_subgrid,
        box_volume_m3=fixture.box_volume_m3,
        radius_bin_edges_m=(1.75e-9, 2.25e-9, 2.75e-9),
        volume_weights=(0.5, 0.5),
        x_b=beta_subgrid_x_b,
        vm_m3_mol=contract.vm_beta_m3_mol,
        provenance={
            "kind": "PRESCRIBED_NON_PREDICTIVE_STORAGE_CONTROL",
            "disclaimer": FIXTURE_CONDITIONED_DISCLAIMER,
            "backend": "compact_host_population_state_v2",
        },
    )
    q_gp = gp.inventory_mol(fixture.box_volume_m3)
    q_subgrid = subgrid.inventory_mol(fixture.box_volume_m3)
    # Fixed resolved inventory is removed exactly once from the source before
    # KWN-like remaining buckets are allocated.
    q_total = fixture.source_total_inventory_mol
    q_resolved = fixture.source_resolved_inventory_mol
    q_matrix_target = q_total - q_resolved - q_gp - q_subgrid
    target_xb, matrix_audit = map_matrix_inventory_preserving_fixture(
        fixture,
        contract,
        q_matrix_target,
    )
    q_matrix = field_matrix_inventory_mol(
        fixture.alpha,
        target_xb,
        fixture.voxel_volume_m3,
        contract.vm_alpha_m3_mol,
    )
    ledger = _ledger_mapping(
        q_total=q_total,
        q_matrix=q_matrix,
        q_gp=q_gp,
        q_subgrid=q_subgrid,
        q_resolved=q_resolved,
        box_volume_m3=fixture.box_volume_m3,
    )
    # The target matrix field is reconstructable as baseline + delta_C/alpha
    # against the fixed fixture.  Store only its scalar baseline, never a
    # second dense 96^3 raw field in the handoff artifact.
    arrays = {
        "matrix_baseline_xB": np.asarray([matrix_audit["baseline_xB"]], dtype=np.float64),
        "gp_radius_bin_edges_m": np.asarray(gp.radius_bin_edges_m, dtype=np.float64),
        "gp_number_density_per_m4": np.asarray(gp.number_density_per_m4, dtype=np.float64),
        "gp_expected_count": np.asarray(gp.expected_count(fixture.box_volume_m3), dtype=np.float64),
        "beta_subgrid_radius_bin_edges_m": np.asarray(subgrid.radius_bin_edges_m, dtype=np.float64),
        "beta_subgrid_number_density_per_m4": np.asarray(subgrid.number_density_per_m4, dtype=np.float64),
        "beta_subgrid_expected_count": np.asarray(subgrid.expected_count(fixture.box_volume_m3), dtype=np.float64),
        "resolved_equivalent_radii_m": np.asarray(fixture.resolved_equivalent_radii_m, dtype=np.float64),
    }
    source_handoff_hash = _source_handoff_hash(fixture, contract)
    metadata: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "validation_only": True,
        "historical_as_run_claim": False,
        "contract_hash": contract.contract_hash,
        "contract_schema_version": contract.document["schema_version"],
        "contract_source_path": str(contract.path),
        "source_kwn_commit": str(contract.document.get("source_commit", "")),
        "validation_contract_source_commit": str(contract.document.get("source_commit", "")),
        "source_pf_commit": (
            fixture.source_details.get("profile_source_commits", [""])[0]
            if len(fixture.source_details.get("profile_source_commits", [])) == 1
            else "SYNTHETIC_OR_PROFILE_COMMIT_NOT_UNIQUE"
        ),
        "fixture_id": fixture.fixture_id,
        "fixture_hash": fixture.fixture_hash,
        "fixture_source_kind": fixture.source_kind,
        "fixture_source_details": dict(fixture.source_details),
        "temperature_K": contract.temperature_K,
        "box": {
            "grid_shape": list(fixture.shape),
            "dx_m": fixture.dx_m,
            "box_volume_m3": fixture.box_volume_m3,
            "periodic": True,
        },
        "time_state_label": "96cube_fixture_conditioned_storage_adapter_t0",
        "conditioning_mode": "FIXED_RESOLVED_BETA_GEOMETRY_REMAINING_BUCKET_ALLOCATION",
        "prescribed_source_disclaimer": FIXTURE_CONDITIONED_DISCLAIMER,
        "source_handoff_hash": source_handoff_hash,
        "auxiliary_state": {
            "schema_version": "AuxPopulationState_V1",
            "frozen": True,
            "dynamics_excluded": [
                "chemical_potential",
                "phi",
                "GP_release",
                "GP_to_beta_conversion",
                "new_beta_seed",
                "online_KWN_call",
            ],
            "GP": {
                "xB_g": gp.x_b,
                "Vm_m3_mol": gp.vm_m3_mol,
                "inventory_mol": q_gp,
                "volume_fraction": gp.volume_fraction(),
                "provenance": dict(gp.provenance),
            },
            "beta_subgrid": {
                "xB_beta": subgrid.x_b,
                "Vm_m3_mol": subgrid.vm_m3_mol,
                "inventory_mol": q_subgrid,
                "volume_fraction": subgrid.volume_fraction(),
                "provenance": dict(subgrid.provenance),
            },
        },
        "fixed_resolved_beta": {
            "inventory_mol": q_resolved,
            "field_storage_contract": "sum(h(phi)*v_B)*voxel_volume/Vm_beta",
            "source_geometry_preserved": True,
            "resolved_equivalent_radius_count": int(fixture.resolved_equivalent_radii_m.size),
        },
        "matrix_field_mapping": {
            "method": "baseline_only_inverse_storage_preserve_phi_and_delta_C_relaxation",
            "source_phi_sha256": fixture.field_hashes["phi"],
            "source_h_phi_sha256": fixture.field_hashes["h_phi"],
            "source_delta_C_relaxation_sha256": fixture.field_hashes["delta_C_relaxation"],
            "matrix_baseline_array_key": "matrix_baseline_xB",
            "reconstruction": "xB_alpha=matrix_baseline_xB+delta_C_relaxation/(1-h(phi))",
            "clipping_used": False,
            "phi_reset": False,
            "delta_C_relaxation_reset": False,
            "audit": matrix_audit,
        },
        "ledger": ledger,
        "unit_definitions": {
            "Q_B": "absolute mol of pseudo-binary B=Ag2Te in complete PF box",
            "C_B": "mol B m^-3, computed as Q_B/box_volume",
            "PF_native_storage": "dimensionless C_B_code=(1-h)*xB_alpha+h*v_B",
            "molar_volume_conversion": "Q_matrix=sum((1-h)*xB_alpha)*voxel_volume/Vm_alpha; Q_resolved=sum(h*v_B)*voxel_volume/Vm_beta",
        },
        "double_count_contract": {
            "fixed_resolved_removed_before_remaining_allocation": True,
            "resolved_bucket_is_not_generated_from_compact_KWN_beta_PSD": True,
            "GP_is_not_added_to_matrix_xB": True,
            "source_total_inventory_mol": fixture.source_total_inventory_mol,
            "source_matrix_inventory_mol": fixture.source_matrix_inventory_mol,
            "source_resolved_inventory_mol": fixture.source_resolved_inventory_mol,
        },
        "pf_raw_initialization_emitted": False,
        "pf_raw_initialization_allowed": False,
        "array_manifest": _array_manifest(arrays),
    }
    return metadata, arrays, {"ledger": ledger, "matrix_audit": matrix_audit}


def _close(lhs: float, rhs: float, scale: float, tolerance: float) -> bool:
    return abs(lhs - rhs) <= tolerance * max(abs(scale), 1.0e-300)


def reconstruct_matrix_xb_alpha(
    fixture: FixtureState, arrays: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Reconstruct the dense matrix field from the compact v2 baseline payload."""

    if "matrix_baseline_xB" not in arrays:
        raise FixtureConditionedHandoffError("v2 arrays have no matrix baseline")
    baseline = np.asarray(arrays["matrix_baseline_xB"], dtype=np.float64)
    if baseline.shape != (1,) or not np.all(np.isfinite(baseline)):
        raise FixtureConditionedHandoffError("matrix baseline array must contain one finite scalar")
    if float(np.min(fixture.alpha)) <= 0.0:
        raise FixtureConditionedHandoffError("fixture has zero matrix support")
    return float(baseline[0]) + fixture.delta_C_relaxation / fixture.alpha


def validate_fixture_conditioned_handoff_v2(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    fixture: FixtureState,
    contract: ValidationContract,
    *,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> Dict[str, Any]:
    """Strictly validate a v2 package against fields, PSDs, units and ledger."""

    if relative_tolerance < 0.0 or not math.isfinite(relative_tolerance):
        raise FixtureConditionedHandoffError("relative tolerance is invalid")
    data = _require_mapping(metadata, "metadata")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise FixtureConditionedHandoffError("wrong v2 package schema")
    if data.get("validation_only") is not True or data.get("historical_as_run_claim") is not False:
        raise FixtureConditionedHandoffError("v2 package scope is not validation-only")
    if data.get("contract_hash") != contract.contract_hash:
        raise FixtureConditionedHandoffError("contract hash mismatch")
    if data.get("fixture_hash") != fixture.fixture_hash:
        raise FixtureConditionedHandoffError("fixture hash mismatch")
    if data.get("prescribed_source_disclaimer") != FIXTURE_CONDITIONED_DISCLAIMER:
        raise FixtureConditionedHandoffError("missing prescribed-source disclaimer")
    expected = {
        "matrix_baseline_xB",
        "gp_radius_bin_edges_m",
        "gp_number_density_per_m4",
        "gp_expected_count",
        "beta_subgrid_radius_bin_edges_m",
        "beta_subgrid_number_density_per_m4",
        "beta_subgrid_expected_count",
        "resolved_equivalent_radii_m",
    }
    if set(arrays) != expected:
        raise FixtureConditionedHandoffError("v2 arrays have missing or unknown entries")
    checked = {name: np.asarray(value, dtype=np.float64) for name, value in arrays.items()}
    if any(not np.all(np.isfinite(value)) for value in checked.values()):
        raise FixtureConditionedHandoffError("v2 arrays contain NaN or Inf")
    x_b = reconstruct_matrix_xb_alpha(fixture, checked)
    if float(np.min(x_b)) < 0.0 or float(np.max(x_b)) > 1.0:
        raise FixtureConditionedHandoffError("mapped matrix field is out of physical bounds")
    gp = FrozenPopulation(
        name="GP",
        radius_bin_edges_m=checked["gp_radius_bin_edges_m"],
        number_density_per_m4=checked["gp_number_density_per_m4"],
        x_b=_finite(data["auxiliary_state"]["GP"]["xB_g"], "GP xB"),
        vm_m3_mol=_finite(data["auxiliary_state"]["GP"]["Vm_m3_mol"], "GP Vm"),
        frozen=bool(data["auxiliary_state"].get("frozen")),
        provenance=_require_mapping(data["auxiliary_state"]["GP"].get("provenance"), "GP provenance"),
    )
    subgrid = FrozenPopulation(
        name="beta_subgrid",
        radius_bin_edges_m=checked["beta_subgrid_radius_bin_edges_m"],
        number_density_per_m4=checked["beta_subgrid_number_density_per_m4"],
        x_b=_finite(data["auxiliary_state"]["beta_subgrid"]["xB_beta"], "subgrid xB"),
        vm_m3_mol=_finite(data["auxiliary_state"]["beta_subgrid"]["Vm_m3_mol"], "subgrid Vm"),
        frozen=bool(data["auxiliary_state"].get("frozen")),
        provenance=_require_mapping(data["auxiliary_state"]["beta_subgrid"].get("provenance"), "subgrid provenance"),
    )
    gp.validate()
    subgrid.validate()
    if not np.allclose(checked["gp_expected_count"], gp.expected_count(fixture.box_volume_m3), rtol=0.0, atol=1.0e-12):
        raise FixtureConditionedHandoffError("GP expected-count array does not match its density")
    if not np.allclose(checked["beta_subgrid_expected_count"], subgrid.expected_count(fixture.box_volume_m3), rtol=0.0, atol=1.0e-12):
        raise FixtureConditionedHandoffError("subgrid expected-count array does not match its density")
    q_matrix = field_matrix_inventory_mol(fixture.alpha, x_b, fixture.voxel_volume_m3, contract.vm_alpha_m3_mol)
    q_resolved = field_resolved_inventory_mol(fixture.h_phi, contract.v_B, fixture.voxel_volume_m3, contract.vm_beta_m3_mol)
    q_gp = gp.inventory_mol(fixture.box_volume_m3)
    q_subgrid = subgrid.inventory_mol(fixture.box_volume_m3)
    ledger = _require_mapping(data.get("ledger"), "ledger")
    q_total = _nonnegative(ledger.get("Q_B_total_mol"), "ledger total")
    checks = {
        "contract_hash_bind": data.get("contract_hash") == contract.contract_hash,
        "fixed_resolved_matches_field": _close(q_resolved, _finite(ledger.get("Q_B_beta_resolved_fixed_mol"), "ledger resolved"), q_total, relative_tolerance),
        "matrix_matches_field": _close(q_matrix, _finite(ledger.get("Q_B_matrix_mol"), "ledger matrix"), q_total, relative_tolerance),
        "gp_matches_compact_psd": _close(q_gp, _finite(ledger.get("Q_B_GP_mol"), "ledger GP"), q_total, relative_tolerance),
        "subgrid_matches_compact_psd": _close(q_subgrid, _finite(ledger.get("Q_B_beta_subgrid_mol"), "ledger subgrid"), q_total, relative_tolerance),
        "ledger_closes": _close(q_total, q_matrix + q_resolved + q_gp + q_subgrid, q_total, relative_tolerance),
        "resolved_not_double_counted": _close(q_resolved, fixture.source_resolved_inventory_mol, q_total, relative_tolerance)
        and data.get("double_count_contract", {}).get("fixed_resolved_removed_before_remaining_allocation") is True
        and data.get("double_count_contract", {}).get("resolved_bucket_is_not_generated_from_compact_KWN_beta_PSD") is True,
        "gp_not_added_to_matrix": data.get("double_count_contract", {}).get("GP_is_not_added_to_matrix_xB") is True,
        "phi_and_delta_reference_preserved": data.get("matrix_field_mapping", {}).get("source_phi_sha256") == fixture.field_hashes["phi"]
        and data.get("matrix_field_mapping", {}).get("source_delta_C_relaxation_sha256") == fixture.field_hashes["delta_C_relaxation"]
        and data.get("matrix_field_mapping", {}).get("matrix_baseline_array_key") == "matrix_baseline_xB"
        and data.get("matrix_field_mapping", {}).get("phi_reset") is False
        and data.get("matrix_field_mapping", {}).get("delta_C_relaxation_reset") is False,
        "no_clipping": data.get("matrix_field_mapping", {}).get("clipping_used") is False,
        "nonnegative_auxiliary": q_gp >= 0.0 and q_subgrid >= 0.0,
        "frozen_auxiliary": gp.frozen and subgrid.frozen,
        "unit_conversion": _close(float(ledger.get("C_B_total_mol_m3")), q_total / fixture.box_volume_m3, q_total / fixture.box_volume_m3, relative_tolerance),
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise FixtureConditionedHandoffError(f"v2 handoff validation failed: {failed}")
    residual = q_total - (q_matrix + q_resolved + q_gp + q_subgrid)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_FIXTURE_CONDITIONED_HANDOFF_V2",
        "contract_hash": contract.contract_hash,
        "fixture_hash": fixture.fixture_hash,
        "fixture_source_kind": fixture.source_kind,
        "checks": checks,
        "ledger": _ledger_mapping(
            q_total=q_total,
            q_matrix=q_matrix,
            q_gp=q_gp,
            q_subgrid=q_subgrid,
            q_resolved=q_resolved,
            box_volume_m3=fixture.box_volume_m3,
        ),
        "field_storage": {
            "Q_B_matrix_mol": q_matrix,
            "Q_B_beta_resolved_mol": q_resolved,
            "field_space_matrix_storage": float(np.sum(fixture.alpha * x_b, dtype=np.float64)),
            "field_space_resolved_storage": float(np.sum(fixture.h_phi * contract.v_B, dtype=np.float64)),
            "phi_sha256": fixture.field_hashes["phi"],
            "delta_C_relaxation_sha256": fixture.field_hashes["delta_C_relaxation"],
        },
        "double_count_check": {
            "status": "PASS_NO_DOUBLE_COUNT",
            "fixed_resolved_inventory_mol": q_resolved,
            "resolved_population_emitted_from_KWN": False,
            "matrix_includes_GP": False,
        },
        "relative_residual": abs(residual) / max(abs(q_total), 1.0e-300),
        "pf_raw_initialization_allowed": True,
        "pf_raw_initialization_emitted": False,
    }
    return result


def _package_hash_payload(metadata: Mapping[str, Any], arrays_sha256: str) -> Dict[str, Any]:
    payload = dict(metadata)
    payload.pop("package_hash", None)
    payload.pop("metadata_canonical_sha256", None)
    # The canonical package identity is based on the exact numeric array
    # manifest, not ZIP-container metadata such as a creation timestamp.  The
    # file hash is still recorded separately and checked on read.
    payload.pop("arrays_file_sha256", None)
    payload["array_content_manifest"] = payload.get("array_manifest", {})
    return payload


def write_fixture_conditioned_handoff_v2(
    directory: Path,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    fixture: FixtureState,
    contract: ValidationContract,
) -> Dict[str, Path]:
    """Validate then write a compact, reconstructable v2 handoff package."""

    output = Path(directory)
    if output.exists():
        raise FixtureConditionedHandoffError(f"refusing to overwrite {output}")
    report = validate_fixture_conditioned_handoff_v2(metadata, arrays, fixture, contract)
    output.mkdir(parents=True)
    array_path = output / ARRAYS_FILENAME
    np.savez_compressed(array_path, **{name: np.asarray(value, dtype=np.float64) for name, value in arrays.items()})
    arrays_hash = sha256_file(array_path)
    document: Dict[str, Any] = dict(metadata)
    document["arrays_file_sha256"] = arrays_hash
    document["pf_raw_initialization_allowed"] = bool(report["pf_raw_initialization_allowed"])
    document["package_hash"] = canonical_sha256(_package_hash_payload(document, arrays_hash))
    document["metadata_canonical_sha256"] = canonical_sha256(
        _package_hash_payload(document, arrays_hash)
    )
    metadata_path = output / METADATA_FILENAME
    metadata_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ledger = _require_mapping(document["ledger"], "ledger")
    rows = [
        _ledger_row("matrix", float(ledger["Q_B_matrix_mol"]), fixture.box_volume_m3),
        _ledger_row("GP", float(ledger["Q_B_GP_mol"]), fixture.box_volume_m3),
        _ledger_row("beta_subgrid", float(ledger["Q_B_beta_subgrid_mol"]), fixture.box_volume_m3),
        _ledger_row("beta_resolved_fixed", float(ledger["Q_B_beta_resolved_fixed_mol"]), fixture.box_volume_m3),
        _ledger_row("total", float(ledger["Q_B_total_mol"]), fixture.box_volume_m3),
    ]
    ledger_path = output / LEDGER_FILENAME
    header = ("bucket", "Q_B_mol", "C_B_mol_m3", "unit_Q_B", "unit_C_B")
    lines = [",".join(header)]
    lines.extend(
        ",".join(str(row[name]) for name in header)
        for row in rows
    )
    ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report_document = dict(report)
    report_document["package_hash"] = document["package_hash"]
    report_document["arrays_file_sha256"] = arrays_hash
    report_document["metadata_file_sha256"] = sha256_file(metadata_path)
    validation_path = output / VALIDATION_REPORT_FILENAME
    validation_path.write_text(json.dumps(report_document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "directory": output,
        "metadata": metadata_path,
        "arrays": array_path,
        "ledger": ledger_path,
        "validation_report": validation_path,
    }


def read_fixture_conditioned_handoff_v2(
    directory: Path,
    fixture: FixtureState,
    contract: ValidationContract,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, Any]]:
    """Round-trip a v2 package and re-run every field/ledger validation."""

    root = Path(directory)
    try:
        metadata = json.loads((root / METADATA_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FixtureConditionedHandoffError("cannot read v2 metadata") from error
    array_path = root / ARRAYS_FILENAME
    if metadata.get("arrays_file_sha256") != sha256_file(array_path):
        raise FixtureConditionedHandoffError("arrays file SHA-256 mismatch")
    with np.load(array_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name], dtype=np.float64) for name in loaded.files}
    manifest = metadata.get("array_manifest")
    if _array_manifest(arrays) != manifest:
        raise FixtureConditionedHandoffError("array manifest mismatch")
    expected_hash = canonical_sha256(_package_hash_payload(metadata, metadata["arrays_file_sha256"]))
    if metadata.get("package_hash") != expected_hash:
        raise FixtureConditionedHandoffError("package hash mismatch")
    report = validate_fixture_conditioned_handoff_v2(metadata, arrays, fixture, contract)
    return dict(metadata), arrays, report
