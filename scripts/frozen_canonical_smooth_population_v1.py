#!/usr/bin/env python3
"""Load the immutable canonical smooth KWN measure across runtime platforms.

The qualified 3200-cell measure was archived as a cell-integrated NPZ rather
than as a recipe whose last bits depend on platform libm/vector reductions.
This module verifies that archival input and rebuilds only the physical
configuration from the read-only fixture.  It never changes the canonical
measure, the radius domain, or the matrix inventory contract.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kwn_mvp.radius_grid import RadiusGrid  # noqa: E402
from kwn_mvp.solver import KWNSolver, SolverConfig  # noqa: E402


FROZEN_BINS = 3200
EXPECTED_CONTRACT_HASH = "d0ff02973ab0f737043e1a40d4f69893a469cbfe2bc4cd22f9e6a410bd0b1333"
EXPECTED_FIXTURE_HASH = "f1247cb66419af764b97de2f7843fc6de2049459d78550bc603edd8e88d9134f"
EXPECTED_CANONICAL_PSD_HASH = "f9bb99dab64de15b946fba8904efab30d1881833577b8a5b28c8ea46a5ccf608"
EXPECTED_CANONICAL_STATE_HASH = "45fc9cdd1a2e2ddfe357122dff26b202fb42893a627202c43904017e428aac95"
EXPECTED_CANONICAL_NPZ_SHA256 = "207e6bcef4ced2eab8a6239931ceadfe73a682e7d952f9f9d4750d5966e50d5d"
EXPECTED_CANONICAL_METADATA_SHA256 = "85f7702caf61fe62ee6f8310d9efc6efffccc91e37338af837b393a3faae7b76"

CANONICAL_NPZ = ROOT / "outputs" / "kwn_lower_boundary_time_accuracy_v1" / "canonical_smooth_population_v1.npz"
CANONICAL_METADATA = ROOT / "outputs" / "kwn_lower_boundary_time_accuracy_v1" / "canonical_smooth_population_v1.json"


class FrozenCanonicalStateError(RuntimeError):
    """Raised when the immutable canonical-state input cannot be verified."""


@dataclass(frozen=True)
class FrozenCanonicalContext:
    """Exact smooth initial state plus the shared physical KWN configuration."""

    solver: KWNSolver
    mapping: dict[str, Any]
    edges_m: np.ndarray
    cell_number_m3: np.ndarray
    contract_hash: str
    fixture_hash: str
    source_initial_psd_hash: str
    canonical_hash: str
    median_radius_m: float
    log_sigma: float
    cell_number_roundtrip_relative_error: float
    snapshot_provenance: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(np.asarray(array, dtype=np.float64)).tobytes())
    return digest.hexdigest()


def _read_metadata() -> dict[str, Any]:
    try:
        value = json.loads(CANONICAL_METADATA.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenCanonicalStateError(
            f"cannot read frozen canonical metadata {CANONICAL_METADATA}"
        ) from error
    if not isinstance(value, dict):
        raise FrozenCanonicalStateError("frozen canonical metadata must be a JSON object")
    return value


def _read_snapshot() -> tuple[dict[str, Any], dict[str, np.ndarray], str, str]:
    if not CANONICAL_NPZ.is_file():
        raise FrozenCanonicalStateError(f"missing frozen canonical NPZ {CANONICAL_NPZ}")
    npz_sha256 = _sha256_file(CANONICAL_NPZ)
    if npz_sha256 != EXPECTED_CANONICAL_NPZ_SHA256:
        raise FrozenCanonicalStateError(
            f"frozen canonical NPZ SHA-256 differs: {npz_sha256}"
        )
    metadata = _read_metadata()
    metadata_sha256 = _sha256_file(CANONICAL_METADATA)
    if metadata_sha256 != EXPECTED_CANONICAL_METADATA_SHA256:
        raise FrozenCanonicalStateError(
            f"frozen canonical metadata SHA-256 differs: {metadata_sha256}"
        )
    checks = {
        "schema": metadata.get("schema_version") == "SMOOTH_POPULATION_CANONICAL_V1",
        "npz_sha256": metadata.get("npz_sha256") == npz_sha256,
        "contract": metadata.get("validation_contract_hash") == EXPECTED_CONTRACT_HASH,
        "fixture": metadata.get("fixture_hash") == EXPECTED_FIXTURE_HASH,
        "state": metadata.get("canonical_hash") == EXPECTED_CANONICAL_STATE_HASH,
        "psd": metadata.get("source_initial_psd_hash") == EXPECTED_CANONICAL_PSD_HASH,
        "bins": metadata.get("canonical_radius_cell_count") == FROZEN_BINS,
        "measure": metadata.get("production_measure") == "cell_integrated",
    }
    if not all(checks.values()):
        raise FrozenCanonicalStateError(f"frozen canonical metadata check failed: {checks}")
    required = (
        "radius_edges_m",
        "cell_number_density_m3",
        "cell_M0_m3",
        "cell_M1_m2",
        "cell_M2_m",
        "cell_M3_dimensionless",
    )
    try:
        with np.load(CANONICAL_NPZ, allow_pickle=False) as archive:
            arrays = {
                key: np.asarray(archive[key], dtype=np.float64).copy()
                for key in required
            }
    except (OSError, KeyError, ValueError) as error:
        raise FrozenCanonicalStateError("cannot load frozen canonical NPZ arrays") from error
    edges = arrays["radius_edges_m"]
    cell_number = arrays["cell_number_density_m3"]
    moments = np.stack(
        [
            arrays["cell_M0_m3"],
            arrays["cell_M1_m2"],
            arrays["cell_M2_m"],
            arrays["cell_M3_dimensionless"],
        ]
    )
    if edges.shape != (FROZEN_BINS + 1,) or cell_number.shape != (FROZEN_BINS,):
        raise FrozenCanonicalStateError("frozen canonical NPZ array shapes are invalid")
    if moments.shape != (4, FROZEN_BINS):
        raise FrozenCanonicalStateError("frozen canonical cell-moment shapes are invalid")
    try:
        RadiusGrid(edges)
    except ValueError as error:
        raise FrozenCanonicalStateError("frozen canonical radius edges are invalid") from error
    if not all(np.all(np.isfinite(array)) for array in (cell_number, moments)) or np.any(cell_number < 0.0):
        raise FrozenCanonicalStateError("frozen canonical state contains non-finite or negative values")
    # These arrays are the archived identity.  Do not substitute moments
    # recomputed from ``edges`` and ``cell_number`` here: last-bit libm and
    # reduction differences are precisely why this frozen input exists.
    canonical_hash = _array_hash(edges, cell_number, moments)
    if canonical_hash != EXPECTED_CANONICAL_STATE_HASH:
        raise FrozenCanonicalStateError(
            f"frozen canonical state hash differs: {canonical_hash}"
        )
    return metadata, arrays, npz_sha256, metadata_sha256


def frozen_canonical_snapshot_provenance() -> dict[str, Any]:
    """Return verified immutable-input provenance without deriving a new PSD."""

    metadata, _arrays, npz_sha256, metadata_sha256 = _read_snapshot()
    return {
        "canonical_state_npz": str(CANONICAL_NPZ.relative_to(ROOT)),
        "canonical_state_npz_sha256": npz_sha256,
        "canonical_state_metadata": str(CANONICAL_METADATA.relative_to(ROOT)),
        "canonical_state_metadata_sha256": metadata_sha256,
        "canonical_state_hash": metadata["canonical_hash"],
        "canonical_psd_hash": metadata["source_initial_psd_hash"],
        "canonical_matrix_xB": metadata["matrix_xB"],
        "canonical_total_inventory_mol_m3": metadata["total_inventory_mol_m3"],
    }


def _load_radius_grid_qualification_runner() -> Any:
    module_name = "_kwn_frozen_canonical_radius_grid_v1"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    path = ROOT / "scripts" / "run_kwn_radius_grid_convergence_v1.py"
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise FrozenCanonicalStateError(f"cannot load fixture configuration builder {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def build_frozen_canonical_context() -> FrozenCanonicalContext:
    """Build a solver from the verified archived measure and shared physics."""

    metadata, arrays, npz_sha256, metadata_sha256 = _read_snapshot()
    runner = _load_radius_grid_qualification_runner()
    _base_solver, construction, contract, fixture = runner._build_fixture_solver(bins=FROZEN_BINS)
    if str(contract.contract_hash) != EXPECTED_CONTRACT_HASH:
        raise FrozenCanonicalStateError("frozen validation-contract hash differs")
    if str(fixture.fixture_hash) != EXPECTED_FIXTURE_HASH:
        raise FrozenCanonicalStateError("frozen fixture hash differs")

    edges = arrays["radius_edges_m"]
    cell_number = arrays["cell_number_density_m3"]
    mapping = deepcopy(construction["config_mapping"])
    mapping["simulation"]["population_measure"] = "cell_integrated"
    mapping["simulation"]["accuracy_radius_cfl"] = None
    mapping["simulation"]["accuracy_active_radius_cfl"] = None
    mapping["radius_grid"] = {
        **mapping["radius_grid"],
        "minimum_m": float(edges[0]),
        "maximum_m": float(edges[-1]),
        "bins": FROZEN_BINS,
        "edges_m": [float(value) for value in edges],
    }
    mapping["populations"]["beta"]["initial"] = {
        "kind": "cell_integrated",
        "radius_edges_m": [float(value) for value in edges],
        "cell_number_density_m3": [float(value) for value in cell_number],
    }
    mapping["matrix"]["initial_xB"] = float(metadata["matrix_xB"])
    mapping["matrix"]["total_b_mol_m3"] = float(metadata["total_inventory_mol_m3"])

    solver = KWNSolver(SolverConfig.from_mapping(mapping))
    beta = solver.population("beta")
    if not np.array_equal(beta.grid.edges_m, edges):
        raise FrozenCanonicalStateError("solver grid differs from frozen canonical edges")
    observed = beta.number_density_per_m4 * beta.grid.widths_m
    roundtrip = float(
        np.max(np.abs(observed - cell_number) / np.maximum(np.abs(cell_number), 1.0e-300))
    )
    if roundtrip > 2.1e-15:
        raise FrozenCanonicalStateError(
            f"frozen cell-integrated round-trip differs by {roundtrip:.3e}"
        )
    if solver.matrix_xb != float(metadata["matrix_xB"]):
        raise FrozenCanonicalStateError("solver matrix composition differs from frozen canonical state")
    ledger = solver.ledger.snapshot(matrix_xb=solver.matrix_xb, populations=solver.population_list())
    if ledger.relative_residual > 1.0e-12:
        raise FrozenCanonicalStateError(
            f"frozen canonical initial ledger residual is {ledger.relative_residual:.3e}"
        )

    return FrozenCanonicalContext(
        solver=solver,
        mapping=mapping,
        edges_m=edges,
        cell_number_m3=cell_number,
        contract_hash=str(contract.contract_hash),
        fixture_hash=str(fixture.fixture_hash),
        source_initial_psd_hash=str(metadata["source_initial_psd_hash"]),
        canonical_hash=str(metadata["canonical_hash"]),
        median_radius_m=float(metadata["median_radius_m"]),
        log_sigma=float(metadata["log_sigma"]),
        cell_number_roundtrip_relative_error=roundtrip,
        snapshot_provenance={
            "canonical_state_npz": str(CANONICAL_NPZ.relative_to(ROOT)),
            "canonical_state_npz_sha256": npz_sha256,
            "canonical_state_metadata": str(CANONICAL_METADATA.relative_to(ROOT)),
            "canonical_state_metadata_sha256": metadata_sha256,
            "canonical_state_hash": str(metadata["canonical_hash"]),
            "canonical_psd_hash": str(metadata["source_initial_psd_hash"]),
        },
    )
