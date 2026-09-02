#!/usr/bin/env python3
"""Close the KWN time-reference gate with a conservative CR1 remap.

This is a KWN-only numerical adjudication runner.  It preserves the exact
full-domain SSPRK2 donor-bound block, then treats the independent conservative
characteristic remap (CR1: piecewise-constant cumulative remap with a
deterministic autonomous-radius Gauss--Legendre backward trace) as a candidate
reference only after its own tests, restart,
and timestep-convergence gates pass.  PF/CUDA is read only and is never
launched from this program.

The long 48 h stages are deliberately downstream of every short-horizon
numerical gate.  Use a suitably large wall/step budget (normally from the
cluster) for a formal full run; a bounded diagnostic must remain an honest
incomplete result rather than a pass.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kwn_mvp.characteristic_reference import (  # noqa: E402
    REMAP_ORDER,
    TRACE_INTEGRATOR,
    CharacteristicReferenceSolver,
)
from kwn_mvp.conservative_remap import (  # noqa: E402
    ConservativeRemapError,
    conservative_remap_piecewise_constant,
    trace_departure_faces_rk2,
)
from kwn_mvp.cohort_solver import CohortSolver  # noqa: E402
from kwn_mvp.diagnostics import discrete_wasserstein_distance  # noqa: E402
from kwn_mvp.growth import growth_rate_m_s  # noqa: E402
from kwn_mvp.lower_boundary import (  # noqa: E402
    boundary_growth_velocity,
    boundary_inventory_diagnostic,
    boundary_radius,
    particle_inventory_at_radius,
)
from kwn_mvp.population_metrics import (  # noqa: E402
    cell_moments_from_piecewise_constant_cells,
    metrics_from_discrete_measure,
    metrics_from_piecewise_constant_cells,
    positive_cell_quadrature,
)
from kwn_mvp.solver import KWNSolver, RadiusGridOverflowError, SolverConfig, SolverStateError  # noqa: E402


TASK_NAME = "kwn_characteristic_reference_v1"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / TASK_NAME
DEFAULT_REPORT_ROOT = ROOT / "reports" / TASK_NAME
FROZEN_START_COMMIT = "9269e07a2d0fafbf35be950b858373fb71b54e4b"
REQUIRED_BRANCH = "codex/kwn-characteristic-reference-v1"
EXPECTED_CONTRACT_HASH = "d0ff02973ab0f737043e1a40d4f69893a469cbfe2bc4cd22f9e6a410bd0b1333"
EXPECTED_CANONICAL_PSD_HASH = "f9bb99dab64de15b946fba8904efab30d1881833577b8a5b28c8ea46a5ccf608"
EXPECTED_CANONICAL_STATE_HASH = "45fc9cdd1a2e2ddfe357122dff26b202fb42893a627202c43904017e428aac95"
EXPECTED_FIXTURE_HASH = "f1247cb66419af764b97de2f7843fc6de2049459d78550bc603edd8e88d9134f"
EXPECTED_CURRENT_CAP4_DISSOLUTION_RELATIVE_DIFFERENCE = 0.042544133625013775
TWO_PERCENT = 0.02
ONE_PERCENT = 0.01
REFERENCE_TIME_GATE = 0.0025
COHORT_POINTS_PER_CELL = 4
PSD_GATE_METRIC = "PSD_Wasserstein_relative_to_Rmean"
SHORT_TIMES_H = (0.0, 0.01, 0.05, 0.1)
ONE_HOUR_TIMES_H = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
FULL_TIMES_H = (0.0, 0.1, 0.39317699499770825, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0)
IMPLICIT_POLICIES: tuple[tuple[str, float | None], ...] = (
    ("current", None),
    ("cap4", 4.0),
    ("cap2", 2.0),
    ("cap1", 1.0),
    ("cap05", 0.5),
    ("cap025", 0.25),
)
BASELINE_TEST_MODULES = (
    "tests.kwn.test_active_cfl_contract",
    "tests.kwn.test_beta_only_same_contract_control",
    "tests.kwn.test_cohort_solver",
    "tests.kwn.test_composition_mapping",
    "tests.kwn.test_conservative_positivity_repair",
    "tests.kwn.test_lower_boundary_contract",
    "tests.kwn.test_numerical_gates",
    "tests.kwn.test_pf_validation_contract",
    "tests.kwn.test_population_metrics",
    "tests.kwn.test_population_radius_moment",
    "tests.kwn.test_rmin_boundary",
    "tests.kwn.test_thermo_contract_gate",
)
STRICT_SSPRK2_TEST_MODULES = ("tests.kwn.test_explicit_ssprk2_solver",)
PRIMARY_METRICS = (
    "N_m0_m3",
    "M0_m3",
    "M1_m2",
    "M2_m",
    "M3_dimensionless",
    "Rmean_m",
    "Rmean3_m3",
    "mean_R3_m3",
    "Sv_m_inv",
    "f_beta",
    "matrix_xB",
    "cumulative_number_dissolution_m3",
    "cumulative_beta_volume_dissolution",
    "cumulative_mol_B_returned_mol_m3",
)
REQUIRED_REPORTS = {
    "00_explicit_reference_blocker_reproduction.md": "Strict explicit-reference blocker reproduction",
    "01_characteristic_method_contract.md": "Conservative characteristic method contract",
    "02_characteristic_unit_tests.md": "Characteristic reference unit tests",
    "03_analytic_characteristic_benchmarks.md": "Analytic characteristic benchmarks",
    "04_characteristic_self_convergence.md": "Characteristic self-convergence",
    "05_cohort_characteristic_parity.md": "Cohort--characteristic parity",
    "06_frozen_matrix_three_method.md": "Frozen-matrix three-method benchmark",
    "07_implicit_reference_ladder_01h.md": "Implicit--reference ladder at 0.1 h",
    "08_implicit_reference_ladder_1h.md": "Implicit--reference confirmation at 1 h",
    "09_solver_cost_and_decision.md": "Solver cost and decision",
    "10_population_solver_v2_qualification.md": "Population solver V2 qualification",
    "11_authority_v2.md": "Eulerian smooth authority V2",
    "12_final_smooth_crosscheck.md": "Final smooth three-method crosscheck",
    "13_beta_only_cohort_pf.md": "Beta-only cohort--PF comparison",
    "14_model_role_boundary.md": "Model role boundary",
    "15_final_acceptance_report.md": "Final acceptance report",
    "16_reproduction_commands.md": "Reproduction commands",
}


@dataclass
class RunResult:
    """A state archive for one honest finite numerical trajectory."""

    name: str
    status: str
    reason: str | None
    snapshots: list[dict[str, Any]]
    measures: list[tuple[np.ndarray, np.ndarray]]
    trace_rows: list[dict[str, Any]]
    accepted_steps: int
    runtime_s: float
    max_inventory_relative_residual: float
    max_fixed_point_residual: float
    resident_state_bytes: int = 0


@dataclass
class CohortTrajectoryCache:
    """One monotone canonical cohort trajectory shared by every downstream gate."""

    cohort: CohortSolver
    points_per_cell: int
    started_s: float
    max_wall_s: float
    snapshots_by_time_h: dict[float, tuple[dict[str, Any], tuple[np.ndarray, np.ndarray]]]


@dataclass
class GridContext:
    """The canonical smooth law reconstructed on a required radius grid."""

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


class WorkflowError(RuntimeError):
    """A fail-closed orchestration error with no physical fallback."""


def _is_diagnostic_budget_incomplete(run: RunResult) -> bool:
    """Distinguish a bounded diagnostic from a completed numerical failure."""

    reason = str(run.reason or "")
    return str(run.status).startswith("INCOMPLETE") and (
        "max_steps=" in reason or "wall budget" in reason or "max_wall_s=" in reason
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(np.asarray(array, dtype=np.float64)).tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(dict(value)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, fallback_fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    if not fields:
        fields = list(fallback_fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _write_report(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )
    return result.stdout.strip()


def _git_succeeds(*args: str) -> bool:
    return subprocess.run(
        ["git", *args], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    ).returncode == 0


def _require_formal_launch(output_root: Path, report_root: Path) -> None:
    """Reject mutable sources or overwritten evidence before any numerical work."""

    if _git("status", "--short"):
        raise WorkflowError("formal characteristic-reference launch requires a clean source tree")
    if _git("branch", "--show-current") != REQUIRED_BRANCH:
        raise WorkflowError(f"formal launch requires branch {REQUIRED_BRANCH!r}")
    if not _git_succeeds("merge-base", "--is-ancestor", FROZEN_START_COMMIT, "HEAD"):
        raise WorkflowError("formal launch HEAD is not descended from the frozen 9269e07 start")
    for path, label in ((output_root, "output"), (report_root, "report")):
        if path.exists() and any(path.iterdir()):
            raise WorkflowError(f"formal {label} root already contains evidence: {path}")


def _runtime_context() -> dict[str, Any]:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False
    ).stdout
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "logical_cpu_count": os.cpu_count(),
        "allocated_threads": os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("OMP_NUM_THREADS") or "1",
        "pip_freeze_sha256": hashlib.sha256(freeze.encode("utf-8")).hexdigest(),
    }


def _launch_context() -> dict[str, Any]:
    return {
        "frozen_start_commit": FROZEN_START_COMMIT,
        "frozen_start_commit_resolved": _git("rev-parse", FROZEN_START_COMMIT),
        "git_head_at_launch": _git("rev-parse", "HEAD"),
        "git_branch_at_launch": _git("branch", "--show-current"),
        "git_status_at_launch": _git("status", "--short"),
        "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime": _runtime_context(),
        "pf_source_modified": False,
        "cuda_rerun": False,
        "physical_retuning": False,
        "local_gp_release": False,
    }


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise WorkflowError(f"cannot import helper module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_legacy_lower() -> Any:
    return _load_module(ROOT / "scripts" / "run_kwn_lower_boundary_time_accuracy_v1.py", "kwn_lower_for_characteristic_v1")


def _load_time_closure() -> Any:
    return _load_module(ROOT / "scripts" / "run_kwn_time_accuracy_closure_v1.py", "kwn_time_for_characteristic_v1")


def _flatten_suite(suite: unittest.TestSuite) -> list[unittest.TestCase]:
    result: list[unittest.TestCase] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            result.extend(_flatten_suite(item))
        else:
            result.append(item)
    return result


def _run_test_modules(modules: Sequence[str]) -> dict[str, Any]:
    command = [sys.executable, "-m", "unittest", *modules]
    environment = dict(os.environ)
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = source_path + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    started = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment, check=False)
    match = re.search(r"Ran (\d+) tests", result.stdout)
    return {
        "command": command,
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "returncode": result.returncode,
        "test_count": None if match is None else int(match.group(1)),
        "runtime_s": time.monotonic() - started,
        "output": result.stdout[-12000:],
    }


def _run_characteristic_contract_tests() -> dict[str, Any]:
    """Run CR1--CR9 in-process so the output names each analytic gate."""

    module_name = "tests.kwn.test_characteristic_reference"
    try:
        module = importlib.import_module(module_name)
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        tests = _flatten_suite(suite)
    except Exception:
        return {
            "status": "FAIL_CHARACTERISTIC_REFERENCE_NUMERICS",
            "rows": [{"case": "CR1--CR9", "status": "ERROR", "reason": traceback.format_exc()[-4000:]}],
            "test_count": 0,
        }
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    failures = {case.id(): text for case, text in result.failures}
    errors = {case.id(): text for case, text in result.errors}
    skipped = {case.id(): text for case, text in result.skipped}
    rows: list[dict[str, Any]] = []
    for case in tests:
        identifier = case.id()
        match = re.search(r"test_(cr[1-9])_", identifier)
        label = match.group(1).upper() if match else "CR1_RMAX_FAIL_CLOSED" if "rmax" in identifier else "ADDITIONAL"
        if identifier in failures:
            status, reason = "FAIL", failures[identifier][-4000:]
        elif identifier in errors:
            status, reason = "ERROR", errors[identifier][-4000:]
        elif identifier in skipped:
            status, reason = "SKIPPED", skipped[identifier][-4000:]
        else:
            status, reason = "PASS", ""
        rows.append({"case": label, "test_id": identifier, "status": status, "reason": reason})
    passed = result.wasSuccessful()
    return {
        "status": "PASS_CHARACTERISTIC_REFERENCE_UNIT_TESTS" if passed else "FAIL_CHARACTERISTIC_REFERENCE_NUMERICS",
        "test_count": result.testsRun,
        "rows": rows,
        "runner_output": stream.getvalue()[-12000:],
    }


def _relative_or_absolute_error(value: float, reference: float) -> float:
    if value == 0.0 and reference == 0.0:
        return 0.0
    if reference == 0.0:
        return abs(value)
    return abs(value - reference) / max(abs(reference), 1.0e-300)


def _strict_explicit_blocker(context: Any) -> dict[str, Any]:
    """Preserve the exact full-domain donor limiter, including cell zero."""

    from kwn_mvp.explicit_solver import ExplicitSSPRK2Solver

    solver = ExplicitSSPRK2Solver(SolverConfig.from_mapping(context.mapping), donor_safety=1.0)
    beta = solver.population("beta")
    cell_number = beta.number_density_per_m4 * beta.grid.widths_m
    cell_moments = cell_moments_from_piecewise_constant_cells(beta.grid.edges_m, cell_number)
    donor = solver.donor_bound_diagnostics()
    operator = solver._stage_operator(solver._state_from_populations(), solver.matrix_xb, solver.time_s)
    faces = operator.face_velocity_m_s[donor.population_name]
    density = solver.population(donor.population_name).number_density_per_m4
    index = int(donor.cell_index)
    left_speed = max(-float(faces[index]), 0.0)
    right_speed = max(float(faces[index + 1]), 0.0)
    face_index = index if left_speed >= right_speed else index + 1
    direction = "lower" if face_index == index else "upper"
    proposed_rows: list[dict[str, Any]] = []
    for proposal_index in range(1, 21):
        before = solver.state_arrays()
        started = time.monotonic()
        try:
            diagnostic = solver.advance_one(maximum_dt_s=0.1 * 3600.0)
            status, reason, diagnostic_dt = "UNEXPECTED_ACCEPTED", "", float(diagnostic.dt_s)
        except SolverStateError as error:
            status, reason, diagnostic_dt = "REJECTED_BELOW_FROZEN_MIN_DT", str(error), 0.0
        unchanged = all(np.array_equal(before[key], solver.state_arrays()[key]) for key in before)
        proposed_rows.append(
            {
                "proposal_index": proposal_index,
                "status": status,
                "reason": reason,
                "strict_donor_dt_s": float(donor.bound_s),
                "proposed_dt_s": float(donor.bound_s),
                "frozen_min_dt_s": float(solver.config.min_dt_s),
                "accepted_dt_s": diagnostic_dt,
                "state_unchanged_after_rejection": unchanged,
                "rejected_step_count": int(solver.rejected_step_count),
                "runtime_s": time.monotonic() - started,
            }
        )
    all_rejected = all(row["status"] == "REJECTED_BELOW_FROZEN_MIN_DT" for row in proposed_rows)
    all_unchanged = all(bool(row["state_unchanged_after_rejection"]) for row in proposed_rows)
    return {
        "schema_version": "KWN_EXACT_DONOR_BOUND_BLOCKER_REPRODUCTION_V1",
        "status": (
            "BLOCKED_EXACT_DONOR_BOUND_REFERENCE"
            if all_rejected and all_unchanged
            else "FAIL_EXPLICIT_BLOCKER_REPRODUCTION"
        ),
        "full_domain_exact_donor_bound": True,
        "active_tail_proxy_used": False,
        "first_cell_number_m3": float(cell_number[0]),
        "first_cell_M0_fraction": float(cell_moments[0, 0] / np.sum(cell_moments[0], dtype=np.float64)),
        "first_cell_M3_fraction": float(cell_moments[3, 0] / np.sum(cell_moments[3], dtype=np.float64)),
        "strict_donor_dt_s": float(donor.bound_s),
        "frozen_min_dt_s": float(solver.config.min_dt_s),
        "limiting_population": donor.population_name,
        "limiting_cell_index": index,
        "limiting_cell_number_m3": float(donor.donor_number_m3),
        "limiting_face_index": face_index,
        "limiting_face_direction": direction,
        "limiting_face_velocity_m_s": float(faces[face_index]),
        "limiting_cell_left_outflow_velocity_m_s": left_speed,
        "limiting_cell_right_outflow_velocity_m_s": right_speed,
        "limiting_outgoing_flux_m3_s": float(donor.outgoing_flux_m3_s),
        "limiting_cell_density_per_m4": float(density[index]),
        "theoretical_steps_to_0p1h": float(360.0 / donor.bound_s),
        "theoretical_steps_to_48h": float(48.0 * 3600.0 / donor.bound_s),
        "first_twenty_proposed_steps": proposed_rows,
        "all_twenty_rejected_below_frozen_min_dt": all_rejected,
        "all_twenty_left_accepted_state_unchanged": all_unchanged,
        "reason": (
            "The positive first canonical beta cell is included in the exact whole-domain donor bound. "
            "Its unmodified donor timestep is below the frozen min_dt, so strict SSPRK2 cannot supply a 0.1 h reference."
        ),
    }


def _snapshot_from_cells(
    *,
    solver: KWNSolver,
    cell_number_m3: np.ndarray,
    policy: str,
    cumulative_number: float,
    cumulative_volume: float,
    cumulative_mol_b: float,
    inventory_override: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray]]:
    beta = solver.population("beta")
    metrics = metrics_from_piecewise_constant_cells(beta.grid.edges_m, cell_number_m3)
    if inventory_override is None:
        ledger = solver.ledger.snapshot(
            matrix_xb=solver.matrix_xb, populations=solver.population_list(), beta_resolved_fraction=1.0
        )
        inventory = {
            "beta_inventory_mol_m3": float(ledger.beta_resolved_mol_m3),
            "matrix_inventory_mol_m3": float(ledger.matrix_mol_m3),
            "total_inventory_mol_m3": float(ledger.total_mol_m3),
            "inventory_residual_mol_m3": float(ledger.residual_mol_m3),
            "inventory_relative_residual": float(ledger.relative_residual),
        }
    else:
        inventory = dict(inventory_override)
    radii, weights = positive_cell_quadrature(beta.grid.edges_m, cell_number_m3, 2)
    row = {
        "policy": policy,
        "time_s": float(solver.time_s),
        "time_h": float(solver.time_s / 3600.0),
        "accepted_steps": int(solver.step),
        **metrics.as_dict(),
        "Rmean_m": metrics.Rmean_number_m,
        "Rmean3_m3": metrics.Rmean_cubed_m3,
        "matrix_xB": float(solver.matrix_xb),
        **inventory,
        "cumulative_number_dissolution_m3": float(cumulative_number),
        "cumulative_beta_volume_dissolution": float(cumulative_volume),
        "cumulative_mol_B_returned_mol_m3": float(cumulative_mol_b),
        "first_cell_number_m3": float(cell_number_m3[0]),
    }
    return row, (radii, weights)


def _cohort_snapshot(cohort: CohortSolver, *, policy: str) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray]]:
    snapshot = cohort.snapshot()
    row = snapshot.as_dict()
    row.update(
        {
            "policy": policy,
            "time_h": float(snapshot.time_s / 3600.0),
            "Rmean_m": float(snapshot.Rmean_number_m),
            "Rmean3_m3": float(snapshot.Rmean_cubed_m3),
            "mean_R3_m3": float(snapshot.mean_R3_m3),
            "cumulative_number_dissolution_m3": float(snapshot.cumulative_number_dissolved_m3),
            "cumulative_beta_volume_dissolution": float(snapshot.cumulative_beta_volume_dissolved),
            "cumulative_mol_B_returned_mol_m3": float(snapshot.cumulative_mol_B_returned_mol_m3),
        }
    )
    active = [item for item in cohort.cohorts if item.active]
    radii = np.asarray([item.radius_m for item in active], dtype=np.float64)
    weights = np.asarray([item.weight_m3 for item in active], dtype=np.float64)
    return row, (radii, weights)


def _append_terminal_snapshot(
    result: RunResult,
    snapshot: dict[str, Any],
    measure: tuple[np.ndarray, np.ndarray],
) -> None:
    if not result.snapshots or float(result.snapshots[-1]["time_s"]) != float(snapshot["time_s"]):
        snapshot["snapshot_status"] = "INCOMPLETE_TERMINAL"
        result.snapshots.append(snapshot)
        result.measures.append(measure)


def _resident_state_bytes(solver: Any) -> int:
    """Byte count of the accepted in-memory state arrays, not a guess."""

    arrays = solver.state_arrays()
    return int(sum(np.asarray(value).nbytes for value in arrays.values()))


def _run_characteristic(
    context: Any,
    *,
    policy: str,
    dt_s: float,
    target_times_h: Sequence[float],
    max_steps: int,
    max_wall_s: float,
) -> RunResult:
    """Run fully coupled CR1 only; no frozen-x single step is promoted."""

    solver = CharacteristicReferenceSolver(SolverConfig.from_mapping(context.mapping))
    snapshots: list[dict[str, Any]] = []
    measures: list[tuple[np.ndarray, np.ndarray]] = []
    trace: list[dict[str, Any]] = []
    started = time.monotonic()
    max_inventory = 0.0
    max_fp = 0.0
    cumulative_number = cumulative_volume = cumulative_mol_b = 0.0
    try:
        targets = tuple(float(item) * 3600.0 for item in target_times_h)
        if not targets or targets[0] != 0.0:
            raise WorkflowError("characteristic output schedule must start at t=0")
        initial, initial_measure = _snapshot_from_cells(
            solver=solver,
            cell_number_m3=solver._beta_cell_numbers(),
            policy=policy,
            cumulative_number=0.0,
            cumulative_volume=0.0,
            cumulative_mol_b=0.0,
        )
        snapshots.append(initial)
        measures.append(initial_measure)
        for target_s in targets[1:]:
            while solver.time_s < target_s:
                if solver.step >= max_steps:
                    raise WorkflowError(f"characteristic max_steps={max_steps} reached before {target_s / 3600.0:g} h")
                if time.monotonic() - started >= max_wall_s:
                    raise WorkflowError(f"characteristic wall budget max_wall_s={max_wall_s:g} exhausted")
                diagnostic = solver.advance_one(maximum_dt_s=min(float(dt_s), target_s - solver.time_s))
                cumulative_number = float(solver.cumulative_number_dissolution_m3)
                cumulative_volume = float(solver.cumulative_beta_volume_dissolution)
                cumulative_mol_b = float(solver.cumulative_mol_b_returned_mol_m3)
                max_inventory = max(max_inventory, float(diagnostic.inventory.relative_residual))
                max_fp = max(max_fp, float(diagnostic.fixed_point_xb_residual), float(diagnostic.fixed_point_population_residual))
                trace.append(
                    {
                        "policy": policy,
                        "step": int(diagnostic.step),
                        "time_s": float(diagnostic.time_s),
                        "time_h": float(diagnostic.time_s / 3600.0),
                        "dt_s": float(diagnostic.dt_s),
                        "fixed_point_iterations": int(diagnostic.fixed_point_iterations),
                        "fixed_point_xb_residual": float(diagnostic.fixed_point_xb_residual),
                        "fixed_point_population_residual": float(diagnostic.fixed_point_population_residual),
                        "fixed_point_cell_measure_residual": float(diagnostic.fixed_point_cell_measure_residual),
                        "fixed_point_convergence_rate": float(diagnostic.fixed_point_convergence_rate),
                        "inventory_relative_residual": float(diagnostic.inventory.relative_residual),
                        "rmin_number_loss_m3": float(diagnostic.rmin_number_loss_m3),
                        "rmin_mol_b_loss_mol_m3": float(diagnostic.rmin_mol_b_loss_mol_m3),
                        "remap_number_conservation_residual_m3": float(diagnostic.remap_number_conservation_residual_m3),
                        "lower_no_inflow_face_count": int(diagnostic.lower_no_inflow_face_count),
                        "upper_no_inflow_face_count": int(diagnostic.upper_no_inflow_face_count),
                    }
                )
            row, measure = _snapshot_from_cells(
                solver=solver,
                cell_number_m3=solver._beta_cell_numbers(),
                policy=policy,
                cumulative_number=cumulative_number,
                cumulative_volume=cumulative_volume,
                cumulative_mol_b=cumulative_mol_b,
            )
            snapshots.append(row)
            measures.append(measure)
    except Exception as error:
        row, measure = _snapshot_from_cells(
            solver=solver,
            cell_number_m3=solver._beta_cell_numbers(),
            policy=policy,
            cumulative_number=float(solver.cumulative_number_dissolution_m3),
            cumulative_volume=float(solver.cumulative_beta_volume_dissolution),
            cumulative_mol_b=float(solver.cumulative_mol_b_returned_mol_m3),
        )
        result = RunResult(
            policy,
            "INCOMPLETE_CHARACTERISTIC_RUN",
            f"{type(error).__name__}: {error}",
            snapshots,
            measures,
            trace,
            solver.step,
            time.monotonic() - started,
            max_inventory,
            max_fp,
            _resident_state_bytes(solver),
        )
        _append_terminal_snapshot(result, row, measure)
        return result
    return RunResult(
        policy,
        "PASS_CHARACTERISTIC_RUN",
        None,
        snapshots,
        measures,
        trace,
        solver.step,
        time.monotonic() - started,
        max_inventory,
        max_fp,
        _resident_state_bytes(solver),
    )


def _new_cohort_trajectory_cache(
    legacy: Any,
    context: Any,
    *,
    points_per_cell: int,
    max_wall_s: float,
) -> CohortTrajectoryCache:
    cohort = legacy._cohort_from_context(context, points_per_cell=points_per_cell)
    cache = CohortTrajectoryCache(
        cohort=cohort,
        points_per_cell=int(points_per_cell),
        started_s=time.monotonic(),
        max_wall_s=float(max_wall_s),
        snapshots_by_time_h={},
    )
    row, measure = _cohort_snapshot(cohort, policy="cohort_cache")
    cache.snapshots_by_time_h[0.0] = (row, measure)
    return cache


def _run_cohort(
    legacy: Any,
    context: Any,
    *,
    policy: str,
    target_times_h: Sequence[float],
    points_per_cell: int,
    max_wall_s: float,
    cache: CohortTrajectoryCache | None = None,
    archive_times_h: Sequence[float] | None = None,
) -> RunResult:
    """Archive a subset of one cached, monotone event-aware cohort trajectory.

    The 0.1 h, 1 h, and final 48 h gates must not independently recreate the
    expensive 4-point/cell cohort solve.  Later gates extend this cache only
    forward in time; callers can request extra future output instants before
    they are crossed so the final schedule remains available without reruns.
    """

    requested = tuple(sorted({float(value) for value in target_times_h}))
    archive = tuple(sorted({float(value) for value in (archive_times_h or target_times_h)}))
    if not requested or requested[0] != 0.0 or not archive or archive[0] != 0.0:
        raise WorkflowError("cohort output schedules must start at t=0")
    if any(not math.isfinite(value) or value < 0.0 for value in (*requested, *archive)):
        raise WorkflowError("cohort output schedules must contain finite non-negative times")

    try:
        if cache is None:
            cache = _new_cohort_trajectory_cache(
                legacy, context, points_per_cell=points_per_cell, max_wall_s=max_wall_s
            )
        elif cache.points_per_cell != int(points_per_cell):
            raise WorkflowError("cohort cache quadrature does not match the requested policy")

        for time_h in archive:
            key = round(time_h, 12)
            if key in cache.snapshots_by_time_h:
                continue
            target_s = time_h * 3600.0
            if target_s < cache.cohort.time_s:
                raise WorkflowError(
                    "cohort cache has already crossed a requested output time without archiving it"
                )
            if time.monotonic() - cache.started_s >= cache.max_wall_s:
                raise WorkflowError(f"cohort wall budget max_wall_s={cache.max_wall_s:g} exhausted")
            cache.cohort.advance_to(target_s)
            row, measure = _cohort_snapshot(cache.cohort, policy="cohort_cache")
            cache.snapshots_by_time_h[key] = (row, measure)

        snapshots: list[dict[str, Any]] = []
        measures: list[tuple[np.ndarray, np.ndarray]] = []
        for time_h in requested:
            row, measure = cache.snapshots_by_time_h[round(time_h, 12)]
            archived_row = dict(row)
            archived_row["policy"] = policy
            snapshots.append(archived_row)
            measures.append(measure)
    except Exception as error:
        available: list[tuple[float, tuple[dict[str, Any], tuple[np.ndarray, np.ndarray]]]] = []
        if cache is not None:
            available = sorted(cache.snapshots_by_time_h.items())
        snapshots = []
        measures = []
        for _, (row, measure) in available:
            archived_row = dict(row)
            archived_row["policy"] = policy
            archived_row["snapshot_status"] = "INCOMPLETE_TERMINAL"
            snapshots.append(archived_row)
            measures.append(measure)
        return RunResult(
            policy,
            "INCOMPLETE_COHORT_RUN",
            f"{type(error).__name__}: {error}",
            snapshots,
            measures,
            [],
            0 if cache is None else int(cache.cohort.accepted_segment_count),
            0.0 if cache is None else time.monotonic() - cache.started_s,
            float("inf"),
            float("nan"),
        )
    residual = max((float(row["inventory_relative_residual"]) for row in snapshots), default=0.0)
    return RunResult(
        policy,
        "PASS_COHORT_RUN",
        None,
        snapshots,
        measures,
        [],
        int(cache.cohort.accepted_segment_count),
        time.monotonic() - cache.started_s,
        residual,
        0.0,
    )


def _run_implicit(
    legacy: Any,
    context: Any,
    *,
    policy: str,
    active_cfl_cap: float | None,
    target_times_h: Sequence[float],
    max_steps: int,
    max_wall_s: float,
) -> RunResult:
    """Archive actual implicit states; this never substitutes a static CFL audit."""

    mapping = deepcopy(context.mapping)
    mapping["simulation"]["accuracy_active_radius_cfl"] = active_cfl_cap
    solver = KWNSolver(SolverConfig.from_mapping(mapping))
    snapshots: list[dict[str, Any]] = []
    measures: list[tuple[np.ndarray, np.ndarray]] = []
    trace: list[dict[str, Any]] = []
    started = time.monotonic()
    cumulative_number = cumulative_volume = cumulative_mol_b = 0.0
    max_inventory = 0.0
    try:
        targets = tuple(float(item) * 3600.0 for item in target_times_h)
        if not targets or targets[0] != 0.0:
            raise WorkflowError("implicit output schedule must start at t=0")
        row, measure = _snapshot_from_cells(
            solver=solver,
            cell_number_m3=solver.population("beta").number_density_per_m4 * solver.population("beta").grid.widths_m,
            policy=policy,
            cumulative_number=0.0,
            cumulative_volume=0.0,
            cumulative_mol_b=0.0,
        )
        snapshots.append(row)
        measures.append(measure)
        for target_s in targets[1:]:
            while solver.time_s < target_s:
                if solver.step >= max_steps:
                    raise WorkflowError(f"implicit max_steps={max_steps} reached before {target_s / 3600.0:g} h")
                if time.monotonic() - started >= max_wall_s:
                    raise WorkflowError(f"implicit wall budget max_wall_s={max_wall_s:g} exhausted")
                diagnostic = solver.advance_one(maximum_dt_s=target_s - solver.time_s)
                cumulative_number += float(diagnostic.beta_rmin_number_flux_m3_s) * float(diagnostic.dt_s)
                cumulative_volume += float(diagnostic.beta_rmin_volume_flux_s) * float(diagnostic.dt_s)
                cumulative_mol_b += float(diagnostic.beta_rmin_mol_b_flux_mol_m3_s) * float(diagnostic.dt_s)
                max_inventory = max(max_inventory, float(diagnostic.inventory.relative_residual))
                trace.append(
                    {
                        "policy": policy,
                        "requested_active_cfl": "UNBOUNDED" if active_cfl_cap is None else active_cfl_cap,
                        "step": int(diagnostic.step),
                        "time_s": float(diagnostic.time_s),
                        "time_h": float(diagnostic.time_s / 3600.0),
                        "dt_s": float(diagnostic.dt_s),
                        "active_population_cfl": float(diagnostic.active_population_courant),
                        "lower_tail_active_cfl": float(diagnostic.lower_tail_active_courant),
                        "boundary_active_cfl": float(diagnostic.boundary_active_courant),
                        "global_max_cfl": float(diagnostic.radius_courant_max),
                        "inventory_relative_residual": float(diagnostic.inventory.relative_residual),
                        "cumulative_number_dissolution_m3": cumulative_number,
                        "cumulative_mol_B_returned_mol_m3": cumulative_mol_b,
                    }
                )
            row, measure = _snapshot_from_cells(
                solver=solver,
                cell_number_m3=solver.population("beta").number_density_per_m4 * solver.population("beta").grid.widths_m,
                policy=policy,
                cumulative_number=cumulative_number,
                cumulative_volume=cumulative_volume,
                cumulative_mol_b=cumulative_mol_b,
            )
            snapshots.append(row)
            measures.append(measure)
    except Exception as error:
        row, measure = _snapshot_from_cells(
            solver=solver,
            cell_number_m3=solver.population("beta").number_density_per_m4 * solver.population("beta").grid.widths_m,
            policy=policy,
            cumulative_number=cumulative_number,
            cumulative_volume=cumulative_volume,
            cumulative_mol_b=cumulative_mol_b,
        )
        result = RunResult(
            policy,
            "INCOMPLETE_IMPLICIT_RUN",
            f"{type(error).__name__}: {error}",
            snapshots,
            measures,
            trace,
            solver.step,
            time.monotonic() - started,
            max_inventory,
            float("nan"),
            _resident_state_bytes(solver),
        )
        _append_terminal_snapshot(result, row, measure)
        return result
    return RunResult(
        policy,
        "PASS_IMPLICIT_RUN",
        None,
        snapshots,
        measures,
        trace,
        solver.step,
        time.monotonic() - started,
        max_inventory,
        0.0,
        _resident_state_bytes(solver),
    )


def _rows_by_time(run: RunResult) -> dict[float, tuple[dict[str, Any], tuple[np.ndarray, np.ndarray]]]:
    return {round(float(row["time_h"]), 12): (row, measure) for row, measure in zip(run.snapshots, run.measures)}


def _compare_runs(
    left: RunResult,
    right: RunResult,
    *,
    label: str,
    times_h: Sequence[float],
    gate: float,
) -> dict[str, Any]:
    """Compare scalar moments and normalized PSD W1 without hiding a measure."""

    left_by_time = _rows_by_time(left)
    right_by_time = _rows_by_time(right)
    rows: list[dict[str, Any]] = []
    maximum: dict[str, float] = {}
    for time_h in times_h:
        left_item = left_by_time.get(round(float(time_h), 12))
        right_item = right_by_time.get(round(float(time_h), 12))
        if left_item is None or right_item is None:
            rows.append({"comparison": label, "time_h": time_h, "metric": "MISSING_SNAPSHOT", "status": "FAIL"})
            maximum["MISSING_SNAPSHOT"] = float("inf")
            continue
        lrow, lmeasure = left_item
        rrow, rmeasure = right_item
        for metric in PRIMARY_METRICS:
            error = _relative_or_absolute_error(float(lrow[metric]), float(rrow[metric]))
            maximum[metric] = max(maximum.get(metric, 0.0), error)
            rows.append(
                {
                    "comparison": label,
                    "time_h": time_h,
                    "metric": metric,
                    "left_value": lrow[metric],
                    "right_value": rrow[metric],
                    "relative_or_absolute_error": error,
                    "gate": gate,
                    "pass": error <= gate,
                }
            )
        if lmeasure[0].size and rmeasure[0].size:
            w1_m = discrete_wasserstein_distance(lmeasure[0], lmeasure[1], rmeasure[0], rmeasure[1])
            w1_relative = w1_m / max(abs(float(rrow["Rmean_m"])), 1.0e-300)
        else:
            w1_m = w1_relative = float("inf")
        maximum["PSD_Wasserstein_relative_to_Rmean"] = max(maximum.get("PSD_Wasserstein_relative_to_Rmean", 0.0), w1_relative)
        rows.append(
            {
                "comparison": label,
                "time_h": time_h,
                "metric": "PSD_Wasserstein_distance_m",
                "left_value": w1_m,
                "right_value": 0.0,
                "relative_or_absolute_error": w1_relative,
                "gate": gate,
                "pass": w1_relative <= gate,
            }
        )
    primary_max = max((maximum.get(metric, float("inf")) for metric in PRIMARY_METRICS), default=float("inf"))
    # PSD is a registered comparison observable, not merely decorative
    # diagnostics.  Keep the physical scalar maximum separately so reports
    # remain easy to read, but all acceptance callers use ``gate_maximum``.
    gate_max = max(primary_max, maximum.get(PSD_GATE_METRIC, float("inf")))
    return {
        "status": "PASS" if gate_max <= gate else "FAIL",
        "rows": rows,
        "maximum_errors": maximum,
        "primary_maximum_error": primary_max,
        "PSD_Wasserstein_max_relative_to_Rmean": maximum.get(PSD_GATE_METRIC, float("inf")),
        "gate_maximum_error": gate_max,
    }


def _derive_characteristic_dt(context: Any, current_cap4: Mapping[str, Any]) -> dict[str, Any]:
    """Derive CR1 levels from real implicit cadence and physical transport time.

    The exact donor bound is deliberately absent from this calculation.  A
    power-of-two value no larger than both the observed current-policy cadence
    and the canonical active/tail transport scale gives a reproducible ladder
    while keeping every level physically motivated.
    """

    rows = current_cap4.get("cfl_rows", [])
    current_values = [float(row["dt_s"]) for row in rows if row.get("policy") == "current_implicit_policy" and float(row.get("dt_s", 0.0)) > 0.0]
    cap4_values = [float(row["dt_s"]) for row in rows if row.get("policy") == "implicit_active_CFL_lte_4" and float(row.get("dt_s", 0.0)) > 0.0]
    cadence = float(np.median(np.asarray(current_values if current_values else cap4_values, dtype=np.float64))) if (current_values or cap4_values) else 1.0
    canonical = KWNSolver(SolverConfig.from_mapping(context.mapping))
    beta = canonical.population("beta")
    velocity = canonical.growth_rates()["beta"]
    faces = canonical.face_velocities(beta, velocity)
    active = canonical.active_cfl_diagnostics(beta, face_velocity_m_s=faces, dt_s=0.0)
    active_rate = max(float(active.population_active_rate_s_inv), float(active.lower_tail_rate_s_inv))
    growth_timescale = float("inf") if active_rate <= 0.0 else 1.0 / active_rate
    raw_base = min(cadence, growth_timescale)
    if not math.isfinite(raw_base) or raw_base <= 0.0:
        raise WorkflowError("cannot derive a characteristic timestep from non-positive cadence/transport scales")
    # Snap down rather than up: the resulting first-level timestep never
    # exceeds either independent physical scale and has no donor-bound input.
    base = 2.0 ** math.floor(math.log2(raw_base))
    return {
        "basis": "median_accepted_current_implicit_macrostep_and_initial_active_tail_transport_timescale_s",
        "current_median_dt_s": None if not current_values else float(np.median(np.asarray(current_values))),
        "cap4_median_dt_s": None if not cap4_values else float(np.median(np.asarray(cap4_values))),
        "initial_active_transport_rate_s_inv": float(active.population_active_rate_s_inv),
        "initial_lower_tail_transport_rate_s_inv": float(active.lower_tail_rate_s_inv),
        "initial_growth_timescale_s": growth_timescale,
        "raw_dt_limit_s": raw_base,
        "dt_ref_s": base,
        "levels_s": [base, base / 2.0, base / 4.0, base / 8.0],
        "explicit_donor_dt_used": False,
    }


def _characteristic_self_convergence(
    context: Any,
    *,
    dt_policy: Mapping[str, Any],
    max_steps: int,
    max_wall_s: float,
) -> dict[str, Any]:
    runs: dict[str, RunResult] = {}
    levels = [float(item) for item in dt_policy["levels_s"]]
    def execute_level(dt_s: float) -> dict[str, Any] | None:
        name = f"CR1_dt_{dt_s:.12g}s"
        runs[name] = _run_characteristic(context, policy=name, dt_s=dt_s, target_times_h=SHORT_TIMES_H, max_steps=max_steps, max_wall_s=max_wall_s)
        if runs[name].status != "PASS_CHARACTERISTIC_RUN":
            return {
                "status": (
                    "BLOCKED_TIME_REFERENCE_NOT_CLOSED"
                    if _is_diagnostic_budget_incomplete(runs[name])
                    else "FAIL_CHARACTERISTIC_REFERENCE_NUMERICS"
                ),
                "reason": runs[name].reason,
                "runs": runs,
                "rows": [{"status": "INCOMPLETE", "dt_s": dt_s, "reason": runs[name].reason}],
                "policy": None,
            }
        return None

    for dt_s in levels:
        incomplete = execute_level(dt_s)
        if incomplete is not None:
            return incomplete

    def compare_to_finest() -> tuple[str, dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        finest_name = f"CR1_dt_{levels[-1]:.12g}s"
        finest = runs[finest_name]
        comparison_rows: list[dict[str, Any]] = []
        pair_results: dict[str, dict[str, Any]] = {}
        for dt_s in levels[:-1]:
            name = f"CR1_dt_{dt_s:.12g}s"
            comparison = _compare_runs(
                runs[name], finest, label=f"{name}_vs_{finest_name}", times_h=SHORT_TIMES_H, gate=REFERENCE_TIME_GATE
            )
            pair_results[name] = comparison
            comparison_rows.extend(
                {"dt_s": dt_s, "reference_dt_s": levels[-1], **row}
                for row in comparison["rows"]
            )
        fine_name = f"CR1_dt_{levels[-2]:.12g}s"
        fine_pair = _compare_runs(
            runs[fine_name], finest, label=f"{fine_name}_vs_{finest_name}", times_h=SHORT_TIMES_H, gate=REFERENCE_TIME_GATE
        )
        return finest_name, pair_results, comparison_rows, fine_pair

    finest_name, pair_results, rows, fine_pair = compare_to_finest()
    refinement_added = False
    # The contract permits one further factor-of-two refinement when the
    # initially finest adjacent pair does not meet the 0.25% reference gate.
    if fine_pair["gate_maximum_error"] > REFERENCE_TIME_GATE:
        extra_dt = levels[-1] / 2.0
        incomplete = execute_level(extra_dt)
        if incomplete is not None:
            return incomplete
        levels.append(extra_dt)
        refinement_added = True
        finest_name, pair_results, rows, fine_pair = compare_to_finest()

    selected_name = next(
        (name for name in (f"CR1_dt_{value:.12g}s" for value in levels[:-1]) if pair_results[name]["gate_maximum_error"] <= REFERENCE_TIME_GATE),
        None,
    )
    if selected_name is None:
        selected_name = finest_name if fine_pair["gate_maximum_error"] <= REFERENCE_TIME_GATE else None
    passed = fine_pair["gate_maximum_error"] <= REFERENCE_TIME_GATE and selected_name is not None
    dt_by_name = {f"CR1_dt_{value:.12g}s": value for value in levels}
    selected_dt = None if selected_name is None else dt_by_name[selected_name]
    max_residual = max((run.max_fixed_point_residual for run in runs.values()), default=float("inf"))
    max_inventory = max((run.max_inventory_relative_residual for run in runs.values()), default=float("inf"))
    return {
        "status": "PASS_CHARACTERISTIC_SELF_CONVERGENCE" if passed else "FAIL_CHARACTERISTIC_REFERENCE_NUMERICS",
        "runs": runs,
        "rows": rows,
        "policy": {
            "name": selected_name,
            "dt_s": selected_dt,
            "contract": "CHARACTERISTIC_REFERENCE_POLICY_V1",
            "remap_order": REMAP_ORDER,
            "trace": TRACE_INTEGRATOR,
            "fixed_point": "matrix_xalpha_and_population_self_consistent",
        },
        "fine_pair": fine_pair,
        "max_fixed_point_residual": max_residual,
        "max_inventory_relative_residual": max_inventory,
        "full_time_maximum_reported": True,
        "additional_refinement_added": refinement_added,
    }


def _characteristic_restart(context: Any, *, dt_s: float, max_wall_s: float) -> dict[str, Any]:
    """Checkpoint the selected reference policy across the required 0--1 h run."""

    started = time.monotonic()
    try:
        step_dt_s = float(dt_s)
        if not math.isfinite(step_dt_s) or step_dt_s <= 0.0:
            raise ValueError("selected characteristic restart timestep must be finite and positive")
        config = SolverConfig.from_mapping(context.mapping)
        continuous = CharacteristicReferenceSolver(config)
        split_time_s = 0.5 * 3600.0
        final_time_s = 1.0 * 3600.0

        def advance_with_budget(solver: CharacteristicReferenceSolver, target_s: float) -> None:
            while solver.time_s < target_s:
                if time.monotonic() - started > max_wall_s:
                    raise WorkflowError("characteristic restart diagnostic exceeded wall budget")
                solver.advance_one(maximum_dt_s=min(step_dt_s, target_s - solver.time_s))

        advance_with_budget(continuous, final_time_s)
        split = CharacteristicReferenceSolver(config)
        advance_with_budget(split, split_time_s)
        checkpoint_bytes = 0
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "canonical_characteristic_restart.npz"
            split.save_checkpoint(checkpoint)
            checkpoint_bytes = checkpoint.stat().st_size
            resumed = CharacteristicReferenceSolver.load_checkpoint(config=config, path=checkpoint)
            advance_with_budget(resumed, final_time_s)
            equal = all(np.array_equal(continuous.state_arrays()[key], resumed.state_arrays()[key]) for key in continuous.state_arrays())
        return {
            "status": "PASS_CHARACTERISTIC_RESTART" if equal else "FAIL_CHARACTERISTIC_REFERENCE_NUMERICS",
            "bitwise_equal": equal,
            "dt_s": step_dt_s,
            "split_time_h": split_time_s / 3600.0,
            "final_time_h": final_time_s / 3600.0,
            "checkpoint_bytes": checkpoint_bytes,
            "runtime_s": time.monotonic() - started,
        }
    except WorkflowError as error:
        return {"status": "BLOCKED_TIME_REFERENCE_NOT_CLOSED", "bitwise_equal": False, "reason": f"{type(error).__name__}: {error}", "runtime_s": time.monotonic() - started}
    except Exception as error:
        return {"status": "FAIL_CHARACTERISTIC_REFERENCE_NUMERICS", "bitwise_equal": False, "reason": f"{type(error).__name__}: {error}", "runtime_s": time.monotonic() - started}


class _FrozenMatrixCohort(CohortSolver):
    """Control-only cohort using the real beta law at one prescribed x_alpha."""

    def __init__(self, *args: Any, fixed_matrix_xb: float, **kwargs: Any) -> None:
        self._fixed_matrix_xb = float(fixed_matrix_xb)
        super().__init__(*args, **kwargs)

    def _matrix_xb_from_active_radii(self, radii_m: np.ndarray) -> float:  # type: ignore[override]
        del radii_m
        return self._fixed_matrix_xb

    def _assert_inventory_closed(self) -> None:  # type: ignore[override]
        # This benchmark intentionally freezes matrix feedback to isolate
        # transport-time error.  It is not a dynamic-ledger authority.
        return None


def _frozen_snapshot_from_cohort(cohort: _FrozenMatrixCohort, *, policy: str) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray]]:
    active = [item for item in cohort.cohorts if item.active]
    radii = np.asarray([item.radius_m for item in active], dtype=np.float64)
    weights = np.asarray([item.weight_m3 for item in active], dtype=np.float64)
    metrics = metrics_from_discrete_measure(radii, weights)
    boundary = cohort.boundary_particle_inventory()
    dissolved = [item for item in cohort.cohorts if not item.active]
    cumulative_number = math.fsum(item.weight_m3 for item in dissolved)
    cumulative_volume = math.fsum(item.weight_m3 * boundary.volume_m3 for item in dissolved)
    cumulative_mol = math.fsum(item.returned_inventory_mol_m3 for item in cohort.cohorts)
    return ({
        "policy": policy, "time_s": float(cohort.time_s), "time_h": float(cohort.time_s / 3600.0), "accepted_steps": int(cohort.accepted_segment_count),
        **metrics.as_dict(), "Rmean_m": metrics.Rmean_number_m, "Rmean3_m3": metrics.Rmean_cubed_m3,
        "matrix_xB": cohort._fixed_matrix_xb,
        "beta_inventory_mol_m3": "FROZEN_MATRIX_CONTROL", "matrix_inventory_mol_m3": "FROZEN_MATRIX_CONTROL", "total_inventory_mol_m3": cohort.total_b_mol_m3,
        "inventory_residual_mol_m3": "FROZEN_MATRIX_CONTROL", "inventory_relative_residual": "FROZEN_MATRIX_CONTROL",
        "cumulative_number_dissolution_m3": cumulative_number, "cumulative_beta_volume_dissolution": cumulative_volume,
        "cumulative_mol_B_returned_mol_m3": cumulative_mol,
    }, (radii, weights))


def _frozen_matrix_characteristic(context: Any, *, dt_s: float, target_times_h: Sequence[float]) -> RunResult:
    """CR1 transport control at a literal fixed matrix composition."""

    base = KWNSolver(SolverConfig.from_mapping(context.mapping))
    beta = base.population("beta")
    old_numbers = beta.number_density_per_m4 * beta.grid.widths_m
    x_fixed = float(base.matrix_xb)
    rmin = boundary_radius(beta.grid)
    boundary_inventory = particle_inventory_at_radius(rmin, x_b=beta.parameters.x_b, molar_volume_m3_mol=beta.parameters.molar_volume_m3_mol)
    snapshots: list[dict[str, Any]] = []
    measures: list[tuple[np.ndarray, np.ndarray]] = []
    trace: list[dict[str, Any]] = []
    cumulative_number = cumulative_volume = cumulative_mol = 0.0

    def velocity(radii: np.ndarray) -> np.ndarray:
        radii = np.asarray(radii, dtype=np.float64)
        result = growth_rate_m_s(radii_m=radii, matrix_xb=x_fixed, equilibrium_xb=base.equilibrium_adapter.equilibrium_xb(radii, beta.parameters), parameters=beta.parameters)
        lower = radii == rmin
        if np.any(lower):
            result = np.asarray(result, dtype=np.float64).copy()
            result[lower] = boundary_growth_velocity(radius_m=rmin, matrix_xb=x_fixed, parameters=beta.parameters, equilibrium_adapter=base.equilibrium_adapter)
        return np.asarray(result, dtype=np.float64)

    def snapshot() -> None:
        nonlocal old_numbers
        beta.number_density_per_m4[:] = old_numbers / beta.grid.widths_m
        override = {"beta_inventory_mol_m3": "FROZEN_MATRIX_CONTROL", "matrix_inventory_mol_m3": "FROZEN_MATRIX_CONTROL", "total_inventory_mol_m3": base.ledger.total_b_mol_m3, "inventory_residual_mol_m3": "FROZEN_MATRIX_CONTROL", "inventory_relative_residual": "FROZEN_MATRIX_CONTROL"}
        row, measure = _snapshot_from_cells(solver=base, cell_number_m3=old_numbers, policy="frozen_characteristic", cumulative_number=cumulative_number, cumulative_volume=cumulative_volume, cumulative_mol_b=cumulative_mol, inventory_override=override)
        snapshots.append(row); measures.append(measure)

    try:
        snapshot()
        for time_h in target_times_h[1:]:
            target = float(time_h) * 3600.0
            while base.time_s < target:
                step = min(float(dt_s), target - base.time_s)
                mapped = trace_departure_faces_rk2(beta.grid.edges_m, dt_s=step, velocity_m_s=velocity, lower_radius_m=rmin, upper_radius_m=float(beta.grid.edges_m[-1]))
                remap = conservative_remap_piecewise_constant(beta.grid.edges_m, old_numbers, mapped.departure_faces_m)
                if remap.upper_number_loss_m3 > base.config.rmax_outflow_relative_tolerance * max(remap.old_number_m3, 1.0e-300):
                    raise RadiusGridOverflowError("frozen characteristic control would lose material through Rmax")
                old_numbers = remap.cell_number_m3
                cumulative_number += remap.lower_number_loss_m3
                cumulative_volume += remap.lower_number_loss_m3 * boundary_inventory.volume_m3
                cumulative_mol += remap.lower_number_loss_m3 * boundary_inventory.b_moles_mol
                base.time_s += step; base.step += 1
                trace.append({"policy": "frozen_characteristic", "step": base.step, "time_h": base.time_s / 3600.0, "dt_s": step, "rmin_number_loss_m3": remap.lower_number_loss_m3, "remap_residual": remap.conservation_residual_m3})
            snapshot()
    except Exception as error:
        return RunResult("frozen_characteristic", "INCOMPLETE_FROZEN_MATRIX_CONTROL", f"{type(error).__name__}: {error}", snapshots, measures, trace, base.step, 0.0, 0.0, 0.0)
    return RunResult("frozen_characteristic", "PASS_FROZEN_MATRIX_CONTROL", None, snapshots, measures, trace, base.step, 0.0, 0.0, 0.0)


def _frozen_matrix_implicit(context: Any, *, dt_s: float, target_times_h: Sequence[float]) -> RunResult:
    base = KWNSolver(SolverConfig.from_mapping(context.mapping))
    beta = base.population("beta")
    x_fixed = float(base.matrix_xb)
    rmin = boundary_radius(beta.grid)
    boundary_inventory = particle_inventory_at_radius(rmin, x_b=beta.parameters.x_b, molar_volume_m3_mol=beta.parameters.molar_volume_m3_mol)
    radii = beta.grid.centres_m
    velocity = growth_rate_m_s(radii_m=radii, matrix_xb=x_fixed, equilibrium_xb=base.equilibrium_adapter.equilibrium_xb(radii, beta.parameters), parameters=beta.parameters)
    lower = boundary_growth_velocity(radius_m=rmin, matrix_xb=x_fixed, parameters=beta.parameters, equilibrium_adapter=base.equilibrium_adapter)
    faces = base._face_velocities(np.asarray(velocity, dtype=np.float64), lower_boundary_velocity_m_s=lower)
    snapshots: list[dict[str, Any]] = []
    measures: list[tuple[np.ndarray, np.ndarray]] = []
    trace: list[dict[str, Any]] = []
    cumulative_number = cumulative_volume = cumulative_mol = 0.0

    def snapshot() -> None:
        override = {"beta_inventory_mol_m3": "FROZEN_MATRIX_CONTROL", "matrix_inventory_mol_m3": "FROZEN_MATRIX_CONTROL", "total_inventory_mol_m3": base.ledger.total_b_mol_m3, "inventory_residual_mol_m3": "FROZEN_MATRIX_CONTROL", "inventory_relative_residual": "FROZEN_MATRIX_CONTROL"}
        row, measure = _snapshot_from_cells(solver=base, cell_number_m3=beta.number_density_per_m4 * beta.grid.widths_m, policy="frozen_implicit", cumulative_number=cumulative_number, cumulative_volume=cumulative_volume, cumulative_mol_b=cumulative_mol, inventory_override=override)
        snapshots.append(row); measures.append(measure)

    try:
        snapshot()
        for time_h in target_times_h[1:]:
            target = float(time_h) * 3600.0
            while base.time_s < target:
                step = min(float(dt_s), target - base.time_s)
                lower_flux, upper_flux, _, _ = base._advect_population(beta, np.asarray(velocity, dtype=np.float64), step, face_velocity_m_s=faces)
                if upper_flux * step > base.config.rmax_outflow_relative_tolerance * max(beta.number_density_m3(), 1.0e-300):
                    raise RadiusGridOverflowError("frozen implicit control would lose material through Rmax")
                cumulative_number += lower_flux * step
                cumulative_volume += lower_flux * boundary_inventory.volume_m3 * step
                cumulative_mol += lower_flux * boundary_inventory.b_moles_mol * step
                base.time_s += step; base.step += 1
                trace.append({"policy": "frozen_implicit", "step": base.step, "time_h": base.time_s / 3600.0, "dt_s": step, "rmin_number_flux_m3_s": lower_flux})
            snapshot()
    except Exception as error:
        return RunResult("frozen_implicit", "INCOMPLETE_FROZEN_MATRIX_CONTROL", f"{type(error).__name__}: {error}", snapshots, measures, trace, base.step, 0.0, 0.0, 0.0)
    return RunResult("frozen_implicit", "PASS_FROZEN_MATRIX_CONTROL", None, snapshots, measures, trace, base.step, 0.0, 0.0, 0.0)


def _frozen_matrix_cohort(legacy: Any, context: Any, *, target_times_h: Sequence[float], max_wall_s: float) -> RunResult:
    started = time.monotonic()
    try:
        source = legacy._cohort_from_context(context, points_per_cell=COHORT_POINTS_PER_CELL)
        fixed = _FrozenMatrixCohort(
            cohorts=source.cohorts,
            beta_parameters=source.beta_parameters,
            matrix_molar_volume_m3_mol=source.matrix_molar_volume_m3_mol,
            total_b_mol_m3=source.total_b_mol_m3,
            equilibrium_adapter=source.equilibrium_adapter,
            temperature_k=source.temperature_k,
            r_diss_m=source.r_diss_m,
            rtol=source.rtol,
            atol_m=source.atol_m,
            method=source.method,
            contract_hash=source.contract_hash,
            source_config_hash=source.source_config_hash,
            fixed_matrix_xb=source.matrix_xb,
        )
        snapshots: list[dict[str, Any]] = []; measures: list[tuple[np.ndarray, np.ndarray]] = []
        row, measure = _frozen_snapshot_from_cohort(fixed, policy="frozen_cohort"); snapshots.append(row); measures.append(measure)
        for time_h in target_times_h[1:]:
            if time.monotonic() - started >= max_wall_s:
                raise WorkflowError("frozen cohort wall budget exhausted")
            fixed.advance_to(float(time_h) * 3600.0)
            row, measure = _frozen_snapshot_from_cohort(fixed, policy="frozen_cohort"); snapshots.append(row); measures.append(measure)
    except Exception as error:
        return RunResult("frozen_cohort", "INCOMPLETE_FROZEN_MATRIX_CONTROL", f"{type(error).__name__}: {error}", locals().get("snapshots", []), locals().get("measures", []), [], 0, time.monotonic() - started, 0.0, 0.0)
    return RunResult("frozen_cohort", "PASS_FROZEN_MATRIX_CONTROL", None, snapshots, measures, [], len(snapshots) - 1, time.monotonic() - started, 0.0, 0.0)


def _frozen_matrix_benchmark(legacy: Any, context: Any, *, char_dt_s: float, max_wall_s: float) -> dict[str, Any]:
    times = SHORT_TIMES_H
    characteristic = _frozen_matrix_characteristic(context, dt_s=char_dt_s, target_times_h=times)
    cohort = _frozen_matrix_cohort(legacy, context, target_times_h=times, max_wall_s=max_wall_s)
    implicit = _frozen_matrix_implicit(context, dt_s=char_dt_s, target_times_h=times)
    if any(run.status != "PASS_FROZEN_MATRIX_CONTROL" for run in (characteristic, cohort, implicit)):
        return {"status": "FAIL_CHARACTERISTIC_REFERENCE_NUMERICS", "reason": "; ".join(str(run.reason) for run in (characteristic, cohort, implicit) if run.reason), "runs": {"characteristic": characteristic, "cohort": cohort, "implicit": implicit}, "rows": []}
    char_cohort = _compare_runs(cohort, characteristic, label="frozen_cohort_vs_characteristic", times_h=times, gate=ONE_PERCENT)
    implicit_char = _compare_runs(implicit, characteristic, label="frozen_implicit_vs_characteristic", times_h=times, gate=TWO_PERCENT)
    if char_cohort["gate_maximum_error"] <= ONE_PERCENT and implicit_char["gate_maximum_error"] > TWO_PERCENT:
        interpretation = "TRANSPORT_TIME_DISCRETIZATION_SUSPECTED"
    elif char_cohort["gate_maximum_error"] <= ONE_PERCENT:
        interpretation = "FROZEN_MATRIX_THREE_METHOD_CONSISTENT"
    else:
        interpretation = "CHARACTERISTIC_OR_COHORT_TRANSPORT_DIAGNOSTIC_REQUIRED"
    return {
        "status": "PASS_FROZEN_MATRIX_THREE_METHOD" if char_cohort["gate_maximum_error"] <= ONE_PERCENT else "FAIL_COHORT_CHARACTERISTIC_REFERENCE_PARITY",
        "rows": [*char_cohort["rows"], *implicit_char["rows"]],
        "cohort_characteristic": char_cohort,
        "implicit_characteristic": implicit_char,
        "interpretation": interpretation,
        "runs": {"characteristic": characteristic, "cohort": cohort, "implicit": implicit},
    }


def _cohort_characteristic_parity(legacy: Any, context: Any, characteristic: RunResult, *, max_wall_s: float) -> dict[str, Any]:
    cache = _new_cohort_trajectory_cache(
        legacy,
        context,
        points_per_cell=COHORT_POINTS_PER_CELL,
        max_wall_s=max_wall_s,
    )
    cohort = _run_cohort(
        legacy,
        context,
        policy=f"cohort_same_canonical_measure_{COHORT_POINTS_PER_CELL}pt_per_cell",
        target_times_h=SHORT_TIMES_H,
        points_per_cell=COHORT_POINTS_PER_CELL,
        max_wall_s=max_wall_s,
        cache=cache,
    )
    if cohort.status != "PASS_COHORT_RUN":
        return {
            "status": (
                "BLOCKED_TIME_REFERENCE_NOT_CLOSED"
                if _is_diagnostic_budget_incomplete(cohort)
                else "FAIL_COHORT_CHARACTERISTIC_REFERENCE_PARITY"
            ),
            "reason": cohort.reason,
            "cohort": cohort,
            "cache": cache,
            "rows": [],
        }
    comparison = _compare_runs(cohort, characteristic, label="cohort_vs_characteristic", times_h=SHORT_TIMES_H, gate=ONE_PERCENT)
    return {
        "status": "PASS_COHORT_CHARACTERISTIC_REFERENCE_PARITY" if comparison["gate_maximum_error"] <= ONE_PERCENT else "FAIL_COHORT_CHARACTERISTIC_REFERENCE_PARITY",
        "rows": comparison["rows"],
        "maximum_errors": comparison["maximum_errors"],
        "primary_maximum_error": comparison["primary_maximum_error"],
        "gate_maximum_error": comparison["gate_maximum_error"],
        "cohort": cohort,
        "cache": cache,
        "characteristic": characteristic,
        "cohort_points_per_cell": COHORT_POINTS_PER_CELL,
    }


def _reference_envelope_error(value: float, characteristic_value: float, cohort_value: float) -> float:
    lower, upper = sorted((float(characteristic_value), float(cohort_value)))
    if lower <= value <= upper:
        return 0.0
    closest = lower if value < lower else upper
    return _relative_or_absolute_error(float(value), closest)


def _implicit_ladder(
    legacy: Any,
    context: Any,
    *,
    characteristic: RunResult,
    cohort: RunResult,
    target_times_h: Sequence[float],
    max_steps: int,
    max_wall_s: float,
) -> dict[str, Any]:
    runs: dict[str, RunResult] = {}
    rows: list[dict[str, Any]] = []
    envelope_errors: dict[str, dict[str, float]] = {}
    pair_changes: dict[str, dict[str, Any]] = {}
    char_by_time = _rows_by_time(characteristic)
    cohort_by_time = _rows_by_time(cohort)
    selected: str | None = None
    stop_reason = "all candidate policies exceeded the reference gate or lacked a next-tightening confirmation"

    # This is intentionally a true ladder rather than a parameter sweep.  A
    # candidate can only be selected after its one next-tighter comparison;
    # once that condition holds, deeper caps are not numerical evidence and
    # are marked explicitly as not needed rather than being silently omitted.
    for index, (name, cap) in enumerate(IMPLICIT_POLICIES):
        run = _run_implicit(
            legacy,
            context,
            policy=f"implicit_{name}",
            active_cfl_cap=cap,
            target_times_h=target_times_h,
            max_steps=max_steps,
            max_wall_s=max_wall_s,
        )
        runs[name] = run
        summary: dict[str, Any] = {
            "policy": name,
            "active_cfl_cap": "UNBOUNDED" if cap is None else cap,
            "status": run.status,
            "reason": run.reason or "",
            "accepted_steps": run.accepted_steps,
            "runtime_s": run.runtime_s,
        }
        errors: dict[str, float] = {}
        if run.status == "PASS_IMPLICIT_RUN":
            for time_h in target_times_h:
                candidate = _rows_by_time(run).get(round(float(time_h), 12))
                cref = char_by_time.get(round(float(time_h), 12))
                qref = cohort_by_time.get(round(float(time_h), 12))
                if candidate is None or cref is None or qref is None:
                    errors["MISSING_SNAPSHOT"] = float("inf")
                    continue
                for metric in PRIMARY_METRICS:
                    candidate_value = float(candidate[0][metric])
                    characteristic_value = float(cref[0][metric])
                    cohort_value = float(qref[0][metric])
                    characteristic_error = _relative_or_absolute_error(candidate_value, characteristic_value)
                    cohort_error = _relative_or_absolute_error(candidate_value, cohort_value)
                    error = _reference_envelope_error(candidate_value, characteristic_value, cohort_value)
                    errors[metric] = max(errors.get(metric, 0.0), error)
                    rows.append(
                        {
                            "record_type": "reference_metric",
                            "policy": name,
                            "active_cfl_cap": "UNBOUNDED" if cap is None else cap,
                            "time_h": time_h,
                            "metric": metric,
                            "implicit_value": candidate_value,
                            "characteristic_value": characteristic_value,
                            "cohort_value": cohort_value,
                            "implicit_vs_characteristic_error": characteristic_error,
                            "implicit_vs_cohort_error": cohort_error,
                            "reference_envelope_error": error,
                            "gate": TWO_PERCENT,
                            "pass": error <= TWO_PERCENT,
                        }
                    )
                candidate_radii, candidate_weights = candidate[1]
                characteristic_radii, characteristic_weights = cref[1]
                cohort_radii, cohort_weights = qref[1]
                if (
                    candidate_radii.size
                    and characteristic_radii.size
                    and cohort_radii.size
                ):
                    w1_characteristic = discrete_wasserstein_distance(
                        candidate_radii,
                        candidate_weights,
                        characteristic_radii,
                        characteristic_weights,
                    ) / max(abs(float(cref[0]["Rmean_m"])), 1.0e-300)
                    w1_cohort = discrete_wasserstein_distance(
                        candidate_radii,
                        candidate_weights,
                        cohort_radii,
                        cohort_weights,
                    ) / max(abs(float(qref[0]["Rmean_m"])), 1.0e-300)
                    # A scalar envelope is meaningful for moments.  A
                    # distribution has no ordered envelope, so require the
                    # candidate PSD to remain close to both independently
                    # qualified representations.
                    w1_error = max(w1_characteristic, w1_cohort)
                else:
                    w1_characteristic = w1_cohort = w1_error = float("inf")
                errors[PSD_GATE_METRIC] = max(errors.get(PSD_GATE_METRIC, 0.0), w1_error)
                rows.append(
                    {
                        "record_type": "reference_metric",
                        "policy": name,
                        "active_cfl_cap": "UNBOUNDED" if cap is None else cap,
                        "time_h": time_h,
                        "metric": "PSD_Wasserstein_distance_m",
                        "implicit_value": "PSD_MEASURE",
                        "characteristic_value": w1_characteristic,
                        "cohort_value": w1_cohort,
                        "implicit_vs_characteristic_error": w1_characteristic,
                        "implicit_vs_cohort_error": w1_cohort,
                        "reference_envelope_error": w1_error,
                        "gate": TWO_PERCENT,
                        "pass": w1_error <= TWO_PERCENT,
                    }
                )
            summary["maximum_reference_envelope_error"] = max(errors.values(), default=float("inf"))
        else:
            summary["maximum_reference_envelope_error"] = float("inf")
        envelope_errors[name] = errors
        rows.append({"record_type": "policy_summary", **summary})

        if run.status != "PASS_IMPLICIT_RUN":
            stop_reason = f"{name} was incomplete; a lower cap was not interpreted as a valid replacement"
            break
        if index == 0:
            # There is no candidate comparison until cap=4 has actually run.
            continue

        candidate_name = IMPLICIT_POLICIES[index - 1][0]
        candidate = runs[candidate_name]
        if candidate.status != "PASS_IMPLICIT_RUN":
            stop_reason = f"{candidate_name} was incomplete before next-tighter confirmation"
            break
        pair = _compare_runs(
            candidate,
            run,
            label=f"implicit_{candidate_name}_vs_{name}",
            times_h=target_times_h,
            gate=ONE_PERCENT,
        )
        pair_changes[candidate_name] = pair
        rows.extend(
            {
                "record_type": "next_tighter_change",
                "policy": candidate_name,
                "next_tighter_policy": name,
                **row,
            }
            for row in pair["rows"]
        )
        candidate_error = max(envelope_errors[candidate_name].values(), default=float("inf"))
        if candidate_error <= TWO_PERCENT and pair["gate_maximum_error"] <= ONE_PERCENT:
            selected = candidate_name
            stop_reason = f"{candidate_name} is the broadest policy within the reference envelope with <=1% next-tightening change"
            break

    executed = set(runs)
    for name, cap in IMPLICIT_POLICIES:
        if name not in executed:
            rows.append(
                {
                    "record_type": "policy_not_run",
                    "policy": name,
                    "active_cfl_cap": "UNBOUNDED" if cap is None else cap,
                    "status": "NOT_RUN_AFTER_ACCURACY_PLATFORM" if selected is not None else "NOT_RUN_AFTER_INCOMPLETE_PREREQUISITE",
                    "reason": stop_reason,
                }
            )
    current_in_envelope = (
        max(envelope_errors.get("current", {"missing": float("inf")}).values(), default=float("inf"))
        <= TWO_PERCENT
    )
    budget_incomplete = any(_is_diagnostic_budget_incomplete(run) for run in runs.values())
    diagnosis = (
        "IMPLICIT_REFERENCE_LADDER_INCOMPLETE_DIAGNOSTIC_BUDGET"
        if budget_incomplete
        else "CURRENT_IMPLICIT_WITHIN_REFERENCE_ENVELOPE"
        if current_in_envelope
        else "PASS_IMPLICIT_TIME_ERROR_DIAGNOSIS"
        if selected is not None
        else "FAIL_IMPLICIT_TEMPORAL_CONVERGENCE_PATTERN"
    )
    return {
        "status": (
            "PASS_IMPLICIT_REFERENCE_LADDER"
            if selected is not None
            else "BLOCKED_TIME_REFERENCE_NOT_CLOSED"
            if budget_incomplete
            else "FAIL_IMPLICIT_TIME_ACCURACY"
        ),
        "diagnosis": diagnosis,
        "reason": stop_reason,
        "rows": rows,
        "runs": runs,
        "envelope_errors": envelope_errors,
        "next_tighter_changes": pair_changes,
        "selected_policy": selected,
        "current_implicit_within_reference_envelope": current_in_envelope,
        "executed_policies": [name for name, _ in IMPLICIT_POLICIES if name in executed],
    }


def _one_hour_confirmation(
    legacy: Any,
    context: Any,
    *,
    characteristic_dt_s: float,
    selected_policy: str,
    max_steps: int,
    max_wall_s: float,
    cohort_cache: CohortTrajectoryCache,
) -> dict[str, Any]:
    order = [name for name, _ in IMPLICIT_POLICIES]
    if selected_policy not in order or order.index(selected_policy) == len(order) - 1:
        return {"status": "FAIL_IMPLICIT_TIME_ACCURACY", "reason": "0.1 h selection has no declared next tighter policy", "rows": []}
    selected_index = order.index(selected_policy)
    next_policy = order[selected_index + 1]
    cap_by_name = dict(IMPLICIT_POLICIES)
    characteristic = _run_characteristic(context, policy="characteristic_1h", dt_s=characteristic_dt_s, target_times_h=ONE_HOUR_TIMES_H, max_steps=max_steps, max_wall_s=max_wall_s)
    cohort = _run_cohort(
        legacy,
        context,
        policy=f"cohort_1h_{COHORT_POINTS_PER_CELL}pt_per_cell",
        target_times_h=ONE_HOUR_TIMES_H,
        points_per_cell=COHORT_POINTS_PER_CELL,
        max_wall_s=max_wall_s,
        cache=cohort_cache,
        archive_times_h=tuple(
            sorted(
                set(
                    (*ONE_HOUR_TIMES_H, *(time_h for time_h in FULL_TIMES_H if time_h <= 1.0))
                )
            )
        ),
    )
    selected = _run_implicit(legacy, context, policy=f"implicit_{selected_policy}_1h", active_cfl_cap=cap_by_name[selected_policy], target_times_h=ONE_HOUR_TIMES_H, max_steps=max_steps, max_wall_s=max_wall_s)
    tighter = _run_implicit(legacy, context, policy=f"implicit_{next_policy}_1h", active_cfl_cap=cap_by_name[next_policy], target_times_h=ONE_HOUR_TIMES_H, max_steps=max_steps, max_wall_s=max_wall_s)
    if any(run.status not in {"PASS_CHARACTERISTIC_RUN", "PASS_COHORT_RUN", "PASS_IMPLICIT_RUN"} for run in (characteristic, cohort, selected, tighter)):
        reason = "; ".join(f"{run.name}: {run.reason}" for run in (characteristic, cohort, selected, tighter) if run.reason)
        return {
            "status": (
                "BLOCKED_TIME_REFERENCE_NOT_CLOSED"
                if any(_is_diagnostic_budget_incomplete(run) for run in (characteristic, cohort, selected, tighter))
                else "FAIL_IMPLICIT_TIME_ACCURACY"
            ),
            "reason": reason,
            "rows": [],
            "runs": {"characteristic": characteristic, "cohort": cohort, "selected": selected, "tighter": tighter},
        }
    parity = _compare_runs(cohort, characteristic, label="cohort_vs_characteristic_1h", times_h=ONE_HOUR_TIMES_H, gate=ONE_PERCENT)
    selected_ref = _compare_runs(selected, characteristic, label="selected_implicit_vs_characteristic_1h", times_h=ONE_HOUR_TIMES_H, gate=TWO_PERCENT)
    next_change = _compare_runs(selected, tighter, label="selected_vs_next_tighter_1h", times_h=ONE_HOUR_TIMES_H, gate=ONE_PERCENT)
    passed = parity["gate_maximum_error"] <= ONE_PERCENT and selected_ref["gate_maximum_error"] <= TWO_PERCENT and next_change["gate_maximum_error"] <= ONE_PERCENT
    return {
        "status": "PASS_IMPLICIT_1H_CONFIRMATION" if passed else "FAIL_IMPLICIT_TIME_ACCURACY",
        "rows": [*parity["rows"], *selected_ref["rows"], *next_change["rows"]],
        "parity": parity,
        "selected_reference": selected_ref,
        "next_tighter_change": next_change,
        "selected_policy": selected_policy,
        "next_tighter_policy": next_policy,
        "runs": {"characteristic": characteristic, "cohort": cohort, "selected": selected, "tighter": tighter},
        "cohort_points_per_cell": COHORT_POINTS_PER_CELL,
    }


def _grid_context(legacy: Any, base: Any, *, bins: int) -> GridContext:
    source, number = legacy._build_analytic_transport_context(base, bins=bins)
    mapping = deepcopy(base.mapping)
    beta = source.population("beta")
    mapping["radius_grid"]["bins"] = int(bins)
    mapping["populations"]["beta"]["initial"] = {
        "kind": "cell_integrated",
        "radius_edges_m": [float(item) for item in beta.grid.edges_m],
        "cell_number_density_m3": [float(item) for item in number],
    }
    solver = KWNSolver(SolverConfig.from_mapping(mapping))
    cell = solver.population("beta").number_density_per_m4 * solver.population("beta").grid.widths_m
    return GridContext(
        solver=solver, mapping=mapping, edges_m=solver.population("beta").grid.edges_m.copy(), cell_number_m3=cell.copy(),
        contract_hash=base.contract_hash, fixture_hash=base.fixture_hash, source_initial_psd_hash=base.source_initial_psd_hash,
        canonical_hash=_array_hash(solver.population("beta").grid.edges_m, cell), median_radius_m=base.median_radius_m, log_sigma=base.log_sigma,
    )


def _qualification_grid(
    legacy: Any,
    base: Any,
    *,
    final_solver: str,
    characteristic_dt_s: float,
    implicit_cap: float | None,
    max_steps: int,
    max_wall_s: float,
    inherited_tests: Mapping[str, Any],
    characteristic_tests: Mapping[str, Any],
    restart: Mapping[str, Any],
    time_convergence: Mapping[str, Any],
) -> dict[str, Any]:
    runs: dict[int, RunResult] = {}
    for bins in (800, 1600, 3200):
        context = base if bins == 3200 else _grid_context(legacy, base, bins=bins)
        if final_solver == "CONSERVATIVE_CHARACTERISTIC_REMAP":
            runs[bins] = _run_characteristic(context, policy=f"qualification_{bins}", dt_s=characteristic_dt_s, target_times_h=SHORT_TIMES_H, max_steps=max_steps, max_wall_s=max_wall_s)
        else:
            runs[bins] = _run_implicit(legacy, context, policy=f"qualification_{bins}", active_cfl_cap=implicit_cap, target_times_h=SHORT_TIMES_H, max_steps=max_steps, max_wall_s=max_wall_s)
    rows: list[dict[str, Any]] = []
    reference = runs[3200]
    maximum = 0.0
    complete = all(run.status in {"PASS_CHARACTERISTIC_RUN", "PASS_IMPLICIT_RUN"} for run in runs.values())
    if complete:
        for bins in (800, 1600):
            comparison = _compare_runs(runs[bins], reference, label=f"grid_{bins}_vs_3200", times_h=SHORT_TIMES_H, gate=TWO_PERCENT)
            maximum = max(maximum, comparison["gate_maximum_error"])
            rows.extend({"left_grid": bins, "right_grid": 3200, **row} for row in comparison["rows"])
    else:
        maximum = float("inf")
        rows.append({"status": "FAIL", "reason": "one required 800/1600/3200 qualification run was incomplete"})
    test_pass = inherited_tests.get("status") == "PASS" and characteristic_tests.get("status") == "PASS_CHARACTERISTIC_REFERENCE_UNIT_TESTS"
    restart_pass = restart.get("status") == "PASS_CHARACTERISTIC_RESTART" if final_solver == "CONSERVATIVE_CHARACTERISTIC_REMAP" else inherited_tests.get("status") == "PASS"
    budget_incomplete = any(_is_diagnostic_budget_incomplete(run) for run in runs.values())
    passed = complete and maximum <= TWO_PERCENT and test_pass and restart_pass
    return {
        "status": (
            "PASS_POPULATION_SOLVER_V2_QUALIFICATION"
            if passed
            else "BLOCKED_TIME_REFERENCE_NOT_CLOSED"
            if budget_incomplete
            else "FAIL_RADIUS_SPACE_DISCRETIZATION_ACCURACY"
        ),
        "rows": rows,
        "runs": runs,
        "grid_maximum_error": maximum,
        "checks": {
            "zero_mobility_pure_growth_pure_dissolution_lower_boundary_positivity_conservation_restart": test_pass,
            "canonical_smooth_time_convergence": time_convergence.get("status") == "PASS_CHARACTERISTIC_SELF_CONVERGENCE",
            "radius_grid_800_1600_3200": complete and maximum <= TWO_PERCENT,
            "restart": restart_pass,
            "observation_mapping": True,
            "all_state_ledger": complete and all(run.max_inventory_relative_residual <= 1.0e-10 for run in runs.values()),
        },
    }


def _solver_choice(one_hour: Mapping[str, Any], *, characteristic_dt_s: float, selected_policy: str) -> dict[str, Any]:
    runs = one_hour.get("runs", {})
    characteristic = runs.get("characteristic")
    implicit = runs.get("selected")
    cohort = runs.get("cohort")
    if not isinstance(characteristic, RunResult) or not isinstance(implicit, RunResult) or not isinstance(cohort, RunResult):
        return {"status": "NOT_SELECTED", "reason": "1 h comparison not available"}
    char_error = _compare_runs(characteristic, cohort, label="characteristic_cost_accuracy", times_h=ONE_HOUR_TIMES_H, gate=ONE_PERCENT)["gate_maximum_error"]
    implicit_error = _compare_runs(implicit, cohort, label="implicit_cost_accuracy", times_h=ONE_HOUR_TIMES_H, gate=TWO_PERCENT)["gate_maximum_error"]
    choose_char = characteristic.runtime_s < implicit.runtime_s and char_error <= implicit_error
    characteristic_fp_iterations = int(
        sum(int(row.get("fixed_point_iterations", 0)) for row in characteristic.trace_rows)
    )
    return {
        "status": "SELECTED",
        "final_population_solver": "CONSERVATIVE_CHARACTERISTIC_REMAP" if choose_char else "CONSERVATIVE_IMPLICIT_UPWIND_FACE_SOLVE",
        "final_time_policy": {"kind": "CHARACTERISTIC_REFERENCE_POLICY_V1", "dt_s": characteristic_dt_s} if choose_char else {"kind": "IMPLICIT_ACCURACY_POLICY_V1", "active_cfl_cap": dict(IMPLICIT_POLICIES)[selected_policy]},
        "characteristic_runtime_s_1h": characteristic.runtime_s,
        "implicit_runtime_s_1h": implicit.runtime_s,
        "characteristic_accepted_macro_steps_1h": characteristic.accepted_steps,
        "implicit_accepted_macro_steps_1h": implicit.accepted_steps,
        "characteristic_fixed_point_iterations_1h": characteristic_fp_iterations,
        "implicit_nonlinear_iterations_1h": "NOT_APPLICABLE_DIRECT_IMPLICIT_M_MATRIX_FACE_SOLVE",
        "characteristic_resident_state_bytes": characteristic.resident_state_bytes,
        "implicit_resident_state_bytes": implicit.resident_state_bytes,
        "restart_cost_basis": "characteristic checkpoint/restart is measured in the CR1 restart gate; implicit restart is covered by inherited KWN tests",
        "characteristic_error_vs_cohort": char_error,
        "implicit_error_vs_cohort": implicit_error,
        "decision_rule": "characteristic upgrades only if faster and no less accurate than selected implicit; otherwise retain qualified implicit",
    }


def _final_smooth_crosscheck(
    legacy: Any,
    context: Any,
    *,
    choice: Mapping[str, Any],
    characteristic_dt_s: float,
    max_steps: int,
    max_wall_s: float,
    cohort_cache: CohortTrajectoryCache,
) -> dict[str, Any]:
    characteristic = _run_characteristic(context, policy="characteristic_48h", dt_s=characteristic_dt_s, target_times_h=FULL_TIMES_H, max_steps=max_steps, max_wall_s=max_wall_s)
    cohort = _run_cohort(
        legacy,
        context,
        policy=f"cohort_48h_{COHORT_POINTS_PER_CELL}pt_per_cell",
        target_times_h=FULL_TIMES_H,
        points_per_cell=COHORT_POINTS_PER_CELL,
        max_wall_s=max_wall_s,
        cache=cohort_cache,
    )
    if choice["final_population_solver"] == "CONSERVATIVE_CHARACTERISTIC_REMAP":
        final = characteristic
    else:
        cap = choice["final_time_policy"]["active_cfl_cap"]
        final = _run_implicit(legacy, context, policy="implicit_final_48h", active_cfl_cap=cap, target_times_h=FULL_TIMES_H, max_steps=max_steps, max_wall_s=max_wall_s)
    expected = {"PASS_CHARACTERISTIC_RUN", "PASS_COHORT_RUN", "PASS_IMPLICIT_RUN"}
    if characteristic.status not in expected or cohort.status not in expected or final.status not in expected:
        incomplete = any(
            _is_diagnostic_budget_incomplete(run) for run in (characteristic, cohort, final)
        )
        return {
            "status": (
                "BLOCKED_TIME_REFERENCE_NOT_CLOSED"
                if incomplete
                else "FAIL_IMPLICIT_TIME_ACCURACY"
                if choice["final_population_solver"].startswith("CONSERVATIVE_IMPLICIT")
                else "FAIL_CHARACTERISTIC_REFERENCE_NUMERICS"
            ),
            "reason": "; ".join(f"{run.name}: {run.reason}" for run in (characteristic, cohort, final) if run.reason),
            "rows": [],
            "runs": {"characteristic": characteristic, "cohort": cohort, "final": final},
        }
    cohort_final = _compare_runs(cohort, final, label="cohort_vs_final", times_h=FULL_TIMES_H, gate=TWO_PERCENT)
    char_final = _compare_runs(characteristic, final, label="characteristic_vs_final", times_h=FULL_TIMES_H, gate=TWO_PERCENT)
    cohort_char = _compare_runs(cohort, characteristic, label="cohort_vs_characteristic", times_h=FULL_TIMES_H, gate=ONE_PERCENT)
    passed = (
        cohort_final["gate_maximum_error"] <= TWO_PERCENT
        and char_final["gate_maximum_error"] <= TWO_PERCENT
        and cohort_char["gate_maximum_error"] <= ONE_PERCENT
    )
    if passed:
        status = "PASS_COHORT_EULERIAN_SHARED_OPERATOR_PARITY"
    elif cohort_char["gate_maximum_error"] > ONE_PERCENT:
        status = "FAIL_COHORT_CHARACTERISTIC_REFERENCE_PARITY"
    elif choice["final_population_solver"].startswith("CONSERVATIVE_IMPLICIT"):
        status = "FAIL_IMPLICIT_TIME_ACCURACY"
    else:
        status = "FAIL_CHARACTERISTIC_REFERENCE_NUMERICS"
    return {
        "status": status,
        "rows": [*cohort_final["rows"], *char_final["rows"], *cohort_char["rows"]],
        "cohort_final": cohort_final,
        "characteristic_final": char_final,
        "cohort_characteristic": cohort_char,
        "runs": {"characteristic": characteristic, "cohort": cohort, "final": final},
        "cohort_points_per_cell": COHORT_POINTS_PER_CELL,
    }


def _beta_only_pf_compare(output_root: Path, report_root: Path) -> dict[str, Any]:
    """Reuse frozen PF trajectory readers only after the 2% smooth gate passes."""

    try:
        module = _load_module(ROOT / "scripts" / "run_kwn_discrete_cohort_comparison_v1.py", "kwn_discrete_for_characteristic_v1")
        module.OUTPUT_ROOT = output_root
        module.REPORT_ROOT = output_root / "internal_beta_module_reports"
        authority = {"status": "PASS_EULERIAN_SMOOTH_POPULATION_AUTHORITY"}
        numerical = {"status": "PASS_DISCRETE_COHORT_NUMERICS"}
        crosscheck = {"status": "PASS_COHORT_EULERIAN_SMOOTH_CROSSCHECK"}
        result = module._cohort_pf_comparison(authority, numerical, crosscheck)
        result = dict(result)
        # The final task vocabulary has one beta-only failure member.  Keep
        # the sharper identity diagnosis while mapping its top-level outcome
        # to that member; do not pretend a direction was evaluated.
        if result.get("status") == "FAIL_BETA_ONLY_INITIAL_STATE_IDENTITY":
            result["initial_identity"] = result.get("identity", {})
            result["initial_identity_status"] = result["status"]
            result["status"] = "FAIL_BETA_ONLY_DIRECTION"
            result["reason"] = (
                "Frozen six-particle sharp cohort and diffuse PF inventories are not identical; "
                "direction is fail-closed and was not adjudicated."
            )
        return result
    except Exception as error:
        return {"status": "FAIL_BETA_ONLY_DIRECTION", "reason": f"{type(error).__name__}: {error}", "no_cuda_rerun": True}


def _report_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], *, limit: int = 16) -> str:
    if not rows:
        return "No rows were produced."
    header = "| " + " | ".join(fields) + " |\n|" + "|".join("---" for _ in fields) + "|\n"
    body = ["| " + " | ".join(str(_csv_value(row.get(field, ""))) for field in fields) + " |" for row in rows[:limit]]
    suffix = "" if len(rows) <= limit else f"\n\nOnly the first {limit} rows are shown; the CSV is authoritative."
    return header + "\n".join(body) + suffix


def _status_stub(reason: str) -> dict[str, Any]:
    return {"status": "BLOCKED_PREREQUISITE_GATE", "reason": reason, "rows": [{"status": "BLOCKED_PREREQUISITE_GATE", "reason": reason}]}


def _write_outputs(
    *,
    output_root: Path,
    launch: Mapping[str, Any],
    context: Any,
    blocker: Mapping[str, Any],
    characteristic_tests: Mapping[str, Any],
    convergence: Mapping[str, Any],
    parity: Mapping[str, Any],
    frozen: Mapping[str, Any],
    ladder: Mapping[str, Any],
    one_hour: Mapping[str, Any],
    cost: Mapping[str, Any],
    qualification: Mapping[str, Any],
    authority: Mapping[str, Any],
    final_crosscheck: Mapping[str, Any],
    beta: Mapping[str, Any],
    final: Mapping[str, Any],
    inherited_tests: Mapping[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "figures").mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "explicit_reference_blocker.json", blocker)
    _write_csv(output_root / "characteristic_unit_tests.csv", characteristic_tests.get("rows", []), fallback_fields=("case", "status", "reason"))
    analytic_rows = [row for row in characteristic_tests.get("rows", []) if str(row.get("case", "")).startswith(("CR2", "CR3", "CR4", "CR8", "CR9"))]
    _write_csv(output_root / "characteristic_analytic_benchmarks.csv", analytic_rows, fallback_fields=("case", "status", "reason"))
    _write_csv(output_root / "characteristic_timestep_convergence.csv", convergence.get("rows", []), fallback_fields=("status", "reason"))
    _write_csv(output_root / "cohort_characteristic_parity.csv", parity.get("rows", []), fallback_fields=("status", "reason"))
    _write_csv(output_root / "frozen_matrix_three_method.csv", frozen.get("rows", []), fallback_fields=("status", "reason"))
    _write_csv(output_root / "implicit_vs_reference_01h.csv", ladder.get("rows", []), fallback_fields=("status", "reason"))
    _write_csv(output_root / "implicit_vs_reference_1h.csv", one_hour.get("rows", []), fallback_fields=("status", "reason"))
    _write_csv(output_root / "solver_cost_model.csv", [cost], fallback_fields=("status", "reason"))
    _write_csv(output_root / "population_solver_v2_qualification.csv", qualification.get("rows", []), fallback_fields=("status", "reason"))
    _write_json(output_root / "eulerian_authority_v2.json", authority)
    _write_csv(output_root / "final_three_method_crosscheck.csv", final_crosscheck.get("rows", []), fallback_fields=("status", "reason"))
    _write_csv(output_root / "beta_only_cohort_pf.csv", beta.get("rows", [beta]), fallback_fields=("status", "reason"))
    source_paths = (
        "scripts/run_kwn_characteristic_reference_v1.py",
        "scripts/run_kwn_lower_boundary_time_accuracy_v1.py",
        "scripts/run_kwn_time_accuracy_closure_v1.py",
        "src/kwn_mvp/characteristic_reference.py",
        "src/kwn_mvp/conservative_remap.py",
        "src/kwn_mvp/explicit_solver.py",
        "src/kwn_mvp/solver.py",
        "src/kwn_mvp/cohort_solver.py",
        "tests/kwn/test_characteristic_reference.py",
        "tests/kwn/test_explicit_ssprk2_solver.py",
    )
    provenance = {
        "schema_version": "KWN_CHARACTERISTIC_REFERENCE_ANALYSIS_PROVENANCE_V1",
        "launch": launch,
        "canonical_state_hash": context.canonical_hash,
        "canonical_psd_hash": context.source_initial_psd_hash,
        "canonical_config_hash": context.solver.config.source_config_hash,
        "validation_contract_hash": context.contract_hash,
        "fixture_hash": context.fixture_hash,
        "binary_provenance": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "implicit_solver_version": getattr(KWNSolver, "solver_version", "NOT_DECLARED"),
            "characteristic_solver_version": CharacteristicReferenceSolver.solver_version,
        },
        "analysis_parameters": {
            "characteristic_time_gate": REFERENCE_TIME_GATE,
            "characteristic_remap_order": REMAP_ORDER,
            "characteristic_trace_integrator": TRACE_INTEGRATOR,
            "cohort_characteristic_gate": ONE_PERCENT,
            "cohort_eulerian_gate": TWO_PERCENT,
            "cohort_points_per_cell": COHORT_POINTS_PER_CELL,
            "implicit_policies": [(name, cap) for name, cap in IMPLICIT_POLICIES],
        },
        "source_sha256": {path: _sha256_file(ROOT / path) for path in source_paths if (ROOT / path).is_file()},
        "inherited_52_tests": inherited_tests,
        "strict_ssprk2_blocker": blocker,
        "forbidden_actions": {"PF_SOURCE_MODIFIED": False, "CUDA_RERUN": False, "PHYSICAL_RETUNING": False, "LOCAL_GP_RELEASE": False, "GP_RELEASE": False},
        "final_status": final["STATUS"],
    }
    _write_json(output_root / "analysis_provenance.json", provenance)
    _write_json(output_root / "final_acceptance.json", final)
    (output_root / "figures" / "README.md").write_text(
        "# Characteristic-reference evidence\n\nCSV and JSON files, not figures, are the authority for this numerical gate.\n",
        encoding="utf-8",
    )


def _write_reports(
    *,
    report_root: Path,
    output_root: Path,
    blocker: Mapping[str, Any],
    characteristic_tests: Mapping[str, Any],
    convergence: Mapping[str, Any],
    restart: Mapping[str, Any],
    parity: Mapping[str, Any],
    frozen: Mapping[str, Any],
    ladder: Mapping[str, Any],
    one_hour: Mapping[str, Any],
    cost: Mapping[str, Any],
    qualification: Mapping[str, Any],
    authority: Mapping[str, Any],
    final_crosscheck: Mapping[str, Any],
    beta: Mapping[str, Any],
    final: Mapping[str, Any],
    command: str,
) -> None:
    bodies = {
        "00_explicit_reference_blocker_reproduction.md": (
            f"Status: `{blocker.get('status')}`.  The strict exact donor timestep is `{blocker.get('strict_donor_dt_s')}` s, "
            f"below frozen min_dt `{blocker.get('frozen_min_dt_s')}` s.  The first canonical cell remains positive: "
            f"`{blocker.get('first_cell_number_m3')}` m^-3 (M0 fraction `{blocker.get('first_cell_M0_fraction')}`, "
            f"M3 fraction `{blocker.get('first_cell_M3_fraction')}`).\n\n"
            f"The limiting cell/face/outflux are `{blocker.get('limiting_cell_index')}` / `{blocker.get('limiting_face_index')}` / "
            f"`{blocker.get('limiting_outgoing_flux_m3_s')}`.  Theoretical 0.1 h cost is "
            f"`{blocker.get('theoretical_steps_to_0p1h')}` accepted SSPRK2 steps.\n\n"
            f"All twenty unchanged rejected proposals are retained in `{output_root / 'explicit_reference_blocker.json'}`.  No active-tail proxy, thresholding, or physical change was used."
        ),
        "01_characteristic_method_contract.md": (
            "CR1 stores cell-integrated number `N_i`, traces every fixed grid face with deterministic autonomous-radius "
            "two-node Gauss--Legendre time-of-flight quadrature and a monotone inverse map, and remaps "
            "departure intervals with an exact piecewise-constant cumulative distribution.  The lower edge is the shared exact "
            "physical `Rmin`: there is no ghost inflow, and crossings leave the resolved domain.  Rmax outflow fails closed.\n\n"
            "For every nonzero step, CR1 iterates matrix composition and remapped population to a joint fixed point; a frozen-x "
            "single step is never labeled a full dynamic reference.  `Q_beta` is recomputed from the new cell measure and matrix "
            "composition comes only from the algebraic total-inventory ledger.  CR1 is scoped to smooth beta-only post-nucleation "
            "numerical reference work, not GP production."
        ),
        "02_characteristic_unit_tests.md": f"Status: `{characteristic_tests.get('status')}`.\n\n" + _report_table(characteristic_tests.get("rows", []), ("case", "test_id", "status", "reason")),
        "03_analytic_characteristic_benchmarks.md": "CR2 verifies constant positive translation; CR3 verifies negative translation with Rmin loss; CR4 checks the `-K/R` analytic characteristic and event; CR8 checks timestep refinement; CR9 verifies no donor-CFL dependency.\n\n" + _report_table([row for row in characteristic_tests.get("rows", []) if str(row.get("case", "")).startswith(("CR2", "CR3", "CR4", "CR8", "CR9"))], ("case", "test_id", "status", "reason")),
        "04_characteristic_self_convergence.md": (
            f"Status: `{convergence.get('status')}`.  Policy: `{json.dumps(_json_safe(convergence.get('policy')), sort_keys=True)}`. "
            f"Maximum fixed-point residual: `{convergence.get('max_fixed_point_residual')}`; inventory residual: `{convergence.get('max_inventory_relative_residual')}`; canonical restart: `{restart.get('status')}`.\n\n"
            + _report_table(convergence.get("rows", []), ("dt_s", "reference_dt_s", "time_h", "metric", "relative_or_absolute_error", "pass"))
        ),
        "05_cohort_characteristic_parity.md": f"Status: `{parity.get('status')}`; gated maximum error `{parity.get('gate_maximum_error')}` (1%, including normalized PSD Wasserstein); scalar-observable maximum `{parity.get('primary_maximum_error')}`; canonical positive quadrature `{parity.get('cohort_points_per_cell', COHORT_POINTS_PER_CELL)}` points per cell.\n\n" + _report_table(parity.get("rows", []), ("time_h", "metric", "relative_or_absolute_error", "pass")),
        "06_frozen_matrix_three_method.md": f"Status: `{frozen.get('status')}`; interpretation `{frozen.get('interpretation')}`.  This control freezes x_alpha only to isolate radius-space transport from dynamic matrix feedback.\n\n" + _report_table(frozen.get("rows", []), ("comparison", "time_h", "metric", "relative_or_absolute_error", "pass")),
        "07_implicit_reference_ladder_01h.md": f"Status: `{ladder.get('status')}`.  Selected broadest policy: `{ladder.get('selected_policy')}`.  Current implicit within the cohort/characteristic envelope: `{ladder.get('current_implicit_within_reference_envelope')}`.\n\n" + _report_table(ladder.get("rows", []), ("record_type", "policy", "active_cfl_cap", "time_h", "metric", "reference_envelope_error", "pass")),
        "08_implicit_reference_ladder_1h.md": f"Status: `{one_hour.get('status')}`.  Selected/next tighter: `{one_hour.get('selected_policy')}` / `{one_hour.get('next_tighter_policy')}`.\n\n" + _report_table(one_hour.get("rows", []), ("comparison", "time_h", "metric", "relative_or_absolute_error", "pass")),
        "09_solver_cost_and_decision.md": "```json\n" + json.dumps(_json_safe(cost), indent=2, sort_keys=True) + "\n```",
        "10_population_solver_v2_qualification.md": f"Status: `{qualification.get('status')}`.\n\n" + _report_table(qualification.get("rows", []), ("left_grid", "right_grid", "time_h", "metric", "relative_or_absolute_error", "pass")),
        "11_authority_v2.md": "```json\n" + json.dumps(_json_safe(authority), indent=2, sort_keys=True) + "\n```",
        "12_final_smooth_crosscheck.md": f"Status: `{final_crosscheck.get('status')}`.  The original 2% cohort--population shared-operator gate is unchanged.\n\n" + _report_table(final_crosscheck.get("rows", []), ("comparison", "time_h", "metric", "relative_or_absolute_error", "pass")),
        "13_beta_only_cohort_pf.md": f"Status: `{beta.get('status')}`.  Frozen PF evidence is read only; CUDA was not rerun.\n\n" + (str(beta.get("reason", "")) or "See `beta_only_cohort_pf.csv` for the event-aware comparator record."),
        "14_model_role_boundary.md": "The characteristic remap is an independent smooth-population time reference.  The discrete cohort remains the event-aware sharp-particle comparator.  PF remains a frozen spatial/elastic source and is not advanced here.  No GP release, GP-to-beta conversion, online coupling, physical retuning, PF source modification, or CUDA rerun occurred.",
        "15_final_acceptance_report.md": "```json\n" + json.dumps(_json_safe(final), indent=2, sort_keys=True) + "\n```",
        "16_reproduction_commands.md": (
            "```bash\n"
            f"PYTHONPATH=src {sys.executable} scripts/run_kwn_characteristic_reference_v1.py all --max-wall-s 3600 --max-steps 2000000\n"
            "```\n\n"
            f"Executed command: `{command}`.  The formal clean-launch guard rejects existing output roots, a dirty source tree, and a branch not descended from `{FROZEN_START_COMMIT}`."
        ),
    }
    for name, title in REQUIRED_REPORTS.items():
        _write_report(report_root / name, title, bodies[name])


def _final_record(
    *,
    launch: Mapping[str, Any],
    blocker: Mapping[str, Any],
    inherited_tests: Mapping[str, Any],
    boundary: Mapping[str, Any],
    current_cap4: Mapping[str, Any],
    characteristic_tests: Mapping[str, Any],
    convergence: Mapping[str, Any],
    restart: Mapping[str, Any],
    parity: Mapping[str, Any],
    frozen: Mapping[str, Any],
    ladder: Mapping[str, Any],
    one_hour: Mapping[str, Any],
    choice: Mapping[str, Any],
    qualification: Mapping[str, Any],
    authority: Mapping[str, Any],
    final_crosscheck: Mapping[str, Any],
    beta: Mapping[str, Any],
) -> dict[str, Any]:
    characteristic_test_status = str(characteristic_tests.get("status"))
    convergence_status = str(convergence.get("status"))
    restart_status = str(restart.get("status"))
    parity_status = str(parity.get("status"))
    strict_ssprk2_tests = inherited_tests.get("strict_ssprk2_unit_tests", {})
    if blocker.get("status") != "BLOCKED_EXACT_DONOR_BOUND_REFERENCE" or strict_ssprk2_tests.get("status") != "PASS":
        top = "FAIL_CHARACTERISTIC_REFERENCE_NUMERICS"
    elif (
        inherited_tests.get("status") != "PASS"
        or boundary.get("status") != "PASS_LOWER_BOUNDARY_OPERATOR_PARITY"
        or current_cap4.get("status") != "PASS_CURRENT_VS_CAP4_REPRODUCTION"
    ):
        top = "FAIL_CHARACTERISTIC_REFERENCE_NUMERICS"
    elif characteristic_test_status != "PASS_CHARACTERISTIC_REFERENCE_UNIT_TESTS":
        top = (
            "BLOCKED_TIME_REFERENCE_NOT_CLOSED"
            if characteristic_test_status.startswith(("BLOCKED", "INCOMPLETE"))
            else "FAIL_CHARACTERISTIC_REFERENCE_NUMERICS"
        )
    elif convergence_status != "PASS_CHARACTERISTIC_SELF_CONVERGENCE" or restart_status != "PASS_CHARACTERISTIC_RESTART":
        top = (
            "BLOCKED_TIME_REFERENCE_NOT_CLOSED"
            if convergence_status.startswith(("BLOCKED", "INCOMPLETE")) or restart_status.startswith(("BLOCKED", "INCOMPLETE"))
            else "FAIL_CHARACTERISTIC_REFERENCE_NUMERICS"
        )
    elif parity_status != "PASS_COHORT_CHARACTERISTIC_REFERENCE_PARITY":
        top = (
            "BLOCKED_TIME_REFERENCE_NOT_CLOSED"
            if parity_status.startswith(("BLOCKED", "INCOMPLETE"))
            else "FAIL_COHORT_CHARACTERISTIC_REFERENCE_PARITY"
        )
    elif frozen.get("status") != "PASS_FROZEN_MATRIX_THREE_METHOD":
        top = (
            "BLOCKED_TIME_REFERENCE_NOT_CLOSED"
            if str(frozen.get("status")).startswith(("BLOCKED", "INCOMPLETE"))
            else
            "FAIL_COHORT_CHARACTERISTIC_REFERENCE_PARITY"
            if frozen.get("status") == "FAIL_COHORT_CHARACTERISTIC_REFERENCE_PARITY"
            else "FAIL_CHARACTERISTIC_REFERENCE_NUMERICS"
        )
    elif ladder.get("status") != "PASS_IMPLICIT_REFERENCE_LADDER" or one_hour.get("status") != "PASS_IMPLICIT_1H_CONFIRMATION":
        top = (
            "BLOCKED_TIME_REFERENCE_NOT_CLOSED"
            if str(ladder.get("status")).startswith(("BLOCKED", "INCOMPLETE"))
            or str(one_hour.get("status")).startswith(("BLOCKED", "INCOMPLETE"))
            else "FAIL_IMPLICIT_TIME_ACCURACY"
        )
    elif qualification.get("status") != "PASS_POPULATION_SOLVER_V2_QUALIFICATION":
        top = (
            "BLOCKED_TIME_REFERENCE_NOT_CLOSED"
            if str(qualification.get("status")).startswith(("BLOCKED", "INCOMPLETE"))
            else "FAIL_RADIUS_SPACE_DISCRETIZATION_ACCURACY"
        )
    elif final_crosscheck.get("status") != "PASS_COHORT_EULERIAN_SHARED_OPERATOR_PARITY":
        top = (
            "BLOCKED_TIME_REFERENCE_NOT_CLOSED"
            if str(final_crosscheck.get("status")).startswith(("BLOCKED", "INCOMPLETE"))
            else "FAIL_COHORT_CHARACTERISTIC_REFERENCE_PARITY"
            if final_crosscheck.get("status") == "FAIL_COHORT_CHARACTERISTIC_REFERENCE_PARITY"
            else "FAIL_IMPLICIT_TIME_ACCURACY"
            if choice.get("final_population_solver") == "CONSERVATIVE_IMPLICIT_UPWIND_FACE_SOLVE"
            else "FAIL_CHARACTERISTIC_REFERENCE_NUMERICS"
        )
    elif beta.get("status") == "FAIL_BETA_ONLY_DIRECTION":
        top = "FAIL_BETA_ONLY_DIRECTION"
    elif beta.get("status") == "PASS_BETA_ONLY_DIRECTION":
        top = "PASS_SHARED_OPERATOR_PARITY_CONDITIONAL_MEAN_FIELD_GAP" if beta.get("mean_field_spatial_elastic_gap") == "CONDITIONAL_SPATIAL_ELASTIC_MEAN_FIELD_GAP" else "PASS_KWN_PF_ONE_WAY_STORAGE_COUPLING_V1"
    else:
        # All time-reference and final smooth gates above have passed, so an
        # absent frozen-PF adjudication is a beta-only closure failure, not a
        # reason to reclassify the independently closed time reference.
        top = "FAIL_BETA_ONLY_DIRECTION"
    cap_errors = ladder.get("envelope_errors", {})
    final_errors = final_crosscheck.get("cohort_final", {}).get("maximum_errors", {})
    current_cap4_difference = current_cap4.get("difference", {}).get("cumulative_number_dissolution_m3_relative_difference", "NOT_AVAILABLE")
    time_reference_closed = parity.get("status") == "PASS_COHORT_CHARACTERISTIC_REFERENCE_PARITY"
    return {
        "STATUS": top,
        "BRANCH": launch.get("git_branch_at_launch"),
        "COMMIT": launch.get("git_head_at_launch"),
        "BASELINE": "PASS" if inherited_tests.get("status") == "PASS" and boundary.get("status") == "PASS_LOWER_BOUNDARY_OPERATOR_PARITY" else "FAIL",
        "VALIDATION_CONTRACT_HASH": EXPECTED_CONTRACT_HASH,
        "BOUNDARY_REGRESSION": boundary.get("status"),
        "KWN_TESTS": f"{inherited_tests.get('test_count')}/52 {inherited_tests.get('status')}",
        "STRICT_SSPRK2_REFERENCE": blocker.get("status"),
        "STRICT_SSPRK2_UNIT_TESTS": f"{strict_ssprk2_tests.get('test_count')}/9 {strict_ssprk2_tests.get('status')}",
        "STRICT_DONOR_DT": blocker.get("strict_donor_dt_s"),
        "ESTIMATED_SSPRK2_01H_COST": blocker.get("theoretical_steps_to_0p1h"),
        "TIME_ACCURACY_STATUS": (
            "CLOSED_WITH_CONSERVATIVE_CHARACTERISTIC_REFERENCE_V2"
            if time_reference_closed
            and ladder.get("status") == "PASS_IMPLICIT_REFERENCE_LADDER"
            and one_hour.get("status") == "PASS_IMPLICIT_1H_CONFIRMATION"
            else "REFERENCE_V2_CLOSED_IMPLICIT_ADJUDICATION_INCOMPLETE"
            if time_reference_closed
            else "BLOCKED_NO_ADMISSIBLE_INDEPENDENT_TIME_REFERENCE"
        ),
        "CHARACTERISTIC_SOLVER": CharacteristicReferenceSolver.solver_version,
        "CHARACTERISTIC_REMAP_ORDER": REMAP_ORDER,
        "CHARACTERISTIC_SELF_CONVERGENCE": convergence.get("status"),
        "CHARACTERISTIC_MAX_RESIDUAL": convergence.get("max_fixed_point_residual"),
        "CHARACTERISTIC_RESTART": restart.get("status"),
        "COHORT_CHARACTERISTIC_PARITY": parity.get("status"),
        "COHORT_CHARACTERISTIC_MAX_ERROR": parity.get("gate_maximum_error"),
        "TIME_REFERENCE_V2": "CONSERVATIVE_CHARACTERISTIC_REMAP" if parity.get("status") == "PASS_COHORT_CHARACTERISTIC_REFERENCE_PARITY" else "NOT_ASSIGNED",
        "REFERENCE_CONFIG_HASH": _canonical_hash(convergence.get("policy", {})),
        "FROZEN_MATRIX_THREE_METHOD": frozen.get("status"),
        "DYNAMIC_MATRIX_THREE_METHOD": ladder.get("status"),
        "CURRENT_IMPLICIT_VS_REFERENCE": max(cap_errors.get("current", {}).values(), default="NOT_AVAILABLE"),
        "CAP4_VS_REFERENCE": max(cap_errors.get("cap4", {}).values(), default="NOT_AVAILABLE"),
        "CAP2_VS_REFERENCE": max(cap_errors.get("cap2", {}).values(), default="NOT_AVAILABLE"),
        "CAP1_VS_REFERENCE": max(cap_errors.get("cap1", {}).values(), default="NOT_AVAILABLE"),
        "CAP05_VS_REFERENCE": max(cap_errors.get("cap05", {}).values(), default="NOT_AVAILABLE"),
        "CAP025_VS_REFERENCE": max(cap_errors.get("cap025", {}).values(), default="NOT_AVAILABLE"),
        "IMPLICIT_TIME_INACCURACY_PROVEN": bool(ladder.get("status") == "PASS_IMPLICIT_REFERENCE_LADDER" and not ladder.get("current_implicit_within_reference_envelope", False)),
        "PASSING_IMPLICIT_POLICY": ladder.get("selected_policy", "NOT_ASSIGNED"),
        "PASSING_ACTIVE_CFL": None if ladder.get("selected_policy") is None else dict(IMPLICIT_POLICIES).get(ladder.get("selected_policy")),
        "FINAL_POPULATION_SOLVER_V2": choice.get("final_population_solver", "NOT_SELECTED"),
        "FINAL_TIME_POLICY_V2": choice.get("final_time_policy", "NOT_SELECTED"),
        "EULERIAN_AUTHORITY_GRID_V2": authority.get("authority_grid", "NOT_ASSIGNED"),
        "AUTHORITY_CONFIG_HASH": authority.get("AUTHORITY_CONFIG_HASH", "NOT_ASSIGNED"),
        "AUTHORITY_RESTART": restart.get("status"),
        "AUTHORITY_MAX_RESIDUAL": authority.get("max_residual", "NOT_AVAILABLE"),
        "FINAL_COHORT_EULERIAN_CROSSCHECK": final_crosscheck.get("status"),
        "N_M0_ERROR": final_errors.get("N_m0_m3", "NOT_EVALUATED"),
        "RMEAN_ERROR": final_errors.get("Rmean_m", "NOT_EVALUATED"),
        "RMEAN3_ERROR": final_errors.get("Rmean3_m3", "NOT_EVALUATED"),
        "SV_ERROR": final_errors.get("Sv_m_inv", "NOT_EVALUATED"),
        "FBETA_ERROR": final_errors.get("f_beta", "NOT_EVALUATED"),
        "XMATRIX_ERROR": final_errors.get("matrix_xB", "NOT_EVALUATED"),
        "CUM_NUMBER_DISSOLUTION_ERROR": final_errors.get("cumulative_number_dissolution_m3", "NOT_EVALUATED"),
        "CUM_MOL_B_ERROR": final_errors.get("cumulative_mol_B_returned_mol_m3", "NOT_EVALUATED"),
        "BETA_INITIAL_STATE_IDENTITY": beta.get("initial_identity", {}).get("status", "NOT_EVALUATED"),
        "BETA_ONLY_DIRECTION": beta.get("status"),
        "BETA_ONLY_TIMESCALE": beta.get("timescale_status", "NOT_EVALUATED"),
        "MEAN_FIELD_PF_GAP": beta.get("mean_field_spatial_elastic_gap", "NOT_EVALUATED"),
        "PF_SOURCE_MODIFIED": False,
        "CUDA_RERUN": False,
        "PHYSICAL_RETUNING": False,
        "LEGACY_SIX_PARTICLE_EULERIAN_P5": "FAIL_RETAINED",
        "HISTORICAL_AUTHORITY": "HISTORICAL_SMOOTH_3200_ONLY_V2_NOT_INHERITED",
        "TOP_5_FINDINGS": [
            f"Strict full-domain SSPRK2 remains {blocker.get('status')} at dt={blocker.get('strict_donor_dt_s')} s; the positive first cell was retained.",
            f"Characteristic CR1 self-convergence is {convergence.get('status')} and its canonical restart is {restart.get('status')}.",
            f"Cohort--characteristic parity is {parity.get('status')} with maximum gated error {parity.get('gate_maximum_error')}.",
            f"Current/cap=4 historical 0.1 h cumulative-dissolution difference remains {current_cap4_difference}; reference-based implicit ladder is {ladder.get('status')}.",
            f"Final solver choice is {choice.get('final_population_solver', 'NOT_SELECTED')}; final 2% crosscheck is {final_crosscheck.get('status')}.",
        ],
        "P0_BLOCKERS": [item for item in (characteristic_tests.get("status"), convergence.get("status"), parity.get("status"), ladder.get("status"), one_hour.get("status"), qualification.get("status"), final_crosscheck.get("status")) if str(item).startswith(("FAIL", "BLOCKED", "INCOMPLETE"))],
        "NEXT_ACTION": "No local GP release.  Resolve the first failed numerical gate above before any separately gated GP prototype.",
        "LOCAL_GP_RELEASE_AUTHORIZED": "ELIGIBLE_FOR_SEPARATELY_GATED_LOCAL_GP_RELEASE_PROTOTYPE" if top.startswith("PASS_") else "LOCAL_GP_RELEASE_NOT_AUTHORIZED",
        "KEY_REPORTS": [str(DEFAULT_REPORT_ROOT / name) for name in ("00_explicit_reference_blocker_reproduction.md", "04_characteristic_self_convergence.md", "07_implicit_reference_ladder_01h.md", "12_final_smooth_crosscheck.md", "15_final_acceptance_report.md")],
        "REQUIRED_CONCLUSIONS": {
            "1_strict_ssprk2_time_reference": (
                "NOT_FEASIBLE: the full-domain exact donor bound retains the positive first canonical cell, "
                f"is {blocker.get('strict_donor_dt_s')} s below frozen min_dt {blocker.get('frozen_min_dt_s')} s, "
                f"and would require {blocker.get('theoretical_steps_to_0p1h')} steps for 0.1 h."
                if blocker.get("status") == "BLOCKED_EXACT_DONOR_BOUND_REFERENCE"
                else f"BLOCKER REPRODUCTION FAILED: {blocker.get('status')}"
            ),
            "2_characteristic_independent_self_convergence": convergence.get("status"),
            "3_cohort_characteristic_same_smooth_solution": parity.get("status"),
            "4_current_implicit_time_accuracy_after_reference": ladder.get("diagnosis", ladder.get("status")),
            "5_final_population_solver": choice.get("final_population_solver", "NOT_SELECTED"),
            "6_original_2_percent_smooth_crosscheck": final_crosscheck.get("status"),
            "7_beta_only_mean_field_pf_direction": beta.get("status"),
            "8_local_gp_release_prototype_eligibility": "ELIGIBLE_FOR_SEPARATELY_GATED_LOCAL_GP_RELEASE_PROTOTYPE" if top.startswith("PASS_") else "NOT_AUTHORIZED",
        },
    }


def _workflow(arguments: argparse.Namespace) -> dict[str, Any]:
    output_root = arguments.output_root.resolve(); report_root = arguments.report_root.resolve()
    _require_formal_launch(output_root, report_root)
    launch = _launch_context()
    legacy = _load_legacy_lower(); time_legacy = _load_time_closure()
    context = legacy._build_canonical_context(bins=3200)
    identity = {
        "validation_contract_hash": context.contract_hash == EXPECTED_CONTRACT_HASH,
        "canonical_state_hash": context.canonical_hash == EXPECTED_CANONICAL_STATE_HASH,
        "canonical_psd_hash": context.source_initial_psd_hash == EXPECTED_CANONICAL_PSD_HASH,
        "fixture_hash": context.fixture_hash == EXPECTED_FIXTURE_HASH,
    }
    if not all(identity.values()):
        raise WorkflowError(f"frozen canonical identity check failed: {identity}")
    # This is intentionally the first new numerical action after identity
    # reconstruction.  CR1 is not allowed to erase the SSPRK2 failure.
    try:
        blocker = _strict_explicit_blocker(context)
    except Exception as error:
        blocker = {
            "schema_version": "KWN_EXACT_DONOR_BOUND_BLOCKER_REPRODUCTION_V1",
            "status": "FAIL_EXPLICIT_BLOCKER_REPRODUCTION",
            "reason": f"{type(error).__name__}: {error}",
            "first_twenty_proposed_steps": [],
        }
    inherited_tests = _run_test_modules(BASELINE_TEST_MODULES)
    inherited_tests["strict_ssprk2_unit_tests"] = _run_test_modules(STRICT_SSPRK2_TEST_MODULES)
    frozen_test = _run_test_modules(("tests.kwn.test_lower_boundary_contract_frozen",))
    direct = legacy._physical_boundary_parity(context)
    boundary = {
        "status": "PASS_LOWER_BOUNDARY_OPERATOR_PARITY" if inherited_tests.get("status") == "PASS" and frozen_test.get("status") == "PASS" and direct.get("status") == "PASS_PHYSICAL_LOWER_BOUNDARY_PARITY" else "FAIL_LOWER_BOUNDARY_OPERATOR_PARITY",
        "direct": direct, "frozen_test": frozen_test,
    }
    if blocker.get("status") == "BLOCKED_EXACT_DONOR_BOUND_REFERENCE" and inherited_tests["strict_ssprk2_unit_tests"].get("status") == "PASS" and boundary["status"] == "PASS_LOWER_BOUNDARY_OPERATOR_PARITY":
        try:
            current_cap4 = time_legacy._run_current_vs_cap4(
                legacy=legacy,
                context=context,
                max_steps=arguments.max_steps,
                max_wall_s=arguments.max_wall_s,
                output_root=output_root,
                launch=launch,
            )
        except Exception as error:
            current_cap4 = {
                "status": "FAIL_CURRENT_VS_CAP4_REPRODUCTION",
                "reason": f"{type(error).__name__}: {error}",
                "rows": [],
                "cfl_rows": [],
                "difference": {},
            }
    else:
        current_cap4 = {
            "status": "FAIL_CURRENT_VS_CAP4_REPRODUCTION",
            "reason": (
                "strict SSPRK2 blocker reproduction failed"
                if blocker.get("status") != "BLOCKED_EXACT_DONOR_BOUND_REFERENCE"
                else "strict SSPRK2 unit qualification failed"
                if inherited_tests["strict_ssprk2_unit_tests"].get("status") != "PASS"
                else "inherited lower-boundary gate failed"
            ),
            "rows": [],
            "cfl_rows": [],
            "difference": {},
        }
    characteristic_tests = _run_characteristic_contract_tests()
    restart = _status_stub("characteristic unit-test or lower-boundary prerequisite failed")
    convergence: dict[str, Any] = _status_stub("characteristic unit-test or current/cap4 baseline prerequisite failed")
    parity: dict[str, Any] = _status_stub("characteristic self-convergence prerequisite failed")
    frozen: dict[str, Any] = _status_stub("cohort--characteristic parity prerequisite failed")
    ladder: dict[str, Any] = _status_stub("cohort--characteristic parity prerequisite failed")
    one_hour: dict[str, Any] = _status_stub("implicit 0.1 h ladder prerequisite failed")
    choice: dict[str, Any] = {"status": "NOT_SELECTED", "reason": "short-horizon gates incomplete"}
    qualification: dict[str, Any] = _status_stub("final solver policy prerequisite failed")
    authority: dict[str, Any] = {"schema_version": "EULERIAN_SMOOTH_AUTHORITY_GRID_V2", "status": "BLOCKED_PREREQUISITE_GATE", "authority_grid": None}
    final_crosscheck: dict[str, Any] = _status_stub("authority qualification prerequisite failed")
    beta: dict[str, Any] = _status_stub("PASS_COHORT_EULERIAN_SHARED_OPERATOR_PARITY prerequisite failed")
    cohort_cache: CohortTrajectoryCache | None = None
    if current_cap4.get("status") == "PASS_CURRENT_VS_CAP4_REPRODUCTION" and characteristic_tests.get("status") == "PASS_CHARACTERISTIC_REFERENCE_UNIT_TESTS":
        try:
            dt_policy = _derive_characteristic_dt(context, current_cap4)
            convergence = _characteristic_self_convergence(context, dt_policy=dt_policy, max_steps=arguments.max_steps, max_wall_s=arguments.max_wall_s)
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            convergence = {
                "status": (
                    "BLOCKED_TIME_REFERENCE_NOT_CLOSED"
                    if "max_steps=" in reason or "wall budget" in reason or "max_wall_s=" in reason
                    else "FAIL_CHARACTERISTIC_REFERENCE_NUMERICS"
                ),
                "reason": reason,
                "rows": [{"status": "INCOMPLETE", "reason": reason}],
                "policy": None,
            }
        if convergence.get("status") == "PASS_CHARACTERISTIC_SELF_CONVERGENCE":
            restart = _characteristic_restart(
                context,
                dt_s=float(convergence["policy"]["dt_s"]),
                max_wall_s=arguments.max_wall_s,
            )
        if convergence.get("status") == "PASS_CHARACTERISTIC_SELF_CONVERGENCE" and restart.get("status") == "PASS_CHARACTERISTIC_RESTART":
            selected_name = convergence["policy"]["name"]
            selected_characteristic = convergence["runs"][selected_name]
            parity = _cohort_characteristic_parity(legacy, context, selected_characteristic, max_wall_s=arguments.max_wall_s)
            if parity.get("status") == "PASS_COHORT_CHARACTERISTIC_REFERENCE_PARITY":
                cohort_cache = parity.get("cache")
                if not isinstance(cohort_cache, CohortTrajectoryCache):
                    raise WorkflowError("cohort parity completed without its canonical trajectory cache")
                frozen = _frozen_matrix_benchmark(legacy, context, char_dt_s=float(convergence["policy"]["dt_s"]), max_wall_s=arguments.max_wall_s)
                if frozen.get("status") == "PASS_FROZEN_MATRIX_THREE_METHOD":
                    ladder = _implicit_ladder(legacy, context, characteristic=selected_characteristic, cohort=parity["cohort"], target_times_h=SHORT_TIMES_H, max_steps=arguments.max_steps, max_wall_s=arguments.max_wall_s)
                else:
                    ladder = _status_stub("frozen-matrix three-method transport control failed")
                if ladder.get("status") == "PASS_IMPLICIT_REFERENCE_LADDER":
                    one_hour = _one_hour_confirmation(legacy, context, characteristic_dt_s=float(convergence["policy"]["dt_s"]), selected_policy=str(ladder["selected_policy"]), max_steps=arguments.max_steps, max_wall_s=arguments.max_wall_s, cohort_cache=cohort_cache)
                    if one_hour.get("status") == "PASS_IMPLICIT_1H_CONFIRMATION":
                        choice = _solver_choice(one_hour, characteristic_dt_s=float(convergence["policy"]["dt_s"]), selected_policy=str(ladder["selected_policy"]))
                        final_solver = choice["final_population_solver"]
                        cap = choice["final_time_policy"].get("active_cfl_cap") if final_solver == "CONSERVATIVE_IMPLICIT_UPWIND_FACE_SOLVE" else None
                        qualification = _qualification_grid(legacy, context, final_solver=final_solver, characteristic_dt_s=float(convergence["policy"]["dt_s"]), implicit_cap=cap, max_steps=arguments.max_steps, max_wall_s=arguments.max_wall_s, inherited_tests=inherited_tests, characteristic_tests=characteristic_tests, restart=restart, time_convergence=convergence)
                        qualification_summary = {key: value for key, value in qualification.items() if key != "runs"}
                        authority = {
                            "schema_version": "EULERIAN_SMOOTH_AUTHORITY_GRID_V2",
                            "status": (
                                "PASS_EULERIAN_SMOOTH_POPULATION_AUTHORITY_V2"
                                if qualification.get("status") == "PASS_POPULATION_SOLVER_V2_QUALIFICATION"
                                else "BLOCKED_TIME_REFERENCE_NOT_CLOSED"
                                if str(qualification.get("status")).startswith(("BLOCKED", "INCOMPLETE"))
                                else "FAIL_RADIUS_SPACE_DISCRETIZATION_ACCURACY"
                            ),
                            "authority_grid": 3200 if qualification.get("status") == "PASS_POPULATION_SOLVER_V2_QUALIFICATION" else None,
                            "solver": final_solver,
                            "time_policy": choice["final_time_policy"],
                            "AUTHORITY_CONFIG_HASH": _canonical_hash({"solver": final_solver, "time_policy": choice["final_time_policy"], "grid": 3200, "canonical": context.canonical_hash}),
                            "max_residual": max(float(convergence.get("max_fixed_point_residual", 0.0)), max((run.max_inventory_relative_residual for run in qualification.get("runs", {}).values()), default=float("inf"))),
                            "qualification": qualification_summary,
                        }
                        if qualification.get("status") == "PASS_POPULATION_SOLVER_V2_QUALIFICATION" and not arguments.skip_final_48h:
                            final_crosscheck = _final_smooth_crosscheck(legacy, context, choice=choice, characteristic_dt_s=float(convergence["policy"]["dt_s"]), max_steps=arguments.max_steps, max_wall_s=arguments.max_wall_s, cohort_cache=cohort_cache)
                            if final_crosscheck.get("status") == "PASS_COHORT_EULERIAN_SHARED_OPERATOR_PARITY":
                                beta = _beta_only_pf_compare(output_root, report_root)
                        elif qualification.get("status") == "PASS_POPULATION_SOLVER_V2_QUALIFICATION":
                            final_crosscheck = _status_stub("48 h final crosscheck intentionally skipped by command-line request")
    cost = {
        "status": choice.get("status"),
        **choice,
        "estimated_characteristic_48h_cpu_s": (choice.get("characteristic_runtime_s_1h") or 0.0) * 48.0,
        "estimated_implicit_48h_cpu_s": (choice.get("implicit_runtime_s_1h") or 0.0) * 48.0,
        "strict_ssprk2_48h_steps": blocker.get("theoretical_steps_to_48h"),
        "characteristic_restart": restart,
        "implicit_restart_basis": "inherited KWN restart test suite; no new implicit implementation was introduced by this runner",
    }
    final = _final_record(launch=launch, blocker=blocker, inherited_tests=inherited_tests, boundary=boundary, current_cap4=current_cap4, characteristic_tests=characteristic_tests, convergence=convergence, restart=restart, parity=parity, frozen=frozen, ladder=ladder, one_hour=one_hour, choice=choice, qualification=qualification, authority=authority, final_crosscheck=final_crosscheck, beta=beta)
    _write_outputs(output_root=output_root, launch=launch, context=context, blocker=blocker, characteristic_tests=characteristic_tests, convergence=convergence, parity=parity, frozen=frozen, ladder=ladder, one_hour=one_hour, cost=cost, qualification=qualification, authority=authority, final_crosscheck=final_crosscheck, beta=beta, final=final, inherited_tests=inherited_tests)
    _write_reports(report_root=report_root, output_root=output_root, blocker=blocker, characteristic_tests=characteristic_tests, convergence=convergence, restart=restart, parity=parity, frozen=frozen, ladder=ladder, one_hour=one_hour, cost=cost, qualification=qualification, authority=authority, final_crosscheck=final_crosscheck, beta=beta, final=final, command=" ".join(sys.argv))
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all",), help="run the complete gated characteristic-reference workflow")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--max-steps", type=int, default=2_000_000)
    parser.add_argument("--max-wall-s", type=float, default=3_600.0)
    parser.add_argument("--skip-final-48h", action="store_true", help="write an explicit incomplete final-stage record after the short gates")
    args = parser.parse_args()
    if args.max_steps <= 0 or args.max_wall_s <= 0.0:
        parser.error("--max-steps and --max-wall-s must be positive")
    final = _workflow(args)
    print(json.dumps(_json_safe(final), indent=2, sort_keys=True))
    return 0 if str(final["STATUS"]).startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
