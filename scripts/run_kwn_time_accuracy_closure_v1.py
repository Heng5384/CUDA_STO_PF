#!/usr/bin/env python3
"""Gate the KWN Eulerian time-accuracy closure workflow.

This runner is intentionally KWN-only.  It carries the physical-Rmin
operator contract forward unchanged, establishes the requested explicit
SSPRK2 donor-bound reference hierarchy, and stops at the first unavailable
reference gate rather than silently substituting an active-tail approximation.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kwn_mvp.solver import SolverConfig, SolverStateError  # noqa: E402


TASK_NAME = "kwn_time_accuracy_closure_v1"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / TASK_NAME
DEFAULT_REPORT_ROOT = ROOT / "reports" / TASK_NAME
EXPECTED_CONTRACT_HASH = "d0ff02973ab0f737043e1a40d4f69893a469cbfe2bc4cd22f9e6a410bd0b1333"
FROZEN_START_COMMIT = "4db6fbc81145d47f9d2e9008760fb85d7da69f85"
REQUIRED_BRANCH = "codex/kwn-time-accuracy-closure-v1"
EXPECTED_CANONICAL_STATE_HASH = "45fc9cdd1a2e2ddfe357122dff26b202fb42893a627202c43904017e428aac95"
EXPECTED_CANONICAL_SOURCE_PSD_HASH = "f9bb99dab64de15b946fba8904efab30d1881833577b8a5b28c8ea46a5ccf608"
EXPECTED_FIXTURE_HASH = "f1247cb66419af764b97de2f7843fc6de2049459d78550bc603edd8e88d9134f"
EXPECTED_CURRENT_CAP4_DISSOLUTION_RELATIVE_DIFFERENCE = 0.042544133625013775
EXPECTED_CURRENT_CAP4_MOL_B_RELATIVE_DIFFERENCE = 0.042544133625013886
BASELINE_REPRODUCTION_RELATIVE_TOLERANCE = 1.0e-10
TWO_PERCENT = 0.02
EXPLICIT_SAFETIES = (1.0, 0.5, 0.25, 0.125)
SHORT_TIMES_H = (0.0, 0.1)
FULL_TIMES_H = (0.0, 0.1, 0.39317699499770825, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0)
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
REQUIRED_REPORTS = {
    "00_baseline_reproduction.md": "Baseline reproduction",
    "01_time_reference_definition.md": "Time-reference definition",
    "02_explicit_ssprk2_qualification.md": "Explicit SSPRK2 qualification",
    "03_cohort_explicit_reference_parity.md": "Cohort--explicit reference parity",
    "04_implicit_active_cfl_01h.md": "Implicit active-CFL 0.1 h",
    "05_implicit_active_cfl_1h.md": "Implicit active-CFL 1 h",
    "06_48h_cost_model.md": "48 h cost model",
    "07_48h_finalists.md": "48 h finalists",
    "08_time_scheme_decision.md": "Time-scheme decision",
    "09_eulerian_authority_v2.md": "Eulerian authority V2",
    "10_final_cohort_eulerian_crosscheck.md": "Final cohort--Eulerian crosscheck",
    "11_beta_only_cohort_pf_comparison.md": "Beta-only cohort--PF comparison",
    "12_model_role_boundary.md": "Model-role boundary",
    "13_final_acceptance_report.md": "Final acceptance report",
    "14_reproduction_commands.md": "Reproduction commands",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(dict(value)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, Mapping)):
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
            if key not in fields:
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
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )
    return completed.stdout.strip()


def _git_succeeds(*args: str) -> bool:
    """Return whether a non-mutating git predicate succeeds."""

    return subprocess.run(
        ["git", *args], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    ).returncode == 0


def _require_formal_launch(*, output_root: Path, report_root: Path) -> None:
    """Reject a launch that cannot be an immutable, independently auditable run."""

    status = _git("status", "--short")
    if status:
        raise RuntimeError("formal time-accuracy launch requires a clean source tree")
    if _git("branch", "--show-current") != REQUIRED_BRANCH:
        raise RuntimeError(f"formal time-accuracy launch requires branch {REQUIRED_BRANCH!r}")
    if not _git_succeeds("merge-base", "--is-ancestor", FROZEN_START_COMMIT, "HEAD"):
        raise RuntimeError("formal launch HEAD is not descended from the frozen 4db6fbc start")
    for path, label in ((output_root, "output"), (report_root, "report")):
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(
                f"formal {label} root already contains evidence: {path}; use a fresh root rather than overwrite it"
            )


def _load_legacy_runner() -> Any:
    """Load the frozen lower-boundary runner without importing a user worktree."""

    path = ROOT / "scripts" / "run_kwn_lower_boundary_time_accuracy_v1.py"
    specification = importlib.util.spec_from_file_location("kwn_lower_boundary_runner_v1", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load frozen KWN runner: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _runtime_context() -> dict[str, Any]:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False
    ).stdout
    try:
        import scipy

        scipy_version: str | None = str(scipy.__version__)
    except ImportError:
        scipy_version = None
    cpu_model = platform.processor()
    if sys.platform == "darwin":
        probe = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        cpu_model = probe.stdout.strip() or cpu_model
    elif Path("/proc/cpuinfo").is_file():
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    allocated_threads = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("OMP_NUM_THREADS") or "1"
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "allocated_threads": allocated_threads,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy_version,
        "pip_freeze_sha256": hashlib.sha256(freeze.encode("utf-8")).hexdigest(),
        "thread_environment": {
            key: os.environ.get(key, "")
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
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


def _run_manifest(
    *,
    output_root: Path,
    name: str,
    launch: Mapping[str, Any],
    context: Any,
    solver: str,
    policy: str,
    config_mapping: Mapping[str, Any],
    result: Mapping[str, Any],
    execution: Mapping[str, Any] | None = None,
    snapshots: Sequence[Mapping[str, Any]] = (),
    trace_rows: Sequence[Mapping[str, Any]] = (),
) -> Path:
    run_root = output_root / "runs" / name
    run_root.mkdir(parents=True, exist_ok=True)
    config_path = run_root / "config.json"
    _write_json(config_path, config_mapping)
    result_path = run_root / "result.json"
    _write_json(result_path, result)
    artifacts: dict[str, str] = {
        "config.json": _sha256_file(config_path),
        "result.json": _sha256_file(result_path),
    }
    if snapshots:
        snapshots_path = run_root / "snapshots.json"
        _write_json(snapshots_path, {"snapshots": list(snapshots)})
        artifacts["snapshots.json"] = _sha256_file(snapshots_path)
    if trace_rows:
        trace_path = run_root / "accepted_step_trace.csv"
        _write_csv(trace_path, trace_rows, fallback_fields=("status", "reason"))
        artifacts["accepted_step_trace.csv"] = _sha256_file(trace_path)
    solver_path = (
        "src/kwn_mvp/explicit_solver.py"
        if solver == "EXPLICIT_UPWIND_DONOR_BOUND_SSPRK2"
        else "src/kwn_mvp/solver.py"
    )
    summary = result.get("summary", result) if isinstance(result, Mapping) else result
    manifest = {
        "schema_version": "KWN_TIME_ACCURACY_CLOSURE_RUN_MANIFEST_V1",
        "task": TASK_NAME,
        "launch": launch,
        "solver": solver,
        "policy": policy,
        "grid_bins": int(context.solver.config.grid.bins),
        "config_hash": _canonical_hash(config_mapping),
        "execution_policy": {} if execution is None else dict(execution),
        "execution_policy_hash": _canonical_hash({} if execution is None else execution),
        "canonical_state_hash": context.canonical_hash,
        "validation_contract_hash": context.contract_hash,
        "solver_source_sha256": _sha256_file(ROOT / solver_path),
        "accepted_steps": summary.get("accepted_steps"),
        "rejected_steps": summary.get("rejected_steps"),
        "runtime_s": summary.get("runtime_s"),
        "output_artifact_sha256": artifacts,
        "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = run_root / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _baseline_artifact_reproduction(legacy: Any) -> dict[str, Any]:
    frozen_root = ROOT / "outputs" / "kwn_lower_boundary_time_accuracy_v1"
    baseline = legacy._baseline_reproduction(frozen_root)
    return {
        **baseline,
        "source_output_root": str(frozen_root),
        "required_start_commit": FROZEN_START_COMMIT,
    }


def _run_unittest_modules(modules: Sequence[str]) -> dict[str, Any]:
    command = [sys.executable, "-m", "unittest", *modules]
    environment = dict(os.environ)
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = source_path + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    started = time.monotonic()
    completed = subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment, check=False
    )
    output = completed.stdout
    match = re.search(r"Ran (\d+) tests", output)
    return {
        "command": command,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "returncode": completed.returncode,
        "test_count": None if match is None else int(match.group(1)),
        "runtime_s": time.monotonic() - started,
        "output": output[-12000:],
    }


def _eulerian_run_row(run: Any, *, baseline_role: str) -> dict[str, Any]:
    final = dict(run.snapshots[-1]) if run.snapshots else {}
    return {
        "baseline_role": baseline_role,
        "policy": run.policy,
        "requested_active_cfl": "UNBOUNDED" if run.requested_active_cfl is None else run.requested_active_cfl,
        "status": run.status,
        "reason": run.reason or "",
        "accepted_steps": run.accepted_steps,
        "rejected_steps": run.rejected_steps,
        "runtime_s": run.runtime_s,
        **final,
    }


def _run_current_vs_cap4(
    *, legacy: Any, context: Any, max_steps: int, max_wall_s: float, output_root: Path, launch: Mapping[str, Any]
) -> dict[str, Any]:
    runs: dict[str, Any] = {}
    run_mappings: dict[str, dict[str, Any]] = {}
    for name, cap in (("current_implicit_policy", None), ("implicit_active_CFL_lte_4", 4.0)):
        mapping = deepcopy(context.mapping)
        mapping["simulation"]["accuracy_active_radius_cfl"] = cap
        solver = legacy._build_canonical_solver_from_context(context, active_cfl_cap=cap)
        run = legacy._advance_eulerian(
            solver,
            policy=name,
            target_times_h=SHORT_TIMES_H,
            active_cfl_cap=cap,
            max_steps=max_steps,
            max_wall_s=max_wall_s,
        )
        runs[name] = run
        run_mappings[name] = mapping
        row = _eulerian_run_row(run, baseline_role="current_vs_cap4")
        _run_manifest(
            output_root=output_root,
            name=f"baseline_{'current' if cap is None else 'cap4'}",
            launch=launch,
            context=context,
            solver="CONSERVATIVE_IMPLICIT_UPWIND_FACE_SOLVE",
            policy=name,
            config_mapping=mapping,
            execution={"active_cfl_cap": "UNBOUNDED" if cap is None else cap, "time_h": list(SHORT_TIMES_H)},
            result={"summary": row, "feasibility": run.feasibility},
            snapshots=run.snapshots,
            trace_rows=run.cfl_rows,
        )
    current, cap4 = runs["current_implicit_policy"], runs["implicit_active_CFL_lte_4"]
    row_current = _eulerian_run_row(current, baseline_role="current_vs_cap4")
    row_cap4 = _eulerian_run_row(cap4, baseline_role="current_vs_cap4")
    metrics = (
        "cumulative_number_dissolution_m3",
        "cumulative_beta_volume_dissolution",
        "cumulative_mol_B_returned_mol_m3",
    )
    difference: dict[str, float | str] = {}
    if current.snapshots and cap4.snapshots:
        for metric in metrics:
            observed = float(current.snapshots[-1][metric])
            reference = float(cap4.snapshots[-1][metric])
            difference[f"{metric}_relative_difference"] = abs(observed - reference) / max(abs(reference), 1.0e-300)
    else:
        difference = {f"{metric}_relative_difference": "NOT_AVAILABLE" for metric in metrics}
    reproduction_checks = {
        "current_short_run": current.status == "PASS_EULERIAN_SHORT_RUN",
        "cap4_short_run": cap4.status == "PASS_EULERIAN_SHORT_RUN",
        "numeric_differences": all(isinstance(value, float) for value in difference.values()),
        "number_dissolution_difference": isinstance(
            difference.get("cumulative_number_dissolution_m3_relative_difference"), float
        ) and math.isclose(
            float(difference["cumulative_number_dissolution_m3_relative_difference"]),
            EXPECTED_CURRENT_CAP4_DISSOLUTION_RELATIVE_DIFFERENCE,
            rel_tol=BASELINE_REPRODUCTION_RELATIVE_TOLERANCE,
            abs_tol=0.0,
        ),
        "mol_B_difference": isinstance(
            difference.get("cumulative_mol_B_returned_mol_m3_relative_difference"), float
        ) and math.isclose(
            float(difference["cumulative_mol_B_returned_mol_m3_relative_difference"]),
            EXPECTED_CURRENT_CAP4_MOL_B_RELATIVE_DIFFERENCE,
            rel_tol=BASELINE_REPRODUCTION_RELATIVE_TOLERANCE,
            abs_tol=0.0,
        ),
    }
    passed = (
        current.status == "PASS_EULERIAN_SHORT_RUN"
        and cap4.status == "PASS_EULERIAN_SHORT_RUN"
        and all(reproduction_checks.values())
    )
    return {
        "status": "PASS_CURRENT_VS_CAP4_REPRODUCTION" if passed else "FAIL_CURRENT_VS_CAP4_REPRODUCTION",
        "rows": [row_current, row_cap4],
        "cfl_rows": [row for run in runs.values() for row in run.cfl_rows],
        "difference": difference,
        "reproduction_checks": reproduction_checks,
        "expected_cumulative_dissolution_relative_difference": EXPECTED_CURRENT_CAP4_DISSOLUTION_RELATIVE_DIFFERENCE,
        "expected_cumulative_mol_B_relative_difference": EXPECTED_CURRENT_CAP4_MOL_B_RELATIVE_DIFFERENCE,
        "reproduction_relative_tolerance": BASELINE_REPRODUCTION_RELATIVE_TOLERANCE,
        "actual_config_hashes": {
            name: _canonical_hash(mapping) for name, mapping in run_mappings.items()
        },
        "current": row_current,
        "cap4": row_cap4,
    }


def _explicit_preflight(
    *, context: Any, output_root: Path, launch: Mapping[str, Any]
) -> dict[str, Any]:
    """Run the exact full-domain donor-bound first-step gate only.

    The requested safety ladder is evaluated from the solver's actual donor
    operator.  A sub-minimum candidate is a computational-reference block,
    not a change to physical inputs and not an active-tail proxy.
    """

    try:
        from kwn_mvp.explicit_solver import ExplicitSSPRK2Solver
    except ImportError as error:
        return {
            "status": "FAIL_EXPLICIT_SSPRK2_IMPLEMENTATION",
            "reason": f"cannot import ExplicitSSPRK2Solver: {error}",
            "rows": [],
        }
    rows: list[dict[str, Any]] = []
    for safety in EXPLICIT_SAFETIES:
        solver = ExplicitSSPRK2Solver(SolverConfig.from_mapping(context.mapping), donor_safety=safety)
        donor = solver.donor_bound_diagnostics()
        bound = float(donor.bound_s)
        proposed = float(safety * bound)
        started = time.monotonic()
        try:
            diagnostic = solver.advance_one(maximum_dt_s=SHORT_TIMES_H[-1] * 3600.0)
            status = "PASS_FIRST_EXPLICIT_STEP"
            reason = ""
            accepted = 1
            diagnostic_dt = float(diagnostic.dt_s)
        except SolverStateError as error:
            status = "REJECTED_BELOW_FROZEN_MIN_DT"
            reason = str(error)
            accepted = 0
            diagnostic_dt = 0.0
        row = {
            "solver": "EXPLICIT_UPWIND_DONOR_BOUND_SSPRK2",
            "grid": int(context.solver.config.grid.bins),
            "donor_safety": safety,
            "exact_donor_bound_s": bound,
            "controlling_population": donor.population_name,
            "controlling_cell_index": donor.cell_index,
            "controlling_cell_number_m3": donor.donor_number_m3,
            "controlling_cell_outgoing_flux_m3_s": donor.outgoing_flux_m3_s,
            "proposed_dt_s": proposed,
            "frozen_min_dt_s": float(solver.config.min_dt_s),
            "estimated_steps_to_0p1h": 360.0 / proposed,
            "estimated_steps_to_48h": 48.0 * 3600.0 / proposed,
            "status": status,
            "reason": reason,
            "accepted_steps": accepted,
            "rejected_steps": int(getattr(solver, "rejected_step_count", 0)),
            "diagnostic_dt_s": diagnostic_dt,
            "runtime_s": time.monotonic() - started,
        }
        rows.append(row)
        _run_manifest(
            output_root=output_root,
            name=f"explicit_donor_safety_{str(safety).replace('.', 'p')}",
            launch=launch,
            context=context,
            solver="EXPLICIT_UPWIND_DONOR_BOUND_SSPRK2",
            policy=f"donor_safety_{safety:g}",
            config_mapping=context.mapping,
            execution={
                "donor_safety": safety,
                "full_domain_exact_donor_bound": True,
                "time_h": list(SHORT_TIMES_H),
            },
            result=row,
            trace_rows=(row,),
        )
    all_rejected = all(row["status"] == "REJECTED_BELOW_FROZEN_MIN_DT" for row in rows)
    status = "BLOCKED_EXACT_DONOR_BOUND_REFERENCE" if all_rejected else "EXPLICIT_FIRST_STEP_AVAILABLE"
    return {
        "status": status,
        "rows": rows,
        "reason": (
            "The frozen canonical measure has a positive first physical cell.  Its exact full-domain donor bound is below "
            "the frozen minimum timestep, so no requested 0--0.1 h SSPRK2 reference trajectory is available."
            if all_rejected else "At least one safety permits an explicit first step; run the full explicit temporal ladder."
        ),
        "self_convergence": "NOT_RUN_REFERENCE_TRAJECTORY_BLOCKED" if all_rejected else "PENDING",
    }


def _blocked_row(reason: str) -> dict[str, Any]:
    return {"status": "BLOCKED_PREREQUISITE_GATE", "reason": reason}


def _not_run_current_cap4(reason: str) -> dict[str, Any]:
    """Return a schema-stable baseline placeholder after an earlier gate stops."""

    return {
        "status": "BLOCKED_PREREQUISITE_GATE",
        "reason": reason,
        "rows": [_blocked_row(reason)],
        "cfl_rows": [],
        "difference": {},
        "current": {},
        "cap4": {},
    }


def _not_run_explicit(reason: str) -> dict[str, Any]:
    return {
        "status": "BLOCKED_PREREQUISITE_GATE",
        "reason": reason,
        "rows": [],
        "self_convergence": "NOT_RUN_PREREQUISITE_GATE",
    }


def _test_gate(test_result: Mapping[str, Any], *, expected_count: int) -> bool:
    return test_result.get("status") == "PASS" and test_result.get("test_count") == expected_count


def _build_final(
    *,
    launch: Mapping[str, Any],
    historical_baseline: Mapping[str, Any],
    frozen: Mapping[str, Any],
    lower_boundary: Mapping[str, Any],
    tests: Mapping[str, Any],
    current_cap4: Mapping[str, Any],
    explicit_tests: Mapping[str, Any],
    explicit: Mapping[str, Any],
) -> dict[str, Any]:
    inherited_tests_pass = _test_gate(tests, expected_count=52)
    explicit_tests_pass = _test_gate(explicit_tests, expected_count=9)
    boundary_pass = (
        frozen.get("status") == "PASS_FROZEN_BOUNDARY_REGRESSION"
        and lower_boundary.get("status") == "PASS_LOWER_BOUNDARY_OPERATOR_PARITY"
    )
    baseline_pass = (
        historical_baseline.get("status") == "PASS_BASELINE_REPRODUCTION"
        and inherited_tests_pass
        and boundary_pass
        and current_cap4.get("status") == "PASS_CURRENT_VS_CAP4_REPRODUCTION"
    )
    reference_blocked = explicit.get("status") == "BLOCKED_EXACT_DONOR_BOUND_REFERENCE"
    if not boundary_pass:
        top = "FAIL_FROZEN_BOUNDARY_REGRESSION"
    else:
        # The task's permitted vocabulary has no explicit-reference-feasibility
        # member.  D is retained for an incomplete time-accuracy closure, with
        # a separate machine-readable assertion below that implicit inaccuracy
        # has not been proven.
        top = "FAIL_EULERIAN_TIME_ACCURACY"
    baseline_status = "PASS_BASELINE_REPRODUCTION" if baseline_pass else "FAIL_BASELINE_REPRODUCTION"
    current_status = current_cap4.get("current", {}).get("status", "NOT_RUN")
    cap4_status = current_cap4.get("cap4", {}).get("status", "NOT_RUN")
    return {
        "STATUS": top,
        "BRANCH": launch.get("git_branch_at_launch"),
        "COMMIT": launch.get("git_head_at_launch"),
        "BASELINE_REPRODUCED": baseline_status,
        "BOUNDARY_REGRESSION": frozen.get("status"),
        "LOWER_BOUNDARY_OPERATOR_PARITY": lower_boundary.get("status"),
        "KWN_TESTS": "52/52 PASS" if inherited_tests_pass else f"{tests.get('test_count')}/52 {tests.get('status')}",
        "EXPLICIT_SSPRK2_TESTS": "9/9 PASS" if explicit_tests_pass else f"{explicit_tests.get('test_count')}/9 {explicit_tests.get('status')}",
        "VALIDATION_CONTRACT_HASH": EXPECTED_CONTRACT_HASH,
        "CANONICAL_STATE_HASH": EXPECTED_CANONICAL_STATE_HASH,
        "CANONICAL_SOURCE_PSD_HASH": EXPECTED_CANONICAL_SOURCE_PSD_HASH,
        "COHORT_EXPLICIT_REFERENCE": "BLOCKED_EXACT_DONOR_BOUND_REFERENCE" if reference_blocked else "NOT_QUALIFIED",
        "EXPLICIT_REFERENCE_POLICY": "NONE_FULL_DOMAIN_DONOR_BOUND_BELOW_FROZEN_MIN_DT" if reference_blocked else "NOT_SELECTED",
        "EXPLICIT_SELF_CONVERGENCE": explicit.get("self_convergence"),
        "COHORT_EXPLICIT_MAX_ERROR": "NOT_EVALUATED",
        "REFERENCE_GATE_STATUS": explicit.get("status"),
        "IMPLICIT_TIME_INACCURACY_PROVEN": False,
        "CURRENT_IMPLICIT_POLICY": current_status,
        "CURRENT_VS_REFERENCE_01H": "NOT_EVALUATED_NO_TIME_REFERENCE",
        "CAP4_VS_REFERENCE_01H": "NOT_EVALUATED_NO_TIME_REFERENCE" if cap4_status != "NOT_RUN" else "NOT_RUN_PREREQUISITE_GATE",
        "CAP2_VS_REFERENCE_01H": "NOT_RUN_PREREQUISITE_GATE",
        "CAP1_VS_REFERENCE_01H": "NOT_RUN_PREREQUISITE_GATE",
        "CAP05_VS_REFERENCE_01H": "NOT_RUN_PREREQUISITE_GATE",
        "CAP025_VS_REFERENCE_01H": "NOT_RUN_PREREQUISITE_GATE",
        "PASSING_IMPLICIT_POLICY": "NOT_ASSIGNED",
        "PASSING_ACTIVE_CFL": "NOT_ASSIGNED",
        "01H_MAX_ERROR": "NOT_EVALUATED",
        "1H_MAX_ERROR": "NOT_EVALUATED",
        "ESTIMATED_48H_STEPS": "NOT_ASSIGNED",
        "ESTIMATED_48H_CPU_HOURS": "NOT_ASSIGNED",
        "FINAL_TIME_SCHEME": "NOT_SELECTED",
        "FINAL_ACCURACY_POLICY": "NOT_SELECTED",
        "EULERIAN_AUTHORITY_GRID_V2": "NOT_ASSIGNED",
        "EULERIAN_AUTHORITY_CONFIG_HASH": "NOT_ASSIGNED",
        "EULERIAN_RESTART": "PASS_EXPLICIT_SSPRK2_UNIT_RESTART" if explicit_tests_pass else "NOT_QUALIFIED",
        "EULERIAN_MAX_RESIDUAL": "NOT_EVALUATED_FINAL_AUTHORITY",
        "COHORT_EULERIAN_CROSSCHECK": "BLOCKED_PREREQUISITE_GATE",
        "N_M0_ERROR": "NOT_EVALUATED",
        "RMEAN_ERROR": "NOT_EVALUATED",
        "RMEAN3_ERROR": "NOT_EVALUATED",
        "SV_ERROR": "NOT_EVALUATED",
        "FBETA_ERROR": "NOT_EVALUATED",
        "XMATRIX_ERROR": "NOT_EVALUATED",
        "CUM_NUMBER_DISSOLUTION_ERROR": "NOT_EVALUATED",
        "CUM_MOL_B_ERROR": "NOT_EVALUATED",
        "BETA_INITIAL_STATE_IDENTITY": "BLOCKED_PREREQUISITE_GATE",
        "BETA_ONLY_DIRECTION": "BLOCKED_PREREQUISITE_GATE",
        "BETA_ONLY_TIMESCALE": "NOT_EVALUATED",
        "MEAN_FIELD_PF_GAP": "NOT_EVALUATED",
        "PF_SOURCE_MODIFIED": False,
        "CUDA_RERUN": False,
        "PHYSICAL_RETUNING": False,
        "LEGACY_SIX_PARTICLE_EULERIAN_P5": "FAIL_RETAINED",
        "HISTORICAL_AUTHORITY": "HISTORICAL_SMOOTH_3200_ONLY_V2_NOT_INHERITED",
        "TOP_5_FINDINGS": [
            f"The frozen 4db6fbc baseline status is {baseline_status}; the historical artifact is retained as provenance only.",
            f"The immutable physical-Rmin regression and direct shared-operator audit are {frozen.get('status')} / {lower_boundary.get('status')}.",
            f"Current versus cap=4 cumulative dissolution difference at 0.1 h is {current_cap4.get('difference', {}).get('cumulative_number_dissolution_m3_relative_difference', 'NOT_AVAILABLE')}.",
            "A strict full-domain SSPRK2 donor bound includes the positive canonical first cell and cannot be replaced by active-tail CFL.",
            "No implicit accuracy, cohort parity, Eulerian authority, beta-only PF direction, or GP-release decision is claimed without an admissible time reference.",
        ],
        "P0_BLOCKERS": [
            frozen.get("status") if not boundary_pass else explicit.get("status"),
            "No self-converged full-domain explicit SSPRK2 trajectory reaches 0.1 h under the frozen canonical measure and min_dt." if reference_blocked else "Explicit reference qualification did not reach a usable trajectory.",
            "Consequently TIME_REFERENCE_V1, the implicit ladder, longer-horizon checks, authority V2, and beta-only PF comparison are blocked by their declared prerequisites.",
        ],
        "NEXT_ACTION": "Authorize a separately gated conservative characteristic/semi-Lagrangian/IMEX reference design, or revise the canonical-measure/time-reference contract explicitly; do not use active-tail CFL as a donor-bound explicit proxy.",
        "LOCAL_GP_RELEASE_AUTHORIZED": "LOCAL_GP_RELEASE_NOT_AUTHORIZED",
        "TERMINAL_MAPPING_NOTE": "The requested final vocabulary has no explicit-reference-computational-blocked member.  STATUS remains the inherited D slot, but this run does not assert that the implicit scheme has been proven inaccurate.",
        "KEY_REPORTS": [
            str(DEFAULT_REPORT_ROOT / name)
            for name in (
                "00_baseline_reproduction.md",
                "02_explicit_ssprk2_qualification.md",
                "04_implicit_active_cfl_01h.md",
                "13_final_acceptance_report.md",
            )
        ],
    }


def _write_outputs(
    *,
    output_root: Path,
    launch: Mapping[str, Any],
    context: Any,
    historical_baseline: Mapping[str, Any],
    frozen: Mapping[str, Any],
    lower_boundary: Mapping[str, Any],
    tests: Mapping[str, Any],
    current_cap4: Mapping[str, Any],
    explicit_tests: Mapping[str, Any],
    explicit: Mapping[str, Any],
    final: Mapping[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "figures").mkdir(parents=True, exist_ok=True)
    _write_json(
        output_root / "baseline_reproduction.json",
        {
            "historical_artifact": historical_baseline,
            "inherited_52_tests": tests,
            "frozen_boundary_regression": frozen,
            "direct_lower_boundary_operator_parity": lower_boundary,
            "current_vs_cap4": current_cap4,
        },
    )
    _write_csv(output_root / "explicit_timestep_ladder.csv", explicit.get("rows", []), fallback_fields=("status", "reason"))
    _write_csv(output_root / "cohort_explicit_reference.csv", [_blocked_row(explicit.get("reason", "explicit reference unavailable"))], fallback_fields=("status", "reason"))
    _write_csv(output_root / "implicit_active_cfl_01h.csv", current_cap4.get("rows", []), fallback_fields=("status", "reason"))
    _write_csv(output_root / "implicit_active_cfl_1h.csv", [_blocked_row("TIME_REFERENCE_V1 unavailable")], fallback_fields=("status", "reason"))
    _write_csv(output_root / "cfl_time_traces.csv", current_cap4.get("cfl_rows", []), fallback_fields=("status", "reason"))
    _write_csv(output_root / "cost_model_48h.csv", [_blocked_row("no time-qualified finalist")], fallback_fields=("status", "reason"))
    _write_csv(output_root / "finalist_48h_trajectories.csv", [_blocked_row("short-time reference gate blocked")], fallback_fields=("status", "reason"))
    _write_json(output_root / "eulerian_authority_v2.json", {"status": "BLOCKED_PREREQUISITE_GATE", "reason": "final time scheme/policy not selected"})
    _write_csv(output_root / "final_cohort_eulerian_crosscheck.csv", [_blocked_row("EULERIAN_SMOOTH_AUTHORITY_V2 unavailable")], fallback_fields=("status", "reason"))
    _write_csv(output_root / "beta_only_cohort_pf_comparison.csv", [_blocked_row("PASS_COHORT_EULERIAN_SHARED_OPERATOR_PARITY unavailable")], fallback_fields=("status", "reason"))
    source_paths = (
        "scripts/run_kwn_time_accuracy_closure_v1.py",
        "scripts/run_kwn_lower_boundary_time_accuracy_v1.py",
        "scripts/run_kwn_radius_grid_convergence_v1.py",
        "src/kwn_mvp/explicit_solver.py",
        "src/kwn_mvp/solver.py",
        "src/kwn_mvp/lower_boundary.py",
        "src/kwn_mvp/cohort_solver.py",
        "tests/kwn/test_lower_boundary_contract_frozen.py",
        "tests/kwn/test_explicit_ssprk2_solver.py",
    )
    provenance = {
        "schema_version": "KWN_TIME_ACCURACY_CLOSURE_ANALYSIS_PROVENANCE_V1",
        "launch": launch,
        "canonical_state_hash": context.canonical_hash,
        "canonical_config_hash": context.solver.config.source_config_hash,
        "validation_contract_hash": context.contract_hash,
        "source_sha256": {path: _sha256_file(ROOT / path) for path in source_paths if (ROOT / path).is_file()},
        "historical_baseline_artifact": historical_baseline,
        "inherited_52_tests": tests,
        "frozen_boundary_regression": frozen,
        "direct_lower_boundary_operator_parity": lower_boundary,
        "explicit_ssprk2_tests": explicit_tests,
        "explicit_reference": explicit,
        "forbidden_actions": {"PF_SOURCE_MODIFIED": False, "CUDA_RERUN": False, "PHYSICAL_RETUNING": False, "LOCAL_GP_RELEASE": False},
        "final_status": final["STATUS"],
    }
    _write_json(output_root / "analysis_provenance.json", provenance)
    _write_json(output_root / "final_acceptance.json", final)
    (output_root / "figures" / "README.md").write_text(
        "# KWN time-accuracy closure evidence\n\nNo figure substitutes for the explicit-reference and parity gates; inspect the CSV and JSON evidence in this directory.\n",
        encoding="utf-8",
    )


def _write_reports(
    *,
    report_root: Path,
    output_root: Path,
    historical_baseline: Mapping[str, Any],
    frozen: Mapping[str, Any],
    lower_boundary: Mapping[str, Any],
    tests: Mapping[str, Any],
    current_cap4: Mapping[str, Any],
    explicit_tests: Mapping[str, Any],
    explicit: Mapping[str, Any],
    final: Mapping[str, Any],
    command: str,
) -> None:
    difference = current_cap4.get("difference", {}).get("cumulative_number_dissolution_m3_relative_difference", "NOT_AVAILABLE")
    bodies = {
        "00_baseline_reproduction.md": (
            f"Status: `{final['BASELINE_REPRODUCED']}`.  Inherited KWN suite: `{tests.get('status')}` with `{tests.get('test_count')}` tests.  "
            f"Frozen-boundary regression: `{frozen.get('status')}`; direct physical shared-operator audit: `{lower_boundary.get('status')}`.  "
            f"The actual 0--0.1 h current/cap=4 cumulative-number dissolution difference is `{difference}`.\n\n"
            f"The older lower-boundary trace is historical input only (`{historical_baseline.get('status')}`); this report records the actual frozen-start reproduction.\n\n"
            f"Structured evidence: `{output_root / 'baseline_reproduction.json'}`."
        ),
        "01_time_reference_definition.md": (
            "The required hierarchy is event-aware discrete cohort → full-domain explicit upwind donor-bound SSPRK2 → implicit upwind candidate.  "
            "No active-support approximation is permitted to replace the full-domain donor reference.\n\n"
            f"Status: `{explicit.get('status')}`."
        ),
        "02_explicit_ssprk2_qualification.md": (
            f"Status: `{explicit.get('status')}`.  Focused implementation tests: `{final['EXPLICIT_SSPRK2_TESTS']}`.\n\n{explicit.get('reason', '')}\n\n"
            f"Safety ladder: `{output_root / 'explicit_timestep_ladder.csv'}`.  The rejection is a numerical-reference feasibility fact, not a physical failure."
        ),
        "03_cohort_explicit_reference_parity.md": "Status: `BLOCKED_PREREQUISITE_GATE`.  A cohort comparison cannot use an unqualified explicit proxy.",
        "04_implicit_active_cfl_01h.md": (
            "Status: `BASELINE_ONLY_NOT_TIME_QUALIFIED`.  Current and cap=4 were reproduced solely to preserve the frozen starting evidence.  "
            "The formal implicit ladder is blocked until TIME_REFERENCE_V1 exists.\n\n"
            f"Rows: `{output_root / 'implicit_active_cfl_01h.csv'}`."
        ),
        "05_implicit_active_cfl_1h.md": "Status: `BLOCKED_PREREQUISITE_GATE`.  No provisional implicit policy has been selected.",
        "06_48h_cost_model.md": "Status: `BLOCKED_PREREQUISITE_GATE`.  A 48 h production-cost model cannot be selected before a time-qualified scheme exists.",
        "07_48h_finalists.md": "Status: `BLOCKED_PREREQUISITE_GATE`.  No time-qualified finalist exists.",
        "08_time_scheme_decision.md": "Status: `NOT_SELECTED`.  The current implicit policy is not declared accurate or inaccurate by this blocked reference gate.",
        "09_eulerian_authority_v2.md": "Status: `BLOCKED_PREREQUISITE_GATE`.  The 800/1600/3200 authority ladder is downstream of final time-scheme selection.",
        "10_final_cohort_eulerian_crosscheck.md": "Status: `BLOCKED_PREREQUISITE_GATE`.  The original 2% cross-representation gate was not rerun without a qualified Eulerian solver.",
        "11_beta_only_cohort_pf_comparison.md": "Status: `BLOCKED_PREREQUISITE_GATE`.  No PF/CUDA source was changed or rerun.",
        "12_model_role_boundary.md": "The work remains a mean-field KWN population calculation.  GP release, GP-to-beta conversion, online KWN--PF coupling, PF/CUDA reruns, physical retuning, and D_scale remain outside this run.",
        "13_final_acceptance_report.md": (
            f"Status: `{final['STATUS']}`; reference gate: `{final['REFERENCE_GATE_STATUS']}`; implicit inaccuracy proven: `{final['IMPLICIT_TIME_INACCURACY_PROVEN']}`.\n\nThe terminal vocabulary lacks an explicit-reference-computational-blocked category; the retained D slot does not mean the implicit scheme has been proven inaccurate.  "
            "The exact full-domain donor-bound reference is blocked by the positive canonical Rmin cell and frozen min_dt.\n\n"
            f"Structured final record: `{output_root / 'final_acceptance.json'}`."
        ),
        "14_reproduction_commands.md": (
            f"```bash\nPYTHONPATH=src {sys.executable} scripts/run_kwn_time_accuracy_closure_v1.py all --max-wall-s 300 --max-steps 100000\n```\n\n"
            f"Invocation recorded as: `{command}`."
        ),
    }
    for filename, title in REQUIRED_REPORTS.items():
        _write_report(report_root / filename, title, bodies[filename])


def _workflow(arguments: argparse.Namespace) -> dict[str, Any]:
    output_root = arguments.output_root.resolve()
    report_root = arguments.report_root.resolve()
    _require_formal_launch(output_root=output_root, report_root=report_root)
    launch = _launch_context()
    legacy = _load_legacy_runner()
    context = legacy._build_canonical_context()
    identity_checks = {
        "validation_contract_hash": context.contract_hash == EXPECTED_CONTRACT_HASH,
        "canonical_state_hash": context.canonical_hash == EXPECTED_CANONICAL_STATE_HASH,
        "source_initial_psd_hash": context.source_initial_psd_hash == EXPECTED_CANONICAL_SOURCE_PSD_HASH,
        "fixture_hash": context.fixture_hash == EXPECTED_FIXTURE_HASH,
    }
    if not all(identity_checks.values()):
        raise RuntimeError(f"canonical frozen-identity gate failed: {identity_checks}")

    historical_baseline = _baseline_artifact_reproduction(legacy)
    tests = _run_unittest_modules(BASELINE_TEST_MODULES)
    _run_manifest(
        output_root=output_root,
        name="inherited_52_kwn_tests",
        launch=launch,
        context=context,
        solver="UNIT_TEST_SUITE",
        policy="inherited_52_kwn_tests",
        config_mapping=context.mapping,
        execution={"modules": list(BASELINE_TEST_MODULES), "expected_test_count": 52},
        result=tests,
    )
    frozen_test = _run_unittest_modules(("tests.kwn.test_lower_boundary_contract_frozen",))
    frozen = {
        "status": "PASS_FROZEN_BOUNDARY_REGRESSION"
        if _test_gate(frozen_test, expected_count=2)
        else "FAIL_FROZEN_BOUNDARY_REGRESSION",
        "test": frozen_test,
    }
    _run_manifest(
        output_root=output_root,
        name="frozen_lower_boundary_regression",
        launch=launch,
        context=context,
        solver="UNIT_TEST_SUITE",
        policy="immutable_physical_rmin_boundary_regression",
        config_mapping=context.mapping,
        execution={"module": "tests.kwn.test_lower_boundary_contract_frozen", "expected_test_count": 2},
        result=frozen_test,
    )

    if frozen["status"] != "PASS_FROZEN_BOUNDARY_REGRESSION":
        lower_boundary = {
            "status": "BLOCKED_FROZEN_BOUNDARY_REGRESSION",
            "reason": "immutable physical-Rmin regression failed; numerical time-policy work stopped immediately",
        }
    elif not _test_gate(tests, expected_count=52):
        lower_boundary = {
            "status": "BLOCKED_INHERITED_KWN_TESTS",
            "reason": "inherited 52-test KWN gate did not reproduce",
        }
    else:
        direct_parity = legacy._physical_boundary_parity(context)
        lower_boundary = {
            **direct_parity,
            "status": (
                "PASS_LOWER_BOUNDARY_OPERATOR_PARITY"
                if direct_parity.get("status") == "PASS_PHYSICAL_LOWER_BOUNDARY_PARITY"
                else "FAIL_LOWER_BOUNDARY_OPERATOR_PARITY"
            ),
        }
        _run_manifest(
            output_root=output_root,
            name="direct_lower_boundary_operator_parity",
            launch=launch,
            context=context,
            solver="SHARED_PHYSICAL_RMIN_OPERATOR_AUDIT",
            policy="direct_helper_fv_cohort_parity",
            config_mapping=context.mapping,
            execution={"physical_rmin": True, "same_canonical_measure": True},
            result=lower_boundary,
        )

    baseline_prerequisite = (
        historical_baseline.get("status") == "PASS_BASELINE_REPRODUCTION"
        and _test_gate(tests, expected_count=52)
        and frozen["status"] == "PASS_FROZEN_BOUNDARY_REGRESSION"
        and lower_boundary.get("status") == "PASS_LOWER_BOUNDARY_OPERATOR_PARITY"
    )
    if baseline_prerequisite:
        current_cap4 = _run_current_vs_cap4(
            legacy=legacy,
            context=context,
            max_steps=arguments.max_steps,
            max_wall_s=arguments.max_wall_s,
            output_root=output_root,
            launch=launch,
        )
    else:
        current_cap4 = _not_run_current_cap4("baseline or frozen lower-boundary prerequisite failed")

    if current_cap4.get("status") == "PASS_CURRENT_VS_CAP4_REPRODUCTION":
        explicit_tests = _run_unittest_modules(("tests.kwn.test_explicit_ssprk2_solver",))
        _run_manifest(
            output_root=output_root,
            name="explicit_ssprk2_qualification_tests",
            launch=launch,
            context=context,
            solver="UNIT_TEST_SUITE",
            policy="explicit_ssprk2_qualification",
            config_mapping=context.mapping,
            execution={"module": "tests.kwn.test_explicit_ssprk2_solver", "expected_test_count": 9},
            result=explicit_tests,
        )
        explicit = (
            _explicit_preflight(context=context, output_root=output_root, launch=launch)
            if _test_gate(explicit_tests, expected_count=9)
            else _not_run_explicit("explicit SSPRK2 focused qualification tests failed")
        )
    else:
        explicit_tests = {"status": "NOT_RUN", "test_count": None}
        explicit = _not_run_explicit("current/cap=4 baseline reproduction did not pass")
    final = _build_final(
        launch=launch,
        historical_baseline=historical_baseline,
        frozen=frozen,
        lower_boundary=lower_boundary,
        tests=tests,
        current_cap4=current_cap4,
        explicit_tests=explicit_tests,
        explicit=explicit,
    )
    final["KEY_REPORTS"] = [
        str(report_root / name)
        for name in (
            "00_baseline_reproduction.md",
            "02_explicit_ssprk2_qualification.md",
            "04_implicit_active_cfl_01h.md",
            "13_final_acceptance_report.md",
        )
    ]
    _write_outputs(
        output_root=output_root,
        launch=launch,
        context=context,
        historical_baseline=historical_baseline,
        frozen=frozen,
        lower_boundary=lower_boundary,
        tests=tests,
        current_cap4=current_cap4,
        explicit_tests=explicit_tests,
        explicit=explicit,
        final=final,
    )
    _write_reports(
        report_root=report_root,
        output_root=output_root,
        historical_baseline=historical_baseline,
        frozen=frozen,
        lower_boundary=lower_boundary,
        tests=tests,
        current_cap4=current_cap4,
        explicit_tests=explicit_tests,
        explicit=explicit,
        final=final,
        command=" ".join(sys.argv),
    )
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all",), help="run the gate sequence through its first blocked or qualified stage")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--max-wall-s", type=float, default=300.0)
    arguments = parser.parse_args()
    if arguments.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if arguments.max_wall_s <= 0.0:
        parser.error("--max-wall-s must be positive")
    final = _workflow(arguments)
    print(json.dumps(_json_safe(final), indent=2, sort_keys=True))
    return 0 if str(final["STATUS"]).startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
