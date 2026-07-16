#!/usr/bin/env python3
"""Freeze the qualified BDF2 baseline and prepare matched T400 planar states.

This script is intentionally limited to provenance and initial-state
construction.  It does not run CUDA and it does not alter any physical or
solver parameter from the qualified BDF2 baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports" / "T400_longtime_v1"
EXAMPLE_MANIFEST = ROOT / "examples" / "T400_longtime_v1_manifest.json"

SOURCE_FILES = (
    "Makefile",
    "main_cuda.cu",
    "cuda_kernels.cu",
    "cuda_kernels.h",
    "pf_params.h",
    "thermo_utils.h",
    "phase_functions.h",
    "cuda_common.h",
    "phase_kkt_utils.h",
    "phase_pdas_reduction.h",
    "ctot_performance_profiler.h",
    "ctot_transport_bound_utils.h",
    "Unit_Psedobinary.py",
)

BASELINE_REPORTS = (
    "current_implementation_audit.md",
    "equation_and_sign_contract.md",
    "history_contract.md",
    "startup_restart_validation.md",
    "mass_storage_validation.md",
    "energy_work_contract.md",
    "step655_order_validation.md",
    "fixed_dt_qualification.md",
    "lie_vs_bdf2_efficiency.md",
    "final_terminal_output.txt",
)

PHYSICAL_KEYS = (
    "temperature_C", "dx", "dy", "dz", "lambda_sm_m", "gamma_Jm2",
    "W", "kappa_phi", "D_alpha", "D_compound", "D_beta_for_calibration",
    "L_phi", "L_phi_physical_value", "v_A", "v_B",
    "coarse_interface_mobility_a_M", "finite_interface_mode",
    "ctot_finite_interface_antitrapping_enabled", "elastic_enabled",
)

SOLVER_KEYS = (
    "dt", "ctot_numerics_contract", "ctot_split_defect_policy",
    "ctot_nonlinear_max_iter", "ctot_phase_linear_max_iter",
    "ctot_outer_max_iter", "ctot_step_max_retries",
    "ctot_automatic_dt_growth", "ctot_retry_shrink_factor",
    "ctot_dt_min_ratio", "ctot_transport_nonlinear_coordinate",
    "ctot_matrix_support_eps", "ctot_phase_semismooth_pdas_enabled",
    "ctot_preconditioner_a_ref", "ctot_outer_acceleration",
    "y_update_mass_projection_enabled", "mechanics_precision_mode",
    "mechanics_acceptance_mode", "eta_accept",
)

EXPECTED = {
    "model": "pbte_ag2te_gp_coarse4_stoich_rd_v2",
    "integrator": "FIXED_STEP_IMEX_BDF2_V1",
    "history": "FIXED_STEP_BDF2_ACCEPTED_HISTORY_V1",
    "phase_context": "PHI_EXTRAPOLATION_2N_MINUS_NM1_ULP64_V1",
    "energy": "BDF2_ENDPOINT_DISCRETE_WORK_V1",
    "dt": 0.000390625,
    "dt_physical_s": 0.016066244306466707,
    "dx_nm": 1.0,
    "lambda_nm": 4.0,
}

CASE_DEFINITIONS = {
    "growth": {
        "matrix_xB": 0.05,
        "initial_beta_half_width_nm": 16.0,
        "target_displacement_nm": 8.0,
        "sense": 1,
    },
    "dissolution": {
        "matrix_xB": 1.0e-4,
        "initial_beta_half_width_nm": 16.0,
        "target_displacement_nm": -8.0,
        "sense": -1,
    },
}

# At xB=1e-4, 512/1024 nm periodic boxes reach finite-inventory
# dissolution equilibrium before an 8 nm retreat.  Larger boxes are therefore
# required by the preregistered physical displacement, not by numerical tuning.
BOXES_BY_DIRECTION = {
    "growth": (512, 1024),
    "dissolution": (3072, 4096),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> str:
    completed = subprocess.run(
        command, text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {command!r}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout


def parse_params(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def parse_sha256(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            result[Path(parts[1]).name] = parts[0]
    return result


def h(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def alpha(phi: np.ndarray) -> np.ndarray:
    value = 1.0 - phi
    return value**3 * (6.0 * value**2 - 15.0 * value + 10.0)


def hp(phi: np.ndarray) -> np.ndarray:
    return 30.0 * phi**2 * (1.0 - phi) ** 2


def gp(phi: np.ndarray) -> np.ndarray:
    return 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)


def gpp(phi: np.ndarray) -> np.ndarray:
    return 2.0 - 12.0 * phi + 12.0 * phi**2


def hpp(phi: np.ndarray) -> np.ndarray:
    return 60.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)


def initial_slab(
    cells: int, dx_nm: float, half_width_nm: float, shift_dx: float,
) -> np.ndarray:
    coordinate = (np.arange(cells, dtype=np.float64) + 0.5) * dx_nm
    center = 0.5 * cells * dx_nm + shift_dx * dx_nm
    half_parameter = 0.5 * EXPECTED["lambda_nm"]
    return 0.5 * (
        np.tanh((coordinate - (center - half_width_nm)) / half_parameter)
        - np.tanh((coordinate - (center + half_width_nm)) / half_parameter)
    )


def refine_stationary_slab(
    phi0: np.ndarray,
    target_h_cells: float,
    dx_nm: float,
    w: float = 1.0,
    kappa: float = 2.0,
    tolerance: float = 3.0e-11,
    max_newton: int = 80,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Constrained 1-D spectral Newton solve for the zero-driving slab shape."""
    n = phi0.size
    wave = 2.0 * math.pi * np.fft.fftfreq(n, d=dx_nm)
    k2 = wave * wave

    def lap(field: np.ndarray) -> np.ndarray:
        return np.fft.ifft(-k2 * np.fft.fft(field)).real

    phi = np.clip(phi0.astype(np.float64, copy=True), 0.0, 1.0)
    representation_support_eps = 64.0 * np.finfo(np.float64).eps
    core = alpha(phi) <= representation_support_eps
    phi[core] = 1.0
    multiplier = 0.0
    history: list[dict[str, float | int]] = []
    for iteration in range(max_newton):
        free = ~core
        indices = np.flatnonzero(free)
        residual = w * gp(phi) + multiplier * hp(phi) - kappa * lap(phi)
        volume = float(np.sum(h(phi)) - target_h_cells)
        residual_linf = float(np.max(np.abs(residual[free])))
        if residual_linf <= tolerance and abs(volume) <= 1.0e-10:
            break
        local = w * gpp(phi) + multiplier * hpp(phi)
        hp_value = hp(phi)
        scale = 1.0 / math.sqrt(max(target_h_cells, 1.0))
        rhs = np.empty(indices.size + 1, dtype=np.float64)
        rhs[:-1] = -residual[indices]
        rhs[-1] = -scale * volume

        def matvec(vector: np.ndarray) -> np.ndarray:
            dphi = np.zeros(n, dtype=np.float64)
            dphi[indices] = vector[:-1]
            output = np.empty_like(vector)
            field = local * dphi - kappa * lap(dphi) + hp_value * vector[-1]
            output[:-1] = field[indices]
            output[-1] = scale * float(np.sum(hp_value * dphi))
            return output

        operator = LinearOperator(
            (indices.size + 1, indices.size + 1),
            matvec=matvec,
            dtype=np.float64,
        )
        shift = max(0.1, float(np.median(local[free])))
        denominator = shift + kappa * k2

        def precondition(vector: np.ndarray) -> np.ndarray:
            field = np.zeros(n, dtype=np.float64)
            field[indices] = vector[:-1]
            z = np.fft.ifft(np.fft.fft(field) / denominator).real
            w_field = np.fft.ifft(np.fft.fft(hp_value) / denominator).real
            z[core] = 0.0
            w_field[core] = 0.0
            schur = float(np.sum(hp_value * w_field))
            if not math.isfinite(schur) or abs(schur) < 1.0e-20:
                raise RuntimeError("stationary slab Schur complement is singular")
            lagrange = (float(np.sum(hp_value * z)) - vector[-1] / scale) / schur
            output = np.empty_like(vector)
            output[:-1] = (z - lagrange * w_field)[indices]
            output[-1] = lagrange
            return output

        preconditioner = LinearOperator(
            (indices.size + 1, indices.size + 1),
            matvec=precondition,
            dtype=np.float64,
        )
        kwargs = {
            "M": preconditioner,
            "atol": 1.0e-13,
            "restart": 80,
            "maxiter": 30,
        }
        try:
            step, info = gmres(operator, rhs, rtol=1.0e-11, **kwargs)
        except TypeError:
            step, info = gmres(operator, rhs, tol=1.0e-11, **kwargs)
        if info != 0 or not np.all(np.isfinite(step)):
            raise RuntimeError(f"stationary slab GMRES failed: info={info}")
        dphi = np.zeros(n, dtype=np.float64)
        dphi[indices] = step[:-1]
        merit0 = max(residual_linf, abs(volume) / max(target_h_cells, 1.0))
        factor = 1.0
        accepted = False
        for _ in range(40):
            trial = np.clip(phi + factor * dphi, 0.0, 1.0)
            trial[core] = 1.0
            trial_multiplier = multiplier + factor * float(step[-1])
            if float(np.min(trial)) < 0.0 or float(np.max(trial)) > 1.0:
                factor *= 0.5
                continue
            trial_residual = (
                w * gp(trial) + trial_multiplier * hp(trial) - kappa * lap(trial)
            )
            trial_volume = float(np.sum(h(trial)) - target_h_cells)
            merit = max(
                float(np.max(np.abs(trial_residual[free]))),
                abs(trial_volume) / max(target_h_cells, 1.0),
            )
            if merit < merit0:
                accepted = True
                break
            factor *= 0.5
        if not accepted:
            raise RuntimeError("stationary slab Newton line search failed")
        phi = trial
        multiplier = trial_multiplier
        core |= alpha(phi) <= representation_support_eps
        phi[core] = 1.0
        history.append({
            "iteration": iteration,
            "residual_linf_before": residual_linf,
            "volume_residual_before": volume,
            "line_factor": factor,
        })
    final_residual = w * gp(phi) + multiplier * hp(phi) - kappa * lap(phi)
    final_free = ~core
    diagnostics: dict[str, Any] = {
        "newton_iterations": len(history),
        "phase_residual_linf": float(np.max(np.abs(final_residual[final_free]))),
        "h_volume_residual_cell_units": float(np.sum(h(phi)) - target_h_cells),
        "volume_multiplier": multiplier,
        "representability_core_cells": int(np.count_nonzero(core)),
        "representation_support_eps": representation_support_eps,
        "phi_min": float(np.min(phi)),
        "phi_max": float(np.max(phi)),
        "history": history,
    }
    if diagnostics["phase_residual_linf"] > tolerance:
        raise RuntimeError(f"stationary slab did not converge: {diagnostics}")
    if abs(diagnostics["h_volume_residual_cell_units"]) > 1.0e-10:
        raise RuntimeError(f"stationary slab volume did not close: {diagnostics}")
    return phi, multiplier, diagnostics


def write_state(
    output: Path,
    direction: str,
    cells: int,
    shift_dx: float,
) -> dict[str, Any]:
    definition = CASE_DEFINITIONS[direction]
    dx_nm = EXPECTED["dx_nm"]
    target_h = 2.0 * float(definition["initial_beta_half_width_nm"]) / dx_nm
    raw = initial_slab(
        cells, dx_nm, float(definition["initial_beta_half_width_nm"]), shift_dx,
    )
    # Keep the host constructor independent of the production PDAS active set.
    # Runtime suitability is decided by a production-BDF2 zero-driving hold,
    # so a separate host obstacle solve cannot hide discrete profile relaxation.
    phi = raw
    wave = 2.0 * math.pi * np.fft.fftfreq(cells, d=dx_nm)
    lap = np.fft.ifft(-(wave * wave) * np.fft.fft(phi)).real
    zero_driving_residual = gp(phi) - 2.0 * lap
    multiplier = 0.0
    diagnostics: dict[str, Any] = {
        "construction": "analytic_continuum_stationary_profile",
        "runtime_hold_required": True,
        "phase_residual_linf": float(np.max(np.abs(zero_driving_residual))),
        "h_volume_residual_cell_units": float(np.sum(h(phi)) - target_h),
        "volume_multiplier": 0.0,
        "phi_min": float(np.min(phi)),
        "phi_max": float(np.max(phi)),
    }
    x_value = float(definition["matrix_xB"])
    x = np.full(cells, x_value, dtype=np.float64)
    C = h(phi) + alpha(phi) * x
    q = C - h(phi)
    shape = (cells, 1, 1)
    output.mkdir(parents=True, exist_ok=True)
    for name, line in (("phi", phi), ("xB", x), ("Ctot", C)):
        line.reshape(shape).astype(np.float64).tofile(output / f"{name}_init.raw")
    metadata: dict[str, Any] = {
        "schema": "T400_longtime_matched_planar_state_v1",
        "direction": direction,
        "Nx": cells, "Ny": 1, "Nz": 1,
        "dx_nm": dx_nm,
        "lambda_nm": EXPECTED["lambda_nm"],
        "lambda_over_dx": EXPECTED["lambda_nm"] / dx_nm,
        "dtype": "float64", "order": "C",
        "authoritative_state": "Ctot",
        "geometry": "periodic_planar_beta_slab_two_interfaces",
        "interface_center_shift_dx": shift_dx,
        "initial_beta_half_width_nm": definition["initial_beta_half_width_nm"],
        "matrix_xB": x_value,
        "h_volume_cell_units": float(np.sum(h(phi))),
        "total_C_cell_units": float(np.sum(C)),
        "mean_Ctot": float(np.mean(C)),
        "q_min": float(np.min(q)),
        "q_capacity_margin_min": float(np.min(alpha(phi) - q)),
        "C_minus_h_min": float(np.min(C - h(phi))),
        "one_minus_C_min": float(np.min(1.0 - C)),
        "finite": bool(np.all(np.isfinite(phi)) and np.all(np.isfinite(C))),
        "zero_driving_stationary_profile": diagnostics,
        "zero_driving_volume_multiplier": multiplier,
        "profile_role": (
            "analytic zero-driving stationary candidate with exact h-volume; "
            "production-runtime zero-driving hold required before use"
        ),
    }
    metadata["files"] = {
        name: {"path": f"{name}_init.raw", "sha256": sha256(output / f"{name}_init.raw")}
        for name in ("phi", "xB", "Ctot")
    }
    (output / "init_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def read_final_markers() -> dict[str, str]:
    path = ROOT / "reports" / "bdf2_v1" / "final_terminal_output.txt"
    return parse_params(path.read_text(encoding="utf-8"))


def remote_baseline(host: str, remote_repo: str) -> dict[str, Any]:
    quoted = shlex.quote(remote_repo)
    script = f"""
set -e
cd {quoted}
echo __SOURCE__
sha256sum main_cuda {' '.join(shlex.quote(name) for name in SOURCE_FILES)}
echo __ENV__
/usr/local/cuda-12.9/bin/nvcc --version | tail -1
g++ --version | head -1
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
echo __QUALIFIED_PARAMS__
find runs/bdf2_v1/fixed_dt_qualification/dt_div8/run -name pf_input.params -print -quit | xargs cat
echo __QUALIFIED_META__
find runs/bdf2_v1/fixed_dt_qualification/dt_div8/run -name ctot_checkpoint_step001000_meta.json -print -quit | xargs cat
echo __QUALIFIED_HASHES__
find runs/bdf2_v1/fixed_dt_qualification/dt_div8/run -type f -name 'ctot_checkpoint_step001000_*' -print0 | sort -z | xargs -0 sha256sum
echo __ORDER_SOURCE_META__
cat runs/frozen_input/ctot_checkpoint_step000054_meta.json
echo __ORDER_SOURCE_HASHES__
sha256sum runs/frozen_input/ctot_checkpoint_step000054*
"""
    output = run(["ssh", host, script])
    labels = (
        "SOURCE", "ENV", "QUALIFIED_PARAMS", "QUALIFIED_META",
        "QUALIFIED_HASHES", "ORDER_SOURCE_META", "ORDER_SOURCE_HASHES",
    )
    sections: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []
    for line in output.splitlines():
        if line.startswith("__") and line.endswith("__"):
            if current is not None:
                sections[current] = "\n".join(buffer).strip()
            current = line.strip("_")
            buffer = []
        else:
            buffer.append(line)
    if current is not None:
        sections[current] = "\n".join(buffer).strip()
    missing = set(labels) - set(sections)
    if missing:
        raise RuntimeError(f"remote baseline sections missing: {sorted(missing)}")

    source_lines = sections["SOURCE"].splitlines()
    binary_sha = source_lines[0].split()[0]
    remote_sources = parse_sha256("\n".join(source_lines[1:]))
    local_sources = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    if remote_sources != local_sources:
        raise RuntimeError("workstation and local source hashes differ")
    params = parse_params(sections["QUALIFIED_PARAMS"])
    qualified_meta = json.loads(sections["QUALIFIED_META"])
    order_meta = json.loads(sections["ORDER_SOURCE_META"])
    return {
        "host_alias": host,
        "binary_sha256": binary_sha,
        "source_hashes": local_sources,
        "environment_lines": sections["ENV"].splitlines(),
        "qualified_params": params,
        "qualified_checkpoint_meta": qualified_meta,
        "qualified_checkpoint_hashes": parse_sha256(sections["QUALIFIED_HASHES"]),
        "order_source_checkpoint_meta": order_meta,
        "order_source_checkpoint_hashes": parse_sha256(sections["ORDER_SOURCE_HASHES"]),
    }


def freeze_baseline(host: str, remote_repo: str) -> dict[str, Any]:
    markers = read_final_markers()
    if markers.get("final_status") != "PASS_FIXED_STEP_IMEX_BDF2_QUALIFIED":
        raise RuntimeError("qualified BDF2 final marker is missing")
    remote = remote_baseline(host, remote_repo)
    params = remote["qualified_params"]
    meta = remote["qualified_checkpoint_meta"]
    assertions = {
        "model": params.get("PF_RESEARCH_MODEL") == EXPECTED["model"],
        "numerics": params.get("ctot_numerics_contract") == "ctot_jichen_imex_bdf2_v1",
        "integrator": meta.get("time_integrator") == EXPECTED["integrator"],
        "history": meta.get("history_contract_version") == EXPECTED["history"],
        "phase_context": meta.get("phase_context_version") == EXPECTED["phase_context"],
        "energy": meta.get("BDF2_energy_contract_version") == EXPECTED["energy"],
        "dt": math.isclose(float(params.get("dt", "nan")), EXPECTED["dt"], rel_tol=0.0, abs_tol=1e-18),
        "dx": math.isclose(float(params.get("dx", "nan")), EXPECTED["dx_nm"], rel_tol=0.0, abs_tol=1e-15),
        "lambda": math.isclose(float(params.get("lambda_sm_m", "nan")) * 1e9, EXPECTED["lambda_nm"], rel_tol=0.0, abs_tol=1e-14),
        "retry_off": params.get("ctot_step_max_retries") == "0",
        "projection_off": params.get("y_update_mass_projection_enabled") == "0",
        "GP_off": all(params.get(key) == "0" for key in (
            "gp_growth_enabled", "gp_to_beta_enabled", "diagnostic_rsmd_enabled",
            "enable_legacy_gp_storage_coupling", "enable_gp_assisted_beta_nucleation",
        )),
    }
    if not all(assertions.values()):
        raise RuntimeError(f"baseline contract mismatch: {assertions}")
    report_hashes = {
        name: sha256(ROOT / "reports" / "bdf2_v1" / name)
        for name in BASELINE_REPORTS
    }
    baseline = {
        "schema": "T400_longtime_v1_baseline_manifest",
        "audit_date": "2026-07-16",
        "git_head": run(["git", "rev-parse", "HEAD"]).strip(),
        "git_branch": run(["git", "branch", "--show-current"]).strip(),
        "qualified_status": markers["final_status"],
        "assertions": assertions,
        "expected_contract": EXPECTED,
        "source_hashes": remote["source_hashes"],
        "binary_sha256": remote["binary_sha256"],
        "environment_lines": remote["environment_lines"],
        "baseline_report_hashes": report_hashes,
        "qualified_checkpoint_hashes": remote["qualified_checkpoint_hashes"],
        "qualified_checkpoint_meta": meta,
        "order_source_checkpoint_hashes": remote["order_source_checkpoint_hashes"],
        "order_source_checkpoint_meta": remote["order_source_checkpoint_meta"],
        "order_source_model_name_note": (
            "The common order-test source predates the v2 model-name metadata. "
            "Its physical/calibration hashes are retained as historical provenance; "
            "the current qualified step1000 checkpoint and runtime parameters carry v2."
        ),
        "physical_parameters": {key: params.get(key) for key in PHYSICAL_KEYS},
        "solver_parameters": {key: params.get(key) for key in SOLVER_KEYS},
        "history_contract": EXPECTED["history"],
        "energy_work_contract": EXPECTED["energy"],
        "mechanics_contract": {
            "precision": params.get("mechanics_precision_mode"),
            "acceptance": params.get("mechanics_acceptance_mode"),
            "eta_accept": params.get("eta_accept"),
        },
        "cluster_used": False,
        "commit_created": False,
        "push_performed": False,
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "baseline_manifest.json").write_text(
        json.dumps(baseline, indent=2) + "\n", encoding="utf-8",
    )
    return baseline


def build_test_manifest() -> dict[str, Any]:
    state_root = REPORT_ROOT / "initial_states"
    states: list[dict[str, Any]] = []
    for direction in CASE_DEFINITIONS:
        for cells in BOXES_BY_DIRECTION[direction]:
            for shift in (0.0, 0.5):
                case_id = f"T400_{direction}_N{cells}_shift{str(shift).replace('.', 'p')}"
                metadata = write_state(state_root / case_id, direction, cells, shift)
                states.append({"case_id": case_id, **metadata})
    manifest = {
        "schema": "T400_longtime_v1_test_manifest",
        "temperature_C": 400.0,
        "physics_model": EXPECTED["model"],
        "numerics": "ctot_jichen_imex_bdf2_v1",
        "selected_dt_code": EXPECTED["dt"],
        "selected_dt_physical_s": EXPECTED["dt_physical_s"],
        "dt_refinement_codes": [EXPECTED["dt"], EXPECTED["dt"] / 2.0, EXPECTED["dt"] / 4.0],
        "box_lengths_nm_by_direction": {
            key: [float(value) for value in values]
            for key, values in BOXES_BY_DIRECTION.items()
        },
        "subcell_shifts_dx": [0.0, 0.5],
        "minimum_displacement_nm": max(5.0 * EXPECTED["dx_nm"], 2.0 * EXPECTED["lambda_nm"]),
        "case_definitions": CASE_DEFINITIONS,
        "boundary_conditions": "periodic_planar_slab_two_interfaces",
        "transverse_grid": [1, 1],
        "references": {
            "direct_JiChen": "independent_host_double_solution_logit_reaction_diffusion",
            "fine_Ctot": {
                "lambda_nm": 0.30, "dx_nm": 0.025,
                "lambda_over_dx": 12.0, "Lphi_over_Ldiff": 0.90,
                "finite_interface_mode": "off",
            },
            "sharp": "finite_box_matrix_diffusion_plus_Stefan_and_finite_Lphi_comparator",
        },
        "states": states,
        "forbidden": [
            "T380", "GP", "S3", "curvature", "multi_particle", "cluster",
            "adaptive_dt", "variable_step_BDF2", "formal_3D_production",
        ],
    }
    EXAMPLE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLE_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def write_reports(baseline: dict[str, Any], manifest: dict[str, Any]) -> None:
    selected = baseline["solver_parameters"]
    REPORT_ROOT.joinpath("baseline_freeze.md").write_text(f"""# T400 Long-Time BDF2 Baseline Freeze

Status: `PASS_BASELINE_FREEZE`

The local candidate source and workstation source hashes are identical.  The
qualified workstation binary hash is `{baseline['binary_sha256']}`.  The
current runtime/checkpoint contract is `{EXPECTED['model']}` with
`{EXPECTED['integrator']}`, `{EXPECTED['history']}`, and
`{EXPECTED['energy']}`.

The selected fixed step is `{EXPECTED['dt']:.17e}` code time or
`{EXPECTED['dt_physical_s']:.17e} s`.  Automatic retry, clipping, physical
projection, GP/S3/source, elasticity, `a_M`, and finite-interface correction
remain off for the qualification baseline.

The order-test step54 checkpoint carries the historical v1 model-name string.
It is retained only as provenance for the already accepted order ladder.  Its
calibration/operator metadata are recorded separately; the current step1000
checkpoint and parameter file carry the v2 model identity.  No old name is
silently promoted to the current model.

Key nonlinear budgets are transport `{selected['ctot_nonlinear_max_iter']}`,
phase linear `{selected['ctot_phase_linear_max_iter']}`, and outer
`{selected['ctot_outer_max_iter']}`.  Full values and hashes are frozen in
`baseline_manifest.json`.
""", encoding="utf-8")

    rows = []
    for direction, definition in CASE_DEFINITIONS.items():
        rows.append(
            f"| {direction} | {definition['matrix_xB']:.8g} | "
            f"{definition['initial_beta_half_width_nm']:.1f} | "
            f"{definition['target_displacement_nm']:+.1f} |"
        )
    REPORT_ROOT.joinpath("test_definition.md").write_text(f"""# T400 Matched Long-Time Test Definition

The tests are periodic planar beta slabs with two equivalent interfaces,
`dx=1 nm` and `lambda=4 nm`.  The generated analytic zero-driving profile
matches the requested h-volume exactly but remains a candidate until it
passes a production-BDF2 zero-driving hold.  Growth/dissolution driving then
enters only through the declared uniform matrix composition; profile shape
is never fitted to a moving runtime trajectory.

| Direction | initial matrix xB | initial beta half-width (nm) | required per-interface displacement (nm) |
|---|---:|---:|---:|
{chr(10).join(rows)}

The growth primary and box-sensitivity lengths are 512 and 1024 nm.  At the
declared `xB=1e-4`, a source-free periodic dissolution slab reaches finite-box
equilibrium before an 8 nm retreat in either box, so the dissolution lengths
are 3072 and 4096 nm.  This follows from the total-inventory constraint and is
not numerical tuning.  Both centered and half-cell-shifted initial interfaces
are generated.  Every state closes
the exact `Ctot=h+(1-h)xB_alpha` storage relation in FP64 and records raw-file
hashes.  The minimum displacement is `{manifest['minimum_displacement_nm']:.1f} nm`.

The reference ensemble is independent direct Ji-Chen, fine fixed-Ctot
(`lambda=0.30 nm`, `dx=0.025 nm`, `lambda/dx=12`), and finite-box sharp
Stefan/mixed-control comparators.  Each reference must self-qualify before it
can contribute to a coarse4 fidelity tier.
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="workstation-tail")
    parser.add_argument(
        "--remote-repo",
        default="/home/zhiheng/PF/CUDA_STO_PF_lie_be_v2_20260716",
    )
    args = parser.parse_args()
    baseline = freeze_baseline(args.host, args.remote_repo)
    manifest = build_test_manifest()
    write_reports(baseline, manifest)
    print("T400_longtime_baseline_status=PASS_BASELINE_FREEZE")
    print(f"qualified_binary_sha256={baseline['binary_sha256']}")
    print(f"matched_initial_states={len(manifest['states'])}")
    print("cluster_used=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
