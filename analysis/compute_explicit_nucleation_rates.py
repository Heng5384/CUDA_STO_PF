#!/usr/bin/env python3
"""
Compute explicit stochastic nucleation rates for Ag2Te precipitation in a PbTe-rich
matrix using a constrained phase-field nucleus library.

This script does not re-estimate the nucleation barrier with a spherical CNT model.
Instead, it reads DeltaG_star and r_eff_star from constrained PF nucleus results
when those quantities are available in summary CSV/JSON/log files or can be
reconstructed from phi VTK fields with the same quintic h(phi) used by the CUDA
code:

    h(phi) = phi^3 * (6 phi^2 - 15 phi + 10)
    V_h    = integral h(phi) dV
    r_eff  = (3 V_h / 4 pi)^(1/3)

The rate structure retains the CNT-like prefactor form:

    J = N_site * Z_r * beta_r_star * exp(-DeltaG_star / (kB T)) * Theta_tr

For this Ag2Te-PbTe pseudo-binary system, the limiting species is the pseudo-binary
solute B = Ag2Te growth unit. The first version uses homogeneous matrix nucleation
with x_lim^0 = xB_loc, y_lim^e = 1, and a limiting diffusivity taken from either:

1. an Arrhenius Ag-in-PbTe diffusivity, consistent with Unit_Psedobinary.py, or
2. a CUDA-like physical diffusivity reconstructed from generated PF parameters.

The first version operates on case-averaged library entries rather than full spatial
VTK fields. When key quantities are missing or have ambiguous units, the script
marks the case incomplete and records warnings instead of silently inventing
defaults.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


K_B_J_PER_K = 1.380649e-23
N_A_PER_MOL = 6.02214076e23
R_GAS_J_PER_MOLK = 8.31446261815324
EV_TO_J = 1.602176634e-19
LOG10 = math.log(10.0)
FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?")
DEFAULT_BARRIER_SUMMARIES = (
    "barrier_summary.csv",
    "current_results_master_table_fitted.csv",
    "growth_summary.csv",
)
DEFAULT_TABLE_FIELDS = [
    "case_dir",
    "base_case_tag",
    "grouped_case_count",
    "T_input",
    "T_K",
    "xB_loc",
    "xB_eq",
    "xB_eq_source",
    "S_ratio",
    "mode",
    "exx",
    "eyy",
    "ezz",
    "exy",
    "DeltaG_star_J",
    "DeltaG_star_eV",
    "DeltaG_star_kBT",
    "DeltaG_star_source",
    "r_eff_star_m",
    "r_eff_star_nm",
    "r_eff_star_source",
    "Vm_alpha_m3_mol",
    "Vm_compound_m3_mol",
    "volume_sources",
    "Omega_site_m3",
    "Omega_g_m3",
    "D_B_m2_s",
    "D_B_source",
    "N_site_m3",
    "Z_r_1_m",
    "Z_r_source",
    "beta_r_star_1_s",
    "Theta_tr",
    "J_m3_s",
    "log10_J_m3_s",
    "DeltaV_nuc_m3",
    "dt_phys_s",
    "P_event",
    "log10_expected_events",
    "case_status",
    "barrier_metadata_source",
    "profile_source",
    "warnings",
]
DIAGNOSTIC_TABLE_FIELDS = [
    "case_dir",
    "base_case_tag",
    "grouped_case_count",
    "T_input",
    "T_K",
    "xB_loc",
    "xB_eq",
    "xB_eq_source",
    "S_ratio",
    "mode",
    "exx",
    "eyy",
    "ezz",
    "exy",
    "DeltaG_star_J",
    "DeltaG_star_eV",
    "DeltaG_star_kBT",
    "DeltaG_star_source",
    "DeltaG_star_current",
    "barrier_kBT_current",
    "DeltaG_star_corrected",
    "barrier_kBT_corrected",
    "r_eff_star_m",
    "r_eff_star_nm",
    "r_eff_star_source",
    "Vm_alpha_m3_mol",
    "Vm_compound_m3_mol",
    "volume_sources",
    "Omega_site_m3",
    "Omega_g_m3",
    "D_B_m2_s",
    "D_B_source",
    "N_site_m3",
    "beta_r_star_1_s",
    "Theta_tr",
    "strict_status",
    "strict_incomplete_reason",
    "Z_r_strict",
    "Z_r_strict_source",
    "J_strict",
    "log10J_strict",
    "Z_r_diagnostic",
    "Z_r_diagnostic_source",
    "Z_r_diagnostic_fit_R2",
    "Z_r_diagnostic_window_points",
    "Z_r_diagnostic_curvature_sign",
    "J_diagnostic",
    "log10J_diagnostic",
    "P_event_diagnostic",
    "log10_expected_events_diagnostic",
    "J_diagnostic_current",
    "log10J_diagnostic_current",
    "P_event_diagnostic_current",
    "log10_expected_events_diagnostic_current",
    "J_diagnostic_corrected",
    "log10J_diagnostic_corrected",
    "P_event_diagnostic_corrected",
    "log10_expected_events_diagnostic_corrected",
    "ln_prefactor",
    "barrier_over_kBT",
    "lnJ",
    "prefactor_log10",
    "barrier_penalty_log10",
    "barrier_over_kBT_corrected",
    "lnJ_corrected",
    "barrier_penalty_log10_corrected",
    "same_strain_reference_found",
    "same_strain_reference_source",
    "same_strain_reference_confidence",
    "background_fraction_of_current_barrier",
    "case_status",
    "barrier_metadata_source",
    "profile_source",
    "warnings",
]
AUDIT_FIELDS = [
    "case_dir",
    "T_input",
    "T_K",
    "xB0",
    "mode",
    "strain_values",
    "F_CNT_peak_hat",
    "F_total_CNT_hat_if_available",
    "barrier_J_current",
    "barrier_eV_current",
    "barrier_kBT_current",
    "r_eff_star_nm",
    "voxel_count",
    "spacing_sim_units",
    "dx_nm_or_spacing_physical",
    "volume_from_voxels_m3",
    "mu_reference_if_used",
    "c_tot_if_used",
    "Vm_alpha",
    "Vm_compound",
    "conversion_formula_used",
    "all_conversion_factors",
    "warnings",
]
REFERENCE_AUDIT_FIELDS = [
    "case_dir",
    "T_input",
    "T_K",
    "xB0",
    "mode",
    "exx",
    "eyy",
    "ezz",
    "exy",
    "Nx",
    "Ny",
    "Nz",
    "dx_nm",
    "V_box_m3",
    "w_phys_J_m3",
    "F_CNT_peak_hat_current",
    "F_CNT_peak_definition_inferred",
    "F_saddle_hat",
    "F_reference_hat_same_strain",
    "F_reference_source",
    "F_reference_hat_unstrained_if_available",
    "DeltaF_hat_current",
    "DeltaF_hat_corrected_same_strain",
    "DeltaF_hat_using_unstrained_reference_if_available",
    "barrier_J_current",
    "barrier_kBT_current",
    "barrier_eV_current",
    "barrier_J_corrected_same_strain",
    "barrier_kBT_corrected_same_strain",
    "barrier_eV_corrected_same_strain",
    "barrier_J_unstrained_reference_if_available",
    "barrier_kBT_unstrained_reference_if_available",
    "background_elastic_hat_same_strain",
    "background_elastic_J_same_strain",
    "background_fraction_of_current_barrier",
    "recommended_barrier",
    "recommended_barrier_kBT",
    "recommended_reference",
    "confidence",
    "warnings",
]
REFERENCE_COMPARISON_FIELDS = [
    "case_dir",
    "base_case_tag",
    "current_barrier_kBT",
    "corrected_same_strain_barrier_kBT",
    "difference_kBT",
    "ratio_current_to_corrected",
    "background_fraction",
    "status",
]
MODE_CANDIDATES = (
    "biaxial_xy",
    "exx_eyy",
    "shear_xy",
    "shear_xz",
    "shear_yz",
    "no_strain",
    "exx",
    "eyy",
    "ezz",
)


@dataclass
class ProfilePoint:
    case_dir: str
    nominal_radius_nm: float | None = None
    r_eff_m: float | None = None
    energy_j: float | None = None
    energy_source: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class CaseData:
    key: str
    base_case_tag: str
    representative_dir: Path
    case_dirs: list[Path]
    params_path: Path | None
    params: dict[str, Any]
    physical_inputs_path: Path | None
    physical_inputs: dict[str, Any]
    parsed_name: dict[str, Any]
    local_metadata_rows: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    global_metadata_rows: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class BarrierData:
    barrier_j: float | None
    source: str
    warnings: list[str] = field(default_factory=list)
    source_field: str = ""
    source_unit: str = ""
    raw_value: float | None = None
    row_source: str = ""
    conversion_formula_used: str = ""
    all_conversion_factors: dict[str, Any] = field(default_factory=dict)
    f_cnt_peak_hat: float | None = None
    f_total_cnt_hat_if_available: float | None = None
    voxel_count: float | None = None
    spacing_sim_units: str = ""
    dx_nm_or_spacing_physical: str = ""
    volume_from_voxels_m3: float | None = None
    mu_reference_if_used: float | None = None
    c_tot_if_used: float | None = None


@dataclass
class ZeldovichResult:
    value: float | None
    source: str
    warnings: list[str] = field(default_factory=list)
    profile_source: str = ""
    fit_r2: float | None = None
    window_points: int | None = None
    curvature_sign: str = ""
    curvature_j_per_m2: float | None = None


@dataclass
class BarrierAuditRecord:
    case_dir: str
    T_input: float | None
    T_K: float | None
    xB0: float | None
    mode: str
    strain_values: str
    F_CNT_peak_hat: float | None
    F_total_CNT_hat_if_available: float | None
    barrier_J_current: float | None
    barrier_eV_current: float | None
    barrier_kBT_current: float | None
    r_eff_star_nm: float | None
    voxel_count: float | None
    spacing_sim_units: str
    dx_nm_or_spacing_physical: str
    volume_from_voxels_m3: float | None
    mu_reference_if_used: float | None
    c_tot_if_used: float | None
    Vm_alpha: float | None
    Vm_compound: float | None
    conversion_formula_used: str
    all_conversion_factors: str
    warnings: str

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReferenceAuditRecord:
    case_dir: str
    T_input: float | None
    T_K: float | None
    xB0: float | None
    mode: str
    exx: float | None
    eyy: float | None
    ezz: float | None
    exy: float | None
    Nx: int | None
    Ny: int | None
    Nz: int | None
    dx_nm: float | None
    V_box_m3: float | None
    w_phys_J_m3: float | None
    F_CNT_peak_hat_current: float | None
    F_CNT_peak_definition_inferred: str
    F_saddle_hat: float | None
    F_reference_hat_same_strain: float | None
    F_reference_source: str
    F_reference_hat_unstrained_if_available: float | None
    DeltaF_hat_current: float | None
    DeltaF_hat_corrected_same_strain: float | None
    DeltaF_hat_using_unstrained_reference_if_available: float | None
    barrier_J_current: float | None
    barrier_kBT_current: float | None
    barrier_eV_current: float | None
    barrier_J_corrected_same_strain: float | None
    barrier_kBT_corrected_same_strain: float | None
    barrier_eV_corrected_same_strain: float | None
    barrier_J_unstrained_reference_if_available: float | None
    barrier_kBT_unstrained_reference_if_available: float | None
    background_elastic_hat_same_strain: float | None
    background_elastic_J_same_strain: float | None
    background_fraction_of_current_barrier: float | None
    recommended_barrier: float | None
    recommended_barrier_kBT: float | None
    recommended_reference: str
    confidence: str
    warnings: str

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReferenceComparisonRecord:
    case_dir: str
    base_case_tag: str
    current_barrier_kBT: float | None
    corrected_same_strain_barrier_kBT: float | None
    difference_kBT: float | None
    ratio_current_to_corrected: float | None
    background_fraction: float | None
    status: str

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SiblingEnergyPoint:
    case_dir: str
    nominal_radius_nm: float | None
    voxel_count: float | None
    mean_h: float | None
    F_surf_hat: float | None
    F_el_hat: float | None
    F_total_excess_hat: float | None
    F_chem_CNT_hat: float | None
    F_total_CNT_hat: float | None
    r_eff_m: float | None
    energy_csv_path: str
    is_reference_like: bool = False


@dataclass
class RateResult:
    case_dir: str
    base_case_tag: str
    grouped_case_count: int
    T_input: float | None
    T_K: float | None
    xB_loc: float | None
    xB_eq: float | None
    xB_eq_source: str
    S_ratio: float | None
    mode: str
    exx: float | None
    eyy: float | None
    ezz: float | None
    exy: float | None
    DeltaG_star_J: float | None
    DeltaG_star_eV: float | None
    DeltaG_star_kBT: float | None
    DeltaG_star_source: str
    r_eff_star_m: float | None
    r_eff_star_nm: float | None
    r_eff_star_source: str
    Vm_alpha_m3_mol: float | None
    Vm_compound_m3_mol: float | None
    volume_sources: str
    Omega_site_m3: float | None
    Omega_g_m3: float | None
    D_B_m2_s: float | None
    D_B_source: str
    N_site_m3: float | None
    Z_r_1_m: float | None
    Z_r_source: str
    beta_r_star_1_s: float | None
    Theta_tr: float | None
    J_m3_s: float | None
    log10_J_m3_s: float | None
    DeltaV_nuc_m3: float | None
    dt_phys_s: float | None
    P_event: float | None
    case_status: str
    barrier_metadata_source: str = ""
    profile_source: str = ""
    log10_expected_events: float | None = None
    strict_status: str = "incomplete"
    strict_incomplete_reason: str = ""
    Z_r_strict: float | None = None
    Z_r_strict_source: str = ""
    J_strict: float | None = None
    log10J_strict: float | None = None
    Z_r_diagnostic: float | None = None
    Z_r_diagnostic_source: str = ""
    Z_r_diagnostic_fit_R2: float | None = None
    Z_r_diagnostic_window_points: int | None = None
    Z_r_diagnostic_curvature_sign: str = ""
    J_diagnostic: float | None = None
    log10J_diagnostic: float | None = None
    P_event_diagnostic: float | None = None
    log10_expected_events_diagnostic: float | None = None
    J_diagnostic_current: float | None = None
    log10J_diagnostic_current: float | None = None
    P_event_diagnostic_current: float | None = None
    log10_expected_events_diagnostic_current: float | None = None
    DeltaG_star_current: float | None = None
    barrier_kBT_current: float | None = None
    DeltaG_star_corrected: float | None = None
    barrier_kBT_corrected: float | None = None
    J_diagnostic_corrected: float | None = None
    log10J_diagnostic_corrected: float | None = None
    P_event_diagnostic_corrected: float | None = None
    log10_expected_events_diagnostic_corrected: float | None = None
    ln_prefactor: float | None = None
    barrier_over_kBT: float | None = None
    lnJ: float | None = None
    prefactor_log10: float | None = None
    barrier_penalty_log10: float | None = None
    barrier_over_kBT_corrected: float | None = None
    lnJ_corrected: float | None = None
    barrier_penalty_log10_corrected: float | None = None
    same_strain_reference_found: int = 0
    same_strain_reference_source: str = ""
    same_strain_reference_confidence: str = ""
    background_fraction_of_current_barrier: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["warnings"] = " | ".join(sorted(dict.fromkeys(self.warnings)))
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute explicit nucleation rates from constrained PF nucleus results."
    )
    parser.add_argument("--root", required=True, help="root directory to recursively scan")
    parser.add_argument("--pattern", default="cntcon_*", help="case directory glob pattern")
    parser.add_argument(
        "--out",
        default=None,
        help="output directory; default is <workflow-root>/analysis/nucleation_rate when --root is a workflow",
    )
    parser.add_argument(
        "--barrier-summary",
        default=",".join(DEFAULT_BARRIER_SUMMARIES),
        help="comma-separated preferred barrier summary filenames",
    )
    parser.add_argument("--profile-glob", default="*profile*.csv", help="profile csv glob")
    parser.add_argument("--params-glob", default="pf_input.params", help="params file glob")
    parser.add_argument("--log-glob", default="*.log", help="log file glob")
    parser.add_argument(
        "--diffusivity-mode",
        choices=("arrhenius_ag", "cuda_like"),
        default="arrhenius_ag",
        help="diffusivity model for matrix-side attachment kinetics",
    )
    parser.add_argument(
        "--theta",
        choices=("steady", "transient"),
        default="steady",
        help="steady-state or transient prefactor",
    )
    parser.add_argument("--tau-inc", type=float, default=None, help="incubation time constant in s")
    parser.add_argument("--t-eval", type=float, default=None, help="evaluation time in s")
    parser.add_argument("--t-act", type=float, default=0.0, help="activation time in s")
    parser.add_argument("--delta-v-nuc", type=float, default=None, help="Poisson insertion volume in m^3")
    parser.add_argument("--dt-phys", type=float, default=None, help="physical time step in s")
    parser.add_argument(
        "--allow-zeldovich-fallback",
        action="store_true",
        help="allow a capillary-scale fallback when no usable profile curvature exists",
    )
    parser.add_argument(
        "--zeldovich-fallback-value",
        type=float,
        default=None,
        help="optional constant fallback Z_r in 1/m",
    )
    parser.add_argument(
        "--fit-points",
        type=int,
        default=5,
        help="number of nearest profile points to use for the quadratic Zeldovich fit",
    )
    parser.add_argument(
        "--audit-reference-energy",
        action="store_true",
        help="emit same-strain reference-energy audit and corrected barrier diagnostics",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="skip matplotlib plot generation",
    )
    return parser.parse_args()


def infer_default_output_dir(root: Path) -> Path:
    if (root / "input").is_dir() and (root / "cnt_scan").is_dir():
        return root / "analysis" / "nucleation_rate"
    return Path.cwd() / "nucleation_rate_outputs"


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "n/a"}:
        return None
    match = FLOAT_RE.search(text.replace("D", "E").replace("d", "e"))
    if not match:
        return None
    try:
        out = float(match.group(0))
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")


def flatten_mapping(obj: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_mapping(value, next_prefix))
    elif isinstance(obj, list):
        if obj and all(not isinstance(x, (dict, list)) for x in obj):
            flat[prefix] = ",".join(str(x) for x in obj)
        else:
            for idx, value in enumerate(obj):
                next_prefix = f"{prefix}[{idx}]"
                flat.update(flatten_mapping(value, next_prefix))
    else:
        flat[prefix] = obj
    return flat


def parse_text_kv(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return data
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        parsed = safe_float(value)
        data[key] = parsed if parsed is not None else value
    return data


def parse_params_file(path: Path) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if not path.exists():
        return params
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        numeric = safe_float(value)
        params[key] = numeric if numeric is not None else value
    return params


def decode_tag_number(token: str) -> float | None:
    raw = token.strip().lower()
    if not raw:
        return None
    sign = 1.0
    if raw.startswith("sm"):
        sign = -1.0
        raw = raw[2:]
    elif raw.startswith("s"):
        raw = raw[1:]
    elif raw.startswith("m"):
        sign = -1.0
        raw = raw[1:]
    raw = raw.replace("p", ".")
    try:
        return sign * float(raw)
    except ValueError:
        return None


def parse_case_name(name: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "case_name": name,
        "base_case_tag": name,
        "nominal_radius_nm": None,
        "T_input": None,
        "xB_loc": None,
        "mode": "unknown",
        "strain_tag": None,
        "strain_value": None,
        "is_reference_case": False,
    }
    if name.endswith("_ref_matrix_only"):
        parsed["base_case_tag"] = name[: -len("_ref_matrix_only")]
        parsed["is_reference_case"] = True
    radius_match = re.search(r"_r([0-9mp]+)$", name)
    if radius_match:
        parsed["nominal_radius_nm"] = decode_tag_number(radius_match.group(1))
        parsed["base_case_tag"] = name[: radius_match.start()]
    temp_match = re.search(r"(?:^|_)T([0-9mp]+)(?:_|$)", name)
    if temp_match:
        parsed["T_input"] = decode_tag_number(temp_match.group(1))
    xb_match = re.search(r"(?:^|_)(?:xB|x0_)([0-9mp]+)(?:_|$)", name)
    if xb_match:
        parsed["xB_loc"] = decode_tag_number(xb_match.group(1))
    strain_match = re.search(r"(?:^|_)(s(?:m)?[0-9p]+)(?:_|$)", name)
    if strain_match:
        parsed["strain_tag"] = strain_match.group(1)
        parsed["strain_value"] = decode_tag_number(strain_match.group(1))
    for mode in MODE_CANDIDATES:
        if re.search(rf"(?:^|_){re.escape(mode)}(?:_|$)", name):
            parsed["mode"] = mode
            break
    return parsed


def find_nearest_case_param_file(case_dir: Path, glob_pattern: str) -> Path | None:
    preferred = case_dir / glob_pattern
    if preferred.exists():
        return preferred
    matches = sorted(case_dir.glob(glob_pattern))
    return matches[0] if matches else None


def find_nearest_physical_inputs(case_dir: Path, root: Path) -> tuple[Path | None, dict[str, Any]]:
    checked: list[Path] = []
    for parent in [case_dir, *case_dir.parents]:
        if parent == root.parent:
            break
        for candidate in (
            parent / "input" / "physical_inputs.json",
            parent / "physical_inputs.json",
        ):
            if candidate in checked:
                continue
            checked.append(candidate)
            if candidate.exists():
                try:
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                    payload = {k: v for k, v in payload.items() if not str(k).startswith("_")}
                    return candidate, payload
                except (OSError, json.JSONDecodeError):
                    continue
    return None, {}


def discover_global_rows(root: Path, preferred_names: Iterable[str]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    seen: set[Path] = set()
    patterns = set(name.strip() for name in preferred_names if name.strip())
    patterns.update({"current_results_master_table_fitted.csv", "growth_summary.csv"})
    for pattern in patterns:
        for path in root.rglob(pattern):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            rows.extend(load_metadata_rows(path))
    return rows


def discover_profile_files(root: Path, pattern: str) -> list[Path]:
    files = [path for path in root.rglob(pattern) if path.is_file()]
    known = list(root.rglob("current_results_master_table_fitted.csv"))
    for path in known:
        if path not in files:
            files.append(path)
    return sorted(files)


def load_metadata_rows(path: Path) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            with path.open(newline="", encoding="utf-8", errors="replace") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    rows.append((str(path), row))
        elif suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        rows.append((str(path), flatten_mapping(item)))
            elif isinstance(payload, dict):
                rows.append((str(path), flatten_mapping(payload)))
        elif suffix in {".txt", ".log"}:
            rows.append((str(path), parse_text_kv(path)))
    except (OSError, json.JSONDecodeError, csv.Error):
        return rows
    return rows


def score_row_match(case: CaseData, row: dict[str, Any]) -> int:
    score = 0
    row_norm = {normalize_key(k): v for k, v in row.items()}
    base = case.base_case_tag.lower()
    case_names = {path.name.lower() for path in case.case_dirs}
    for key in ("base_case_tag", "case_tag", "case_name"):
        value = row_norm.get(key)
        if value is None:
            continue
        text = str(value).lower()
        if text == base:
            score += 100
        if text in case_names:
            score += 80
    for key in ("case_dir", "case_dir_rel", "summary_path", "phi_vtk_file", "energy_csv_path"):
        value = row_norm.get(key)
        if value and base in str(value).lower():
            score += 60
    mode = infer_mode(case)
    row_mode = row_norm.get("mode")
    if row_mode and str(row_mode).lower() == mode.lower():
        score += 15
    t_case = infer_temperature_c(case)
    t_row = safe_float(row_norm.get("t_c") or row_norm.get("temperature_c") or row_norm.get("t_input"))
    if t_case is not None and t_row is not None and abs(t_case - t_row) < 1.0e-9:
        score += 10
    xb_case = infer_xb_loc(case)
    xb_row = safe_float(row_norm.get("xb_out") or row_norm.get("xb_loc") or row_norm.get("x_b_out"))
    if xb_case is not None and xb_row is not None and abs(xb_case - xb_row) < 1.0e-12:
        score += 10
    strain_case = infer_primary_strain(case)
    strain_row = safe_float(row_norm.get("strain"))
    if strain_case is not None and strain_row is not None and abs(strain_case - strain_row) < 1.0e-12:
        score += 10
    return score


def scan_case_dirs(
    root: Path,
    pattern: str,
    params_glob: str,
    log_glob: str,
    global_rows: list[tuple[str, dict[str, Any]]],
) -> list[CaseData]:
    grouped: dict[str, list[Path]] = {}
    for path in sorted(root.rglob(pattern)):
        if not path.is_dir():
            continue
        if not any(path.glob(params_glob)) and not (path / "summary.txt").exists() and not any(path.glob("energy_minimize_*.csv")):
            continue
        parsed = parse_case_name(path.name)
        key = parsed["base_case_tag"]
        grouped.setdefault(key, []).append(path)

    cases: list[CaseData] = []
    for key, case_dirs in sorted(grouped.items()):
        representative_dir = sorted(case_dirs)[0]
        params_path = find_nearest_case_param_file(representative_dir, params_glob)
        params = parse_params_file(params_path) if params_path else {}
        phys_path, physical_inputs = find_nearest_physical_inputs(representative_dir, root)
        parsed = parse_case_name(representative_dir.name)
        local_rows: list[tuple[str, dict[str, Any]]] = []
        local_files: list[Path] = []
        for case_dir in case_dirs:
            summary_file = case_dir / "summary.txt"
            if summary_file.exists():
                local_files.append(summary_file)
            local_files.extend(sorted(case_dir.glob(log_glob)))
            local_files.extend(sorted(case_dir.glob("*.json")))
            local_files.extend(sorted(case_dir.glob("reference_energy.csv")))
            local_files.extend(sorted(case_dir.glob("*profile*.csv")))
            local_files.extend(sorted(case_dir.glob("*barrier*.csv")))
        seen_local: set[Path] = set()
        for file_path in local_files:
            if file_path in seen_local or file_path.name == "pf_input.params":
                continue
            seen_local.add(file_path)
            local_rows.extend(load_metadata_rows(file_path))
        global_matches = [(source, row) for source, row in global_rows if score_row_match(
            CaseData(
                key=key,
                base_case_tag=key,
                representative_dir=representative_dir,
                case_dirs=case_dirs,
                params_path=params_path,
                params=params,
                physical_inputs_path=phys_path,
                physical_inputs=physical_inputs,
                parsed_name=parsed,
            ),
            row,
        ) >= 20]
        cases.append(
            CaseData(
                key=key,
                base_case_tag=key,
                representative_dir=representative_dir,
                case_dirs=sorted(case_dirs),
                params_path=params_path,
                params=params,
                physical_inputs_path=phys_path,
                physical_inputs=physical_inputs,
                parsed_name=parsed,
                local_metadata_rows=local_rows,
                global_metadata_rows=global_matches,
            )
        )
    return cases


def infer_temperature_c(case: CaseData) -> float | None:
    for key in ("temperature_C", "T_C"):
        value = safe_float(case.params.get(key))
        if value is not None:
            return value
    value = safe_float(case.physical_inputs.get("temperature_C"))
    if value is not None:
        return value
    return case.parsed_name.get("T_input")


def infer_temperature_k(case: CaseData) -> float | None:
    temp_c = infer_temperature_c(case)
    if temp_c is None:
        return None
    temp_k = temp_c + 273.15
    return temp_k if temp_k > 0.0 else None


def infer_xb_loc(case: CaseData) -> float | None:
    candidates = (
        "ic_23d_xB_out",
        "ic_xB_out",
        "xB_out",
        "xB_loc",
    )
    for key in candidates:
        value = safe_float(case.params.get(key))
        if value is not None and value > 0.0:
            return value
    value = safe_float(case.physical_inputs.get("xB_out"))
    if value is not None:
        return value
    value = case.parsed_name.get("xB_loc")
    return value


def infer_primary_strain(case: CaseData) -> float | None:
    return case.parsed_name.get("strain_value")


def infer_mode(case: CaseData) -> str:
    mode = case.parsed_name.get("mode")
    if mode and mode != "unknown":
        return mode
    exx, eyy, ezz, exy = infer_strain_components(case)
    exz = safe_float(case.params.get("E0_xz"))
    eyz = safe_float(case.params.get("E0_yz"))
    components = [x if x is not None else 0.0 for x in (exx, eyy, ezz, exy, exz, eyz)]
    if max(abs(x) for x in components) < 1.0e-15:
        return "no_strain"
    if exy and abs(exy) > 1.0e-15:
        return "shear_xy"
    if exz and abs(exz) > 1.0e-15:
        return "shear_xz"
    if eyz and abs(eyz) > 1.0e-15:
        return "shear_yz"
    if exx and eyy and abs(exx - eyy) < 1.0e-15 and abs(exx) > 1.0e-15:
        return "exx_eyy"
    if exx and abs(exx) > 1.0e-15:
        return "exx"
    if eyy and abs(eyy) > 1.0e-15:
        return "eyy"
    if ezz and abs(ezz) > 1.0e-15:
        return "ezz"
    return "unknown"


def infer_strain_components(case: CaseData) -> tuple[float | None, float | None, float | None, float | None]:
    exx = safe_float(case.params.get("E0_xx"))
    eyy = safe_float(case.params.get("E0_yy"))
    ezz = safe_float(case.params.get("E0_zz"))
    exy = safe_float(case.params.get("E0_xy"))
    if all(value is None for value in (exx, eyy, ezz, exy)):
        mode = case.parsed_name.get("mode", "unknown")
        strain = case.parsed_name.get("strain_value")
        if strain is not None:
            if mode == "exx":
                exx = strain
                eyy = 0.0
                ezz = 0.0
                exy = 0.0
            elif mode == "eyy":
                exx = 0.0
                eyy = strain
                ezz = 0.0
                exy = 0.0
            elif mode in {"exx_eyy", "biaxial_xy"}:
                exx = strain
                eyy = strain
                ezz = 0.0
                exy = 0.0
            elif mode == "ezz":
                exx = 0.0
                eyy = 0.0
                ezz = strain
                exy = 0.0
            elif mode == "shear_xy":
                exx = 0.0
                eyy = 0.0
                ezz = 0.0
                exy = strain
    return exx, eyy, ezz, exy


def get_row_candidates(case: CaseData) -> list[tuple[str, dict[str, Any]]]:
    ranked = []
    for source, row in case.local_metadata_rows + case.global_metadata_rows:
        score = score_row_match(case, row)
        ranked.append((score, source, row))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [(source, row) for score, source, row in ranked if score > 0]


def prioritize_summary_rows(candidates: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    prioritized: list[tuple[int, int, str, dict[str, Any]]] = []
    for idx, (source, row) in enumerate(candidates):
        normalized_keys = {normalize_key(key) for key in row.keys()}
        is_summary = source.endswith("current_results_master_table_fitted.csv")
        has_excess_barrier = any(
            key in normalized_keys
            for key in (
                "f_cnt_peak_hat_excess",
                "f_cnt_peak_hat_used",
                "f_cnt_fit_hat_excess",
                "f_cnt_fit_hat_used",
                "f_ref_hat_same_strain",
            )
        )
        priority = 0
        if is_summary and has_excess_barrier:
            priority = 3
        elif is_summary:
            priority = 2
        elif has_excess_barrier:
            priority = 1
        prioritized.append((-priority, idx, source, row))
    prioritized.sort()
    return [(source, row) for _priority, _idx, source, row in prioritized]


def extract_by_keys(row: dict[str, Any], keys: Iterable[str]) -> tuple[float | None, str]:
    normalized = {normalize_key(k): (k, v) for k, v in row.items()}
    for key in keys:
        if key not in normalized:
            continue
        raw_key, raw_value = normalized[key]
        numeric = safe_float(raw_value)
        if numeric is None:
            continue
        return numeric, raw_key
    return None, ""


def extract_vector3_by_keys(row: dict[str, Any], keys: Iterable[str]) -> tuple[tuple[float, float, float] | None, str]:
    normalized = {normalize_key(k): (k, v) for k, v in row.items()}
    for key in keys:
        if key not in normalized:
            continue
        raw_key, raw_value = normalized[key]
        matches = FLOAT_RE.findall(str(raw_value).replace("D", "E").replace("d", "e"))
        if len(matches) < 3:
            continue
        try:
            values = (float(matches[0]), float(matches[1]), float(matches[2]))
        except ValueError:
            continue
        if all(math.isfinite(value) for value in values):
            return values, raw_key
    return None, ""


def json_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def format_strain_values(exx: float | None, eyy: float | None, ezz: float | None, exy: float | None) -> str:
    payload = {
        "exx": exx,
        "eyy": eyy,
        "ezz": ezz,
        "exy": exy,
    }
    return json_compact(payload)


def convert_barrier_to_j(value: float, unit: str, T_K: float) -> float:
    if unit == "J":
        return value
    if unit == "eV":
        return value * EV_TO_J
    if unit == "kBT":
        return value * K_B_J_PER_K * T_K
    raise ValueError(f"unsupported barrier unit: {unit}")


def infer_energy_density_scale_j_m3(case: CaseData) -> tuple[float | None, str, list[str]]:
    warnings: list[str] = []
    gamma = safe_float(case.physical_inputs.get("gamma"))
    lambda_sm = safe_float(case.physical_inputs.get("lambda_sm"))
    if gamma is not None and lambda_sm is not None and gamma > 0.0 and lambda_sm > 0.0:
        return 12.0 * gamma / lambda_sm, f"{case.physical_inputs_path}:12*gamma/lambda_sm", warnings
    gamma = safe_float(case.params.get("gamma_Jm2"))
    lambda_sm = safe_float(case.params.get("lambda_sm_m"))
    if gamma is not None and lambda_sm is not None and gamma > 0.0 and lambda_sm > 0.0:
        return 12.0 * gamma / lambda_sm, f"{case.params_path}:12*gamma_Jm2/lambda_sm_m", warnings
    warnings.append("missing gamma/lambda_sm; cannot convert hat free energy density to physical units")
    return None, "", warnings


def infer_c_tot_mol_per_m3(case: CaseData) -> float | None:
    value = safe_float((((case.physical_inputs.get("metadata") or {}).get("scales") or {}).get("c_tot_mol_per_m3")))
    if value is not None and value > 0.0:
        return value
    vm_alpha = safe_float(case.physical_inputs.get("Vm_alpha_0"))
    if vm_alpha is None or vm_alpha <= 0.0:
        vm_alpha = safe_float(case.params.get("Vm_alpha_0_phys_m3mol"))
    if vm_alpha is not None and vm_alpha > 0.0:
        return 1.0 / vm_alpha
    return None


def infer_row_cell_size_m(case: CaseData, row: dict[str, Any], axis_key: str) -> tuple[float | None, str]:
    spacing_value, raw_key = extract_by_keys(row, (f"spacing_{axis_key}",))
    phys_dx_ref = safe_float(case.physical_inputs.get("phys_dx_ref"))
    if spacing_value is not None and phys_dx_ref is not None and spacing_value > 0.0 and phys_dx_ref > 0.0:
        return spacing_value * phys_dx_ref, f"{raw_key} * {case.physical_inputs_path}:phys_dx_ref"
    return infer_phys_dx_m(case, axis_key)


def infer_system_volume_from_row(case: CaseData, row: dict[str, Any]) -> tuple[float | None, str, list[str]]:
    warnings: list[str] = []
    nx, kx = extract_by_keys(row, ("grid_nx",))
    ny, ky = extract_by_keys(row, ("grid_ny",))
    nz, kz = extract_by_keys(row, ("grid_nz",))
    dims_vec, dims_key = extract_vector3_by_keys(row, ("grid_dimensions",))
    if dims_vec is not None:
        if nx is None:
            nx = dims_vec[0]
            kx = dims_key
        if ny is None:
            ny = dims_vec[1]
            ky = dims_key
        if nz is None:
            nz = dims_vec[2]
            kz = dims_key
    if nx is None:
        nx = safe_float(case.params.get("Nx"))
        kx = "pf_input.params:Nx" if nx is not None else kx
    if ny is None:
        ny = safe_float(case.params.get("Ny"))
        ky = "pf_input.params:Ny" if ny is not None else ky
    if nz is None:
        nz = safe_float(case.params.get("Nz"))
        kz = "pf_input.params:Nz" if nz is not None else kz
    if None in (nx, ny, nz):
        for _source, candidate in get_row_candidates(case):
            if nx is None:
                nx, kx = extract_by_keys(candidate, ("grid_nx",))
            if ny is None:
                ny, ky = extract_by_keys(candidate, ("grid_ny",))
            if nz is None:
                nz, kz = extract_by_keys(candidate, ("grid_nz",))
            dims_vec, dims_key = extract_vector3_by_keys(candidate, ("grid_dimensions",))
            if dims_vec is not None:
                if nx is None:
                    nx = dims_vec[0]
                    kx = dims_key
                if ny is None:
                    ny = dims_vec[1]
                    ky = dims_key
                if nz is None:
                    nz = dims_vec[2]
                    kz = dims_key
            if None not in (nx, ny, nz):
                break
    dx_m, sx = infer_row_cell_size_m(case, row, "x")
    dy_m, sy = infer_row_cell_size_m(case, row, "y")
    dz_m, sz = infer_row_cell_size_m(case, row, "z")
    if None in (nx, ny, nz, dx_m, dy_m, dz_m):
        warnings.append("missing grid dimensions or cell sizes for hat-energy conversion")
        return None, "", warnings
    volume = int(round(nx)) * int(round(ny)) * int(round(nz)) * dx_m * dy_m * dz_m
    return volume, f"{kx},{ky},{kz}; {sx}; {sy}; {sz}", warnings


def build_hat_energy_conversion(case: CaseData, row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    scale_j_m3, scale_source, scale_warnings = infer_energy_density_scale_j_m3(case)
    warnings.extend(scale_warnings)
    system_volume_m3, volume_source, volume_warnings = infer_system_volume_from_row(case, row)
    warnings.extend(volume_warnings)
    nx, _ = extract_by_keys(row, ("grid_nx",))
    ny, _ = extract_by_keys(row, ("grid_ny",))
    nz, _ = extract_by_keys(row, ("grid_nz",))
    dims_vec, _ = extract_vector3_by_keys(row, ("grid_dimensions",))
    if dims_vec is not None:
        if nx is None:
            nx = dims_vec[0]
        if ny is None:
            ny = dims_vec[1]
        if nz is None:
            nz = dims_vec[2]
    if nx is None:
        nx = safe_float(case.params.get("Nx"))
    if ny is None:
        ny = safe_float(case.params.get("Ny"))
    if nz is None:
        nz = safe_float(case.params.get("Nz"))
    dx_m, dx_source = infer_row_cell_size_m(case, row, "x")
    dy_m, dy_source = infer_row_cell_size_m(case, row, "y")
    dz_m, dz_source = infer_row_cell_size_m(case, row, "z")
    voxel_count, _ = extract_by_keys(row, ("voxel_count",))
    spacing_vec, spacing_key = extract_vector3_by_keys(row, ("spacing_sim_units",))
    f_cnt_peak_hat, _ = extract_by_keys(row, ("f_cnt_peak_hat",))
    f_total_cnt_hat, _ = extract_by_keys(row, ("f_total_cnt_hat",))
    mu_reference = safe_float(case.params.get("mu_reference_scale"))
    if mu_reference is None:
        mu_reference = safe_float(case.physical_inputs.get("mu_reference_scale"))
    c_tot = infer_c_tot_mol_per_m3(case)
    voxel_volume_m3 = None
    if None not in (dx_m, dy_m, dz_m):
        voxel_volume_m3 = dx_m * dy_m * dz_m
    volume_from_voxels_m3 = None
    if voxel_count is not None and voxel_volume_m3 is not None:
        volume_from_voxels_m3 = voxel_count * voxel_volume_m3
    spacing_text = ""
    if spacing_vec is not None:
        spacing_text = f"{spacing_key}={spacing_vec}"
    elif any(value is not None for value in (dx_m, dy_m, dz_m)):
        spacing_text = "spacing_sim_units unavailable"
    dx_text = ""
    if None not in (dx_m, dy_m, dz_m):
        dx_text = json_compact(
            {
                "dx_nm": dx_m * 1.0e9,
                "dy_nm": dy_m * 1.0e9,
                "dz_nm": dz_m * 1.0e9,
                "dx_source": dx_source,
                "dy_source": dy_source,
                "dz_source": dz_source,
            }
        )
    formula = (
        "barrier_J = F_hat * w_phys * V_box, "
        "with w_phys = 12*gamma/lambda_sm from Unit_Psedobinary.py / main_cuda.cu; "
        "F_total_CNT_hat is box-averaged hat free-energy density in main_cuda.cu; "
        "no extra mu_reference_scale, c_tot, Vm_alpha, Vm_compound, or per-voxel volume factor "
        "is multiplied beyond V_box unless explicitly stated."
    )
    factors = {
        "Nx": int(round(nx)) if nx is not None else None,
        "Ny": int(round(ny)) if ny is not None else None,
        "Nz": int(round(nz)) if nz is not None else None,
        "energy_density_scale_J_per_m3": scale_j_m3,
        "energy_density_scale_source": scale_source,
        "box_volume_m3": system_volume_m3,
        "box_volume_source": volume_source,
        "voxel_volume_m3": voxel_volume_m3,
        "voxel_count": voxel_count,
        "volume_from_voxels_m3": volume_from_voxels_m3,
        "spacing_sim_units": spacing_vec,
        "mu_reference_scale_J_per_mol_available_not_multiplied": mu_reference,
        "c_tot_mol_per_m3_available_not_multiplied": c_tot,
    }
    return {
        "scale_j_m3": scale_j_m3,
        "scale_source": scale_source,
        "system_volume_m3": system_volume_m3,
        "volume_source": volume_source,
        "Nx": int(round(nx)) if nx is not None else None,
        "Ny": int(round(ny)) if ny is not None else None,
        "Nz": int(round(nz)) if nz is not None else None,
        "dx_nm": (dx_m * 1.0e9 if dx_m is not None else None),
        "voxel_count": voxel_count,
        "spacing_sim_units": spacing_text,
        "dx_nm_or_spacing_physical": dx_text,
        "volume_from_voxels_m3": volume_from_voxels_m3,
        "mu_reference_if_used": mu_reference,
        "c_tot_if_used": c_tot,
        "conversion_formula_used": formula,
        "all_conversion_factors": factors,
        "f_cnt_peak_hat": f_cnt_peak_hat,
        "f_total_cnt_hat_if_available": f_total_cnt_hat,
    }, warnings


def make_single_case(case: CaseData, case_dir: Path) -> CaseData:
    params_path = find_nearest_case_param_file(case_dir, "pf_input.params")
    return CaseData(
        key=case.key,
        base_case_tag=case.base_case_tag,
        representative_dir=case_dir,
        case_dirs=[case_dir],
        params_path=params_path,
        params=parse_params_file(params_path) if params_path else case.params,
        physical_inputs_path=case.physical_inputs_path,
        physical_inputs=case.physical_inputs,
        parsed_name=parse_case_name(case_dir.name),
        local_metadata_rows=load_metadata_rows(case_dir / "summary.txt") if (case_dir / "summary.txt").exists() else [],
        global_metadata_rows=case.global_metadata_rows,
    )


def convert_hat_energy_to_j(
    value_hat: float,
    case: CaseData,
    row: dict[str, Any],
) -> tuple[float | None, str, list[str], dict[str, Any]]:
    audit, warnings = build_hat_energy_conversion(case, row)
    scale_j_m3 = audit.get("scale_j_m3")
    system_volume_m3 = audit.get("system_volume_m3")
    if scale_j_m3 is None or system_volume_m3 is None:
        return None, "", warnings, audit
    barrier_j = value_hat * scale_j_m3 * system_volume_m3
    source = f"{audit.get('scale_source', '')}; V_box={audit.get('volume_source', '')}"
    return barrier_j, source, warnings, audit


def split_profile_groups_by_spacing(points: list[ProfilePoint]) -> tuple[list[ProfilePoint], list[str]]:
    warnings: list[str] = []
    ordered = sorted(
        [point for point in points if point.nominal_radius_nm is not None],
        key=lambda point: float(point.nominal_radius_nm),
    )
    if len(ordered) < 4:
        return points, warnings
    radii = np.array([float(point.nominal_radius_nm) for point in ordered], dtype=float)
    dr = np.diff(radii)
    positive_dr = dr[dr > 1.0e-12]
    if len(positive_dr) == 0:
        return points, warnings
    dr_min = float(np.min(positive_dr))
    threshold = 1.5 * dr_min
    split_idx = None
    for idx, step in enumerate(dr):
        if step >= threshold:
            split_idx = idx + 1
            break
    if split_idx is None or split_idx <= 1:
        return points, warnings
    fine_points = ordered[:split_idx]
    warnings.append(
        f"detected mixed radius-spacing groups; used first fine-step group with {len(fine_points)} points for barrier/profile fitting"
    )
    return fine_points, warnings


def load_barrier_data(case: CaseData) -> BarrierData:
    warnings: list[str] = []
    T_K = infer_temperature_k(case)
    if T_K is None:
        warnings.append("missing temperature; cannot convert barrier units that depend on T")
    barrier_keys = {
        "J": (
            "deltag_star_j",
            "delta_g_star_j",
            "barrier_j",
            "delta_g_j",
            "f_saddle_minus_f_matrix_j",
            "free_energy_barrier_j",
        ),
        "eV": (
            "deltag_star_ev",
            "delta_g_star_ev",
            "barrier_ev",
            "delta_g_ev",
            "f_saddle_minus_f_matrix_ev",
            "free_energy_barrier_ev",
        ),
        "kBT": (
            "deltag_star_kbt",
            "delta_g_star_kbt",
            "barrier_kbt",
            "delta_g_kbt",
            "f_saddle_minus_f_matrix_kbt",
            "free_energy_barrier_kbt",
        ),
        "ambiguous": (
            "f_cnt_peak_hat_excess",
            "f_cnt_peak_hat_used",
            "f_cnt_fit_hat_excess",
            "f_cnt_fit_hat_used",
            "deltag_star",
            "delta_g_star",
            "barrier",
            "free_energy_barrier",
            "f_saddle_minus_f_matrix",
            "f_cnt_peak_hat_absolute",
            "f_cnt_fit_hat_absolute",
            "f_cnt_peak_hat",
            "f_cnt_fit_hat",
            "f_total_cnt_hat",
            "f_total_excess_hat",
        ),
    }
    candidates = prioritize_summary_rows(get_row_candidates(case))
    for source, row in candidates:
        for unit, keys in barrier_keys.items():
            value, raw_key = extract_by_keys(row, keys)
            if value is None:
                continue
            if unit == "ambiguous":
                barrier_j, hat_source, hat_warnings, audit = convert_hat_energy_to_j(value, case, row)
                warnings.extend(hat_warnings)
                if barrier_j is not None:
                    ref_type = str(row.get("barrier_reference_type", "")).strip()
                    if raw_key in {"f_cnt_peak_hat_absolute", "f_cnt_fit_hat_absolute", "f_cnt_peak_hat", "f_cnt_fit_hat", "f_total_cnt_hat"}:
                        warnings.append("missing_same_strain_matrix_reference; using absolute or smallest-sibling reference")
                    elif raw_key in {"f_cnt_peak_hat_excess", "f_cnt_peak_hat_used", "f_cnt_fit_hat_excess", "f_cnt_fit_hat_used"}:
                        warnings.append("using same-strain matrix-reference-subtracted excess barrier from CNT summary table")
                    elif ref_type == "matrix_only_same_strain":
                        warnings.append("using same-strain matrix-reference-subtracted barrier indicated by summary metadata")
                    warnings.append(
                        f"converted {raw_key} from hat free-energy density to J using physical energy density scale and box volume"
                    )
                    return BarrierData(
                        barrier_j=barrier_j,
                        source=f"{source}:{raw_key} -> {hat_source}",
                        warnings=warnings,
                        source_field=raw_key,
                        source_unit="hat_density",
                        raw_value=value,
                        row_source=source,
                        conversion_formula_used=str(audit.get("conversion_formula_used", "")),
                        all_conversion_factors=dict(audit.get("all_conversion_factors", {})),
                        f_cnt_peak_hat=audit.get("f_cnt_peak_hat"),
                        f_total_cnt_hat_if_available=audit.get("f_total_cnt_hat_if_available"),
                        voxel_count=audit.get("voxel_count"),
                        spacing_sim_units=str(audit.get("spacing_sim_units", "")),
                        dx_nm_or_spacing_physical=str(audit.get("dx_nm_or_spacing_physical", "")),
                        volume_from_voxels_m3=audit.get("volume_from_voxels_m3"),
                        mu_reference_if_used=audit.get("mu_reference_if_used"),
                        c_tot_if_used=audit.get("c_tot_if_used"),
                    )
                warnings.append(
                    f"ambiguous barrier unit in {source} field '{raw_key}'; explicit J/eV/kBT required"
                )
                break
            if T_K is None and unit == "kBT":
                warnings.append(
                    f"cannot convert {raw_key} from kBT without temperature"
                )
                break
            barrier_j = convert_barrier_to_j(value, unit, T_K if T_K is not None else 1.0)
            audit, audit_warnings = build_hat_energy_conversion(case, row)
            warnings.extend(audit_warnings)
            return BarrierData(
                barrier_j=barrier_j,
                source=f"{source}:{raw_key}",
                warnings=warnings,
                source_field=raw_key,
                source_unit=unit,
                raw_value=value,
                row_source=source,
                conversion_formula_used=(
                    f"barrier_J = {raw_key}" if unit == "J" else
                    f"barrier_J = {raw_key} * eV_to_J" if unit == "eV" else
                    f"barrier_J = {raw_key} * kB * T_K"
                ),
                all_conversion_factors={
                    "explicit_unit": unit,
                    "eV_to_J": EV_TO_J if unit == "eV" else None,
                    "T_K": T_K,
                    "kB_J_per_K": K_B_J_PER_K if unit == "kBT" else None,
                    "hat_conversion_context_not_used": audit.get("all_conversion_factors"),
                },
                f_cnt_peak_hat=audit.get("f_cnt_peak_hat"),
                f_total_cnt_hat_if_available=audit.get("f_total_cnt_hat_if_available"),
                voxel_count=audit.get("voxel_count"),
                spacing_sim_units=str(audit.get("spacing_sim_units", "")),
                dx_nm_or_spacing_physical=str(audit.get("dx_nm_or_spacing_physical", "")),
                volume_from_voxels_m3=audit.get("volume_from_voxels_m3"),
                mu_reference_if_used=audit.get("mu_reference_if_used"),
                c_tot_if_used=audit.get("c_tot_if_used"),
            )
    return BarrierData(barrier_j=None, source="", warnings=warnings)


def pick_phi_vtk(case_dir: Path) -> Path | None:
    finals = sorted(case_dir.glob("phi_final_*.vtk"))
    if finals:
        return finals[0]
    numeric: list[tuple[int, Path]] = []
    for path in case_dir.glob("phi_*.vtk"):
        suffix = path.stem.removeprefix("phi_")
        if suffix.isdigit():
            numeric.append((int(suffix), path))
    if numeric:
        numeric.sort()
        return numeric[-1][1]
    init = case_dir / "phi_init.vtk"
    return init if init.exists() else None


def read_legacy_scalar_vtk(path: Path) -> tuple[np.ndarray, tuple[int, int, int], tuple[float, float, float]]:
    dims: tuple[int, int, int] | None = None
    spacing = (1.0, 1.0, 1.0)
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"unexpected EOF while parsing {path}")
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("DIMENSIONS"):
                _, sx, sy, sz = text.split()
                dims = (int(sx), int(sy), int(sz))
            elif text.startswith("SPACING"):
                _, sx, sy, sz = text.split()
                spacing = (float(sx), float(sy), float(sz))
            elif text.startswith("ASPECT_RATIO"):
                _, sx, sy, sz = text.split()
                spacing = (float(sx), float(sy), float(sz))
            elif text.startswith("LOOKUP_TABLE"):
                break
        if dims is None:
            raise ValueError(f"missing DIMENSIONS in {path}")
        payload = handle.read().decode("ascii", errors="replace")
    values = np.fromstring(payload, sep=" ", dtype=np.float64)
    expected = dims[0] * dims[1] * dims[2]
    if values.size != expected:
        raise ValueError(f"VTK scalar count mismatch in {path}: got {values.size}, expected {expected}")
    return values.reshape(dims, order="C"), dims, spacing


def h_of_phi(phi: np.ndarray) -> np.ndarray:
    p2 = phi * phi
    p3 = p2 * phi
    return p3 * (6.0 * p2 - 15.0 * phi + 10.0)


def infer_phys_dx_m(case: CaseData, axis_key: str) -> tuple[float | None, str]:
    physical_inputs = case.physical_inputs
    key_pf = f"pf_d{axis_key}"
    if key_pf in physical_inputs:
        value = safe_float(physical_inputs.get(key_pf))
        if value is not None and value > 0.0:
            return value, f"{case.physical_inputs_path}:{key_pf}"
    if "pf_dx" in physical_inputs:
        value = safe_float(physical_inputs.get("pf_dx"))
        if value is not None and value > 0.0:
            return value, f"{case.physical_inputs_path}:pf_dx"
    key_legacy = f"d{axis_key}"
    if key_legacy in physical_inputs:
        value = safe_float(physical_inputs.get(key_legacy))
        if value is not None and value > 0.0:
            return value, f"{case.physical_inputs_path}:{key_legacy}"
    phys_dx_ref = safe_float(physical_inputs.get("phys_dx_ref"))
    code_spacing = safe_float(case.params.get(f"d{axis_key}"))
    if phys_dx_ref is not None and code_spacing is not None and phys_dx_ref > 0.0 and code_spacing > 0.0:
        return phys_dx_ref * code_spacing, f"{case.params_path}:d{axis_key} * {case.physical_inputs_path}:phys_dx_ref"
    return None, ""


def compute_r_eff_from_phi(case: CaseData, case_dir: Path) -> tuple[float | None, str, list[str]]:
    warnings: list[str] = []
    phi_vtk = pick_phi_vtk(case_dir)
    if phi_vtk is None:
        return None, "", [f"missing phi VTK in {case_dir}"]
    dx_m, sx = infer_phys_dx_m(case, "x")
    dy_m, sy = infer_phys_dx_m(case, "y")
    dz_m, sz = infer_phys_dx_m(case, "z")
    if dx_m is None or dy_m is None or dz_m is None:
        return None, "", [f"missing physical voxel size for {case_dir}"]
    try:
        phi, _dims, _spacing = read_legacy_scalar_vtk(phi_vtk)
    except Exception as exc:
        return None, "", [f"failed to read {phi_vtk}: {exc}"]
    phi = np.clip(phi, 0.0, 1.0)
    vh_m3 = float(np.sum(h_of_phi(phi)) * dx_m * dy_m * dz_m)
    if vh_m3 <= 0.0:
        warnings.append(f"non-positive V_h from {phi_vtk}")
        return None, "", warnings
    r_eff = ((3.0 * vh_m3) / (4.0 * math.pi)) ** (1.0 / 3.0)
    source = f"{phi_vtk}:V_h from h(phi), cell sizes [{sx}; {sy}; {sz}]"
    return r_eff, source, warnings


def extract_r_eff_from_rows(case: CaseData) -> tuple[float | None, str, list[str], float | None]:
    warnings: list[str] = []
    radius_keys_nm = (
        "r_eff_star_nm",
        "r_eff_nm",
        "radius_eff_nm",
        "effective_radius_nm",
        "r_eff_nm_star",
        "r_eff_nm_fit",
        "r_star_eff_nm",
        "r_eff",
        "r_eff_star",
        "r_effective_nm",
        "r_eff_cnt_nm",
    )
    radius_keys_m = (
        "r_eff_star_m",
        "r_eff_m",
        "radius_eff_m",
        "effective_radius_m",
    )
    volume_keys_m3 = (
        "v_h_m3",
        "vh_m3",
        "precipitate_volume_m3",
        "volume_m3",
    )
    volume_keys_nm3 = (
        "v_h_nm3",
        "vh_nm3",
        "precipitate_volume_nm3",
        "volume_nm3",
    )
    nominal_radius_keys = ("rc_cnt_fit_nm", "rc_cnt_nm", "radius_nm", "critical_radius_nm")
    for source, row in get_row_candidates(case):
        value_m, raw_key = extract_by_keys(row, radius_keys_m)
        if value_m is not None and value_m > 0.0:
            return value_m, f"{source}:{raw_key}", warnings, None
        value_nm, raw_key = extract_by_keys(row, radius_keys_nm)
        if value_nm is not None and value_nm > 0.0:
            return value_nm * 1.0e-9, f"{source}:{raw_key}", warnings, None
        volume_m3, raw_key = extract_by_keys(row, volume_keys_m3)
        if volume_m3 is not None and volume_m3 > 0.0:
            r_eff = ((3.0 * volume_m3) / (4.0 * math.pi)) ** (1.0 / 3.0)
            return r_eff, f"{source}:{raw_key} -> r_eff", warnings, None
        volume_nm3, raw_key = extract_by_keys(row, volume_keys_nm3)
        if volume_nm3 is not None and volume_nm3 > 0.0:
            volume_m3 = volume_nm3 * 1.0e-27
            r_eff = ((3.0 * volume_m3) / (4.0 * math.pi)) ** (1.0 / 3.0)
            return r_eff, f"{source}:{raw_key} -> r_eff", warnings, None
        voxel_count, raw_key = extract_by_keys(row, ("voxel_count",))
        if voxel_count is not None and voxel_count > 0.0:
            dx_m, sx = infer_row_cell_size_m(case, row, "x")
            dy_m, sy = infer_row_cell_size_m(case, row, "y")
            dz_m, sz = infer_row_cell_size_m(case, row, "z")
            if None not in (dx_m, dy_m, dz_m):
                volume_m3 = voxel_count * dx_m * dy_m * dz_m
                r_eff = ((3.0 * volume_m3) / (4.0 * math.pi)) ** (1.0 / 3.0)
                warnings.append(
                    "r_eff_star reconstructed from thresholded voxel_count volume; this approximates V_h with phi>0.5 geometry"
                )
                return r_eff, f"{source}:{raw_key} * [{sx}; {sy}; {sz}] -> r_eff", warnings, None
        nominal_nm, raw_key = extract_by_keys(row, nominal_radius_keys)
        if nominal_nm is not None and nominal_nm > 0.0:
            return None, "", warnings, nominal_nm
    return None, "", warnings, None


def load_r_eff_star(case: CaseData) -> tuple[float | None, str, list[str]]:
    warnings: list[str] = []
    value, source, row_warnings, nominal_nm = extract_r_eff_from_rows(case)
    warnings.extend(row_warnings)
    if value is not None:
        return value, source, warnings
    if nominal_nm is not None:
        closest_dir = min(
            case.case_dirs,
            key=lambda path: abs((parse_case_name(path.name).get("nominal_radius_nm") or math.inf) - nominal_nm),
        )
        r_eff, phi_source, phi_warnings = compute_r_eff_from_phi(case, closest_dir)
        warnings.extend(phi_warnings)
        if r_eff is not None:
            warnings.append(f"used sibling phi VTK nearest nominal radius {nominal_nm:.6f} nm")
            return r_eff, phi_source, warnings
    for case_dir in case.case_dirs:
        r_eff, phi_source, phi_warnings = compute_r_eff_from_phi(case, case_dir)
        warnings.extend(phi_warnings)
        if r_eff is not None:
            warnings.append("used first sibling phi VTK because explicit r_eff_star was not found")
            return r_eff, phi_source, warnings
    warnings.append("missing r_eff_star and unable to reconstruct from phi VTK")
    return None, "", warnings


def read_last_csv_row(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return None
    return rows[-1] if rows else None


def convert_energy_value_to_j(value: float, unit: str, T_K: float) -> float:
    if unit == "J":
        return value
    if unit == "eV":
        return value * EV_TO_J
    if unit == "kBT":
        return value * K_B_J_PER_K * T_K
    raise ValueError(f"unsupported energy unit: {unit}")


def extract_profile_energy(row: dict[str, Any], T_K: float) -> tuple[float | None, str, list[str]]:
    warnings: list[str] = []
    energy_keys = {
        "J": (
            "deltag_j",
            "delta_g_j",
            "deltag_star_j",
            "delta_g_star_j",
            "f_total_excess_j",
            "f_total_cnt_j",
        ),
        "eV": (
            "deltag_ev",
            "delta_g_ev",
            "deltag_star_ev",
            "delta_g_star_ev",
            "f_total_excess_ev",
            "f_total_cnt_ev",
        ),
        "kBT": (
            "deltag_kbt",
            "delta_g_kbt",
            "deltag_star_kbt",
            "delta_g_star_kbt",
            "f_total_excess_kbt",
            "f_total_cnt_kbt",
        ),
        "ambiguous": (
            "f_total_cnt_hat",
            "f_total_excess_hat",
            "f_cnt_peak_hat",
            "deltag_star",
            "delta_g_star",
        ),
    }
    for unit, keys in energy_keys.items():
        value, raw_key = extract_by_keys(row, keys)
        if value is None:
            continue
        if unit == "ambiguous":
            warnings.append(f"ambiguous profile energy unit in field '{raw_key}'")
            return None, "", warnings
        return convert_energy_value_to_j(value, unit, T_K), raw_key, warnings
    return None, "", warnings


def extract_profile_energy_for_case(
    case: CaseData,
    row: dict[str, Any],
    T_K: float,
) -> tuple[float | None, str, list[str]]:
    energy_j, source, warnings = extract_profile_energy(row, T_K)
    if energy_j is not None:
        return energy_j, source, warnings
    value_hat, raw_key = extract_by_keys(row, ("f_total_cnt_hat", "f_cnt_peak_hat", "f_cnt_fit_hat", "f_total_excess_hat"))
    if value_hat is None:
        return None, "", warnings
    barrier_j, hat_source, hat_warnings, _audit = convert_hat_energy_to_j(value_hat, case, row)
    if barrier_j is not None:
        warnings = [warning for warning in warnings if "ambiguous profile energy unit" not in warning]
        warnings.extend(hat_warnings)
        warnings.append(f"converted profile field '{raw_key}' from hat units to J")
        return barrier_j, f"{raw_key} -> {hat_source}", warnings
    warnings.extend(hat_warnings)
    warnings.append(f"ambiguous profile energy unit in field '{raw_key}'")
    return None, "", warnings


def load_profile_from_explicit_files(
    case: CaseData,
    profile_files: list[Path],
) -> tuple[list[ProfilePoint], str, list[str]]:
    warnings: list[str] = []
    points: list[ProfilePoint] = []
    T_K = infer_temperature_k(case)
    if T_K is None:
        return points, "", ["missing temperature for profile conversion"]
    for path in profile_files:
        if case.base_case_tag not in str(path):
            continue
        for source, row in load_metadata_rows(path):
            if score_row_match(case, row) < 15:
                continue
            r_eff, r_key = extract_by_keys(
                row,
                (
                    "r_eff_star_m",
                    "r_eff_m",
                    "r_eff_star_nm",
                    "r_eff_nm",
                    "radius_eff_nm",
                    "effective_radius_nm",
                    "rc_cnt_fit_nm",
                    "rc_cnt_nm",
                ),
            )
            if r_eff is None:
                continue
            if str(r_key).endswith("_nm") or "nm" in normalize_key(r_key):
                r_eff_m = r_eff * 1.0e-9
            else:
                r_eff_m = r_eff
            energy_j, energy_source, energy_warnings = extract_profile_energy_for_case(case, row, T_K)
            warnings.extend(energy_warnings)
            if energy_j is None:
                continue
            points.append(
                ProfilePoint(
                    case_dir=str(path),
                    nominal_radius_nm=None,
                    r_eff_m=r_eff_m,
                    energy_j=energy_j,
                    energy_source=f"{source}:{energy_source}",
                )
            )
    points, split_warnings = split_profile_groups_by_spacing(points)
    warnings.extend(split_warnings)
    source = "explicit_profile_csv" if points else ""
    return points, source, warnings


def build_profile_from_sibling_cases(case: CaseData) -> tuple[list[ProfilePoint], str, list[str]]:
    warnings: list[str] = []
    points: list[ProfilePoint] = []
    T_K = infer_temperature_k(case)
    if T_K is None:
        return points, "", ["missing temperature for profile conversion"]
    for case_dir in case.case_dirs:
        energy_csvs = sorted(case_dir.glob("energy_minimize_*.csv"))
        if not energy_csvs:
            continue
        row = read_last_csv_row(energy_csvs[0])
        if row is None:
            continue
        sibling_case = make_single_case(case, case_dir)
        r_eff_m, r_source, r_warnings = load_r_eff_star(sibling_case)
        warnings.extend(r_warnings)
        energy_j, energy_source, e_warnings = extract_profile_energy_for_case(sibling_case, row, T_K)
        warnings.extend(e_warnings)
        points.append(
            ProfilePoint(
                case_dir=str(case_dir),
                nominal_radius_nm=parse_case_name(case_dir.name).get("nominal_radius_nm"),
                r_eff_m=r_eff_m,
                energy_j=energy_j,
                energy_source=f"{energy_csvs[0]}:{energy_source}" if energy_source else "",
                warnings=r_warnings + e_warnings + ([r_source] if r_source else []),
            )
        )
    points, split_warnings = split_profile_groups_by_spacing(points)
    warnings.extend(split_warnings)
    source = "sibling_case_profile" if points else ""
    return points, source, warnings


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def collect_sibling_energy_points(case: CaseData) -> tuple[list[SiblingEnergyPoint], list[str]]:
    warnings: list[str] = []
    points: list[SiblingEnergyPoint] = []
    for case_dir in case.case_dirs:
        reference_csv = case_dir / "reference_energy.csv"
        if reference_csv.exists():
            ref_rows = read_csv_rows(reference_csv)
            ref_row = ref_rows[-1] if ref_rows else None
            if ref_row is not None:
                points.append(
                    SiblingEnergyPoint(
                        case_dir=str(case_dir),
                        nominal_radius_nm=0.0,
                        voxel_count=safe_float(ref_row.get("voxel_count")),
                        mean_h=safe_float(ref_row.get("mean_h")),
                        F_surf_hat=safe_float(ref_row.get("F_surf_ref_hat")),
                        F_el_hat=safe_float(ref_row.get("F_el_ref_hat")),
                        F_total_excess_hat=safe_float(ref_row.get("F_total_ref_hat")),
                        F_chem_CNT_hat=safe_float(ref_row.get("F_chem_ref_hat")),
                        F_total_CNT_hat=safe_float(ref_row.get("F_total_ref_hat")),
                        r_eff_m=0.0,
                        energy_csv_path=str(reference_csv),
                        is_reference_like=True,
                    )
                )
                continue
        energy_csvs = sorted(case_dir.glob("energy_minimize_*.csv"))
        if not energy_csvs:
            continue
        rows = read_csv_rows(energy_csvs[0])
        if not rows:
            continue
        last = rows[-1]
        sibling_case = make_single_case(case, case_dir)
        r_eff_m, _r_source, r_warnings = load_r_eff_star(sibling_case)
        warnings.extend(r_warnings)
        voxel_count = None
        for _source, row in sibling_case.local_metadata_rows + sibling_case.global_metadata_rows:
            voxel_count, _raw = extract_by_keys(row, ("voxel_count",))
            if voxel_count is not None:
                break
        point = SiblingEnergyPoint(
            case_dir=str(case_dir),
            nominal_radius_nm=parse_case_name(case_dir.name).get("nominal_radius_nm"),
            voxel_count=voxel_count,
            mean_h=safe_float(last.get("mean_h")),
            F_surf_hat=safe_float(last.get("F_surf_hat")),
            F_el_hat=safe_float(last.get("F_el_hat")),
            F_total_excess_hat=safe_float(last.get("F_total_excess_hat")),
            F_chem_CNT_hat=safe_float(last.get("F_chem_CNT_hat")),
            F_total_CNT_hat=safe_float(last.get("F_total_CNT_hat")),
            r_eff_m=r_eff_m,
            energy_csv_path=str(energy_csvs[0]),
        )
        points.append(point)
    points = [point for point in points if point.F_total_CNT_hat is not None]
    points.sort(
        key=lambda point: (
            point.voxel_count if point.voxel_count is not None else math.inf,
            point.mean_h if point.mean_h is not None else math.inf,
            point.nominal_radius_nm if point.nominal_radius_nm is not None else math.inf,
        )
    )
    return points, warnings


def infer_zero_like_strain(case: CaseData) -> bool:
    exx, eyy, ezz, exy = infer_strain_components(case)
    return max(abs(value or 0.0) for value in (exx, eyy, ezz, exy)) < 1.0e-15


def match_same_thermo(case_a: CaseData, case_b: CaseData) -> bool:
    ta = infer_temperature_c(case_a)
    tb = infer_temperature_c(case_b)
    xa = infer_xb_loc(case_a)
    xb = infer_xb_loc(case_b)
    if ta is None or tb is None or xa is None or xb is None:
        return False
    return abs(ta - tb) < 1.0e-9 and abs(xa - xb) < 1.0e-12


def select_reference_point(points: list[SiblingEnergyPoint]) -> tuple[SiblingEnergyPoint | None, str, str, list[str]]:
    warnings: list[str] = []
    if not points:
        return None, "", "missing", warnings
    for point in points:
        if point.voxel_count is not None and point.voxel_count <= 1.0:
            point.is_reference_like = True
            return point, f"explicit_volume_zero:{point.case_dir}", "high", warnings
    candidate = points[0]
    candidate.is_reference_like = True
    warnings.append("same-strain reference approximated by smallest-volume sibling; explicit matrix-only reference not found")
    return candidate, f"same_base_smallest_volume_sibling:{candidate.case_dir}", "low", warnings


def find_unstrained_reference_case(case: CaseData, all_cases: list[CaseData]) -> CaseData | None:
    candidates = [
        other for other in all_cases
        if other.base_case_tag != case.base_case_tag
        and match_same_thermo(case, other)
        and infer_mode(other) == infer_mode(case)
        and infer_zero_like_strain(other)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.base_case_tag)[0]


def collect_profile_points(case: CaseData, profile_files: list[Path]) -> tuple[list[ProfilePoint], str, list[str]]:
    warnings: list[str] = []
    explicit_points, explicit_source, explicit_warnings = load_profile_from_explicit_files(case, profile_files)
    warnings.extend(explicit_warnings)
    points = [point for point in explicit_points if point.r_eff_m is not None and point.energy_j is not None]
    profile_source = explicit_source
    if points:
        return points, profile_source, warnings
    sibling_points, sibling_source, sibling_warnings = build_profile_from_sibling_cases(case)
    warnings.extend(sibling_warnings)
    points = [point for point in sibling_points if point.r_eff_m is not None and point.energy_j is not None]
    profile_source = sibling_source
    return points, profile_source, warnings


def evaluate_quadratic_fit(
    items: list[tuple[float, float]],
    T_K: float,
    source_label: str,
    weights: np.ndarray | None = None,
) -> ZeldovichResult:
    if len(items) < 3:
        return ZeldovichResult(value=None, source=source_label, warnings=[f"{source_label} requires at least 3 points"])
    radii = np.array([item[0] for item in items], dtype=float)
    energies = np.array([item[1] for item in items], dtype=float)
    order = np.argsort(radii)
    radii = radii[order]
    energies = energies[order]
    if len(np.unique(radii)) < 3:
        return ZeldovichResult(value=None, source=source_label, warnings=[f"{source_label} has fewer than 3 unique radii"])
    try:
        coeff = np.polyfit(radii, energies, 2, w=weights[order] if weights is not None else None)
    except np.linalg.LinAlgError as exc:
        return ZeldovichResult(value=None, source=source_label, warnings=[f"{source_label} failed: {exc}"])
    predicted = np.polyval(coeff, radii)
    ss_res = float(np.sum((energies - predicted) ** 2))
    ss_tot = float(np.sum((energies - np.mean(energies)) ** 2))
    fit_r2 = 1.0 if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
    curvature = 2.0 * float(coeff[0])
    curvature_sign = "negative" if curvature < 0.0 else "positive" if curvature > 0.0 else "zero"
    z_value = None
    if math.isfinite(curvature) and abs(curvature) > 0.0 and T_K > 0.0:
        z_value = math.sqrt(abs(curvature) / (2.0 * math.pi * K_B_J_PER_K * T_K))
    warnings: list[str] = []
    if not math.isfinite(curvature) or curvature == 0.0:
        warnings.append(f"{source_label} gave zero or non-finite curvature")
    return ZeldovichResult(
        value=z_value,
        source=source_label,
        warnings=warnings,
        fit_r2=fit_r2,
        window_points=len(items),
        curvature_sign=curvature_sign,
        curvature_j_per_m2=curvature,
    )


def compute_strict_zeldovich(
    case: CaseData,
    profile_files: list[Path],
    r_eff_star_m: float | None,
    fit_points: int,
    allow_fallback: bool,
    fallback_value: float | None,
    barrier_j: float | None,
) -> ZeldovichResult:
    warnings: list[str] = []
    if r_eff_star_m is None:
        return ZeldovichResult(value=None, source="fallback_missing_profile", warnings=["missing r_eff_star; cannot evaluate Zeldovich factor"])
    points, profile_source, point_warnings = collect_profile_points(case, profile_files)
    warnings.extend(point_warnings)
    T_K = infer_temperature_k(case)
    if T_K is None:
        return ZeldovichResult(value=None, source="fallback_missing_profile", warnings=warnings + ["missing temperature for Zeldovich factor"], profile_source=profile_source)
    if len(points) >= 3:
        unique_points = sorted({point.r_eff_m: point.energy_j for point in points}.items(), key=lambda item: item[0])
        sorted_by_distance = sorted(unique_points, key=lambda item: abs(item[0] - r_eff_star_m))
        selected = sorted(sorted_by_distance[: max(3, fit_points)], key=lambda item: item[0])
        bracketed = selected[0][0] < r_eff_star_m < selected[-1][0]
        if not bracketed:
            warnings.append("profile points do not bracket r_eff_star; using nearest-point quadratic fit")
        strict_fit = evaluate_quadratic_fit(
            selected,
            T_K,
            "profile_quadratic_fit" if bracketed else "profile_nearest_quadratic_fit",
        )
        warnings.extend(strict_fit.warnings)
        if strict_fit.value is not None and strict_fit.curvature_sign == "negative":
            strict_fit.warnings = warnings
            strict_fit.profile_source = profile_source
            return strict_fit

        peak_index = max(range(len(unique_points)), key=lambda idx: unique_points[idx][1])
        if 0 < peak_index < len(unique_points) - 1:
            left = max(0, peak_index - 2)
            right = min(len(unique_points), peak_index + 3)
            peak_window = unique_points[left:right]
            peak_fit = evaluate_quadratic_fit(peak_window, T_K, "profile_peak_quadratic_fit")
            warnings.extend(peak_fit.warnings)
            if peak_fit.value is not None and peak_fit.curvature_sign == "negative":
                peak_fit.warnings = warnings
                peak_fit.profile_source = profile_source
                return peak_fit
    else:
        warnings.append("fewer than 3 usable profile points with explicit energy units")
    if not allow_fallback:
        warnings.append("missing usable profile curvature; strict_status stays incomplete and diagnostic output will use a labeled fallback")
        return ZeldovichResult(value=None, source="fallback_missing_profile", warnings=warnings, profile_source=profile_source)
    if fallback_value is not None and fallback_value > 0.0:
        warnings.append("strict mode used user-specified constant Zeldovich fallback")
        return ZeldovichResult(value=fallback_value, source="fallback_user_constant", warnings=warnings, profile_source=profile_source)
    if barrier_j is not None and r_eff_star_m is not None and r_eff_star_m > 0.0:
        gamma_eff = 3.0 * barrier_j / (4.0 * math.pi * r_eff_star_m * r_eff_star_m)
        if gamma_eff > 0.0:
            z_value = math.sqrt(4.0 * gamma_eff / (K_B_J_PER_K * T_K))
            warnings.append("strict mode used capillary_from_barrier_radius fallback")
            return ZeldovichResult(value=z_value, source="fallback_capillary_from_barrier_radius", warnings=warnings, profile_source=profile_source)
    return ZeldovichResult(value=None, source="fallback_missing_profile", warnings=warnings, profile_source=profile_source)


def compute_diagnostic_zeldovich(
    case: CaseData,
    profile_files: list[Path],
    r_eff_star_m: float | None,
    barrier_j: float | None,
    strict_result: ZeldovichResult,
    fallback_value: float | None,
) -> ZeldovichResult:
    warnings = list(strict_result.warnings)
    if strict_result.value is not None:
        result = ZeldovichResult(**asdict(strict_result))
        result.source = "strict_reused"
        return result
    if r_eff_star_m is None:
        return ZeldovichResult(value=None, source="diagnostic_missing_radius", warnings=warnings + ["missing r_eff_star for diagnostic Zeldovich"])
    T_K = infer_temperature_k(case)
    points, profile_source, point_warnings = collect_profile_points(case, profile_files)
    warnings.extend(point_warnings)
    if T_K is not None and len(points) >= 3:
        unique_points = sorted({point.r_eff_m: point.energy_j for point in points}.items(), key=lambda item: item[0])
        order_by_distance = sorted(unique_points, key=lambda item: abs(item[0] - r_eff_star_m))
        window = sorted(order_by_distance[: min(len(order_by_distance), max(5, min(9, len(order_by_distance))))], key=lambda item: item[0])
        radii = np.array([item[0] for item in window], dtype=float)
        scale = float(np.median(np.abs(radii - r_eff_star_m))) if len(window) > 1 else max(r_eff_star_m, 1.0e-30)
        scale = max(scale, max(r_eff_star_m, 1.0e-30) * 0.1, 1.0e-30)
        weights = 1.0 / (1.0 + ((radii - r_eff_star_m) / scale) ** 2)
        global_fit = evaluate_quadratic_fit(window, T_K, "profile_global_fit", weights=weights)
        global_fit.profile_source = profile_source
        warnings.extend(global_fit.warnings)
        if global_fit.value is not None:
            if global_fit.curvature_sign != "negative":
                warnings.append("profile_global_fit curvature sign is not negative; using diagnostic-only fit")
            if global_fit.fit_r2 is not None and global_fit.fit_r2 < 0.5:
                warnings.append("profile_global_fit R2 is poor; using diagnostic-only fit")
            global_fit.warnings = warnings
            return global_fit
    if T_K is not None and barrier_j is not None and r_eff_star_m > 0.0:
        gamma_eff = 3.0 * barrier_j / (4.0 * math.pi * r_eff_star_m * r_eff_star_m)
        if gamma_eff > 0.0:
            z_value = math.sqrt(4.0 * gamma_eff / (K_B_J_PER_K * T_K))
            warnings.append("using capillary_from_barrier_radius diagnostic fallback")
            return ZeldovichResult(
                value=z_value,
                source="capillary_from_barrier_radius",
                warnings=warnings,
                profile_source=profile_source,
                curvature_sign="negative",
            )
    const_value = fallback_value if fallback_value is not None and fallback_value > 0.0 else 1.0e9
    warnings.append("using user_constant diagnostic fallback")
    return ZeldovichResult(
        value=const_value,
        source="user_constant",
        warnings=warnings,
        profile_source=profile_source,
    )


def compute_xB_eq_regular_solution(T_K: float) -> float:
    L_val = 41212.9 - 18.05 * T_K
    RT = R_GAS_J_PER_MOLK * T_K
    x = math.exp(-L_val / RT)
    x = max(1.0e-9, min(0.5, x))
    for _ in range(50):
        one_minus_x = 1.0 - x
        f = RT * math.log(x) + L_val * one_minus_x * one_minus_x
        df = RT / x - 2.0 * L_val * one_minus_x
        if abs(df) < 1.0e-30:
            break
        step = f / df
        x_new = max(1.0e-10, min(0.999, x - step))
        if abs(x_new - x) < 1.0e-14:
            x = x_new
            break
        x = x_new
    return x


def infer_xB_eq(case: CaseData) -> tuple[float | None, str, list[str]]:
    warnings: list[str] = []
    candidates = [
        ("pf_params:ic_xB_eq_matrix", safe_float(case.params.get("ic_xB_eq_matrix"))),
        ("physical_inputs:xB_eq", safe_float(case.physical_inputs.get("xB_eq"))),
    ]
    for source, value in candidates:
        if value is not None and value > 0.0:
            return value, source, warnings
    for source, row in get_row_candidates(case):
        value, raw_key = extract_by_keys(row, ("xB_eq", "x_eq", "xag2te_eq", "ic_xb_eq_matrix"))
        if value is not None and value > 0.0:
            return value, f"{source}:{raw_key}", warnings
    T_K = infer_temperature_k(case)
    if T_K is None:
        warnings.append("missing temperature; cannot compute xB_eq from regular solution")
        return None, "", warnings
    return compute_xB_eq_regular_solution(T_K), "regular_solution", warnings


def infer_volume_data(case: CaseData) -> tuple[float | None, float | None, str, list[str]]:
    warnings: list[str] = []
    params = case.params
    phys = case.physical_inputs
    vm_alpha_phys = safe_float(phys.get("Vm_alpha_0"))
    source_alpha = ""
    if vm_alpha_phys is not None and vm_alpha_phys > 0.0:
        source_alpha = f"{case.physical_inputs_path}:Vm_alpha_0"
    else:
        vm_alpha_phys = safe_float(params.get("Vm_alpha_0_phys_m3mol"))
        if vm_alpha_phys is not None and vm_alpha_phys > 0.0:
            source_alpha = f"{case.params_path}:Vm_alpha_0_phys_m3mol"
    vm_comp_phys = safe_float(phys.get("Vm_compound"))
    source_comp = ""
    if vm_comp_phys is not None and vm_comp_phys > 0.0:
        source_comp = f"{case.physical_inputs_path}:Vm_compound"
    raw_vm_comp = safe_float(params.get("Vm_compound"))
    raw_vm_alpha = safe_float(params.get("Vm_alpha_0"))
    if vm_alpha_phys is None and raw_vm_alpha is not None and raw_vm_alpha > 1.0e-8:
        if raw_vm_alpha <= 10.0 and vm_alpha_phys is None:
            warnings.append("pf_input.params Vm_alpha_0 appears dimensionless; waiting for physical Vm_alpha_0 from physical_inputs/logs")
        else:
            vm_alpha_phys = raw_vm_alpha
            source_alpha = f"{case.params_path}:Vm_alpha_0"
            warnings.append("Vm_alpha_0 taken directly from pf_input.params; verify that it is already in m^3/mol")
    if vm_comp_phys is None and raw_vm_comp is not None and raw_vm_comp > 1.0e-8:
        if vm_alpha_phys is not None and raw_vm_comp <= 10.0:
            vm_comp_phys = raw_vm_comp * vm_alpha_phys
            source_comp = f"{case.params_path}:Vm_compound * {source_alpha}"
            warnings.append("Vm_compound interpreted as dimensionless ratio from pf_input.params and converted with Vm_alpha_0_phys")
        elif raw_vm_comp > 1.0e-8 and raw_vm_comp < 1.0e-3:
            vm_comp_phys = raw_vm_comp
            source_comp = f"{case.params_path}:Vm_compound"
            warnings.append("Vm_compound taken directly from pf_input.params; verify that it is already in m^3/mol")
    if vm_comp_phys is None and vm_alpha_phys is not None:
        vm_comp_phys = vm_alpha_phys
        source_comp = source_alpha
        warnings.append("Vm_compound missing; fell back to Vm_alpha")
    volume_source = "; ".join(part for part in (source_alpha, source_comp) if part)
    return vm_alpha_phys, vm_comp_phys, volume_source, warnings


def infer_vm_alpha_at_x(case: CaseData, xB_loc: float, vm_alpha_0: float) -> tuple[float, list[str]]:
    warnings: list[str] = []
    raw = safe_float(case.params.get("dVm_alpha_dxB"))
    if raw is None:
        return vm_alpha_0, warnings
    if abs(raw) < 1.0e-2:
        return vm_alpha_0 * (1.0 + raw * xB_loc), warnings
    if abs(raw) < 1.0e-3:
        warnings.append("dVm_alpha_dxB interpretation is ambiguous; using direct additive physical form")
    return vm_alpha_0 + raw * xB_loc, warnings


def compute_arrhenius_ag_diffusivity(T_K: float) -> tuple[float, str]:
    d0_cm2_s = 4.251e-11
    q_j_mol = 3.403e4
    d_cm2_s = d0_cm2_s * math.exp(-q_j_mol / (R_GAS_J_PER_MOLK * T_K))
    return d_cm2_s * 1.0e-4, "Unit_Psedobinary.py:D_Ag_in_PbTe_m2_per_s"


def compute_diffusivity(case: CaseData, mode: str) -> tuple[float | None, str, list[str]]:
    warnings: list[str] = []
    T_K = infer_temperature_k(case)
    if T_K is None:
        warnings.append("missing temperature; cannot compute diffusivity")
        return None, "", warnings
    if mode == "arrhenius_ag":
        value, source = compute_arrhenius_ag_diffusivity(T_K)
        return value, source, warnings
    t_real_unit = safe_float(case.params.get("t_real_unit"))
    d_alpha_code = safe_float(case.params.get("D_alpha"))
    phys_dx_ref = safe_float(case.physical_inputs.get("phys_dx_ref"))
    if d_alpha_code is not None and t_real_unit is not None and phys_dx_ref is not None:
        if d_alpha_code > 0.0 and t_real_unit > 0.0 and phys_dx_ref > 0.0:
            d_phys = d_alpha_code * phys_dx_ref * phys_dx_ref / t_real_unit
            return d_phys, "pf_input.params:D_alpha with physical_inputs:phys_dx_ref,t_real_unit", warnings
    metadata_json = case.representative_dir / "generated_pf.json"
    if metadata_json.exists():
        try:
            payload = json.loads(metadata_json.read_text(encoding="utf-8"))
            value = safe_float(
                (((payload.get("metadata") or {}).get("diffusivity_physical") or {}).get("D_alpha_m2_per_s"))
            )
            if value is not None and value > 0.0:
                return value, f"{metadata_json}:metadata.diffusivity_physical.D_alpha_m2_per_s", warnings
        except (OSError, json.JSONDecodeError):
            pass
    warnings.append("cuda_like diffusivity could not be reconstructed from code-unit parameters")
    warnings.append("TODO: add D_mix/Gamma-based local mobility extraction when spatial-field mode is added")
    return None, "", warnings


def compute_theta(args: argparse.Namespace) -> tuple[float | None, list[str]]:
    warnings: list[str] = []
    if args.theta == "steady":
        return 1.0, warnings
    if args.t_eval is None:
        warnings.append("theta=transient requires --t-eval in s")
        return None, warnings
    if args.tau_inc is None or args.tau_inc <= 0.0:
        warnings.append("theta=transient requires positive --tau-inc in s")
        return None, warnings
    if args.t_eval <= args.t_act:
        return 0.0, warnings
    theta = 1.0 - math.exp(-(args.t_eval - args.t_act) / args.tau_inc)
    return theta, warnings


def infer_delta_v_nuc(case: CaseData, arg_value: float | None) -> tuple[float | None, str, list[str]]:
    warnings: list[str] = []
    if arg_value is not None:
        return arg_value, "cli:--delta-v-nuc", warnings
    dx_m, _ = infer_phys_dx_m(case, "x")
    dy_m, _ = infer_phys_dx_m(case, "y")
    dz_m, _ = infer_phys_dx_m(case, "z")
    if dx_m is not None and dy_m is not None and dz_m is not None:
        warnings.append("DeltaV_nuc defaulted to one physical voxel volume")
        return dx_m * dy_m * dz_m, "derived:physical_voxel_volume", warnings
    warnings.append("missing physical voxel size; cannot derive DeltaV_nuc")
    return None, "", warnings


def infer_dt_phys(case: CaseData, arg_value: float | None) -> tuple[float | None, str, list[str]]:
    warnings: list[str] = []
    if arg_value is not None:
        return arg_value, "cli:--dt-phys", warnings
    dt_code = safe_float(case.params.get("dt"))
    t_real_unit = safe_float(case.params.get("t_real_unit"))
    if dt_code is not None and t_real_unit is not None and dt_code > 0.0 and t_real_unit > 0.0:
        return dt_code * t_real_unit, "pf_input.params:dt * t_real_unit", warnings
    warnings.append("missing dt or t_real_unit; cannot derive physical dt")
    return None, "", warnings


def compute_log_rate_terms(
    n_site: float | None,
    z_r: float | None,
    beta: float | None,
    barrier_j: float | None,
    theta: float | None,
    temp_k: float | None,
) -> tuple[float | None, float | None, float | None, float | None, float | None, float | None]:
    if None in (n_site, z_r, beta, barrier_j, theta, temp_k):
        return None, None, None, None, None, None
    if n_site <= 0.0 or z_r <= 0.0 or beta <= 0.0 or temp_k <= 0.0:
        return None, None, None, None, None, None
    if theta <= 0.0:
        ln_prefactor = math.log(n_site) + math.log(z_r) + math.log(beta)
        barrier_over_kbt = barrier_j / (K_B_J_PER_K * temp_k)
        return ln_prefactor, barrier_over_kbt, float("-inf"), ln_prefactor / LOG10, -barrier_over_kbt / LOG10, float("-inf")
    ln_prefactor = math.log(n_site) + math.log(z_r) + math.log(beta) + math.log(theta)
    barrier_over_kbt = barrier_j / (K_B_J_PER_K * temp_k)
    ln_j = ln_prefactor - barrier_over_kbt
    prefactor_log10 = ln_prefactor / LOG10
    barrier_penalty_log10 = -barrier_over_kbt / LOG10
    log10_j = ln_j / LOG10
    return ln_prefactor, barrier_over_kbt, ln_j, prefactor_log10, barrier_penalty_log10, log10_j


def maybe_exp_from_ln(ln_value: float | None) -> float | None:
    if ln_value is None:
        return None
    if ln_value <= -745.0:
        return 0.0
    if ln_value >= 709.0:
        return float("inf")
    return math.exp(ln_value)


def compute_poisson_outputs(j_value: float | None, log10_j: float | None, delta_v_nuc: float | None, dt_phys_s: float | None) -> tuple[float | None, float | None]:
    if log10_j is None or delta_v_nuc is None or dt_phys_s is None or delta_v_nuc <= 0.0 or dt_phys_s <= 0.0:
        return None, None
    log10_expected_events = log10_j + math.log10(delta_v_nuc) + math.log10(dt_phys_s)
    if j_value is None or j_value <= 0.0 or not math.isfinite(j_value):
        return None, log10_expected_events
    return -math.expm1(-j_value * delta_v_nuc * dt_phys_s), log10_expected_events


def convert_hat_delta_to_barrier(case: CaseData, delta_hat: float | None, reference_row: dict[str, Any] | None = None) -> tuple[float | None, dict[str, Any], list[str]]:
    warnings: list[str] = []
    row = reference_row if reference_row is not None else {}
    audit, audit_warnings = build_hat_energy_conversion(case, row)
    warnings.extend(audit_warnings)
    scale = audit.get("scale_j_m3")
    box_volume = audit.get("system_volume_m3")
    if delta_hat is None or scale is None or box_volume is None:
        return None, audit, warnings
    return delta_hat * scale * box_volume, audit, warnings


def build_reference_energy_audit(
    case: CaseData,
    all_cases: list[CaseData],
    barrier_data: BarrierData,
    barrier_j_used: float | None,
    barrier_kbt_used: float | None,
    barrier_ev_used: float | None,
    r_eff_m: float | None,
) -> tuple[ReferenceAuditRecord, ReferenceComparisonRecord, dict[str, Any], list[str]]:
    warnings: list[str] = []
    exx, eyy, ezz, exy = infer_strain_components(case)
    row_candidates = get_row_candidates(case)
    primary_row = row_candidates[0][1] if row_candidates else {}
    hat_audit, hat_warnings = build_hat_energy_conversion(case, primary_row)
    warnings.extend(hat_warnings)
    sibling_points, sibling_warnings = collect_sibling_energy_points(case)
    warnings.extend(sibling_warnings)

    explicit_ref_hat = None
    explicit_ref_source = ""
    explicit_ref_confidence = ""
    explicit_peak_excess = None
    explicit_peak_absolute = None
    for source, row in prioritize_summary_rows(row_candidates):
        value, _ = extract_by_keys(row, ("f_ref_hat_same_strain",))
        if value is not None:
            explicit_ref_hat = value
            explicit_ref_source = f"{source}:F_ref_hat_same_strain"
            explicit_ref_confidence = str(row.get("F_ref_confidence") or row.get("f_ref_confidence") or "high")
            explicit_peak_excess, _ = extract_by_keys(row, ("f_cnt_peak_hat_excess", "f_cnt_peak_hat_used"))
            explicit_peak_absolute, _ = extract_by_keys(row, ("f_cnt_peak_hat_absolute", "f_cnt_peak_hat"))
            break

    current_hat = barrier_data.raw_value if barrier_data.source_unit == "hat_density" else barrier_data.f_cnt_peak_hat
    if current_hat is None:
        current_hat = safe_float(barrier_data.all_conversion_factors.get("current_hat_value")) if barrier_data.all_conversion_factors else None
    f_saddle_hat = None
    peak_definition = "unknown"
    if sibling_points:
        saddle = max(sibling_points, key=lambda point: point.F_total_CNT_hat if point.F_total_CNT_hat is not None else -math.inf)
        f_saddle_hat = saddle.F_total_CNT_hat
        peak_definition = "absolute peak of last-row F_total_CNT_hat across sibling radius cases; no same-strain reference subtraction detected in summarize_cnt_scan_from_guide.py"
        if current_hat is None:
            current_hat = f_saddle_hat
    same_ref_point, same_ref_source, confidence, ref_warnings = select_reference_point(sibling_points)
    warnings.extend(ref_warnings)
    same_ref_hat = same_ref_point.F_total_CNT_hat if same_ref_point is not None else None
    if explicit_ref_hat is not None:
        same_ref_hat = explicit_ref_hat
        same_ref_source = explicit_ref_source
        confidence = explicit_ref_confidence or "high"
    delta_hat_current = explicit_peak_absolute if explicit_peak_absolute is not None else (f_saddle_hat if f_saddle_hat is not None else current_hat)
    delta_hat_corrected = None
    if explicit_peak_excess is not None:
        delta_hat_corrected = explicit_peak_excess
    elif f_saddle_hat is not None and same_ref_hat is not None:
        delta_hat_corrected = f_saddle_hat - same_ref_hat
        if delta_hat_corrected < 0.0:
            warnings.append("same-strain corrected delta_hat is negative; marking corrected barrier unavailable")
            delta_hat_corrected = None
    unstrained_ref_hat = None
    delta_hat_unstrained = None
    unstrained_case = find_unstrained_reference_case(case, all_cases)
    if unstrained_case is not None:
        unstrained_points, uw = collect_sibling_energy_points(unstrained_case)
        warnings.extend(uw)
        unstrained_ref_point, _u_source, _u_conf, uw2 = select_reference_point(unstrained_points)
        warnings.extend(uw2)
        if unstrained_ref_point is not None:
            unstrained_ref_hat = unstrained_ref_point.F_total_CNT_hat
            if f_saddle_hat is not None:
                delta_hat_unstrained = f_saddle_hat - unstrained_ref_hat
    else:
        warnings.append("missing unstrained reference case for comparison")

    barrier_j_corrected = None
    barrier_kbt_corrected = None
    barrier_ev_corrected = None
    barrier_j_unstrained = None
    barrier_kbt_unstrained = None
    barrier_j_absolute = barrier_j_used
    barrier_kbt_absolute = barrier_kbt_used
    barrier_ev_absolute = barrier_ev_used
    if delta_hat_current is not None:
        barrier_j_absolute, _audit_unused0, conv_warn0 = convert_hat_delta_to_barrier(case, delta_hat_current, primary_row)
        warnings.extend(conv_warn0)
    if delta_hat_corrected is not None:
        barrier_j_corrected, _audit_unused, conv_warn = convert_hat_delta_to_barrier(case, delta_hat_corrected, primary_row)
        warnings.extend(conv_warn)
    if delta_hat_unstrained is not None:
        barrier_j_unstrained, _audit_unused2, conv_warn2 = convert_hat_delta_to_barrier(case, delta_hat_unstrained, primary_row)
        warnings.extend(conv_warn2)
    temp_k = infer_temperature_k(case)
    if temp_k is not None and barrier_j_corrected is not None:
        barrier_kbt_corrected = barrier_j_corrected / (K_B_J_PER_K * temp_k)
        barrier_ev_corrected = barrier_j_corrected / EV_TO_J
    if temp_k is not None and barrier_j_absolute is not None:
        barrier_kbt_absolute = barrier_j_absolute / (K_B_J_PER_K * temp_k)
        barrier_ev_absolute = barrier_j_absolute / EV_TO_J
    if temp_k is not None and barrier_j_unstrained is not None:
        barrier_kbt_unstrained = barrier_j_unstrained / (K_B_J_PER_K * temp_k)
    background_elastic_hat = same_ref_point.F_el_hat if same_ref_point is not None else None
    background_elastic_j = None
    if background_elastic_hat is not None:
        background_elastic_j, _audit_unused3, conv_warn3 = convert_hat_delta_to_barrier(case, background_elastic_hat, primary_row)
        warnings.extend(conv_warn3)
    background_fraction = None
    if background_elastic_hat is not None and delta_hat_current is not None and delta_hat_current > 0.0:
        background_fraction = background_elastic_hat / delta_hat_current
    recommended_barrier = barrier_j_corrected if barrier_j_corrected is not None else barrier_j_absolute
    recommended_barrier_kbt = barrier_kbt_corrected if barrier_kbt_corrected is not None else barrier_kbt_absolute
    recommended_reference = same_ref_source if barrier_j_corrected is not None else "current_absolute_peak"
    if same_ref_hat is None:
        confidence = "missing_same_strain_reference"
        warnings.append("missing_same_strain_reference")
    if same_ref_point is not None and not same_ref_source.startswith("explicit_volume_zero") and explicit_ref_hat is None:
        warnings.append("same-strain reference is approximate, not explicit matrix-only")

    record = ReferenceAuditRecord(
        case_dir=str(case.representative_dir),
        T_input=infer_temperature_c(case),
        T_K=temp_k,
        xB0=infer_xb_loc(case),
        mode=infer_mode(case),
        exx=exx,
        eyy=eyy,
        ezz=ezz,
        exy=exy,
        Nx=hat_audit.get("Nx"),
        Ny=hat_audit.get("Ny"),
        Nz=hat_audit.get("Nz"),
        dx_nm=hat_audit.get("dx_nm"),
        V_box_m3=hat_audit.get("system_volume_m3"),
        w_phys_J_m3=hat_audit.get("scale_j_m3"),
        F_CNT_peak_hat_current=delta_hat_current,
        F_CNT_peak_definition_inferred=peak_definition,
        F_saddle_hat=f_saddle_hat,
        F_reference_hat_same_strain=same_ref_hat,
        F_reference_source=same_ref_source,
        F_reference_hat_unstrained_if_available=unstrained_ref_hat,
        DeltaF_hat_current=delta_hat_current,
        DeltaF_hat_corrected_same_strain=delta_hat_corrected,
        DeltaF_hat_using_unstrained_reference_if_available=delta_hat_unstrained,
        barrier_J_current=barrier_j_absolute,
        barrier_kBT_current=barrier_kbt_absolute,
        barrier_eV_current=barrier_ev_absolute,
        barrier_J_corrected_same_strain=barrier_j_corrected,
        barrier_kBT_corrected_same_strain=barrier_kbt_corrected,
        barrier_eV_corrected_same_strain=barrier_ev_corrected,
        barrier_J_unstrained_reference_if_available=barrier_j_unstrained,
        barrier_kBT_unstrained_reference_if_available=barrier_kbt_unstrained,
        background_elastic_hat_same_strain=background_elastic_hat,
        background_elastic_J_same_strain=background_elastic_j,
        background_fraction_of_current_barrier=background_fraction,
        recommended_barrier=recommended_barrier,
        recommended_barrier_kBT=recommended_barrier_kbt,
        recommended_reference=recommended_reference,
        confidence=confidence,
        warnings=" | ".join(sorted(dict.fromkeys(warnings))),
    )
    comparison = ReferenceComparisonRecord(
        case_dir=str(case.representative_dir),
        base_case_tag=case.base_case_tag,
        current_barrier_kBT=barrier_kbt_absolute,
        corrected_same_strain_barrier_kBT=barrier_kbt_corrected,
        difference_kBT=(barrier_kbt_absolute - barrier_kbt_corrected) if barrier_kbt_absolute is not None and barrier_kbt_corrected is not None else None,
        ratio_current_to_corrected=(barrier_kbt_absolute / barrier_kbt_corrected) if barrier_kbt_absolute is not None and barrier_kbt_corrected not in (None, 0.0) else None,
        background_fraction=background_fraction,
        status=("same_strain_reference_found" if same_ref_hat is not None else "missing_same_strain_reference"),
    )
    extras = {
        "barrier_j_corrected": barrier_j_corrected,
        "barrier_kbt_corrected": barrier_kbt_corrected,
        "barrier_j_absolute": barrier_j_absolute,
        "barrier_kbt_absolute": barrier_kbt_absolute,
        "barrier_j_used": barrier_j_used,
        "barrier_kbt_used": barrier_kbt_used,
        "same_strain_reference_found": 1 if same_ref_hat is not None else 0,
        "same_strain_reference_source": same_ref_source,
        "same_strain_reference_confidence": confidence,
        "background_fraction_of_current_barrier": background_fraction,
    }
    return record, comparison, extras, warnings


def compute_rate(case: CaseData, all_cases: list[CaseData], args: argparse.Namespace, profile_files: list[Path]) -> tuple[RateResult, BarrierAuditRecord, ReferenceAuditRecord, ReferenceComparisonRecord]:
    warnings = list(case.warnings)
    strict_reasons: list[str] = []
    diagnostic_reasons: list[str] = []

    temp_c = infer_temperature_c(case)
    temp_k = infer_temperature_k(case)
    xb_loc = infer_xb_loc(case)
    mode = infer_mode(case)
    exx, eyy, ezz, exy = infer_strain_components(case)
    barrier_data = load_barrier_data(case)
    warnings.extend(barrier_data.warnings)
    barrier_j = barrier_data.barrier_j
    barrier_source = barrier_data.source
    r_eff_m, r_eff_source, r_eff_warnings = load_r_eff_star(case)
    warnings.extend(r_eff_warnings)
    xB_eq, xB_eq_source, xeq_warnings = infer_xB_eq(case)
    warnings.extend(xeq_warnings)
    vm_alpha_0, vm_comp, volume_source, volume_warnings = infer_volume_data(case)
    warnings.extend(volume_warnings)
    theta, theta_warnings = compute_theta(args)
    warnings.extend(theta_warnings)
    d_b, d_source, d_warnings = compute_diffusivity(case, args.diffusivity_mode)
    warnings.extend(d_warnings)
    delta_v_nuc, _delta_v_source, dv_warnings = infer_delta_v_nuc(case, args.delta_v_nuc)
    warnings.extend(dv_warnings)
    dt_phys_s, _dt_source, dt_warnings = infer_dt_phys(case, args.dt_phys)
    warnings.extend(dt_warnings)

    omega_site = None
    omega_g = None
    n_site = None
    beta = None
    s_ratio = None
    delta_g_ev = None
    delta_g_kbt = None
    profile_source = ""

    if temp_k is None:
        warnings.append("missing temperature_C")
        strict_reasons.append("missing_temperature")
        diagnostic_reasons.append("missing_temperature")
    if xb_loc is None:
        warnings.append("missing xB_loc / xB_out")
        strict_reasons.append("missing_xB_loc")
        diagnostic_reasons.append("missing_xB_loc")
    if xB_eq is not None and xb_loc is not None and xB_eq > 0.0:
        s_ratio = xb_loc / xB_eq
    if barrier_j is None:
        warnings.append("missing explicit DeltaG_star with clear units")
        strict_reasons.append("missing_barrier")
        diagnostic_reasons.append("missing_barrier")
    if r_eff_m is None:
        warnings.append("missing r_eff_star")
        strict_reasons.append("missing_r_eff_star")
        diagnostic_reasons.append("missing_r_eff_star")
    if vm_alpha_0 is None:
        warnings.append("missing Vm_alpha physical data")
        strict_reasons.append("missing_Vm_alpha")
        diagnostic_reasons.append("missing_Vm_alpha")
    if vm_comp is None:
        warnings.append("missing Vm_compound physical data")
        strict_reasons.append("missing_Vm_compound")
        diagnostic_reasons.append("missing_Vm_compound")
    if theta is None:
        strict_reasons.append("missing_theta")
        diagnostic_reasons.append("missing_theta")
    if temp_k is not None and barrier_j is not None:
        delta_g_ev = barrier_j / EV_TO_J
        delta_g_kbt = barrier_j / (K_B_J_PER_K * temp_k)
    if vm_alpha_0 is not None and xb_loc is not None:
        vm_alpha_loc, vm_alpha_warnings = infer_vm_alpha_at_x(case, xb_loc, vm_alpha_0)
        warnings.extend(vm_alpha_warnings)
        if vm_alpha_loc > 0.0:
            omega_site = vm_alpha_loc / N_A_PER_MOL
            n_site = 1.0 / omega_site
        else:
            warnings.append("Vm_alpha(xB_loc) is non-positive")
            strict_reasons.append("nonpositive_Omega_site")
            diagnostic_reasons.append("nonpositive_Omega_site")
    if vm_comp is not None and vm_comp > 0.0:
        omega_g = vm_comp / N_A_PER_MOL
    elif vm_comp is not None:
        strict_reasons.append("nonpositive_Omega_g")
        diagnostic_reasons.append("nonpositive_Omega_g")
    if r_eff_m is not None and d_b is not None and omega_g is not None and xb_loc is not None:
        beta = 4.0 * math.pi * r_eff_m * d_b * xb_loc / omega_g
        if beta <= 0.0:
            warnings.append("beta_r_star is non-positive")
            strict_reasons.append("nonpositive_beta")
            diagnostic_reasons.append("nonpositive_beta")

    strict_z = compute_strict_zeldovich(
        case=case,
        profile_files=profile_files,
        r_eff_star_m=r_eff_m,
        fit_points=args.fit_points,
        allow_fallback=args.allow_zeldovich_fallback,
        fallback_value=args.zeldovich_fallback_value,
        barrier_j=barrier_j,
    )
    warnings.extend(strict_z.warnings)
    profile_source = strict_z.profile_source
    if strict_z.value is None:
        strict_reasons.append("unstable_zeldovich_curvature")

    diagnostic_z = compute_diagnostic_zeldovich(
        case=case,
        profile_files=profile_files,
        r_eff_star_m=r_eff_m,
        barrier_j=barrier_j,
        strict_result=strict_z,
        fallback_value=args.zeldovich_fallback_value,
    )
    warnings.extend(diagnostic_z.warnings)
    if not profile_source:
        profile_source = diagnostic_z.profile_source
    if diagnostic_z.value is None:
        diagnostic_reasons.append("missing_diagnostic_zeldovich")

    reference_audit, reference_comparison, reference_extras, reference_warnings = build_reference_energy_audit(
        case=case,
        all_cases=all_cases,
        barrier_data=barrier_data,
        barrier_j_used=barrier_j,
        barrier_kbt_used=delta_g_kbt,
        barrier_ev_used=delta_g_ev,
        r_eff_m=r_eff_m,
    )
    warnings.extend(reference_warnings)
    barrier_j_absolute = reference_extras.get("barrier_j_absolute")
    barrier_kbt_absolute = reference_extras.get("barrier_kbt_absolute")
    barrier_j_corrected = reference_extras.get("barrier_j_corrected")
    barrier_kbt_corrected = reference_extras.get("barrier_kbt_corrected")

    ln_prefactor = None
    barrier_over_kbt = None
    ln_j_diagnostic = None
    prefactor_log10 = None
    barrier_penalty_log10 = None
    log10_j_diagnostic = None
    j_diagnostic = None
    p_event_diagnostic = None
    log10_expected_events_diagnostic = None

    if diagnostic_z.value is not None:
        (
            ln_prefactor,
            barrier_over_kbt,
            ln_j_diagnostic,
            prefactor_log10,
            barrier_penalty_log10,
            log10_j_diagnostic,
        ) = compute_log_rate_terms(n_site, diagnostic_z.value, beta, barrier_j, theta, temp_k)
        j_diagnostic = maybe_exp_from_ln(ln_j_diagnostic)
        p_event_diagnostic, log10_expected_events_diagnostic = compute_poisson_outputs(
            j_diagnostic,
            log10_j_diagnostic,
            delta_v_nuc,
            dt_phys_s,
        )
        if log10_j_diagnostic is None:
            diagnostic_reasons.append("diagnostic_log_rate_failed")
    diagnostic_z_current_value = diagnostic_z.value
    if diagnostic_z.source == "capillary_from_barrier_radius" and barrier_j_absolute is not None and r_eff_m is not None and temp_k is not None and r_eff_m > 0.0:
        gamma_eff_abs = 3.0 * barrier_j_absolute / (4.0 * math.pi * r_eff_m * r_eff_m)
        if gamma_eff_abs > 0.0:
            diagnostic_z_current_value = math.sqrt(4.0 * gamma_eff_abs / (K_B_J_PER_K * temp_k))
    j_diagnostic_current = None
    log10_j_diagnostic_current = None
    p_event_diagnostic_current = None
    log10_expected_events_diagnostic_current = None
    if diagnostic_z_current_value is not None and barrier_j_absolute is not None:
        (
            _ln_pref_current,
            _barrier_over_kbt_current,
            ln_j_current,
            _prefactor_log10_current,
            _barrier_penalty_log10_current,
            log10_j_diagnostic_current,
        ) = compute_log_rate_terms(n_site, diagnostic_z_current_value, beta, barrier_j_absolute, theta, temp_k)
        j_diagnostic_current = maybe_exp_from_ln(ln_j_current)
        p_event_diagnostic_current, log10_expected_events_diagnostic_current = compute_poisson_outputs(
            j_diagnostic_current,
            log10_j_diagnostic_current,
            delta_v_nuc,
            dt_phys_s,
        )

    j_diagnostic_corrected = None
    log10_j_diagnostic_corrected = None
    p_event_diagnostic_corrected = None
    log10_expected_events_diagnostic_corrected = None
    barrier_over_kbt_corrected = None
    ln_j_corrected = None
    barrier_penalty_log10_corrected = None
    diagnostic_z_corrected_value = diagnostic_z.value
    if diagnostic_z.source == "capillary_from_barrier_radius" and barrier_j_corrected is not None and r_eff_m is not None and temp_k is not None and r_eff_m > 0.0:
        gamma_eff_corr = 3.0 * barrier_j_corrected / (4.0 * math.pi * r_eff_m * r_eff_m)
        if gamma_eff_corr > 0.0:
            diagnostic_z_corrected_value = math.sqrt(4.0 * gamma_eff_corr / (K_B_J_PER_K * temp_k))
    if diagnostic_z_corrected_value is not None and barrier_j_corrected is not None:
        (
            _ln_pref_unused,
            barrier_over_kbt_corrected,
            ln_j_corrected,
            _prefactor_log10_unused,
            barrier_penalty_log10_corrected,
            log10_j_diagnostic_corrected,
        ) = compute_log_rate_terms(n_site, diagnostic_z_corrected_value, beta, barrier_j_corrected, theta, temp_k)
        j_diagnostic_corrected = maybe_exp_from_ln(ln_j_corrected)
        p_event_diagnostic_corrected, log10_expected_events_diagnostic_corrected = compute_poisson_outputs(
            j_diagnostic_corrected,
            log10_j_diagnostic_corrected,
            delta_v_nuc,
            dt_phys_s,
        )
    strict_ln_j = None
    log10_j_strict = None
    j_strict = None
    p_event_strict = None
    log10_expected_events_strict = None
    if strict_z.value is not None:
        _, _, strict_ln_j, _, _, log10_j_strict = compute_log_rate_terms(
            n_site, strict_z.value, beta, barrier_j, theta, temp_k
        )
        j_strict = maybe_exp_from_ln(strict_ln_j)
        p_event_strict, log10_expected_events_strict = compute_poisson_outputs(
            j_strict,
            log10_j_strict,
            delta_v_nuc,
            dt_phys_s,
        )
        if log10_j_strict is None:
            strict_reasons.append("strict_log_rate_failed")

    strict_status = "complete" if not strict_reasons else "incomplete"
    diagnostic_status = "complete" if not diagnostic_reasons and log10_j_diagnostic is not None else "incomplete"
    case_status = strict_status

    result = RateResult(
        case_dir=str(case.representative_dir),
        base_case_tag=case.base_case_tag,
        grouped_case_count=len(case.case_dirs),
        T_input=temp_c,
        T_K=temp_k,
        xB_loc=xb_loc,
        xB_eq=xB_eq,
        xB_eq_source=xB_eq_source,
        S_ratio=s_ratio,
        mode=mode,
        exx=exx,
        eyy=eyy,
        ezz=ezz,
        exy=exy,
        DeltaG_star_J=barrier_j,
        DeltaG_star_eV=delta_g_ev,
        DeltaG_star_kBT=delta_g_kbt,
        DeltaG_star_source=barrier_source,
        r_eff_star_m=r_eff_m,
        r_eff_star_nm=(r_eff_m * 1.0e9 if r_eff_m is not None else None),
        r_eff_star_source=r_eff_source,
        Vm_alpha_m3_mol=vm_alpha_0,
        Vm_compound_m3_mol=vm_comp,
        volume_sources=volume_source,
        Omega_site_m3=omega_site,
        Omega_g_m3=omega_g,
        D_B_m2_s=d_b,
        D_B_source=d_source,
        N_site_m3=n_site,
        Z_r_1_m=strict_z.value,
        Z_r_source=strict_z.source,
        beta_r_star_1_s=beta,
        Theta_tr=theta,
        J_m3_s=j_strict,
        log10_J_m3_s=log10_j_strict,
        DeltaV_nuc_m3=delta_v_nuc,
        dt_phys_s=dt_phys_s,
        P_event=p_event_strict,
        case_status=case_status,
        barrier_metadata_source=barrier_source,
        profile_source=profile_source,
        log10_expected_events=log10_expected_events_strict,
        DeltaG_star_current=barrier_j_absolute,
        barrier_kBT_current=barrier_kbt_absolute,
        DeltaG_star_corrected=barrier_j_corrected,
        barrier_kBT_corrected=barrier_kbt_corrected,
        strict_status=strict_status,
        strict_incomplete_reason=";".join(sorted(dict.fromkeys(strict_reasons))),
        Z_r_strict=strict_z.value,
        Z_r_strict_source=strict_z.source,
        J_strict=j_strict,
        log10J_strict=log10_j_strict,
        Z_r_diagnostic=diagnostic_z.value,
        Z_r_diagnostic_source=diagnostic_z.source,
        Z_r_diagnostic_fit_R2=diagnostic_z.fit_r2,
        Z_r_diagnostic_window_points=diagnostic_z.window_points,
        Z_r_diagnostic_curvature_sign=diagnostic_z.curvature_sign,
        J_diagnostic=j_diagnostic,
        log10J_diagnostic=log10_j_diagnostic,
        P_event_diagnostic=p_event_diagnostic,
        log10_expected_events_diagnostic=log10_expected_events_diagnostic,
        J_diagnostic_current=j_diagnostic_current,
        log10J_diagnostic_current=log10_j_diagnostic_current,
        P_event_diagnostic_current=p_event_diagnostic_current,
        log10_expected_events_diagnostic_current=log10_expected_events_diagnostic_current,
        J_diagnostic_corrected=j_diagnostic_corrected,
        log10J_diagnostic_corrected=log10_j_diagnostic_corrected,
        P_event_diagnostic_corrected=p_event_diagnostic_corrected,
        log10_expected_events_diagnostic_corrected=log10_expected_events_diagnostic_corrected,
        ln_prefactor=ln_prefactor,
        barrier_over_kBT=barrier_over_kbt,
        lnJ=ln_j_diagnostic,
        prefactor_log10=prefactor_log10,
        barrier_penalty_log10=barrier_penalty_log10,
        barrier_over_kBT_corrected=barrier_over_kbt_corrected,
        lnJ_corrected=ln_j_corrected,
        barrier_penalty_log10_corrected=barrier_penalty_log10_corrected,
        same_strain_reference_found=reference_extras.get("same_strain_reference_found", 0),
        same_strain_reference_source=reference_extras.get("same_strain_reference_source", ""),
        same_strain_reference_confidence=reference_extras.get("same_strain_reference_confidence", ""),
        background_fraction_of_current_barrier=reference_extras.get("background_fraction_of_current_barrier"),
        warnings=sorted(dict.fromkeys(warnings)),
    )

    audit = BarrierAuditRecord(
        case_dir=str(case.representative_dir),
        T_input=temp_c,
        T_K=temp_k,
        xB0=xb_loc,
        mode=mode,
        strain_values=format_strain_values(exx, eyy, ezz, exy),
        F_CNT_peak_hat=barrier_data.f_cnt_peak_hat,
        F_total_CNT_hat_if_available=barrier_data.f_total_cnt_hat_if_available,
        barrier_J_current=barrier_j_absolute,
        barrier_eV_current=(barrier_j_absolute / EV_TO_J if barrier_j_absolute is not None else None),
        barrier_kBT_current=barrier_kbt_absolute,
        r_eff_star_nm=(r_eff_m * 1.0e9 if r_eff_m is not None else None),
        voxel_count=barrier_data.voxel_count,
        spacing_sim_units=barrier_data.spacing_sim_units,
        dx_nm_or_spacing_physical=barrier_data.dx_nm_or_spacing_physical,
        volume_from_voxels_m3=barrier_data.volume_from_voxels_m3,
        mu_reference_if_used=barrier_data.mu_reference_if_used,
        c_tot_if_used=barrier_data.c_tot_if_used,
        Vm_alpha=vm_alpha_0,
        Vm_compound=vm_comp,
        conversion_formula_used=barrier_data.conversion_formula_used,
        all_conversion_factors=json_compact(barrier_data.all_conversion_factors),
        warnings=" | ".join(sorted(dict.fromkeys(barrier_data.warnings + r_eff_warnings))),
    )
    result.case_status = strict_status
    if diagnostic_status == "incomplete" and "diagnostic_incomplete" not in result.warnings:
        result.warnings.append(f"diagnostic_status={diagnostic_status}:{';'.join(sorted(dict.fromkeys(diagnostic_reasons)))}")
    return result, audit, reference_audit, reference_comparison


def maybe_git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def make_plots(results: list[RateResult], out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def group_label(item: RateResult) -> str:
        t_txt = "T=NA" if item.T_input is None else f"T={item.T_input:.0f} C"
        return f"{t_txt}, {item.mode}"

    def line_plot(
        filename: str,
        title: str,
        x_attr: str,
        y_attr: str,
        xlabel: str,
        ylabel: str,
    ) -> None:
        valid = [
            item for item in results
            if getattr(item, x_attr) is not None and getattr(item, y_attr) is not None
            and math.isfinite(float(getattr(item, x_attr)))
            and math.isfinite(float(getattr(item, y_attr)))
        ]
        if not valid:
            return
        fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
        grouped: dict[str, list[RateResult]] = {}
        for item in valid:
            grouped.setdefault(group_label(item), []).append(item)
        for label, items in sorted(grouped.items()):
            items_sorted = sorted(items, key=lambda item: float(getattr(item, x_attr)))
            ax.plot(
                [float(getattr(item, x_attr)) for item in items_sorted],
                [float(getattr(item, y_attr)) for item in items_sorted],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                label=label,
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if len(grouped) > 1:
            ax.legend(fontsize=8)
        target = plot_dir / filename
        fig.savefig(target, dpi=180)
        plt.close(fig)
        paths.append(str(target))

    line_plot(
        "barrier_over_kBT_vs_S.png",
        "Barrier reduction dominates the change in nucleation rate",
        "S_ratio",
        "DeltaG_star_kBT",
        "Supersaturation ratio S = xB_loc / xB_eq",
        "DeltaG_star / (kB T)",
    )
    line_plot(
        "barrier_over_kBT_vs_xB.png",
        "Barrier term remains large across the scanned supersaturation range",
        "xB_loc",
        "DeltaG_star_kBT",
        "Local matrix composition xB_loc",
        "DeltaG_star / (kB T)",
    )

    valid = [
        item for item in results
        if item.r_eff_star_nm is not None and item.DeltaG_star_eV is not None
        and math.isfinite(item.r_eff_star_nm) and math.isfinite(item.DeltaG_star_eV)
    ]
    if valid:
        fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
        grouped: dict[str, list[RateResult]] = {}
        for item in valid:
            grouped.setdefault(group_label(item), []).append(item)
        for label, items in sorted(grouped.items()):
            ax.scatter(
                [float(item.r_eff_star_nm) for item in items],
                [float(item.DeltaG_star_eV) for item in items],
                s=36,
                label=label,
            )
        ax.set_title("Current converted barriers are extremely large for nanometer-scale nuclei")
        ax.set_xlabel("r_eff_star [nm]")
        ax.set_ylabel("DeltaG_star [eV]")
        ax.grid(True, alpha=0.25)
        if len(grouped) > 1:
            ax.legend(fontsize=8)
        target = plot_dir / "barrier_eV_vs_r_eff_star_nm.png"
        fig.savefig(target, dpi=180)
        plt.close(fig)
        paths.append(str(target))

    valid_diag = [
        item for item in results
        if item.DeltaG_star_kBT is not None and item.log10J_diagnostic is not None
        and math.isfinite(item.DeltaG_star_kBT) and math.isfinite(item.log10J_diagnostic)
    ]
    if valid_diag:
        fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
        grouped: dict[str, list[RateResult]] = {}
        for item in valid_diag:
            grouped.setdefault(group_label(item), []).append(item)
        for label, items in sorted(grouped.items()):
            ax.scatter(
                [float(item.DeltaG_star_kBT) for item in items],
                [float(item.log10J_diagnostic) for item in items],
                s=36,
                label=label,
            )
        ax.set_title("Diagnostic Zeldovich fallback does not rescue the rate if DeltaG*/kBT is too high")
        ax.set_xlabel("DeltaG_star / (kB T)")
        ax.set_ylabel("log10 J_diagnostic [1/(m^3 s)]")
        ax.grid(True, alpha=0.25)
        if len(grouped) > 1:
            ax.legend(fontsize=8)
        target = plot_dir / "log10J_diagnostic_vs_barrier_kBT.png"
        fig.savefig(target, dpi=180)
        plt.close(fig)
        paths.append(str(target))

    valid_pref = [
        item for item in results
        if item.prefactor_log10 is not None and item.barrier_penalty_log10 is not None
        and math.isfinite(item.prefactor_log10) and math.isfinite(item.barrier_penalty_log10)
    ]
    if valid_pref:
        fig, ax = plt.subplots(figsize=(10.0, 5.0), constrained_layout=True)
        xs = np.arange(len(valid_pref))
        ax.scatter(xs, [item.prefactor_log10 for item in valid_pref], label="prefactor_log10", s=30)
        ax.scatter(xs, [item.barrier_penalty_log10 for item in valid_pref], label="barrier_penalty_log10", s=30)
        ax.set_title("Barrier term dominates the predicted nucleation rate")
        ax.set_xlabel("Case index")
        ax.set_ylabel("log10 contribution")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([item.base_case_tag for item in valid_pref], rotation=70, ha="right", fontsize=7)
        target = plot_dir / "prefactor_vs_barrier_penalty.png"
        fig.savefig(target, dpi=180)
        plt.close(fig)
        paths.append(str(target))

    valid_corr = [
        item for item in results
        if item.barrier_kBT_current is not None and item.barrier_kBT_corrected is not None
        and math.isfinite(item.barrier_kBT_current) and math.isfinite(item.barrier_kBT_corrected)
    ]
    if valid_corr:
        fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
        xvals = [float(item.barrier_kBT_current) for item in valid_corr]
        yvals = [float(item.barrier_kBT_corrected) for item in valid_corr]
        ax.scatter(xvals, yvals, s=38)
        limit = max(xvals + yvals)
        ax.plot([0.0, limit], [0.0, limit], linestyle="--", linewidth=1.2, color="black", alpha=0.5)
        ax.set_title("Same-strain matrix reference subtraction changes the CNT barrier")
        ax.set_xlabel("Current barrier [kB T]")
        ax.set_ylabel("Corrected same-strain barrier [kB T]")
        ax.grid(True, alpha=0.25)
        target = plot_dir / "current_vs_corrected_barrier_kBT.png"
        fig.savefig(target, dpi=180)
        plt.close(fig)
        paths.append(str(target))

    valid_bg = [
        item for item in results
        if item.background_fraction_of_current_barrier is not None and math.isfinite(item.background_fraction_of_current_barrier)
    ]
    if valid_bg:
        fig, ax = plt.subplots(figsize=(10.0, 5.0), constrained_layout=True)
        xs = np.arange(len(valid_bg))
        ax.bar(xs, [float(item.background_fraction_of_current_barrier) for item in valid_bg], width=0.8)
        ax.set_title("External-strain background energy removed from barrier")
        ax.set_xlabel("Case index")
        ax.set_ylabel("background_fraction_of_current_barrier")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xticks(xs)
        ax.set_xticklabels([item.base_case_tag for item in valid_bg], rotation=70, ha="right", fontsize=7)
        target = plot_dir / "background_fraction_by_case.png"
        fig.savefig(target, dpi=180)
        plt.close(fig)
        paths.append(str(target))

    valid_corr_mode = [
        item for item in results
        if item.barrier_kBT_corrected is not None and math.isfinite(item.barrier_kBT_corrected)
    ]
    if valid_corr_mode:
        fig, ax = plt.subplots(figsize=(10.0, 5.0), constrained_layout=True)
        grouped: dict[str, list[RateResult]] = {}
        for item in valid_corr_mode:
            grouped.setdefault(item.mode, []).append(item)
        labels = sorted(grouped)
        data = [[float(item.barrier_kBT_corrected) for item in grouped[label]] for label in labels]
        ax.boxplot(data, tick_labels=labels)
        ax.set_title("Corrected nucleation barrier after subtracting strained matrix reference")
        ax.set_xlabel("Strain mode")
        ax.set_ylabel("Corrected barrier [kB T]")
        ax.grid(True, axis="y", alpha=0.25)
        target = plot_dir / "corrected_barrier_kBT_vs_mode.png"
        fig.savefig(target, dpi=180)
        plt.close(fig)
        paths.append(str(target))

    valid_corr_reff = [
        item for item in results
        if item.barrier_kBT_corrected is not None and item.r_eff_star_nm is not None
        and math.isfinite(item.barrier_kBT_corrected) and math.isfinite(item.r_eff_star_nm)
    ]
    if valid_corr_reff:
        fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
        ax.scatter(
            [float(item.r_eff_star_nm) for item in valid_corr_reff],
            [float(item.barrier_kBT_corrected) for item in valid_corr_reff],
            s=38,
        )
        ax.set_title("Excess nucleation barrier from same-strain matrix reference")
        ax.set_xlabel("r_eff_star [nm]")
        ax.set_ylabel("Corrected barrier [kB T]")
        ax.grid(True, alpha=0.25)
        target = plot_dir / "corrected_barrier_kBT_vs_r_eff_star_nm.png"
        fig.savefig(target, dpi=180)
        plt.close(fig)
        paths.append(str(target))

    valid_logj_compare = [
        item for item in results
        if item.log10J_diagnostic_current is not None and item.log10J_diagnostic_corrected is not None
        and math.isfinite(item.log10J_diagnostic_current) and math.isfinite(item.log10J_diagnostic_corrected)
    ]
    if valid_logj_compare:
        fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
        xvals = [float(item.log10J_diagnostic_current) for item in valid_logj_compare]
        yvals = [float(item.log10J_diagnostic_corrected) for item in valid_logj_compare]
        ax.scatter(xvals, yvals, s=38)
        lower = min(xvals + yvals)
        upper = max(xvals + yvals)
        ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.2, color="black", alpha=0.5)
        ax.set_title("Explicit nucleation rate after same-strain reference subtraction")
        ax.set_xlabel("log10 J_diagnostic using current barrier")
        ax.set_ylabel("log10 J_diagnostic using corrected barrier")
        ax.grid(True, alpha=0.25)
        target = plot_dir / "diagnostic_log10J_current_vs_corrected.png"
        fig.savefig(target, dpi=180)
        plt.close(fig)
        paths.append(str(target))
    return paths


def write_outputs(
    results: list[RateResult],
    audits: list[BarrierAuditRecord],
    reference_audits: list[ReferenceAuditRecord],
    reference_comparisons: list[ReferenceComparisonRecord],
    out_dir: Path,
    root: Path,
    args: argparse.Namespace,
    profile_files: list[Path],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "nucleation_rate_table.csv"
    rows = [result.to_row() for result in results]
    fieldnames: list[str] = list(DEFAULT_TABLE_FIELDS)
    seen: set[str] = set(fieldnames)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    diagnostic_path = out_dir / "nucleation_rate_table_diagnostic.csv"
    diagnostic_rows = [result.to_row() for result in results]
    with diagnostic_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIAGNOSTIC_TABLE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    audit_path = out_dir / "barrier_conversion_audit.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([audit.to_row() for audit in audits])

    reference_audit_path = out_dir / "reference_energy_audit.csv"
    with reference_audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REFERENCE_AUDIT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([audit.to_row() for audit in reference_audits])

    comparison_path = out_dir / "barrier_reference_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REFERENCE_COMPARISON_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([item.to_row() for item in reference_comparisons])

    plot_paths: list[str] = []
    if not args.no_plots:
        plot_paths = make_plots(results, out_dir)

    strict_completed = sum(1 for item in results if item.strict_status == "complete")
    diagnostic_completed = sum(1 for item in results if item.log10J_diagnostic is not None)
    same_strain_found = sum(1 for item in results if item.same_strain_reference_found)
    same_strain_missing = len(results) - same_strain_found
    absolute_barriers = [item.barrier_kBT_current for item in results if item.barrier_kBT_current is not None and math.isfinite(item.barrier_kBT_current)]
    excess_barriers = [item.barrier_kBT_corrected for item in results if item.barrier_kBT_corrected is not None and math.isfinite(item.barrier_kBT_corrected)]
    diag_logs = [item.log10J_diagnostic for item in results if item.log10J_diagnostic is not None and math.isfinite(item.log10J_diagnostic)]
    bg_fracs = [item.background_fraction_of_current_barrier for item in results if item.background_fraction_of_current_barrier is not None and math.isfinite(item.background_fraction_of_current_barrier)]
    pref_terms = [abs(item.prefactor_log10) for item in results if item.prefactor_log10 is not None and math.isfinite(item.prefactor_log10)]
    barrier_terms = [abs(item.barrier_penalty_log10) for item in results if item.barrier_penalty_log10 is not None and math.isfinite(item.barrier_penalty_log10)]
    if excess_barriers and absolute_barriers and np.median(excess_barriers) < 0.2 * np.median(absolute_barriers):
        dominant_issue = "missing background subtraction likely dominates current barrier"
    elif barrier_terms and pref_terms and np.median(barrier_terms) > np.median(pref_terms) + 2.0:
        dominant_issue = "barrier magnitude"
    elif strict_completed == 0 and diagnostic_completed > 0:
        dominant_issue = "strict Z_r extraction"
    else:
        dominant_issue = "mixed / unresolved"

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "git_commit": maybe_git_commit(root),
        "scan_root": str(root),
        "case_pattern": args.pattern,
        "params_glob": args.params_glob,
        "log_glob": args.log_glob,
        "profile_glob": args.profile_glob,
        "preferred_barrier_summaries": [name.strip() for name in args.barrier_summary.split(",") if name.strip()],
        "diffusivity_mode": args.diffusivity_mode,
        "theta_mode": args.theta,
        "tau_inc_s": args.tau_inc,
        "t_eval_s": args.t_eval,
        "t_act_s": args.t_act,
        "delta_v_nuc_m3_cli": args.delta_v_nuc,
        "dt_phys_s_cli": args.dt_phys,
        "allow_zeldovich_fallback": args.allow_zeldovich_fallback,
        "zeldovich_fallback_value": args.zeldovich_fallback_value if args.zeldovich_fallback_value is not None else 1.0e9,
        "fit_points": args.fit_points,
        "audit_reference_energy": args.audit_reference_energy,
        "diagnostic_note": "diagnostic_J is not a final physical nucleation rate. It is used only to test whether the barrier magnitude already suppresses nucleation.",
        "parameter_sources": {
            "thermo": "thermo_utils.h:get_L_param + xB_eq_from_temperature",
            "phase_fraction": "phase_functions.h:h_of_phi",
            "diffusivity": "Unit_Psedobinary.py:D_Ag_in_PbTe_m2_per_s",
            "time_scale": "Unit_Psedobinary.py:build_main_cuda_overrides -> t_real_unit",
            "barrier_hat_conversion": "main_cuda.cu:F_total_CNT_hat is box-averaged hat free-energy density; Unit_Psedobinary.py:w_phys=12*gamma/lambda_sm; barrier_J = F_hat * w_phys * V_box",
        },
        "unit_conversions": {
            "temperature": "T_K = T_C + 273.15",
            "diffusivity": "cm^2/s -> m^2/s by multiplying 1e-4",
            "radius": "nm -> m by multiplying 1e-9",
            "volume": "nm^3 -> m^3 by multiplying 1e-27",
            "barrier": {
                "eV_to_J": EV_TO_J,
                "kBT_to_J": "DeltaG_J = DeltaG_kBT * kB * T_K",
            },
        },
        "profile_file_count": len(profile_files),
        "completed_cases": strict_completed,
        "strict_completed_cases": strict_completed,
        "diagnostic_completed_cases": diagnostic_completed,
        "cases_with_same_strain_reference_found": same_strain_found,
        "cases_missing_same_strain_reference": same_strain_missing,
        "incomplete_cases": sum(1 for item in results if item.case_status != "complete"),
        "median_barrier_kBT_absolute": float(np.median(absolute_barriers)) if absolute_barriers else None,
        "median_barrier_kBT_excess": float(np.median(excess_barriers)) if excess_barriers else None,
        "min_barrier_kBT_absolute": float(np.min(absolute_barriers)) if absolute_barriers else None,
        "max_barrier_kBT_absolute": float(np.max(absolute_barriers)) if absolute_barriers else None,
        "min_barrier_kBT_excess": float(np.min(excess_barriers)) if excess_barriers else None,
        "max_barrier_kBT_excess": float(np.max(excess_barriers)) if excess_barriers else None,
        "median_background_fraction": float(np.median(bg_fracs)) if bg_fracs else None,
        "median_log10J_diagnostic": float(np.median(diag_logs)) if diag_logs else None,
        "dominant_issue": dominant_issue,
        "output_files": {
            "table_csv": str(csv_path),
            "diagnostic_table_csv": str(diagnostic_path),
            "barrier_audit_csv": str(audit_path),
            "reference_energy_audit_csv": str(reference_audit_path),
            "barrier_reference_comparison_csv": str(comparison_path),
            "metadata_json": str(out_dir / "nucleation_rate_metadata.json"),
            "plots": plot_paths,
        },
        "cases": [
            {
                "base_case_tag": item.base_case_tag,
                "case_dir": item.case_dir,
                "status": item.case_status,
                "strict_status": item.strict_status,
                "strict_incomplete_reason": item.strict_incomplete_reason,
                "diagnostic_z_source": item.Z_r_diagnostic_source,
                "barrier_source": item.DeltaG_star_source,
                "r_eff_source": item.r_eff_star_source,
                "z_source": item.Z_r_source,
                "diffusivity_source": item.D_B_source,
                "warnings": item.warnings,
            }
            for item in results
        ],
    }
    metadata_path = out_dir / "nucleation_rate_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve() if args.out else infer_default_output_dir(root)
    preferred_summaries = [name.strip() for name in args.barrier_summary.split(",") if name.strip()]
    global_rows = discover_global_rows(root, preferred_summaries)
    profile_files = discover_profile_files(root, args.profile_glob)
    cases = scan_case_dirs(
        root=root,
        pattern=args.pattern,
        params_glob=args.params_glob,
        log_glob=args.log_glob,
        global_rows=global_rows,
    )
    if not cases:
        print(f"[warn] no case directories matched pattern '{args.pattern}' under {root}", file=sys.stderr)
    computed = [compute_rate(case, cases, args, profile_files) for case in cases]
    results = [item[0] for item in computed]
    audits = [item[1] for item in computed]
    reference_audits = [item[2] for item in computed]
    reference_comparisons = [item[3] for item in computed]
    write_outputs(results, audits, reference_audits, reference_comparisons, out_dir, root, args, profile_files)
    absolute_barriers = [item.barrier_kBT_current for item in results if item.barrier_kBT_current is not None and math.isfinite(item.barrier_kBT_current)]
    excess_barriers = [item.barrier_kBT_corrected for item in results if item.barrier_kBT_corrected is not None and math.isfinite(item.barrier_kBT_corrected)]
    bg_fracs = [item.background_fraction_of_current_barrier for item in results if item.background_fraction_of_current_barrier is not None and math.isfinite(item.background_fraction_of_current_barrier)]
    diag_logs = [item.log10J_diagnostic for item in results if item.log10J_diagnostic is not None and math.isfinite(item.log10J_diagnostic)]
    pref_terms = [abs(item.prefactor_log10) for item in results if item.prefactor_log10 is not None and math.isfinite(item.prefactor_log10)]
    barrier_terms = [abs(item.barrier_penalty_log10) for item in results if item.barrier_penalty_log10 is not None and math.isfinite(item.barrier_penalty_log10)]
    if excess_barriers and absolute_barriers and np.median(excess_barriers) < 0.2 * np.median(absolute_barriers):
        dominant_issue = "missing background subtraction likely dominates current barrier"
    elif barrier_terms and pref_terms and np.median(barrier_terms) > np.median(pref_terms) + 2.0:
        dominant_issue = "barrier magnitude"
    elif sum(1 for item in results if item.strict_status == "complete") == 0 and sum(1 for item in results if item.log10J_diagnostic is not None) > 0:
        dominant_issue = "strict Z_r extraction"
    else:
        dominant_issue = "mixed / unresolved"
    print(f"cases_scanned={len(cases)}")
    print(f"cases_with_same_strain_reference_found={sum(1 for item in results if item.same_strain_reference_found)}")
    print(f"cases_missing_same_strain_reference={sum(1 for item in results if not item.same_strain_reference_found)}")
    print(f"completed_strict_cases={sum(1 for item in results if item.strict_status == 'complete')}")
    print(f"diagnostic_completed_cases={sum(1 for item in results if item.log10J_diagnostic is not None)}")
    print(f"incomplete_cases={sum(1 for item in results if item.case_status != 'complete')}")
    print(f"median_barrier_kBT_absolute={float(np.median(absolute_barriers)) if absolute_barriers else 'nan'}")
    print(f"median_barrier_kBT_excess={float(np.median(excess_barriers)) if excess_barriers else 'nan'}")
    print(f"median_background_fraction={float(np.median(bg_fracs)) if bg_fracs else 'nan'}")
    print(f"min_barrier_kBT_excess={float(np.min(excess_barriers)) if excess_barriers else 'nan'}")
    print(f"max_barrier_kBT_excess={float(np.max(excess_barriers)) if excess_barriers else 'nan'}")
    print(f"median_log10J_diagnostic={float(np.median(diag_logs)) if diag_logs else 'nan'}")
    print(f"dominant_issue={dominant_issue}")
    print(f"output_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
