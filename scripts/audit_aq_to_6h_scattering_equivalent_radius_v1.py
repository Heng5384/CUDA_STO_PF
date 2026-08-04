#!/usr/bin/env python3
"""Read-only AQ-to-6 h scattering-equivalent-radius audit.

This audit deliberately separates Yu 2024's registered AQ two-population
calculation from Sheskin et al. 2018 ("Tailoring") observations.  It does
not run phase field, tune an input, or convert a scattering-equivalent radius
into a material or PF-seed radius.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy.polynomial.legendre import leggauss


STATUS_PASS_YU = "PASS_YU_AQ_REPRODUCTION_AND_6H_EQUIVALENT_RADIUS_FOUND"
STATUS_INVENTORY = "PASS_SCATTERING_EQUIVALENCE_BUT_INVENTORY_CONFLICT"
YU_AQ_N_SMALL = 7.5e24
YU_AQ_R_SMALL_NM = 2.0
YU_AQ_N_BIG = 1.9e21
YU_AQ_R_BIG_NM = 30.0
TAILORING_AQ_N = 3.61e21
TAILORING_AQ_N_SIGMA = 1.51e21
TAILORING_6H_N = 1.68e24
TAILORING_6H_N_LOW = 0.76e24
TAILORING_6H_N_HIGH = 2.60e24
TAILORING_AQ_XAG = 0.0078
YU_AQ_XAG = 0.0069
TAILORING_6H_XAG = 0.0062
TAILORING_6H_XAG_LOW = 0.0058
TAILORING_6H_XAG_HIGH = 0.0066
TEMPERATURE_K = 300.00
YU_SOURCE_T_K = 303.06
TAILORING_LOWEST_T_K = 303.15  # 30 degC, the lowest plotted Figure 5c point.
FIGURE_DIGITIZATION_SIGMA = 0.015
ROOT_TOLERANCE = 1.0e-6


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/aq_to_6h_equivalent_radius_v1"),
    )
    parser.add_argument(
        "--yu-config",
        type=Path,
        default=Path("data/qualification/yu2024_transport_v1/yu_AQ_parameters.json"),
    )
    parser.add_argument(
        "--yu-script",
        type=Path,
        default=Path("scripts/reproduce_yu2024_debye_callaway.py"),
    )
    parser.add_argument(
        "--pf-script",
        type=Path,
        default=Path("scripts/pf_full_psd_no_dislocation_transport_v1.py"),
    )
    parser.add_argument(
        "--yu-report",
        type=Path,
        default=Path("reports/yu2024_debye_callaway_reproduction_v1/reproduction_report.md"),
    )
    parser.add_argument(
        "--yu-main-pdf",
        type=Path,
        default=Path(
            "/Users/heng/Library/CloudStorage/OneDrive-GuangdongTechnion-"
            "IsraelInstituteofTechnology/Project/Diffusion/Paper/Advanced Energy "
            "Materials - 2024 - Yu - Ostwald Ripening of Ag2Te Precipitates in "
            "Thermoelectric PbTe  Effects of.pdf"
        ),
    )
    parser.add_argument(
        "--yu-si-pdf",
        type=Path,
        default=Path(
            "/Users/heng/Library/CloudStorage/OneDrive-GuangdongTechnion-"
            "IsraelInstituteofTechnology/Project/Diffusion/Paper/AEM Yu 2024 SM.pdf"
        ),
    )
    parser.add_argument(
        "--tailoring-pdf",
        type=Path,
        default=Path(
            "/Users/heng/Library/CloudStorage/OneDrive-GuangdongTechnion-"
            "IsraelInstituteofTechnology/Project/Diffusion/Paper/"
            "tailoring-thermoelectric-transport-properties-of-ag-alloyed-pbte-"
            "effects-of-microstructure-evolution.pdf"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FrozenYuCalculator:
    """Yu S7--S10 calculation with arbitrary named precipitate populations.

    The host is the public AQ parameter set.  Changing only ``xag`` is the
    explicit cross-paper matrix diagnostic requested by the task, not a claim
    that Tailoring supplies the other Yu host parameters.
    """

    def __init__(self, pf: Any, config: dict[str, Any], order: int = 256):
        self.pf = pf
        self.config = config
        self.order = order
        self._cache: dict[tuple[float, float], dict[str, np.ndarray | float]] = {}
        self.v = float(config["shared_parameters"]["average_sound_velocity_m_s"])
        self.kb = float(config["physical_constants"]["k_B_J_K"])
        self.hbar = float(config["physical_constants"]["hbar_J_s"])
        self.theta = float(config["shared_parameters"]["debye_temperature_K"])
        shared = config["shared_parameters"]
        self.density_contrast = float(shared["density_difference_kg_m3"]) / float(
            shared["matrix_density_kg_m3"]
        )

    def state(self, temperature_k: float, xag: float) -> dict[str, np.ndarray | float]:
        key = (float(temperature_k), float(xag))
        if key not in self._cache:
            nodes, weights = leggauss(self.order)
            xmax = self.theta / temperature_k
            x = 0.5 * (nodes + 1.0) * xmax
            omega = x * self.kb * temperature_k / self.hbar
            base = self.pf.base_scattering_rates(omega, temperature_k, xag, self.config)
            prefactor = self.kb / (2.0 * math.pi**2 * self.v) * (
                self.kb * temperature_k / self.hbar
            ) ** 3
            self._cache[key] = {
                "x": x,
                "omega": omega,
                "weights": weights,
                "xmax": xmax,
                "base": base["phonon_phonon"] + base["boundary"] + base["point_defect"],
                "rates": base,
                "prefactor": prefactor,
            }
        return self._cache[key]

    def cross_section(self, omega: np.ndarray, radius_nm: float) -> np.ndarray:
        radius_m = radius_nm * 1.0e-9
        short = 2.0 * math.pi * radius_m**2
        long = (
            4.0
            / 9.0
            * math.pi
            * radius_m**2
            * self.density_contrast**2
            * (omega * radius_m / self.v) ** 4
        )
        return short * long / (short + long)

    def kappa(
        self, temperature_k: float, xag: float, populations: list[tuple[float, float]]
    ) -> float:
        state = self.state(temperature_k, xag)
        omega = np.asarray(state["omega"])
        precipitate = np.zeros_like(omega)
        for number_density, radius_nm in populations:
            precipitate += self.v * number_density * self.cross_section(omega, radius_nm)
        kernel = (
            float(state["prefactor"])
            * self.pf.bose_weight(np.asarray(state["x"]))
            / (np.asarray(state["base"]) + precipitate)
        )
        return float(0.5 * float(state["xmax"]) * np.sum(np.asarray(state["weights"]) * kernel))

    def detail(
        self, temperature_k: float, xag: float, populations: list[tuple[float, float]]
    ) -> dict[str, np.ndarray | float]:
        state = self.state(temperature_k, xag)
        omega = np.asarray(state["omega"])
        precipitate = np.zeros_like(omega)
        components: list[np.ndarray] = []
        for number_density, radius_nm in populations:
            component = self.v * number_density * self.cross_section(omega, radius_nm)
            components.append(component)
            precipitate += component
        total = np.asarray(state["base"]) + precipitate
        kernel = float(state["prefactor"]) * self.pf.bose_weight(np.asarray(state["x"])) / total
        weights = 0.5 * float(state["xmax"]) * np.asarray(state["weights"])
        return {
            **state,
            "precipitate": precipitate,
            "components": components,
            "total": total,
            "kernel": kernel,
            "quadrature_weights": weights,
            "kappa": float(np.sum(weights * kernel)),
        }


def radius_grid() -> np.ndarray:
    return np.concatenate(
        (
            np.arange(0.10, 2.0000001, 0.01),
            np.arange(2.02, 10.0000001, 0.02),
            np.arange(10.05, 40.0000001, 0.05),
            np.arange(40.2, 100.0000001, 0.2),
            np.arange(101.0, 200.0000001, 1.0),
        )
    )


def find_roots(
    calculator: FrozenYuCalculator,
    temperature_k: float,
    xag: float,
    number_density: float,
    target: float,
) -> tuple[list[dict[str, float | str]], dict[str, float]]:
    radii = radius_grid()
    values = np.array(
        [calculator.kappa(temperature_k, xag, [(number_density, float(radius))]) for radius in radii]
    )
    brackets: list[tuple[float, float]] = []
    residuals = values - target
    for index in range(len(radii) - 1):
        if residuals[index] == 0.0 or residuals[index] * residuals[index + 1] < 0.0:
            brackets.append((float(radii[index]), float(radii[index + 1])))
    roots: list[dict[str, float | str]] = []
    for lower, upper in brackets:
        flo = calculator.kappa(temperature_k, xag, [(number_density, lower)]) - target
        for _ in range(70):
            middle = 0.5 * (lower + upper)
            fmid = calculator.kappa(temperature_k, xag, [(number_density, middle)]) - target
            if abs(fmid) <= 1.0e-13:
                lower = upper = middle
                break
            if flo * fmid <= 0.0:
                upper = middle
            else:
                lower = middle
                flo = fmid
        root = 0.5 * (lower + upper)
        detail = calculator.detail(temperature_k, xag, [(number_density, root)])
        contribution = np.asarray(detail["quadrature_weights"]) * np.asarray(detail["kernel"])
        cumulative = np.cumsum(contribution) / float(detail["kappa"])
        omega50 = float(np.asarray(detail["omega"])[min(np.searchsorted(cumulative, 0.5), len(cumulative) - 1)])
        radius_m = root * 1.0e-9
        ratio = (
            (2.0 / 9.0)
            * calculator.density_contrast**2
            * (omega50 * radius_m / calculator.v) ** 4
        )
        roots.append(
            {
                "radius_nm": root,
                "kappa_W_mK": float(detail["kappa"]),
                "absolute_mismatch_W_mK": abs(float(detail["kappa"]) - target),
                "omega50_rad_s": omega50,
                "sigma_long_over_short_at_omega50": ratio,
                "branch": "RAYLEIGH_SIDE" if ratio < 1.0 else "GEOMETRIC_SIDE",
            }
        )
    mismatch = np.abs(values - target)
    closest = int(np.argmin(mismatch))
    return roots, {
        "reachable_kappa_min_W_mK": float(values.min()),
        "reachable_kappa_max_W_mK": float(values.max()),
        "closest_radius_nm": float(radii[closest]),
        "minimum_absolute_mismatch_W_mK": float(mismatch[closest]),
    }


def render_tailoring_figure(pdf: Path, output: Path) -> None:
    """Save the Figure 5c graph with axes; the source PDF itself is untouched."""
    if shutil.which("pdftoppm") is None:
        raise RuntimeError("pdftoppm is required to preserve the digitized Figure 5c source")
    with tempfile.TemporaryDirectory(prefix="aq_6h_tailoring_") as tmp:
        prefix = Path(tmp) / "page"
        subprocess.run(
            ["pdftoppm", "-f", "5", "-l", "5", "-r", "600", "-png", str(pdf), str(prefix)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        page = Image.open(Path(f"{prefix}-5.png"))
        # Pixel box verified against the 600 dpi page-5 render: it preserves
        # the full Figure 5c axes, labels, symbols, legend and caption start.
        page.crop((300, 3200, 2800, 5200)).save(output)


def tailoring_digitization() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    """Return the auditable Figure 5c point data and 300 K interpolation.

    Axis calibration comes from labelled major ticks in a 600 dpi PDF render:
    x=805/1066/.../2358 for 0/50/.../300 degC and y=3622/.../4745 for
    1.6/.../0.8 W m^-1 K^-1.  The first data point is at 30 degC.  At 300 K
    (26.85 degC) the report therefore performs a short extrapolation rather
    than relabelling a 30 degC point as 300 K.
    """
    x_ticks = [(805.0, 0.0), (1066.0, 50.0), (1324.0, 100.0), (1583.0, 150.0),
               (1842.0, 200.0), (2100.0, 250.0), (2358.0, 300.0)]
    y_ticks = [(3622.0, 1.6), (3761.0, 1.5), (3902.0, 1.4), (4044.0, 1.3),
               (4185.0, 1.2), (4324.0, 1.1), (4466.0, 1.0), (4606.0, 0.9),
               (4745.0, 0.8)]
    ax, bx = np.polyfit([item[0] for item in x_ticks], [item[1] for item in x_ticks], 1)
    ay, by = np.polyfit([item[0] for item in y_ticks], [item[1] for item in y_ticks], 1)
    # Marker centres from colour-mask components.  The red 30 degC marker is
    # partly occluded by the blue square; its nominal x is fixed to the shared
    # temperature marker while its y comes from its visible red outline.
    pixels = [
        ("AQ", 30.0, 963.85, 4127.67),
        ("6h", 30.0, 963.85, 4094.09),
        ("AQ", 50.0, 1067.61, 4257.37),
        ("6h", 50.0, 1059.69, 4345.01),
    ]
    rows: list[dict[str, Any]] = []
    for state, nominal_c, px, py in pixels:
        rows.append(
            {
                "state": state,
                "nominal_temperature_C": nominal_c,
                "nominal_temperature_K": nominal_c + 273.15,
                "marker_x_pixel": px,
                "marker_y_pixel": py,
                "x_axis_calibrated_temperature_C": ax * px + bx,
                "digitized_kappa_W_mK": ay * py + by,
                "digitization_uncertainty_W_mK": FIGURE_DIGITIZATION_SIGMA,
                "identity": "FIGURE_5C_DIGITIZED_NOT_TABLE_VALUE",
            }
        )
    values: dict[str, float] = {}
    for state in ("AQ", "6h"):
        low, high = [row for row in rows if row["state"] == state]
        low_c = float(low["nominal_temperature_C"])
        high_c = float(high["nominal_temperature_C"])
        k_low = float(low["digitized_kappa_W_mK"])
        k_high = float(high["digitized_kappa_W_mK"])
        at_300 = k_low + (k_high - k_low) * ((TEMPERATURE_K - 273.15 - low_c) / (high_c - low_c))
        values[f"{state}_300K"] = at_300
        values[f"{state}_303p15K"] = k_low
    calibration = [
        {"axis": "x", "pixel": pixel, "value": value, "unit": "degC"}
        for pixel, value in x_ticks
    ] + [
        {"axis": "y", "pixel": pixel, "value": value, "unit": "W m^-1 K^-1"}
        for pixel, value in y_ticks
    ]
    return calibration, rows, values


def volume_equivalent_radius_nm(length_nm: float, aspect_ratio: float) -> float:
    # Prolate spheroid: major semiaxis a=L/2, minor b=a/AR.
    a = 0.5 * length_nm
    b = a / aspect_ratio
    return (a * b * b) ** (1.0 / 3.0)


def physicality(number_density: float, radius_nm: float) -> dict[str, Any]:
    radius_m = radius_nm * 1.0e-9
    separation_nm = number_density ** (-1.0 / 3.0) * 1.0e9
    f = number_density * 4.0 * math.pi * radius_m**3 / 3.0
    area = 4.0 * math.pi * number_density * radius_m**2
    m6 = number_density * radius_m**6
    count246 = number_density * (246.0e-9) ** 3
    count512 = number_density * (512.0e-9) ** 3
    return {
        "mean_separation_nm": separation_nm,
        "spherical_volume_fraction": f,
        "interface_area_density_m-1": area,
        "M6_m3": m6,
        "equivalent_count_246cube": count246,
        "equivalent_count_512cube": count512,
        "diameter_over_separation": 2.0 * radius_nm / separation_nm,
        "overlap_status": "SERIOUS_SPHERICAL_OVERLAP" if 2.0 * radius_nm >= separation_nm else "NO_GEOMETRIC_SPHERE_OVERLAP",
        "inventory_status": "INVENTORY_OR_OBJECT_DEFINITION_CONFLICT" if f > 0.03 else "BELOW_3_PERCENT_SPHERICAL_FRACTION",
        "pf_resolution_status": "PF_RESOLVED_GE_8NM" if radius_nm >= 8.0 else "BELOW_PF_RESOLVED_RANGE",
        "profile_library_status": "WITHIN_CURRENT_8_TO_11p5NM_LIBRARY" if 8.0 <= radius_nm <= 11.5 else "OUTSIDE_CURRENT_PROFILE_LIBRARY",
        "integer_population_246_status": "AT_LEAST_ONE_EXPECTED" if count246 >= 1.0 else "SUBUNITY_EXPECTED_COUNT",
    }


def main() -> None:
    args = arguments()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite existing audit directory: {args.out}")
    required = (args.yu_config, args.yu_script, args.pf_script, args.yu_report, args.yu_main_pdf, args.yu_si_pdf, args.tailoring_pdf)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Required source is missing: " + ", ".join(missing))
    args.out.mkdir(parents=True)

    reproduce = import_module("yu_reproduction", args.yu_script)
    pf = import_module("pf_full_psd", args.pf_script)
    config = pf.load_json(args.yu_config)
    pf.validate_yu_base_config(config, args.yu_config)
    calculator = FrozenYuCalculator(pf, config)

    # Stage 1 - exact public Yu AQ stepwise calculation.
    mechanisms = reproduce.STAGES
    step_rows: list[dict[str, Any]] = []
    yu_step: dict[tuple[float, str], float] = {}
    stage_map = {
        "HOST_ONLY": mechanisms["U_N_plus_GB"],
        "HOST_PLUS_POINT_DEFECT": mechanisms["U_N_plus_GB_PD"],
        "HOST_PLUS_POINT_DEFECT_PLUS_SMALL": mechanisms["U_N_plus_GB_PD_Pre_small"],
        "HOST_PLUS_POINT_DEFECT_PLUS_SMALL_PLUS_BIG": mechanisms["U_N_plus_GB_PD_Pre_all"],
    }
    for temperature in (TEMPERATURE_K, YU_SOURCE_T_K):
        for name, selected in stage_map.items():
            value = reproduce.integrate_gauss_legendre(temperature, config, selected)
            yu_step[(temperature, name)] = value
            step_rows.append(
                {
                    "T_K": temperature,
                    "model_stage": name,
                    "kappa_W_mK": value,
                    "incremental_delta_from_previous_W_mK": "",
                    "mechanisms": "+".join(selected),
                    "A_N": 1.5,
                    "dislocation_mode": "OFF",
                    "identity": "YU_2024_AQ_PUBLIC_PARAMETER_CALCULATION",
                }
            )
        stage_start = len(step_rows) - len(stage_map)
        for idx in range(stage_start + 1, len(step_rows)):
            step_rows[idx]["incremental_delta_from_previous_W_mK"] = (
                float(step_rows[idx]["kappa_W_mK"]) - float(step_rows[idx - 1]["kappa_W_mK"])
            )
    write_csv(args.out / "yu_aq_300K_stepwise_kappa.csv", list(step_rows[0]), step_rows)

    # Independent numerical equality check against the existing qualified AQ row.
    expected_303p06 = 1.65602381323
    observed_303p06 = yu_step[(YU_SOURCE_T_K, "HOST_PLUS_POINT_DEFECT_PLUS_SMALL_PLUS_BIG")]
    yu_relative_error = abs(observed_303p06 - expected_303p06) / expected_303p06
    yu_pass = yu_relative_error <= 1.0e-10

    spectral_rows: list[dict[str, Any]] = []
    for temperature in (TEMPERATURE_K, YU_SOURCE_T_K):
        detail = calculator.detail(
            temperature, YU_AQ_XAG, [(YU_AQ_N_SMALL, YU_AQ_R_SMALL_NM), (YU_AQ_N_BIG, YU_AQ_R_BIG_NM)]
        )
        rates = detail["rates"]
        for index, omega in enumerate(np.asarray(detail["omega"])):
            spectral_rows.append(
                {
                    "T_K": temperature,
                    "omega_rad_s": omega,
                    "frequency_THz": omega / (2.0 * math.pi * 1.0e12),
                    "phonon_phonon_rate_s-1": np.asarray(rates["phonon_phonon"])[index],
                    "boundary_rate_s-1": np.asarray(rates["boundary"])[index],
                    "point_defect_rate_s-1": np.asarray(rates["point_defect"])[index],
                    "small_precipitate_rate_s-1": np.asarray(detail["components"])[0][index],
                    "big_precipitate_rate_s-1": np.asarray(detail["components"])[1][index],
                    "total_rate_s-1": np.asarray(detail["total"])[index],
                    "dkappa_dx_W_mK": np.asarray(detail["kernel"])[index],
                }
            )
    write_csv(args.out / "yu_aq_300K_spectral_rates.csv", list(spectral_rows[0]), spectral_rows)

    # Source identity and digitization provenance.
    source_rows = [
        {
            "source": "Yu et al. 2024", "sample_state": "AQ", "observation_method": "Table S2 / AQ structure statistics",
            "population_name": "small Ag2Te precipitate population", "radius_or_size": "R=2 nm", "number_density": "7.5e24 m^-3",
            "matrix_Ag": "0.0069", "grain_size": "10.4 um", "lattice_parameter": "6.4286 angstrom", "dislocation_condition": "0 m^-2",
            "data_identity": "YU_AQ_PUBLIC_TRANSPORT_INPUT", "uncertainty": "not tabulated", "source_location": "Yu 2024 SI Table S2",
        },
        {
            "source": "Yu et al. 2024", "sample_state": "AQ", "observation_method": "Table S2 / AQ structure statistics",
            "population_name": "big Ag2Te precipitate population", "radius_or_size": "R=30 nm", "number_density": "1.9e21 m^-3",
            "matrix_Ag": "0.0069", "grain_size": "10.4 um", "lattice_parameter": "6.4286 angstrom", "dislocation_condition": "0 m^-2",
            "data_identity": "YU_AQ_PUBLIC_TRANSPORT_INPUT", "uncertainty": "not tabulated", "source_location": "Yu 2024 SI Table S2",
        },
        {
            "source": "Sheskin et al. 2018", "sample_state": "AQ", "observation_method": "complementary TEM/APT",
            "population_name": "Ag-rich objects, total at nanometre scale", "radius_or_size": "no unique radius; small <=10 nm and elongated up to ~100 nm", "number_density": "(3.61 +/- 1.51)e21 m^-3",
            "matrix_Ag": "0.0078 +/- 0.0004", "grain_size": "not supplied for Yu host substitution", "lattice_parameter": "not supplied for Yu host substitution", "dislocation_condition": "lattice strain qualitatively discussed; no Yu-equivalent transport input",
            "data_identity": "TAILORING_AQ_TEM_APT_TOTAL", "uncertainty": "+/-1.51e21 m^-3", "source_location": "Tailoring Table 1, Figures 4 and 6",
        },
        {
            "source": "Sheskin et al. 2018", "sample_state": "AQ", "observation_method": "SE/FIB",
            "population_name": "large-scale objects only", "radius_or_size": "typically micrometre-scale imaging sensitivity", "number_density": "(5.0 +/- 1.5)e17 m^-3",
            "matrix_Ag": "0.0078 +/- 0.0004", "grain_size": "not a Yu input", "lattice_parameter": "not a Yu input", "dislocation_condition": "not a transport input",
            "data_identity": "TAILORING_AQ_FIB_SE_DIAGNOSTIC_ONLY", "uncertainty": "+/-1.5e17 m^-3", "source_location": "Tailoring Table 1 and Figure 4",
        },
        {
            "source": "Sheskin et al. 2018", "sample_state": "AQ", "observation_method": "one APT reconstruction",
            "population_name": "count-fraction diagnostic", "radius_or_size": "37 small; 6 elongated large", "number_density": "not an independent bulk density", "matrix_Ag": "0.0078 +/- 0.0004", "grain_size": "n/a", "lattice_parameter": "n/a", "dislocation_condition": "n/a",
            "data_identity": "TAILORING_APT_COUNT_FRACTION_DIAGNOSTIC", "uncertainty": "single reconstruction", "source_location": "Tailoring Figure 4",
        },
        {
            "source": "Sheskin et al. 2018", "sample_state": "6 h at 380 degC", "observation_method": "complementary TEM/APT",
            "population_name": "Ag-rich objects, total at nanometre scale", "radius_or_size": "mainly 1--10 nm; no unique radius", "number_density": "(1.68 +/- 0.92)e24 m^-3",
            "matrix_Ag": "0.0062 +/- 0.0004", "grain_size": "not supplied for Yu host substitution", "lattice_parameter": "not supplied for Yu host substitution", "dislocation_condition": "not a Yu-equivalent transport input",
            "data_identity": "TAILORING_6H_TEM_APT_TOTAL", "uncertainty": "+/-0.92e24 m^-3", "source_location": "Tailoring Table 1 and Figure 6",
        },
        {
            "source": "Sheskin et al. 2018", "sample_state": "6 h at 380 degC", "observation_method": "SE/FIB",
            "population_name": "large-scale objects only", "radius_or_size": "different detection scale", "number_density": "(1.07 +/- 0.45)e18 m^-3",
            "matrix_Ag": "0.0062 +/- 0.0004", "grain_size": "n/a", "lattice_parameter": "n/a", "dislocation_condition": "n/a",
            "data_identity": "TAILORING_6H_FIB_SE_DIAGNOSTIC_ONLY", "uncertainty": "+/-0.45e18 m^-3", "source_location": "Tailoring Table 1",
        },
    ]
    write_csv(args.out / "source_identity_table.csv", list(source_rows[0]), source_rows)
    render_tailoring_figure(args.tailoring_pdf, args.out / "tailoring_figure5c_original.png")
    calibration, digitized_rows, tailoring_kappa = tailoring_digitization()
    write_csv(args.out / "tailoring_figure5c_calibration.csv", list(calibration[0]), calibration)
    write_csv(args.out / "tailoring_figure5c_digitization.csv", list(digitized_rows[0]), digitized_rows)

    source_audit = f"""# AQ-to-6 h source identity audit

## Strict source separation

Yu 2024 Table S2 supplies the registered AQ two-population Debye--Callaway input.  Sheskin et al. 2018 (Tailoring) supplies independent measured transport and microstructure observations.  No Yu population, grain size, lattice parameter, or dislocation field is represented as a Tailoring measurement.

The Tailoring supporting information was not locally available.  Its main text states that the supporting material contains the raw temperature-dependent thermal-diffusivity and heat-capacity data; it does not provide a local thermal-conductivity table for this audit.  Thus the AQ/6 h thermal-conductivity values below are explicitly Figure 5c digitizations, not table values.

## Figure 5c digitization

`tailoring_figure5c_original.png` is a 600 dpi crop from page 5.  The calibration and marker coordinates are retained in the companion CSV files.  The lowest plotted marker is 30 degC = {TAILORING_LOWEST_T_K:.2f} K.  Therefore 300.00 K is a short {TAILORING_LOWEST_T_K - TEMPERATURE_K:.2f} K extrapolation from the 30/50 degC segment; it is not a direct 300 K measurement.  The assigned {FIGURE_DIGITIZATION_SIGMA:.3f} W m^-1 K^-1 uncertainty covers marker thickness, coordinate calibration and this short extrapolation only.

## Detection-scale rule

The Tailoring TEM/APT total density and the SE/FIB large-scale density are not summed.  They use different detection scales and could overlap in object identity.  The APT 37/43 and 6/43 count fractions are a single-reconstruction diagnostic, not a validated bulk two-population density.
"""
    (args.out / "source_identity_audit.md").write_text(source_audit, encoding="utf-8")

    # Tailoring AQ literal diagnosis: T-A scalar total, T-B count-fraction envelope, T-C FIB diagnostic.
    ta_radii = np.concatenate((np.arange(0.5, 10.000001, 0.05), np.arange(10.1, 40.000001, 0.1), np.arange(40.5, 100.000001, 0.5)))
    literal_rows: list[dict[str, Any]] = []
    ta_values: list[float] = []
    for host_mode, xag in (("HOST_YU_AQ_FIXED", YU_AQ_XAG), ("HOST_YU_COMMON_WITH_TAILORING_MATRIX_UPDATED", TAILORING_AQ_XAG)):
        for radius in ta_radii:
            value = calculator.kappa(TEMPERATURE_K, xag, [(TAILORING_AQ_N, float(radius))])
            literal_rows.append({"branch": "T-A", "host_mode": host_mode, "matrix_xAg": xag, "number_density_m-3": TAILORING_AQ_N, "small_radius_nm": radius, "large_length_nm": "", "large_AR": "", "large_radius_definition": "", "kappa_300K_W_mK": value, "identity": "CROSS_PAPER_DIAGNOSTIC_NOT_UNIQUE_TAILORING_MODEL"})
            ta_values.append(value)
    frac_small = 37.0 / 43.0
    n_small, n_large = frac_small * TAILORING_AQ_N, (1.0 - frac_small) * TAILORING_AQ_N
    tb_values: list[float] = []
    for small_radius in np.arange(0.5, 5.000001, 0.25):
        for length in np.arange(10.0, 100.000001, 2.0):
            for ar in (2.0, 5.0, 10.0):
                for definition, large_radius in (("SPHERICAL_RADIUS_ASSUMPTION_L_OVER_2", 0.5 * length), ("PROLATE_SPHEROID_VOLUME_EQUIVALENT", volume_equivalent_radius_nm(length, ar))):
                    value = calculator.kappa(TEMPERATURE_K, TAILORING_AQ_XAG, [(n_small, float(small_radius)), (n_large, large_radius)])
                    literal_rows.append({"branch": "T-B", "host_mode": "HOST_YU_COMMON_WITH_TAILORING_MATRIX_UPDATED", "matrix_xAg": TAILORING_AQ_XAG, "number_density_m-3": TAILORING_AQ_N, "small_radius_nm": small_radius, "large_length_nm": length, "large_AR": ar, "large_radius_definition": definition, "kappa_300K_W_mK": value, "identity": "SINGLE_APT_RECONSTRUCTION_COUNT_FRACTION_DIAGNOSTIC"})
                    tb_values.append(value)
    for radius_nm in (500.0, 1000.0, 2000.0):
        value = calculator.kappa(TEMPERATURE_K, TAILORING_AQ_XAG, [(5.0e17, radius_nm)])
        literal_rows.append({"branch": "T-C", "host_mode": "HOST_YU_COMMON_WITH_TAILORING_MATRIX_UPDATED", "matrix_xAg": TAILORING_AQ_XAG, "number_density_m-3": 5.0e17, "small_radius_nm": "", "large_length_nm": 2.0 * radius_nm, "large_AR": "", "large_radius_definition": "MICROMETRE_SCALE_DIAGNOSTIC_SPHERE", "kappa_300K_W_mK": value, "identity": "FIB_LARGE_SCALE_DIAGNOSTIC_NOT_ADDED_TO_TA_OR_TB"})
    write_csv(args.out / "tailoring_aq_source_literal_cases.csv", list(literal_rows[0]), literal_rows)
    envelope_rows = []
    for label, values in (("T-A_all_host_modes", ta_values), ("T-B_count_fraction_diagnostic", tb_values), ("T-A_plus_T-B_literal_envelope", ta_values + tb_values)):
        envelope_rows.append({"population_branch": label, "T_K": TEMPERATURE_K, "kappa_min_W_mK": min(values), "kappa_max_W_mK": max(values), "source_identity": "LITERAL_INPUT_ENVELOPE_NOT_A_UNIQUE_TAILORING_MODEL"})
    write_csv(args.out / "tailoring_aq_300K_kappa_envelope.csv", list(envelope_rows[0]), envelope_rows)
    (args.out / "tailoring_aq_population_assumption_audit.md").write_text(
        "# Tailoring AQ population assumption audit\n\n"
        "T-A uses the TEM/APT total density as one monodisperse diagnostic and scans its radius; it cannot yield a unique conductivity because Tailoring reports no unique AQ radius.  `HOST_YU_AQ_FIXED` retains the Yu AQ host including xAg=0.0069.  `HOST_YU_COMMON_WITH_TAILORING_MATRIX_UPDATED` changes only the point-defect xAg to the Tailoring AQ value 0.0078; all other host fields remain Yu inputs and are therefore explicitly cross-paper diagnostics.\n\n"
        "T-B applies the 37/43 and 6/43 counts only as a single-APT-reconstruction fraction.  Large-object length is not treated as a sphere radius.  It is bracketed by a deliberately nonphysical L/2 sphere and a prolate-spheroid volume-equivalent radius for AR=2,5,10.\n\n"
        "T-C is a separate micro-scale FIB diagnostic.  It is never added to the TEM/APT density, preventing double counting.\n",
        encoding="utf-8",
    )

    # Stage 3 targets: T1, T2 and source-literal envelope endpoints.
    yu_small_only = yu_step[(TEMPERATURE_K, "HOST_PLUS_POINT_DEFECT_PLUS_SMALL")]
    yu_big_only = reproduce.integrate_gauss_legendre(
        TEMPERATURE_K, config, ("phonon_phonon", "boundary", "point_defect", "precipitate_big")
    )
    yu_all = yu_step[(TEMPERATURE_K, "HOST_PLUS_POINT_DEFECT_PLUS_SMALL_PLUS_BIG")]
    targets = [
        {"target_identity": "T1_YU_AQ_TWO_POPULATION_300K", "target_kappa": yu_all, "target_uncertainty": 0.0},
        {"target_identity": "T2_TAILORING_AQ_FIG5C_EXTRAPOLATED_300K", "target_kappa": tailoring_kappa["AQ_300K"], "target_uncertainty": FIGURE_DIGITIZATION_SIGMA},
        {"target_identity": "T3_LITERAL_ENVELOPE_MIN", "target_kappa": min(ta_values + tb_values), "target_uncertainty": 0.0},
        {"target_identity": "T3_LITERAL_ENVELOPE_MAX", "target_kappa": max(ta_values + tb_values), "target_uncertainty": 0.0},
    ]
    root_rows: list[dict[str, Any]] = []
    root_details: list[dict[str, Any]] = []
    for target in targets:
        for mode, xag in (("E1", YU_AQ_XAG), ("E2", TAILORING_6H_XAG)):
            roots, reach = find_roots(calculator, TEMPERATURE_K, xag, TAILORING_6H_N, float(target["target_kappa"]))
            if roots:
                for root in roots:
                    record = {**target, "equivalent_mode": mode, "matrix_xAg": xag, "N_6h_m-3": TAILORING_6H_N, "root_exists": True, **root, **reach}
                    root_rows.append(record)
                    root_details.append(record)
            else:
                root_rows.append({**target, "equivalent_mode": mode, "matrix_xAg": xag, "N_6h_m-3": TAILORING_6H_N, "root_exists": False, "radius_nm": "", "kappa_W_mK": "", "absolute_mismatch_W_mK": "", "omega50_rad_s": "", "sigma_long_over_short_at_omega50": "", "branch": "NO_ROOT", **reach})
    write_csv(args.out / "tailoring_6h_equivalent_radius_roots.csv", list(root_rows[0]), root_rows)

    # Deterministic uncertainty grid: N6, E2 matrix xAg, digitized T2 target,
    # T-A density error and the T-B size/count diagnostic envelope.
    uncertainty_scenarios: list[dict[str, Any]] = []
    literal_target_variants: list[tuple[str, float]] = []
    for n_aq in (TAILORING_AQ_N - TAILORING_AQ_N_SIGMA, TAILORING_AQ_N, TAILORING_AQ_N + TAILORING_AQ_N_SIGMA):
        values = [calculator.kappa(TEMPERATURE_K, TAILORING_AQ_XAG, [(n_aq, float(r))]) for r in ta_radii]
        literal_target_variants.extend(((f"T3_TA_NAQ_{n_aq:.3e}_MIN", min(values)), (f"T3_TA_NAQ_{n_aq:.3e}_MAX", max(values))))
    literal_target_variants.extend((("T3_TB_SIZE_ENVELOPE_MIN", min(tb_values)), ("T3_TB_SIZE_ENVELOPE_MAX", max(tb_values))))
    sampled_targets = [("T1_YU_AQ", yu_all)] + [
        (f"T2_TAILORING_AQ_DIGITIZED_{sign}", tailoring_kappa["AQ_300K"] + sign * FIGURE_DIGITIZATION_SIGMA)
        for sign in (-1.0, 0.0, 1.0)
    ] + literal_target_variants
    for target_id, target_value in sampled_targets:
        for n6 in (TAILORING_6H_N_LOW, TAILORING_6H_N, TAILORING_6H_N_HIGH):
            for mode, xags in (("E1", (YU_AQ_XAG,)), ("E2", (TAILORING_6H_XAG_LOW, TAILORING_6H_XAG, TAILORING_6H_XAG_HIGH))):
                for xag in xags:
                    roots, reach = find_roots(calculator, TEMPERATURE_K, xag, n6, target_value)
                    uncertainty_scenarios.append({"target_identity": target_id, "target_kappa_W_mK": target_value, "N_6h_m-3": n6, "equivalent_mode": mode, "matrix_xAg": xag, "root_count": len(roots), "roots_nm": ";".join(f"{float(root['radius_nm']):.9g}" for root in roots), "closest_radius_nm": reach["closest_radius_nm"], "minimum_absolute_mismatch_W_mK": reach["minimum_absolute_mismatch_W_mK"], "reachable_kappa_min_W_mK": reach["reachable_kappa_min_W_mK"], "reachable_kappa_max_W_mK": reach["reachable_kappa_max_W_mK"]})
    finite_roots = [float(item) for row in uncertainty_scenarios for item in str(row["roots_nm"]).split(";") if item]
    uncertainty_summary = {
        "summary_kind": "aggregate_deterministic_grid",
        "sample_count": len(uncertainty_scenarios),
        "root_sample_count": len(finite_roots),
        "R_6h_eq_median_nm": float(np.quantile(finite_roots, 0.5)),
        "R_6h_eq_q05_nm": float(np.quantile(finite_roots, 0.05)),
        "R_6h_eq_q25_nm": float(np.quantile(finite_roots, 0.25)),
        "R_6h_eq_q75_nm": float(np.quantile(finite_roots, 0.75)),
        "R_6h_eq_q95_nm": float(np.quantile(finite_roots, 0.95)),
        "multiple_root_probability": float(sum(row["root_count"] > 1 for row in uncertainty_scenarios) / len(uncertainty_scenarios)),
        "no_root_probability": float(sum(row["root_count"] == 0 for row in uncertainty_scenarios) / len(uncertainty_scenarios)),
    }
    uncertainty_rows = uncertainty_scenarios + [{key: value for key, value in uncertainty_summary.items()}]
    fields = sorted({key for row in uncertainty_rows for key in row})
    write_csv(args.out / "tailoring_6h_equivalent_radius_uncertainty.csv", fields, uncertainty_rows)

    physical_rows: list[dict[str, Any]] = []
    for root in root_details:
        physical_rows.append({**{key: root[key] for key in ("target_identity", "target_kappa", "equivalent_mode", "matrix_xAg", "N_6h_m-3", "radius_nm", "branch")}, **physicality(float(root["N_6h_m-3"]), float(root["radius_nm"])), "interpretation": "SCATTERING_EQUIVALENT_RADIUS_NOT_MATERIAL_INVENTORY_EQUIVALENT"})
    write_csv(args.out / "equivalent_radius_physicality_audit.csv", list(physical_rows[0]), physical_rows)

    # Plots use central E1/E2 curves and preserve all target identities as lines.
    grid = radius_grid()
    fig, axis = plt.subplots(figsize=(8.3, 5.4), constrained_layout=True)
    for mode, xag, color in (("E1, AQ host xAg=0.0069", YU_AQ_XAG, "#1f77b4"), ("E2, 6 h matrix xAg=0.0062", TAILORING_6H_XAG, "#d62728")):
        values = [calculator.kappa(TEMPERATURE_K, xag, [(TAILORING_6H_N, float(radius))]) for radius in grid]
        axis.plot(grid, values, color=color, lw=1.4, label=mode)
    for target in targets:
        axis.axhline(float(target["target_kappa"]), lw=1.0, ls="--", label=str(target["target_identity"]))
    axis.set_xscale("log")
    axis.set_xlabel("monodisperse 6 h scattering-equivalent radius (nm)")
    axis.set_ylabel(r"$\kappa_{lat}$ (W m$^{-1}$ K$^{-1}$)")
    axis.set_title("Tailoring 6 h density in frozen Yu AQ host, 300.00 K")
    axis.legend(fontsize=7, ncol=2)
    axis.grid(alpha=0.25)
    fig.savefig(args.out / "kappa_6h_vs_radius.png", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8.3, 5.4), constrained_layout=True)
    zoom = np.arange(0.1, 12.001, 0.01)
    for mode, xag, color in (("E1", YU_AQ_XAG, "#1f77b4"), ("E2", TAILORING_6H_XAG, "#d62728")):
        values = [calculator.kappa(TEMPERATURE_K, xag, [(TAILORING_6H_N, float(radius))]) for radius in zoom]
        axis.plot(zoom, values, color=color, label=mode)
    for target in targets:
        axis.axhline(float(target["target_kappa"]), lw=1.0, ls="--", label=str(target["target_identity"]))
    for root in root_details:
        axis.plot(float(root["radius_nm"]), float(root["kappa_W_mK"]), "ko", ms=3)
    axis.axvspan(8.0, 11.5, color="gray", alpha=0.16, label="current PF library 8--11.5 nm")
    axis.set_xlim(0.1, 12.0)
    axis.set_xlabel("radius (nm)")
    axis.set_ylabel(r"$\kappa_{lat}$ (W m$^{-1}$ K$^{-1}$)")
    axis.set_title("AQ targets and all central 6 h crossings")
    axis.legend(fontsize=7, ncol=2)
    axis.grid(alpha=0.25)
    fig.savefig(args.out / "AQ_target_intersections.png", dpi=180)
    plt.close(fig)

    primary = next(row for row in root_details if row["target_identity"] == "T1_YU_AQ_TWO_POPULATION_300K" and row["equivalent_mode"] == "E1")
    primary_physical = physicality(TAILORING_6H_N, float(primary["radius_nm"]))
    feasible_primary = primary_physical["inventory_status"] != "INVENTORY_OR_OBJECT_DEFINITION_CONFLICT"
    final_status = STATUS_PASS_YU if feasible_primary else STATUS_INVENTORY
    resolved_primary = [float(row["radius_nm"]) for row in root_details if row["target_identity"] == "T1_YU_AQ_TWO_POPULATION_300K" and row["equivalent_mode"] == "E1" and float(row["radius_nm"]) >= 8.0]
    library_primary = [radius for radius in resolved_primary if radius <= 11.5]
    decision = f"""# AQ-to-6 h scattering-equivalent-radius decision

## Answer

At 300.00 K, the frozen Yu AQ two-population calculation gives **{yu_all:.9f} W m^-1 K^-1**.  Its exact check at 303.06 K is {observed_303p06:.11f} W m^-1 K^-1, matching the existing qualified AQ calculation {expected_303p06:.11f} within relative error {yu_relative_error:.3e}.

The Yu AQ small-population-only total is {yu_small_only:.9f} W m^-1 K^-1 and the big-population-only total is {yu_big_only:.9f} W m^-1 K^-1.  From the stepwise sequence, adding small objects changes the host-plus-point-defect result by {yu_small_only - yu_step[(TEMPERATURE_K, 'HOST_PLUS_POINT_DEFECT')]:.9f} W m^-1 K^-1; adding big objects after small changes it by {yu_all - yu_small_only:.9f} W m^-1 K^-1.  These are sequential Matthiessen differences, not unique additive attribution shares.

Tailoring AQ does not provide the Yu-style unique pair of radius and density inputs.  Its literal T-A/T-B constructions therefore return a conductivity envelope, not a unique Tailoring calculation.  Figure 5c gives a digitized low-temperature estimate of AQ {tailoring_kappa['AQ_300K']:.6f} and 6 h {tailoring_kappa['6h_300K']:.6f} W m^-1 K^-1 at 300 K; both are short extrapolations below the plotted 30 degC point and retain a {FIGURE_DIGITIZATION_SIGMA:.3f} W m^-1 K^-1 digitization uncertainty.

For the primary target (Yu AQ at 300 K) in E1, the only scanned crossing is **R = {float(primary['radius_nm']):.9f} nm**, on the {primary['branch']} side at the median heat-carrying angular frequency.  There are no primary roots in the PF-resolved range (R >= 8 nm), and none in the current 8--11.5 nm library.

Its spherical proxy volume fraction is **{primary_physical['spherical_volume_fraction']:.6f}**, well above 0.03.  This violates the stated inventory/object-definition check.  It must not be proposed as a PF seed radius.  The only supported reading is `SCATTERING_EQUIVALENT_RADIUS_NOT_MATERIAL_INVENTORY_EQUIVALENT`.

## Decision

```text
final_status={final_status}
Yu_AQ_reproduction={'PASS' if yu_pass else 'FAIL'}
Tailoring_AQ_unique_kappa=false
primary_equivalence_exists=true
primary_resolved_PF_root_count={len(resolved_primary)}
primary_inventory_feasible={feasible_primary}
```

## Recommended next action

Do not convert this effective scattering root into the 6 h PF library.  If the project later seeks to model the low-temperature AQ-to-6 h thermal-conductivity contrast, introduce a separately identified unresolved Ag-rich scattering population (and its independent inventory/observation contract), then compare it against resolved-beta, strain, and any experimentally supplied dislocation background without retuning the frozen Yu host.
"""
    (args.out / "final_AQ_to_6h_equivalence_decision.md").write_text(decision, encoding="utf-8")
    yu_report = f"""# Yu AQ 300 K reproduction

The exact original Yu host-only reproduction function was used for the four staged AQ calculations.  The public AQ input is `A_N=1.5`, both Yu precipitate populations, and zero dislocation density.  No PF population, refit, or case-specific scale is used.

At 303.06 K the full AQ calculation is {observed_303p06:.12f} W m^-1 K^-1 versus the existing qualified value {expected_303p06:.12f}; relative error {yu_relative_error:.3e}.  This passes the prerequisite for the following frozen-formula equivalence diagnostic.
"""
    (args.out / "yu_aq_reproduction_report.md").write_text(yu_report, encoding="utf-8")
    terminal = "\n".join((
        f"yu_AQ_kappa_300K={yu_all:.12g}",
        f"yu_AQ_kappa_303p06K={observed_303p06:.12g}",
        f"yu_AQ_small_only_kappa_300K={yu_small_only:.12g}",
        f"yu_AQ_big_only_kappa_300K={yu_big_only:.12g}",
        f"yu_AQ_small_plus_big_kappa_300K={yu_all:.12g}",
        f"tailoring_AQ_experimental_kappa_300K={tailoring_kappa['AQ_300K']:.12g}",
        f"tailoring_6h_experimental_kappa_300K={tailoring_kappa['6h_300K']:.12g}",
        f"tailoring_AQ_6h_relative_difference_percent={100.0 * (tailoring_kappa['6h_300K'] - tailoring_kappa['AQ_300K']) / tailoring_kappa['AQ_300K']:.12g}",
        f"tailoring_AQ_literal_kappa_min_300K={min(ta_values + tb_values):.12g}",
        f"tailoring_AQ_literal_kappa_max_300K={max(ta_values + tb_values):.12g}",
        "equivalent_target_identity=T1_YU_AQ_TWO_POPULATION_300K",
        "equivalent_mode=E1",
        f"R_6h_eq_root_count={len([row for row in root_details if row['target_identity'] == 'T1_YU_AQ_TWO_POPULATION_300K' and row['equivalent_mode'] == 'E1'])}",
        "R_6h_eq_all_roots_nm=" + ";".join(f"{float(row['radius_nm']):.12g}" for row in root_details if row["target_identity"] == "T1_YU_AQ_TWO_POPULATION_300K" and row["equivalent_mode"] == "E1"),
        "R_6h_eq_resolved_roots_nm=" + ";".join(f"{value:.12g}" for value in resolved_primary),
        "R_6h_eq_library_roots_nm=" + ";".join(f"{value:.12g}" for value in library_primary),
        f"R_6h_eq_inventory_feasible={feasible_primary}",
        f"R_6h_eq_volume_fraction={primary_physical['spherical_volume_fraction']:.12g}",
        f"R_6h_eq_count_246cube={primary_physical['equivalent_count_246cube']:.12g}",
        f"R_6h_eq_count_512cube={primary_physical['equivalent_count_512cube']:.12g}",
        "equivalent_radius_interpretation=SCATTERING_EQUIVALENT_RADIUS_NOT_MATERIAL_INVENTORY_EQUIVALENT",
        "recommended_next_action=do_not_use_as_PF_seed; keep_unresolved_scattering_population_separate",
        f"final_status={final_status}",
        "",
    ))
    (args.out / "final_terminal_output.txt").write_text(terminal, encoding="utf-8")
    write_json(args.out / "audit_manifest.json", {
        "schema": "AQ_TO_6H_SCATTERING_EQUIVALENT_RADIUS_AUDIT_V1",
        "source_sha256": {str(path): sha256(path) for path in required},
        "dislocation_mode": "OFF",
        "pf_runs_started": False,
        "production_results_modified": False,
        "case_specific_parameter_adjustment": False,
        "final_status": final_status,
    })
    print(terminal, end="")


if __name__ == "__main__":
    main()
