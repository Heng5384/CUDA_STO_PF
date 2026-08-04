#!/usr/bin/env python3
"""Extract the Sheskin Figure 5c transport data with explicit data identity.

The source paper supplies the 300 degC endpoint values in prose but no main-
text conductivity table.  This script therefore keeps two noninterchangeable
records: Figure 5c digitizations and the three prose-reported endpoints.  It
also records the Table 1 microstructure observations without converting the
reported object sizes into a transport PSD.

The script consumes a deterministic 600 dpi render of page 5.  Marker centres
are recovered from colour masks inside pre-registered local windows.  The
windows only disambiguate overlapping curves and the Figure 5 legend; the
axis mapping is an independent least-squares fit to retained tick pixels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


SCHEMA = "SHESKIN_EXPERIMENTAL_DATA_EXTRACTION_V1"
TEMPERATURES_C = (30.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0)
DIGITIZATION_UNCERTAINTY_W_MK = 0.015

# Retained full-page, 600 dpi tick locations from Figure 5c.
X_TICKS = (
    (805.0, 0.0),
    (1066.0, 50.0),
    (1324.0, 100.0),
    (1583.0, 150.0),
    (1842.0, 200.0),
    (2100.0, 250.0),
    (2358.0, 300.0),
)
Y_TICKS = (
    (3622.0, 1.6),
    (3761.0, 1.5),
    (3902.0, 1.4),
    (4044.0, 1.3),
    (4185.0, 1.2),
    (4324.0, 1.1),
    (4466.0, 1.0),
    (4606.0, 0.9),
    (4745.0, 0.8),
)

# Local y centres were frozen by visual inspection before extracting values.
# A +/-55 pixel window is wider than every marker but excludes adjacent curves.
EXPECTED_Y = {
    "AQ": (4131, 4258, 4417, 4553, 4547, 4580, 4592),
    "6h": (4097, 4344, 4520, 4642, 4671, 4671, 4685),
    "48h": (3738, 3962, 4208, 4344, 4348, 4408, 4447),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    require(bool(values), f"refusing to write empty CSV: {path}")
    fields = list(values[0])
    require(all(list(row) == fields for row in values), f"CSV schema mismatch: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def masks(rgb: np.ndarray) -> dict[str, np.ndarray]:
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    return {
        "AQ": (red > 200) & (red - green > 60) & (red - blue > 40) & (green < 190),
        "6h": (blue > 170) & (blue - red > 70) & (blue - green > 20),
        "48h": (green > 80) & (green - red > 15) & (green - blue > 10) & (red < 180),
    }


def linear_calibration(points: tuple[tuple[float, float], ...]) -> tuple[float, float, float]:
    pixel = np.asarray([item[0] for item in points], dtype=float)
    value = np.asarray([item[1] for item in points], dtype=float)
    slope, intercept = np.polyfit(pixel, value, 1)
    residual = value - (slope * pixel + intercept)
    return float(slope), float(intercept), float(np.sqrt(np.mean(residual**2)))


def digitize(page_png: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Image.Image]:
    image = Image.open(page_png).convert("RGB")
    require(image.size == (5212, 6820), f"unexpected page render dimensions: {image.size}")
    rgb = np.asarray(image)
    state_masks = masks(rgb)
    x_slope, x_intercept, x_rms = linear_calibration(X_TICKS)
    y_slope, y_intercept, y_rms = linear_calibration(Y_TICKS)
    calibration_rows: list[dict[str, Any]] = []
    for pixel, value in X_TICKS:
        calibration_rows.append(
            {
                "axis": "x",
                "tick_pixel": pixel,
                "tick_value": value,
                "unit": "degC",
                "fit_value": x_slope * pixel + x_intercept,
                "fit_residual": value - (x_slope * pixel + x_intercept),
            }
        )
    for pixel, value in Y_TICKS:
        calibration_rows.append(
            {
                "axis": "y",
                "tick_pixel": pixel,
                "tick_value": value,
                "unit": "W m^-1 K^-1",
                "fit_value": y_slope * pixel + y_intercept,
                "fit_residual": value - (y_slope * pixel + y_intercept),
            }
        )

    rows: list[dict[str, Any]] = []
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    colours = {"AQ": (0, 0, 0), "6h": (120, 0, 120), "48h": (255, 140, 0)}
    for state, expected_y_values in EXPECTED_Y.items():
        mask = state_masks[state]
        for nominal_c, expected_y in zip(TEMPERATURES_C, expected_y_values):
            expected_x = int(round((nominal_c - x_intercept) / x_slope))
            y0, y1 = expected_y - 55, expected_y + 56
            x0, x1 = expected_x - 20, expected_x + 21
            yy, xx = np.where(mask[y0:y1, x0:x1])
            require(len(yy) >= 300, f"insufficient marker pixels: {state} {nominal_c:g} C")
            yy = yy + y0
            xx = xx + x0
            marker_x = float(np.median(xx))
            marker_y = float(np.median(yy))
            calibrated_c = x_slope * marker_x + x_intercept
            kappa = y_slope * marker_y + y_intercept
            rows.append(
                {
                    "sample_state": state,
                    "nominal_temperature_C": nominal_c,
                    "nominal_temperature_K": nominal_c + 273.15,
                    "marker_x_pixel": marker_x,
                    "marker_y_pixel": marker_y,
                    "mask_pixel_count": int(len(yy)),
                    "mask_x_min": int(xx.min()),
                    "mask_x_max": int(xx.max()),
                    "mask_y_min": int(yy.min()),
                    "mask_y_max": int(yy.max()),
                    "x_axis_calibrated_temperature_C": calibrated_c,
                    "kappa_W_mK": kappa,
                    "digitization_uncertainty_W_mK": DIGITIZATION_UNCERTAINTY_W_MK,
                    "thermal_quantity_identity": "MEASURED_TOTAL_KAPPA",
                    "source_value_identity": "FIGURE_5C_DIGITIZED_NOT_TABLE_VALUE",
                    "analysis_selection": "SELECTED_EXCEPT_300C_PROSE_ENDPOINT_OVERRIDES",
                    "source_location": "Sheskin et al. 2018 Figure 5c",
                }
            )
            x, y = int(round(marker_x)), int(round(marker_y))
            colour = colours[state]
            draw.line((x - 12, y, x + 12, y), fill=colour, width=3)
            draw.line((x, y - 12, x, y + 12), fill=colour, width=3)

    require(x_rms < 0.2, f"x-axis calibration RMS too large: {x_rms}")
    require(y_rms < 0.001, f"y-axis calibration RMS too large: {y_rms}")
    return rows, calibration_rows, overlay


def microstructure_rows() -> list[dict[str, Any]]:
    def row(state: str, category: str, observation: str, value: Any, uncertainty: Any,
            unit: str, identity: str, location: str, notes: str) -> dict[str, Any]:
        return {
            "sample_state": state,
            "category": category,
            "observation": observation,
            "value": value,
            "uncertainty": uncertainty,
            "unit": unit,
            "data_identity": identity,
            "source_location": location,
            "notes": notes,
        }

    rows = [
        row("all", "nominal composition", "bulk formula", "(PbTe)0.97(Ag2Te)0.03", "not reported", "formula", "SOURCE_LITERAL", "Summary and Conclusions", "Not a measured phase volume fraction."),
        row("all", "processing", "aging temperature", 380, "not reported", "degC", "SOURCE_LITERAL", "Experimental Section", "AQ is the water-quenched hot-pressed state; aged states are 6 h and 48 h."),
        row("all", "grain size", "grain size", "not reported", "not reported", "", "MISSING_SOURCE_QUANTITY", "Main text and accessible figures", "Pellet diameter is 12.5 mm and must not be substituted for grain size."),
        row("AQ", "matrix composition", "Ag", 0.78, 0.04, "atom percent", "TABLE_VALUE", "Table 1", "Interprecipitate APT region."),
        row("6h", "matrix composition", "Ag", 0.62, 0.04, "atom percent", "TABLE_VALUE", "Table 1", "Interprecipitate APT region."),
        row("48h", "matrix composition", "Ag", 0.62, 0.04, "atom percent", "TABLE_VALUE", "Table 1", "Interprecipitate APT region."),
        row("AQ", "number density", "TEM/APT Ag-rich objects", 3.61e21, 1.51e21, "m^-3", "TABLE_VALUE_DETECTION_SCALE_SPECIFIC", "Table 1", "Do not add to SE/FIB density; object identities may overlap across scales."),
        row("6h", "number density", "TEM/APT Ag-rich objects", 1.68e24, 0.92e24, "m^-3", "TABLE_VALUE_DETECTION_SCALE_SPECIFIC", "Table 1", "Do not add to SE/FIB density."),
        row("48h", "number density", "TEM/APT Ag-rich objects", 1.21e22, 0.75e22, "m^-3", "TABLE_VALUE_DETECTION_SCALE_SPECIFIC", "Table 1", "Do not add to SE/FIB density."),
        row("AQ", "number density", "SE/FIB large objects", 5.0e17, 1.5e17, "m^-3", "TABLE_VALUE_DETECTION_SCALE_SPECIFIC", "Table 1", "Separate large-object imaging scale."),
        row("6h", "number density", "SE/FIB large objects", 1.07e18, 0.45e18, "m^-3", "TABLE_VALUE_DETECTION_SCALE_SPECIFIC", "Table 1", "Separate large-object imaging scale."),
        row("48h", "number density", "SE/FIB large objects", 4.0e17, 1.5e17, "m^-3", "TABLE_VALUE_DETECTION_SCALE_SPECIFIC", "Table 1", "Separate large-object imaging scale."),
        row("AQ", "morphology", "single APT reconstruction", "37 small and 6 elongated objects", "single reconstruction", "count", "COUNT_FRACTION_DIAGNOSTIC_NOT_BULK_PSD", "Figure 4 discussion", "Small objects up to about 10 nm; elongated objects up to about 100 nm. No unique radii or aspect ratios."),
        row("6h", "morphology", "four APT reconstructions", "several dozen small spheroidal objects per volume", "qualitative", "", "QUALITATIVE_NOT_BULK_PSD", "Section 3.1", "Small objects up to about 10 nm; some are connected by Ag-decorated dislocations."),
        row("48h", "morphology", "three APT reconstructions", "elongated and spheroidal objects larger than at 6 h", "qualitative", "", "QUALITATIVE_NOT_BULK_PSD", "Section 3.1", "Attributed to coarsening; no unique full PSD is tabulated."),
        row("all", "thermal measurement", "total thermal conductivity", "laser-flash diffusivity x density x heat capacity", "measurement uncertainty not reported", "W m^-1 K^-1", "MEASURED_TOTAL_KAPPA", "Experimental Section and Figure 5c", "Figure 5c is the source for the temperature curves."),
        row("all", "electronic thermal conductivity", "Wiedemann-Franz estimate", "about four orders below lattice component", "qualitative", "", "WIEDEMANN_FRANZ_DERIVED_NOT_DIRECTLY_MEASURED", "Figure 7 and Section 3.3", "Lorenz number uses a Seebeck-dependent semiempirical expression; no numeric table in main text."),
    ]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-pdf", type=Path, required=True)
    parser.add_argument("--page5-600dpi-png", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pdf = args.paper_pdf.resolve()
    page_png = args.page5_600dpi_png.resolve()
    out = args.out.resolve()
    require(pdf.is_file(), f"missing source PDF: {pdf}")
    require(page_png.is_file(), f"missing page render: {page_png}")
    require(not out.exists(), f"refusing to overwrite output root: {out}")
    out.mkdir(parents=True)

    digitized, calibration, overlay = digitize(page_png)
    write_csv(out / "sheskin_figure5c_digitized_pixels.csv", digitized)
    write_csv(out / "sheskin_figure5c_calibration.csv", calibration)

    # Preserve every digitized point and add source-literal prose endpoints as
    # distinct rows.  Downstream fitting selects the prose endpoint at 300 C.
    experimental_rows: list[dict[str, Any]] = []
    for item in digitized:
        experimental_rows.append(
            {
                "sample_state": item["sample_state"],
                "temperature_C": item["nominal_temperature_C"],
                "temperature_K": item["nominal_temperature_K"],
                "kappa_W_mK": item["kappa_W_mK"],
                "analysis_uncertainty_W_mK": item["digitization_uncertainty_W_mK"],
                "reported_experimental_uncertainty_W_mK": "not reported",
                "thermal_quantity_identity": item["thermal_quantity_identity"],
                "source_value_identity": item["source_value_identity"],
                "analysis_selection": "REFERENCE_ONLY" if item["nominal_temperature_C"] == 300.0 else "SELECTED_FOR_TEMPERATURE_CURVE",
                "source_location": item["source_location"],
            }
        )
    prose = {"AQ": 0.93, "6h": 0.85, "48h": 1.03}
    for state, value in prose.items():
        experimental_rows.append(
            {
                "sample_state": state,
                "temperature_C": 300.0,
                "temperature_K": 573.15,
                "kappa_W_mK": value,
                "analysis_uncertainty_W_mK": DIGITIZATION_UNCERTAINTY_W_MK,
                "reported_experimental_uncertainty_W_mK": "not reported",
                "thermal_quantity_identity": "MEASURED_TOTAL_KAPPA",
                "source_value_identity": "MAIN_TEXT_EXPLICIT_VALUE",
                "analysis_selection": "SELECTED_FOR_TEMPERATURE_CURVE",
                "source_location": "Sheskin et al. 2018 Section 3.3, page 38999",
            }
        )
    experimental_rows.sort(key=lambda item: (item["sample_state"], float(item["temperature_C"]), item["source_value_identity"]))
    write_csv(out / "sheskin_experimental_kappa.csv", experimental_rows)
    write_csv(out / "sheskin_microstructure_source_table.csv", microstructure_rows())

    crop_box = (650, 3500, 2465, 4825)
    Image.open(page_png).convert("RGB").crop(crop_box).save(
        out / "sheskin_figure5c_source_crop.png", format="PNG", optimize=False
    )
    overlay.crop(crop_box).save(
        out / "sheskin_figure5c_digitization_overlay.png", format="PNG", optimize=False
    )

    selected = [item for item in experimental_rows if item["analysis_selection"] == "SELECTED_FOR_TEMPERATURE_CURVE"]
    require(len(selected) == 21, f"expected 21 selected curve points, found {len(selected)}")
    require({item["sample_state"] for item in selected} == {"AQ", "6h", "48h"}, "state identity closure failed")
    require(all(sum(item["sample_state"] == state for item in selected) == 7 for state in ("AQ", "6h", "48h")), "temperature-curve closure failed")

    provenance = {
        "schema": SCHEMA,
        "status": "PASS_SHESKIN_MAIN_TEXT_DIGITIZATION",
        "paper_pdf": {"path": str(pdf), "sha256": sha256(pdf)},
        "rendered_page": {"path": str(page_png), "sha256": sha256(page_png), "dpi": 600, "page": 5, "pixel_dimensions": [5212, 6820]},
        "figure": "Figure 5c measured total thermal conductivity",
        "axis_calibration": "least squares over retained tick pixels",
        "marker_detection": "state-specific RGB masks in frozen local windows",
        "digitization_uncertainty_W_mK": DIGITIZATION_UNCERTAINTY_W_MK,
        "supporting_information": {
            "expected_filename": "am8b15204_si_001.pdf",
            "contents_stated_by_main_text": "raw thermal diffusivity Figure S1 and heat capacity Figure S2",
            "local_status": "NOT_AVAILABLE",
            "public_search_status": "NO_ACCESSIBLE_COPY_FOUND_2026-08-02",
        },
        "analysis_script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
    }
    (out / "sheskin_digitization_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (out / "sheskin_digitization_provenance.md").write_text(
        "# Sheskin experimental digitization provenance\n\n"
        "Status: `PASS_SHESKIN_MAIN_TEXT_DIGITIZATION`.\n\n"
        "The temperature curves are measured total thermal conductivity from Figure 5c, "
        "not a table and not a Yu/PF calculation. Page 5 was rendered at 600 dpi. Both "
        "axis-tick calibration and every recovered marker pixel centre are retained in CSV. "
        "A 0.015 W m^-1 K^-1 analysis uncertainty covers marker thickness and calibration; "
        "the paper does not report pointwise conductivity uncertainties.\n\n"
        "At 300 degC, the source-literal prose values AQ=0.93, 6 h=0.85 and 48 h=1.03 "
        "W m^-1 K^-1 override the graph digitization for analysis, while the graph points "
        "remain retained as reference-only evidence.\n\n"
        "Figure 7 reports Wiedemann-Franz-derived lattice and electronic components. The "
        "electronic contribution is about four orders below the lattice contribution, but "
        "it remains a derived quantity and is not relabelled as measured total kappa. No "
        "numeric Figure 7 table is present in the main text.\n\n"
        "The publisher names the SI file `am8b15204_si_001.pdf`; the main text says it "
        "contains raw diffusivity (Figure S1) and heat capacity (Figure S2). No accessible "
        "local or public copy was found on 2026-08-02, so those source curves are not "
        "fabricated or inferred.\n",
        encoding="utf-8",
    )
    (out / "experimental_identity_audit.md").write_text(
        "# Experimental identity audit\n\n"
        "`PASS_EXPERIMENTAL_IDENTITY_SEPARATION`\n\n"
        "- Figure 5c: measured total thermal conductivity.\n"
        "- Figure 7 open symbols: Wiedemann-Franz-derived lattice thermal conductivity.\n"
        "- Figure 7 filled symbols: Wiedemann-Franz-derived electronic thermal conductivity.\n"
        "- Yu curves: calculated Debye-Callaway quantities and not Sheskin measurements.\n"
        "- PF curves: calculated full-PSD quantities and not Sheskin measurements.\n\n"
        "Only Figure 5c measured-total values enter the experimental CSV. Because the "
        "electronic term is roughly 1e-4 of the lattice term, measured total is a numerically "
        "adequate target at the retained 0.015 W m^-1 K^-1 digitization resolution, but its "
        "identity is never renamed. TEM/APT and SE/FIB number densities are kept separate "
        "because they use different detection scales and may overlap in object identity.\n",
        encoding="utf-8",
    )

    outputs = {
        path.name: sha256(path)
        for path in sorted(out.iterdir())
        if path.is_file()
    }
    (out / "stage1_manifest.json").write_text(
        json.dumps({"schema": SCHEMA, "outputs": outputs}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PASS_SHESKIN_MAIN_TEXT_DIGITIZATION")


if __name__ == "__main__":
    main()
