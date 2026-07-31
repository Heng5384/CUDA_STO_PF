#!/usr/bin/env python3
"""Deterministically digitize the two dashed Yu-model curves in Figure 6b."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import label


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_temperatures(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [float(row["T_K"]) for row in csv.DictReader(handle)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-image", type=Path, required=True)
    parser.add_argument("--source-pdf", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/qualification/yu2024_transport_v1"),
    )
    args = parser.parse_args()

    image = np.asarray(Image.open(args.page_image).convert("RGB"))
    if image.shape[:2] != (2608, 1985):
        raise ValueError(
            f"Expected 240 dpi page render of 1985x2608 pixels, got {image.shape}"
        )

    # Figure 6b plot-box calibration in full-page pixel coordinates.
    x_left, x_right = 1101.0, 1564.0
    temperature_left, temperature_right = 275.0, 600.0
    y_top, y_bottom = 267.0, 622.0
    kappa_top, kappa_bottom = 3.0, 1.0

    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    navy = (blue > 90) & (blue < 150) & (red < 70) & (green < 70)
    roi = np.zeros_like(navy)
    roi[430:623, 1101:1565] = True
    components, count = label(navy & roi)

    pixel_rows: list[dict[str, object]] = []
    curves: dict[str, list[tuple[float, float]]] = {"AQ": [], "48h": []}
    curve_component_counts = {"AQ": 0, "48h": 0}
    for component_id in range(1, count + 1):
        ys, xs = np.where(components == component_id)
        if len(xs) < 20:
            continue
        x = float(xs.mean())
        y = float(ys.mean())
        # Empirical midpoint between the two visibly separated dashed curves.
        midpoint_y = 0.316 * (x - 1000.0) + 201.6 + 230.0
        state = "48h" if y < midpoint_y else "AQ"
        curve_component_counts[state] += 1
        temperature = temperature_left + (x - x_left) * (
            temperature_right - temperature_left
        ) / (x_right - x_left)
        kappa = kappa_bottom + (y_bottom - y) * (
            kappa_top - kappa_bottom
        ) / (y_bottom - y_top)
        for x_column in sorted(set(int(value) for value in xs)):
            column_ys = ys[xs == x_column]
            y_column = float(np.median(column_ys))
            temperature_column = temperature_left + (x_column - x_left) * (
                temperature_right - temperature_left
            ) / (x_right - x_left)
            kappa_column = kappa_bottom + (y_bottom - y_column) * (
                kappa_top - kappa_bottom
            ) / (y_bottom - y_top)
            curves[state].append((temperature_column, kappa_column))
            pixel_rows.append(
                {
                    "state": state,
                    "component_id": component_id,
                    "column_pixel_count": len(column_ys),
                    "x_px": x_column,
                    "y_median_px": f"{y_column:.6f}",
                    "T_K": f"{temperature_column:.8f}",
                    "kappa_lat_W_mK": f"{kappa_column:.8f}",
                }
            )

    if curve_component_counts["AQ"] < 10 or curve_component_counts["48h"] < 10:
        raise RuntimeError(
            f"Insufficient curve components: AQ={curve_component_counts['AQ']}, "
            f"48h={curve_component_counts['48h']}"
        )
    pixel_rows.sort(key=lambda row: (str(row["state"]), float(row["T_K"])))
    write_csv(
        args.data_dir / "figure6b_model_curve_pixels.csv",
        list(pixel_rows[0]),
        pixel_rows,
    )

    uncertainty = 2.0 * (kappa_top - kappa_bottom) / (y_bottom - y_top)
    for state, experiment_filename, output_filename in (
        ("AQ", "yu_AQ_kappaL_digitized.csv", "yu_AQ_model_digitized.csv"),
        ("48h", "yu_48h_kappaL_digitized.csv", "yu_48h_model_digitized.csv"),
    ):
        points = sorted(curves[state])
        point_temperature = np.array([point[0] for point in points])
        point_kappa = np.array([point[1] for point in points])
        interpolator = PchipInterpolator(point_temperature, point_kappa)
        rows = []
        for temperature in read_temperatures(args.data_dir / experiment_filename):
            if not point_temperature.min() <= temperature <= point_temperature.max():
                raise ValueError(f"{state} temperature {temperature} requires extrapolation")
            rows.append(
                {
                    "state": state,
                    "T_K": f"{temperature:.8g}",
                    "yu_model_kappa_lat_W_mK": (
                        f"{float(interpolator(temperature)):.12g}"
                    ),
                    "digitization_uncertainty_W_mK": f"{uncertainty:.12g}",
                    "source_location": "main text Figure 6b dashed Callaway curve",
                    "interpolation": "PCHIP_between_masked_curve_pixels",
                }
            )
        write_csv(args.data_dir / output_filename, list(rows[0]), rows)

    contract = {
        "source_pdf": str(args.source_pdf),
        "source_pdf_sha256": sha256(args.source_pdf),
        "rendered_page_image": str(args.page_image),
        "rendered_page_image_sha256": sha256(args.page_image),
        "source_pdf_page": 8,
        "render_contract": {
            "renderer": "Poppler pdftoppm",
            "dpi": 240,
            "color_mode": "PNG RGB",
            "expected_pixel_dimensions": [1985, 2608],
        },
        "plot_box_calibration": {
            "x_left_px": x_left,
            "x_right_px": x_right,
            "temperature_left_K": temperature_left,
            "temperature_right_K": temperature_right,
            "y_top_px": y_top,
            "y_bottom_px": y_bottom,
            "kappa_top_W_mK": kappa_top,
            "kappa_bottom_W_mK": kappa_bottom,
        },
        "curve_extraction": {
            "navy_mask": "90 < B < 150 and R < 70 and G < 70",
            "minimum_connected_component_pixels": 20,
            "legend_exclusion": "restrict y to 430:622 in full-page coordinates",
            "state_classification": (
                "connected-component centroid relative to the midpoint between "
                "the two dashed curves"
            ),
            "interpolation": "monotonic piecewise cubic PCHIP",
            "extrapolation": False,
            "AQ_dash_components": curve_component_counts["AQ"],
            "48h_dash_components": curve_component_counts["48h"],
        },
        "uncertainty_contract": {
            "assumed_vertical_pixel_uncertainty": 2.0,
            "kappa_uncertainty_W_mK": uncertainty,
            "note": (
                "This covers line thickness and antialiasing in the 240 dpi "
                "render; it does not represent experimental uncertainty."
            ),
        },
    }
    (args.data_dir / "figure6b_digitization_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "AQ_dash_components": curve_component_counts["AQ"],
                "48h_dash_components": curve_component_counts["48h"],
                "uncertainty_W_mK": uncertainty,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
