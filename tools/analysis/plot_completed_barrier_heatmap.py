#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_matrix(
    rows: list[dict[str, str]], metric: str
) -> tuple[list[float], list[float], np.ndarray]:
    temps = sorted({float(row["T_C"]) for row in rows})
    xbs = sorted({float(row["xB"]) for row in rows})
    lookup = {(float(row["T_C"]), float(row["xB"])): float(row[metric]) for row in rows}
    matrix = np.full((len(temps), len(xbs)), np.nan, dtype=float)
    for i, temp in enumerate(temps):
        for j, xb in enumerate(xbs):
            matrix[i, j] = lookup[(temp, xb)]
    return temps, xbs, matrix


def annotate(ax: plt.Axes, matrix: np.ndarray, fmt: str) -> None:
    nrows, ncols = matrix.shape
    for i in range(nrows):
        for j in range(ncols):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                format(value, fmt),
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                fontweight="semibold",
            )


def style_axis(ax: plt.Axes, title: str, temps: list[float], xbs: list[float]) -> None:
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xticks(range(len(xbs)), [f"{xb:.3f}" for xb in xbs], fontsize=10)
    ax.set_yticks(range(len(temps)), [f"{temp:.0f}" for temp in temps], fontsize=10)
    ax.set_xlabel("xB", fontsize=11)
    ax.set_ylabel("Temperature (°C)", fontsize=11)
    ax.set_xticks(np.arange(-0.5, len(xbs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(temps), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render publication-style heatmaps for a completed CNT barrier overview table."
    )
    parser.add_argument("--input", type=Path, required=True, help="overview csv")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="output path without extension",
    )
    args = parser.parse_args()

    rows = read_rows(args.input.expanduser().resolve())
    case_count = len(rows)
    temps, xbs, barrier = build_matrix(rows, "cnt_refsub_peak_kBT")
    _, _, rc = build_matrix(rows, "rc_schur_nm")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.8), dpi=220, constrained_layout=True)
    fig.patch.set_facecolor("white")

    im0 = axes[0].imshow(barrier, cmap="YlOrRd", aspect="auto")
    style_axis(axes[0], "CNT Refsub Barrier (kBT)", temps, xbs)
    annotate(axes[0], barrier, ".1f")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    cbar0.set_label("kBT", fontsize=10)

    im1 = axes[1].imshow(rc, cmap="GnBu", aspect="auto")
    style_axis(axes[1], "Schur Critical Radius (nm)", temps, xbs)
    annotate(axes[1], rc, ".3f")
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    cbar1.set_label("nm", fontsize=10)

    fig.suptitle(
        f"Ag2Te-PbTe No-Strain Completed CNT Cases ({case_count}-case overview)",
        fontsize=15,
        y=1.02,
    )

    prefix = args.output_prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(prefix.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)

    print(prefix.with_suffix(".png"))
    print(prefix.with_suffix(".pdf"))
    print(prefix.with_suffix(".svg"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
