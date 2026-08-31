#!/usr/bin/env python3
"""Render evidence-bounded figures for the KWN--PF state-closure v1 report.

The figures intentionally distinguish host-only validation and storage controls
from a CUDA PF smoke.  They never infer a PF trajectory from KWN or host-state
results: missing CUDA trajectories are drawn as ``NOT RUN CUDA``.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "kwn_pf_state_closure_v1"
FIGURE_ROOT = OUTPUT_ROOT / "figures"

VALIDATION = "#2563eb"
HOST = "#15803d"
FIXTURE = "#7c3aed"
NOT_RUN = "#b91c1c"
NEUTRAL = "#475569"
BUCKET_COLORS = {
    "matrix": "#60a5fa",
    "GP": "#f59e0b",
    "beta_subgrid": "#a78bfa",
    "beta_resolved_fixed": "#10b981",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: object, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def floor_for_log(value: float) -> float:
    """Show an exact zero on logarithmic controls without hiding its label."""

    return max(abs(value), 1.0e-20)


def compact_hash(value: str | None) -> str:
    return (value or "unavailable")[:12]


def control_tag(fig: plt.Figure, label: str, color: str) -> None:
    fig.text(
        0.012,
        0.986,
        label,
        ha="left",
        va="top",
        fontsize=8,
        color="white",
        weight="bold",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": color, "edgecolor": color},
    )


def save(fig: plt.Figure, filename: str) -> Path:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    path = FIGURE_ROOT / filename
    fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    svg_path = path.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)
    return path


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    color: str,
    *,
    dashed: bool = False,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.5,
        edgecolor=color,
        facecolor="white",
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=9,
        color="#0f172a",
        wrap=True,
    )


def draw_arrow(ax: plt.Axes, source: tuple[float, float], target: tuple[float, float], color: str) -> None:
    ax.add_patch(
        FancyArrowPatch(
            source,
            target,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.25,
            color=color,
            connectionstyle="arc3,rad=0.0",
        )
    )


def plot_authority_map(
    thermo_summary: dict[str, Any],
    four_bucket_summary: dict[str, Any],
    fixture_validation: dict[str, Any],
    beta_summary: dict[str, Any],
) -> Path:
    fig, ax = plt.subplots(figsize=(12.5, 7.3))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    contract_hash = compact_hash(thermo_summary.get("contract_hash"))

    draw_box(
        ax,
        0.04,
        0.73,
        0.25,
        0.15,
        f"PF_KWN_VALIDATION_CONTRACT_V1\nhash {contract_hash}…\nVALIDATION CONTROL",
        VALIDATION,
    )
    draw_box(
        ax,
        0.37,
        0.73,
        0.25,
        0.15,
        "Python contract loader +\ngenerated PF C++ header\nshared coefficients/functions",
        VALIDATION,
    )
    draw_box(
        ax,
        0.70,
        0.73,
        0.25,
        0.15,
        f"Thermo parity\n{thermo_summary.get('status', 'UNKNOWN')}\n{thermo_summary.get('rows', 'unknown')} host probe rows",
        VALIDATION,
    )
    draw_arrow(ax, (0.29, 0.805), (0.37, 0.805), VALIDATION)
    draw_arrow(ax, (0.62, 0.805), (0.70, 0.805), VALIDATION)

    draw_box(
        ax,
        0.04,
        0.40,
        0.25,
        0.17,
        "96³ resolved-beta fixture\nraw profile fields hash-validated\nFIXTURE-CONDITIONED",
        FIXTURE,
    )
    draw_box(
        ax,
        0.37,
        0.40,
        0.25,
        0.17,
        "Four buckets\nmatrix | GP | beta-subgrid |\nfixed resolved beta\nfrozen auxiliary state",
        HOST,
    )
    draw_box(
        ax,
        0.70,
        0.40,
        0.25,
        0.17,
        "V6 checkpoint contract\nV2–V5 backward reads\nhost persistence control only",
        HOST,
    )
    draw_arrow(ax, (0.29, 0.485), (0.37, 0.485), FIXTURE)
    draw_arrow(ax, (0.62, 0.485), (0.70, 0.485), HOST)
    draw_arrow(ax, (0.495, 0.73), (0.495, 0.57), NEUTRAL)

    draw_box(
        ax,
        0.04,
        0.10,
        0.25,
        0.16,
        "Historical as-run authority\nUNRECOVERED\nconditional downstream PSD ≠ authority",
        NEUTRAL,
        dashed=True,
    )
    draw_box(
        ax,
        0.37,
        0.10,
        0.25,
        0.16,
        "Beta-only t0 sign\nshared-contract self-consistency\nKWN runtime incomplete",
        VALIDATION,
    )
    draw_box(
        ax,
        0.70,
        0.10,
        0.25,
        0.16,
        "96³ CUDA PF smoke A–E\nNOT RUN CUDA\nno PF trajectories inferred",
        NOT_RUN,
    )
    draw_arrow(ax, (0.495, 0.40), (0.495, 0.26), NEUTRAL)
    draw_arrow(ax, (0.825, 0.40), (0.825, 0.26), NOT_RUN)

    ax.text(
        0.5,
        0.955,
        "KWN–PF state-closure v1 evidence and authority map",
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
    )
    ax.text(
        0.5,
        0.02,
        "No GP release, GP→beta conversion, new beta seeding, online KWN call, or CUDA PF dynamics is represented here.",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#334155",
    )
    control_tag(fig, "EVIDENCE MAP — validation + host storage + fixture-conditioned inputs", NEUTRAL)
    return save(fig, "01_authority_map.png")


def pretty_relative_field(name: str) -> str:
    replacements = {
        "relative_error_G_alpha_J_mol": "Gα",
        "relative_error_mu_A_J_mol": "μA",
        "relative_error_mu_B_J_mol": "μB",
        "relative_error_dGdx_J_mol": "dG/dx",
        "relative_error_d2Gdx2_J_mol": "d²G/dx²",
        "relative_error_driving_force_beta_J_mol": "ΔGβ",
        "relative_error_D_alpha_m2_s": "Dα",
        "relative_error_curvature_equilibrium_xB": "xB curvature",
        "relative_error_Vm_alpha_m3_mol": "Vmα",
        "relative_error_Vm_beta_m3_mol": "Vmβ",
    }
    return replacements.get(name, name.removeprefix("relative_error_"))


def plot_thermo_parity(rows: list[dict[str, str]], summary: dict[str, Any]) -> Path:
    relative_fields = [key for key in rows[0] if key.startswith("relative_error_")]
    maxima = {key: max(as_float(row[key]) for row in rows) for key in relative_fields}
    labels = [pretty_relative_field(key) for key in relative_fields]
    values = [floor_for_log(maxima[key]) for key in relative_fields]
    tolerance = as_float(summary.get("relative_tolerance"), 1.0e-10)

    fig, (ax_error, ax_solvus) = plt.subplots(1, 2, figsize=(13.2, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
    ax_error.bar(range(len(values)), values, color=VALIDATION, edgecolor="#1e3a8a")
    ax_error.axhline(tolerance, color=NOT_RUN, linestyle="--", linewidth=1.3, label=f"relative tolerance = {tolerance:.0e}")
    ax_error.set_yscale("log")
    ax_error.set_xticks(
        range(len(values)), labels, rotation=35, ha="right", rotation_mode="anchor"
    )
    ax_error.set_ylabel("maximum relative error (zero shown at 1e-20 floor)")
    ax_error.set_title("Host C++ probe vs Python contract")
    ax_error.legend(loc="upper right", fontsize=8)
    for index, key in enumerate(relative_fields):
        value = maxima[key]
        text = "0" if value == 0.0 else f"{value:.1e}"
        ax_error.text(
            index,
            values[index] * 1.55,
            text,
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
            rotation_mode="anchor",
        )

    by_radius: dict[float, tuple[float, float, float]] = {}
    for row in rows:
        radius_nm = as_float(row["radius_m"]) * 1.0e9
        candidate = (
            as_float(row["curvature_equilibrium_xB"]),
            as_float(row["python_curvature_equilibrium_xB"]),
            as_float(row["absolute_error_planar_solvus_xB"]),
        )
        by_radius[radius_nm] = candidate
    radii = sorted(by_radius)
    cpp_solvus = [by_radius[radius][0] for radius in radii]
    py_solvus = [by_radius[radius][1] for radius in radii]
    planar_absolute_error = max(item[2] for item in by_radius.values())
    ax_solvus.plot(radii, cpp_solvus, "o-", color="#0f766e", label="PF generated header")
    ax_solvus.plot(radii, py_solvus, "x--", color="#7c3aed", label="Python contract")
    ax_solvus.set_xlabel("radius (nm)")
    ax_solvus.set_ylabel("curvature equilibrium xB")
    ax_solvus.set_title("Gibbs–Thomson / solvus parity")
    ax_solvus.legend(fontsize=8)
    ax_solvus.text(
        0.03,
        0.04,
        f"planar-solvus |ΔxB|max = {planar_absolute_error:.1e}\nlimit = {as_float(summary.get('solvus_absolute_tolerance'), 1e-8):.0e}",
        transform=ax_solvus.transAxes,
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eff6ff", "edgecolor": VALIDATION},
    )

    fig.suptitle(
        f"Thermodynamic parity — {summary.get('status', 'UNKNOWN')} (contract {compact_hash(summary.get('contract_hash'))}…)",
        fontsize=13,
        weight="bold",
        y=0.93,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    control_tag(fig, "VALIDATION CONTROL — host C++ probe, not CUDA PF dynamics", VALIDATION)
    return save(fig, "02_thermo_parity_and_solvus.png")


def scenario_label(scenario: str) -> str:
    labels = {
        "S1_ZERO_AUX_IDENTITY": "S1 zero auxiliary\nidentity",
        "S2_NONZERO_FROZEN_AUX_STORAGE_ONLY": "S2 nonzero frozen auxiliary\n(total intentionally increases)",
        "S2E_FIXTURE_CONDITIONED_REBALANCED_AUX": "S2E fixture-conditioned\nrebalanced auxiliary",
        "S3_MATRIX_TO_GP_AND_REVERSE": "S3 matrix→GP→reverse\nconservative transfer",
    }
    return labels.get(scenario, scenario)


def plot_four_bucket_ledger(rows: list[dict[str, str]]) -> Path:
    scenarios = [
        "S1_ZERO_AUX_IDENTITY",
        "S2_NONZERO_FROZEN_AUX_STORAGE_ONLY",
        "S2E_FIXTURE_CONDITIONED_REBALANCED_AUX",
        "S3_MATRIX_TO_GP_AND_REVERSE",
    ]
    rows_by_scenario: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        rows_by_scenario[row["scenario"]][row["bucket"]] = row

    fig, (ax_stack, ax_residual) = plt.subplots(1, 2, figsize=(13.2, 5.6), gridspec_kw={"width_ratios": [1.45, 0.8]})
    buckets = ["matrix", "GP", "beta_subgrid", "beta_resolved_fixed"]
    for index, scenario in enumerate(scenarios):
        items = rows_by_scenario[scenario]
        total = as_float(items["total"]["Q_B_mol"])
        left = 0.0
        for bucket in buckets:
            fraction = as_float(items.get(bucket, {}).get("Q_B_mol")) / total if total else 0.0
            ax_stack.barh(index, fraction, left=left, color=BUCKET_COLORS[bucket], edgecolor="white", height=0.68)
            if fraction >= 0.05:
                ax_stack.text(left + fraction / 2, index, f"{fraction * 100:.1f}%", ha="center", va="center", fontsize=8)
            left += fraction
        auxiliary_fraction = sum(
            as_float(items.get(bucket, {}).get("Q_B_mol")) / total for bucket in ("GP", "beta_subgrid")
        )
        if auxiliary_fraction > 0.0:
            ax_stack.text(1.015, index, f"aux = {auxiliary_fraction * 100:.3f}%", va="center", fontsize=7.5, color=HOST)
    ax_stack.set_yticks(range(len(scenarios)), [scenario_label(item) for item in scenarios])
    ax_stack.set_xlim(0, 1.20)
    ax_stack.set_xlabel("fraction of each scenario's four-bucket total B inventory")
    ax_stack.set_title("Four-bucket ledger allocations")
    ax_stack.legend(
        [plt.Rectangle((0, 0), 1, 1, color=BUCKET_COLORS[item]) for item in buckets],
        ["matrix", "GP", "beta subgrid", "fixed resolved beta"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.31),
        ncol=2,
        fontsize=8,
        frameon=False,
    )

    residuals: list[float] = []
    for scenario in scenarios:
        total_row = rows_by_scenario[scenario]["total"]
        residuals.append(as_float(total_row.get("ledger_relative_residual")))
    ax_residual.bar(range(len(scenarios)), [floor_for_log(value) for value in residuals], color=HOST)
    ax_residual.axhline(1.0e-10, color=NOT_RUN, linestyle="--", linewidth=1.2, label="Closure limit 1e-10")
    ax_residual.set_yscale("log")
    ax_residual.set_xticks(range(len(scenarios)), ["S1", "S2", "S2E", "S3"])
    ax_residual.set_ylabel("four-bucket relative residual\n(zero shown at 1e-20 floor)")
    ax_residual.set_title("Ledger closure")
    ax_residual.legend(fontsize=8)
    for index, value in enumerate(residuals):
        ax_residual.text(index, floor_for_log(value) * 1.6, "0" if value == 0.0 else f"{value:.1e}", ha="center", fontsize=7)

    fig.suptitle("Four-bucket storage controls — no PF dynamics executed", fontsize=13, weight="bold", y=0.93)
    fig.tight_layout(rect=(0, 0.05, 1, 0.86))
    control_tag(fig, "HOST STORAGE CONTROL — NOT CUDA PF smoke", HOST)
    return save(fig, "03_four_bucket_ledger_scenarios.png")


def plot_identity_and_roundtrip(summary: dict[str, Any]) -> Path:
    s1 = summary["S1_zero_aux_identity"]
    s3 = summary["S3_exact_matrix_to_GP_inverse_transfer_and_reverse"]
    preservation = s1["source_field_preservation"]
    identity = [
        ("phi max |Δ|", as_float(preservation["source_phi_max_abs_difference"])),
        ("xB_alpha max |Δ|", as_float(preservation["source_xB_alpha_max_abs_difference"])),
        ("fixed resolved\ninventory rel. residual", as_float(preservation["resolved_inventory_relative_residual"])),
        ("mapped matrix\nmax |ΔxB|", as_float(s1["max_mapped_matrix_xB_difference_from_source"])),
    ]
    roundtrip = [
        ("forward matrix\nledger residual", as_float(s3["forward_matrix_inverse_audit"]["relative_residual"])),
        ("matrix xB\nroundtrip error", as_float(s3["max_xB_roundtrip_error"])),
        ("reverse matrix\nledger residual", as_float(s3["reverse_relative_residual"])),
        ("clipping used", as_float(s3["forward_matrix_inverse_audit"]["clipping_used"])),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.7, 5.0))
    for ax, data, title, color in (
        (axes[0], identity, "S1 zero-aux identity", VALIDATION),
        (axes[1], roundtrip, "S3 exact matrix↔GP roundtrip", HOST),
    ):
        values = [floor_for_log(value) for _, value in data]
        ax.bar(range(len(data)), values, color=color)
        ax.axhline(1.0e-10, color=NOT_RUN, linestyle="--", linewidth=1.2, label="Control limit 1e-10")
        ax.set_yscale("log")
        ax.set_xticks(range(len(data)), [label for label, _ in data])
        ax.set_ylabel("absolute / relative control quantity\n(zero shown at 1e-20 floor)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        for index, (_, value) in enumerate(data):
            ax.text(index, values[index] * 1.7, "0" if value == 0.0 else f"{value:.1e}", ha="center", fontsize=8)

    fig.suptitle("Field identity and conservative matrix↔GP roundtrip", fontsize=13, weight="bold", y=0.93)
    fig.text(
        0.5,
        0.015,
        "The fixture fields are preserved in the host control; this is not a PF evolution, matrix-pulse, or CUDA restart result.",
        ha="center",
        fontsize=8.5,
        color="#334155",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    control_tag(fig, "HOST STORAGE CONTROL — field-preserving transfer audit", HOST)
    return save(fig, "04_identity_roundtrip_residual.png")


def plot_handoff_constraint_feasibility(
    fixture_validation: dict[str, Any], fixture_metadata: dict[str, Any]
) -> Path:
    """Show the closed, compact v2 source allocation without implying a PF run."""

    ledger = fixture_validation["ledger"]
    audit = fixture_metadata["matrix_field_mapping"]["audit"]
    rows = [
        ("Total source", "Q_B_total_mol", "C_B_total_mol_m3", "fixed source inventory"),
        (
            "Fixed resolved β",
            "Q_B_beta_resolved_fixed_mol",
            "C_B_beta_resolved_fixed_mol_m3",
            "fixture geometry; counted once",
        ),
        ("Matrix", "Q_B_matrix_mol", "C_B_matrix_mol_m3", "inverse storage mapping"),
        ("Frozen GP", "Q_B_GP_mol", "C_B_GP_mol_m3", "prescribed, non-predictive"),
        (
            "β subgrid",
            "Q_B_beta_subgrid_mol",
            "C_B_beta_subgrid_mol_m3",
            "frozen compact PSD",
        ),
    ]
    cell_rows = [
        [name, f"{as_float(ledger[q_key]):.6e}", f"{as_float(ledger[c_key]):.6g}", rule]
        for name, q_key, c_key, rule in rows
    ]

    fig, (ax_table, ax_margin) = plt.subplots(
        1, 2, figsize=(13.2, 5.4), gridspec_kw={"width_ratios": [1.68, 0.82]}
    )
    ax_table.set_axis_off()
    table = ax_table.table(
        cellText=cell_rows,
        colLabels=["Bucket", "Q_B (mol)", "C_B (mol m⁻³)", "Constraint"],
        colWidths=[0.22, 0.22, 0.20, 0.36],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.65)
    for column in range(4):
        table[(0, column)].set_facecolor("#ede9fe")
        table[(0, column)].set_text_props(weight="bold")
    for column in range(4):
        table[(2, column)].set_facecolor("#ecfdf5")
        table[(3, column)].set_facecolor("#eff6ff")
        table[(4, column)].set_facecolor("#fffbeb")
    ax_table.set_title("Fixture-conditioned v2 constraint ledger", fontsize=12, weight="bold", pad=12)
    ax_table.text(
        0.0,
        0.07,
        f"ledger relative residual = {as_float(ledger['relative_residual']):.2e}; "
        f"matrix inverse residual = {as_float(audit['relative_residual']):.2e}; "
        "resolved β is not re-emitted from KWN.",
        transform=ax_table.transAxes,
        fontsize=8,
        color="#334155",
    )

    margins = [
        ("lower xB margin", as_float(audit["lower_bound_margin"])),
        ("upper xB margin", as_float(audit["upper_bound_margin"])),
        ("clipping used", as_float(audit["clipping_used"])),
    ]
    values = [floor_for_log(value) for _, value in margins]
    ax_margin.bar(range(len(margins)), values, color=[FIXTURE, FIXTURE, HOST])
    ax_margin.set_yscale("log")
    ax_margin.set_xticks(range(len(margins)), [label for label, _ in margins], rotation=25, ha="right", rotation_mode="anchor")
    ax_margin.set_ylabel("dimensionless bound margin / flag\n(zero shown at 1e-20 floor)")
    ax_margin.set_title("Matrix feasibility margins")
    for index, (_, value) in enumerate(margins):
        ax_margin.text(
            index,
            values[index] * 1.6,
            "0" if value == 0.0 else f"{value:.2e}",
            ha="center",
            fontsize=8,
        )
    ax_margin.text(
        0.5,
        0.04,
        "Package validation passed; it contains no dense PF raw fields.",
        transform=ax_margin.transAxes,
        ha="center",
        fontsize=7.5,
        color=NOT_RUN,
    )

    fig.suptitle("KWN→PF handoff v2 — feasible compact allocation, not a GP prediction", fontsize=13, weight="bold", y=0.94)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    control_tag(fig, "FIXTURE-CONDITIONED — NON-PREDICTIVE PRESCRIBED SOURCE", FIXTURE)
    return save(fig, "08_handoff_constraint_feasibility.png")


def plot_checkpoint_provenance(summary: dict[str, Any]) -> Path:
    checkpoint = summary["S4_cpp_V6_checkpoint_dependency"]
    fig, ax = plt.subplots(figsize=(12.2, 5.5))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.92, "PF checkpoint provenance — host C++ persistence contract", ha="center", fontsize=14, weight="bold")

    versions = [
        ("V2–V4", "legacy read supported\nzero auxiliary default", "legacy / unbound"),
        ("V5", "legacy read supported\npackage identity absent", "active-aux restart rejected"),
        ("V6", "four buckets + PSD bins\nwrite / read persistence", "validation contract hash\n+ package provenance"),
    ]
    positions = [0.04, 0.36, 0.68]
    for index, ((version, behavior, provenance), x) in enumerate(zip(versions, positions)):
        color = HOST if version == "V6" else NEUTRAL
        draw_box(ax, x, 0.43, 0.22, 0.28, f"{version}\n{behavior}\n\n{provenance}", color, dashed=version != "V6")
        if index:
            draw_arrow(ax, (positions[index - 1] + 0.28, 0.55), (x, 0.55), NEUTRAL)

    ax.text(
        0.5,
        0.25,
        checkpoint.get("status", "UNKNOWN"),
        ha="center",
        va="center",
        color="white",
        weight="bold",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": HOST, "edgecolor": HOST},
    )
    ax.text(
        0.5,
        0.12,
        "Host persistence/backward-read evidence only. CUDA PF restart smoke: NOT RUN CUDA.",
        ha="center",
        va="center",
        fontsize=10,
        color=NOT_RUN,
        weight="bold",
    )
    control_tag(fig, "HOST STORAGE CONTROL — make test_pf_zero_mode_checkpoint", HOST)
    return save(fig, "05_checkpoint_provenance_host_only.png")


def plot_beta_direction_and_runtime(rows: list[dict[str, str]], summary: dict[str, Any]) -> Path:
    direction_rows = [row for row in rows if row["record_type"] == "t0_size_class_direction"]
    runtime_rows = [row for row in rows if row["record_type"] == "kwn_time_snapshot"]
    failure_row = next((row for row in runtime_rows if row["kwn_run_status"] == "FAILED_KWN_RUNTIME"), None)

    fig, (ax_direction, ax_timeline) = plt.subplots(1, 2, figsize=(13.2, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    radii_nm = [as_float(row["size_class_radius_m"]) * 1.0e9 for row in direction_rows]
    rates_nm_per_h = [as_float(row["kwn_growth_rate_m_s"]) * 1.0e9 * 3600.0 for row in direction_rows]
    colors = ["#dc2626" if value < 0 else "#16a34a" for value in rates_nm_per_h]
    ax_direction.bar(range(len(radii_nm)), rates_nm_per_h, color=colors)
    ax_direction.axhline(0.0, color="#0f172a", linewidth=1.0)
    ax_direction.set_xticks(range(len(radii_nm)), [f"{radius:.2f}" for radius in radii_nm])
    ax_direction.set_xlabel("initial fixed-resolved size class radius (nm)")
    ax_direction.set_ylabel("KWN t0 growth rate (nm h⁻¹)")
    ax_direction.set_title("Exact-contract curvature direction at t0")
    ax_direction.set_ylim(min(rates_nm_per_h) - 0.24, max(rates_nm_per_h) + 0.25)
    for index, row in enumerate(direction_rows):
        direction = row["pf_contract_direction"]
        ax_direction.text(
            index,
            rates_nm_per_h[index] + (0.05 if rates_nm_per_h[index] >= 0 else -0.09),
            f"shared-contract sign: {direction}\nKWN t0 agrees={row['direction_match']}",
            ha="center",
            va="bottom" if rates_nm_per_h[index] >= 0 else "top",
            fontsize=8,
        )

    ax_timeline.axhline(0.5, color="#94a3b8", linewidth=2)
    ax_timeline.scatter([0.0], [0.5], color=VALIDATION, s=80, zorder=3)
    ax_timeline.annotate(
        "t0 evaluated\nshared-contract sign check",
        xy=(0.0, 0.5),
        xytext=(2.2, 0.83),
        ha="left",
        va="top",
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": VALIDATION, "linewidth": 1.0},
    )
    failure_time = as_float(failure_row["kwn_time_h"]) if failure_row else math.nan
    if math.isfinite(failure_time):
        ax_timeline.scatter([failure_time], [0.5], color=NOT_RUN, marker="X", s=100, zorder=3)
        ax_timeline.annotate(
            f"KWN runtime stopped\n{failure_time:.3f} h",
            xy=(failure_time, 0.5),
            xytext=(2.2, 0.16),
            ha="left",
            va="bottom",
            fontsize=8,
            color=NOT_RUN,
            arrowprops={"arrowstyle": "-", "color": NOT_RUN, "linewidth": 1.0},
        )
    for requested in (6.0, 48.0):
        ax_timeline.scatter([requested], [0.5], facecolors="none", edgecolors=NOT_RUN, marker="s", s=90, linewidths=1.8, zorder=3)
        ax_timeline.text(requested, 0.72, f"{requested:.0f} h\nnot reached", ha="center", fontsize=8, color=NOT_RUN)
    ax_timeline.set_xlim(-1.0, 52.0)
    ax_timeline.set_ylim(0.05, 1.0)
    ax_timeline.set_yticks([])
    ax_timeline.set_xlabel("requested code-control time (h)")
    ax_timeline.set_title("Runtime boundary — no actual PF trajectory comparison")
    ax_timeline.text(
        0.5,
        0.06,
        f"CUDA PF: {summary['pf_cuda_trajectory']['status']}",
        transform=ax_timeline.transAxes,
        ha="center",
        fontsize=9,
        color=NOT_RUN,
        weight="bold",
    )

    fig.suptitle(
        f"Beta-only same-contract control — {summary.get('status', 'UNKNOWN')}",
        fontsize=13,
        weight="bold",
        y=0.93,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    control_tag(fig, "VALIDATION CONTROL — t0 directions only; runtime incomplete", VALIDATION)
    return save(fig, "06_beta_t0_directions_runtime_incomplete.png")


def plot_smoke_matrix(fixture_validation: dict[str, Any]) -> Path:
    cases = [
        ("A", "BASELINE_LEGACY_ZERO_AUX", "No host PF trajectory record"),
        ("B", "IDENTITY_ADAPTER_ZERO_AUX", "S1 host zero-aux identity"),
        ("C", "NONZERO_FROZEN_AUX_STORAGE", "S2 host frozen-aux storage"),
        ("D", "CONSERVATIVE_MATRIX_TO_GP_CONTROL", "S3 host conservative transfer"),
        ("E", "FIXTURE_CONDITIONED_KWN_HANDOFF_V2", "V2 package passed; no dense raw fields are embedded"),
    ]
    fig, (ax_matrix, ax_note) = plt.subplots(1, 2, figsize=(13.5, 6.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax_matrix.set_xlim(0, 3)
    ax_matrix.set_ylim(0, len(cases))
    ax_matrix.invert_yaxis()
    ax_matrix.set_xticks([0.5, 1.5, 2.5], ["0 h", "6 h", "48 h"])
    ax_matrix.xaxis.tick_top()
    ax_matrix.set_yticks([index + 0.5 for index in range(len(cases))], [f"{key}. {name}" for key, name, _ in cases])
    for row_index in range(len(cases)):
        for column_index in range(3):
            ax_matrix.add_patch(
                FancyBboxPatch(
                    (column_index + 0.05, row_index + 0.08),
                    0.90,
                    0.84,
                    boxstyle="round,pad=0.01,rounding_size=0.03",
                    facecolor="#fee2e2",
                    edgecolor=NOT_RUN,
                    linewidth=1.0,
                )
            )
            ax_matrix.text(column_index + 0.5, row_index + 0.5, "NOT RUN\nCUDA", ha="center", va="center", fontsize=8, color=NOT_RUN, weight="bold")
    ax_matrix.set_title("Actual 96³ PF smoke trajectory matrix")
    ax_matrix.set_xlabel("No CUDA PF binary/trajectory was available", labelpad=15, color=NOT_RUN, weight="bold")
    for spine in ax_matrix.spines.values():
        spine.set_visible(False)

    ax_note.set_axis_off()
    ax_note.text(0.5, 0.92, "Available evidence is host-side only", ha="center", fontsize=12, weight="bold")
    y = 0.78
    for key, name, evidence in cases:
        color = FIXTURE if key == "E" else HOST
        ax_note.text(
            0.03,
            y,
            f"{key}. {name}\n{evidence}",
            ha="left",
            va="top",
            fontsize=8.8,
            bbox={"boxstyle": "round,pad=0.34", "facecolor": "#f8fafc", "edgecolor": color},
        )
        y -= 0.145
    checks = fixture_validation.get("checks", {})
    ax_note.text(
        0.03,
        0.04,
        f"E package validation checks: {sum(bool(value) for value in checks.values())}/{len(checks)} true.\n"
        "This confirms package closure, not a CUDA PF trajectory.",
        ha="left",
        va="bottom",
        fontsize=7.2,
        wrap=True,
        color="#334155",
    )

    fig.suptitle("96³ PF smoke A–E — execution status, not an inferred result", fontsize=13, weight="bold", y=0.93)
    fig.text(
        0.5,
        0.015,
        "No matrix-pulse, particle-loss, free-energy, clipping, NaN/Inf, or restart-drift trajectory result is claimed without CUDA execution.",
        ha="center",
        fontsize=8.5,
        color="#334155",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    control_tag(fig, "NOT RUN CUDA — do not interpret host controls as PF smoke passes", NOT_RUN)
    return save(fig, "07_pf_smoke_matrix_not_run_cuda.png")


def main() -> int:
    thermo_summary = load_json(OUTPUT_ROOT / "thermo_cross_language_summary.json")
    thermo_rows = read_csv(OUTPUT_ROOT / "thermo_cross_language.csv")
    four_bucket_summary = load_json(OUTPUT_ROOT / "four_bucket_storage_control_summary.json")
    ledger_rows = read_csv(OUTPUT_ROOT / "four_bucket_test_ledger.csv")
    beta_summary = load_json(OUTPUT_ROOT / "beta_only_code_control_summary.json")
    beta_rows = read_csv(OUTPUT_ROOT / "beta_only_code_control.csv")
    fixture_validation = load_json(OUTPUT_ROOT / "kwn_pf_handoff_fixture_conditioned_v2" / "validation_report.json")
    fixture_metadata = load_json(OUTPUT_ROOT / "kwn_pf_handoff_fixture_conditioned_v2" / "metadata.json")

    paths: Iterable[Path] = (
        plot_authority_map(thermo_summary, four_bucket_summary, fixture_validation, beta_summary),
        plot_thermo_parity(thermo_rows, thermo_summary),
        plot_four_bucket_ledger(ledger_rows),
        plot_identity_and_roundtrip(four_bucket_summary),
        plot_handoff_constraint_feasibility(fixture_validation, fixture_metadata),
        plot_checkpoint_provenance(four_bucket_summary),
        plot_beta_direction_and_runtime(beta_rows, beta_summary),
        plot_smoke_matrix(fixture_validation),
    )
    for path in paths:
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
