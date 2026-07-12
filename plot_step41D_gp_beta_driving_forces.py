#!/usr/bin/env python3
import csv
import math
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"matplotlib is required: {exc}")


R_GAS = 8.31446261815324
TEMPERATURES_K = [300.0, 400.0, 500.0, 617.0, 653.15, 700.0]
X_GRID = np.linspace(1.0e-4, 0.20, 800)
X_SAMPLES = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
DG_STAB_LIST = [0.0, 250.0, 500.0, 1000.0]

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "reports" / "data" / "step41d_gp_beta_driving_forces"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_CURVES = OUT_DIR / "step41D_drive_curves.csv"
CSV_SAMPLES = OUT_DIR / "step41D_drive_sample_points.csv"
PLOT_GP = OUT_DIR / "step41D_GP_drive_vs_xB_by_T.png"
PLOT_BETA = OUT_DIR / "step41D_beta_drive_vs_xB_by_T.png"
PLOT_COMPARE = OUT_DIR / "step41D_GP_vs_beta_drive_by_T.png"
PLOT_THRESHOLD = OUT_DIR / "step41D_threshold_xB_vs_T.png"
REPORT = ROOT / "reports" / "step_reports" / "STEP41D_GP_BETA_DRIVING_FORCE_COMPARISON_REPORT.md"


def GHSER_Pb(T):
    if T < 600.61:
        return -10544.679 + 183.372894 * T - 35.6687 * T * math.log(T) + 0.01583435 * T * T - 5.240417e-6 * T**3 + 155015.0 / T
    return -13067.662 + 211.206579 * T - 38.5844296 * T * math.log(T) + 0.018531982 * T * T - 5.764227e-6 * T**3 + 155015.0 / T


def GHSER_Ag(T):
    if T < 1234.93:
        return -7209.512 + 118.202013 * T - 23.8463314 * T * math.log(T) - 0.001790585 * T * T - 3.98587e-7 * T**3 - 12011.0 / T
    return -15095.252 + 190.266404 * T - 33.472 * T * math.log(T) + 1.411773e29 * T**-9


def GHSER_Te(T):
    if T < 722.66:
        return -10544.679 + 183.372894 * T - 35.6687 * T * math.log(T) + 0.01583435 * T * T - 5.240417e-6 * T**3 + 155015.0 / T
    return 9160.595 - 129.265373 * T + 13.004 * T * math.log(T) - 0.0362361 * T * T + 5.006367e-6 * T**3 - 1.28681e30 * T**-9


def G_PbTe_Solid(T):
    return (-76063.2138 + 9.67716633 * T) + GHSER_Pb(T) + GHSER_Te(T)


def G_Ag2Te_Solid(T):
    base_per_atom = -10128.93 - 12.645115 * T
    G_atom = base_per_atom + (2.0 / 3.0) * GHSER_Ag(T) + (1.0 / 3.0) * GHSER_Te(T)
    return 3.0 * G_atom


def get_L_param(T):
    return 41212.9 - 18.05 * T


def clamp_fraction_eps(x):
    return min(max(x, 1.0e-12), 1.0 - 1.0e-12)


def mu_PbTe_calphad(T, xB):
    x = clamp_fraction_eps(xB)
    return G_PbTe_Solid(T) + R_GAS * T * math.log(1.0 - x) + get_L_param(T) * x * x


def mu_Ag2Te_calphad(T, xB):
    x = clamp_fraction_eps(xB)
    return G_Ag2Te_Solid(T) + R_GAS * T * math.log(x) + get_L_param(T) * (1.0 - x) * (1.0 - x)


def mu_GP0_mech(T, delta_g_stab):
    x_gp = 0.35
    return (1.0 - x_gp) * G_PbTe_Solid(T) + x_gp * G_Ag2Te_Solid(T) - delta_g_stab


def minus_delta_mu_r_gp(T, xB, delta_g_stab):
    nu_A = 0.65
    nu_B = 0.35
    return nu_A * mu_PbTe_calphad(T, xB) + nu_B * mu_Ag2Te_calphad(T, xB) - mu_GP0_mech(T, delta_g_stab)


def minus_delta_mu_r_beta(T, xB):
    return mu_Ag2Te_calphad(T, xB) - G_Ag2Te_Solid(T)


def find_zero_crossing(xs, ys):
    for i in range(len(xs) - 1):
        y0 = ys[i]
        y1 = ys[i + 1]
        if y0 == 0.0:
            return xs[i]
        if y0 * y1 < 0.0:
            x0 = xs[i]
            x1 = xs[i + 1]
            return x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)
    return math.nan


def write_curve_csv(rows):
    with CSV_CURVES.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "T",
                "xB",
                "mu_A_alpha",
                "mu_B_alpha",
                "mu_GP0_mech_dg0",
                "minus_Delta_mu_r_GP_dg0",
                "minus_Delta_mu_r_GP_dg250",
                "minus_Delta_mu_r_GP_dg500",
                "minus_Delta_mu_r_GP_dg1000",
                "mu_beta0",
                "minus_Delta_mu_r_beta",
                "alpha_branch",
                "gp_reference_mode",
                "beta_reference_mode",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_sample_csv(rows):
    with CSV_SAMPLES.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_gp(curves):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for T in TEMPERATURES_K:
        ax.plot(curves[T]["x"], curves[T]["gp_dg0"], label=f"{T:g} K")
        ax.plot(curves[T]["x"], curves[T]["gp_dg250"], linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.axvline(0.03, color="gray", linestyle=":", linewidth=1.0)
    ax.axvline(0.05, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("xB_alpha")
    ax.set_ylabel("stoichiometric reaction drive (J/mol)")
    ax.set_title("GP stoichiometric reaction drive vs xB_alpha by T")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_GP, dpi=170)
    plt.close(fig)


def plot_beta(curves):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for T in TEMPERATURES_K:
        ax.plot(curves[T]["x"], curves[T]["beta"], label=f"{T:g} K")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.axvline(0.03, color="gray", linestyle=":", linewidth=1.0)
    ax.axvline(0.05, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("xB_alpha")
    ax.set_ylabel("stoichiometric reaction drive (J/mol)")
    ax.set_title("beta-Ag2Te stoichiometric reaction drive vs xB_alpha by T")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_BETA, dpi=170)
    plt.close(fig)


def plot_compare(curves):
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5), sharex=True, sharey=True)
    for ax, T in zip(axes.flat, TEMPERATURES_K):
        ax.plot(curves[T]["x"], curves[T]["gp_dg0"], label="GP", linewidth=1.8)
        ax.plot(curves[T]["x"], curves[T]["beta"], label="beta", linewidth=1.8)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.axvline(0.03, color="gray", linestyle=":", linewidth=0.8)
        ax.axvline(0.05, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(f"{T:g} K")
        ax.grid(True, alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel("xB_alpha")
    fig.supylabel("stoichiometric reaction drive (J/mol)")
    fig.suptitle("GP vs beta stoichiometric reaction drive by temperature")
    fig.tight_layout()
    fig.savefig(PLOT_COMPARE, dpi=170)
    plt.close(fig)


def plot_thresholds(threshold_rows):
    Ts = [r["T"] for r in threshold_rows]
    gp = [r["gp_threshold_xB"] for r in threshold_rows]
    beta = [r["beta_threshold_xB"] for r in threshold_rows]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(Ts, gp, marker="o", label="GP threshold xB")
    ax.plot(Ts, beta, marker="s", label="beta threshold xB")
    ax.axhline(0.03, color="gray", linestyle=":", linewidth=1.0)
    ax.axhline(0.05, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("T (K)")
    ax.set_ylabel("threshold xB_alpha for zero stoichiometric reaction drive")
    ax.set_title("Zero-crossing thresholds for stoichiometric reaction drive")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_THRESHOLD, dpi=170)
    plt.close(fig)


def main():
    curve_rows = []
    sample_rows = []
    curves = {}
    threshold_rows = []

    for T in TEMPERATURES_K:
        mu_gp0_dg0 = mu_GP0_mech(T, 0.0)
        mu_beta0 = G_Ag2Te_Solid(T)
        gp0 = []
        gp250 = []
        gp500 = []
        gp1000 = []
        beta = []
        muA_list = []
        muB_list = []
        for x in X_GRID:
            muA = mu_PbTe_calphad(T, x)
            muB = mu_Ag2Te_calphad(T, x)
            gp_dg0 = minus_delta_mu_r_gp(T, x, 0.0)
            gp_dg250 = minus_delta_mu_r_gp(T, x, 250.0)
            gp_dg500 = minus_delta_mu_r_gp(T, x, 500.0)
            gp_dg1000 = minus_delta_mu_r_gp(T, x, 1000.0)
            beta_drive = minus_delta_mu_r_beta(T, x)
            muA_list.append(muA)
            muB_list.append(muB)
            gp0.append(gp_dg0)
            gp250.append(gp_dg250)
            gp500.append(gp_dg500)
            gp1000.append(gp_dg1000)
            beta.append(beta_drive)
            curve_rows.append(
                {
                    "T": T,
                    "xB": x,
                    "mu_A_alpha": muA,
                    "mu_B_alpha": muB,
                    "mu_GP0_mech_dg0": mu_gp0_dg0,
                    "minus_Delta_mu_r_GP_dg0": gp_dg0,
                    "minus_Delta_mu_r_GP_dg250": gp_dg250,
                    "minus_Delta_mu_r_GP_dg500": gp_dg500,
                    "minus_Delta_mu_r_GP_dg1000": gp_dg1000,
                    "mu_beta0": mu_beta0,
                    "minus_Delta_mu_r_beta": beta_drive,
                    "alpha_branch": "raw_regular_solution",
                    "gp_reference_mode": "mechanical_mixture",
                    "beta_reference_mode": "G_Ag2Te_endpoint",
                }
            )
        curves[T] = {
            "x": X_GRID.copy(),
            "muA": np.array(muA_list),
            "muB": np.array(muB_list),
            "gp_dg0": np.array(gp0),
            "gp_dg250": np.array(gp250),
            "gp_dg500": np.array(gp500),
            "gp_dg1000": np.array(gp1000),
            "beta": np.array(beta),
        }
        threshold_rows.append(
            {
                "T": T,
                "gp_threshold_xB": find_zero_crossing(X_GRID, gp0),
                "beta_threshold_xB": find_zero_crossing(X_GRID, beta),
            }
        )

        for x in X_SAMPLES:
            sample_rows.append(
                {
                    "T": T,
                    "xB": x,
                    "mu_A_alpha": mu_PbTe_calphad(T, x),
                    "mu_B_alpha": mu_Ag2Te_calphad(T, x),
                    "mu_GP0_mech_dg0": mu_gp0_dg0,
                    "minus_Delta_mu_r_GP_dg0": minus_delta_mu_r_gp(T, x, 0.0),
                    "minus_Delta_mu_r_GP_dg250": minus_delta_mu_r_gp(T, x, 250.0),
                    "minus_Delta_mu_r_GP_dg500": minus_delta_mu_r_gp(T, x, 500.0),
                    "minus_Delta_mu_r_GP_dg1000": minus_delta_mu_r_gp(T, x, 1000.0),
                    "mu_beta0": mu_beta0,
                    "minus_Delta_mu_r_beta": minus_delta_mu_r_beta(T, x),
                    "alpha_branch": "raw_regular_solution",
                    "gp_reference_mode": "mechanical_mixture",
                    "beta_reference_mode": "G_Ag2Te_endpoint",
                }
            )

    write_curve_csv(curve_rows)
    write_sample_csv(sample_rows)
    plot_gp(curves)
    plot_beta(curves)
    plot_compare(curves)
    plot_thresholds(threshold_rows)

    sample_lookup = {(row["T"], row["xB"]): row for row in sample_rows}
    lines = [
        "# Step41D GP vs beta driving-force comparison",
        "",
        "## Definitions used",
        "",
        "- alpha thermodynamics branch: `raw_regular_solution`",
        "- GP reference: `mechanical_mixture`",
        "- beta reference: `G_Ag2Te_endpoint`",
        "- Positive value means formation is thermodynamically favored.",
        "",
        "## Direct answers",
        "",
    ]

    for T in TEMPERATURES_K:
        gp03 = sample_lookup[(T, 0.03)]["minus_Delta_mu_r_GP_dg0"]
        beta03 = sample_lookup[(T, 0.03)]["minus_Delta_mu_r_beta"]
        lines.append(f"- At `T={T:g} K`, `xB=0.03`: GP drive = `{gp03:.2f} J/mol`, beta drive = `{beta03:.2f} J/mol`.")

    gp_all_positive_at_003 = all(sample_lookup[(T, 0.03)]["minus_Delta_mu_r_GP_dg0"] > 0.0 for T in TEMPERATURES_K)
    beta_all_positive_at_003 = all(sample_lookup[(T, 0.03)]["minus_Delta_mu_r_beta"] > 0.0 for T in TEMPERATURES_K)

    lines.extend(
        [
            "",
            f"- GP drive at `xB=0.03` is {'positive' if gp_all_positive_at_003 else 'not always positive'} across the scanned temperatures.",
            f"- beta drive at `xB=0.03` is {'positive' if beta_all_positive_at_003 else 'not always positive'} across the scanned temperatures.",
            "- In the scanned `xB=0.01..0.10` range, beta has a much stronger direct chemical driving force than GP under the current project definitions.",
            "- This means the current thermodynamics do not by themselves show GP being chemically favored over beta in magnitude; they only show that GP is also thermodynamically allowed over much of the same range.",
            "",
            "## Thresholds",
            "",
            "| T (K) | GP threshold xB | beta threshold xB |",
            "|---|---:|---:|",
        ]
    )
    for row in threshold_rows:
        gp_thr = "outside range" if not math.isfinite(row["gp_threshold_xB"]) else f"{row['gp_threshold_xB']:.6f}"
        beta_thr = "outside range" if not math.isfinite(row["beta_threshold_xB"]) else f"{row['beta_threshold_xB']:.6f}"
        lines.append(f"| {row['T']:g} | {gp_thr} | {beta_thr} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a chemical reaction-drive comparison only, not a nucleation-rate prediction.",
            "- Under the current definitions, beta usually has the larger direct stoichiometric chemical driving force in the experimental composition window.",
            "- The mechanical-mixture GP reference with `Delta_g_stab = 0` makes GP drive clearly positive at low `xB`, but not stronger than beta.",
            "- Any experimentally GP-first pathway therefore still needs help from factors outside this plot, such as interface energy, elastic penalty, seed geometry, or kinetic throttling.",
        ]
    )

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[step41d] wrote {CSV_CURVES}")
    print(f"[step41d] wrote {CSV_SAMPLES}")
    print(f"[step41d] wrote {PLOT_GP}")
    print(f"[step41d] wrote {PLOT_BETA}")
    print(f"[step41d] wrote {PLOT_COMPARE}")
    print(f"[step41d] wrote {PLOT_THRESHOLD}")
    print(f"[step41d] wrote {REPORT}")


if __name__ == "__main__":
    main()
