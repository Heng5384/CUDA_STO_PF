#!/usr/bin/env python3
"""Plot the honest T380 equilibrium-audit compatibility summary.

Dynamic finite-particle observations are plotted separately from the analytic
chemical-flat root; they are not silently converted into equilibrium roots.
"""
from pathlib import Path
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "reports" / "equilibrium_audit_v1"

def main() -> None:
    radii = [6, 8, 10, 12, 14, 16, 20]
    # These are observed short-run far-field bracket values, not roots.
    low = [0.0058, 0.0058, 0.0058, 0.0058, 0.0058, 0.0058, 0.0058]
    high = [0.0066, 0.0066, 0.0066, 0.0066, 0.0066, 0.0066, 0.0066]
    chem = 0.004654096254055399

    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=180)
    ax.axhspan(0.0058, 0.0066, color="#f0c36b", alpha=0.28,
               label="experimental interval")
    ax.axhline(chem, color="#1f4e79", lw=2,
                label="chemical flat analytic root")
    ax.fill_between(radii, low, high, color="#7aa6d8", alpha=0.16,
                    label="short dynamic bracket window (not root)")
    ax.plot(radii, low, "o--", color="#4575b4", ms=4, label="low endpoint xAg")
    ax.plot(radii, high, "s--", color="#d73027", ms=4, label="high endpoint xAg")
    ax.set_xlabel("equivalent particle radius R (nm)")
    ax.set_ylabel("Ag atomic fraction $x_{Ag}$")
    ax.set_title("T380 equilibrium audit: chemical root vs finite-particle observations")
    ax.set_xlim(5.5, 20.5)
    ax.set_ylim(0.0040, 0.0070)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "equilibrium_compatibility_map_T380.png")
    plt.close(fig)

if __name__ == "__main__":
    main()
