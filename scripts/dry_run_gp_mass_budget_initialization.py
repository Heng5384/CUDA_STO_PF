#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "gp_assisted_beta_mass_ledger"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_cases(out_dir: Path) -> list[dict[str, object]]:
    domain_size = 32 * 32 * 32
    xB_total_target = 0.030
    xB_matrix_fixed = 0.030
    n_gp_sites = 4
    m_B_GP_per_site = 8.0
    m_B_GP_total = n_gp_sites * m_B_GP_per_site
    m_target = xB_total_target * domain_size

    rows: list[dict[str, object]] = []

    if m_B_GP_total > m_target:
        rows.append({
            "mode": "total_composition_fixed",
            "domain_size": domain_size,
            "xB_total_target": xB_total_target,
            "xB_matrix_initial": "",
            "n_gp_sites": n_gp_sites,
            "m_B_GP_total": m_B_GP_total,
            "M_B_matrix": "",
            "M_B_GP_active": m_B_GP_total,
            "M_B_beta": 0.0,
            "M_B_total": "",
            "relative_error": "",
            "status": "abort: GP reservoir exceeds total target",
        })
    else:
        matrix = m_target - m_B_GP_total
        xB_matrix_initial = matrix / domain_size
        total = matrix + m_B_GP_total
        rows.append({
            "mode": "total_composition_fixed",
            "domain_size": domain_size,
            "xB_total_target": xB_total_target,
            "xB_matrix_initial": xB_matrix_initial,
            "n_gp_sites": n_gp_sites,
            "m_B_GP_total": m_B_GP_total,
            "M_B_matrix": matrix,
            "M_B_GP_active": m_B_GP_total,
            "M_B_beta": 0.0,
            "M_B_total": total,
            "relative_error": (total - m_target) / m_target,
            "status": "pass",
        })

    matrix = xB_matrix_fixed * domain_size
    total = matrix + m_B_GP_total
    rows.append({
        "mode": "matrix_composition_fixed",
        "domain_size": domain_size,
        "xB_total_target": "",
        "xB_matrix_initial": xB_matrix_fixed,
        "n_gp_sites": n_gp_sites,
        "m_B_GP_total": m_B_GP_total,
        "M_B_matrix": matrix,
        "M_B_GP_active": m_B_GP_total,
        "M_B_beta": 0.0,
        "M_B_total": total,
        "relative_error": 0.0,
        "status": f"pass: effective_global_xB={total/domain_size:.12e}",
    })

    write_csv(out_dir / "gp_mass_budget_initialization_test.csv", rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run explicit GP reservoir initialization mass ledger.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    rows = run_cases(args.out_dir.resolve())
    passed = all(str(r["status"]).startswith("pass") for r in rows)
    print(f"dry_run_initial_mass_budget_passed={passed}")
    print(f"output_csv={args.out_dir.resolve() / 'gp_mass_budget_initialization_test.csv'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
