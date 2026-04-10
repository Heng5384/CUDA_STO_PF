#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def make_run_dir_name(t_c: float, nx: int, ny: int, nz: int, dt: float, nsteps: int, r_nm: float, xB_out: float) -> str:
    return f"chel_T{t_c:.0f}_cuda_{nx}x{ny}x{nz}_dt{dt:g}_steps{nsteps}_r{r_nm:.3f}nm_xB{xB_out:.3f}"


def group_name(mode: str, strain: float) -> str:
    if abs(strain) < 1e-15:
        return f"{mode}_s0"
    sign = "p" if strain > 0 else "m"
    mag = f"{abs(strain):.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{mode}_{sign}{mag}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create strain-grouped symlink folders under Results/by_external_strain.")
    parser.add_argument("--case-dir", default="Results_scan/constraint_cnt_strictref_test_T400_x0_0p03")
    parser.add_argument("--results-root", default="Results")
    parser.add_argument("--nx", type=int, default=400)
    parser.add_argument("--ny", type=int, default=400)
    parser.add_argument("--nz", type=int, default=400)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--nsteps", type=int, default=30000)
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    results_root = Path(args.results_root)
    by_strain_root = results_root / "by_external_strain"
    by_strain_root.mkdir(parents=True, exist_ok=True)

    created = 0
    for csv_path in sorted(case_dir.glob("case_*_L5.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            row = next(csv.DictReader(f))

        base_case_tag = row["base_case_tag"]
        mode = row["mode"]
        strain = float(row["strain"])
        t_c = float(row["T_C"])
        xB_out = float(row["xB_out"])
        rc = float(row["rc_schur_nm"])
        window = float(row["window_nm"])
        step = float(row["step_nm"])
        n = int(round(window / step))

        group_dir = by_strain_root / group_name(mode, strain)
        group_dir.mkdir(parents=True, exist_ok=True)

        for k in range(-n, n + 1):
            r_nm = rc + k * step
            if r_nm <= 0:
                continue
            r_str = f"{r_nm:.6f}"
            case_tag = f"{base_case_tag}_r{r_str.replace('.', 'p')}"
            run_dir = make_run_dir_name(t_c, args.nx, args.ny, args.nz, args.dt, args.nsteps, r_nm, xB_out)
            target = results_root / run_dir / case_tag
            link = group_dir / case_tag
            if link.exists() or link.is_symlink():
                continue
            link.symlink_to(Path("../../") / run_dir / case_tag)
            created += 1

    print(f"case_dir={case_dir}")
    print(f"results_root={results_root}")
    print(f"created_symlinks={created}")


if __name__ == "__main__":
    main()
