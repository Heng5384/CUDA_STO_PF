#!/usr/bin/env python3
"""Purely mathematical sub-8-nm extension of the global transport audit."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("global_density", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import global density audit")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    names = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve() if args.out.is_absolute() else (root / args.out).resolve()
    output = out / "global_2d_non_pf_mathematical_diagnostic.csv"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite: {output}")
    global_script = root / "scripts/analyze_global_resolved_psd_density_v1.py"
    density = load_module(global_script)
    transport = density.load_transport(root / "scripts/pf_full_psd_no_dislocation_transport_v1.py")
    config = transport.load_json(root / "data/qualification/yu2024_transport_v1/yu_48h_parameters.json")
    # No step is prescribed for the extra mathematical range.  This coarse,
    # fully integrated grid deliberately trades volume for a transparent
    # diagnostic; it is never a current-library/PF seed recommendation.
    r6_values = [round(.5 + .25 * index, 10) for index in range(31)]
    r48_values = sorted({
        *r6_values,
        *[round(8.0 + .5 * index, 10) for index in range(45)],
        *[float(31 + index) for index in range(30)],
        *[float(65 + 5 * index) for index in range(28)],
    })
    cache: dict[tuple[float, float], tuple[dict[str, Any], float]] = {}
    for cv in density.CVS:
        for radius in r48_values:
            pop = density.lognormal_population(radius, cv)
            kappa, _ = density.kappa_and_spectrum(transport, config, pop)
            cache[(radius, cv)] = (pop, kappa)
    rows: list[dict[str, Any]] = []
    for cv6 in density.CVS:
        for cv48 in density.CVS:
            for r6 in r6_values:
                for r48 in r48_values:
                    if r48 < r6:
                        continue
                    pop6, k6 = cache[(r6, cv6)]
                    pop48, k48 = cache[(r48, cv48)]
                    rows.append({
                        "classification": "NON_PF_RESOLVED_MATHEMATICAL_DIAGNOSTIC",
                        "CV6": cv6, "CV48": cv48, "R6_nm": r6, "R48_nm": r48,
                        "R48_over_R6": r48 / r6,
                        "N6_equivalent": pop6["equivalent_N"], "N48_equivalent": pop48["equivalent_N"],
                        "kappa6_W_mK": k6, "kappa48_W_mK": k48,
                        "delta_kappa_W_mK": k48-k6, "relative_change": (k48-k6)/k6,
                        "J_abs": ((k6-density.EXPERIMENT_K6)/density.EXPERIMENT_K6)**2 + ((k48-density.EXPERIMENT_K48)/density.EXPERIMENT_K48)**2,
                        "J_delta": ((k48-k6)-density.EXPERIMENT_DELTA)**2,
                        "J_relative": ((k48-k6)/k6-density.EXPERIMENT_RELATIVE)**2,
                        "both_endpoints_within_5percent": abs(k6-density.EXPERIMENT_K6)/density.EXPERIMENT_K6 <= .05 and abs(k48-density.EXPERIMENT_K48)/density.EXPERIMENT_K48 <= .05,
                        "PF_coarsening_compatible": False,
                        "beta_inventory_h_volume_nm3": density.TARGET_H_VOLUME_NM3,
                    })
    write_csv(output, rows)
    best = min(rows, key=lambda row: float(row["J_abs"]))
    max_delta = max(rows, key=lambda row: float(row["delta_kappa_W_mK"]))
    summary = {
        "schema": "NON_PF_MATHEMATICAL_GLOBAL_PAIR_DIAGNOSTIC_V1",
        "grid": {"R6_nm": "0.5--8 step 0.25", "R48_nm": "0.5--30 step 0.5; 31--60 step 1; 65--200 step 5", "CV6_CV48": list(density.CVS)},
        "all_rows": len(rows), "best_absolute": best, "maximum_positive_delta": max_delta,
        "any_endpoint_pair_within_5percent": any(bool(row["both_endpoints_within_5percent"]) for row in rows),
        "scientific_role": "Mathematical diagnostic only; no R<8 nm PF fixture is authorized or recommended.",
    }
    (out / "global_2d_non_pf_mathematical_diagnostic_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary["all_rows"])


if __name__ == "__main__":
    main()
