#!/usr/bin/env python3
"""Compare the minimal oracle with real short PF multiparticle trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

from particle_coarsening_oracle import OracleParams, run_oracle


MASS_RE = re.compile(r"CTOT_MIMETIC_BE_ACCEPT.*mass_error=([^ ]+)")
DT_RE = re.compile(r"^dt=([^#\s]+)", re.MULTILINE)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    args.report_root.mkdir(parents=True, exist_ok=True)

    # Derived from the unchanged T400 runtime inputs, not fitted to these runs.
    temperature_k = 673.15
    gamma = 0.168
    molar_volume = 4.1009e-5
    gas_constant = 8.314462618
    capillary_nm = 2.0 * gamma * molar_volume / (gas_constant * temperature_k) * 1e9
    diffusivity_nm2_s = 400.0 / 41.1295854245547687
    params = OracleParams(0.005896193199132673, capillary_nm,
                          diffusivity_nm2_s, 0.05)
    rows = []
    all_pass = True
    for tag, steps in (("two_particle", 200), ("three_particle_extinction", 1200)):
        case = args.run_root / tag
        initial = read(case / "handoff_initial.json")
        final = read(case / "handoff_final.json")
        dt_values = DT_RE.findall((case / "runtime.params").read_text())
        if not dt_values:
            raise RuntimeError(f"missing dt in {case / 'runtime.params'}")
        dt_code = float(dt_values[-1])
        end_time = steps * dt_code * 41.1295854245547687
        trajectory, summary = run_oracle(initial, params, end_time)
        (args.report_root / f"{tag}_oracle_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n")
        initial_by_id = {p["particle_id"]: p for p in initial["particles"]}
        pf_final = sorted(p["equivalent_radius_nm"] for p in final["particles"])
        oracle_final = sorted(summary["final_radii_nm"])
        radius_errors = [abs(a-b)/max(abs(b), 1e-300)
                         for a, b in zip(pf_final, oracle_final)]
        count_match = len(pf_final) == len(oracle_final)
        pf_beta = final["ledger"]["beta_inventory_hvB"]
        oracle_beta = summary["final_beta_inventory"]
        phase_error = abs(pf_beta-oracle_beta)/max(abs(pf_beta), 1e-300)
        missing_pf = len(initial_by_id) - len(pf_final)
        extinction_match = (tag == "two_particle" or
                            (missing_pf > 0 and bool(summary["extinction_order"])))
        log = (case / "run.log").read_text(errors="replace")
        mass_errors = [abs(float(x)) for x in MASS_RE.findall(log)]
        max_pf_mass = max(mass_errors, default=math.inf)
        passed = (count_match and max(radius_errors, default=0.0) <= 0.05 and
                  phase_error <= 0.05 and extinction_match and
                  summary["max_mass_error_rel"] <= 1e-12 and max_pf_mass <= 1e-10)
        all_pass &= passed
        rows.append({
            "case": tag, "PF_steps": steps, "physical_time_s": end_time,
            "PF_dt_code": dt_code,
            "PF_initial_count": initial["particle_count"],
            "PF_final_count": final["particle_count"],
            "oracle_final_count": summary["final_particle_count"],
            "PF_final_radii_nm": ";".join(map(str, pf_final)),
            "oracle_final_radii_nm": ";".join(map(str, oracle_final)),
            "max_radius_error_rel": max(radius_errors, default=0.0),
            "phase_amount_error_rel": phase_error,
            "PF_mass_error_rel_max": max_pf_mass,
            "oracle_mass_error_rel_max": summary["max_mass_error_rel"],
            "extinction_order_match": extinction_match,
            "status": "PASS" if passed else "FAIL",
        })
    with (args.report_root / "particle_oracle_overlap_metrics.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    status = "HANDOFF_TREND_MODEL_READY" if all_pass else "PF_ORACLE_OVERLAP_FAILED"
    report = "# Minimal PF/particle overlap\n\n" + "\n".join(
        f"- {r['case']}: {r['status']}; radius error={r['max_radius_error_rel']:.6e}, "
        f"phase error={r['phase_amount_error_rel']:.6e}, counts="
        f"{r['PF_final_count']}/{r['oracle_final_count']}." for r in rows)
    report += f"\n\nParameters were derived from unchanged runtime inputs and were not fitted " \
              f"to these trajectories. Status: `{status}`.\n"
    (args.report_root / "particle_oracle_overlap.md").write_text(report)
    print(status)
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
