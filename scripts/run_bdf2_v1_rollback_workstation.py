#!/usr/bin/env python3
"""Workstation-only rejected-BDF2 history rollback oracle."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from run_bdf2_v1_startup_restart_workstation import run_case, sha256


PROVENANCE_KEYS = (
    "bdf2_history_valid",
    "bdf2_dt_n",
    "bdf2_dt_nm1",
    "bdf2_restart_fallback_pending",
    "bdf2_accepted_step",
    "bdf2_time_code",
    "bdf2_physical_time_s",
    "time_integrator",
    "history_contract_version",
    "phase_context_version",
    "BDF2_energy_contract_version",
    "bdf2_last_fallback_reason",
)


def meta_projection(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: data.get(key) for key in PROVENANCE_KEYS}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = repo / "runs/bdf2_v1/rollback_validation"
    if root.exists():
        if not args.overwrite:
            raise RuntimeError(f"refusing to overwrite {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True)

    dt = 0.003125
    half = 0.5 * dt
    seed = run_case(repo, root, "seed_startup", dt, 1)
    forced = run_case(
        repo,
        root,
        "forced_bdf2_reject_retry",
        dt,
        3,
        seed,
        {
            "ctot_step_max_retries": "1",
            "ctot_retry_shrink_factor": "0.5",
            "ctot_dt_min_ratio": "0.5",
            "ctot_debug_force_first_bdf2_attempt_reject": "1",
        },
    )
    control_pre = run_case(repo, root, "clean_control_pre", dt, 1, seed)
    control = run_case(repo, root, "clean_control_half_dt", half, 2, control_pre)

    comparisons = []
    for field in ("phi", "xB", "Ctot", "phi_nm1", "Ctot_nm1"):
        forced_hash = sha256(forced[field])
        control_hash = sha256(control[field])
        comparisons.append(
            {
                "field": field,
                "forced_hash": forced_hash,
                "control_hash": control_hash,
                "bitwise_equal": forced_hash == control_hash,
            }
        )
    forced_log = Path(forced["log"]).read_text(encoding="utf-8")
    required_markers = {
        "forced_reject": "debug_forced_first_bdf2_attempt_reject" in forced_log,
        "retry_fallback": "fallback_reason=previous_attempt_rejected" in forced_log,
        "bdf2_resumed": forced_log.count("integrator=BDF2") >= 2,
        "no_history_commit_on_rejected_attempt": (
            forced_log.count("accepted_integrator=BDF2") == 1
        ),
    }
    forced_meta = meta_projection(Path(forced["meta"]))
    control_meta = meta_projection(Path(control["meta"]))
    summary = {
        "dt_initial": dt,
        "dt_retry": half,
        "field_comparisons": comparisons,
        "all_fields_bitwise_equal": all(row["bitwise_equal"] for row in comparisons),
        "required_markers": required_markers,
        "forced_meta": forced_meta,
        "control_meta": control_meta,
        "provenance_equal": forced_meta == control_meta,
    }
    summary["status"] = (
        "PASS_BDF2_ROLLBACK_BITWISE"
        if summary["all_fields_bitwise_equal"]
        and all(required_markers.values())
        and summary["provenance_equal"]
        else "FAIL_BDF2_ROLLBACK"
    )
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
