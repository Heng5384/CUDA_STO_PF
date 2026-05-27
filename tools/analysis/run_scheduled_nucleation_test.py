#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis.test_scheduled_nucleation_insertions_only import run_insertions_only


def _write_segment_plan(config_path: Path, out_dir: Path) -> None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    global_cfg = cfg.get("global", {})
    total_steps = int(global_cfg.get("total_steps", 1000))
    events = sorted({int(e["step"]) for e in cfg.get("nucleation_events", [])})
    cuts = [0, *[s for s in events if 0 < s < total_steps], total_steps]
    rows = []
    for idx, (start, stop) in enumerate(zip(cuts[:-1], cuts[1:])):
        rows.append(
            {
                "segment": idx,
                "global_step_start": start,
                "global_step_stop": stop,
                "segment_steps": stop - start,
                "event_after_segment": stop if stop in events else "",
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scheduled_pf_segment_plan.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (out_dir / "scheduled_pf_segment_plan.csv").open("w", encoding="utf-8") as f:
        f.write("segment,global_step_start,global_step_stop,segment_steps,event_after_segment\n")
        for r in rows:
            f.write(
                f"{r['segment']},{r['global_step_start']},{r['global_step_stop']},"
                f"{r['segment_steps']},{r['event_after_segment']}\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scheduled nucleation function-test wrapper. By default it runs the pure-Python "
            "insertion dry-run so production CUDA code is untouched. PF segmented execution is "
            "kept behind --run-pf and currently emits a segment plan plus a clear stop."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--insertions-only", action="store_true", help="Run only Python insertion/mass-compensation dry-run.")
    parser.add_argument("--run-pf", action="store_true", help="Prepare PF segmented plan. Full PF orchestration is intentionally not default.")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if args.run_pf and args.insertions_only:
        raise SystemExit("--run-pf and --insertions-only are mutually exclusive")

    if args.run_pf:
        _write_segment_plan(config_path, out_dir)
        cmd = [
            args.python,
            str(REPO_ROOT / "tools/analysis/test_scheduled_nucleation_insertions_only.py"),
            "--config",
            str(config_path),
            "--out-dir",
            str(out_dir / "insertions_only_reference"),
        ]
        (out_dir / "insertions_only_reference_command.txt").write_text(" ".join(shlex.quote(x) for x in cmd) + "\n", encoding="utf-8")
        print(f"[ok] wrote PF segment plan: {out_dir / 'scheduled_pf_segment_plan.json'}")
        print("[stop] Full PF segmented execution is not enabled by default in this safety wrapper.")
        print("[next] First validate the insertion-only reference, then wire segment runs to main_cuda raw_fields restart.")
        print("[cmd] " + " ".join(shlex.quote(x) for x in cmd))
        return 2

    # Default safe behavior.
    run_insertions_only(config_path, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
