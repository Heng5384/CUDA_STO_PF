#!/usr/bin/env python3
"""Workstation-only startup, restart, and deterministic-history validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

from prepare_bdf2_v1_startup_smoke import rewrite_meta, rewrite_params


def only(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} below {root}, got {matches}")
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_case(
    repo: Path,
    root: Path,
    tag: str,
    dt: float,
    steps: int,
    current: dict[str, Path] | None = None,
    param_overrides: dict[str, str] | None = None,
) -> dict[str, Path | int]:
    case = root / tag
    case.mkdir(parents=True)
    params = case / "runtime.params"
    meta = case / "init_meta.json"
    base = repo / "runs/preparation_v2/cases/T400_step655_lie_be_v2_dt_div1_full"
    rewrite_params(
        base / "runtime.params", params, tag, dt,
        extra_replacements=param_overrides,
    )
    if current is None:
        rewrite_meta(base / "init_meta.json", meta)
        frozen = repo / "runs/frozen_input"
        phi = frozen / "ctot_checkpoint_step000054_phi.raw"
        xb = frozen / "ctot_checkpoint_step000054_xB_alpha.raw"
        ctot = frozen / "ctot_checkpoint_step000054_Ctot.raw"
        history_args: list[str] = []
    else:
        shutil.copy2(current["meta"], meta)
        phi = current["phi"]
        xb = current["xB"]
        ctot = current["Ctot"]
        history_args = [
            "--init-Ctot-nm1-raw", str(current["Ctot_nm1"]),
            "--init-phi-nm1-raw", str(current["phi_nm1"]),
        ]
    command = [
        str(repo / "main_cuda"), "512", "1", "1", f"{dt:.17e}",
        str(steps), str(steps), "1", "0", "--mode", "dynamics",
        "--pf-param-file", str(params), "--init-mode", "raw_fields",
        "--init-phi-raw", str(phi), "--init-xB-raw", str(xb),
        "--init-Ctot-raw", str(ctot), "--init-meta", str(meta),
        "--init-case-tag", tag,
        *history_args,
    ]
    (case / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    run = case / "run"
    run.mkdir()
    started = time.perf_counter()
    with (run / "run.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=run, stdout=log, stderr=subprocess.STDOUT)
    wall_seconds = time.perf_counter() - started
    (run / "exit_code.txt").write_text(f"{completed.returncode}\n", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"{tag} failed; inspect {run / 'run.log'}")
    result = {
        "phi": only(run, f"ctot_checkpoint_step{steps:06d}_phi.raw"),
        "xB": only(run, f"ctot_checkpoint_step{steps:06d}_xB_alpha.raw"),
        "Ctot": only(run, f"ctot_checkpoint_step{steps:06d}_Ctot.raw"),
        "phi_nm1": only(run, f"ctot_checkpoint_step{steps:06d}_phi_nm1.raw"),
        "Ctot_nm1": only(run, f"ctot_checkpoint_step{steps:06d}_Ctot_nm1.raw"),
        "meta": only(run, f"ctot_checkpoint_step{steps:06d}_meta.json"),
        "log": run / "run.log",
        "returncode": completed.returncode,
        "wall_seconds": wall_seconds,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = repo / "runs/bdf2_v1/startup_restart_validation"
    if root.exists():
        if not args.overwrite:
            raise RuntimeError(f"refusing to overwrite {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    dt = 0.003125
    startup = run_case(repo, root, "startup_one", dt, 1)
    restart_after_startup = run_case(
        repo, root, "restart_after_startup", dt, 2, startup
    )
    active_source_root = repo / "runs/bdf2_v1/startup_smoke/run"
    active = {
        key: only(active_source_root, f"ctot_checkpoint_step000003_{suffix}.raw")
        for key, suffix in {
            "phi": "phi", "xB": "xB_alpha", "Ctot": "Ctot",
            "phi_nm1": "phi_nm1", "Ctot_nm1": "Ctot_nm1",
        }.items()
    }
    active["meta"] = only(active_source_root, "ctot_checkpoint_step000003_meta.json")
    active_a = run_case(repo, root, "active_restart_a", dt, 2, active)
    active_b = run_case(repo, root, "active_restart_b", dt, 2, active)
    comparisons = []
    for field in ("phi", "Ctot", "phi_nm1", "Ctot_nm1"):
        hash_a = sha256(active_a[field])
        hash_b = sha256(active_b[field])
        comparisons.append(
            {"field": field, "sha_a": hash_a, "sha_b": hash_b,
             "bitwise_equal": hash_a == hash_b}
        )
    summary = {
        "startup": str(startup["log"]),
        "restart_after_startup": str(restart_after_startup["log"]),
        "active_restart_a": str(active_a["log"]),
        "active_restart_b": str(active_b["log"]),
        "bitwise_comparisons": comparisons,
        "all_bitwise_equal": all(row["bitwise_equal"] for row in comparisons),
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_bitwise_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
