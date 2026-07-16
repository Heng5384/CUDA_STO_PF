#!/usr/bin/env python3
"""Prepare restart-safe JC4 fixed-dt probes from an accepted startup state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DT = (0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0)


def tag(value: float) -> str:
    return f"{value:.9g}".replace(".", "p")


def replace_param(text: str, key: str, value: object) -> str:
    replacement = f"{key}={value}"
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(replacement, text)
    return text.rstrip() + "\n" + replacement + "\n"


def final_checkpoint(source: Path) -> tuple[int, dict[str, Path]]:
    candidates: dict[int, dict[str, Path]] = {}
    for field, suffix in (
        ("phi", "_phi.raw"),
        ("xB", "_xB_alpha.raw"),
        ("Ctot", "_Ctot.raw"),
        ("meta", "_meta.json"),
    ):
        for path in (source / "run").rglob(f"ctot_checkpoint_step*{suffix}"):
            match = re.search(r"step(\d+)_", path.name)
            if match:
                candidates.setdefault(int(match.group(1)), {})[field] = path
    complete = [(step, fields) for step, fields in candidates.items()
                if set(fields) == {"phi", "xB", "Ctot", "meta"}]
    if not complete:
        raise RuntimeError(f"no complete Ctot checkpoint under {source}")
    return max(complete, key=lambda row: row[0])


def build(source_root: Path, output_root: Path, dt_values: tuple[float, ...],
          probe_steps: int,
          performance_profile_enabled: bool = False) -> dict[str, object]:
    if probe_steps <= 0 or any(dt <= 0.0 for dt in dt_values):
        raise ValueError("dt values and probe steps must be positive")
    output_root.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    for direction in ("growth", "dissolution"):
        matching = sorted((source_root / "cases").glob(f"T400_{direction}_*"))
        completed = [case for case in matching if (case / "workstation_status.json").is_file()]
        if len(completed) != 1:
            raise RuntimeError(
                f"expected one completed {direction} startup case, got {len(completed)}"
            )
        source = completed[0]
        source_manifest = json.loads(
            (source / "runtime_manifest.json").read_text(encoding="utf-8")
        )
        status = json.loads(
            (source / "workstation_status.json").read_text(encoding="utf-8")
        )
        if not (
            int(status["returncode"]) == 0
            and int(status["accepted_steps"]) == int(status["expected_steps"])
            and int(status["retry_count"]) == 0
            and int(status["reject_count"]) == 0
        ):
            raise RuntimeError(f"startup case is not accepted: {source}")
        source_step, checkpoint = final_checkpoint(source)
        source_meta = json.loads(checkpoint["meta"].read_text(encoding="utf-8"))
        source_time_code = float(source_meta["time_code"])
        params_text = (source / "runtime.params").read_text(encoding="utf-8")
        for dt in dt_values:
            case_id = f"T400_{direction}_continued_dt{tag(dt)}"
            case = output_root / "cases" / case_id
            case.mkdir(parents=True, exist_ok=True)
            for field, target in (
                ("phi", "phi_init.raw"),
                ("xB", "xB_init.raw"),
                ("Ctot", "Ctot_init.raw"),
                ("meta", "init_meta.json"),
            ):
                shutil.copy2(checkpoint[field], case / target)
            case_params = replace_param(params_text, "dt", f"{dt:.17e}")
            case_params = replace_param(case_params, "init_case_tag", case_id)
            case_params = replace_param(
                case_params, "ctot_performance_profile_enabled",
                int(performance_profile_enabled),
            )
            (case / "runtime.params").write_text(case_params, encoding="utf-8")
            manifest = {
                "schema": "jc4_dt_continuation_probe_v1",
                "case_id": case_id,
                "direction": direction,
                "grid": source_manifest["grid"],
                "dx_nm": source_manifest["dx_nm"],
                "lambda_nm": source_manifest["lambda_nm"],
                "temperature_C": source_manifest["temperature_C"],
                "dt_code": dt,
                "nsteps": probe_steps,
                "out_every": probe_steps,
                "source_case_id": source_manifest["case_id"],
                "source_checkpoint_step": source_step,
                "source_checkpoint_time_code": source_time_code,
                "authoritative_restart": "Ctot",
                "purpose": "post_startup_fixed_dt_acceptance_probe_not_adaptive_BE",
                "cluster_used": False,
                "performance_profile_enabled": performance_profile_enabled,
            }
            (case / "runtime_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            cases.append(manifest)
    top = {
        "schema": "jc4_dt_continuation_probe_matrix_v1",
        "cases": cases,
        "source_root": str(source_root),
        "adaptive_BE": False,
        "cluster_used": False,
        "performance_profile_enabled": performance_profile_enabled,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(top, indent=2) + "\n", encoding="utf-8"
    )
    return top


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root", type=Path,
        default=ROOT / "tmp/jc4_long_time_planar_stability_refined",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "tmp/jc4_dt_continuation_probe",
    )
    parser.add_argument("--dt", type=float, action="append")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--enable-performance-profile", action="store_true")
    args = parser.parse_args()
    dt_values = tuple(args.dt) if args.dt else DEFAULT_DT
    manifest = build(
        args.source_root.resolve(), args.output_root.resolve(),
        dt_values, args.steps, args.enable_performance_profile,
    )
    print(f"jc4_dt_continuation_probe_cases={len(manifest['cases'])}")
    for case in manifest["cases"]:
        print(
            f"{case['case_id']} source_step={case['source_checkpoint_step']} "
            f"dt={case['dt_code']:.9g} steps={case['nsteps']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
