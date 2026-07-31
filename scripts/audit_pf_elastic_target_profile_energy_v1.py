#!/usr/bin/env python3
"""Freeze elastic-energy provenance for a materialized profile library."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def radius_tag(radius_nm: float) -> str:
    return f"R{radius_nm:.1f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--case-source-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Run root containing the original cases/ and profiles/ trees. "
            "May be repeated for a selected library assembled from qualified "
            "partial runs. Defaults to --run-root."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    library_path = args.run_root / "library" / "library_manifest.json"
    if not library_path.is_file():
        raise SystemExit(f"missing library manifest: {library_path}")
    library = json.loads(library_path.read_text(encoding="utf-8"))
    if library.get("schema") != "PF_ELASTIC_TARGET_PROFILE_LIBRARY_V1":
        raise SystemExit("wrong library schema")
    case_source_roots = args.case_source_root or [args.run_root]

    rows: list[dict[str, Any]] = []
    for entry in library["profiles"]:
        radius = float(entry["target_radius_nm"])
        tag = radius_tag(radius)
        profile_path = (
            library_path.parent / entry["profile_manifest_path"]
        ).resolve()
        selected_profile_sha256 = sha256(profile_path)
        matching_sources: list[tuple[Path, Path, Path]] = []
        for source_root in case_source_roots:
            case_root = source_root / "cases" / tag
            run_log = case_root / "run.stdout"
            traces = list(case_root.glob("Results/**/energy_minimize_*.csv"))
            source_profile = (
                source_root / "profiles" / tag / "profile_manifest.json"
            )
            if (
                run_log.is_file()
                and len(traces) == 1
                and source_profile.is_file()
                and sha256(source_profile) == selected_profile_sha256
            ):
                matching_sources.append((source_root, run_log, traces[0]))
        if len(matching_sources) != 1:
            raise SystemExit(
                f"R={radius:g}: expected exactly one qualified case source, "
                f"found {len(matching_sources)}"
            )
        source_root, run_log, trace_path = matching_sources[0]
        traces = [trace_path]
        run_text = run_log.read_text(encoding="utf-8", errors="replace")
        if "  elastic                  : 启用" not in run_text:
            raise SystemExit(f"R={radius:g}: elastic runtime marker is absent")
        with traces[0].open("r", encoding="utf-8", newline="") as handle:
            trace_rows = list(csv.DictReader(handle))
        if not trace_rows:
            raise SystemExit(f"R={radius:g}: empty energy trace")
        final = trace_rows[-1]
        required = (
            "iter",
            "F_surf_hat",
            "F_el_hat",
            "F_chem_excess_hat",
            "F_total_excess_hat",
            "total_interface_sum",
            "total_el_core_sum",
            "rms_res",
            "mass_err_rel",
        )
        if any(name not in final for name in required):
            raise SystemExit(f"R={radius:g}: incomplete energy trace schema")
        values = {name: float(final[name]) for name in required}
        if not all(math.isfinite(value) for value in values.values()):
            raise SystemExit(f"R={radius:g}: non-finite final energy record")
        if values["F_el_hat"] <= 0.0 or values["total_el_core_sum"] <= 0.0:
            raise SystemExit(f"R={radius:g}: elastic energy is not positive")

        source_profile_path = (
            source_root / "profiles" / tag / "profile_manifest.json"
        )
        if sha256(source_profile_path) != selected_profile_sha256:
            raise SystemExit(
                f"R={radius:g}: selected profile does not match case source"
            )
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        converged_iter = int(
            profile["constraint_contract"]["final_marker"][
                "convergence_converged_iter"
            ]
        )
        if int(values["iter"]) != converged_iter:
            raise SystemExit(
                f"R={radius:g}: energy trace and profile endpoint differ"
            )
        params = profile["parameter_values"]
        eigenstrain = {
            key: float(params[key])
            for key in (
                "eps_xx00",
                "eps_yy00",
                "eps_zz00",
                "eps_xy00",
                "eps_xz00",
                "eps_yz00",
            )
        }
        rows.append(
            {
                "target_radius_nm": radius,
                "converged_iter": converged_iter,
                "F_surf_hat": values["F_surf_hat"],
                "F_el_hat": values["F_el_hat"],
                "F_chem_excess_hat": values["F_chem_excess_hat"],
                "F_total_excess_hat": values["F_total_excess_hat"],
                "total_interface_sum": values["total_interface_sum"],
                "total_el_core_sum": values["total_el_core_sum"],
                "projected_kkt_rms": values["rms_res"],
                "mass_error_relative": values["mass_err_rel"],
                "energy_trace_sha256": sha256(traces[0]),
                "run_log_sha256": sha256(run_log),
                "profile_manifest_sha256": sha256(profile_path),
                "case_source_root": str(source_root),
                "case_source_profile_sha256": sha256(source_profile_path),
                "eigenstrain": eigenstrain,
                "orientation_label": profile["orientation_label"],
                "elastic_runtime_marker": "PASS",
            }
        )

    csv_path = args.out / "elastic_energy_endpoints.csv"
    csv_fields = [
        key for key in rows[0] if key not in ("eigenstrain",)
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in row.items() if key in csv_fields}
            for row in rows
        )
    audit = {
        "schema": "PF_ELASTIC_TARGET_PROFILE_ENERGY_AUDIT_V1",
        "status": "PASS_ELASTIC_TARGET_PROFILE_ENERGY_PROVENANCE_V1",
        "run_root": str(args.run_root),
        "case_source_roots": [str(path) for path in case_source_roots],
        "library_manifest_sha256": sha256(library_path),
        "source_commit": library["source_commit"],
        "source_tree_sha256": library["source_tree_sha256"],
        "boundary_contract": "periodic_fixed_cell",
        "orientation_scope": library["orientation_scope"],
        "profile_count": len(rows),
        "profiles": rows,
    }
    audit_path = args.out / "elastic_energy_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out / "final_terminal_output.txt").write_text(
        f"elastic_energy_status={audit['status']}\n"
        f"profile_count={len(rows)}\n"
        f"library_manifest_sha256={audit['library_manifest_sha256']}\n"
        f"audit_sha256={sha256(audit_path)}\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
