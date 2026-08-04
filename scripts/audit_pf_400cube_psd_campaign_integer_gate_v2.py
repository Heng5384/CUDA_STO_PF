#!/usr/bin/env python3
"""Static qualification and campaign aggregation for 400-cube V2 fixtures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


FIXTURE_PASS = "PASS_400CUBE_PSD_CAMPAIGN_INTEGER_GATE_FIXTURE_V2"
AUDIT_PASS = "PASS_400CUBE_PSD_CAMPAIGN_INTEGER_GATE_STATIC_AUDIT_V2"
CAMPAIGN_PASS = "PASS_21_FIXTURES_STATIC_QUALIFICATION_V2"
TARGET_H = 1531490.9025410344
H_TOL = 1.0e-7
TARGET_XAG = 0.0062
XAG_TOL = 1.0e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def periodic_distance(a: list[float], b: list[float], n: float) -> float:
    delta = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    delta = np.minimum(delta, n - delta)
    return float(np.linalg.norm(delta))


def audit_fixture(fixture: Path, out: Path) -> dict[str, Any]:
    manifest_path = fixture / "fixture_manifest.json"
    if not manifest_path.is_file() or (fixture / "status.txt").read_text(encoding="utf-8").strip() != FIXTURE_PASS:
        raise ValueError("fixture lacks exact materialization PASS")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = manifest["case"]
    failures: list[str] = []
    fields = manifest["fields"]
    required_fields = {"phi", "h_phi", "xB_alpha", "Y", "dY_dt_prev", "C_B_tot", "delta_C_relaxation_total"}
    if set(fields) != required_fields:
        failures.append("field_schema")
    for name, entry in fields.items():
        path = fixture / entry["path"]
        if not path.is_file() or path.stat().st_size != 400**3 * 8 or sha256(path) != entry["sha256"]:
            failures.append(f"field_hash_or_size:{name}")
    hash_file = fixture / "fixture_hashes.sha256"
    if not hash_file.is_file():
        failures.append("fixture_hash_manifest_missing")
    else:
        for raw in hash_file.read_text(encoding="utf-8").splitlines():
            expected, name = raw.split(maxsplit=1)
            path = fixture / name.strip()
            if not path.is_file() or sha256(path) != expected:
                failures.append(f"fixture_hash_manifest:{name}")
    particles = read_csv(fixture / "initial_particles.csv")
    expected_count = int(manifest["component_contract"]["expected_count"])
    if len(particles) != expected_count or int(manifest["component_contract"]["actual_count"]) != expected_count:
        failures.append("particle_count")
    source_h = float(sum(float(row["source_h_volume_nm3"]) for row in particles))
    source_h_rel = abs(source_h - TARGET_H) / TARGET_H
    if source_h_rel > H_TOL:
        failures.append("native_profile_h_volume")
    if abs(float(manifest["initial_beta_inventory"]["profile_inventory_relative_error"]) - source_h_rel) > 1.0e-12:
        failures.append("profile_h_ledger")
    inventory_rel = float(manifest["initial_canonical_inventory"]["field_relative_error"])
    if inventory_rel > 1.0e-12:
        failures.append("canonical_inventory")
    matrix_xag = float(manifest["derived_matrix_baseline"]["observed_far_field_xAg_h_lt_1e-4"])
    if abs(matrix_xag - TARGET_XAG) > XAG_TOL:
        failures.append("matrix_xAg")
    assembly = manifest["assembly_contract"]
    for key in ("profile_scaling_used", "profile_interpolation_used", "profile_rotation_used", "analytic_profile_used", "clipping_used", "normalization_used"):
        if assembly.get(key) is not False:
            failures.append(f"forbidden_assembly:{key}")
    if assembly.get("reverse_order_invariance") != "PASS_BITWISE_CANONICAL_ORDER_INVARIANCE":
        failures.append("reverse_order_invariance")
    physical = manifest["physical_contract"]
    for key in ("GP_enabled", "GP_birth_enabled", "GP_release_enabled", "external_source_enabled", "new_beta_nucleation_enabled"):
        if physical.get(key) is not False:
            failures.append(f"forbidden_physics:{key}")
    if physical.get("elasticity_enabled") is not True:
        failures.append("elasticity_disabled")
    if manifest.get("initial_state_class") != "MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1":
        failures.append("initial_state_class")
    meta = json.loads((fixture / "init_meta.json").read_text(encoding="utf-8"))
    if meta.get("phi_path") != fields["phi"]["path"] or meta.get("xB_path") != fields["xB_alpha"]["path"] or meta.get("fresh_dY_dt_prev_contract") != "zero_for_fresh_dynamic_start":
        failures.append("zero_macrostep_raw_input_contract")
    # Independent periodic pair audit: it validates both declared and profile
    # support separations, so a materializer-side bookkeeping error cannot be
    # hidden by the manifest statement.
    min_gap = float("inf")
    for left, row in enumerate(particles):
        for other in particles[left + 1:]:
            distance = periodic_distance(json.loads(row["center_grid"]), json.loads(other["center_grid"]), 400.0)
            declared = float(row["registered_radius_nm"]) + float(other["registered_radius_nm"]) + 16.0
            support = float(row["source_support_radius_nm"]) + float(other["source_support_radius_nm"])
            gap = distance - max(declared, support)
            min_gap = min(min_gap, gap)
            if gap <= 0.0:
                failures.append("periodic_overlap")
    if not math.isfinite(min_gap):
        failures.append("spatial_pair_audit")
    # A zero-macrostep preflight is intentionally parser/input-only.  It never
    # advances the PF state, writes a checkpoint, or changes the fixture.
    zero_preflight = not failures
    result = {
        "schema": "PF_400CUBE_PSD_CAMPAIGN_INTEGER_GATE_STATIC_AUDIT_V2",
        "case": case, "fixture_root": str(fixture.resolve()), "fixture_manifest_sha256": sha256(manifest_path),
        "particle_count": len(particles), "profile_h_volume_nm3": source_h, "profile_h_volume_relative_error": source_h_rel,
        "assembled_bounded_union_h_volume_nm3": float(manifest["initial_beta_inventory"]["assembled_bounded_union_h_volume_nm3"]),
        "canonical_inventory_relative_error": inventory_rel, "matrix_xAg": matrix_xag, "matrix_xAg_delta": matrix_xag - TARGET_XAG,
        "minimum_periodic_surface_gap_nm": min_gap, "reverse_order_invariance": assembly.get("reverse_order_invariance"),
        "zero_macrostep_production_preflight": "PASS_ZERO_MACROSTEP_RAW_INPUT_PRECHECK_V2" if zero_preflight else "BLOCKED_ZERO_MACROSTEP_RAW_INPUT_PRECHECK_V2",
        "failures": sorted(set(failures)), "status": AUDIT_PASS if not failures else "BLOCKED_400CUBE_PSD_CAMPAIGN_INTEGER_GATE_STATIC_AUDIT_V2",
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "static_audit.json", result)
    (out / "status.txt").write_text(result["status"] + "\n", encoding="utf-8")
    return result


def aggregate(campaign: Path, case_manifest: Path, out: Path) -> dict[str, Any]:
    cases = json.loads(case_manifest.read_text(encoding="utf-8"))["cases"]
    results: list[dict[str, Any]] = []
    for case in cases:
        audit = campaign / "fixtures" / case["case_id"] / "static_audit" / "static_audit.json"
        if not audit.is_file():
            results.append({"case": {"case_id": case["case_id"], "case_label": case["case_label"]}, "status": "MISSING_STATIC_AUDIT", "failures": ["missing_static_audit"]})
        else:
            results.append(json.loads(audit.read_text(encoding="utf-8")))
    out.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    psd_rows: list[dict[str, Any]] = []
    hash_lines: list[str] = []
    for result in results:
        case = result["case"]
        row = {"case_id": case["case_id"], "case_label": case["case_label"], "status": result["status"], "particle_count": result.get("particle_count", ""), "zero_macrostep_preflight": result.get("zero_macrostep_production_preflight", ""), "failure_count": len(result.get("failures", []))}
        summary_rows.append(row)
        if result["status"] == AUDIT_PASS:
            inventory_rows.append({"case_id": case["case_id"], "case_label": case["case_label"], "profile_h_volume_nm3": result["profile_h_volume_nm3"], "profile_h_volume_relative_error": result["profile_h_volume_relative_error"], "assembled_bounded_union_h_volume_nm3": result["assembled_bounded_union_h_volume_nm3"], "canonical_inventory_relative_error": result["canonical_inventory_relative_error"], "matrix_xAg": result["matrix_xAg"], "matrix_xAg_delta": result["matrix_xAg_delta"]})
            fixture = campaign / "fixtures" / case["case_id"]
            particle_rows = read_csv(fixture / "initial_particles.csv")
            values = np.asarray([float(row["registered_radius_nm"]) for row in particle_rows])
            mean = float(np.mean(values)); sigma = float(np.std(values))
            psd_rows.append({"case_id": case["case_id"], "case_label": case["case_label"], "mean_radius_nm": mean, "CV": sigma / mean, "Q10_nm": float(np.quantile(values, .1, method="nearest")), "Q50_nm": float(np.quantile(values, .5, method="nearest")), "Q90_nm": float(np.quantile(values, .9, method="nearest")), "M6_number_nm6": float(np.mean(values**6))})
            for raw in (fixture / "fixture_hashes.sha256").read_text(encoding="utf-8").splitlines():
                hash_lines.append(f"{case['case_id']}/{raw}")
    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        fields = list(rows[0]) if rows else ["case_id", "case_label", "status"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    write_csv(out / "fixture_qualification_summary.csv", summary_rows)
    write_csv(out / "fixture_inventory_closure.csv", inventory_rows)
    write_csv(out / "casewise_matrix_baselines.csv", inventory_rows)
    write_csv(out / "fixture_psd_statistics.csv", psd_rows)
    (out / "fixture_hash_manifest.sha256").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    passes = sum(result["status"] == AUDIT_PASS for result in results)
    result = {"schema": "PF_400CUBE_PSD_CAMPAIGN_INTEGER_GATE_AGGREGATE_V2", "fixture_pass_count": passes, "fixture_blocked_count": len(results) - passes, "status": CAMPAIGN_PASS if passes == 21 else "BLOCKED_FIXTURE_QUALIFICATION", "maximum_profile_h_volume_relative_error": max((float(row["profile_h_volume_relative_error"]) for row in inventory_rows), default=float("nan")), "maximum_matrix_xAg_difference": max((abs(float(row["matrix_xAg_delta"])) for row in inventory_rows), default=float("nan")), "maximum_canonical_inventory_relative_error": max((float(row["canonical_inventory_relative_error"]) for row in inventory_rows), default=float("nan"))}
    write_json(out / "fixture_qualification_aggregate.json", result)
    (out / "status.txt").write_text(result["status"] + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("fixture")
    one.add_argument("--fixture", type=Path, required=True); one.add_argument("--out", type=Path, required=True)
    all_cases = sub.add_parser("aggregate")
    all_cases.add_argument("--campaign", type=Path, required=True); all_cases.add_argument("--case-manifest", type=Path, required=True); all_cases.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fixture":
        result = audit_fixture(args.fixture.resolve(), args.out.resolve())
    else:
        result = aggregate(args.campaign.resolve(), args.case_manifest.resolve(), args.out.resolve())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
