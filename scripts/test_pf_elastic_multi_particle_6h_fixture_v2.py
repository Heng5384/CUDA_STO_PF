#!/usr/bin/env python3
"""Static regression checks for the V2 phase-consistent fixture contract."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import materialize_pf_elastic_multi_particle_6h_fixture_v1 as v1
import materialize_pf_elastic_multi_particle_6h_fixture_v2 as target
import test_pf_elastic_multi_particle_6h_fixture_v1 as v1test


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def v2_state(root: Path) -> Dict[str, Any]:
    state = v1test.build_synthetic_library(root)
    profiles_root = state["library_path"].parent.parent / "profiles"
    for path in profiles_root.glob("*/profile_manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["composition"] = {"far_field_xB_mean": 0.0062}
        manifest["runtime_load_contract"] = {"dY_dt_prev_initialization": "zero_for_fresh_dynamic_start"}
        write_json(path, manifest)
    library = json.loads(state["library_path"].read_text(encoding="utf-8"))
    for entry in library["profiles"]:
        path = (state["library_path"].parent / entry["profile_manifest_path"]).resolve()
        entry["profile_manifest_sha256"] = digest(path)
    write_json(state["library_path"], library)
    write_json(state["selection_path"], {
        "library_manifest_sha256": digest(state["library_path"]),
        "selection_status": "PASS_REUSE_ONLY_INDIVIDUALLY_QUALIFIED_PROFILES",
    })
    return state


def particle(radius: float, center: List[int], particle_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
    row = v1test.particle(radius, center, particle_id, state)
    manifest_path = state["library_path"].parent.parent / "profiles" / f"R{str(radius).replace('.', 'p')}" / "profile_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row["source_xB_alpha_sha256"] = manifest["fields"]["xB_alpha"]["sha256"]
    return row


def spec(state: Dict[str, Any], radii: List[float]) -> Dict[str, Any]:
    provisional = [particle(radius, [0, 0, 0], f"MP{i + 1:03d}", state) for i, radius in enumerate(radii)]
    centers = v1.replay_hard_core_centers(provisional, (96, 96, 96), 1.0, 4.0, 2026073091, 20000)
    rows = [particle(radius, center, f"MP{i + 1:03d}", state) for i, (radius, center) in enumerate(zip(radii, centers))]
    return {
        "schema": target.SPEC_SCHEMA, "composition_contract": target.COMPOSITION_CONTRACT,
        "fixture_id": "synthetic_v2", "validation_only": True, "scientific_status": "SYNTHETIC_STATIC_TEST_ONLY",
        "selected_library_manifest_sha256": digest(state["library_path"]),
        "selected_library_selection_sha256": digest(state["selection_path"]),
        "target": {"grid": [96, 96, 96], "dx_nm": 1.0, "lambda_sm_nm": 4.0, "temperature_C": 380.0,
                   "dt_code": 0.02, "mean_C_B_tot": 0.03, "matrix_xB_reference": 0.0062,
                   "xB_max_safe": 0.499999, "component_h_threshold": 1.0e-4, "phase_storage_alpha_floor": 1.0e-10},
        "placement": {"method": "deterministic_hard_core_rejection_integer_grid_v1", "seed": 2026073091,
                      "max_trials_per_particle": 20000, "random_non_lattice": True},
        "particles": rows,
    }


def run(state: Dict[str, Any], spec_value: Dict[str, Any], output: Path, kind: str = "E2") -> Dict[str, Any]:
    spec_path = output.parent / f"{output.name}_{kind}.json"
    write_json(spec_path, spec_value)
    args = type("Args", (), {"library_root": state["library_path"].parent.parent, "selection_provenance": state["selection_path"], "spec": spec_path, "kind": kind, "out": output})
    return target.materialize(args)


def rejects(state: Dict[str, Any], source: Dict[str, Any], patch: Any, label: str, root: Path) -> bool:
    candidate = copy.deepcopy(source)
    patch(candidate)
    try:
        run(state, candidate, root / f"reject_{label}")
    except ValueError:
        return True
    return False


def main() -> int:
    checks: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="pf_elastic_multi_particle_v2_test_") as raw:
        root = Path(raw)
        state = v2_state(root / "state")
        multi = spec(state, [10.5, 9.5, 8.0, 8.0])
        one = spec(state, [8.0])
        run(state, one, root / "one")
        manifest_one = json.loads((root / "one" / "fixture_manifest.json").read_text())
        row = manifest_one["particles"][0]
        source_dir = Path(row["source_profile_path"]).parent / "../../shared"
        source_delta = np.fromfile(source_dir / "delta_C_relaxation.raw.f64", dtype="<f8").reshape((96, 96, 96))
        output_delta = np.fromfile(root / "one" / "delta_C_relaxation_total.raw.f64", dtype="<f8").reshape((96, 96, 96))
        shift = tuple(int(value) - 48 for value in row["center_grid"])
        first = run(state, multi, root / "first")
        second = run(state, multi, root / "second")
        e2 = json.loads((root / "first" / "fixture_manifest.json").read_text())
        s0 = run(state, multi, root / "s0", "S0")
        e2_xb = np.fromfile(root / "first" / "xB_alpha.raw.f64", dtype="<f8")
        e2_c = np.fromfile(root / "first" / "C_B_tot.raw.f64", dtype="<f8")
        e2_h = np.fromfile(root / "first" / "h_phi.raw.f64", dtype="<f8")
        e2_dx = np.fromfile(root / "first" / "delta_x_alpha_total.raw.f64", dtype="<f8")
        reconstructed = e2_h + (1.0 - e2_h) * (e2["inventory"]["matrix_xB"] + e2_dx)
        formula_max_abs = float(np.max(np.abs(e2_c - reconstructed)))
        checks.extend([
            {"test": "one_particle_exact_delta_C_identity", "pass": np.array_equal(output_delta, np.roll(source_delta, shift, axis=(0, 1, 2))), "detail": ""},
            {"test": "one_particle_shortcut_declared", "pass": manifest_one["combination"]["one_profile_exact_delta_C_shortcut"], "detail": ""},
            {"test": "repeated_materialization_bytewise", "pass": all(digest(root / "first" / name) == digest(root / "second" / name) for name in ("phi.raw.f64", "xB_alpha.raw.f64", "C_B_tot.raw.f64", "fixture_manifest.json")), "detail": ""},
            {"test": "phase_consistent_formula", "pass": formula_max_abs <= 1.0e-15, "detail": f"max_abs={formula_max_abs:.3e}"},
            {"test": "canonical_inventory", "pass": e2["inventory"]["relative_error"] <= 1.0e-12, "detail": str(e2["inventory"]["relative_error"])},
            {"test": "bounded_xB_without_clipping", "pass": float(np.min(e2_xb)) > 0.0 and float(np.max(e2_xb)) < 0.499999 and not e2["combination"]["clipping_used"], "detail": ""},
            {"test": "absolute_xB_never_superposed", "pass": not e2["combination"]["absolute_xB_superposed"] and not e2["combination"]["direct_delta_C_sum_used"], "detail": ""},
            {"test": "component_count", "pass": e2["component_contract"]["actual_count"] == len(multi["particles"]), "detail": ""},
            {"test": "all_field_hashes", "pass": all(digest(root / "first" / item["path"]) == item["sha256"] for item in e2["fields"].values()), "detail": ""},
            {"test": "S0_E2_inventory_equal", "pass": abs(s0["inventory"]["actual_total_C_B_tot"] - first["inventory"]["actual_total_C_B_tot"]) <= 1.0e-10, "detail": ""},
            {"test": "S0_E2_h_volume_equal", "pass": abs(s0["inventory"]["combined_h_volume_nm3"] - first["inventory"]["combined_h_volume_nm3"]) <= 1.0e-9, "detail": ""},
            {"test": "S0_E2_matrix_baseline_equal", "pass": abs(s0["inventory"]["matrix_xB"] - first["inventory"]["matrix_xB"]) <= 1.0e-15, "detail": ""},
            {"test": "S0_projection_conserves_without_discard", "pass": s0["combination"]["s0_phase_storage_projection"]["mass_discarded"] == 0.0 and abs(s0["combination"]["s0_phase_storage_projection"]["correction_sum_error"]) <= 1.0e-12, "detail": ""},
        ])
        shuffled = copy.deepcopy(multi)
        shuffled["particles"] = list(reversed(shuffled["particles"]))
        run(state, shuffled, root / "shuffled")
        checks.append({"test": "seed_order_bytewise", "pass": all(digest(root / "first" / name) == digest(root / "shuffled" / name) for name in ("phi.raw.f64", "delta_x_alpha_total.raw.f64", "C_B_tot.raw.f64", "fixture_manifest.json")), "detail": ""})
        checks.extend([
            {"test": "duplicate_id_rejected", "pass": rejects(state, multi, lambda x: x["particles"][1].update({"particle_id": x["particles"][0]["particle_id"]}), "duplicate", root), "detail": ""},
            {"test": "xB_hash_mismatch_rejected", "pass": rejects(state, multi, lambda x: x["particles"][0].update({"source_xB_alpha_sha256": "0" * 64}), "xbhash", root), "detail": ""},
            {"test": "wrong_contract_rejected", "pass": rejects(state, multi, lambda x: x.update({"composition_contract": "DIRECT_DELTA_C_V1"}), "contract", root), "detail": ""},
            {"test": "periodic_overlap_rejected", "pass": rejects(state, multi, lambda x: x["particles"][1].update({"center_grid": x["particles"][0]["center_grid"]}), "overlap", root), "detail": ""},
            {"test": "fresh_history_explicit", "pass": e2["time_level_contract"]["dY_dt_prev"] == "zero_for_fresh_dynamic_start", "detail": ""},
        ])
    failed = [item for item in checks if not item["pass"]]
    print(json.dumps({"status": "PASS" if not failed else "FAIL", "test_count": len(checks), "failed_count": len(failed), "checks": checks}, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
