#!/usr/bin/env python3
"""Static/regression tests for the V1 elastic multi-particle materializer."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import materialize_pf_elastic_multi_particle_6h_fixture_v1 as target


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_raw(path: Path, value: np.ndarray) -> None:
    np.asarray(value, dtype="<f8").ravel(order="C").tofile(path)


def build_synthetic_library(root: Path) -> Dict[str, Any]:
    """Small provenance-equivalent library; all radius entries share raw fields."""
    shape = (96, 96, 96)
    shared = root / "shared"
    profiles_root = root / "profiles"
    library_dir = root / "library"
    shared.mkdir(parents=True)
    profiles_root.mkdir()
    library_dir.mkdir()
    coords = np.arange(96, dtype=np.float64)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    rr = np.sqrt((xx - 48.0) ** 2 + (yy - 48.0) ** 2 + (zz - 48.0) ** 2)
    phi = 0.5 * (1.0 + np.tanh((4.0 - rr) / 2.0))
    h = target.h_of_phi(phi)
    xb = 0.0062 + 2.0e-5 * np.exp(-(rr / 12.0) ** 2)
    delta = (1.0 - h) * (xb - 0.0062)
    fields_values = {
        "phi": phi, "h_phi": h, "delta_C_relaxation": delta,
        "xB_alpha": xb, "Y": target.logit(xb),
        "C_B_tot": (1.0 - h) * xb + h, "dY_dt_prev": np.zeros(shape),
    }
    field_meta: Dict[str, Dict[str, str]] = {}
    for name, value in fields_values.items():
        path = shared / f"{name}.raw.f64"
        write_raw(path, value)
        field_meta[name] = {"path": f"../../shared/{path.name}", "sha256": digest(path), "dtype": "float64-le", "order": "C"}
    h_volume = float(np.sum(h) * 1.0)
    actual_radius = (3.0 * h_volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    entries = []
    manifests: Dict[float, Dict[str, Any]] = {}
    for radius in (8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5):
        label = f"R{str(radius).replace('.', 'p')}"
        profile_dir = profiles_root / label
        profile_dir.mkdir()
        manifest = {
            "schema": target.PROFILE_SCHEMA,
            "grid": {"Nx": 96, "Ny": 96, "Nz": 96, "dx_nm": 1.0},
            "lambda_sm_nm": 4.0, "temperature_C": 380.0, "v_B": 1.0,
            "orientation_label": "variant_100_identity",
            "geometry": {"target_equivalent_radius_nm": radius, "actual_equivalent_radius_nm": actual_radius, "h_volume_nm3": h_volume, "axis_ratio_major_minor": 1.0},
            "fields": field_meta,
        }
        path = profile_dir / "profile_manifest.json"
        write_json(path, manifest)
        manifests[radius] = manifest
        entries.append({"target_radius_nm": radius, "profile_manifest_path": f"../profiles/{label}/profile_manifest.json", "profile_manifest_sha256": digest(path)})
    library = {
        "schema": target.LIBRARY_SCHEMA, "profile_count": 8, "radius_ladder_nm": [x["target_radius_nm"] for x in entries],
        "grid": {"Nx": 96, "Ny": 96, "Nz": 96, "dx_nm": 1.0}, "lambda_sm_nm": 4.0,
        "temperature_C": 380.0, "v_B": 1.0, "source_tree_sha256": "a" * 64, "binary_sha256": "b" * 64, "profiles": entries,
    }
    library_path = library_dir / "library_manifest.json"
    write_json(library_path, library)
    selection = {"library_manifest_sha256": digest(library_path), "selection_status": "PASS_REUSE_ONLY_INDIVIDUALLY_QUALIFIED_PROFILES"}
    selection_path = root / "selection_provenance.json"
    write_json(selection_path, selection)
    return {"library": library, "library_path": library_path, "selection_path": selection_path, "fields": field_meta, "manifests": manifests}


def particle(radius: float, center: List[int], item_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
    profile_dir = state["library_path"].parent.parent / "profiles" / f"R{str(radius).replace('.', 'p')}"
    manifest_path = profile_dir / "profile_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return {
        "particle_id": item_id, "registered_radius_nm": radius, "center_grid": center,
        "orientation_label": "variant_100_identity", "source_profile_manifest_sha256": digest(manifest_path),
        "source_phi_sha256": manifest["fields"]["phi"]["sha256"],
        "source_delta_C_relaxation_sha256": manifest["fields"]["delta_C_relaxation"]["sha256"],
    }


def base_spec(state: Dict[str, Any], particles: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Coordinates reproduce deterministic hard-core draws for the listed radii.
    return {
        "schema": target.SPEC_SCHEMA, "fixture_id": "synthetic", "validation_only": True,
        "selected_library_manifest_sha256": digest(state["library_path"]),
        "selected_library_selection_sha256": digest(state["selection_path"]),
        "target": {"grid": [96, 96, 96], "dx_nm": 1.0, "lambda_sm_nm": 4.0, "temperature_C": 380.0,
                   "dt_code": 0.02, "mean_C_B_tot": 0.03, "matrix_xB_reference": 0.0062,
                   "xB_max_safe": 0.499999, "component_h_threshold": 1.0e-4, "phase_storage_alpha_floor": 1.0e-10},
        "placement": {"method": "deterministic_hard_core_rejection_integer_grid_v1", "seed": 2026073091,
                      "max_trials_per_particle": 20000, "random_non_lattice": True},
        "particles": particles,
    }


def spec_with_replayed_centers(state: Dict[str, Any], radii: List[float]) -> Dict[str, Any]:
    provisional = [particle(radius, [0, 0, 0], f"MP{i + 1:03d}", state) for i, radius in enumerate(radii)]
    centers = target.replay_hard_core_centers(provisional, (96, 96, 96), 1.0, 4.0, 2026073091, 20000)
    particles = [particle(radius, center, f"MP{i + 1:03d}", state) for i, (radius, center) in enumerate(zip(radii, centers))]
    return base_spec(state, particles)


def run(state: Dict[str, Any], spec: Dict[str, Any], out: Path, kind: str = "E2") -> Dict[str, Any]:
    spec_path = out.parent / f"{out.name}_{kind}.spec.json"
    write_json(spec_path, spec)
    ns = type("Args", (), {"library_root": state["library_path"].parent.parent, "selection_provenance": state["selection_path"], "spec": spec_path, "kind": kind, "out": out})
    return target.materialize(ns)


def rejects(state: Dict[str, Any], spec: Dict[str, Any], patch: Any, label: str) -> bool:
    bad = copy.deepcopy(spec)
    patch(bad)
    try:
        run(state, bad, state["library_path"].parent.parent / f"reject_{label}")
    except ValueError:
        return True
    return False


def main() -> int:
    checks: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="pf_elastic_multi_particle_test_") as raw:
        root = Path(raw)
        state = build_synthetic_library(root / "library_root")
        spec = spec_with_replayed_centers(state, [10.5, 9.5, 8.0, 8.0])
        one = spec_with_replayed_centers(state, [8.0])
        run(state, one, root / "one")
        one_manifest = json.loads((root / "one" / "fixture_manifest.json").read_text())
        one_particle = one_manifest["particles"][0]
        source_phi = np.fromfile(
            Path(one_particle["source_profile_path"]) .parent / "../../shared/phi.raw.f64",
            dtype="<f8",
        ).reshape((96, 96, 96))
        source_delta = np.fromfile(
            Path(one_particle["source_profile_path"]) .parent / "../../shared/delta_C_relaxation.raw.f64",
            dtype="<f8",
        ).reshape((96, 96, 96))
        shift = tuple(int(v) - 48 for v in one_particle["center_grid"])
        output_phi = np.fromfile(root / "one" / "phi.raw.f64", dtype="<f8").reshape((96, 96, 96))
        output_delta = np.fromfile(root / "one" / "delta_C_relaxation_total.raw.f64", dtype="<f8").reshape((96, 96, 96))
        first = run(state, spec, root / "first")
        second = run(state, spec, root / "second")
        manifest_first = json.loads((root / "first" / "fixture_manifest.json").read_text())
        manifest_second = json.loads((root / "second" / "fixture_manifest.json").read_text())
        checks.extend([
            {"test": "one_particle_translation", "pass": np.array_equal(output_phi, np.roll(source_phi, shift, axis=(0, 1, 2))) and np.array_equal(output_delta, np.roll(source_delta, shift, axis=(0, 1, 2))), "detail": ""},
            {"test": "repeated_materialization_bytewise", "pass": all(digest(root / "first" / name) == digest(root / "second" / name) for name in ("phi.raw.f64", "delta_C_relaxation_total.raw.f64", "fixture_manifest.json")), "detail": ""},
            {"test": "canonical_inventory", "pass": manifest_first["inventory"]["relative_error"] <= 1.0e-12, "detail": str(manifest_first["inventory"]["relative_error"])},
            {"test": "matrix_baseline_reported", "pass": math.isfinite(manifest_first["inventory"]["matrix_xB"]), "detail": ""},
            {"test": "component_count", "pass": manifest_first["component_contract"]["actual_count"] == 4, "detail": ""},
            {"test": "bounded_no_clipping", "pass": manifest_first["combination"]["normalization_used"] is False, "detail": ""},
            {"test": "manifest_field_hashes", "pass": all(digest(root / "first" / meta["path"]) == meta["sha256"] for meta in manifest_first["fields"].values()), "detail": ""},
            {"test": "registered_radius_only", "pass": manifest_first["combination"]["radial_interpolation_used"] is False, "detail": ""},
        ])
        shuffled = copy.deepcopy(spec)
        shuffled["particles"] = list(reversed(shuffled["particles"]))
        run(state, shuffled, root / "shuffled")
        checks.append({"test": "complete_fixture_seed_order_bytewise", "pass": all(digest(root / "first" / name) == digest(root / "shuffled" / name) for name in ("phi.raw.f64", "delta_C_relaxation_total.raw.f64", "fixture_manifest.json")), "detail": ""})
        two = spec_with_replayed_centers(state, [10.5, 8.0])
        run(state, two, root / "two")
        two_flip = copy.deepcopy(two)
        two_flip["particles"] = list(reversed(two_flip["particles"]))
        run(state, two_flip, root / "two_flip")
        checks.append({"test": "two_particle_seed_order_bytewise", "pass": digest(root / "two" / "phi.raw.f64") == digest(root / "two_flip" / "phi.raw.f64"), "detail": ""})
        checks.extend([
            {"test": "duplicate_id_rejection", "pass": rejects(state, spec, lambda s: s["particles"][1].update({"particle_id": s["particles"][0]["particle_id"]}), "duplicate"), "detail": ""},
            {"test": "unregistered_radius_rejection", "pass": rejects(state, spec, lambda s: s["particles"][0].update({"registered_radius_nm": 8.25}), "radius"), "detail": ""},
            {"test": "rotated_variant_rejection", "pass": rejects(state, spec, lambda s: s["particles"][0].update({"orientation_label": "rotated"}), "rotated"), "detail": ""},
            {"test": "overlap_rejection", "pass": rejects(state, spec, lambda s: s["particles"][1].update({"center_grid": s["particles"][0]["center_grid"]}), "overlap"), "detail": ""},
            {"test": "periodic_image_overlap_rejection", "pass": rejects(state, spec, lambda s: s["particles"][1].update({"center_grid": [95, 95, 95]}), "periodic"), "detail": ""},
            {"test": "out_of_domain_rejection", "pass": rejects(state, spec, lambda s: s["particles"][0].update({"center_grid": [96, 0, 0]}), "domain"), "detail": ""},
            {"test": "source_hash_mismatch_rejection", "pass": rejects(state, spec, lambda s: s["particles"][0].update({"source_phi_sha256": "0" * 64}), "hash"), "detail": ""},
        ])
        nonfinite_state = build_synthetic_library(root / "nonfinite_library")
        phi_path = nonfinite_state["library_path"].parent.parent / "shared" / "phi.raw.f64"
        phi = np.fromfile(phi_path, dtype="<f8")
        phi[0] = float("nan")
        phi.tofile(phi_path)
        for profile_manifest in (nonfinite_state["library_path"].parent.parent / "profiles").glob("*/profile_manifest.json"):
            value = json.loads(profile_manifest.read_text())
            value["fields"]["phi"]["sha256"] = digest(phi_path)
            write_json(profile_manifest, value)
        library = json.loads(nonfinite_state["library_path"].read_text())
        for entry in library["profiles"]:
            profile_manifest = (nonfinite_state["library_path"].parent / entry["profile_manifest_path"]).resolve()
            entry["profile_manifest_sha256"] = digest(profile_manifest)
        write_json(nonfinite_state["library_path"], library)
        write_json(nonfinite_state["selection_path"], {"library_manifest_sha256": digest(nonfinite_state["library_path"]), "selection_status": "PASS_REUSE_ONLY_INDIVIDUALLY_QUALIFIED_PROFILES"})
        nonfinite_spec = spec_with_replayed_centers(nonfinite_state, [8.0])
        try:
            run(nonfinite_state, nonfinite_spec, root / "nonfinite_output")
            nonfinite_rejected = False
        except ValueError:
            nonfinite_rejected = True
        checks.append({"test": "nonfinite_field_rejection", "pass": nonfinite_rejected, "detail": ""})
        checks.append({"test": "fresh_dY_dt_prev_contract", "pass": manifest_first["time_level_contract"]["dY_dt_prev"] == "zero_for_fresh_dynamic_start", "detail": ""})
    failures = [item for item in checks if not item["pass"]]
    print(json.dumps({"status": "PASS" if not failures else "FAIL", "test_count": len(checks), "failed_count": len(failures), "checks": checks}, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
