#!/usr/bin/env python3
"""Fail-closed static audit for the three 246^3 library handoff fixtures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


SCHEMA = "PF_246CUBE_LIBRARY_HANDOFF_MANIFEST_V1"
EXPECTED_LABELS = ("replicate_A", "replicate_B", "replicate_C")
EXPECTED_SEEDS = {
    "replicate_A": 18278234711707939752,
    "replicate_B": 6256128897973916905,
    "replicate_C": 4209997954605651191,
}
EXPECTED_HISTOGRAM = {
    "8.0": 4,
    "8.5": 19,
    "9.0": 19,
    "9.5": 17,
    "10.0": 17,
    "10.5": 16,
    "11.0": 4,
    "11.5": 0,
}
LIBRARY_SHA256 = (
    "58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe"
)
SELECTION_SHA256 = (
    "56c44d8f72b27bb462dffe89b59cb2fcb2ff0bf8807dec9d9cac31d2bd7fcbe3"
)
INITIAL_STATE_CLASS = (
    "MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1"
)
FIELD_NAMES = (
    "phi",
    "h_phi",
    "delta_C_relaxation_total",
    "C_B_tot",
    "xB_alpha",
    "Y",
    "dY_dt_prev",
)
GRID = (246, 246, 246)
VALUE_COUNT = math.prod(GRID)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_fixture(value: str) -> Tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("fixture must be LABEL=PATH")
    return label, Path(path)


def raw_stats(path: Path) -> Dict[str, Any]:
    if path.stat().st_size != VALUE_COUNT * 8:
        raise ValueError(f"wrong raw field size: {path}")
    values = np.memmap(path, dtype="<f8", mode="r", shape=GRID)
    return {
        "finite": bool(np.all(np.isfinite(values))),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values, dtype=np.float64)),
    }


def deterministic_probe(
    materializer: Path,
    historical: Path,
    library_root: Path,
    selection: Path,
    label: str,
    manifest: Path,
    order: str,
) -> Dict[str, Any]:
    command = [
        "python3",
        str(materializer),
        "--historical-manifest",
        str(historical),
        "--library-root",
        str(library_root),
        "--selection-provenance",
        str(selection),
        "--replicate",
        label,
        "--input-order",
        order,
        "--compare-to-manifest",
        str(manifest),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            f"{label}/{order} deterministic probe failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return json.loads(result.stdout)


def audit_fixture(
    label: str,
    root: Path,
    materializer: Path,
    historical: Path,
    library_root: Path,
    selection: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    manifest_path = root / "fixture_manifest.json"
    manifest = load_json(manifest_path)
    gates: Dict[str, bool] = {}
    gates["schema"] = manifest.get("schema") == SCHEMA
    gates["replicate_id"] = manifest.get("replicate_id") == label
    gates["initial_state_class"] = (
        manifest.get("initial_state_class") == INITIAL_STATE_CLASS
    )
    gates["validation_only"] = manifest.get("validation_only") is True
    gates["grid"] = manifest.get("grid", {}).get("Nx") == GRID[0] and all(
        manifest.get("grid", {}).get(key) == GRID[axis]
        for axis, key in enumerate(("Nx", "Ny", "Nz"))
    )
    gates["library_hash"] = (
        manifest.get("profile_library_manifest_sha256") == LIBRARY_SHA256
        and manifest.get("selection_provenance_sha256") == SELECTION_SHA256
    )
    placement = manifest.get("placement", {})
    gates["seed"] = (
        int(placement.get("seed_unsigned64", -1)) == EXPECTED_SEEDS[label]
    )
    gates["separation"] = (
        float(placement.get("minimum_registered_separation_margin_nm", -1.0))
        > 0.0
        and float(placement.get("minimum_actual_support_margin_nm", -1.0))
        > 0.0
        and placement.get("periodic_image_overlap_status") == "PASS"
    )
    gates["non_lattice"] = placement.get("regular_lattice_rejected") is True

    mappings = manifest.get("particle_library_mappings", [])
    particle_ids = [str(row.get("particle_id")) for row in mappings]
    centers = [tuple(row.get("center_grid", [])) for row in mappings]
    histogram = Counter(
        f"{float(row['registered_radius_nm']):.1f}" for row in mappings
    )
    histogram_full = {
        radius: int(histogram.get(radius, 0))
        for radius in EXPECTED_HISTOGRAM
    }
    required_mapping = (
        "particle_id",
        "replicate_id",
        "registered_radius_nm",
        "library_entry_id",
        "library_entry_sha256",
        "center_nm",
        "orientation",
        "source_phi_sha256",
        "source_delta_C_relaxation_sha256",
        "effective_h_volume",
        "canonical_particle_inventory",
    )
    gates["particle_count"] = len(mappings) == 96
    gates["particle_ids_unique"] = len(set(particle_ids)) == len(particle_ids)
    gates["centers_unique_and_in_domain"] = (
        len(set(centers)) == len(centers)
        and all(
            len(center) == 3
            and all(0 <= int(value) < GRID[axis] for axis, value in enumerate(center))
            for center in centers
        )
    )
    gates["mapping_complete"] = all(
        all(key in row for key in required_mapping) for row in mappings
    )
    gates["mapping_library_exact"] = all(
        row.get("replicate_id") == label
        and str(row.get("library_entry_id", "")).startswith("R")
        and len(str(row.get("library_entry_sha256", ""))) == 64
        and len(str(row.get("source_phi_sha256", ""))) == 64
        and len(str(row.get("source_delta_C_relaxation_sha256", ""))) == 64
        for row in mappings
    )
    gates["histogram"] = histogram_full == EXPECTED_HISTOGRAM

    assembly = manifest.get("assembly_contract", {})
    gates["no_interpolation_scaling_or_minimizer"] = all(
        assembly.get(key) is False
        for key in (
            "radial_interpolation_used",
            "profile_scaling_used",
            "spatial_resampling_used",
            "rotation_used",
            "analytic_tanh_used",
            "clipping_used",
            "normalization_used",
            "optimizer_invoked",
            "common_multi_particle_pre_relaxation_run",
        )
    )
    physical = manifest.get("physical_contract", {})
    gates["prohibited_paths_off"] = all(
        physical.get(key) is False
        for key in (
            "GP_enabled",
            "GP_birth_enabled",
            "GP_release_enabled",
            "external_source_enabled",
            "new_beta_nucleation_enabled",
        )
    )
    inventory = manifest.get("initial_canonical_inventory", {})
    gates["mass_inventory"] = (
        inventory.get("status") == "PASS_MACHINE_PRECISION"
        and float(inventory.get("field_relative_error", math.inf)) <= 1.0e-14
        and float(
            inventory.get("decomposition_relative_error", math.inf)
        )
        <= 1.0e-14
    )
    zero_mode = manifest.get("initial_zero_mode_provenance", {})
    gates["zero_mode_provenance"] = (
        zero_mode.get("zero_mode") == "PF_CONSERVED_Y_ZERO_MODE_V1"
        and zero_mode.get("backend") == "HOST_NEWTON_BISECTION_V1"
        and zero_mode.get("explicit_context") == "SM_EXPLICIT_CONTEXT_N_V1"
        and zero_mode.get("reaction_discretization") == "SM_TANGENT_N_V1"
    )
    component = manifest.get("component_contract", {})
    gates["component_count"] = (
        int(component.get("expected_count", -1)) == 96
        and int(component.get("actual_count", -1)) == 96
        and len(component.get("particle_to_component", {})) == 96
        and len(set(component.get("particle_to_component", {}).values())) == 96
    )

    field_audits: Dict[str, Any] = {}
    for name in FIELD_NAMES:
        row = manifest.get("fields", {}).get(name, {})
        path = root / str(row.get("path", ""))
        if not path.is_file() or sha256(path) != row.get("sha256"):
            raise ValueError(f"{label}: field identity mismatch: {name}")
        field_audits[name] = {
            "sha256": row["sha256"],
            **raw_stats(path),
        }
    gates["raw_fields_finite"] = all(
        row["finite"] for row in field_audits.values()
    )
    gates["field_bounds"] = (
        -1.0e-6 <= field_audits["phi"]["minimum"]
        and field_audits["phi"]["maximum"] <= 1.0 + 1.0e-6
        and 0.0 < field_audits["xB_alpha"]["minimum"]
        and field_audits["xB_alpha"]["maximum"] < 0.499999
    )

    probes = {
        order: deterministic_probe(
            materializer,
            historical,
            library_root,
            selection,
            label,
            manifest_path,
            order,
        )
        for order in ("canonical", "reverse")
    }
    gates["deterministic_materialization"] = all(
        row.get("status") == "PASS_DETERMINISTIC_MATERIALIZATION"
        and row.get("raw_field_hashes_equal") is True
        for row in probes.values()
    )
    if not all(gates.values()):
        failed = [key for key, value in gates.items() if not value]
        raise ValueError(f"{label}: failed static gates: {failed}")
    return (
        {
            "status": "PASS_246CUBE_LIBRARY_HANDOFF_STATIC_V1",
            "replicate": label,
            "fixture_root": str(root),
            "fixture_manifest_sha256": sha256(manifest_path),
            "gates": gates,
            "field_audits": field_audits,
            "deterministic_probes": probes,
            "histogram": histogram_full,
            "placement": placement,
            "inventory": inventory,
            "matrix_baseline": manifest.get("derived_matrix_baseline"),
        },
        mappings,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture", action="append", type=parse_fixture, required=True
    )
    parser.add_argument("--materializer", type=Path, required=True)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--selection-provenance", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fixtures = dict(args.fixture)
    if set(fixtures) != set(EXPECTED_LABELS):
        raise SystemExit(
            f"fixtures must be exactly {EXPECTED_LABELS}, got {tuple(fixtures)}"
        )
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)
    try:
        audits: Dict[str, Any] = {}
        all_mappings: Dict[str, List[Dict[str, Any]]] = {}
        for label in EXPECTED_LABELS:
            audits[label], all_mappings[label] = audit_fixture(
                label,
                fixtures[label],
                args.materializer,
                args.historical_manifest,
                args.library_root,
                args.selection_provenance,
            )
        first_histogram = audits[EXPECTED_LABELS[0]]["histogram"]
        psd_equal = all(
            audits[label]["histogram"] == first_histogram
            for label in EXPECTED_LABELS
        )
        center_sets = [
            tuple(tuple(row["center_grid"]) for row in all_mappings[label])
            for label in EXPECTED_LABELS
        ]
        centers_different = len(set(center_sets)) == len(EXPECTED_LABELS)
        phi_hashes = [
            audits[label]["field_audits"]["phi"]["sha256"]
            for label in EXPECTED_LABELS
        ]
        x_b_hashes = [
            audits[label]["field_audits"]["xB_alpha"]["sha256"]
            for label in EXPECTED_LABELS
        ]
        field_hashes_different = (
            len(set(phi_hashes)) == len(EXPECTED_LABELS)
            and len(set(x_b_hashes)) == len(EXPECTED_LABELS)
        )
        ensemble_gates = {
            "discrete_PSD_identity": psd_equal,
            "spatial_center_independence": centers_different,
            "field_hash_independence": field_hashes_different,
            "replicate_seed_independence": len(set(EXPECTED_SEEDS.values())) == 3,
        }
        if not all(ensemble_gates.values()):
            failed = [key for key, value in ensemble_gates.items() if not value]
            raise ValueError(f"ensemble static gates failed: {failed}")
        summary = {
            "status": "PASS_246CUBE_THREE_SEED_STATIC_QUALIFICATION_V1",
            "replicates": audits,
            "ensemble_gates": ensemble_gates,
            "expected_histogram": EXPECTED_HISTOGRAM,
            "profile_library_manifest_sha256": LIBRARY_SHA256,
            "selection_provenance_sha256": SELECTION_SHA256,
        }
        write_json(args.out / "static_audit.json", summary)
        with (args.out / "spatial_statistics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            keys = (
                "replicate",
                "seed_unsigned64",
                "minimum_periodic_pair_distance_nm",
                "minimum_registered_separation_margin_nm",
                "minimum_actual_support_margin_nm",
                "nearest_neighbor_mean_nm",
                "nearest_neighbor_std_nm",
                "nearest_neighbor_cv",
            )
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for label in EXPECTED_LABELS:
                placement = audits[label]["placement"]
                writer.writerow(
                    {
                        "replicate": label,
                        **{key: placement[key] for key in keys[1:]},
                    }
                )
        with (args.out / "mass_inventory.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            keys = (
                "replicate",
                "matrix_xB",
                "matrix_xAg",
                "target_total_code",
                "actual_total_code",
                "field_relative_error",
                "decomposition_relative_error",
            )
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for label in EXPECTED_LABELS:
                inventory = audits[label]["inventory"]
                matrix = audits[label]["matrix_baseline"]
                writer.writerow(
                    {
                        "replicate": label,
                        "matrix_xB": matrix["xB_alpha"],
                        "matrix_xAg": matrix["xAg"],
                        **{key: inventory[key] for key in keys[3:]},
                    }
                )
        (args.out / "status.txt").write_text(
            "PASS_246CUBE_THREE_SEED_STATIC_QUALIFICATION_V1\n",
            encoding="utf-8",
        )
    except ValueError as exc:
        (args.out / "status.txt").write_text(
            f"BLOCKED_246CUBE_THREE_SEED_STATIC_QUALIFICATION_V1\n{exc}\n",
            encoding="utf-8",
        )
        raise SystemExit(f"[fatal] {exc}") from exc
    print(json.dumps({"status": summary["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
