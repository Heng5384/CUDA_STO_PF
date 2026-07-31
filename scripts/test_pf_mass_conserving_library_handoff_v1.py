#!/usr/bin/env python3
"""Static/negative regression matrix for the conditional handoff schema."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import materialize_pf_elastic_multi_particle_6h_fixture_v1 as base
import materialize_pf_mass_conserving_library_handoff_v1 as target
import test_pf_elastic_multi_particle_6h_fixture_v1 as synthetic


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def conditional_spec(
    state: Dict[str, Any], radii: List[float]
) -> Dict[str, Any]:
    value = synthetic.spec_with_replayed_centers(state, radii)
    value["schema"] = target.SPEC_SCHEMA
    value["initial_state_class"] = target.INITIAL_STATE_CLASS
    value["profile_library_manifest_sha256"] = synthetic.digest(
        state["library_path"]
    )
    value["selection_provenance_sha256"] = synthetic.digest(
        state["selection_path"]
    )
    return value


def run(
    state: Dict[str, Any], spec: Dict[str, Any], spec_path: Path, out: Path
) -> Dict[str, Any]:
    write_json(spec_path, spec)
    args = type(
        "Args",
        (),
        {
            "library_root": state["library_path"].parent.parent,
            "selection_provenance": state["selection_path"],
            "spec": spec_path,
            "out": out,
        },
    )
    return target.materialize(args)


def rejected(
    state: Dict[str, Any],
    source: Dict[str, Any],
    patch,
    root: Path,
    label: str,
) -> bool:
    value = copy.deepcopy(source)
    patch(value)
    try:
        run(
            state,
            value,
            root / f"{label}.json",
            root / f"reject_{label}",
        )
    except ValueError:
        return True
    return False


def main() -> int:
    checks: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="pf_mass_conserving_handoff_v1_"
    ) as raw:
        root = Path(raw)
        state = synthetic.build_synthetic_library(root / "library_root")
        # The synthetic fixture keeps the production logic but uses local
        # test hashes.  Production constants are restored when this process
        # exits.
        target.FROZEN_LIBRARY_SHA256 = synthetic.digest(
            state["library_path"]
        )
        target.FROZEN_SELECTION_SHA256 = synthetic.digest(
            state["selection_path"]
        )
        target.FROZEN_SOURCE_TREE_SHA256 = "a" * 64
        target.FROZEN_BINARY_SHA256 = "b" * 64

        spec = conditional_spec(state, [10.5, 9.5, 8.0, 8.0])
        first = run(state, spec, root / "first.json", root / "first")
        second = run(state, spec, root / "second.json", root / "second")
        reverse = copy.deepcopy(spec)
        reverse["particles"] = list(reversed(reverse["particles"]))
        run(state, reverse, root / "reverse.json", root / "reverse")

        exact_files = (
            "phi.raw.f64",
            "xB_alpha.raw.f64",
            "C_B_tot.raw.f64",
            "delta_C_relaxation_total.raw.f64",
            "fixture_manifest.json",
        )
        checks.extend(
            [
                {
                    "test": "01_exact_registered_radius_load",
                    "pass": len(first["particle_library_mappings"]) == 4,
                },
                {
                    "test": "02_unregistered_radius_rejected",
                    "pass": rejected(
                        state,
                        spec,
                        lambda x: x["particles"][0].update(
                            {"registered_radius_nm": 8.25}
                        ),
                        root,
                        "unregistered",
                    ),
                },
                {
                    "test": "03_library_hash_mismatch_rejected",
                    "pass": rejected(
                        state,
                        spec,
                        lambda x: x.update(
                            {"profile_library_manifest_sha256": "0" * 64}
                        ),
                        root,
                        "library_hash",
                    ),
                },
                {
                    "test": "04_duplicate_particle_id_rejected",
                    "pass": rejected(
                        state,
                        spec,
                        lambda x: x["particles"][1].update(
                            {
                                "particle_id": x["particles"][0][
                                    "particle_id"
                                ]
                            }
                        ),
                        root,
                        "duplicate",
                    ),
                },
                {
                    "test": "05_profile_support_overlap_rejected",
                    "pass": rejected(
                        state,
                        spec,
                        lambda x: x["particles"][1].update(
                            {"center_grid": x["particles"][0]["center_grid"]}
                        ),
                        root,
                        "overlap",
                    ),
                },
                {
                    "test": "06_periodic_image_overlap_rejected",
                    "pass": rejected(
                        state,
                        spec,
                        lambda x: (
                            x["particles"][0].update(
                                {"center_grid": [1, 20, 20]}
                            ),
                            x["particles"][1].update(
                                {"center_grid": [95, 20, 20]}
                            ),
                        ),
                        root,
                        "periodic_overlap",
                    ),
                },
                {
                    "test": "07_out_of_domain_center_rejected",
                    "pass": rejected(
                        state,
                        spec,
                        lambda x: x["particles"][0].update(
                            {"center_grid": [96, 0, 0]}
                        ),
                        root,
                        "out_of_domain",
                    ),
                },
                {
                    "test": "09_manifest_order_invariance",
                    "pass": all(
                        synthetic.digest(root / "first" / name)
                        == synthetic.digest(root / "reverse" / name)
                        for name in exact_files
                    ),
                },
                {
                    "test": "10_repeated_bytewise_determinism",
                    "pass": all(
                        synthetic.digest(root / "first" / name)
                        == synthetic.digest(root / "second" / name)
                        for name in exact_files
                    ),
                },
                {
                    "test": "11_particle_library_traceability",
                    "pass": all(
                        row["library_entry_sha256"]
                        and row["phi_profile_hash"]
                        and row["delta_C_relaxation_hash"]
                        for row in first["particle_library_mappings"]
                    ),
                },
                {
                    "test": "12_global_initial_inventory",
                    "pass": first["initial_canonical_inventory"][
                        "field_relative_error"
                    ]
                    <= 1.0e-14
                    and first["initial_canonical_inventory"][
                        "decomposition_relative_error"
                    ]
                    <= 1.0e-14,
                },
                {
                    "test": "13_zero_mode_provenance_declared",
                    "pass": first["initial_zero_mode_provenance"][
                        "zero_mode"
                    ]
                    == "PF_CONSERVED_Y_ZERO_MODE_V1",
                },
                {
                    "test": "17_gp_and_new_nucleation_off",
                    "pass": all(
                        first["physical_contract"][key] is False
                        for key in (
                            "GP_enabled",
                            "GP_birth_enabled",
                            "GP_release_enabled",
                            "external_source_enabled",
                            "new_beta_nucleation_enabled",
                        )
                    ),
                },
                {
                    "test": "18_no_common_equilibrium_optimizer",
                    "pass": first["assembly_contract"][
                        "optimizer_invoked"
                    ]
                    is False
                    and first[
                        "common_multi_particle_equilibrium_required"
                    ]
                    is False,
                },
            ]
        )

        # Non-finite source-field rejection is tested last because it
        # deliberately corrupts the synthetic library's raw field and then
        # re-pins every parent manifest in the test-only chain.
        profile_path = (
            state["library_path"].parent.parent
            / "profiles"
            / "R10p5"
            / "profile_manifest.json"
        )
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        phi_path = (
            profile_path.parent / profile["fields"]["phi"]["path"]
        ).resolve()
        phi = np.fromfile(phi_path, dtype="<f8")
        phi[0] = np.nan
        phi.astype("<f8").tofile(phi_path)
        profile["fields"]["phi"]["sha256"] = synthetic.digest(phi_path)
        write_json(profile_path, profile)
        library = json.loads(
            state["library_path"].read_text(encoding="utf-8")
        )
        for row in library["profiles"]:
            if float(row["target_radius_nm"]) == 10.5:
                row["profile_manifest_sha256"] = synthetic.digest(
                    profile_path
                )
        write_json(state["library_path"], library)
        selection = json.loads(
            state["selection_path"].read_text(encoding="utf-8")
        )
        selection["library_manifest_sha256"] = synthetic.digest(
            state["library_path"]
        )
        write_json(state["selection_path"], selection)
        target.FROZEN_LIBRARY_SHA256 = synthetic.digest(
            state["library_path"]
        )
        target.FROZEN_SELECTION_SHA256 = synthetic.digest(
            state["selection_path"]
        )
        nonfinite = conditional_spec(
            state, [10.5, 9.5, 8.0, 8.0]
        )
        checks.append(
            {
                "test": "08_non_finite_field_rejected",
                "pass": rejected(
                    state, nonfinite, lambda _x: None, root, "nonfinite"
                ),
            }
        )

    checks.sort(key=lambda row: row["test"])
    failed = [row for row in checks if not row["pass"]]
    print(
        json.dumps(
            {
                "status": "PASS" if not failed else "FAIL",
                "test_count": len(checks),
                "failed_count": len(failed),
                "checks": checks,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
