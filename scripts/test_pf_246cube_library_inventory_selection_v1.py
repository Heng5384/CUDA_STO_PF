#!/usr/bin/env python3
"""Static regression for the hash-pinned 246-cube inventory selection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import materialize_pf_246cube_library_handoff_v1 as materializer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--inventory-selection", type=Path, required=True)
    args = parser.parse_args()
    _manifest, particles, audit = materializer.load_historical(
        args.historical_manifest, materializer.DEFAULT_REGISTERED_RADII_NM
    )
    selected, selected_audit, identity = materializer.apply_inventory_selection(
        particles,
        audit,
        args.inventory_selection,
        materializer.DEFAULT_REGISTERED_RADII_NM,
        materializer.DEFAULT_LIBRARY_SHA256,
        materializer.DEFAULT_SELECTION_SHA256,
    )
    if identity is None or len(selected) != 96:
        raise SystemExit("[fatal] inventory selection was not applied")
    if sum(selected_audit["histogram"].values()) != 96:
        raise SystemExit("[fatal] selected histogram does not contain 96 particles")
    if len(selected_audit["inventory_selection_transitions"]) != 5:
        raise SystemExit("[fatal] transition audit is incomplete")
    expected = float(selected_audit["target_effective_h_volume_nm3"])
    actual = float(selected_audit["selected_effective_h_volume_nm3"])
    tolerance = json.loads(
        args.inventory_selection.read_text(encoding="utf-8")
    )["effective_h_volume_tolerance_nm3"]
    if not math.isfinite(actual) or abs(actual - expected) > tolerance:
        raise SystemExit("[fatal] selected effective h-volume misses target")
    if any(
        row["registered_radius_nm"]
        not in materializer.DEFAULT_REGISTERED_RADII_NM
        for row in selected
    ):
        raise SystemExit("[fatal] unregistered radius selected")
    print(
        json.dumps(
            {
                "status": "PASS_PF_246CUBE_LIBRARY_INVENTORY_SELECTION_V1",
                "particle_count": len(selected),
                "selected_histogram": selected_audit["histogram"],
                "selected_effective_h_volume_nm3": actual,
                "target_effective_h_volume_nm3": expected,
                "effective_h_volume_error_nm3": actual - expected,
                "inventory_selection_sha256": identity["sha256"],
                "profile_scaling_used": False,
                "interpolation_used": False
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
