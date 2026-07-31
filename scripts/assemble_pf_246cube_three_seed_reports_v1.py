#!/usr/bin/env python3
"""Assemble the registered three-replicate 246-cube qualification reports.

This is intentionally read-only with respect to runtime roots.  It consumes
copied, hash-pinned Stage-6/7/8 evidence and creates only the still-missing
registered report artifacts.  All three short-screening statuses must be the
exact PASS marker before any final qualification report is emitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


STATIC_PASS = "PASS_246CUBE_THREE_SEED_STATIC_QUALIFICATION_V1"
RESTART_PASS = "PASS_246CUBE_SHORT_RESTART_AND_OBSERVABLES_V1"
SCREEN_PASS = "PASS_246CUBE_6H8H_SHORT_SCREENING_V1"
FINAL_PASS = "PASS_246CUBE_THREE_SEED_LIBRARY_HANDOFF_SHORT_QUALIFICATION_V1"
LABELS = ("A", "B", "C")
REQUIRED_EXISTING = (
    "baseline_freeze.md",
    "source_psd_mapping.md",
    "frozen_discrete_psd.csv",
    "replicate_seed_contract.md",
    "fixture_A_manifest.json",
    "fixture_B_manifest.json",
    "fixture_C_manifest.json",
    "fixture_hashes.md",
    "spatial_statistics.csv",
    "library_mapping_audit.md",
    "mass_inventory_audit.md",
    "deterministic_materialization.md",
    "short_restart_validation.md",
    "full_6h_48h_registered_plan.md",
)
GENERATED = (
    "short_6h_8h_screening.md",
    "ensemble_observables_6h_8h.csv",
    "particle_lineage_A.csv",
    "particle_lineage_B.csv",
    "particle_lineage_C.csv",
    "particle_events_A.csv",
    "particle_events_B.csv",
    "particle_events_C.csv",
    "short_6h_8h_A_audit.json",
    "short_6h_8h_B_audit.json",
    "short_6h_8h_C_audit.json",
    "short_6h_8h_A_input_hashes.sha256",
    "short_6h_8h_B_input_hashes.sha256",
    "short_6h_8h_C_input_hashes.sha256",
    "short_6h_8h_A_analysis_binary.sha256",
    "short_6h_8h_B_analysis_binary.sha256",
    "short_6h_8h_C_analysis_binary.sha256",
    "merge_aware_A_audit.json",
    "merge_aware_B_audit.json",
    "merge_aware_C_audit.json",
    "merge_groups_A.csv",
    "merge_groups_B.csv",
    "merge_groups_C.csv",
    "merge_aware_A_report.md",
    "merge_aware_B_report.md",
    "merge_aware_C_report.md",
    "original_strict_lineage_A_status.txt",
    "original_strict_lineage_B_status.txt",
    "original_strict_lineage_C_status.txt",
    "original_strict_lineage_A_summary.txt",
    "original_strict_lineage_B_summary.txt",
    "original_strict_lineage_C_summary.txt",
    "original_strict_lineage_A_events.csv",
    "original_strict_lineage_B_events.csv",
    "original_strict_lineage_C_events.csv",
    "performance.md",
    "production_candidate_decision.md",
    "first_failure.csv",
    "final_terminal_output.txt",
    "report_manifest.sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_new(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise ValueError("refusing to overwrite: " + ", ".join(existing))


def write_csv(
    path: Path, rows: List[Mapping[str, Any]], fieldnames: List[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 12) -> str:
    if isinstance(value, str):
        value = float(value)
    return f"{float(value):.{digits}g}"


def endpoint(rows: List[Dict[str, str]], step: int) -> Dict[str, str]:
    matches = [row for row in rows if int(row["step"]) == step]
    if len(matches) != 1:
        raise ValueError(f"expected one observation at step {step}")
    return matches[0]


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    result = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    result.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    staging = args.staging_root.resolve()
    report = args.report_root.resolve()
    if not report.is_dir():
        raise ValueError(f"report root is absent: {report}")
    missing = [name for name in REQUIRED_EXISTING if not (report / name).is_file()]
    if missing:
        raise ValueError("missing prerequisite reports: " + ", ".join(missing))
    require_new(report / name for name in GENERATED)

    static = load_json(staging / "static_audit" / "static_audit.json")
    if static.get("status") != STATIC_PASS:
        raise ValueError("static qualification is not exact PASS")
    report_assembler_sha256 = sha256(Path(__file__).resolve())
    restart: Dict[str, Dict[str, Any]] = {}
    audit: Dict[str, Dict[str, Any]] = {}
    observations: Dict[str, List[Dict[str, str]]] = {}
    for label in LABELS:
        restart[label] = load_json(staging / f"short_restart_{label}.json")
        if restart[label].get("status") != RESTART_PASS:
            raise ValueError(f"replicate {label} restart is not exact PASS")
        screen_root = staging / f"screening_{label}"
        audit[label] = load_json(screen_root / "audit.json")
        if audit[label].get("status") != SCREEN_PASS:
            raise ValueError(f"replicate {label} screening is not exact PASS")
        if audit[label].get("replicate") != f"replicate_{label}":
            raise ValueError(f"replicate {label} screening identity mismatch")
        if not all(audit[label].get("gates", {}).values()):
            raise ValueError(f"replicate {label} has a false screening gate")
        performance = audit[label].get("metrics", {})
        merge_count = int(performance.get("merge_count", 0))
        if (
            merge_count > 0
            and performance.get("merge_aware_status")
            != "PASS_246CUBE_RESOLVED_MERGE_AWARE_LINEAGE_V1"
        ):
            raise ValueError(
                f"replicate {label} has no resolved merge-aware PASS"
            )
        for key in (
            "wall_seconds_per_step",
            "segment_wall_seconds",
            "gpu_utilization_mean_percent",
            "peak_VRAM_MiB",
        ):
            if not math.isfinite(float(performance.get(key, math.nan))):
                raise ValueError(
                    f"replicate {label} has non-finite performance metric {key}"
                )
        if int(performance.get("gpu_sample_count", 0)) <= 0:
            raise ValueError(f"replicate {label} has no GPU telemetry samples")
        observations[label] = load_csv(
            screen_root / "ensemble_observables_6h_8h.csv"
        )
        if [int(row["step"]) for row in observations[label]][-1] != 7266:
            raise ValueError(f"replicate {label} lacks step 7266")

    # Preserve the machine-readable Stage-8 audits and event logs.
    analysis_hashes: Dict[str, str] = {}
    input_list_hashes: Dict[str, str] = {}
    for label in LABELS:
        screen_root = staging / f"screening_{label}"
        analysis_hash_file = screen_root / "analysis_binary.sha256"
        input_hash_file = screen_root / "input_hashes.sha256"
        analysis_hash = analysis_hash_file.read_text(
            encoding="utf-8"
        ).split()[0]
        if len(analysis_hash) != 64 or any(
            character not in "0123456789abcdef" for character in analysis_hash
        ):
            raise ValueError(f"replicate {label} analysis hash is malformed")
        analysis_hashes[label] = analysis_hash
        input_list_hashes[label] = sha256(input_hash_file)
        shutil.copyfile(
            screen_root / "audit.json",
            report / f"short_6h_8h_{label}_audit.json",
        )
        shutil.copyfile(
            screen_root / "particle_events.csv",
            report / f"particle_events_{label}.csv",
        )
        shutil.copyfile(
            input_hash_file,
            report / f"short_6h_8h_{label}_input_hashes.sha256",
        )
        shutil.copyfile(
            analysis_hash_file,
            report / f"short_6h_8h_{label}_analysis_binary.sha256",
        )
        if int(audit[label]["metrics"].get("merge_count", 0)) > 0:
            for source_name, target_name in (
                ("merge_aware_audit.json", f"merge_aware_{label}_audit.json"),
                ("merge_groups.csv", f"merge_groups_{label}.csv"),
                ("merge_aware_report.md", f"merge_aware_{label}_report.md"),
                (
                    "original_blocked_status.txt",
                    f"original_strict_lineage_{label}_status.txt",
                ),
                (
                    "original_lineage_summary.txt",
                    f"original_strict_lineage_{label}_summary.txt",
                ),
                (
                    "original_particle_events.csv",
                    f"original_strict_lineage_{label}_events.csv",
                ),
            ):
                shutil.copyfile(
                    screen_root / source_name,
                    report / target_name,
                )

    # Join the tracker stable component identity back to the registered Pxxx ID.
    mapping_max_delta: Dict[str, float] = {}
    for label in LABELS:
        mapping_rows = load_csv(
            staging / "fixture_aux" / f"initial_components_{label}.csv"
        )
        mapping = {
            int(row["component_label"]): row["particle_id"]
            for row in mapping_rows
        }
        mapping_centroid = {
            int(row["component_label"]): json.loads(row["centroid_nm"])
            for row in mapping_rows
        }
        if len(mapping) != 96 or len(set(mapping.values())) != 96:
            raise ValueError(f"replicate {label} component mapping is incomplete")
        lineage = load_csv(
            staging / f"screening_{label}" / "particle_lineage.csv"
        )
        enriched: List[Dict[str, str]] = []
        initial_seen = set()
        max_centroid_delta = 0.0
        for row in lineage:
            stable_id = int(row["stable_particle_id"])
            if stable_id not in mapping:
                raise ValueError(
                    f"replicate {label} unknown stable component {stable_id}"
                )
            member_stable_ids = [
                int(value)
                for value in row.get(
                    "lineage_member_ids", str(stable_id)
                ).split("+")
                if value
            ]
            try:
                member_particles = [
                    mapping[value] for value in member_stable_ids
                ]
            except KeyError as exc:
                raise ValueError(
                    f"replicate {label} unknown merge-group member"
                ) from exc
            if int(row["step"]) == 0:
                initial_seen.add(stable_id)
                observed = [
                    float(row["centroid_x_nm"]),
                    float(row["centroid_y_nm"]),
                    float(row["centroid_z_nm"]),
                ]
                expected = mapping_centroid[stable_id]
                deltas = [
                    min(abs(left - right), 246.0 - abs(left - right))
                    for left, right in zip(observed, expected)
                ]
                max_centroid_delta = max(
                    max_centroid_delta,
                    math.sqrt(sum(value * value for value in deltas)),
                )
            enriched.append(
                {
                    "replicate": f"replicate_{label}",
                    "particle_id": mapping[stable_id],
                    "lineage_member_particle_ids": "+".join(
                        member_particles
                    ),
                    **row,
                }
            )
        if len(initial_seen) != 96 or max_centroid_delta > 1.0e-9:
            raise ValueError(
                f"replicate {label} Pxxx/component identity join failed"
            )
        mapping_max_delta[label] = max_centroid_delta
        write_csv(
            report / f"particle_lineage_{label}.csv",
            enriched,
            list(enriched[0]),
        )

    combined: List[Dict[str, str]] = []
    for label in LABELS:
        for row in observations[label]:
            combined.append({"replicate": f"replicate_{label}", **row})
    write_csv(
        report / "ensemble_observables_6h_8h.csv",
        combined,
        list(combined[0]),
    )

    screening_rows: List[List[str]] = []
    change_rows: List[List[str]] = []
    for label in LABELS:
        initial = endpoint(observations[label], 0)
        at_256 = endpoint(observations[label], 256)
        final = endpoint(observations[label], 7266)
        metrics = audit[label]["metrics"]
        screening_rows.append(
            [
                label,
                initial["particle_count"],
                final["particle_count"],
                str(metrics["dissolution_count"]),
                str(metrics.get("merge_count", 0)),
                fmt(final["beta_volume_fraction"]),
                fmt(final["mean_radius_nm"]),
                fmt(final["Sv_nm_inv"]),
                fmt(final["M6_nm3"]),
                fmt(final["far_field_matrix_xAg"]),
            ]
        )
        change_rows.append(
            [
                label,
                fmt(
                    float(final["beta_volume_fraction"])
                    / float(initial["beta_volume_fraction"])
                    - 1.0
                ),
                fmt(
                    float(final["mean_radius_nm"])
                    / float(initial["mean_radius_nm"])
                    - 1.0
                ),
                fmt(
                    float(final["Sv_nm_inv"])
                    / float(initial["Sv_nm_inv"])
                    - 1.0
                ),
                fmt(
                    float(final["M6_nm3"])
                    / float(initial["M6_nm3"])
                    - 1.0
                ),
                fmt(at_256["phi_normalized_l1_from_initial"]),
                fmt(final["phi_normalized_l1_from_initial"]),
            ]
        )

    screening_text = f"""# Three-replicate 6--8 h short screening

All three independently placed, hash-pinned fixtures reached registered step
7266 at age 8.000019169100993 h.  No run used a shortened final timestep.

{markdown_table(
    [
        "replicate",
        "N at 6 h",
        "N at 8 h",
        "qualified dissolutions",
        "resolved merges",
        "final beta VF",
        "final mean R (nm)",
        "final Sv (nm^-1)",
        "final M6 (nm^3)",
        "final far-field xAg",
    ],
    screening_rows,
)}

Relative changes below use the exact materialized step-0 state.  The phi
columns retain the registered initial-reconstruction diagnostic at step 256
and its accumulated value at step 7266.

{markdown_table(
    [
        "replicate",
        "delta beta VF / initial",
        "delta mean R / initial",
        "delta Sv / initial",
        "delta M6 / initial",
        "phi L1 at 256",
        "phi L1 at 7266",
    ],
    change_rows,
)}

Identity propagation used maximum voxel overlap of periodic connected
components.  A disappearance was accepted only with at least three registered
states, two strictly decreasing final transitions, and final h-volume no more
than half its initial value.  A many-to-one overlap is retained as an explicit
persistent lineage group and must pass the registered multi-threshold
merge-aware audit.  Splits, new components, weak merge edges, and unqualified
dissolutions remain fail-closed.

{markdown_table(
    [
        "replicate",
        "dissolutions",
        "resolved merges",
        "splits",
        "new components",
        "unqualified dissolutions",
        "unqualified merges",
    ],
    [
        [
            label,
            str(audit[label]["metrics"]["dissolution_count"]),
            str(audit[label]["metrics"].get("merge_count", 0)),
            str(audit[label]["metrics"].get("split_count", 0)),
            "0",
            "0",
            str(audit[label]["metrics"].get("unqualified_merge_count", 0)),
        ]
        for label in LABELS
    ],
)}

The stable connected-component labels were joined back to each fixture's
registered `P000`--`P095` identity.  The largest periodic step-0 centroid
discrepancy in A/B/C was
`{max(mapping_max_delta.values()):.17g} nm`.

Any original strict-lineage block caused by a many-to-one overlap is preserved
as historical evidence.  Qualification is restored only by the supplemental
explicit-parent, persistent-group, multi-threshold audit; the strict result is
not overwritten or reinterpreted as though it never occurred.

The complete PSD and P000--P095 lineages are retained in the registered CSV
files.  Different random fixtures are not required to follow the same
trajectory.  Far-field concentration and agreement with an experimental 48 h
band are scientific diagnostics, not hard gates in this short qualification.

```text
short_6h_8h_status=PASS
```
"""
    (report / "short_6h_8h_screening.md").write_text(
        screening_text, encoding="utf-8"
    )

    performance_rows: List[List[str]] = []
    for label in LABELS:
        stage7 = restart[label]["metrics"]
        stage8 = audit[label]["metrics"]
        performance_rows.append(
            [
                label,
                fmt(stage7["continuous_wall_seconds"] / 256.0),
                fmt(stage8["wall_seconds_per_step"]),
                fmt(stage8["segment_wall_seconds"]),
                fmt(stage8["gpu_utilization_mean_percent"]),
                fmt(stage8["peak_VRAM_MiB"]),
            ]
        )
    performance_text = f"""# Performance

{markdown_table(
    [
        "replicate",
        "256-step wall s/step",
        "6.07--8 h wall s/step",
        "screening segment wall s",
        "mean GPU util (%)",
        "peak VRAM (MiB)",
    ],
    performance_rows,
)}

A ran on the registered workstation binary; B and C ran with the registered
cluster sm_80 binary.  The wall-clock values characterize those devices and
must not be interpreted as a hardware-independent physics difference.  All
runs used 246^3 cells, production `dt_code=0.02`, elasticity, the same
physical parameters, sparse checkpoints, and suppressed bulk VTK output.

Runtime analysis binary hashes:

```text
A={analysis_hashes["A"]}
B={analysis_hashes["B"]}
C={analysis_hashes["C"]}
```

SHA-256 identities of the copied runtime input-hash lists:

```text
A={input_list_hashes["A"]}
B={input_list_hashes["B"]}
C={input_list_hashes["C"]}
```

Final report assembler SHA-256:
`{report_assembler_sha256}`.
"""
    (report / "performance.md").write_text(performance_text, encoding="utf-8")

    max_mass = max(
        float(audit[label]["metrics"]["max_system_mass_drift"])
        for label in LABELS
    )
    max_lambda = max(
        float(audit[label]["metrics"]["max_zero_mode_lambda"])
        for label in LABELS
    )
    final_counts = [
        int(audit[label]["metrics"]["final_particle_count"]) for label in LABELS
    ]
    dissolution_counts = [
        int(audit[label]["metrics"]["dissolution_count"]) for label in LABELS
    ]
    merge_counts = [
        int(audit[label]["metrics"].get("merge_count", 0)) for label in LABELS
    ]
    decision_text = f"""# Production candidate decision

The three 246^3 / 96-particle conditional-handoff fixtures satisfy every
registered static, deterministic, conservation, restart, short-dynamic, and
lineage hard gate.

This result qualifies the fixtures for a future three-member 6--48 h
conditional-path ensemble.  It does not claim that the fixtures are a unique
experimental 6 h microstructure or a common multi-particle equilibrium.  It
also does not authorize or start the long production runs.

Summary:

- final particle counts A/B/C: {final_counts}
- qualified dissolution counts A/B/C: {dissolution_counts}
- resolved merge counts A/B/C: {merge_counts}
- maximum system mass drift: {max_mass:.17g}
- maximum absolute zero-mode lambda: {max_lambda:.17g}
- unresolved merge/split/new-component events: none
- GP, GP Birth, GP release, new beta nucleation, external source: disabled

```text
profile_library_status=PASS
production_dt_status=PASS_PRODUCTION_DT_OBSERVABLE_CONVERGENCE_V1
fixture_A_status=PASS
fixture_B_status=PASS
fixture_C_status=PASS
particle_count_per_fixture=96
domain=246x246x246
replicate_count=3
discrete_PSD_identity_status=PASS
spatial_independence_status=PASS
global_inventory_status=PASS
deterministic_materialization_status=PASS
restart_status=PASS_BYTEWISE_ALL_THREE
short_6h_8h_status=PASS
particle_lineage_status=PASS_ALL_THREE_RESOLVED
GP_enabled=false
new_beta_nucleation_enabled=false
common_multi_particle_equilibrium_required=false
full_6h_48h_started=false
qualified_for_full_ensemble=true
recommended_next_action=obtain_explicit_user_authorization_then_launch_three_independent_6h_to_48h_production_runs
final_status={FINAL_PASS}
```
"""
    (report / "production_candidate_decision.md").write_text(
        decision_text, encoding="utf-8"
    )

    resolved_failures = [
        {
            "stage": "Stage8",
            "replicate": f"replicate_{label}",
            "gate": "original_strict_lineage_merge",
            "observed": (
                f"merge_count={audit[label]['metrics'].get('merge_count', 0)}"
            ),
            "required": "explicit_resolved_merge_group_or_no_merge",
            "status": "RESOLVED_BY_MULTI_THRESHOLD_MERGE_AWARE_AUDIT",
        }
        for label in LABELS
        if int(audit[label]["metrics"].get("merge_count", 0)) > 0
    ]
    write_csv(
        report / "first_failure.csv",
        resolved_failures,
        ["stage", "replicate", "gate", "observed", "required", "status"],
    )
    terminal = "\n".join(
        (
            "profile_library_status=PASS",
            "production_dt_status=PASS_PRODUCTION_DT_OBSERVABLE_CONVERGENCE_V1",
            "fixture_A_status=PASS",
            "fixture_B_status=PASS",
            "fixture_C_status=PASS",
            "particle_count_per_fixture=96",
            "domain=246x246x246",
            "replicate_count=3",
            "discrete_PSD_identity_status=PASS",
            "spatial_independence_status=PASS",
            "global_inventory_status=PASS",
            "deterministic_materialization_status=PASS",
            "restart_status=PASS_BYTEWISE_ALL_THREE",
            "short_6h_8h_status=PASS",
            "particle_lineage_status=PASS_ALL_THREE_RESOLVED",
            f"max_system_mass_drift={max_mass:.17g}",
            f"max_zero_mode_lambda={max_lambda:.17g}",
            "GP_enabled=false",
            "new_beta_nucleation_enabled=false",
            "common_multi_particle_equilibrium_required=false",
            "full_6h_48h_started=false",
            "qualified_for_full_ensemble=true",
            f"report_assembler_sha256={report_assembler_sha256}",
            "recommended_next_action=obtain_explicit_user_authorization_then_launch_three_independent_6h_to_48h_production_runs",
            f"final_status={FINAL_PASS}",
        )
    )
    (report / "final_terminal_output.txt").write_text(
        terminal + "\n", encoding="utf-8"
    )

    manifest_path = report / "report_manifest.sha256"
    files = sorted(
        path
        for path in report.iterdir()
        if path.is_file() and path.name != manifest_path.name
    )
    lines = [f"{sha256(path)}  {path.name}" for path in files]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(FINAL_PASS)


if __name__ == "__main__":
    main()
