#!/usr/bin/env python3
"""Fail-closed merge/dissolution audit for hour-cadence PF checkpoints.

The underlying C++ tracker deliberately requires three recorded decreases to
call a dissolution.  That criterion is appropriate for dense output, but it
cannot classify a particle that becomes unresolved between two one-hour
production checkpoints.  This audit does not invent an exact event time.  It
qualifies only a bounded extinction interval when three nested h-threshold
trackers agree, topology is closed, and the full-field inventory remains
closed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Sequence, Set, Tuple


PASS = "PASS_246CUBE_HOURLY_MERGE_DISSOLUTION_AUDIT_V1"
FAIL = "BLOCKED_246CUBE_HOURLY_MERGE_DISSOLUTION_AUDIT_V1"
ALLOWED_TRACKER_STATUS = {
    "PASS_PERIODIC_OVERLAP_PARTICLE_LINEAGE_V1",
    "BLOCKED_PERIODIC_OVERLAP_PARTICLE_LINEAGE_V1",
}
THRESHOLDS = (
    ("low", 1.0e-4),
    ("medium", 1.0e-3),
    ("strong", 5.0e-3),
)


class AuditFailure(ValueError):
    """A deterministic fail-closed gate failure."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_key_values(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = raw.partition("=")
        if separator:
            result[key] = value
    return result


def periodic_distance(
    left: Sequence[float], right: Sequence[float], domain_nm: float
) -> float:
    squared = 0.0
    for a, b in zip(left, right):
        delta = abs(a - b)
        delta = min(delta, domain_nm - delta)
        squared += delta * delta
    return math.sqrt(squared)


def split_ids(value: str) -> Tuple[int, ...]:
    result = tuple(int(item) for item in value.split("+") if item)
    if not result:
        raise AuditFailure("empty lineage member set")
    if len(set(result)) != len(result):
        raise AuditFailure("duplicated lineage member id")
    return result


def member_label(members: FrozenSet[str]) -> str:
    return "+".join(sorted(members))


def parse_expected_steps(path: Path) -> List[int]:
    values: List[int] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped:
            values.append(int(stripped.split()[0]))
    result = [0] + values
    if result != sorted(set(result)) or len(result) < 2:
        raise AuditFailure("expected checkpoint steps are not strict and unique")
    return result


def initial_identity_map(
    lineage: Sequence[Mapping[str, str]],
    registered: Sequence[Mapping[str, str]],
    expected_count: int,
    domain_nm: float,
    tolerance_nm: float,
) -> Tuple[Dict[int, str], float]:
    initial = [row for row in lineage if int(row["step"]) == 0]
    if len(initial) != expected_count or len(registered) != expected_count:
        raise AuditFailure("initial identity population mismatch")
    candidates: List[Tuple[float, int, str]] = []
    for row in initial:
        stable = int(row["stable_particle_id"])
        observed = [
            float(row["centroid_x_nm"]),
            float(row["centroid_y_nm"]),
            float(row["centroid_z_nm"]),
        ]
        for target in registered:
            expected = json.loads(target["centroid_nm"])
            candidates.append(
                (
                    periodic_distance(observed, expected, domain_nm),
                    stable,
                    target["particle_id"],
                )
            )
    mapping: Dict[int, str] = {}
    used: Set[str] = set()
    maximum = 0.0
    for distance, stable, particle in sorted(candidates):
        if stable in mapping or particle in used:
            continue
        mapping[stable] = particle
        used.add(particle)
        maximum = max(maximum, distance)
    if len(mapping) != expected_count or len(used) != expected_count:
        raise AuditFailure("initial identity assignment is not bijective")
    if maximum > tolerance_nm:
        raise AuditFailure(
            f"initial identity delta {maximum} exceeds {tolerance_nm} nm"
        )
    return mapping, maximum


@dataclass(frozen=True)
class Component:
    step: int
    stable_id: int
    component_label: int
    members: FrozenSet[str]
    raw_members: FrozenSet[str]
    h_volume_nm3: float
    equivalent_radius_nm: float
    centroid_nm: Tuple[float, float, float]


@dataclass
class ThresholdData:
    name: str
    threshold: float
    root: Path
    summary: Dict[str, str]
    raw_lineage: List[Dict[str, str]]
    events: List[Dict[str, str]]
    observations: List[Dict[str, str]]
    identity_map: Dict[int, str]
    identity_max_delta_nm: float
    components_by_step: Dict[int, List[Component]]
    alive_by_step: Dict[int, FrozenSet[str]]
    dissolution_groups: List[Dict[str, Any]]
    merge_groups: List[Dict[str, Any]]
    resolved_demerges: List[Dict[str, Any]]
    incidental_overlap_events: List[Dict[str, Any]]


def assignment_to_anchors(
    components: Sequence[Component],
    members: FrozenSet[str],
    anchors: Mapping[str, Sequence[float]],
    domain_nm: float,
) -> Tuple[Dict[int, str], float, float]:
    """Resolve a demerged raw lineage family without volume sorting.

    The old tracker duplicates the merged stable id onto every child after a
    threshold neck opens.  We recover identity only when periodic centroid
    geometry gives a unique bijection to the last separately observed anchors.
    This is deliberately fail-closed: ambiguous assignments are not guessed.
    """
    ordered_components = sorted(components, key=lambda item: item.component_label)
    ordered_members = sorted(members)
    if len(ordered_components) != len(ordered_members):
        raise AuditFailure("demerge child count does not equal lineage member count")
    candidates: List[Tuple[float, float, Tuple[str, ...]]] = []
    for permutation in itertools.permutations(ordered_members):
        distances = tuple(
            periodic_distance(component.centroid_nm, anchors[member], domain_nm)
            for component, member in zip(ordered_components, permutation)
        )
        candidates.append((sum(distances), max(distances, default=0.0), permutation))
    candidates.sort(key=lambda item: (item[0], item[2]))
    best_total, best_maximum, best = candidates[0]
    second_total = candidates[1][0] if len(candidates) > 1 else math.inf
    scale = max(best_total, second_total if math.isfinite(second_total) else 0.0, 1.0)
    margin = second_total - best_total
    if math.isfinite(second_total) and margin <= 1.0e-9 * scale:
        raise AuditFailure("demerge centroid assignment is not unique")
    if len(ordered_members) > 1:
        minimum_anchor_separation = min(
            periodic_distance(anchors[left], anchors[right], domain_nm)
            for index, left in enumerate(ordered_members)
            for right in ordered_members[index + 1 :]
        )
        if minimum_anchor_separation <= 0.0:
            raise AuditFailure("demerge identity anchors are coincident")
        if best_maximum >= 0.5 * minimum_anchor_separation:
            raise AuditFailure("demerge child crossed the anchor Voronoi gate")
    return (
        {
            component.component_label: member
            for component, member in zip(ordered_components, best)
        },
        best_maximum,
        margin,
    )


def nearest_surviving_member(
    component: Component,
    members: FrozenSet[str],
    anchors: Mapping[str, Sequence[float]],
    domain_nm: float,
) -> Tuple[str, float, float]:
    distances = sorted(
        (
            periodic_distance(component.centroid_nm, anchors[member], domain_nm),
            member,
        )
        for member in members
    )
    if not distances:
        raise AuditFailure("no candidate identity remains for demerged family")
    best_distance, best_member = distances[0]
    second_distance = distances[1][0] if len(distances) > 1 else math.inf
    margin = second_distance - best_distance
    scale = max(best_distance, second_distance if math.isfinite(second_distance) else 0.0, 1.0)
    if math.isfinite(second_distance) and margin <= 1.0e-9 * scale:
        raise AuditFailure("surviving demerge identity is spatially ambiguous")
    return best_member, best_distance, margin


def parse_overlap_detail(detail: str) -> Tuple[List[float], float]:
    marker = "parent_id:overlap_voxels:parent_fraction:child_fraction="
    if not detail.startswith(marker):
        raise AuditFailure("merge overlap detail has the wrong schema")
    parent_fractions: List[float] = []
    child_sum = 0.0
    for item in detail[len(marker) :].split("+"):
        fields = item.split(":")
        if len(fields) != 4:
            raise AuditFailure("malformed merge overlap tuple")
        int(fields[0])
        int(fields[1])
        parent_fraction = float(fields[2])
        child_fraction = float(fields[3])
        if not all(math.isfinite(value) for value in (parent_fraction, child_fraction)):
            raise AuditFailure("non-finite merge overlap fraction")
        parent_fractions.append(parent_fraction)
        child_sum += child_fraction
    return parent_fractions, child_sum


def load_threshold(
    name: str,
    threshold: float,
    root: Path,
    registered: Sequence[Mapping[str, str]],
    expected_steps: Sequence[int],
    expected_count: int,
    domain_nm: float,
    identity_tolerance_nm: float,
    mass_tolerance: float,
) -> ThresholdData:
    required = (
        "lineage_summary.txt",
        "particle_lineage.csv",
        "particle_events.csv",
        "ensemble_observables.csv",
        "status.txt",
    )
    for filename in required:
        if not (root / filename).is_file():
            raise AuditFailure(f"missing {name} tracker file: {filename}")
    summary = read_key_values(root / "lineage_summary.txt")
    if summary.get("status") not in ALLOWED_TRACKER_STATUS:
        raise AuditFailure(f"unexpected {name} tracker status")
    lineage = read_rows(root / "particle_lineage.csv")
    events = read_rows(root / "particle_events.csv")
    observations = read_rows(root / "ensemble_observables.csv")
    observed_steps = [int(row["step"]) for row in observations]
    if observed_steps != list(expected_steps):
        raise AuditFailure(f"{name} snapshot sequence mismatch")
    identity_map, identity_delta = initial_identity_map(
        lineage,
        registered,
        expected_count,
        domain_nm,
        identity_tolerance_nm,
    )
    registered_centroids = {
        row["particle_id"]: tuple(float(value) for value in json.loads(row["centroid_nm"]))
        for row in registered
    }
    events_by_step: Dict[int, List[Tuple[int, Dict[str, str]]]] = {}
    for event_index, event in enumerate(events):
        step = int(event["step"])
        if step not in expected_steps or step == 0:
            raise AuditFailure("event step is outside the snapshot contract")
        if event["event_type"] not in {
            "merge",
            "split",
            "dissolution",
            "new_component_without_overlap",
        }:
            raise AuditFailure("unknown lineage event type")
        if event["event_type"] == "new_component_without_overlap":
            raise AuditFailure(f"{name} contains new_component_without_overlap")
        events_by_step.setdefault(step, []).append((event_index, event))

    raw_components_by_step: Dict[int, List[Component]] = {
        step: [] for step in expected_steps
    }
    for row in lineage:
        step = int(row["step"])
        if step not in raw_components_by_step:
            raise AuditFailure(f"{name} lineage contains an unregistered step")
        stable_id = int(row["stable_particle_id"])
        raw_members = split_ids(row["lineage_member_ids"])
        try:
            members = frozenset(identity_map[item] for item in raw_members)
        except KeyError as exc:
            raise AuditFailure("lineage member is not an initial identity") from exc
        if int(row["lineage_group_size"]) != len(members):
            raise AuditFailure("lineage group size does not match member set")
        component = Component(
            step=step,
            stable_id=stable_id,
            component_label=int(row["component_label"]),
            members=members,
            raw_members=members,
            h_volume_nm3=float(row["h_volume_nm3"]),
            equivalent_radius_nm=float(row["equivalent_radius_nm"]),
            centroid_nm=(
                float(row["centroid_x_nm"]),
                float(row["centroid_y_nm"]),
                float(row["centroid_z_nm"]),
            ),
        )
        if (
            not math.isfinite(component.h_volume_nm3)
            or not math.isfinite(component.equivalent_radius_nm)
            or component.h_volume_nm3 <= 0.0
            or component.equivalent_radius_nm <= 0.0
        ):
            raise AuditFailure("invalid component volume or radius")
        raw_components_by_step[step].append(component)

    components_by_step: Dict[int, List[Component]] = {
        step: [] for step in expected_steps
    }
    anchors: Dict[str, Tuple[float, float, float]] = dict(registered_centroids)
    demerged_families: Set[FrozenSet[str]] = set()
    resolved_demerges: List[Dict[str, Any]] = []
    previously_alive: Set[str] = set(registered_centroids)
    for step in expected_steps:
        raw_families: Dict[FrozenSet[str], List[Component]] = {}
        for component in raw_components_by_step[step]:
            raw_families.setdefault(component.raw_members, []).append(component)
        family_members_seen: Set[str] = set()
        for family, family_components in sorted(
            raw_families.items(), key=lambda item: member_label(item[0])
        ):
            if family_members_seen.intersection(family):
                raise AuditFailure("overlapping raw lineage families at one step")
            family_members_seen.update(family)
            if len(family_components) > 1:
                if len(family_components) != len(family):
                    raise AuditFailure("duplicate lineage family has unresolved child count")
                assignment, maximum_delta, assignment_margin = assignment_to_anchors(
                    family_components, family, anchors, domain_nm
                )
                demerged_families.add(family)
                resolved_demerges.append(
                    {
                        "threshold_name": name,
                        "h_threshold": threshold,
                        "step": step,
                        "member_particle_ids": member_label(family),
                        "member_count": len(family),
                        "child_count": len(family_components),
                        "maximum_anchor_delta_nm": maximum_delta,
                        "assignment_margin_nm": assignment_margin,
                    }
                )
                for raw_component in family_components:
                    member = assignment[raw_component.component_label]
                    repaired = Component(
                        **{
                            **raw_component.__dict__,
                            "members": frozenset({member}),
                        }
                    )
                    components_by_step[step].append(repaired)
                    anchors[member] = repaired.centroid_nm
                continue
            raw_component = family_components[0]
            live_family = frozenset(family.intersection(previously_alive))
            if not live_family:
                raise AuditFailure("extinct lineage family reappeared")
            qualified_merge_here = any(
                event["event_type"] == "merge"
                and int(event["qualified_merge"]) == 1
                and int(event["child_stable_id"]) == raw_component.stable_id
                for _event_index, event in events_by_step.get(step, [])
            )
            if family in demerged_families and not qualified_merge_here:
                member, _distance, _margin = nearest_surviving_member(
                    raw_component, live_family, anchors, domain_nm
                )
                repaired_members = frozenset({member})
            else:
                repaired_members = live_family
            repaired = Component(
                **{**raw_component.__dict__, "members": repaired_members}
            )
            components_by_step[step].append(repaired)
            if len(repaired_members) == 1:
                anchors[next(iter(repaired_members))] = repaired.centroid_nm
        current_alive = {
            member
            for component in components_by_step[step]
            for member in component.members
        }
        if step != expected_steps[0] and not current_alive.issubset(previously_alive):
            raise AuditFailure("an extinct identity reappeared")
        previously_alive = current_alive

    alive_by_step: Dict[int, FrozenSet[str]] = {}
    for row, step in zip(observations, expected_steps):
        seen: Set[str] = set()
        for component in components_by_step[step]:
            if seen.intersection(component.members):
                raise AuditFailure("one identity belongs to two components")
            seen.update(component.members)
        alive_by_step[step] = frozenset(seen)
        if int(row["particle_count"]) != len(components_by_step[step]):
            raise AuditFailure("observable particle count does not close")
        finite = row.get("finite") == "1"
        bounds = row.get("bounds") == "1"
        mass_error = float(row["mass_relative_error"])
        if not finite or not bounds or not math.isfinite(mass_error):
            raise AuditFailure("non-finite or out-of-bounds field state")
        if abs(mass_error) > mass_tolerance:
            raise AuditFailure("canonical inventory drift exceeds tolerance")
        global_h = float(row["beta_volume_fraction"]) * domain_nm**3
        connected_h = sum(
            component.h_volume_nm3 for component in components_by_step[step]
        )
        residual_h = global_h - connected_h
        rounding = max(1.0e-8 * max(global_h, 1.0), 1.0e-6)
        if residual_h < -rounding:
            raise AuditFailure("connected h-volume exceeds full-field h-volume")
        if residual_h > threshold * domain_nm**3 + rounding:
            raise AuditFailure("unresolved h-volume exceeds threshold bound")

    if len(components_by_step[0]) != expected_count:
        raise AuditFailure(f"{name} initial component count mismatch")
    if int(summary.get("initial_particle_count", -1)) != expected_count:
        raise AuditFailure(f"{name} summary initial count mismatch")
    if int(summary.get("snapshot_count", -1)) != len(expected_steps):
        raise AuditFailure(f"{name} summary snapshot count mismatch")
    if int(summary.get("final_particle_count", -1)) != len(
        components_by_step[expected_steps[-1]]
    ):
        raise AuditFailure(f"{name} summary final count mismatch")

    dissolutions: List[Dict[str, Any]] = []
    merges: List[Dict[str, Any]] = []
    classified_event_indices: Set[int] = set()
    resolved_demerge_keys = {
        (int(row["step"]), frozenset(row["member_particle_ids"].split("+")))
        for row in resolved_demerges
    }
    for index in range(1, len(expected_steps)):
        previous_step = expected_steps[index - 1]
        step = expected_steps[index]
        previous = components_by_step[previous_step]
        current = components_by_step[step]
        previous_by_member = {
            member: component for component in previous for member in component.members
        }
        current_by_member = {
            member: component for component in current for member in component.members
        }
        for component in current:
            parents = {
                previous_by_member[member]
                for member in component.members
                if member in previous_by_member
            }
            if not parents:
                raise AuditFailure("component appears without a parent identity")
            parent_union = frozenset(
                member for parent in parents for member in parent.members
            )
            if parent_union != component.members:
                demerge_parent = next(
                    (
                        parent
                        for parent in parents
                        if component.members.issubset(parent.members)
                        and (step, parent.members) in resolved_demerge_keys
                    ),
                    None,
                )
                if demerge_parent is None:
                    raise AuditFailure("lineage membership changed without closed topology")
                continue
            if len(parents) > 1:
                candidates = [
                    (event_index, event)
                    for event_index, event in events_by_step.get(step, [])
                    if event["event_type"] == "merge"
                    and int(event["child_stable_id"]) == component.stable_id
                ]
                if len(candidates) != 1:
                    raise AuditFailure("merge transition lacks one exact event")
                event_index, event = candidates[0]
                if int(event["qualified_merge"]) != 1:
                    raise AuditFailure("merge overlap is not qualified")
                fractions, child_sum = parse_overlap_detail(event["detail"])
                if any(value < 0.5 for value in fractions) or child_sum < 0.5:
                    raise AuditFailure("merge overlap fractions fail hard gates")
                classified_event_indices.add(event_index)
                merges.append(
                    {
                        "threshold_name": name,
                        "h_threshold": threshold,
                        "step": step,
                        "previous_step": previous_step,
                        "members": component.members,
                        "stable_particle_ids": member_label(component.members),
                        "child_stable_id": component.stable_id,
                        "overlap_detail": event["detail"],
                    }
                )
        for component in previous:
            children = {
                current_by_member[member]
                for member in component.members
                if member in current_by_member
            }
            if len(children) > 1:
                if (step, component.members) not in resolved_demerge_keys:
                    raise AuditFailure("one lineage group split without resolved demerge")
                candidates = [
                    (event_index, event)
                    for event_index, event in events_by_step.get(step, [])
                    if event["event_type"] == "split"
                    and int(event["stable_particle_id"]) == component.stable_id
                ]
                if len(candidates) != 1:
                    raise AuditFailure("resolved demerge lacks one split provenance row")
                classified_event_indices.add(candidates[0][0])
                continue
            if children:
                continue
            candidates = [
                (event_index, event)
                for event_index, event in events_by_step.get(step, [])
                if event["event_type"] == "dissolution"
                and int(event["stable_particle_id"]) == component.stable_id
            ]
            if len(candidates) != 1:
                raise AuditFailure("extinction transition lacks one exact event")
            event_index, event = candidates[0]
            classified_event_indices.add(event_index)
            dissolutions.append(
                {
                    "threshold_name": name,
                    "h_threshold": threshold,
                    "previous_step": previous_step,
                    "step": step,
                    "members": component.members,
                    "stable_particle_ids": member_label(component.members),
                    "last_h_volume_nm3": component.h_volume_nm3,
                    "last_equivalent_radius_nm": component.equivalent_radius_nm,
                    "legacy_temporal_qualified": int(
                        event["qualified_dissolution"]
                    ),
                }
            )

    incidental_overlap_events: List[Dict[str, Any]] = []
    for step, indexed_events in events_by_step.items():
        remaining = [
            (event_index, event)
            for event_index, event in indexed_events
            if event_index not in classified_event_indices
        ]
        used_remaining: Set[int] = set()
        for event_index, event in remaining:
            if event_index in used_remaining:
                continue
            if event["event_type"] != "merge" or int(event["qualified_merge"]) != 0:
                raise AuditFailure("tracker event ledger has an unclassified topology row")
            fractions, child_sum = parse_overlap_detail(event["detail"])
            if all(value >= 0.5 for value in fractions) and child_sum >= 0.5:
                raise AuditFailure("unqualified merge has physically dominant overlap")
            matching_splits = [
                (other_index, other)
                for other_index, other in remaining
                if other_index not in used_remaining
                and other["event_type"] == "split"
                and int(other["stable_particle_id"]) == int(event["stable_particle_id"])
            ]
            if len(matching_splits) != 1:
                raise AuditFailure("weak overlap merge lacks one paired split row")
            split_index, _split = matching_splits[0]
            used_remaining.update({event_index, split_index})
            classified_event_indices.update({event_index, split_index})
            incidental_overlap_events.append(
                {
                    "threshold_name": name,
                    "h_threshold": threshold,
                    "step": step,
                    "stable_particle_id": int(event["stable_particle_id"]),
                    "overlap_detail": event["detail"],
                    "classification": "INCIDENTAL_WEAK_OVERLAP_EDGE_PRUNED",
                }
            )
    if len(classified_event_indices) != len(events):
        raise AuditFailure("tracker event ledger classification is incomplete")

    for index in range(1, len(expected_steps)):
        before = alive_by_step[expected_steps[index - 1]]
        after = alive_by_step[expected_steps[index]]
        if not after.issubset(before):
            raise AuditFailure("an extinct identity reappeared")

    if int(summary.get("new_component_count", -1)) != 0:
        raise AuditFailure(f"{name} summary reports a new component")
    if int(summary.get("merge_count", -1)) != sum(
        event["event_type"] == "merge" for event in events
    ):
        raise AuditFailure(f"{name} raw merge count does not close")
    if int(summary.get("split_count", -1)) != sum(
        event["event_type"] == "split" for event in events
    ):
        raise AuditFailure(f"{name} raw split count does not close")
    if int(summary.get("dissolution_count", -1)) != sum(
        event["event_type"] == "dissolution" for event in events
    ):
        raise AuditFailure(f"{name} raw dissolution count does not close")

    return ThresholdData(
        name=name,
        threshold=threshold,
        root=root,
        summary=summary,
        raw_lineage=lineage,
        events=events,
        observations=observations,
        identity_map=identity_map,
        identity_max_delta_nm=identity_delta,
        components_by_step=components_by_step,
        alive_by_step=alive_by_step,
        dissolution_groups=dissolutions,
        merge_groups=merges,
        resolved_demerges=resolved_demerges,
        incidental_overlap_events=incidental_overlap_events,
    )


def first_absent_steps(
    data: ThresholdData, identities: Iterable[str], steps: Sequence[int]
) -> Dict[str, int | None]:
    result: Dict[str, int | None] = {}
    for identity in identities:
        result[identity] = next(
            (step for step in steps[1:] if identity not in data.alive_by_step[step]),
            None,
        )
    return result


def classify_merges(
    threshold_data: Mapping[str, ThresholdData],
) -> List[Dict[str, Any]]:
    rows = [
        dict(group)
        for name, _threshold in THRESHOLDS
        for group in threshold_data[name].merge_groups
    ]
    member_sets = [row["members"] for row in rows]
    for left_index, left in enumerate(member_sets):
        for right in member_sets[left_index + 1 :]:
            if left.intersection(right) and not (
                left.issubset(right) or right.issubset(left)
            ):
                raise AuditFailure("cross-threshold merge groups are not laminar")
    by_members: Dict[FrozenSet[str], List[Dict[str, Any]]] = {}
    for row in rows:
        by_members.setdefault(row["members"], []).append(row)
    order = {"low": 0, "medium": 1, "strong": 2}
    output: List[Dict[str, Any]] = []
    for members, group_rows in sorted(
        by_members.items(), key=lambda item: member_label(item[0])
    ):
        seen = {row["threshold_name"]: int(row["step"]) for row in group_rows}
        available = [name for name, _value in THRESHOLDS if name in seen]
        ordered_steps = [seen[name] for name in available]
        if ordered_steps != sorted(ordered_steps):
            raise AuditFailure("merge threshold ordering is inverted")
        strongest = max(available, key=lambda name: order[name])
        demerge_steps: Dict[str, int] = {}
        for name, _threshold in THRESHOLDS:
            candidates = [
                int(row["step"])
                for row in threshold_data[name].resolved_demerges
                if frozenset(row["member_particle_ids"].split("+")) == members
            ]
            if candidates:
                demerge_steps[name] = min(candidates)
        persistent_strong = strongest == "strong" and "strong" not in demerge_steps
        classification = (
            "PERSISTENT_STRONG_CORE_MERGE"
            if persistent_strong
            else "THRESHOLD_CONTACT_NOT_PHYSICAL_MERGE"
        )
        for row in sorted(group_rows, key=lambda item: order[item["threshold_name"]]):
            output.append(
                {
                    **{key: value for key, value in row.items() if key != "members"},
                    "merge_group_id": "MG_" + member_label(members).replace("+", "_"),
                    "member_particle_ids": member_label(members),
                    "contact_end_step": demerge_steps.get(row["threshold_name"], ""),
                    "group_classification": classification,
                }
            )
    return output


def qualify_dissolutions(
    threshold_data: Mapping[str, ThresholdData],
    identities: Sequence[str],
    steps: Sequence[int],
    physical_dt_s: float,
    start_age_h: float,
) -> List[Dict[str, Any]]:
    absent = {
        name: first_absent_steps(threshold_data[name], identities, steps)
        for name, _threshold in THRESHOLDS
    }
    low = threshold_data["low"]
    medium = threshold_data["medium"]
    medium_events_by_member: Dict[str, Dict[str, Any]] = {}
    for event in medium.dissolution_groups:
        for member in event["members"]:
            if member in medium_events_by_member:
                raise AuditFailure("medium identity has duplicate dissolution events")
            medium_events_by_member[member] = event
    rows: List[Dict[str, Any]] = []
    covered: Set[str] = set()
    for member in identities:
        strong_step = absent["strong"][member]
        medium_step = absent["medium"][member]
        low_step = absent["low"][member]
        if medium_step is None:
            continue
        if strong_step is None or strong_step > medium_step:
            raise AuditFailure("resolved-core extinction lacks ordered strong evidence")
        if low_step is not None and low_step < medium_step:
            raise AuditFailure("dissolution threshold ordering is inverted")
        event = medium_events_by_member.get(member)
        if event is None or int(event["step"]) != medium_step:
            raise AuditFailure("resolved-core extinction lacks medium event provenance")
        if member in covered:
            raise AuditFailure("identity is covered by multiple dissolution intervals")
        covered.add(member)
        start_step = int(event["previous_step"])
        end_step = int(event["step"])
        temporal = bool(event["legacy_temporal_qualified"])
        if low_step is None:
            classification = "CORE_EXTINCTION_WITH_ATTACHED_LOW_H_TAIL"
        elif strong_step == medium_step == low_step:
            classification = (
                "TEMPORAL_DECAY_AND_FULL_SUPPORT_EXTINCTION"
                if temporal
                else "NESTED_SAME_INTERVAL_FULL_SUPPORT_EXTINCTION"
            )
        else:
            classification = "ORDERED_NESTED_CORE_TO_SUPPORT_EXTINCTION"
        rows.append(
            {
                "dissolution_group_id": "DG_" + member.replace("+", "_"),
                "member_particle_ids": member,
                "member_count": 1,
                "interval_start_step": start_step,
                "interval_end_step": end_step,
                "interval_start_age_h": start_age_h
                + start_step * physical_dt_s / 3600.0,
                "interval_end_age_h": start_age_h
                + end_step * physical_dt_s / 3600.0,
                "interval_width_s": (end_step - start_step) * physical_dt_s,
                "strong_first_absent_step": strong_step,
                "medium_first_absent_step": medium_step,
                "low_first_absent_step": "" if low_step is None else low_step,
                "last_medium_h_volume_nm3": event["last_h_volume_nm3"],
                "last_medium_equivalent_radius_nm": event["last_equivalent_radius_nm"],
                "legacy_temporal_qualified": int(temporal),
                "qualification_class": classification,
                "exact_event_time_claimed": 0,
            }
        )
    final_alive = set(medium.alive_by_step[steps[-1]])
    expected_dissolved = set(identities) - final_alive
    if covered != expected_dissolved:
        raise AuditFailure("resolved-core dissolution coverage does not close identities")
    return sorted(rows, key=lambda row: (row["interval_end_step"], row["dissolution_group_id"]))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def input_hashes(roots: Mapping[str, Path], initial: Path, steps: Path) -> Dict[str, str]:
    result = {
        "initial_components.csv": sha256(initial),
        "expected_steps.txt": sha256(steps),
    }
    for name, root in sorted(roots.items()):
        for filename in (
            "lineage_summary.txt",
            "particle_events.csv",
            "particle_lineage.csv",
            "ensemble_observables.csv",
            "status.txt",
        ):
            result[f"{name}/{filename}"] = sha256(root / filename)
    return result


def run(args: argparse.Namespace) -> Dict[str, Any]:
    expected_steps = parse_expected_steps(args.expected_steps)
    registered = read_rows(args.initial_components)
    identities = sorted(row["particle_id"] for row in registered)
    if len(identities) != args.expected_initial_count or len(set(identities)) != len(identities):
        raise AuditFailure("registered initial particle ids are not unique and complete")
    roots = {"low": args.low, "medium": args.medium, "strong": args.strong}
    data = {
        name: load_threshold(
            name,
            threshold,
            roots[name],
            registered,
            expected_steps,
            args.expected_initial_count,
            args.domain_nm,
            args.identity_tolerance_nm,
            args.mass_tolerance,
        )
        for name, threshold in THRESHOLDS
    }

    for step in expected_steps:
        if not data["strong"].alive_by_step[step].issubset(
            data["medium"].alive_by_step[step]
        ):
            raise AuditFailure("strong-threshold identities are not nested in medium")
        if not data["medium"].alive_by_step[step].issubset(
            data["low"].alive_by_step[step]
        ):
            raise AuditFailure("medium-threshold identities are not nested in low")
        reference = data["low"].observations[expected_steps.index(step)]
        for name in ("medium", "strong"):
            other = data[name].observations[expected_steps.index(step)]
            for field in ("canonical_inventory_code", "beta_volume_fraction"):
                left = float(reference[field])
                right = float(other[field])
                tolerance = 1.0e-12 * max(abs(left), abs(right), 1.0)
                if abs(left - right) > tolerance:
                    raise AuditFailure(f"full-field {field} differs across thresholds")

    merges = classify_merges(data)
    dissolutions = qualify_dissolutions(
        data,
        identities,
        expected_steps,
        args.physical_dt_s,
        args.start_age_h,
    )
    maximum_interval = max(
        (float(row["interval_width_s"]) for row in dissolutions), default=0.0
    )
    maximum_step_gap = max(
        later - earlier for earlier, later in zip(expected_steps, expected_steps[1:])
    )
    if maximum_interval > maximum_step_gap * args.physical_dt_s + 1.0e-9:
        raise AuditFailure("dissolution interval exceeds observed checkpoint gap")

    legacy_unqualified = int(
        data["medium"].summary.get("unqualified_dissolution_count", -1)
    )
    final_support_components = len(data["low"].components_by_step[expected_steps[-1]])
    final_support_members = len(data["low"].alive_by_step[expected_steps[-1]])
    final_core_components = len(data["medium"].components_by_step[expected_steps[-1]])
    final_core_members = len(data["medium"].alive_by_step[expected_steps[-1]])
    resolved_demerge_episode_count = len(
        {
            (name, row["member_particle_ids"])
            for name, _threshold in THRESHOLDS
            for row in data[name].resolved_demerges
        }
    )
    incidental_overlap_count = sum(
        len(data[name].incidental_overlap_events) for name, _threshold in THRESHOLDS
    )
    physical_merge_groups = {
        row["merge_group_id"]
        for row in merges
        if row["group_classification"] == "PERSISTENT_STRONG_CORE_MERGE"
    }
    contact_groups = {
        row["merge_group_id"]
        for row in merges
        if row["group_classification"] == "THRESHOLD_CONTACT_NOT_PHYSICAL_MERGE"
    }
    gates = {
        "snapshot_sequence_exact": True,
        "initial_identity_bijection": True,
        "periodic_identity_mapping": True,
        "component_partition_closed": True,
        "no_unresolved_split_all_thresholds": True,
        "demerge_identity_assignment_unique": True,
        "weak_overlap_edges_pruned_fail_closed": True,
        "no_new_component_all_thresholds": True,
        "merge_overlap_qualified": True,
        "merge_groups_laminar": True,
        "nested_threshold_alive_sets": True,
        "nested_threshold_extinction_order": True,
        "no_identity_reappearance": True,
        "dissolution_interval_coverage_exact": True,
        "full_field_h_volume_bound": True,
        "canonical_inventory_closed": True,
        "finite_and_bounds": True,
        "exact_event_time_not_claimed": True,
    }
    audit = {
        "schema": "PF_246CUBE_HOURLY_MERGE_DISSOLUTION_AUDIT_V1",
        "status": PASS,
        "gates": gates,
        "contract": {
            "thresholds": {name: value for name, value in THRESHOLDS},
            "expected_initial_count": args.expected_initial_count,
            "domain_nm": args.domain_nm,
            "physical_dt_s": args.physical_dt_s,
            "start_age_h": args.start_age_h,
            "identity_tolerance_nm": args.identity_tolerance_nm,
            "mass_tolerance": args.mass_tolerance,
            "event_time_semantics": "BOUNDED_BY_CONSECUTIVE_SNAPSHOTS",
            "particle_core_semantics": "H_GE_1E3_WITH_H_GE_5E3_ORDERED_EVIDENCE",
            "low_support_semantics": "H_GE_1E4_NOT_A_STANDALONE_PARTICLE_CORE",
            "exact_event_time_claimed": False,
        },
        "metrics": {
            "snapshot_count": len(expected_steps),
            "initial_particle_count": args.expected_initial_count,
            "final_low_support_component_count": final_support_components,
            "final_low_support_identity_count": final_support_members,
            "final_resolved_core_component_count": final_core_components,
            "final_resolved_core_identity_count": final_core_members,
            "dissolution_interval_count": len(dissolutions),
            "dissolved_initial_identity_count": sum(
                int(row["member_count"]) for row in dissolutions
            ),
            "legacy_medium_unqualified_dissolution_count": legacy_unqualified,
            "merge_group_count": len({row["merge_group_id"] for row in merges}),
            "merge_event_rows": len(merges),
            "threshold_contact_group_count": len(contact_groups),
            "persistent_strong_core_merge_group_count": len(physical_merge_groups),
            "resolved_demerge_episode_count": resolved_demerge_episode_count,
            "incidental_weak_overlap_event_count": incidental_overlap_count,
            "attached_low_h_tail_dissolution_count": sum(
                row["qualification_class"]
                == "CORE_EXTINCTION_WITH_ATTACHED_LOW_H_TAIL"
                for row in dissolutions
            ),
            "maximum_dissolution_interval_s": maximum_interval,
            "maximum_checkpoint_gap_steps": maximum_step_gap,
            "max_mass_relative_error": max(
                abs(float(row["mass_relative_error"]))
                for row in data["low"].observations
            ),
        },
        "threshold_identity_max_delta_nm": {
            name: data[name].identity_max_delta_nm for name, _value in THRESHOLDS
        },
        "threshold_summaries": {
            name: data[name].summary for name, _value in THRESHOLDS
        },
        "dissolution_intervals": dissolutions,
        "merge_groups": merges,
        "resolved_demerges": [
            row
            for name, _threshold in THRESHOLDS
            for row in data[name].resolved_demerges
        ],
        "incidental_overlap_events": [
            row
            for name, _threshold in THRESHOLDS
            for row in data[name].incidental_overlap_events
        ],
        "input_sha256": input_hashes(roots, args.initial_components, args.expected_steps),
    }
    return audit


def write_success(out: Path, audit: Mapping[str, Any]) -> None:
    dissolution_fields = (
        "dissolution_group_id",
        "member_particle_ids",
        "member_count",
        "interval_start_step",
        "interval_end_step",
        "interval_start_age_h",
        "interval_end_age_h",
        "interval_width_s",
        "strong_first_absent_step",
        "medium_first_absent_step",
        "low_first_absent_step",
        "last_medium_h_volume_nm3",
        "last_medium_equivalent_radius_nm",
        "legacy_temporal_qualified",
        "qualification_class",
        "exact_event_time_claimed",
    )
    merge_fields = (
        "merge_group_id",
        "member_particle_ids",
        "threshold_name",
        "h_threshold",
        "previous_step",
        "step",
        "stable_particle_ids",
        "child_stable_id",
        "overlap_detail",
        "contact_end_step",
        "group_classification",
    )
    write_csv(
        out / "dissolution_intervals.csv",
        audit["dissolution_intervals"],
        dissolution_fields,
    )
    write_csv(out / "merge_groups.csv", audit["merge_groups"], merge_fields)
    write_csv(
        out / "resolved_demerge_observations.csv",
        audit["resolved_demerges"],
        (
            "threshold_name",
            "h_threshold",
            "step",
            "member_particle_ids",
            "member_count",
            "child_count",
            "maximum_anchor_delta_nm",
            "assignment_margin_nm",
        ),
    )
    write_csv(
        out / "incidental_overlap_events.csv",
        audit["incidental_overlap_events"],
        (
            "threshold_name",
            "h_threshold",
            "step",
            "stable_particle_id",
            "overlap_detail",
            "classification",
        ),
    )
    (out / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics = audit["metrics"]
    report = [
        "# Hour-cadence merge and dissolution qualification",
        "",
        f"`{PASS}`",
        "",
        "A dissolution is registered as an interval bounded by two existing "
        "snapshots, never as an invented exact event time. Qualification "
        "requires ordered resolved-core extinction at h=5e-3 and 1e-3, while "
        "h=1e-4 is retained only as a diffuse-support diagnostic. Reversible "
        "threshold necks are resolved by a unique periodic-centroid bijection; "
        "unresolved split, new component, or reappearance still fails closed; "
        "and full-field h-volume and canonical-inventory closure.",
        "",
        "## Metrics",
        "",
        f"- snapshots: {metrics['snapshot_count']}",
        f"- initial identities: {metrics['initial_particle_count']}",
        f"- final resolved-core components: {metrics['final_resolved_core_component_count']}",
        f"- final low-support components: {metrics['final_low_support_component_count']}",
        f"- dissolved initial identities: {metrics['dissolved_initial_identity_count']}",
        f"- dissolution intervals: {metrics['dissolution_interval_count']}",
        f"- merge groups: {metrics['merge_group_count']}",
        f"- threshold-contact groups (not physical merges): {metrics['threshold_contact_group_count']}",
        f"- persistent strong-core merges: {metrics['persistent_strong_core_merge_group_count']}",
        f"- attached low-h tails after core extinction: {metrics['attached_low_h_tail_dissolution_count']}",
        f"- legacy unqualified core dissolutions re-audited: {metrics['legacy_medium_unqualified_dissolution_count']}",
        f"- maximum bounded interval: {metrics['maximum_dissolution_interval_s']:.12g} s",
        f"- maximum mass relative error: {metrics['max_mass_relative_error']:.12g}",
        "",
        "The original strict tracker outputs remain unchanged. This report "
        "supplements them with an hour-cadence evidence contract.",
    ]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (out / "first_failure.csv").write_text(
        "stage,gate,detail\n", encoding="utf-8"
    )
    (out / "status.txt").write_text(PASS + "\n", encoding="utf-8")
    terminal = [
        f"hourly_lineage_status={PASS}",
        f"snapshot_count={metrics['snapshot_count']}",
        f"dissolution_interval_count={metrics['dissolution_interval_count']}",
        f"dissolved_initial_identity_count={metrics['dissolved_initial_identity_count']}",
        f"merge_group_count={metrics['merge_group_count']}",
        f"threshold_contact_group_count={metrics['threshold_contact_group_count']}",
        f"persistent_strong_core_merge_group_count={metrics['persistent_strong_core_merge_group_count']}",
        "unresolved_split_count=0",
        "new_component_count=0",
        "exact_event_time_claimed=false",
        f"final_status={PASS}",
    ]
    (out / "final_terminal_output.txt").write_text(
        "\n".join(terminal) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--medium", type=Path, required=True)
    parser.add_argument("--strong", type=Path, required=True)
    parser.add_argument("--initial-components", type=Path, required=True)
    parser.add_argument("--expected-steps", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-initial-count", type=int, default=96)
    parser.add_argument("--domain-nm", type=float, default=246.0)
    parser.add_argument("--physical-dt-s", type=float, default=0.9909260953431841)
    parser.add_argument("--start-age-h", type=float, default=6.0)
    parser.add_argument("--identity-tolerance-nm", type=float, default=1.0e-3)
    parser.add_argument("--mass-tolerance", type=float, default=1.0e-10)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)
    try:
        audit = run(args)
        write_success(args.out, audit)
        print(PASS)
    except (AuditFailure, KeyError, OSError, ValueError) as exc:
        detail = str(exc).replace("\n", " ")
        (args.out / "status.txt").write_text(
            f"{FAIL}\n{detail}\n", encoding="utf-8"
        )
        with (args.out / "first_failure.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(("stage", "gate", "detail"))
            writer.writerow(("hourly_lineage", "fail_closed", detail))
        (args.out / "final_terminal_output.txt").write_text(
            f"hourly_lineage_status={FAIL}\n"
            f"recommended_next_action=inspect_first_failure\n"
            f"final_status={FAIL}\n",
            encoding="utf-8",
        )
        raise SystemExit(f"[fatal] {detail}") from exc


if __name__ == "__main__":
    main()
