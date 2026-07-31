#!/usr/bin/env python3
"""Fail-closed single-particle identity audit for elastic profile handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

from analyze_pf_zero_mode_checkpoints import read_checkpoint
from materialize_pf_elastic_target_profile_v1 import h_of_phi


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def periodic_components(mask: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(3, 1)
    raw, count = ndimage.label(mask, structure=structure)
    parent = np.arange(count + 1, dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: np.ndarray, right: np.ndarray) -> None:
        pairs = np.stack((left.ravel(), right.ravel()), axis=1)
        pairs = pairs[(pairs[:, 0] != 0) & (pairs[:, 1] != 0)]
        for a, b in np.unique(pairs, axis=0):
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[rb] = ra

    if count:
        union(raw[0, :, :], raw[-1, :, :])
        union(raw[:, 0, :], raw[:, -1, :])
        union(raw[:, :, 0], raw[:, :, -1])
    roots = np.arange(count + 1, dtype=np.int32)
    for index in range(1, count + 1):
        roots[index] = find(index)
    unique_roots = sorted(set(int(value) for value in roots[1:]))
    remap = {root: index + 1 for index, root in enumerate(unique_roots)}
    dense = np.zeros_like(raw, dtype=np.int32)
    for label in range(1, count + 1):
        dense[raw == label] = remap[find(label)]
    return dense


def load_profile_phi(manifest_path: Path) -> tuple[np.ndarray, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shape = tuple(int(manifest["grid"][key]) for key in ("Nx", "Ny", "Nz"))
    field = manifest["fields"]["phi"]
    path = manifest_path.parent / field["path"]
    if sha256(path) != field["sha256"]:
        raise SystemExit("profile phi hash mismatch")
    values = np.fromfile(path, dtype="<f8")
    if values.size != math.prod(shape):
        raise SystemExit("profile phi size mismatch")
    return values.reshape(shape, order="C"), manifest


def load_checkpoint_phi(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    record = read_checkpoint(path)
    if tuple(record[1]) != shape:
        raise SystemExit(f"{path}: checkpoint shape mismatch")
    return np.asarray(record[5]).reshape(shape, order="C")


def single_component(mask: np.ndarray, label: str) -> dict:
    labels = periodic_components(mask)
    count = int(labels.max())
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one component, got {count}")
    return {
        "component_count": count,
        "support_voxels": int(np.count_nonzero(labels)),
        "mask": labels > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-manifest", type=Path, required=True)
    parser.add_argument("--initial-probe-checkpoint", type=Path, required=True)
    parser.add_argument("--continuous-checkpoint", type=Path, required=True)
    parser.add_argument("--restart-checkpoint", type=Path, required=True)
    parser.add_argument("--refined-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--h-threshold", type=float, default=1.0e-4)
    parser.add_argument("--minimum-overlap-fraction", type=float, default=0.90)
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    phi_initial, manifest = load_profile_phi(args.profile_manifest)
    shape = phi_initial.shape
    fields = {
        "initial": phi_initial,
        "initial_probe": load_checkpoint_phi(
            args.initial_probe_checkpoint, shape
        ),
        "continuous": load_checkpoint_phi(args.continuous_checkpoint, shape),
        "restart": load_checkpoint_phi(args.restart_checkpoint, shape),
        "refined": load_checkpoint_phi(args.refined_checkpoint, shape),
    }
    states = {
        label: single_component(
            h_of_phi(np.clip(phi, 0.0, 1.0)) > args.h_threshold,
            label,
        )
        for label, phi in fields.items()
    }
    initial_mask = states["initial"]["mask"]
    overlaps = {}
    for label in ("initial_probe", "continuous", "restart", "refined"):
        candidate = states[label]["mask"]
        overlap = np.count_nonzero(initial_mask & candidate)
        denominator = max(
            min(np.count_nonzero(initial_mask), np.count_nonzero(candidate)),
            1,
        )
        overlaps[label] = float(overlap / denominator)
    continuous_restart_exact = np.array_equal(
        fields["continuous"], fields["restart"]
    )
    passed = (
        continuous_restart_exact
        and all(
            value >= args.minimum_overlap_fraction
            for value in overlaps.values()
        )
    )
    audit = {
        "schema": "PF_ELASTIC_TARGET_PROFILE_IDENTITY_AUDIT_V1",
        "status": (
            "PASS_ELASTIC_TARGET_PROFILE_SINGLE_PARTICLE_IDENTITY_V1"
            if passed
            else "FAIL_ELASTIC_TARGET_PROFILE_IDENTITY_V1"
        ),
        "profile_manifest": str(args.profile_manifest),
        "profile_manifest_sha256": sha256(args.profile_manifest),
        "target_radius_nm": manifest["geometry"][
            "target_equivalent_radius_nm"
        ],
        "h_threshold": args.h_threshold,
        "minimum_overlap_fraction": args.minimum_overlap_fraction,
        "continuous_restart_phi_exact": continuous_restart_exact,
        "states": {
            label: {
                "component_count": state["component_count"],
                "support_voxels": state["support_voxels"],
            }
            for label, state in states.items()
        },
        "initial_overlap_fraction": overlaps,
        "checkpoint_sha256": {
            "initial_probe": sha256(args.initial_probe_checkpoint),
            "continuous": sha256(args.continuous_checkpoint),
            "restart": sha256(args.restart_checkpoint),
            "refined": sha256(args.refined_checkpoint),
        },
    }
    audit_path = args.out / "particle_identity_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out / "final_terminal_output.txt").write_text(
        f"particle_identity_status={audit['status']}\n"
        f"target_radius_nm={audit['target_radius_nm']}\n"
        f"continuous_restart_phi_exact={str(continuous_restart_exact).lower()}\n"
        f"audit_sha256={sha256(audit_path)}\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
