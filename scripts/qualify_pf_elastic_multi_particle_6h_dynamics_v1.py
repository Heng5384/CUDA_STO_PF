#!/usr/bin/env python3
"""Fail-closed short handoff/dt/restart audit for a V1 multi-particle fixture."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from analyze_pf_zero_mode_checkpoints import read_checkpoint
from materialize_pf_elastic_multi_particle_6h_fixture_v1 import components, h_of_phi


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def field(manifest: Dict[str, Any], root: Path, name: str) -> np.ndarray:
    row = manifest["fields"][name]
    path = root / row["path"]
    if sha256(path) != row["sha256"]:
        raise ValueError(f"fixture field hash mismatch: {name}")
    shape = tuple(int(manifest["grid"][axis]) for axis in ("Nx", "Ny", "Nz"))
    value = np.fromfile(path, dtype="<f8")
    if value.size != math.prod(shape):
        raise ValueError(f"fixture field size mismatch: {name}")
    return value.reshape(shape, order="C")


def checkpoint(path: Path, shape: Tuple[int, int, int]) -> Dict[str, Any]:
    step, actual_shape, dt, temp, target, phi, y, xb, dy = read_checkpoint(path)
    if tuple(actual_shape) != shape:
        raise ValueError(f"checkpoint shape mismatch: {path}")
    return {
        "path": path, "step": step, "dt": dt, "temperature": temp, "target": target,
        "phi": np.asarray(phi).reshape(shape, order="C"),
        "Y": np.asarray(y).reshape(shape, order="C"),
        "xB": np.asarray(xb).reshape(shape, order="C"),
        "dY": np.asarray(dy).reshape(shape, order="C"),
    }


def match_components(initial: np.ndarray, current: np.ndarray, count: int) -> Tuple[bool, Dict[int, int], Dict[int, float]]:
    """Maximum-overlap bijection, suitable for the small validation fixture."""
    scores = np.zeros((count, count), dtype=np.int64)
    for previous in range(count):
        cells = initial == previous
        if np.any(cells):
            values, hits = np.unique(current[cells], return_counts=True)
            for value, hit in zip(values, hits):
                if 0 <= int(value) < count:
                    scores[previous, int(value)] = int(hit)
    best = None
    for perm in itertools.permutations(range(count)):
        total = sum(int(scores[index, value]) for index, value in enumerate(perm))
        key = (total, tuple(-value for value in perm))
        if best is None or key > best[0]:
            best = (key, perm)
    assert best is not None
    mapping = {index: int(value) for index, value in enumerate(best[1])}
    fractions: Dict[int, float] = {}
    for previous, value in mapping.items():
        denominator = int(np.count_nonzero(initial == previous))
        fractions[previous] = (float(scores[previous, value]) / denominator) if denominator else 0.0
    return all(value > 0.0 for value in fractions.values()), mapping, fractions


def tracked_state(phi: np.ndarray, threshold: float, dx_nm: float) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    labels, rows = components(h_of_phi(phi), threshold, dx_nm)
    return labels, sorted(rows, key=lambda row: int(row["component_label"]))


def finite_elastic_log(text: str) -> bool:
    lowered = text.lower()
    return "elastic" in lowered and "nan" not in lowered and "inf" not in lowered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--initial-probe-checkpoint", type=Path, required=True)
    parser.add_argument("--continuous-checkpoint", type=Path, required=True)
    parser.add_argument("--restart-checkpoint", type=Path, required=True)
    parser.add_argument("--refined-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-probe-stdout", type=Path, required=True)
    parser.add_argument("--continuous-stdout", type=Path, required=True)
    parser.add_argument("--restart-stdout", type=Path, required=True)
    parser.add_argument("--refined-stdout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.fixture_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "PF_ELASTIC_MULTI_PARTICLE_6H_FIXTURE_V1" or manifest.get("fixture_kind") != "E2":
        raise SystemExit("expected an E2 multi-particle V1 fixture")
    root = args.fixture_manifest.parent
    shape = tuple(int(manifest["grid"][axis]) for axis in ("Nx", "Ny", "Nz"))
    dx_nm = float(manifest["grid"]["dx_nm"])
    threshold = float(manifest["component_contract"]["h_threshold"])
    expected_count = int(manifest["component_contract"]["expected_count"])
    phi0 = field(manifest, root, "phi")
    xb0 = field(manifest, root, "xB_alpha")
    c0 = field(manifest, root, "C_B_tot")
    initial_labels, initial_rows = tracked_state(phi0, threshold, dx_nm)
    if len(initial_rows) != expected_count:
        raise SystemExit("fixture component count is not self-consistent")
    probes = {
        "one_step": checkpoint(args.initial_probe_checkpoint, shape),
        "continuous": checkpoint(args.continuous_checkpoint, shape),
        "restart": checkpoint(args.restart_checkpoint, shape),
        "refined": checkpoint(args.refined_checkpoint, shape),
    }
    if probes["one_step"]["step"] != 1:
        raise SystemExit("one-step probe did not end at step 1")
    if probes["continuous"]["step"] != probes["restart"]["step"]:
        raise SystemExit("continuous/restart endpoints differ")
    if abs(probes["continuous"]["step"] * probes["continuous"]["dt"] - probes["refined"]["step"] * probes["refined"]["dt"]) > 1e-12:
        raise SystemExit("dt endpoints differ")
    states: Dict[str, Dict[str, Any]] = {}
    trajectories: List[Dict[str, Any]] = []
    for name, item in probes.items():
        labels, rows = tracked_state(item["phi"], threshold, dx_nm)
        identity_ok = len(rows) == expected_count
        mapping: Dict[int, int] = {}
        overlap: Dict[int, float] = {}
        if identity_ok:
            identity_ok, mapping, overlap = match_components(initial_labels, labels, expected_count)
        h = h_of_phi(item["phi"])
        c = (1.0 - h) * item["xB"] + h
        states[name] = {"labels": labels, "rows": rows, "identity_ok": identity_ok, "mapping": mapping, "overlap": overlap,
                        "mass": float(np.sum(c, dtype=np.float64)), "h": h}
        inverse = {value: key for key, value in mapping.items()}
        for row in rows:
            copy = dict(row)
            copy.update({"state": name, "step": item["step"], "dt_code": item["dt"],
                         "initial_component_label": inverse.get(int(row["component_label"]), -1),
                         "initial_support_overlap": overlap.get(inverse.get(int(row["component_label"]), -1), 0.0)})
            trajectories.append(copy)
    restart_raw_equal = args.continuous_checkpoint.read_bytes() == args.restart_checkpoint.read_bytes()
    restart_fields_equal = all(np.array_equal(probes["continuous"][name], probes["restart"][name]) for name in ("phi", "xB", "Y", "dY"))
    phi_l1 = float(np.sum(np.abs(probes["continuous"]["phi"] - probes["refined"]["phi"]), dtype=np.float64) / max(float(np.sum(states["continuous"]["h"])), 1.0))
    xb_mae = float(np.mean(np.abs(probes["continuous"]["xB"] - probes["refined"]["xB"]), dtype=np.float64))
    axis_rel = 0.0
    for left, right in zip(states["continuous"]["rows"], states["refined"]["rows"]):
        a = np.asarray(left["semi_axes_nm"], dtype=float)
        b = np.asarray(right["semi_axes_nm"], dtype=float)
        axis_rel = max(axis_rel, float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-30))))
    logs = [path.read_text(encoding="utf-8", errors="replace") for path in (
        args.initial_probe_stdout, args.continuous_stdout, args.restart_stdout, args.refined_stdout)]
    no_clipping = all("raw init required clamping" not in text for text in logs)
    zero_mode = all("PF_ZERO_MODE_FINAL_AUDIT status=PASS" in text for text in logs)
    prohibited = all(marker not in "\n".join(logs) for marker in ("GP_EVENT", "GP_BIRTH", "BETA_NUCLEATION_EVENT", "source_event"))
    elastic = all(finite_elastic_log(text) for text in logs)
    mass_rel = max(abs(states[name]["mass"] - float(np.sum(c0, dtype=np.float64))) for name in states) / max(abs(float(np.sum(c0, dtype=np.float64))), 1.0)
    gates = {
        "component_identity": all(states[name]["identity_ok"] for name in states),
        "one_step_overlap": min(states["one_step"]["overlap"].values(), default=0.0) >= 0.90,
        "restart_bytewise": restart_raw_equal and restart_fields_equal,
        "dt_phi_L1": phi_l1 <= 2.0e-2,
        "dt_axis": axis_rel <= 2.0e-2,
        "dt_xB_hard": xb_mae <= 5.0e-5,
        "mass": mass_rel <= 1.0e-10,
        "zero_mode": zero_mode,
        "no_clipping": no_clipping,
        "prohibited_paths": prohibited,
        "elastic_diagnostics": elastic,
    }
    status = "PASS_CONSERVED_ELASTIC_MULTI_PARTICLE_HANDOFF_RESTART_DT_V1" if all(gates.values()) else "FAIL_CONSERVED_ELASTIC_MULTI_PARTICLE_HANDOFF_RESTART_DT_V1"
    with (args.out / "particle_trajectories.json").open("w", encoding="utf-8") as handle:
        json.dump(trajectories, handle, indent=2, sort_keys=True)
        handle.write("\n")
    audit = {"schema": "PF_ELASTIC_MULTI_PARTICLE_DYNAMIC_QUALIFICATION_V1", "status": status,
             "fixture_manifest_sha256": sha256(args.fixture_manifest), "checkpoints": {key: sha256(value["path"]) for key, value in probes.items()},
             "gates": gates, "metrics": {"mass_relative": mass_rel, "dt_phi_L1": phi_l1, "dt_xB_mean_absolute": xb_mae,
             "dt_xB_preferred": xb_mae <= 2.0e-5, "dt_axis_relative": axis_rel, "restart_raw_equal": restart_raw_equal,
             "restart_fields_equal": restart_fields_equal}, "states": {name: {"step": value["step"], "dt": value["dt"], "component_count": len(states[name]["rows"]), "identity_mapping": states[name]["mapping"], "overlap": states[name]["overlap"]} for name, value in probes.items()}}
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "final_terminal_output.txt").write_text("\n".join(f"{key}={value}" for key, value in {
        "handoff_status": status, "component_identity_status": gates["component_identity"], "restart_status": gates["restart_bytewise"],
        "dt_refinement_status": gates["dt_phi_L1"] and gates["dt_axis"] and gates["dt_xB_hard"], "zero_mode_status": gates["zero_mode"],
        "raw_field_clipping_status": gates["no_clipping"], "mass_relative": mass_rel, "dt_phi_L1": phi_l1,
        "dt_xB_mean_absolute": xb_mae, "dt_axis_relative": axis_rel}.items()) + "\n", encoding="utf-8")
    print(status)


if __name__ == "__main__":
    main()
