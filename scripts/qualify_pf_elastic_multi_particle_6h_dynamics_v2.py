#!/usr/bin/env python3
"""Fail-closed handoff, timestep and restart audit for qualified fixtures."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np

import qualify_pf_elastic_multi_particle_6h_dynamics_v1 as base


V2_SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_6H_FIXTURE_V2"
V3_SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_COMMON_MATRIX_PROFILE_V3"
V4_SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_FIXED_PHI_COMMON_MATRIX_PROFILE_V4"
V5_SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_COMPONENT_VOLUME_PROFILE_V5"
CONDITIONAL_SCHEMA = "PF_MASS_CONSERVING_LIBRARY_HANDOFF_MANIFEST_V1"
CONDITIONAL_INITIAL_STATE_CLASS = (
    "MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1"
)
V2_PASS = "PASS_CONSERVED_ELASTIC_MULTI_PARTICLE_HANDOFF_RESTART_DT_V2"
V2_FAIL = "FAIL_CONSERVED_ELASTIC_MULTI_PARTICLE_HANDOFF_RESTART_DT_V2"
V3_PASS = "PASS_CONSERVED_ELASTIC_MULTI_PARTICLE_HANDOFF_RESTART_DT_V3"
V3_FAIL = "FAIL_CONSERVED_ELASTIC_MULTI_PARTICLE_HANDOFF_RESTART_DT_V3"
V4_PASS = "PASS_CONSERVED_ELASTIC_MULTI_PARTICLE_HANDOFF_RESTART_DT_V4"
V4_FAIL = "FAIL_CONSERVED_ELASTIC_MULTI_PARTICLE_HANDOFF_RESTART_DT_V4"
V5_PASS = "PASS_CONSERVED_ELASTIC_MULTI_PARTICLE_HANDOFF_RESTART_DT_V5"
V5_FAIL = "FAIL_CONSERVED_ELASTIC_MULTI_PARTICLE_HANDOFF_RESTART_DT_V5"
CONDITIONAL_PASS = (
    "PASS_MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1"
)
CONDITIONAL_FAIL = (
    "FAIL_MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1"
)


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
    schema = manifest.get("schema")
    if schema == V2_SCHEMA:
        if manifest.get("fixture_kind") != "E2":
            raise SystemExit("expected an E2 multi-particle V2 fixture")
        if manifest.get("combination", {}).get("composition_contract") != "PHASE_CONSISTENT_DELTA_X_ALPHA_V2":
            raise SystemExit("fixture does not declare the required V2 composition contract")
        pass_marker, fail_marker = V2_PASS, V2_FAIL
        history_required = False
    elif schema == V3_SCHEMA:
        if manifest.get("fixture_kind") != "E2_COMMON_MATRIX_CONSTRAINED_PROFILE":
            raise SystemExit("expected a constrained common-matrix V3 E2 fixture")
        if manifest.get("composition_contract", {}).get("final_profile") != "FULL_MODEL_ELASTIC_MASS_CONSTRAINED_MINIMIZATION_V3":
            raise SystemExit("V3 fixture lacks the constrained-profile contract")
        pass_marker, fail_marker = V3_PASS, V3_FAIL
        history_required = bool(manifest.get("time_level_contract", {}).get("dynamic_raw_history_required"))
    elif schema == V4_SCHEMA:
        if manifest.get("fixture_kind") != "E2_FIXED_PHI_COMMON_MATRIX_CONSTRAINED_PROFILE":
            raise SystemExit("expected a fixed-phi constrained common-matrix V4 E2 fixture")
        if manifest.get("composition_contract", {}).get("final_profile") != "FIXED_PHI_FULL_MODEL_CONSERVED_COMPOSITION_MINIMIZATION_V4":
            raise SystemExit("V4 fixture lacks the fixed-phi composition contract")
        if manifest.get("composition_contract", {}).get("phi_evolution") != "FROZEN_BYTE_IDENTICAL_TO_COMMON_MATRIX_SEED_V4":
            raise SystemExit("V4 fixture lacks the byte-identical phi contract")
        if manifest.get("minimization", {}).get("seed_to_final_phi_bytewise_equal") is not True:
            raise SystemExit("V4 fixture did not prove seed/final phi identity")
        pass_marker, fail_marker = V4_PASS, V4_FAIL
        history_required = bool(manifest.get("time_level_contract", {}).get("dynamic_raw_history_required"))
    elif schema == V5_SCHEMA:
        if manifest.get("fixture_kind") != "E2_COMPONENT_VOLUME_CONSTRAINED_PROFILE":
            raise SystemExit("expected a component-volume constrained V5 E2 fixture")
        contract = manifest.get("composition_contract", {})
        if contract.get("final_profile") != "JOINT_PHI_Y_COMPONENT_H_VOLUME_CONSTRAINED_MINIMIZATION_V5":
            raise SystemExit("V5 fixture lacks the joint component-constrained profile contract")
        if contract.get("inter_component_volume_exchange") is not False:
            raise SystemExit("V5 fixture does not prohibit inter-component volume exchange")
        component_contract = manifest.get("component_contract", {})
        if component_contract.get("per_particle_h_volume_constrained") is not True:
            raise SystemExit("V5 fixture lacks per-particle volume provenance")
        if not isinstance(manifest.get("minimization", {}).get("component_target_h_sums"), list):
            raise SystemExit("V5 fixture lacks component h-volume targets")
        pass_marker, fail_marker = V5_PASS, V5_FAIL
        history_required = bool(manifest.get("time_level_contract", {}).get("dynamic_raw_history_required"))
    elif schema == CONDITIONAL_SCHEMA:
        if manifest.get("initial_state_class") != CONDITIONAL_INITIAL_STATE_CLASS:
            raise SystemExit("wrong conditional-handoff initial_state_class")
        if manifest.get("validation_only") is not True:
            raise SystemExit("conditional handoff must remain validation-only")
        if manifest.get("common_multi_particle_equilibrium_required") is not False:
            raise SystemExit("conditional handoff incorrectly requires common equilibrium")
        if manifest.get("common_multi_particle_equilibrium_claim") is not False:
            raise SystemExit("conditional handoff incorrectly claims common equilibrium")
        if manifest.get("initial_relaxation_is_physical_evolution") is not True:
            raise SystemExit("conditional handoff does not count initial relaxation as physical time")
        if manifest.get("particle_profiles_locked_after_t0") is not False:
            raise SystemExit("conditional handoff incorrectly locks particle profiles")
        if manifest.get("full_field_xB_dt_MAE_blocking") is not False:
            raise SystemExit("conditional handoff has the wrong dt-xB policy")
        if manifest.get("assembly_contract", {}).get("optimizer_invoked") is not False:
            raise SystemExit("conditional handoff invoked a prohibited optimizer")
        pass_marker, fail_marker = CONDITIONAL_PASS, CONDITIONAL_FAIL
        history_required = False
    else:
        raise SystemExit("unsupported multi-particle fixture schema")
    root = args.fixture_manifest.parent
    shape = tuple(int(manifest["grid"][axis]) for axis in ("Nx", "Ny", "Nz"))
    dx_nm = float(manifest["grid"]["dx_nm"])
    threshold = float(manifest["component_contract"]["h_threshold"])
    expected_count = int(manifest["component_contract"]["expected_count"])
    phi0 = base.field(manifest, root, "phi")
    xb0 = base.field(manifest, root, "xB_alpha")
    c0 = base.field(manifest, root, "C_B_tot")
    initial_labels, initial_rows = base.tracked_state(phi0, threshold, dx_nm)
    if len(initial_rows) != expected_count:
        raise SystemExit("fixture component count is not self-consistent")
    paths = {
        "one_step": args.initial_probe_checkpoint, "continuous": args.continuous_checkpoint,
        "restart": args.restart_checkpoint, "refined": args.refined_checkpoint,
    }
    probes = {name: base.checkpoint(path, shape) for name, path in paths.items()}
    if probes["one_step"]["step"] != 1:
        raise SystemExit("one-step probe did not end at step 1")
    if probes["continuous"]["step"] != probes["restart"]["step"]:
        raise SystemExit("continuous/restart endpoints differ")
    if abs(probes["continuous"]["step"] * probes["continuous"]["dt"] - probes["refined"]["step"] * probes["refined"]["dt"]) > 1.0e-12:
        raise SystemExit("dt endpoints differ")
    states: Dict[str, Dict[str, Any]] = {}
    trajectories = []
    for name, item in probes.items():
        labels, rows = base.tracked_state(item["phi"], threshold, dx_nm)
        identity_ok = len(rows) == expected_count
        mapping: Dict[int, int] = {}
        overlap: Dict[int, float] = {}
        if identity_ok:
            identity_ok, mapping, overlap = base.match_components(initial_labels, labels, expected_count)
        h = base.h_of_phi(item["phi"])
        c = (1.0 - h) * item["xB"] + h
        states[name] = {"labels": labels, "rows": rows, "identity_ok": identity_ok, "mapping": mapping, "overlap": overlap, "mass": float(np.sum(c, dtype=np.float64)), "h": h}
        inverse = {value: key for key, value in mapping.items()}
        for row in rows:
            copy = dict(row)
            prior = inverse.get(int(row["component_label"]), -1)
            copy.update({"state": name, "step": item["step"], "dt_code": item["dt"], "initial_component_label": prior, "initial_support_overlap": overlap.get(prior, 0.0)})
            trajectories.append(copy)
    restart_raw_equal = args.continuous_checkpoint.read_bytes() == args.restart_checkpoint.read_bytes()
    restart_fields_equal = all(np.array_equal(probes["continuous"][name], probes["restart"][name]) for name in ("phi", "xB", "Y", "dY"))
    phi_l1 = float(np.sum(np.abs(probes["continuous"]["phi"] - probes["refined"]["phi"]), dtype=np.float64) / max(float(np.sum(states["continuous"]["h"])), 1.0))
    xb_mae = float(np.mean(np.abs(probes["continuous"]["xB"] - probes["refined"]["xB"]), dtype=np.float64))
    xb_difference = np.abs(probes["continuous"]["xB"] - probes["refined"]["xB"])
    alpha_continuous = 1.0 - states["continuous"]["h"]
    def masked_mae(mask: np.ndarray) -> float:
        return float(np.mean(xb_difference[mask], dtype=np.float64)) if np.any(mask) else math.nan
    xb_mae_alpha_gt_1e_2 = masked_mae(alpha_continuous > 1.0e-2)
    xb_mae_alpha_gt_1e_1 = masked_mae(alpha_continuous > 1.0e-1)
    xb_mae_far_matrix = masked_mae(states["continuous"]["h"] < 1.0e-4)
    axis_rel = 0.0
    for left, right in zip(states["continuous"]["rows"], states["refined"]["rows"]):
        a, b = np.asarray(left["semi_axes_nm"], dtype=float), np.asarray(right["semi_axes_nm"], dtype=float)
        axis_rel = max(axis_rel, float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1.0e-30))))
    logs = [path.read_text(encoding="utf-8", errors="replace") for path in (args.initial_probe_stdout, args.continuous_stdout, args.restart_stdout, args.refined_stdout)]
    joined = "\n".join(logs)
    conditional = schema == CONDITIONAL_SCHEMA
    fixture_sha256 = base.sha256(args.fixture_manifest)
    library_sha256 = manifest.get("profile_library_manifest_sha256", "")
    provenance_tokens = (
        f"initial_state_class={manifest.get('initial_state_class', '')}",
        f"fixture_manifest_sha256={fixture_sha256}",
        f"profile_library_manifest_sha256={library_sha256}",
    )
    first_step_profile_changed = not np.array_equal(
        phi0, probes["one_step"]["phi"]
    )
    gates = {
        "component_identity": all(states[name]["identity_ok"] for name in states),
        "one_step_overlap": min(states["one_step"]["overlap"].values(), default=0.0) >= 0.90,
        "restart_bytewise": restart_raw_equal and restart_fields_equal,
        "dt_phi_L1": phi_l1 <= 2.0e-2,
        "dt_axis": axis_rel <= 2.0e-2,
        # The old xB threshold remains a reported diagnostic.  It is not an
        # initial-state identity gate for the conditional handoff class.
        "dt_xB_hard": (xb_mae <= 5.0e-5) if not conditional else True,
        "mass": max(abs(states[name]["mass"] - float(np.sum(c0, dtype=np.float64))) for name in states) / max(abs(float(np.sum(c0, dtype=np.float64))), 1.0) <= 1.0e-10,
        "zero_mode": all("PF_ZERO_MODE_FINAL_AUDIT status=PASS" in text for text in logs),
        "no_clipping": all("raw init required clamping" not in text for text in logs),
        "prohibited_paths": all(token not in joined for token in ("GP_EVENT", "GP_BIRTH", "BETA_NUCLEATION_EVENT", "source_event")),
        # ``inferred`` appears in ordinary physical-unit diagnostics, so a
        # raw substring search for ``inf`` is not a finite-value test.
        "elastic_diagnostics": all(
            "elastic" in text.lower()
            and re.search(r"(?<![a-z])(?:nan|[-+]?inf(?:inity)?)(?![a-z])", text.lower()) is None
            for text in logs
        ),
    }
    if conditional:
        gates["checkpoint_initial_state_provenance"] = all(
            all(token in text for token in provenance_tokens)
            for text in logs
        )
        gates["initial_profile_free_after_t0"] = first_step_profile_changed
    if history_required:
        gates["raw_time_level_history"] = all(
            "dY_dt_prev initialization : provenance_pinned_raw_history" in text
            # Restart is required to restore its checksum-protected history
            # from the checkpoint, so only fresh raw-field launches must
            # report the raw-history load marker.
            for text in (logs[0], logs[1], logs[3])
        )
    status = pass_marker if all(gates.values()) else fail_marker
    mass_rel = max(abs(states[name]["mass"] - float(np.sum(c0, dtype=np.float64))) for name in states) / max(abs(float(np.sum(c0, dtype=np.float64))), 1.0)
    (args.out / "particle_trajectories.json").write_text(json.dumps(trajectories, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "schema": "PF_ELASTIC_MULTI_PARTICLE_DYNAMIC_QUALIFICATION_V2_V3_V4_V5_CONDITIONAL_V1", "fixture_schema": schema, "status": status,
        "fixture_manifest_sha256": fixture_sha256, "checkpoints": {key: base.sha256(value["path"]) for key, value in probes.items()},
        "gates": gates, "metrics": {"mass_relative": mass_rel, "dt_phi_L1": phi_l1, "dt_xB_mean_absolute": xb_mae, "dt_xB_alpha_gt_1e_2_mean_absolute": xb_mae_alpha_gt_1e_2, "dt_xB_alpha_gt_1e_1_mean_absolute": xb_mae_alpha_gt_1e_1, "dt_xB_far_matrix_mean_absolute": xb_mae_far_matrix, "dt_xB_preferred": xb_mae <= 2.0e-5, "dt_xB_within_historical_hard_threshold": xb_mae <= 5.0e-5, "dt_xB_blocking": not conditional, "dt_axis_relative": axis_rel, "restart_raw_equal": restart_raw_equal, "restart_fields_equal": restart_fields_equal, "initial_profile_changed_after_one_step": first_step_profile_changed},
        "states": {name: {"step": probes[name]["step"], "dt": probes[name]["dt"], "component_count": len(states[name]["rows"]), "identity_mapping": states[name]["mapping"], "overlap": states[name]["overlap"]} for name in states},
    }
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    terminal = {
        "handoff_status": status, "component_identity_status": gates["component_identity"], "restart_status": gates["restart_bytewise"],
        "dt_refinement_status": gates["dt_phi_L1"] and gates["dt_axis"] and gates["dt_xB_hard"], "zero_mode_status": gates["zero_mode"],
        "raw_field_clipping_status": gates["no_clipping"], "mass_relative": mass_rel, "dt_phi_L1": phi_l1,
        "dt_xB_mean_absolute": xb_mae, "dt_xB_alpha_gt_1e_2_mean_absolute": xb_mae_alpha_gt_1e_2, "dt_xB_alpha_gt_1e_1_mean_absolute": xb_mae_alpha_gt_1e_1, "dt_xB_far_matrix_mean_absolute": xb_mae_far_matrix, "dt_axis_relative": axis_rel,
    }
    if conditional:
        terminal.update({
            "full_field_xB_dt_MAE_blocking": False,
            "initial_profile_free_after_t0": first_step_profile_changed,
            "checkpoint_initial_state_provenance": gates[
                "checkpoint_initial_state_provenance"
            ],
        })
    (args.out / "final_terminal_output.txt").write_text("\n".join(f"{key}={value}" for key, value in terminal.items()) + "\n", encoding="utf-8")
    print(status)


if __name__ == "__main__":
    main()
