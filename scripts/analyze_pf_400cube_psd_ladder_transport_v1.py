#!/usr/bin/env python3
"""No-dislocation 300 K transport audit for one 400^3 PSD ladder trajectory.

It reports a conditional Yu--Debye response of the resolved PF population.
It deliberately does not fit a scale factor, infer a dislocation density, or
claim an absolute reconstruction of an experimental conductivity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PASS = "PASS_400CUBE_PSD_LADDER_TRANSPORT_V1"
TEMPERATURE_K = 300.0
BOX_VOLUME_M3 = (400.0e-9) ** 3
MODELS = ("Nv_plus_mean_R", "Sv_geometric_limit", "Sv_plus_M6", "full_PSD")
MATRIX_MODES = ("fixed_6h_matrix", "pf_time_varying_matrix")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    if not values:
        raise ValueError(f"refusing empty CSV: {path}")
    fields = list(values[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(values)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def integrate(transport: Any, config: dict[str, Any], matrix_xag: float, radii_m: np.ndarray, model: str) -> float:
    return float(transport.integrate_kappa_gauss(TEMPERATURE_K, matrix_xag, radii_m, BOX_VOLUME_M3, config, {
        "Nv_plus_mean_R": "Nv_plus_mean_R_monodisperse",
        "Sv_geometric_limit": "Sv_geometric_limit",
        "Sv_plus_M6": "Sv_plus_M6_moment_reconstruction",
        "full_PSD": "full_psd",
    }[model]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observables", required=True, type=Path)
    parser.add_argument("--particles", required=True, type=Path)
    parser.add_argument("--yu-config", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--transport-module", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)
    try:
        config = json.loads(args.yu_config.read_text(encoding="utf-8"))
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        if contract.get("status") != "FROZEN" or any(contract["disabled_inputs"].get(key) is not True for key in ("S11_dislocation_core", "S13_dislocation_strain", "yu_refit_scale_0p1172768")):
            raise ValueError("no-dislocation transport contract is not frozen")
        transport = load_module("pf400_transport", args.transport_module)
        transport.validate_yu_base_config(config, args.yu_config)
        transport.validate_interface_contract(contract, args.contract, args.yu_config)
        observed = rows(args.observables)
        particle_rows = rows(args.particles)
        by_step: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in particle_rows:
            by_step[int(row["step"])].append(row)
        if not observed or int(observed[0]["step"]) != 0 or int(observed[-1]["step"]) != 152585:
            raise ValueError("observations do not span 6 h to 48 h")
        fixed_matrix = float(observed[0]["far_field_matrix_xAg"])
        predictions: list[dict[str, Any]] = []
        descriptors: list[dict[str, Any]] = []
        reference: dict[tuple[str, str], float] = {}
        for row in observed:
            step = int(row["step"])
            group = by_step.get(step, [])
            if len(group) != int(row["particle_count"]):
                raise ValueError(f"particle count/PSD mismatch at step {step}")
            radii_nm = np.asarray(sorted(float(item["equivalent_radius_nm"]) for item in group), dtype=float)
            if np.any(~np.isfinite(radii_nm)) or np.any(radii_nm <= 0.0):
                raise ValueError(f"invalid PSD at step {step}")
            moments = transport.moments_from_radii(radii_nm * 1.0e-9, BOX_VOLUME_M3)
            descriptors.append({"step": step, "age_h": float(row["experimental_age_h"]), "particle_count": len(radii_nm), "Nv_m-3": moments["Nv_m-3"], "mean_radius_nm": float(np.mean(radii_nm)), "std_radius_nm": float(np.std(radii_nm)), "Sv_nm-1": moments["Sv_m-1"] * 1.0e-9, "M6_nm3": moments["M6_m3"] * 1.0e27, "matrix_xAg": float(row["far_field_matrix_xAg"]), "full_psd_sha256": transport.canonical_sha256(radii_nm.tolist())})
            for mode in MATRIX_MODES:
                xag = fixed_matrix if mode == "fixed_6h_matrix" else float(row["far_field_matrix_xAg"])
                values = {model: integrate(transport, config, xag, radii_nm * 1.0e-9, model) for model in MODELS}
                for model, value in values.items():
                    key = (mode, model)
                    if step == 0:
                        reference[key] = value
                    predictions.append({"step": step, "age_h": float(row["experimental_age_h"]), "matrix_mode": mode, "matrix_xAg_used": xag, "descriptor_model": model, "kappa_W_mK": value, "full_PSD_kappa_W_mK": values["full_PSD"], "kappa_relative_error": abs(value - values["full_PSD"]) / max(abs(values["full_PSD"]), np.finfo(float).tiny), "delta_kappa_from_6h_W_mK": value - reference[key], "full_PSD_delta_kappa_from_6h_W_mK": values["full_PSD"] - reference[(mode, "full_PSD")]})
        for row in predictions:
            row["delta_kappa_error_W_mK"] = row["delta_kappa_from_6h_W_mK"] - row["full_PSD_delta_kappa_from_6h_W_mK"]
        summary: list[dict[str, Any]] = []
        for mode in MATRIX_MODES:
            for model in MODELS:
                data = [row for row in predictions if row["matrix_mode"] == mode and row["descriptor_model"] == model]
                err = np.asarray([float(row["kappa_relative_error"]) for row in data])
                delta = np.asarray([float(row["delta_kappa_error_W_mK"]) for row in data if int(row["step"]) != 0])
                signal = np.asarray([float(row["full_PSD_delta_kappa_from_6h_W_mK"]) for row in data if int(row["step"]) != 0])
                summary.append({"matrix_mode": mode, "descriptor_model": model, "kappa_MAPE_percent": 100.0 * float(np.mean(err)), "kappa_max_relative_percent": 100.0 * float(np.max(err)), "delta_kappa_signal_NRMSE_percent": 100.0 * float(np.sqrt(np.mean(delta**2)) / np.sqrt(np.mean(signal**2))) if np.any(np.abs(signal) > 0.0) else 0.0})
        write_csv(args.out / "hourly_full_psd_descriptors.csv", descriptors)
        write_csv(args.out / "transport_predictions_300K.csv", predictions)
        write_csv(args.out / "descriptor_sufficiency_300K.csv", summary)
        for mode in MATRIX_MODES:
            data = [row for row in predictions if row["matrix_mode"] == mode and row["descriptor_model"] == "full_PSD"]
            plt.plot([row["age_h"] for row in data], [row["kappa_W_mK"] for row in data], marker="o", ms=2.5, label=mode)
        plt.xlabel("aging time (h)"); plt.ylabel("conditional κL,no-dis (W m⁻¹ K⁻¹)"); plt.legend(); plt.tight_layout(); plt.savefig(args.out / "kappa_300K_6h_to_48h.png", dpi=180); plt.close()
        first = [row for row in particle_rows if int(row["step"]) == 0]
        last = [row for row in particle_rows if int(row["step"]) == 152585]
        plt.hist([float(row["equivalent_radius_nm"]) for row in first], bins=16, alpha=0.6, label="6 h")
        plt.hist([float(row["equivalent_radius_nm"]) for row in last], bins=16, alpha=0.6, label="48 h")
        plt.xlabel("equivalent radius (nm)"); plt.ylabel("particle count"); plt.legend(); plt.tight_layout(); plt.savefig(args.out / "psd_6h_vs_48h.png", dpi=180); plt.close()
        endpoint = {(row["matrix_mode"], row["descriptor_model"], int(row["step"])): row for row in predictions}
        fixed6 = endpoint[("fixed_6h_matrix", "full_PSD", 0)]
        fixed48 = endpoint[("fixed_6h_matrix", "full_PSD", 152585)]
        delta = float(fixed48["kappa_W_mK"]) - float(fixed6["kappa_W_mK"])
        sign = "POSITIVE" if delta > 0.0 else "NEGATIVE" if delta < 0.0 else "ZERO_WITHIN_NUMERIC_PRECISION"
        provenance = {"schema": "PF_400CUBE_PSD_LADDER_TRANSPORT_V1", "status": PASS, "temperature_K": TEMPERATURE_K, "fixed_matrix_xAg": fixed_matrix, "disabled_dislocation_terms": True, "forbidden_refit_scale_used": False, "absolute_experimental_kappa_claim": False, "input_sha256": {str(path): sha256(path) for path in (args.observables, args.particles, args.yu_config, args.contract, args.transport_module)}}
        (args.out / "transport_manifest.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (args.out / "status.txt").write_text(PASS + "\n", encoding="utf-8")
        (args.out / "final_terminal_output.txt").write_text(f"transport_status={PASS}\ndelta_kappa_PSD_300K_W_mK={delta:.17g}\ndelta_kappa_PSD_300K_sign={sign}\nabsolute_experimental_kappa_claim=false\n", encoding="utf-8")
        print(PASS)
    except Exception as exc:
        (args.out / "status.txt").write_text("BLOCKED_400CUBE_PSD_LADDER_TRANSPORT_V1\n", encoding="utf-8")
        (args.out / "first_failure.csv").write_text("stage,detail\ntransport," + str(exc).replace(",", ";").replace("\n", " ") + "\n", encoding="utf-8")
        raise SystemExit(f"[fatal] {exc}") from exc


if __name__ == "__main__":
    main()
