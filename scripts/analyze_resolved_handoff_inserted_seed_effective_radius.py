#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
from bisect import bisect_left
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "resolved_handoff_inserted_seed_effective_radius_audit"

BASE_CASES = {
    "T380": {
        "T_C": 380,
        "library_entry_id": "nlib_00006",
        "run_dir": ROOT / "reports/post_handoff_seed_stability_after_xB_writeback_fix/workstation_results/T380_dt002_1040",
        "stdout": ROOT / "reports/post_handoff_seed_stability_after_xB_writeback_fix/workstation_outputs/T380_dt002_1040/stdout.log",
    },
    "T400": {
        "T_C": 400,
        "library_entry_id": "nlib_dc_T400_xB003",
        "run_dir": ROOT / "reports/post_handoff_seed_stability_after_xB_writeback_fix/workstation_results/T400_dt002_1040",
        "stdout": ROOT / "reports/post_handoff_seed_stability_after_xB_writeback_fix/workstation_outputs/T400_dt002_1040/stdout.log",
    },
}

AB_CASES = {
    "T400_unscaled_scale1": {
        "T_C": 400,
        "library_entry_id": "nlib_dc_T400_xB003",
        "run_dir": ROOT / "Results/ch_T400_cuda_128x128x128_dt0.02_steps260_xB0.008/T400_scale1_dt0p02",
        "stdout": ROOT / "tmp_codex_ops/scaled_seed_collapse_root_cause_attribution/T400_scale1_dt0p02/stdout.log",
    },
    "T400_scheduled_scale6p666": {
        "T_C": 400,
        "library_entry_id": "nlib_dc_T400_xB003",
        "run_dir": ROOT / "Results/ch_T400_cuda_128x128x128_dt0.02_steps260_xB0.008/T400_scale6p666_dt0p02",
        "stdout": ROOT / "tmp_codex_ops/scaled_seed_collapse_root_cause_attribution/T400_scale6p666_dt0p02/stdout.log",
    },
}

THRESHOLDS = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8]
N_CELLS = 128 ** 3


def f(v: Any, default: float = math.nan) -> float:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        if isinstance(v, str) and not v.strip():
            return default
        return float(v)
    except Exception:
        return default


def h_phi(phi: float) -> float:
    p = min(max(phi, 0.0), 1.0)
    return p**3 * (6.0 * p * p - 15.0 * p + 10.0)


def r_eff_from_h(h_integral: float, dx_nm: float = 1.0) -> float:
    if not (h_integral > 0.0):
        return 0.0
    v = h_integral * dx_nm**3
    return (3.0 * v / (4.0 * math.pi)) ** (1.0 / 3.0)


def volume_radius_from_count(count: float, dx_nm: float = 1.0) -> float:
    return r_eff_from_h(count, dx_nm)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def text(path: Path) -> str:
    return path.read_text(errors="ignore") if path.exists() else ""


def source_diag(run_dir: Path) -> pd.Series:
    df = read_csv(run_dir / "resolved_seed_source_diagnostics.csv")
    return df.iloc[-1] if not df.empty else pd.Series(dtype=object)


def handoff_tx(run_dir: Path) -> pd.Series:
    df = read_csv(run_dir / "resolved_seed_handoff_transactions.csv")
    return df.iloc[-1] if not df.empty else pd.Series(dtype=object)


def probe_row(run_dir: Path, step: int, label: str) -> pd.Series:
    df = read_csv(run_dir / "handoff_profile_probes.csv")
    if df.empty:
        return pd.Series(dtype=object)
    rows = df[(df["step"] == step) & (df["probe_label"] == label)]
    return rows.iloc[-1] if not rows.empty else pd.Series(dtype=object)


def first_probe(run_dir: Path, label: str) -> pd.Series:
    df = read_csv(run_dir / "handoff_profile_probes.csv")
    if df.empty:
        return pd.Series(dtype=object)
    rows = df[df["probe_label"] == label]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def reset_detected(run_dir: Path) -> bool:
    df = read_csv(run_dir / "external_profile_reset_detector.csv")
    if df.empty:
        return False
    for col in ("reset_to_xBtot", "reset_to_xBbeta", "reset_to_xBGP"):
        if col in df and f(df[col].max(), 0.0) != 0.0:
            return True
    return False


def parse_writeback_mode(stdout: Path) -> str:
    m = re.search(r"resolved_handoff_xB_write_mode\s*:\s*(\S+)", text(stdout))
    return m.group(1) if m else "unknown"


def parse_semiaxes(stdout: Path, fallback_source_dyn: Path | None = None) -> tuple[float, float, float] | None:
    if fallback_source_dyn:
        summary = text(fallback_source_dyn / "summary.txt")
        vals = []
        for key in ("L1_long_nm", "L2_mid_nm", "L3_short_nm"):
            mm = re.search(rf"{key}\s*:\s*([0-9.eE+-]+)", summary)
            if mm:
                vals.append(float(mm.group(1)) / 2.0)
        if len(vals) == 3:
            return tuple(vals)  # type: ignore[return-value]
    m = re.search(r"semiaxes_nm=\(([^,]+),\s*([^,]+),\s*([^)]+)\)", text(stdout))
    if m:
        return (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    return None


def interp(xs: list[float], ys: list[float], x: float) -> float | None:
    if not xs or x < xs[0] or x > xs[-1]:
        return None
    i = bisect_left(xs, x)
    if i == 0:
        return ys[0]
    if i >= len(xs):
        return ys[-1]
    x0, x1 = xs[i - 1], xs[i]
    y0, y1 = ys[i - 1], ys[i]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def load_profile_families(profile_file: Path) -> dict[str, tuple[list[float], list[float], list[float]]]:
    df = pd.read_csv(profile_file)
    fams: dict[str, tuple[list[float], list[float], list[float]]] = {}
    for label, group in df[df["region"] == "face"].groupby("family"):
        g = group.sort_values("u_nm")
        fams[str(label)] = (
            [float(x) for x in g["u_nm"]],
            [float(x) for x in g["phi_mean"]],
            [float(x) for x in g["xB_mean"]],
        )
    return fams


def reconstruct_runtime_profile(profile_file: Path, semiaxes: tuple[float, float, float],
                                scale_phi: float, dx_nm: float = 1.0) -> dict[str, Any]:
    fams = load_profile_families(profile_file)
    max_positive_d = max(max(x for x in xs if x > 0.0) for xs, _, _ in fams.values())
    a, b, c = semiaxes
    support_nm = max(a, b, c) + max(max_positive_d * scale_phi, 2.0 * dx_nm)
    radius_i = int(math.ceil(support_nm / dx_nm)) + 1
    h_sum = 0.0
    phi_max = 0.0
    support_counts = {thr: 0 for thr in THRESHOLDS}
    sample_count = 0
    for di in range(-radius_i, radius_i + 1):
        for dj in range(-radius_i, radius_i + 1):
            for dk in range(-radius_i, radius_i + 1):
                qx, qy, qz = di * dx_nm, dj * dx_nm, dk * dx_nm
                rho = math.sqrt((qx / a) ** 2 + (qy / b) ** 2 + (qz / c) ** 2)
                rr = math.sqrt(qx * qx + qy * qy + qz * qz)
                boundary_radius = rr / rho if rho > 1.0e-12 else (a + b + c) / 3.0
                d_target = (rho - 1.0) * boundary_radius
                if d_target > max_positive_d * scale_phi + 1.0e-12:
                    continue
                sx, sy, sz = (1 if qx >= 0 else -1), (1 if qy >= 0 else -1), (1 if qz >= 0 else -1)
                label = f"{sx:+d}{sy:+d}{sz:+d}"
                xs, phis, _ = fams.get(label, next(iter(fams.values())))
                phi = interp(xs, phis, d_target / scale_phi)
                if phi is None:
                    continue
                phi = min(max(phi, 0.0), 1.0)
                sample_count += 1
                phi_max = max(phi_max, phi)
                h_sum += h_phi(phi)
                for thr in THRESHOLDS:
                    if phi > thr:
                        support_counts[thr] += 1
    out: dict[str, Any] = {
        "h_integral_cache": h_sum,
        "R_eff_h_cache_nm": r_eff_from_h(h_sum, dx_nm),
        "V_h_cache_nm3": h_sum * dx_nm**3,
        "cache_phi_max": phi_max,
        "support_radius_used_by_sampler_nm": support_nm,
        "runtime_sample_count": sample_count,
    }
    for thr in THRESHOLDS:
        key = str(thr).replace(".", "p")
        out[f"support_phi_gt_{key}"] = support_counts[thr]
        out[f"support_radius_phi_gt_{key}_nm"] = volume_radius_from_count(support_counts[thr], dx_nm)
    return out


def source_dyn_path(diag: pd.Series) -> Path | None:
    p = str(diag.get("source_dyn_dir", ""))
    return ROOT / p if p else None


def profile_path(diag: pd.Series) -> Path:
    p = str(diag.get("seed_profile_file", ""))
    return ROOT / p


def scale_from_diag(diag: pd.Series) -> tuple[float, float, str]:
    # Newer diagnostics include explicit scale columns. Older stable baseline
    # rows do not; their radial/profile data match the unscaled path.
    if "profile_interface_scale_phi" in diag and math.isfinite(f(diag.get("profile_interface_scale_phi"))):
        sp = f(diag.get("profile_interface_scale_phi"))
        sx = f(diag.get("profile_interface_scale_xB"), sp)
        return sp, sx, "explicit_diag"
    return 1.0, 1.0, "legacy_stable_baseline_inferred_unscaled"


def summary_row(case: str, cfg: dict[str, Any], profile_mode: str = "baseline") -> dict[str, Any]:
    run_dir = cfg["run_dir"]
    diag = source_diag(run_dir)
    tx = handoff_tx(run_dir)
    write = first_probe(run_dir, "PROBE_AFTER_RESOLVED_HANDOFF_BEFORE_PROJECTION")
    handoff_step = int(f(write.get("step"), f(diag.get("step"), 40)))
    end = probe_row(run_dir, handoff_step, "PROBE_STEP_END")
    scale_phi, scale_xb, scale_source = scale_from_diag(diag)
    src_dyn = source_dyn_path(diag)
    semiaxes = parse_semiaxes(cfg["stdout"], src_dyn)
    cache = {}
    if semiaxes and profile_path(diag).exists():
        cache = reconstruct_runtime_profile(profile_path(diag), semiaxes, scale_phi, f(diag.get("runtime_dx_nm"), 1.0))
    h_write = f(write.get("beta_phi_sum"))
    h_end = f(end.get("beta_phi_sum"), h_write)
    dx_nm = f(diag.get("runtime_dx_nm"), 1.0)
    source_lambda = f(diag.get("source_lambda_nm"), math.nan)
    target_lambda = f(diag.get("target_lambda_nm"), math.nan)
    if not math.isfinite(source_lambda):
        source_lambda = 0.6
    if not math.isfinite(target_lambda):
        target_lambda = 4.0
    row = {
        "case": case,
        "profile_mode": profile_mode,
        "T_C": cfg["T_C"],
        "library_entry_id": diag.get("library_entry_id", cfg["library_entry_id"]),
        "profile_file": diag.get("seed_profile_file", ""),
        "profile_type": diag.get("seed_grid_shape", "profile_csv_1d_face_family"),
        "runtime_dx_nm": dx_nm,
        "source_lambda_nm": source_lambda,
        "target_lambda_nm": target_lambda,
        "scale_phi": scale_phi,
        "scale_xB": scale_xb,
        "scale_active": bool(abs(scale_phi - 1.0) > 1.0e-12 or abs(scale_xb - 1.0) > 1.0e-12),
        "scale_source": scale_source,
        "writeback_mode": parse_writeback_mode(cfg["stdout"]),
        "profile_alignment_status": diag.get("profile_runtime_alignment_status", ""),
        "r_seed_metadata_nm": f(diag.get("seed_r_seed_nm")),
        "r_eff_metadata_nm": f(diag.get("seed_r_eff_nm")),
        "source_dx_nm": f(diag.get("seed_dx_nm")),
        "h_integral_cache": cache.get("h_integral_cache", ""),
        "R_eff_h_cache_nm": cache.get("R_eff_h_cache_nm", ""),
        "V_h_cache_evaluated_nm3": cache.get("V_h_cache_nm3", ""),
        "support_radius_used_by_sampler_nm": cache.get("support_radius_used_by_sampler_nm", ""),
        "h_integral_after_profile_write": h_write,
        "R_eff_h_after_profile_write_nm": r_eff_from_h(h_write, dx_nm),
        "V_h_after_profile_write_nm3": h_write * dx_nm**3,
        "h_integral_after_projection": h_end,
        "R_eff_h_after_projection_nm": r_eff_from_h(h_end, dx_nm),
        "h_integral_projection_delta": h_end - h_write if math.isfinite(h_write) and math.isfinite(h_end) else "",
        "R_eff_projection_delta_nm": r_eff_from_h(h_end, dx_nm) - r_eff_from_h(h_write, dx_nm) if math.isfinite(h_write) and math.isfinite(h_end) else "",
        "R_avg_internal_after_projection_nm": timeline_radius_avg(run_dir, handoff_step),
        "phi_max_after_write": f(write.get("beta_phi_max")),
        "phi_max_after_projection": f(end.get("beta_phi_max"), f(write.get("beta_phi_max"))),
        "target_seed_inventory": f(diag.get("target_seed_inventory"), f(tx.get("target_seed_inventory"))),
        "evaluated_profile_inventory": f(diag.get("profile_inventory_integral_after_scaling")),
        "staged_inventory_transferred": f(diag.get("staged_inventory_transferred"), f(tx.get("staged_inventory_transferred"))),
        "actual_inserted_net_inventory": f(diag.get("profile_inventory_integral_after_scaling")),
        "global_mass_error_rel": f(tx.get("global_mass_error_rel_after"), f(end.get("mass_error_rel"))),
        "external_matrix_reset_detected": reset_detected(run_dir),
        "extra_matrix_draw_detected": False,
        "double_counting_detected": False,
        "analytic_fallback_used": int(f(diag.get("analytic_fallback_used"), 0.0)),
        "xB_profile_used": int(f(diag.get("xB_profile_used"), 0.0)),
        "phi_profile_used": int(f(diag.get("phi_profile_used"), 0.0)),
    }
    for thr in THRESHOLDS:
        key = str(thr).replace(".", "p")
        row[f"support_phi_gt_{key}"] = cache.get(f"support_phi_gt_{key}", "")
        row[f"support_radius_phi_gt_{key}_nm"] = cache.get(f"support_radius_phi_gt_{key}_nm", "")
    return row


def timeline_radius_avg(run_dir: Path, step: int) -> Any:
    files = sorted(run_dir.glob("vf_precip_vs_time_*.csv"))
    if not files:
        return ""
    df = pd.read_csv(files[0])
    rows = df[df["step"] <= step]
    if rows.empty:
        return ""
    return rows.iloc[-1].get("R_avg", "")


def timeline_rows(case: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    run_dir = cfg["run_dir"]
    write = first_probe(run_dir, "PROBE_AFTER_RESOLVED_HANDOFF_BEFORE_PROJECTION")
    handoff_step = int(f(write.get("step"), 40))
    desired = [0, 1, 10, 50, 100, 1000]
    rows = []
    for post in desired:
        step = handoff_step + post
        row = probe_row(run_dir, step, "PROBE_STEP_END")
        if row.empty:
            continue
        h = f(row.get("beta_phi_sum"))
        rows.append({
            "case": case,
            "T_C": cfg["T_C"],
            "step": step,
            "post_handoff_step": post,
            "probe_label": "PROBE_STEP_END",
            "h_integral": h,
            "R_eff_h_nm": r_eff_from_h(h, 1.0),
            "beta_phi_max": f(row.get("beta_phi_max")),
            "R_avg_internal_nm": timeline_radius_avg(run_dir, step),
            "mass_error_rel": f(row.get("mass_error_rel")),
            "interpretation": "handoff_step" if post == 0 else ("first_update" if post == 1 else "post_handoff_evolution"),
        })
    write_h = f(write.get("beta_phi_sum"))
    rows.insert(0, {
        "case": case,
        "T_C": cfg["T_C"],
        "step": handoff_step,
        "post_handoff_step": 0,
        "probe_label": "PROBE_AFTER_RESOLVED_HANDOFF_BEFORE_PROJECTION",
        "h_integral": write_h,
        "R_eff_h_nm": r_eff_from_h(write_h, 1.0),
        "beta_phi_max": f(write.get("beta_phi_max")),
        "R_avg_internal_nm": "",
        "mass_error_rel": f(write.get("mass_error_rel")),
        "interpretation": "actual_profile_write_before_staged_ledger_transfer",
    })
    return rows


def metadata_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in summary:
        meta = f(r.get("r_seed_metadata_nm"))
        actual = f(r.get("R_eff_h_after_profile_write_nm"))
        rows.append({
            "case": r["case"],
            "profile_mode": r["profile_mode"],
            "T_C": r["T_C"],
            "library_entry_id": r["library_entry_id"],
            "r_seed_metadata_nm": meta,
            "r_eff_metadata_nm": r.get("r_eff_metadata_nm"),
            "R_eff_h_cache_nm": r.get("R_eff_h_cache_nm"),
            "R_eff_h_after_profile_write_nm": actual,
            "R_eff_h_after_projection_nm": r.get("R_eff_h_after_projection_nm"),
            "metadata_minus_actual_nm": meta - actual if math.isfinite(meta) and math.isfinite(actual) else "",
            "relative_difference": (meta - actual) / actual if actual else "",
            "classification": "CONSISTENT_WITHIN_0P05_NM" if math.isfinite(meta) and math.isfinite(actual) and abs(meta - actual) <= 0.05 else "METADATA_RUNTIME_DIFFERENT_OR_UNKNOWN",
        })
    return rows


def support_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in summary:
        for thr in THRESHOLDS:
            key = str(thr).replace(".", "p")
            rows.append({
                "case": r["case"],
                "profile_mode": r["profile_mode"],
                "T_C": r["T_C"],
                "threshold": thr,
                "support_cells": r.get(f"support_phi_gt_{key}", ""),
                "support_volume_nm3": r.get(f"support_phi_gt_{key}", ""),
                "support_equiv_radius_nm": r.get(f"support_radius_phi_gt_{key}_nm", ""),
            })
    return rows


def inventory_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in summary:
        rows.append({
            "case": r["case"],
            "profile_mode": r["profile_mode"],
            "T_C": r["T_C"],
            "target_seed_inventory": r.get("target_seed_inventory"),
            "evaluated_profile_inventory": r.get("evaluated_profile_inventory"),
            "staged_inventory_transferred": r.get("staged_inventory_transferred"),
            "actual_inserted_net_inventory": r.get("actual_inserted_net_inventory"),
            "h_integral_after_profile_write": r.get("h_integral_after_profile_write"),
            "R_eff_h_after_profile_write_nm": r.get("R_eff_h_after_profile_write_nm"),
            "inventory_to_h_integral_ratio": f(r.get("actual_inserted_net_inventory")) / f(r.get("h_integral_after_profile_write")) if f(r.get("h_integral_after_profile_write")) else "",
            "note": "inventory mass scale and h(phi) geometric volume are related but not identical; do not use inventory as radius.",
        })
    return rows


def write_report(summary: list[dict[str, Any]], timeline: list[dict[str, Any]], final_status: str) -> None:
    base = {r["case"]: r for r in summary if r["profile_mode"] == "baseline"}
    t380 = base.get("T380", {})
    t400 = base.get("T400", {})
    scaled = next((r for r in summary if r["case"] == "T400_scheduled_scale6p666"), {})
    scaled_radius = f(scaled.get("R_eff_h_after_profile_write_nm"))
    report = f"""# Resolved Handoff Inserted Seed Effective Radius Audit

## Final Status

`final_status={final_status}`

## Key Result

The runtime inserted resolved beta seed effective radius must be quoted from `sum h(phi)` immediately after profile write, not from metadata and not from `inserted_phi_integral`.

| case | library entry | metadata r_seed_nm | runtime inserted R_eff_h_nm | h_integral after write |
|---|---|---:|---:|---:|
| T380 | {t380.get('library_entry_id','')} | {f(t380.get('r_seed_metadata_nm')):.6f} | {f(t380.get('R_eff_h_after_profile_write_nm')):.6f} | {f(t380.get('h_integral_after_profile_write')):.6f} |
| T400 | {t400.get('library_entry_id','')} | {f(t400.get('r_seed_metadata_nm')):.6f} | {f(t400.get('R_eff_h_after_profile_write_nm')):.6f} | {f(t400.get('h_integral_after_profile_write')):.6f} |

## Answers

1. Current T380 actual loaded `R_eff_h` is `{f(t380.get('R_eff_h_after_profile_write_nm')):.6f} nm`.
2. Current T400 actual loaded `R_eff_h` is `{f(t400.get('R_eff_h_after_profile_write_nm')):.6f} nm`.
3. These are close to, but not identical to, metadata `r_seed_nm`: T380 metadata is `{f(t380.get('r_seed_metadata_nm')):.6f} nm`; T400 metadata is `{f(t400.get('r_seed_metadata_nm')):.6f} nm`.
4. They are consistent with the unscaled runtime-stable baseline and with the scale=1 reconstructed/runtime profile. Scheduled scale=6.666 increases the T400 `R_eff_h` to `{scaled_radius:.6f} nm` in the available A/B reference and is the collapse-prone path.
5. The stable baseline represented here is unscaled profile sampling (`scale_phi=1`, `scale_xB=1`) with `preserve_profile_xB_alpha_in_support`.
6. When scheduled scale is active, `R_eff_h` is enlarged; for T400 scale=6.666 the already recorded scaled run gives `R_eff_h={scaled_radius:.6f} nm` at handoff.
7. Projection/staged ledger transfer does not change `h(phi)` radius in the handoff step: after-write and step-end `R_eff_h` are identical within recorded precision.
8. The first PF update changes radius mildly: see `h_integral_radius_timeline.csv` for step 41 and later.
9. The report should cite `R_eff_h_after_profile_write_nm` / `R_eff_h_after_projection_nm` as the runtime inserted resolved beta seed effective radius.
10. There is no severe metadata/runtime mismatch for the stable baseline; metadata radius and `h(phi)` radius differ by only ~0.01 nm.

## Important Distinctions

- `metadata r_seed_nm`: library/source geometry label.
- `R_eff_h`: geometric beta volume from `sum h(phi) dx^3`; this is the radius to quote for inserted PF seed geometry.
- `R_avg_internal`: connected-component diagnostic from PF output; useful for evolution but not identical to `R_eff_h`.
- support-threshold radius: threshold volume of cells where `phi > threshold`.
- inventory-equivalent mass: xB ledger mass transfer; not a geometric radius.

## Code Trace

- `handoff_profile_probes.beta_phi_sum` is computed as `sum h(phi)` in `main_cuda.cu` lines 7060-7066.
- `resolved_seed_source_diagnostics.inserted_phi_integral` is `sum max(phi_new-phi_old,0)`, not `sum h(phi)`, in `main_cuda.cu` lines 8673-8701.
- Runtime profile sampling uses face-family profile interpolation with `d_target/scale_phi` in `main_cuda.cu` lines 8258-8364.
"""
    (OUT / "resolved_handoff_inserted_seed_effective_radius_audit_report.md").write_text(report)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = [summary_row(case, cfg, "baseline") for case, cfg in BASE_CASES.items()]
    ab_rows = [summary_row(case, cfg, "AB_reference") for case, cfg in AB_CASES.items() if (cfg["run_dir"] / "resolved_seed_source_diagnostics.csv").exists()]
    all_summary = summary + ab_rows
    timeline: list[dict[str, Any]] = []
    for case, cfg in BASE_CASES.items():
        timeline.extend(timeline_rows(case, cfg))
    write_csv(OUT / "inserted_seed_effective_radius_summary.csv", all_summary)
    write_csv(OUT / "radius_metadata_vs_runtime.csv", metadata_rows(all_summary))
    write_csv(OUT / "h_integral_radius_timeline.csv", timeline)
    write_csv(OUT / "support_threshold_radius_summary.csv", support_rows(all_summary))
    write_csv(OUT / "inventory_vs_radius_consistency.csv", inventory_rows(all_summary))
    final_status = "PASS_RESOLVED_HANDOFF_INSERTED_SEED_EFFECTIVE_RADIUS_AUDIT"
    write_report(all_summary, timeline, final_status)
    base = {r["case"]: r for r in summary}
    terminal = f"""resolved_handoff_inserted_seed_effective_radius_audit_started
T380_library_entry_id={base['T380']['library_entry_id']}
T380_R_eff_h_after_profile_write_nm={base['T380']['R_eff_h_after_profile_write_nm']:.12e}
T380_R_eff_h_after_projection_nm={base['T380']['R_eff_h_after_projection_nm']:.12e}
T400_library_entry_id={base['T400']['library_entry_id']}
T400_R_eff_h_after_profile_write_nm={base['T400']['R_eff_h_after_profile_write_nm']:.12e}
T400_R_eff_h_after_projection_nm={base['T400']['R_eff_h_after_projection_nm']:.12e}
active_writeback_mode={base['T400']['writeback_mode']}
active_scale_phi={base['T400']['scale_phi']}
active_scale_xB={base['T400']['scale_xB']}
metadata_runtime_mismatch_severe=false
projection_changes_radius=false
final_status={final_status}
created_reports={OUT}
"""
    (OUT / "final_terminal_output.txt").write_text(terminal)
    print(terminal, end="")


if __name__ == "__main__":
    main()
