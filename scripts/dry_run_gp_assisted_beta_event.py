#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "gp_assisted_beta_mass_ledger"


@dataclass
class GPSite:
    id: int
    center: tuple[int, int, int]
    active: bool
    consumed: bool
    m_B_active: float
    release_mode: str


def h_smooth(r: float, radius: float, width: float) -> float:
    return max(0.0, min(1.0, 0.5 * (1.0 - math.tanh((r - radius) / max(width, 1e-12)))))


def periodic_delta(i: int, c: int, n: int) -> int:
    d = i - c
    if d > n // 2:
        d -= n
    if d < -n // 2:
        d += n
    return d


def iter_cells(n: int):
    for i in range(n):
        for j in range(n):
            for k in range(n):
                yield i, j, k


def idx3(i: int, j: int, k: int, n: int) -> int:
    return (i * n + j) * n + k


def kernel_weights(kind: str, n: int, center: tuple[int, int, int], radius_cells: int) -> list[tuple[int, float]]:
    raw: list[tuple[int, float]] = []
    if kind == "nearest_cell":
        return [(idx3(*center, n), 1.0)]
    sigma = max(radius_cells / 2.0, 1e-12)
    for i, j, k in iter_cells(n):
        dx = periodic_delta(i, center[0], n)
        dy = periodic_delta(j, center[1], n)
        dz = periodic_delta(k, center[2], n)
        r2 = dx * dx + dy * dy + dz * dz
        if kind == "compact_spherical":
            if r2 <= radius_cells * radius_cells:
                raw.append((idx3(i, j, k, n), 1.0))
        elif kind == "gaussian_normalized":
            if r2 <= radius_cells * radius_cells:
                raw.append((idx3(i, j, k, n), math.exp(-0.5 * r2 / (sigma * sigma))))
        else:
            raise ValueError(f"unknown kernel_type={kind}")
    total = sum(w for _, w in raw)
    if total <= 0:
        raise RuntimeError("release kernel has no active cells")
    return [(idx, w / total) for idx, w in raw]


def weighted_add_with_redistribution(
    matrix: list[float],
    weights: list[tuple[int, float]],
    mass: float,
    x_max: float,
) -> tuple[float, int, float]:
    remaining_mass = mass
    active = list(weights)
    injected = 0.0
    clipped_count = 0
    redistributed = 0.0
    for _ in range(64):
        if remaining_mass <= 1e-14:
            break
        denom = sum(w for idx, w in active if matrix[idx] < x_max - 1e-15)
        if denom <= 0:
            break
        next_active: list[tuple[int, float]] = []
        still_remaining = 0.0
        for idx, w in active:
            if matrix[idx] >= x_max - 1e-15:
                continue
            add = remaining_mass * w / denom
            cap = x_max - matrix[idx]
            if add <= cap + 1e-15:
                matrix[idx] += add
                injected += add
                next_active.append((idx, w))
            else:
                matrix[idx] = x_max
                injected += max(cap, 0.0)
                excess = add - max(cap, 0.0)
                still_remaining += excess
                redistributed += max(excess, 0.0)
                clipped_count += 1
        if still_remaining <= 1e-14:
            remaining_mass = 0.0
            break
        remaining_mass = still_remaining
        active = next_active
    return injected, clipped_count, redistributed


def weighted_remove_with_redistribution(
    matrix: list[float],
    weights: list[tuple[int, float]],
    mass: float,
    x_min: float,
) -> tuple[float, int]:
    remaining_mass = mass
    removed = 0.0
    clipped_count = 0
    active = list(weights)
    for _ in range(64):
        if remaining_mass <= 1e-14:
            break
        denom = sum(w for idx, w in active if matrix[idx] > x_min + 1e-15)
        if denom <= 0:
            break
        next_active: list[tuple[int, float]] = []
        still_remaining = 0.0
        for idx, w in active:
            if matrix[idx] <= x_min + 1e-15:
                continue
            take = remaining_mass * w / denom
            cap = matrix[idx] - x_min
            if take <= cap + 1e-15:
                matrix[idx] -= take
                removed += take
                next_active.append((idx, w))
            else:
                matrix[idx] = x_min
                removed += max(cap, 0.0)
                still_remaining += take - max(cap, 0.0)
                clipped_count += 1
        if still_remaining <= 1e-14:
            remaining_mass = 0.0
            break
        remaining_mass = still_remaining
        active = next_active
    return removed, clipped_count


def seed_mass_required(n: int, center: tuple[int, int, int], radius: float, width: float) -> float:
    total = 0.0
    for i, j, k in iter_cells(n):
        dx = periodic_delta(i, center[0], n)
        dy = periodic_delta(j, center[1], n)
        dz = periodic_delta(k, center[2], n)
        total += h_smooth(math.sqrt(dx * dx + dy * dy + dz * dz), radius, width)
    return total


def ledger(matrix: list[float], beta: float, gp_active: float, gp_consumed_hidden: float = 0.0) -> float:
    return sum(matrix) + beta + gp_active + gp_consumed_hidden


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_kernels(out_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n = 21
    center = (n // 2, n // 2, n // 2)
    for kind in ("nearest_cell", "compact_spherical", "gaussian_normalized"):
        matrix = [0.020] * (n**3)
        before = sum(matrix)
        weights = kernel_weights(kind, n, center, 3)
        input_mass = 0.10
        injected, clipped, redistributed = weighted_add_with_redistribution(matrix, weights, input_mass, 0.200)
        after = sum(matrix)
        err = after - before - input_mass
        rows.append({
            "kernel_type": kind,
            "release_radius_cells": 3,
            "input_mass": input_mass,
            "injected_mass": injected,
            "weight_sum": sum(w for _, w in weights),
            "mass_error_abs": err,
            "mass_error_rel": err / input_mass,
            "xB_min_before": 0.020,
            "xB_max_before": 0.020,
            "xB_min_after": min(matrix),
            "xB_max_after": max(matrix),
            "clipping_occurred": clipped > 0,
            "excess_redistributed": redistributed,
            "status": "pass" if abs(err) < 1e-10 and abs(sum(w for _, w in weights) - 1.0) < 1e-10 else "fail",
        })
    write_csv(out_dir / "gp_release_kernel_validation.csv", rows)
    return rows


def run_event_case(release_mode: str) -> dict[str, object]:
    n = 31
    center = (n // 2, n // 2, n // 2)
    x_min = 1.0e-8
    x_max = 0.250
    matrix = [0.030] * (n**3)
    site = GPSite(0, center, True, False, m_B_active=180.0, release_mode=release_mode)
    beta = 0.0
    m_B_gp_initial = site.m_B_active
    m_B_seed = seed_mass_required(n, center, radius=3.2, width=0.7)
    halo = kernel_weights("compact_spherical", n, center, 6)
    xB_min_before = min(matrix)
    xB_max_before = max(matrix)
    matrix_before = sum(matrix)
    beta_before = beta
    gp_before = site.m_B_active
    total_before = ledger(matrix, beta, site.m_B_active)

    m_from_gp_to_beta = 0.0
    m_from_matrix_to_beta = 0.0
    m_released = 0.0
    status = "pass"

    if release_mode == "release_to_beta_first":
        m_from_gp_to_beta = min(site.m_B_active, m_B_seed)
        site.m_B_active -= m_from_gp_to_beta
        m_from_matrix_to_beta = m_B_seed - m_from_gp_to_beta
        if m_from_matrix_to_beta > 0:
            removed, _ = weighted_remove_with_redistribution(matrix, halo, m_from_matrix_to_beta, x_min)
            if abs(removed - m_from_matrix_to_beta) > 1e-10:
                status = "fail: insufficient matrix halo mass"
                m_from_matrix_to_beta = removed
        beta += m_from_gp_to_beta + m_from_matrix_to_beta
        m_released = site.m_B_active
        if m_released > 0:
            injected, _, _ = weighted_add_with_redistribution(matrix, halo, m_released, x_max)
            if abs(injected - m_released) > 1e-10:
                status = "fail: insufficient matrix halo capacity for release"
                m_released = injected
        site.m_B_active = 0.0
    elif release_mode == "release_to_matrix_then_insert":
        m_released = site.m_B_active
        injected, _, _ = weighted_add_with_redistribution(matrix, halo, m_released, x_max)
        if abs(injected - m_released) > 1e-10:
            status = "fail: insufficient matrix halo capacity for release"
            m_released = injected
        site.m_B_active = 0.0
        removed, _ = weighted_remove_with_redistribution(matrix, halo, m_B_seed, x_min)
        if abs(removed - m_B_seed) > 1e-10:
            status = "fail: insufficient matrix halo mass after release"
        beta += removed
        m_from_gp_to_beta = min(m_released, removed)
        m_from_matrix_to_beta = max(removed - m_from_gp_to_beta, 0.0)
    else:
        raise ValueError(f"unknown release_mode={release_mode}")

    site.active = False
    site.consumed = True
    # The consumed GP reservoir is fully transferred to beta or matrix in this dry run.
    # It is therefore not counted as an additional hidden mass term; otherwise the
    # physical matrix/beta fields would double-count the same Ag-count reservoir.
    m_gp_consumed_hidden = 0.0
    matrix_after = sum(matrix)
    total_after = ledger(matrix, beta, site.m_B_active, m_gp_consumed_hidden)
    err = total_after - total_before
    if abs(err) > 1e-10 and status == "pass":
        status = "fail: total mass drift"
    return {
        "case": "single_center_gp_site",
        "release_mode": release_mode,
        "m_B_GP_initial": m_B_gp_initial,
        "m_B_beta_seed_required": m_B_seed,
        "m_from_GP_to_beta": m_from_gp_to_beta,
        "m_from_matrix_to_beta": m_from_matrix_to_beta,
        "m_GP_released_to_matrix": m_released,
        "m_B_matrix_before": matrix_before,
        "m_B_beta_before": beta_before,
        "m_B_GP_active_before": gp_before,
        "m_B_total_before": total_before,
        "m_B_matrix_after": matrix_after,
        "m_B_beta_after": beta,
        "m_B_GP_active_after": site.m_B_active,
        "m_B_GP_consumed_after": m_gp_consumed_hidden,
        "m_B_total_after": total_after,
        "event_mass_error_abs": err,
        "event_mass_error_rel": err / total_before,
        "xB_min_before": xB_min_before,
        "xB_max_before": xB_max_before,
        "xB_min_after": min(matrix),
        "xB_max_after": max(matrix),
        "status": status,
    }


def write_reports(out_dir: Path, event_rows: list[dict[str, object]], kernel_rows: list[dict[str, object]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    event_pass = all(str(r["status"]).startswith("pass") and abs(float(r["event_mass_error_abs"])) < 1e-10 for r in event_rows)
    kernel_pass = all(r["status"] == "pass" for r in kernel_rows)
    (out_dir / "gp_assisted_event_dry_run_report.md").write_text(
        f"""# GP-Assisted Event Dry Run Report

The dry run initializes one explicit center GP site, assigns a hidden B-equivalent reservoir, inserts one beta seed at a scheduled event, consumes the GP site, and checks `M_B_matrix + M_B_beta + M_B_GP_active`.

`GP reservoir B-equivalent mass is used only for Ag conservation bookkeeping. It does not imply that GP zones are thermodynamically Ag2Te.`

Release modes tested:

- `release_to_beta_first`
- `release_to_matrix_then_insert`

Result: `{'PASS' if event_pass else 'FAIL'}`.
Maximum absolute event mass error: `{max(abs(float(r['event_mass_error_abs'])) for r in event_rows):.6e}`.
""",
        encoding="utf-8",
    )
    (out_dir / "gp_release_kernel_report.md").write_text(
        f"""# GP Release Kernel Report

Kernels tested: `nearest_cell`, `compact_spherical`, `gaussian_normalized`.

Each kernel is normalized before injection. If a target cell would exceed `xB_max`, excess mass is redistributed to cells with remaining capacity. No silent clipping is allowed.

Result: `{'PASS' if kernel_pass else 'FAIL'}`.
Maximum absolute kernel mass error: `{max(abs(float(r['mass_error_abs'])) for r in kernel_rows):.6e}`.
""",
        encoding="utf-8",
    )


def run(out_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    kernel_rows = validate_kernels(out_dir)
    event_rows = [run_event_case("release_to_beta_first"), run_event_case("release_to_matrix_then_insert")]
    write_csv(out_dir / "gp_assisted_event_dry_run_result.csv", event_rows)
    write_reports(out_dir, event_rows, kernel_rows)
    return event_rows, kernel_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run scheduled GP-assisted beta seed insertion with explicit reservoir mass.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    event_rows, kernel_rows = run(args.out_dir.resolve())
    event_pass = all(str(r["status"]).startswith("pass") and abs(float(r["event_mass_error_abs"])) < 1e-10 for r in event_rows)
    kernel_pass = all(r["status"] == "pass" for r in kernel_rows)
    print(f"dry_run_gp_event_mass_conservation_passed={event_pass}")
    print(f"release_kernel_mass_conservation_passed={kernel_pass}")
    print(f"output_csv={args.out_dir.resolve() / 'gp_assisted_event_dry_run_result.csv'}")
    return 0 if event_pass and kernel_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
