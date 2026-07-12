#!/usr/bin/env python3
"""Core utilities for GP-zone-assisted beta nucleation validation.

This module is intentionally solver-adjacent rather than a CUDA kernel rewrite.
It implements the data contract required by the runtime path:

    barrier library -> local s_GP barrier multiplier -> explicit event rate
    -> parametric beta seed -> mass ledger.

The convention is always ``DeltaG_eff = s_GP * DeltaG_bare``.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

K_B_J_PER_K = 1.380649e-23


def clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        out = float(text)
    except ValueError:
        return default
    return out if math.isfinite(out) else default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class BarrierEntry:
    T_C: float
    T_K: float
    xB: float
    strain: float
    DeltaG_bare_kBT: float
    DeltaG_bare_J: float | None
    r_star_nm: float
    source_case: str
    valid: bool = True
    validity_reason: str = "ok"
    Z_n: float = 1.0
    Z_r: float = 1.0


class BarrierLibrary:
    def __init__(self, entries: Iterable[BarrierEntry]):
        self.entries = sorted(entries, key=lambda e: (e.T_C, e.xB, e.strain, e.DeltaG_bare_kBT))

    @classmethod
    def from_csv(cls, path: Path, *, allow_incomplete: bool = False) -> "BarrierLibrary":
        rows = read_csv(path)
        entries: list[BarrierEntry] = []
        for row in rows:
            T_C = safe_float(row.get("T_C"))
            T_K = safe_float(row.get("T_K"))
            xB = safe_float(row.get("xB"))
            strain = safe_float(row.get("strain"), 0.0)
            barrier_kBT = (
                safe_float(row.get("DeltaG_bare_kBT"))
                or safe_float(row.get("DeltaG_star_kBT"))
                or safe_float(row.get("cnt_refsub_peak_kBT"))
                or safe_float(row.get("barrier_kBT_corrected"))
                or safe_float(row.get("barrier_over_kBT"))
            )
            barrier_J = (
                safe_float(row.get("DeltaG_bare_J"))
                or safe_float(row.get("DeltaG_star_J"))
                or safe_float(row.get("cnt_refsub_peak_J"))
            )
            r_star_nm = (
                safe_float(row.get("r_star_nm"))
                or safe_float(row.get("r_star_CNT_nm"))
                or safe_float(row.get("rc_schur_nm"))
                or safe_float(row.get("r_eff_star_nm"))
            )
            n_points = safe_float(row.get("n_points"), 99.0)
            valid = bool(n_points is None or n_points >= 3.0)
            reason = "ok" if valid else f"incomplete_radius_scan:n_points={n_points:g}"
            if not allow_incomplete and not valid:
                continue
            if None in (T_C, xB, strain, barrier_kBT, r_star_nm):
                continue
            if T_K is None:
                T_K = T_C + 273.15
            z_r = safe_float(row.get("Z_r"))
            z_n = safe_float(row.get("Z_n"))
            omega_g = safe_float(row.get("beta_rate_Omega_g_m3"), safe_float(row.get("Omega_g_m3")))
            if z_n is None and z_r is not None and r_star_nm is not None and omega_g is not None and omega_g > 0.0:
                r_star_m = r_star_nm * 1.0e-9
                if r_star_m > 0.0:
                    z_n = z_r * omega_g / (4.0 * math.pi * r_star_m * r_star_m)
            entries.append(
                BarrierEntry(
                    T_C=float(T_C),
                    T_K=float(T_K),
                    xB=float(xB),
                    strain=float(strain),
                    DeltaG_bare_kBT=float(barrier_kBT),
                    DeltaG_bare_J=barrier_J,
                    r_star_nm=float(r_star_nm),
                    source_case=row.get("workflow_name") or row.get("condition_id") or row.get("case_id") or str(path),
                    valid=valid,
                    validity_reason=reason,
                    Z_n=z_n or 1.0,
                    Z_r=z_r or 1.0,
                )
            )
        return cls(entries)

    def select(self, T_C: float, xB: float, strain: float = 0.0) -> tuple[BarrierEntry, str]:
        valid = [e for e in self.entries if e.valid]
        if not valid:
            raise ValueError("barrier library has no valid entries")
        best_T = min({e.T_C for e in valid}, key=lambda t: abs(t - T_C))
        slice_T = [e for e in valid if e.T_C == best_T]
        best_strain = min({e.strain for e in slice_T}, key=lambda s: abs(s - strain))
        slice_TS = sorted([e for e in slice_T if e.strain == best_strain], key=lambda e: e.xB)
        if not slice_TS:
            raise ValueError("barrier library has no entries after T/strain filtering")

        lower = None
        upper = None
        for entry in slice_TS:
            if entry.xB <= xB:
                lower = entry
            if entry.xB >= xB and upper is None:
                upper = entry
        if lower and upper and lower is not upper and lower.xB != upper.xB:
            f = (xB - lower.xB) / (upper.xB - lower.xB)
            T_K = T_C + 273.15
            barrier_kBT = lower.DeltaG_bare_kBT + f * (upper.DeltaG_bare_kBT - lower.DeltaG_bare_kBT)
            r_star = lower.r_star_nm + f * (upper.r_star_nm - lower.r_star_nm)
            barrier_J = barrier_kBT * K_B_J_PER_K * T_K
            interp = BarrierEntry(
                T_C=T_C,
                T_K=T_K,
                xB=xB,
                strain=best_strain,
                DeltaG_bare_kBT=barrier_kBT,
                DeltaG_bare_J=barrier_J,
                r_star_nm=r_star,
                source_case=f"linear_xB:{lower.source_case}|{upper.source_case}",
                valid=True,
                validity_reason="interpolated_xB_nearest_T",
                Z_n=lower.Z_n + f * (upper.Z_n - lower.Z_n),
                Z_r=lower.Z_r + f * (upper.Z_r - lower.Z_r),
            )
            return interp, f"nearest_T={best_T:g}; nearest_strain={best_strain:g}; linear_xB={lower.xB:g}->{upper.xB:g}"
        nearest = min(slice_TS, key=lambda e: abs(e.xB - xB))
        return nearest, f"nearest_T={best_T:g}; nearest_strain={best_strain:g}; nearest_xB={nearest.xB:g}"


@dataclass(frozen=True)
class NucleusDescriptor:
    id: str
    T_C: float | None
    xB: float | None
    strain: float | None
    r_seed_nm: float
    shape_type: str = "isotropic_fallback"
    semiaxes_nm: tuple[float, float, float] | None = None
    orientation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None = None
    source: str = ""


class NucleusCatalog:
    def __init__(self, entries: Iterable[NucleusDescriptor]):
        self.entries = list(entries)

    @classmethod
    def from_json(cls, path: Path) -> "NucleusCatalog":
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_entries = payload.get("entries", payload if isinstance(payload, list) else [])
        entries: list[NucleusDescriptor] = []
        for i, row in enumerate(raw_entries):
            r_seed = safe_float(row.get("r_seed_nm")) or safe_float(row.get("rc_nm")) or safe_float(row.get("r_star_nm"))
            if r_seed is None:
                continue
            semiaxes = row.get("semiaxes_nm") or row.get("semiaxes")
            sem_tuple = None
            if isinstance(semiaxes, list) and len(semiaxes) >= 3:
                sem_tuple = (float(semiaxes[0]), float(semiaxes[1]), float(semiaxes[2]))
            entries.append(
                NucleusDescriptor(
                    id=str(row.get("id") or row.get("template_id") or f"catalog_{i}"),
                    T_C=safe_float(row.get("T_C")),
                    xB=safe_float(row.get("xB")),
                    strain=safe_float(row.get("strain"), 0.0),
                    r_seed_nm=float(r_seed),
                    shape_type=str(row.get("shape_type") or row.get("seed_type") or "isotropic_fallback"),
                    semiaxes_nm=sem_tuple,
                    source=str(row.get("source_dir") or row.get("source_dyn_dir") or path),
                )
            )
        return cls(entries)

    def select(self, T_C: float, xB: float, strain: float, r_star_nm: float) -> tuple[NucleusDescriptor, str]:
        if not self.entries:
            return (
                NucleusDescriptor(
                    id="isotropic_fallback_from_barrier",
                    T_C=T_C,
                    xB=xB,
                    strain=strain,
                    r_seed_nm=r_star_nm,
                    shape_type="isotropic_fallback",
                    semiaxes_nm=(r_star_nm, r_star_nm, r_star_nm),
                    source="barrier_r_star",
                ),
                "fallback_no_catalog_entries",
            )
        def score(e: NucleusDescriptor) -> tuple[float, float, float, float]:
            return (
                abs((e.T_C if e.T_C is not None else T_C) - T_C),
                abs((e.xB if e.xB is not None else xB) - xB),
                abs((e.strain if e.strain is not None else strain) - strain),
                abs(e.r_seed_nm - r_star_nm),
            )
        best = min(self.entries, key=score)
        return best, f"nearest_catalog_score={score(best)}"


@dataclass
class NucleationCandidate:
    index: tuple[int, int, int]
    xB_loc: float
    phi_beta: float
    gp_density: float = 0.0
    gp_reservoir_mass: float = 0.0


@dataclass
class RateResult:
    gp_present: bool
    s_GP: float
    DeltaG_bare_kBT: float
    DeltaG_eff_kBT: float
    J_bare: float
    J_eff: float
    probability: float
    allowed: bool
    reason: str
    selection_reason: str
    r_star_nm: float


def gp_present(candidate: NucleationCandidate, source: str, threshold: float) -> bool:
    if source == "none":
        return False
    if source in {"gp_density", "eta", "gp_site_array"}:
        return candidate.gp_density >= threshold
    if source == "reservoir":
        return candidate.gp_reservoir_mass > threshold
    raise ValueError(f"unsupported GP mask source: {source}")


def effective_barrier_kBT(deltaG_bare_kBT: float, gp_is_present: bool, s_gp_user: float) -> tuple[float, float]:
    s = clamp(s_gp_user if gp_is_present else 1.0, 1.0e-12, 1.0)
    return s * deltaG_bare_kBT, s


def nucleation_probability(J: float, dV_nuc: float, dt: float) -> float:
    intensity = max(J * dV_nuc * dt, 0.0)
    if intensity > 700.0:
        return 1.0
    return 1.0 - math.exp(-intensity)


def evaluate_candidate(
    candidate: NucleationCandidate,
    barrier_library: BarrierLibrary,
    *,
    T_C: float,
    strain: float,
    s_gp_user: float,
    gp_presence_threshold: float,
    gp_site_mask_source: str,
    mode: str,
    phi_threshold: float,
    N_site: float,
    beta_r_star: float,
    theta_tr: float,
    dV_nuc: float,
    dt: float,
) -> RateResult:
    if candidate.phi_beta >= phi_threshold:
        return RateResult(False, 1.0, math.nan, math.nan, 0.0, 0.0, 0.0, False, "blocked_existing_beta", "", math.nan)
    present = gp_present(candidate, gp_site_mask_source, gp_presence_threshold)
    if mode == "GP_only" and not present:
        return RateResult(False, 1.0, math.nan, math.nan, 0.0, 0.0, 0.0, False, "blocked_no_GP", "", math.nan)
    if mode not in {"GP_only", "homogeneous_plus_GP"}:
        raise ValueError(f"unsupported nucleation mode: {mode}")
    entry, reason = barrier_library.select(T_C, candidate.xB_loc, strain)
    eff_kBT, s_used = effective_barrier_kBT(entry.DeltaG_bare_kBT, present, s_gp_user)
    prefactor = max(N_site * entry.Z_n * beta_r_star * theta_tr, 0.0)
    J_bare = prefactor * math.exp(-min(entry.DeltaG_bare_kBT, 700.0))
    J_eff = prefactor * math.exp(-min(eff_kBT, 700.0))
    P = nucleation_probability(J_eff, dV_nuc, dt)
    return RateResult(
        gp_present=present,
        s_GP=s_used,
        DeltaG_bare_kBT=entry.DeltaG_bare_kBT,
        DeltaG_eff_kBT=eff_kBT,
        J_bare=J_bare,
        J_eff=J_eff,
        probability=P,
        allowed=True,
        reason="ok",
        selection_reason=reason,
        r_star_nm=entry.r_star_nm,
    )


def periodic_delta(i: int, c: int, n: int) -> int:
    d = i - c
    if d > n // 2:
        d -= n
    if d < -n // 2:
        d += n
    return d


def analytic_seed_phi(n: int, center: tuple[int, int, int], radius_cells: float, width_cells: float) -> list[float]:
    phi = [0.0] * (n**3)
    inv_width = 1.0 / max(width_cells, 1.0e-12)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                dx = periodic_delta(i, center[0], n)
                dy = periodic_delta(j, center[1], n)
                dz = periodic_delta(k, center[2], n)
                r = math.sqrt(dx * dx + dy * dy + dz * dz)
                value = 0.5 * (1.0 - math.tanh((r - radius_cells) * inv_width))
                phi[(i * n + j) * n + k] = clamp(value, 0.0, 1.0)
    return phi


def insert_parametric_seed_mass_conserving(
    matrix_xB: list[float],
    beta_phi: list[float],
    gp_reservoir_mass: float,
    *,
    n: int,
    center: tuple[int, int, int],
    radius_cells: float,
    width_cells: float,
    xB_min: float,
) -> tuple[list[float], list[float], dict[str, float]]:
    if len(matrix_xB) != n**3 or len(beta_phi) != n**3:
        raise ValueError("field sizes do not match n^3")
    total_before = sum(matrix_xB) + sum(beta_phi) + gp_reservoir_mass
    seed = analytic_seed_phi(n, center, radius_cells, width_cells)
    seed_mass_required = 0.0
    for idx, value in enumerate(seed):
        add = max(value - beta_phi[idx], 0.0)
        beta_phi[idx] += add
        seed_mass_required += add
    from_gp = min(gp_reservoir_mass, seed_mass_required)
    gp_after = gp_reservoir_mass - from_gp
    from_matrix_needed = seed_mass_required - from_gp
    removed = 0.0
    if from_matrix_needed > 0.0:
        # Deplete a matrix halo using the same smooth seed weights first.
        weights = [(idx, max(seed[idx], 0.0)) for idx in range(n**3) if seed[idx] > 1.0e-10]
        denom = sum(w for _, w in weights)
        if denom <= 0.0:
            raise ValueError("seed has no support")
        remaining = from_matrix_needed
        active = weights
        for _ in range(128):
            if remaining <= 1.0e-13:
                break
            denom = sum(w for idx, w in active if matrix_xB[idx] > xB_min + 1.0e-15)
            if denom <= 0.0:
                break
            next_active: list[tuple[int, float]] = []
            next_remaining = 0.0
            for idx, w in active:
                if matrix_xB[idx] <= xB_min + 1.0e-15:
                    continue
                take = remaining * w / denom
                cap = matrix_xB[idx] - xB_min
                if take <= cap + 1.0e-15:
                    matrix_xB[idx] -= take
                    removed += take
                    next_active.append((idx, w))
                else:
                    matrix_xB[idx] = xB_min
                    removed += max(cap, 0.0)
                    next_remaining += take - max(cap, 0.0)
            if next_remaining <= 1.0e-13:
                remaining = 0.0
                break
            remaining = next_remaining
            active = next_active
    total_after = sum(matrix_xB) + sum(beta_phi) + gp_after
    mass_error = total_after - total_before
    ledger = {
        "seed_mass_required": seed_mass_required,
        "mass_from_GP": from_gp,
        "mass_from_matrix": removed,
        "mass_released": 0.0,
        "total_mass_before": total_before,
        "total_mass_after": total_after,
        "mass_error": mass_error,
        "relative_mass_error": mass_error / max(abs(total_before), 1.0e-30),
        "gp_reservoir_after": gp_after,
    }
    return matrix_xB, beta_phi, ledger
