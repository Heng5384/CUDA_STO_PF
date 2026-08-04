#!/usr/bin/env python3
"""Python-3.6 execution layer for the latest V1 audit definitions.

The cluster's supported Python is 3.6.8.  This module contains the same
read-only V1 formulas used by audit_pf_dynamic_microstructure_metrics_v1.py,
without newer annotation syntax; it has no solver or output mutation path.
"""

import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

R_GAS = 8.31446261815324


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_vtk_ascii(path, n):
    payload = path.read_bytes()
    marker = b"LOOKUP_TABLE default"
    pos = payload.find(marker)
    if pos < 0:
        raise RuntimeError("VTK marker missing: %s" % path)
    pos = payload.find(b"\n", pos)
    values = np.fromstring(payload[pos + 1:], sep=" ", dtype=np.float64, count=n ** 3)
    if values.size != n ** 3:
        raise RuntimeError("%s: expected %d, got %d" % (path, n ** 3, values.size))
    return values.reshape((n, n, n))


def l_param(t):
    return 41504.29119633958 - 18.469276826409214 * t


def ghser_pb(t):
    if t < 600.61:
        return (-7650.085 + 101.700244 * t - 24.5242231 * t * math.log(t)
                - 0.00365895 * t * t - 2.4395e-7 * t ** 3)
    return (-10531.095 + 154.243182 * t - 32.4913959 * t * math.log(t)
            + 0.00154613 * t * t + 8.05448e25 * t ** -9)


def ghser_ag(t):
    if t < 1234.93:
        return (-7209.512 + 118.202013 * t - 23.8463314 * t * math.log(t)
                - 0.001790585 * t * t - 3.98587e-7 * t ** 3 - 12011.0 / t)
    return -15095.252 + 190.266404 * t - 33.472 * t * math.log(t) + 1.411773e29 * t ** -9


def ghser_te(t):
    if t < 722.66:
        return (-10544.679 + 183.372894 * t - 35.6687 * t * math.log(t)
                + 0.01583435 * t * t - 5.240417e-6 * t ** 3 + 155015.0 / t)
    return (9160.595 - 129.265373 * t + 13.004 * t * math.log(t)
            - 0.0362361 * t * t + 5.006367e-6 * t ** 3 - 1.28681e30 * t ** -9)


def g_pbte_solid(t):
    return -76063.2138 + 9.67716633 * t + ghser_pb(t) + ghser_te(t)


def g_ag2te_solid(t):
    return 3.0 * (-10128.93 - 12.645115 * t + (2.0 / 3.0) * ghser_ag(t) + (1.0 / 3.0) * ghser_te(t))


def solve_x_eq(t):
    interaction = l_param(t); rt = R_GAS * t
    x = min(max(math.exp(-interaction / rt), 1.0e-9), 0.99)
    if x >= 0.99:
        x = 0.5
    for _ in range(20):
        residual = rt * math.log(x) + interaction * (1.0 - x) ** 2
        derivative = rt / x - 2.0 * interaction * (1.0 - x)
        delta = residual / derivative
        x = min(max(x - delta, 1.0e-10), 0.999)
        if abs(delta) < 1.0e-8:
            break
    return x


def mu_mix_hat_array(xb, temperature_k, energy_scale):
    x = np.clip(xb, 1.0e-12, 1.0 - 1.0e-12); interaction = l_param(temperature_k)
    mu_a = (g_pbte_solid(temperature_k) + R_GAS * temperature_k * np.log1p(-x) + interaction * x * x) / energy_scale
    mu_b = (g_ag2te_solid(temperature_k) + R_GAS * temperature_k * np.log(x) + interaction * (1.0 - x) * (1.0 - x)) / energy_scale
    return (1.0 - x) * mu_a + x * mu_b


def mu_b_hat_scalar(xb, temperature_k, energy_scale):
    x = min(max(xb, 1.0e-12), 1.0 - 1.0e-12)
    return (g_ag2te_solid(temperature_k) + R_GAS * temperature_k * math.log(x) + l_param(temperature_k) * (1.0 - x) ** 2) / energy_scale


def reconstruct_energy(phi_path, xb_path, n, dx_code, kappa_phi, well_w, temperature_k, mu_reference_scale, mu0_compound_hat, xb_reference, elastic_off=True, chunk_size=1000000):
    if not elastic_off:
        raise RuntimeError("exact elastic trace required")
    phi = read_vtk_ascii(phi_path, n); xb = read_vtk_ascii(xb_path, n); total = phi.size
    reference_mix = float(mu_mix_hat_array(np.asarray([xb_reference]), temperature_k, mu_reference_scale)[0])
    chemical_sum = 0.0; double_well_sum = 0.0; pflat = phi.ravel(); xbflat = xb.ravel()
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total); p = np.clip(pflat[start:stop], 0.0, 1.0)
        hp = p ** 3 * (6.0 * p ** 2 - 15.0 * p + 10.0)
        mix = mu_mix_hat_array(xbflat[start:stop], temperature_k, mu_reference_scale)
        chemical_sum += float(np.sum((1.0 - hp) * mix + hp * mu0_compound_hat - reference_mix))
        double_well_sum += float(np.sum(well_w * p * p * (1.0 - p) * (1.0 - p)))
    gradient_sum = 0.0
    for axis in range(3):
        derivative = (np.roll(phi, -1, axis=axis) - np.roll(phi, 1, axis=axis)) / (2.0 * dx_code)
        gradient_sum += float(np.sum(derivative * derivative))
    gradient_sum *= 0.5 * kappa_phi
    return {"chemical_excess_hat_mean": chemical_sum / total, "interface_gradient_hat_mean": gradient_sum / total, "interface_double_well_hat_mean": double_well_sum / total, "interface_total_hat_mean": (gradient_sum + double_well_sum) / total, "elastic_hat_mean": 0.0}


def derivative(values, times):
    if len(values) < 2:
        return np.full(len(values), np.nan)
    return np.gradient(np.asarray(values, dtype=np.float64), np.asarray(times), edge_order=2 if len(values) >= 3 else 1)


def summarize_particles(rows, growth_tolerance):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["step"])].append(row)
    summary = {}; previous = None
    for step in sorted(grouped):
        current_rows = grouped[step]; current = dict((int(row["particle_id"]), float(row["equivalent_radius_nm"])) for row in current_rows)
        radii = np.asarray(list(current.values()), dtype=np.float64); growth = shrinkage = stable = 0
        dissolved = newly_resolved = 0
        if previous is not None:
            common = set(previous) & set(current)
            for pid in common:
                change = current[pid] / previous[pid] - 1.0
                if change > growth_tolerance: growth += 1
                elif change < -growth_tolerance: shrinkage += 1
                else: stable += 1
            dissolved = len(set(previous) - set(current)); newly_resolved = len(set(current) - set(previous))
        shrinking_total = shrinkage + dissolved
        ratio = float(growth / shrinking_total) if shrinking_total else (math.inf if growth else None)
        summary[step] = {"particle_count": int(radii.size), "mean_radius_nm": float(np.mean(radii)) if radii.size else math.nan, "std_radius_nm": float(np.std(radii)) if radii.size else math.nan, "median_radius_nm": float(np.median(radii)) if radii.size else math.nan, "radius_p10_nm": float(np.percentile(radii, 10)) if radii.size else math.nan, "radius_p90_nm": float(np.percentile(radii, 90)) if radii.size else math.nan, "sum_r2_nm2": float(np.sum(radii ** 2)), "mean_r6_nm6": float(np.mean(radii ** 6)) if radii.size else math.nan, "sum_r6_nm6": float(np.sum(radii ** 6)), "growing_count": growth, "shrinking_survivor_count": shrinkage, "stable_count": stable, "dissolved_count": dissolved, "newly_resolved_count": newly_resolved, "shrinking_including_dissolved_count": shrinking_total, "growing_to_shrinking_ratio": ratio}
        previous = current
    return summary, grouped
