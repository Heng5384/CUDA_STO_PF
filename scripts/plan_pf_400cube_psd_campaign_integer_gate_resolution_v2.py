#!/usr/bin/env python3
"""Plan the V2 400-cube integer-profile PSD campaign before materialization.

The frozen 0.25 nm target-profile library has integer particle populations.
Consequently its available h-volume values form a very fine, but not
machine-continuous, lattice.  This program is intentionally only a *planner*:
it selects deterministic integer histograms from recorded native-profile
volumes and writes immutable case specifications.  It never rescales,
interpolates, edits, or materializes a profile field.

V2 accepts the physically negligible integer lattice error at 1e-7 relative
h-volume.  The subsequent materializer independently rechecks every planned
histogram against the actual assembled field before any Slurm job is eligible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


SCHEMA = "PF_400CUBE_PSD_SPATIAL_DENSITY_INTEGER_GATE_PLAN_V2"
GRID_N = 400
TARGET_H = 1531490.9025410344
H_REL_TOL = 1.0e-7
H_ABS_TOL = TARGET_H * H_REL_TOL
TARGET_MEAN_CBTOT = 0.03
TARGET_MATRIX_XAG = 0.0062
MATRIX_XAG_TOL = 1.0e-5
LAMBDA_NM = 4.0
SEPARATION_NM = 39.0  # 2*11.5 + 4*lambda; native support is also rechecked later.


@dataclass(frozen=True)
class Profile:
    radius_nm: float
    h_volume_nm3: float
    manifest_path: Path
    manifest_sha256: str
    phi_sha256: str
    delta_sha256: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def xag_from_xb(xb: float) -> float:
    return 2.0 * xb / (2.0 + xb)


def profile_rows(library_manifest: Path) -> tuple[dict[str, Any], list[Profile]]:
    library = json.loads(library_manifest.read_text(encoding="utf-8"))
    if library.get("profile_count") != 15:
        raise ValueError("V2 requires the frozen 15-entry Method-1 library")
    profiles: list[Profile] = []
    for entry in library["profiles"]:
        manifest_path = (library_manifest.parent / entry["profile_manifest_path"]).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        geometry = manifest["geometry"]
        h_volume = float(geometry.get("final_h_volume_nm3", geometry["h_volume_nm3"]))
        profiles.append(Profile(
            radius_nm=float(entry["target_radius_nm"]),
            h_volume_nm3=h_volume,
            manifest_path=manifest_path,
            manifest_sha256=sha256(manifest_path),
            phi_sha256=str(manifest["fields"]["phi"]["sha256"]),
            delta_sha256=str(manifest["fields"]["delta_C_relaxation"]["sha256"]),
        ))
    profiles.sort(key=lambda row: row.radius_nm)
    expected = [8.0 + 0.25 * index for index in range(15)]
    if [row.radius_nm for row in profiles] != expected:
        raise ValueError("unexpected registered radius ladder")
    return library, profiles


def percentile_from_histogram(radii: np.ndarray, counts: np.ndarray, q: float) -> float:
    rank = int(math.ceil(q * int(np.sum(counts))))
    rank = max(1, rank)
    cumulative = np.cumsum(counts)
    return float(radii[int(np.searchsorted(cumulative, rank, side="left"))])


def weighted_stats(radii: np.ndarray, counts: np.ndarray) -> dict[str, float]:
    total = float(np.sum(counts))
    values = np.repeat(radii, counts.astype(int))
    mean = float(np.mean(values))
    centered = values - mean
    sigma = float(np.sqrt(np.mean(centered ** 2)))
    skew = 0.0 if sigma == 0.0 else float(np.mean(centered ** 3) / sigma ** 3)
    kurtosis = 0.0 if sigma == 0.0 else float(np.mean(centered ** 4) / sigma ** 4)
    return {
        "count": int(total),
        "mean_radius_nm": mean,
        "std_radius_nm": sigma,
        "CV": 0.0 if mean == 0.0 else sigma / mean,
        "skewness": skew,
        "kurtosis": kurtosis,
        "Q10_nm": percentile_from_histogram(radii, counts, 0.10),
        "Q50_nm": percentile_from_histogram(radii, counts, 0.50),
        "Q90_nm": percentile_from_histogram(radii, counts, 0.90),
        "M3_number_nm3": float(np.mean(values ** 3)),
        "M6_number_nm6": float(np.mean(values ** 6)),
    }


def make_linear_constraints(
    volumes_scaled: np.ndarray,
    count: int,
    lower: float,
    upper: float,
    constraints: Iterable[tuple[np.ndarray, float, float]],
) -> list[LinearConstraint]:
    rows = [np.ones_like(volumes_scaled), volumes_scaled]
    lows = [float(count), lower]
    highs = [float(count), upper]
    for row, lo, hi in constraints:
        rows.append(np.asarray(row, dtype=np.float64))
        lows.append(float(lo)); highs.append(float(hi))
    return [LinearConstraint(np.vstack(rows), np.asarray(lows), np.asarray(highs))]


def solve_histogram(
    family: str,
    count: int,
    radii: np.ndarray,
    volumes: np.ndarray,
    method1_histogram: dict[str, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the frozen integer PSD contract with HiGHS MILP.

    There are no continuous profile radii in this optimization: only the 15
    registered entries are integer decision variables.  A small tolerance
    buffer avoids depending on a solver feasibility tolerance at the V2 gate.
    """
    scale = 1.0e4
    vols = volumes / scale
    target = TARGET_H / scale
    # Leave a deliberate 10% margin above the HiGHS scaled-primal tolerance;
    # the unscaled, independently recomputed ledger must still satisfy the
    # full V2 0.15314909 nm^3 gate after integer rounding.
    tolerance = H_ABS_TOL / scale * 0.90
    nbin = len(radii)
    shape_constraints: list[tuple[np.ndarray, float, float]] = []
    metadata: dict[str, Any] = {"family": family, "integer_optimizer": "SCIPY_HIGHS_MILP_V2"}
    lower_bounds = np.zeros(nbin, dtype=np.float64)
    upper_bounds = np.full(nbin, float(count), dtype=np.float64)
    base_cost = np.zeros(nbin, dtype=np.float64)

    small = radii <= 9.0
    tail = radii >= 10.5
    valley = (radii >= 9.0) & (radii <= 9.75)
    small_peak = radii <= 8.5
    large_peak = radii >= 10.25

    if family == "NARROW":
        reff = (TARGET_H / (count * (4.0 * math.pi / 3.0))) ** (1.0 / 3.0)
        base_cost = (radii - reff) ** 2 + 1.0e-8 * np.abs(radii - reff)
        metadata["narrow_reference_radius_nm"] = reff
    elif family == "REF_BROAD_METHOD1LIKE":
        # A rank/CDF remapping, not a profile transformation: Method-1's
        # 0.5-nm count CDF is mapped to available 0.25-nm *entries* around
        # the 512-particle fixed-inventory effective radius.
        source_counts = np.asarray([method1_histogram.get(f"{radius:.1f}", 0) for radius in np.arange(8.0, 11.6, 0.5)])
        source_radii = np.arange(8.0, 11.6, 0.5)
        src_mean = float(np.average(source_radii, weights=source_counts))
        target_reff = (TARGET_H / (count * (4.0 * math.pi / 3.0))) ** (1.0 / 3.0)
        # Preserve broad ordering but fit into the frozen discrete domain.
        mapped = np.clip(target_reff + 0.95 * (source_radii - src_mean), radii[0], radii[-1])
        desired = np.zeros(nbin, dtype=np.float64)
        for value, multiplicity in zip(mapped, source_counts):
            position = int(np.argmin(np.abs(radii - value)))
            desired[position] += count * float(multiplicity) / float(np.sum(source_counts))
        # Linear L1 deviation auxiliaries turn the qualitative Method-1-like
        # CDF condition into a deterministic, auditable objective.
        variables = nbin * 2
        c = np.concatenate((np.full(nbin, 1.0e-4), np.ones(nbin)))
        integrality = np.concatenate((np.ones(nbin), np.zeros(nbin)))
        bounds = Bounds(np.zeros(variables), np.concatenate((upper_bounds, np.full(nbin, np.inf))))
        base = make_linear_constraints(vols, count, target - tolerance, target + tolerance, [
            (small.astype(float), 0.15 * count, count),
            ((radii >= 10.0).astype(float), 0.08 * count, count),
            (((radii >= 8.0) & (radii <= 10.25)).astype(float), 0.55 * count, count),
        ])
        # Expand x-only constraints to (x,d), then append d >= +/- (x-desired).
        original = base[0]
        A = np.hstack((original.A, np.zeros((original.A.shape[0], nbin))))
        eye = np.eye(nbin)
        # x - d <= desired; -x -d <= -desired.
        A = np.vstack((A, np.hstack((eye, -eye)), np.hstack((-eye, -eye))))
        lb = np.concatenate((original.lb, np.full(nbin * 2, -np.inf)))
        ub = np.concatenate((original.ub, desired, -desired))
        result = milp(c=c, integrality=integrality, bounds=bounds, constraints=LinearConstraint(A, lb, ub), options={"disp": False})
        if not result.success or result.x is None:
            raise RuntimeError(f"{family}/N{count}: MILP failed: {result.message}")
        answer = np.rint(result.x[:nbin]).astype(np.int64)
        metadata.update({"CDF_source": "Method-1 96-particle selected histogram", "CDF_remapping": "rank-preserving discrete 0.5nm-to-0.25nm entry remapping", "CDF_L1_deviation": float(np.sum(np.abs(answer - desired)))})
        return answer, metadata
    elif family == "SMALL_RICH_LONGTAIL":
        shape_constraints += [
            (small.astype(float), math.ceil(0.75 * count), count),
            (tail.astype(float), math.ceil(0.05 * count), math.floor(0.18 * count)),
            (((radii >= 9.25) & (radii <= 10.25)).astype(float), math.ceil(0.02 * count), math.floor(0.20 * count)),
            ((radii == 10.5).astype(float), 1.0, count),
            ((radii == 11.5).astype(float), 1.0, count),
            ((radii == 9.25).astype(float), 1.0, count),
            ((radii == 9.5).astype(float), 1.0, count),
            ((radii == 9.75).astype(float), 1.0, count),
            ((radii == 10.0).astype(float), 1.0, count),
            ((radii == 10.25).astype(float), 1.0, count),
        ]
        # Largest sixth moment subject to a non-bimodal small-rich contract.
        base_cost = -(radii ** 6)
        metadata["shape_contract"] = "small<=9nm >=75%; tail>=10.5nm 5-18%; continuous middle retained"
    elif family == "BIMODAL":
        shape_constraints += [
            (small_peak.astype(float), math.ceil(0.35 * count), math.floor(0.65 * count)),
            (large_peak.astype(float), math.ceil(0.30 * count), math.floor(0.65 * count)),
            (valley.astype(float), 0.0, math.floor(0.05 * count)),
            ((radii == 8.0).astype(float), math.ceil(0.15 * count), count),
            ((radii >= 10.5).astype(float), math.ceil(0.15 * count), count),
        ]
        # Maximise inter-peak separation with a small central-population penalty.
        base_cost = np.where(small_peak | large_peak, -np.abs(radii - 9.5), np.abs(radii - 9.5))
        metadata["shape_contract"] = "small 8-8.5nm and large 10.25-11.5nm peaks; 9-9.75nm valley <=5%"
    else:
        raise ValueError(f"unknown PSD family {family}")

    result = milp(
        c=base_cost,
        integrality=np.ones(nbin),
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=make_linear_constraints(vols, count, target - tolerance, target + tolerance, shape_constraints),
        options={"disp": False},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"{family}/N{count}: MILP failed: {result.message}")
    answer = np.rint(result.x).astype(np.int64)
    return answer, metadata


def build_case_definitions() -> list[dict[str, Any]]:
    rows: list[tuple[str, str, int, str]] = [
        ("001", "REF_BROAD_N512_A", 512, "REF_BROAD_METHOD1LIKE"),
        ("002", "REF_BROAD_N512_B", 512, "REF_BROAD_METHOD1LIKE"),
        ("003", "REF_BROAD_N512_C", 512, "REF_BROAD_METHOD1LIKE"),
        ("004", "NARROW_N512_A", 512, "NARROW"),
        ("005", "NARROW_N512_B", 512, "NARROW"),
        ("006", "NARROW_N512_C", 512, "NARROW"),
        ("007", "LONGTAIL_N512_A", 512, "SMALL_RICH_LONGTAIL"),
        ("008", "LONGTAIL_N512_B", 512, "SMALL_RICH_LONGTAIL"),
        ("009", "LONGTAIL_N512_C", 512, "SMALL_RICH_LONGTAIL"),
        ("010", "BIMODAL_N512_A", 512, "BIMODAL"),
        ("011", "BIMODAL_N512_B", 512, "BIMODAL"),
        ("012", "BIMODAL_N512_C", 512, "BIMODAL"),
        ("013", "NARROW_N256_A", 256, "NARROW"),
        ("014", "NARROW_N256_B", 256, "NARROW"),
        ("015", "NARROW_N256_C", 256, "NARROW"),
        ("016", "NARROW_N640_A", 640, "NARROW"),
        ("017", "NARROW_N640_B", 640, "NARROW"),
        ("018", "NARROW_N640_C", 640, "NARROW"),
        ("019", "NARROW_N704_A", 704, "NARROW"),
        ("020", "NARROW_N704_B", 704, "NARROW"),
        ("021", "NARROW_N704_C", 704, "NARROW"),
    ]
    return [{"case_id": case_id, "case_label": label, "particle_count": count, "psd_family": family, "replicate": label[-1]} for case_id, label, count, family in rows]


def seed64(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big", signed=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library-manifest", type=Path, required=True)
    parser.add_argument("--method1-selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"[fatal] refusing to overwrite V2 planning root: {args.out}")
    library_manifest = args.library_manifest.resolve()
    method1_selection = json.loads(args.method1_selection.read_text(encoding="utf-8"))
    library, profiles = profile_rows(library_manifest)
    radii = np.asarray([row.radius_nm for row in profiles], dtype=np.float64)
    volumes = np.asarray([row.h_volume_nm3 for row in profiles], dtype=np.float64)
    method1_histogram = {str(key): int(value) for key, value in method1_selection["selected_histogram"].items()}
    cases = build_case_definitions()

    # Stage A: all four particle counts and all registered PSD-shape contracts
    # must be satisfiable about one common *target* inventory.  The original
    # target itself is this common V_star; individual integer sums are allowed
    # to differ only inside the new 1e-7 sub-voxel gate.
    needed = [("NARROW", 256), ("NARROW", 512), ("NARROW", 640), ("NARROW", 704),
              ("REF_BROAD_METHOD1LIKE", 512), ("SMALL_RICH_LONGTAIL", 512), ("BIMODAL", 512)]
    selected: dict[tuple[str, int], tuple[np.ndarray, dict[str, Any]]] = {}
    search_rows: list[dict[str, Any]] = []
    for family, count in needed:
        histogram, solver_meta = solve_histogram(family, count, radii, volumes, method1_histogram)
        selected[(family, count)] = (histogram, solver_meta)
        actual = float(np.dot(histogram, volumes))
        rel = abs(actual - TARGET_H) / TARGET_H
        if int(np.sum(histogram)) != count or rel > H_REL_TOL:
            raise RuntimeError(f"Stage A h-volume contract failed for {family}/N{count}")
        stats = weighted_stats(radii, histogram)
        search_rows.append({
            "family": family, "particle_count": count, "candidate_common_target_h_volume_nm3": TARGET_H,
            "selected_h_volume_nm3": actual, "absolute_h_volume_error_nm3": abs(actual - TARGET_H),
            "relative_h_volume_error": rel, "shape_contract_status": "PASS", **stats,
        })
    inventory_policy = "COMMON_INTEGER_REALIZABLE_H_VOLUME"

    args.out.mkdir(parents=True)
    specs_dir = args.out / "case_specs"
    specs_dir.mkdir()
    profile_table = [{
        "radius_nm": row.radius_nm, "h_volume_nm3": row.h_volume_nm3,
        "profile_manifest_path": str(row.manifest_path), "profile_manifest_sha256": row.manifest_sha256,
        "phi_sha256": row.phi_sha256, "delta_C_relaxation_sha256": row.delta_sha256,
    } for row in profiles]
    library_sha = sha256(library_manifest)
    histogram_rows: list[dict[str, Any]] = []
    h_rows: list[dict[str, Any]] = []
    psd_rows: list[dict[str, Any]] = []
    case_manifest_rows: list[dict[str, Any]] = []
    for case in cases:
        histogram, solver_meta = selected[(case["psd_family"], case["particle_count"])]
        actual_h = float(np.dot(histogram, volumes))
        stats = weighted_stats(radii, histogram)
        histogram_map = {f"{radius:.2f}": int(number) for radius, number in zip(radii, histogram) if number}
        placement_key = f"N{case['particle_count']}_{case['replicate']}"
        spec = {
            "schema": SCHEMA,
            "fixture_id": f"pf_400cube_psd_campaign_v2_{case['case_id']}_{case['case_label'].lower()}",
            "case": case,
            "grid": {"Nx": GRID_N, "Ny": GRID_N, "Nz": GRID_N, "dx_nm": 1.0, "lambda_sm_nm": LAMBDA_NM},
            "physical_contract": {
                "temperature_C": 380.0, "elasticity_enabled": True, "elastic_boundary": "periodic_fixed_cell",
                "orientation_label": "variant_100_identity", "eigenstrain": [0.046, -0.022, -0.017, 0.0, 0.0, 0.0],
                "GP_enabled": False, "GP_birth_enabled": False, "GP_release_enabled": False,
                "external_source_enabled": False, "new_beta_nucleation_enabled": False,
                "common_multi_particle_equilibrium_claim": False, "initial_relaxation_is_physical_evolution": True,
            },
            "inventory_contract": {
                "original_target_h_volume_nm3": TARGET_H, "selected_common_target_h_volume_nm3": TARGET_H,
                "inventory_policy": inventory_policy, "h_volume_relative_tolerance": H_REL_TOL,
                "mean_C_B_tot": TARGET_MEAN_CBTOT, "matrix_xAg_target": TARGET_MATRIX_XAG,
                "matrix_xAg_absolute_tolerance": MATRIX_XAG_TOL, "experimental_matrix_xAg_interval": [0.0058, 0.0066],
                "planned_profile_h_volume_nm3": actual_h,
            },
            "profile_library_contract": {
                "library_manifest_path": str(library_manifest), "library_manifest_sha256": library_sha,
                "expected_radii_nm": [float(value) for value in radii], "native_grid": [96, 96, 96],
                "profile_scaling_used": False, "profile_interpolation_used": False,
                "profile_rotation_used": False, "analytic_profile_used": False,
                "entries": profile_table,
            },
            "discrete_psd": {
                "family": case["psd_family"], "registered_radius_histogram": histogram_map,
                "particle_count": case["particle_count"], "statistics": stats, "solver": solver_meta,
                "canonical_radius_assignment": "SHA256(profile_assignment_v2, case family, replicate, particle_id); entries only",
            },
            "placement_contract": {
                "algorithm": "PCG64_PERIODIC_HARD_CORE_V2", "placement_key": placement_key,
                "center_seed_unsigned64": seed64(f"PF400V2:center:{placement_key}"),
                "assignment_seed_unsigned64": seed64(f"PF400V2:assignment:{case['psd_family']}:{case['replicate']}:N{case['particle_count']}"),
                "minimum_declared_center_distance_nm": SEPARATION_NM,
                "periodic_separation_rule": "distance > max(Ri+Rj+4lambda, support_i+support_j)",
                "h_threshold": 1.0e-4, "max_trials_per_particle": 2000000,
                "density_center_sets_nested": False,
            },
            "time_contract": {
                "initial_age_h": 6.0, "dt_code": 0.02, "dt_physical_s": 0.9909260953431841,
                "final_step": 152585, "checkpoint_cadence_steps": 3633,
                "registered_science_steps": [0, 21798, 43596, 65393, 108989, 152585],
            },
        }
        spec["spec_sha256"] = canonical_json_sha256(spec)
        spec_path = specs_dir / f"{case['case_id']}_{case['case_label']}.json"
        write_json(spec_path, spec)
        case_manifest_rows.append({**case, "spec_path": str(spec_path), "spec_sha256": sha256(spec_path), "placement_key": placement_key})
        h_rows.append({"case_id": case["case_id"], "case_label": case["case_label"], "family": case["psd_family"], "particle_count": case["particle_count"], "original_target_h_volume_nm3": TARGET_H, "selected_target_h_volume_nm3": TARGET_H, "planned_h_volume_nm3": actual_h, "absolute_error_nm3": abs(actual_h - TARGET_H), "relative_error": abs(actual_h - TARGET_H) / TARGET_H, "policy": inventory_policy})
        psd_rows.append({"case_id": case["case_id"], "case_label": case["case_label"], "family": case["psd_family"], **stats})
        for radius, number in zip(radii, histogram):
            histogram_rows.append({"case_id": case["case_id"], "case_label": case["case_label"], "family": case["psd_family"], "particle_count": case["particle_count"], "radius_nm": radius, "integer_count": int(number), "profile_h_volume_nm3": float(volumes[np.where(radii == radius)[0][0]])})

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            raise ValueError(f"no rows for {path}")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    write_csv(args.out / "common_realizable_target_search.csv", search_rows)
    write_csv(args.out / "casewise_integer_histograms.csv", histogram_rows)
    write_csv(args.out / "casewise_h_volume_errors.csv", h_rows)
    write_csv(args.out / "casewise_psd_statistics_forecast.csv", psd_rows)
    write_csv(args.out / "campaign_case_manifest.csv", case_manifest_rows)
    write_json(args.out / "campaign_case_manifest.json", {"schema": SCHEMA, "cases": case_manifest_rows, "unique_scientific_cases": 21, "logical_case_count": 24})
    report = f"""# V2 common integer-realizable target search\n\nThe previous block is frozen historical evidence.  Its corrected identity is\n`OVERCONSTRAINED_INTEGER_H_VOLUME_GATE`, not an invalid physical radius range.\n\n- Original target: `{TARGET_H:.10f} nm³`\n- Selected common target: `{TARGET_H:.10f} nm³`\n- V2 relative tolerance: `{H_REL_TOL:.1e}` (`{H_ABS_TOL:.8f} nm³`)\n- Policy: `{inventory_policy}`\n- Tested contracts: 4 N values and 4 PSD shape families\n- Maximum planned relative error: `{max(row['relative_h_volume_error'] for row in search_rows):.12e}`\n\nEvery solution uses only a non-negative integer count of recorded native\nprofiles.  This planning result is not fixture qualification: actual field\nassembly, mass closure, periodic non-overlap, re-materialization/reverse-order\ninvariance and zero-macrostep parser preflight remain mandatory.\n"""
    (args.out / "common_realizable_target_report.md").write_text(report, encoding="utf-8")
    write_json(args.out / "planning_provenance.json", {
        "schema": SCHEMA, "library_manifest": str(library_manifest), "library_manifest_sha256": library_sha,
        "method1_selection": str(args.method1_selection.resolve()), "method1_selection_sha256": sha256(args.method1_selection),
        "planner_script_sha256": sha256(Path(__file__)), "original_target_h_volume_nm3": TARGET_H,
        "selected_common_target_h_volume_nm3": TARGET_H, "inventory_policy": inventory_policy,
        "h_volume_relative_tolerance": H_REL_TOL, "matrix_xAg_target": TARGET_MATRIX_XAG,
    })
    (args.out / "status.txt").write_text("PASS_COMMON_INTEGER_REALIZABLE_INVENTORY_V2\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_COMMON_INTEGER_REALIZABLE_INVENTORY_V2", "case_count": 21, "selected_common_target_h_volume_nm3": TARGET_H, "maximum_relative_error": max(row["relative_error"] for row in h_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
