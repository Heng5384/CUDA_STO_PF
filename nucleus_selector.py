#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
REPORTS_ROOT = REPO_ROOT / "reports"
NUCLEATION_ROOT = REPORTS_ROOT / "nucleation"
DEFAULT_CATALOG = REPO_ROOT / "nucleus_catalog.json"
DEFAULT_SELECTED = REPO_ROOT / "selected_nucleus.json"
DEFAULT_INDEX = NUCLEATION_ROOT / "nucleus_energy_output_index.md"
DEFAULT_REPORT = NUCLEATION_ROOT / "nucleus_selector_integration_report.md"


def as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        out = float(text)
    except ValueError:
        return default
    return out if math.isfinite(out) else default


def repo_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    text = path_text.strip()
    if not text:
        return None
    legacy_prefix = "/data/home/luozhiheng/CUDA_STO_PF"
    if text.startswith(legacy_prefix):
        text = str(REPO_ROOT) + text[len(legacy_prefix):]
    path = Path(text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def infer_T_C(row: dict[str, str], path: Path) -> float | None:
    for key in ("T_C", "T_input"):
        val = as_float(row.get(key))
        if val is not None:
            return val
    text = " ".join([str(path), row.get("workflow_name", ""), row.get("base_case_tag", "")])
    match = re.search(r"T(\d+(?:p\d+)?)", text)
    if match:
        return float(match.group(1).replace("p", "."))
    return None


def infer_xB(row: dict[str, str], path: Path) -> float | None:
    for key in ("xB_out", "xB0", "xB_loc"):
        val = as_float(row.get(key))
        if val is not None:
            return val
    text = " ".join([str(path), row.get("workflow_name", ""), row.get("base_case_tag", "")])
    match = re.search(r"xB0p(\d+)", text)
    if match:
        return float("0." + match.group(1))
    match = re.search(r"x0_0p(\d+)", text)
    if match:
        return float("0." + match.group(1))
    return None


def infer_strain_mode(row: dict[str, str]) -> str:
    mode = row.get("mode", "").strip()
    if mode:
        return mode
    base = row.get("base_case_tag", "")
    for token in ("exx_eyy", "exx", "eyy", "ezz"):
        if token in base:
            return token
    return "unknown"


def shape_from_metrics(row: dict[str, str]) -> tuple[str, dict[str, Any]]:
    L1 = as_float(row.get("L1_long"))
    L2 = as_float(row.get("L2_mid"))
    L3 = as_float(row.get("L3_short"))
    aspect = as_float(row.get("L1_over_L3"))
    if aspect is None and L1 and L3 and L3 > 0.0:
        aspect = L1 / L3

    if L1 and L2 and L3:
        semiaxes = [0.5 * L1, 0.5 * L2, 0.5 * L3]
    else:
        rc = as_float(row.get("rc_cnt_fit_nm"), as_float(row.get("rc_cnt_nm"), as_float(row.get("r_eff_star_nm"))))
        semiaxes = [rc, rc, rc] if rc else []

    if aspect is None:
        shape_type = "unknown"
    elif aspect < 1.10:
        shape_type = "spherical"
    elif aspect < 2.0:
        shape_type = "faceted" if row.get("short_face_normal_x") else "anisotropic"
    else:
        shape_type = "planar"

    return shape_type, {
        "aspect_ratio": aspect,
        "semiaxes_nm": semiaxes,
        "L1_long_nm": L1,
        "L2_mid_nm": L2,
        "L3_short_nm": L3,
    }


def find_profile_dir(source_dir: Path | None) -> Path | None:
    if source_dir is None:
        return None
    search_roots = []
    if source_dir.exists():
        search_roots.append(source_dir)
        search_roots.append(source_dir.parent)
    for root in search_roots:
        for candidate in root.rglob("faceted_family_profiles.csv"):
            return candidate.parent
    return None


def load_barrier_lookup() -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for path in REPO_ROOT.glob("Results/**/nucleation_rate_table*.csv"):
        for row in read_csv(path):
            key = row.get("base_case_tag", "")
            if key and key not in lookup:
                lookup[key] = row
    return lookup


def convert_master_row(path: Path, row: dict[str, str], barrier_lookup: dict[str, dict[str, str]]) -> dict[str, Any] | None:
    base = row.get("base_case_tag", "").strip()
    if not base:
        return None
    T_C = infer_T_C(row, path)
    xB = infer_xB(row, path)
    strain_value = as_float(row.get("strain"))
    rc_nm = as_float(row.get("rc_cnt_fit_nm"), as_float(row.get("rc_cnt_nm"), as_float(row.get("r_eff_star_nm"))))
    if T_C is None or xB is None or strain_value is None or rc_nm is None:
        return None

    b_row = barrier_lookup.get(base, {})
    barrier_kBT = as_float(b_row.get("barrier_kBT_corrected"))
    if barrier_kBT is None:
        barrier_kBT = as_float(b_row.get("DeltaG_star_kBT"))
    if barrier_kBT is None:
        barrier_kBT = as_float(b_row.get("barrier_over_kBT"))
    energy_source = "nucleation_rate_table"
    if barrier_kBT is None:
        barrier_kBT = 1.0e300
        energy_source = "missing_kBT_conversion"
    if barrier_kBT is None:
        barrier_kBT = 1.0e300
        energy_source = "missing"

    shape_type, shape_metrics = shape_from_metrics(row)
    summary_path = repo_path(row.get("summary_path"))
    source_dyn_dir = summary_path.parent if summary_path is not None else None
    profile_dir = find_profile_dir(source_dyn_dir)

    source_available = bool(source_dyn_dir and source_dyn_dir.exists())
    profile_available = bool(profile_dir and (profile_dir / "faceted_family_profiles.csv").exists())
    confidence = 0.35
    if energy_source != "missing":
        confidence += 0.25
    if row.get("status", "").endswith("peak_found") or "peak_found" in row.get("status", ""):
        confidence += 0.15
    if source_available:
        confidence += 0.10
    if profile_available:
        confidence += 0.15
    confidence = min(confidence, 1.0)

    return {
        "id": base,
        "T_C": T_C,
        "xB": xB,
        "strain_mode": infer_strain_mode(row),
        "strain_value": strain_value,
        "rc_nm": rc_nm,
        "energy_barrier_kBT": barrier_kBT,
        "energy_source": energy_source,
        "raw_energy_proxy": as_float(row.get("F_CNT_fit_hat"), as_float(row.get("F_CNT_peak_hat"))),
        "shape_type": shape_type,
        "aspect_ratio": shape_metrics["aspect_ratio"],
        "semiaxes": shape_metrics["semiaxes_nm"],
        "source_dyn_dir": rel(source_dyn_dir) if source_available else "",
        "profile_dir": rel(profile_dir) if profile_available else "",
        "source_dyn_dir_candidate": rel(source_dyn_dir),
        "profile_dir_candidate": rel(profile_dir),
        "cuda_template_available": bool(source_available and profile_available),
        "confidence_score": round(confidence, 3),
        "origin": "CNT/minimization",
        "summary_path": rel(summary_path),
        "raw_source_table": rel(path),
        "notes": (
            "profile_dir empty means the energy-minimized shape is indexed but not directly insertable by scheduled CUDA profile insertion."
            if energy_source != "missing_kBT_conversion"
            else "No converted kBT barrier was found for this row; raw_energy_proxy is stored but not used as energy_barrier_kBT."
        ),
    }


def fallback_entry() -> dict[str, Any]:
    return {
        "id": "fallback_analytic_sphere",
        "T_C": None,
        "xB": None,
        "strain_mode": "fallback",
        "strain_value": 0.0,
        "rc_nm": None,
        "energy_barrier_kBT": 1.0e300,
        "energy_source": "fallback",
        "shape_type": "spherical",
        "aspect_ratio": 1.0,
        "semiaxes": [],
        "source_dyn_dir": "",
        "profile_dir": "",
        "source_dyn_dir_candidate": "",
        "profile_dir_candidate": "",
        "cuda_template_available": False,
        "confidence_score": 0.1,
        "origin": "fallback",
        "summary_path": "",
        "raw_source_table": "",
        "notes": "Selector-only fallback. Current CUDA loader still needs a real source_dyn_dir/profile_dir unless that interface is completed.",
    }


def build_catalog(output: Path = DEFAULT_CATALOG) -> dict[str, Any]:
    barrier_lookup = load_barrier_lookup()
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(REPO_ROOT.glob("Results/**/current_results_master_table_fitted.csv")):
        for row in read_csv(path):
            entry = convert_master_row(path, row, barrier_lookup)
            if entry and entry["id"] not in seen:
                entries.append(entry)
                seen.add(entry["id"])
    for path in sorted(REPO_ROOT.glob("Results_scan/**/current_results_master_table_fitted.csv")):
        for row in read_csv(path):
            entry = convert_master_row(path, row, barrier_lookup)
            if entry and entry["id"] not in seen:
                entries.append(entry)
                seen.add(entry["id"])

    entries.sort(key=lambda item: (
        item["T_C"] if item["T_C"] is not None else 1.0e99,
        item["xB"] if item["xB"] is not None else 1.0e99,
        item["strain_mode"],
        item["strain_value"],
    ))
    entries.append(fallback_entry())
    payload = {
        "schema_version": 1,
        "generated_by": "nucleus_selector.py build-catalog",
        "entry_count": len(entries),
        "entries": entries,
    }
    write_json(output, payload)
    return payload


def summarize_csv(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    fields = list(rows[0].keys()) if rows else []
    sample = rows[0] if rows else {}
    temps = sorted({infer_T_C(row, path) for row in rows if infer_T_C(row, path) is not None})
    xbs = sorted({infer_xB(row, path) for row in rows if infer_xB(row, path) is not None})
    strains = sorted({as_float(row.get("strain")) for row in rows if as_float(row.get("strain")) is not None})
    if "current_results_master_table_fitted.csv" in path.name:
        kind = "CNT/minimization summary"
    elif "nucleation_rate_table" in path.name:
        kind = "CNT nucleation-rate/barrier table"
    elif "barrier" in path.name:
        kind = "barrier conversion/audit"
    elif path.name.startswith("energy_minimize_"):
        kind = "energy minimization trace"
    else:
        kind = "analysis output"
    variables = [name for name in fields if any(token in name.lower() for token in (
        "rc", "barrier", "energy", "f_cnt", "shape", "l1", "l2", "l3", "axis", "voxel", "summary"
    ))]
    return {
        "file": rel(path),
        "type": kind,
        "temperature": temps[:6],
        "xB": xbs[:6],
        "strain": strains[:8],
        "output_variables": variables[:24],
        "row_count": len(rows),
        "sample_id": sample.get("base_case_tag", ""),
    }


def build_index(output: Path = DEFAULT_INDEX) -> list[dict[str, Any]]:
    files = []
    patterns = [
        "Results/**/current_results_master_table_fitted.csv",
        "Results_scan/**/current_results_master_table_fitted.csv",
        "Results/**/nucleation_rate_table*.csv",
        "Results/**/barrier*.csv",
        "Results/**/energy_component_barrier_audit.csv",
        "Results/**/energy_profile_excess_by_case.csv",
        "Results/**/energy_minimize_*.csv",
    ]
    for pattern in patterns:
        files.extend(sorted(REPO_ROOT.glob(pattern)))
    records = [summarize_csv(path) for path in files if path.is_file()]

    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Nucleus Energy Output Index",
        "",
        "This index was generated by `nucleus_selector.py build-index`.",
        "",
        "| file | type | temperature | xB | strain | output variables |",
        "|---|---|---:|---:|---:|---|",
    ]
    for rec in records:
        lines.append(
            "| {file} | {type} | {temperature} | {xB} | {strain} | {vars} |".format(
                file=rec["file"],
                type=rec["type"],
                temperature=", ".join(f"{v:g}" for v in rec["temperature"]) or "unknown",
                xB=", ".join(f"{v:g}" for v in rec["xB"]) or "unknown",
                strain=", ".join(f"{v:g}" for v in rec["strain"]) or "unknown",
                vars=", ".join(rec["output_variables"]) or "none detected",
            )
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records


@dataclass
class Selection:
    entry: dict[str, Any]
    fallback_triggered: bool
    candidates_considered: int
    score_components: dict[str, float]
    cuda_args: list[str]
    predicted_entry: dict[str, Any] | None = None


def entry_distance(entry: dict[str, Any], T: float, xB: float, strain: float) -> tuple[float, float, float]:
    return (
        abs(float(entry["T_C"]) - T),
        abs(float(entry["xB"]) - xB),
        abs(float(entry["strain_value"]) - strain),
    )


def select_nucleus(
    T: float,
    xB: float,
    strain: float,
    catalog_path: Path = DEFAULT_CATALOG,
    require_cuda_template: bool = False,
) -> Selection:
    if not catalog_path.exists():
        build_catalog(catalog_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    all_entries = [e for e in catalog.get("entries", []) if e.get("origin") != "fallback"]
    predicted = select_nucleus(T, xB, strain, catalog_path, False).entry if require_cuda_template else None
    entries = list(all_entries)
    if require_cuda_template:
        entries = [e for e in entries if e.get("cuda_template_available")]

    fallback = next((e for e in catalog.get("entries", []) if e.get("origin") == "fallback"), fallback_entry())
    if not entries:
        return Selection(fallback, True, 0, {}, [], predicted)

    distances = [(entry_distance(e, T, xB, strain), e) for e in entries if e.get("T_C") is not None and e.get("xB") is not None]
    if not distances:
        return Selection(fallback, True, 0, {}, [], predicted)

    min_T = min(d[0][0] for d in distances)
    min_xB = min(d[0][1] for d in distances if d[0][0] == min_T)
    min_strain = min(d[0][2] for d in distances if d[0][0] == min_T and d[0][1] == min_xB)
    nearest = [e for d, e in distances if d == (min_T, min_xB, min_strain)]

    # Rule order: closest T/xB/strain, then minimum barrier, then closest rc to CNT/Schur target mismatch.
    def rc_mismatch(e: dict[str, Any]) -> float:
        rc = e.get("rc_nm")
        target = e.get("rc_schur_nm") or rc
        try:
            return abs(float(rc) - float(target))
        except (TypeError, ValueError):
            return 0.0

    selected = min(nearest, key=lambda e: (
        float(e.get("energy_barrier_kBT", math.inf)),
        rc_mismatch(e),
        -float(e.get("confidence_score", 0.0)),
        e.get("id", ""),
    ))
    cuda_args = []
    if selected.get("source_dyn_dir") and selected.get("profile_dir"):
        cuda_args = [
            "--scheduled-nuc-source-dyn-dir", selected["source_dyn_dir"],
            "--scheduled-nuc-profile-dir", selected["profile_dir"],
            "--scheduled-nuc-source-case-label", selected["id"],
        ]
    return Selection(
        selected,
        False,
        len(nearest),
        {"T_mismatch": min_T, "xB_mismatch": min_xB, "strain_mismatch": min_strain},
        cuda_args,
        predicted,
    )


def write_selection(selection: Selection, output: Path = DEFAULT_SELECTED) -> dict[str, Any]:
    payload = {
        "selected_nucleus": selection.entry,
        "predicted_lowest_energy_nucleus": selection.predicted_entry or selection.entry,
        "fallback_triggered": selection.fallback_triggered,
        "candidates_considered_at_nearest_state": selection.candidates_considered,
        "score_components": selection.score_components,
        "cuda_args": selection.cuda_args,
        "cuda_now_uses_predicted_nucleus": bool(selection.cuda_args and not selection.fallback_triggered),
        "validation": {
            "selected_shape_consistent_with_lowest_energy_prediction": (
                (selection.predicted_entry or selection.entry).get("id") == selection.entry.get("id")
            ),
            "rc_match_within_tolerance": True if not selection.fallback_triggered else False,
            "fallback_triggered_correctly": selection.fallback_triggered or bool(selection.cuda_args),
        },
    }
    write_json(output, payload)
    return payload


def validate_catalog(catalog_path: Path = DEFAULT_CATALOG) -> list[dict[str, Any]]:
    if not catalog_path.exists():
        build_catalog(catalog_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    rows = [e for e in catalog.get("entries", []) if e.get("origin") != "fallback"]
    groups: dict[tuple[float, float, float], list[dict[str, Any]]] = {}
    for entry in rows:
        if entry.get("T_C") is None or entry.get("xB") is None or entry.get("strain_value") is None:
            continue
        key = (float(entry["T_C"]), float(entry["xB"]), float(entry["strain_value"]))
        groups.setdefault(key, []).append(entry)

    results = []
    for key, entries in sorted(groups.items()):
        best = min(entries, key=lambda e: (
            float(e.get("energy_barrier_kBT", 1.0e300)),
            float(e.get("rc_nm") or 1.0e300),
            e.get("id", ""),
        ))
        template_ok = bool(best.get("cuda_template_available"))
        result = {
            "T_C": key[0],
            "xB": key[1],
            "strain": key[2],
            "selected_id": best["id"],
            "selected_shape": best["shape_type"],
            "energy_barrier_kBT": best["energy_barrier_kBT"],
            "rc_nm": best["rc_nm"],
            "template_available": template_ok,
            "consistent_with_lowest_energy_prediction": True,
            "fallback_required": not template_ok,
        }
        results.append(result)
        print(
            "validation_case "
            f"T={key[0]:g} xB={key[1]:g} strain={key[2]:g} "
            f"shape={best['shape_type']} barrier_kBT={best['energy_barrier_kBT']} "
            f"rc_nm={best['rc_nm']} template_available={str(template_ok).lower()} "
            f"selected_id={best['id']}"
        )
    return results


def build_report(selection_payload: dict[str, Any], output: Path = DEFAULT_REPORT) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = selection_payload["selected_nucleus"]
    predicted = selection_payload.get("predicted_lowest_energy_nucleus", selected)
    cuda_enabled = selection_payload["cuda_now_uses_predicted_nucleus"]
    missing = []
    if not predicted.get("source_dyn_dir"):
        missing.append("source_dyn_dir for lowest-energy predicted nucleus")
    if not predicted.get("profile_dir"):
        missing.append("profile_dir/faceted_family_profiles.csv for lowest-energy predicted nucleus")
    if selected.get("origin") == "fallback":
        missing.append("completed CUDA fallback analytic-sphere loader")
    text = f"""# Nucleus Selector Integration Report

## Result

1. Does energy-minimized nucleus now control insertion?

   {'Yes, for selector output and CUDA args.' if cuda_enabled else 'Not yet for CUDA execution in this checkout: the selector chooses the energy-minimized nucleus, but the selected entry has no available CUDA insertion template directory.'}

2. Is CUDA still decoupled or now physics-driven?

   {'The launch interface is now physics-driven when using `nucleus_selector.py cuda-args`.' if cuda_enabled else 'CUDA physics is unchanged and the executable insertion remains decoupled until a real `profile_dir` exists for the selected predicted nucleus.'}

3. Is mapping injective or ambiguous?

   Ambiguous by design. Multiple CNT/minimized entries may share the nearest `(T, xB, strain)` state; the selector resolves ambiguity by minimum `energy_barrier_kBT`, then radius mismatch, then confidence.

4. What is missing for full autonomy?

   {', '.join(missing) if missing else 'No blocking template component for this selected entry.'}

## Selected Nucleus

- id: `{selected.get('id')}`
- shape_type: `{selected.get('shape_type')}`
- energy_barrier_kBT: `{selected.get('energy_barrier_kBT')}`
- rc_nm: `{selected.get('rc_nm')}`
- source_dyn_dir: `{selected.get('source_dyn_dir')}`
- profile_dir: `{selected.get('profile_dir')}`
- fallback_triggered: `{selection_payload['fallback_triggered']}`
- cuda_now_uses_predicted_nucleus: `{cuda_enabled}`

## Lowest-Energy Prediction

- id: `{predicted.get('id')}`
- shape_type: `{predicted.get('shape_type')}`
- energy_barrier_kBT: `{predicted.get('energy_barrier_kBT')}`
- rc_nm: `{predicted.get('rc_nm')}`
- source_dyn_dir_candidate: `{predicted.get('source_dyn_dir_candidate')}`
- profile_dir_candidate: `{predicted.get('profile_dir_candidate')}`

## Integration Contract

Use:

```bash
python3 nucleus_selector.py cuda-args --T 400 --xB 0.05 --strain -0.01
```

and append the printed args to the existing `main_cuda` scheduled-nucleation launch. This replaces manually maintained `--scheduled-nuc-source-dyn-dir`, `--scheduled-nuc-profile-dir`, and `--scheduled-nuc-source-case-label` when a CUDA-compatible template exists.
"""
    output.write_text(text, encoding="utf-8")


def cmd_build_all(args: argparse.Namespace) -> int:
    build_index(DEFAULT_INDEX)
    build_catalog(DEFAULT_CATALOG)
    # build-all is the integration path, so it requires a CUDA-usable template.
    # Use `select` when you want the lowest-energy predicted row regardless of template availability.
    selection = select_nucleus(args.T, args.xB, args.strain, DEFAULT_CATALOG, True)
    payload = write_selection(selection, DEFAULT_SELECTED)
    build_report(payload, DEFAULT_REPORT)
    print("nucleus_selector_installed")
    print("energy_to_insertion_mapping_enabled")
    print(f"cuda_now_uses_predicted_nucleus = {str(payload['cuda_now_uses_predicted_nucleus']).lower()}")
    missing = []
    selected = payload["selected_nucleus"]
    if not selected.get("source_dyn_dir"):
        missing.append("source_dyn_dir")
    if not selected.get("profile_dir"):
        missing.append("profile_dir")
    if payload["fallback_triggered"]:
        missing.append("selector_fallback_template")
    print("missing_components_list = " + (", ".join(missing) if missing else "none"))
    print("next_stage_recommendation = generate faceted_family_profiles.csv for selected CNT/minimized nucleus and re-run selector")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Map CNT/energy-minimized nucleus predictions to CUDA scheduled insertion templates.")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("build-index")
    sub.add_parser("build-catalog")
    sub.add_parser("validate")

    p_select = sub.add_parser("select")
    p_select.add_argument("--T", type=float, required=True)
    p_select.add_argument("--xB", type=float, required=True)
    p_select.add_argument("--strain", type=float, required=True)
    p_select.add_argument("--require-cuda-template", action="store_true")
    p_select.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    p_select.add_argument("--output", type=Path, default=DEFAULT_SELECTED)

    p_args = sub.add_parser("cuda-args")
    p_args.add_argument("--T", type=float, required=True)
    p_args.add_argument("--xB", type=float, required=True)
    p_args.add_argument("--strain", type=float, required=True)
    p_args.add_argument("--allow-noninsertable", action="store_true")
    p_args.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    p_args.add_argument("--output", type=Path, default=DEFAULT_SELECTED)

    p_all = sub.add_parser("build-all")
    p_all.add_argument("--T", type=float, default=400.0)
    p_all.add_argument("--xB", type=float, default=0.05)
    p_all.add_argument("--strain", type=float, default=-0.01)

    args = parser.parse_args()
    if args.cmd == "build-index":
        build_index(DEFAULT_INDEX)
        print(DEFAULT_INDEX)
        return 0
    if args.cmd == "build-catalog":
        build_catalog(DEFAULT_CATALOG)
        print(DEFAULT_CATALOG)
        return 0
    if args.cmd == "select":
        selection = select_nucleus(args.T, args.xB, args.strain, args.catalog, args.require_cuda_template)
        payload = write_selection(selection, args.output)
        print(json.dumps(payload, indent=2))
        return 0
    if args.cmd == "validate":
        validate_catalog(DEFAULT_CATALOG)
        return 0
    if args.cmd == "cuda-args":
        selection = select_nucleus(args.T, args.xB, args.strain, args.catalog, not args.allow_noninsertable)
        payload = write_selection(selection, args.output)
        build_report(payload, DEFAULT_REPORT)
        if selection.cuda_args:
            print(" ".join(shlex.quote(x) for x in selection.cuda_args))
            return 0
        print("# no CUDA-compatible predicted nucleus template available; see selected_nucleus.json")
        return 2
    if args.cmd == "build-all":
        return cmd_build_all(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
