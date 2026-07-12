#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATE_ROOT = REPO_ROOT / "Results" / "nucleus_templates"
DEFAULT_CONTINUOUS_CSV = REPO_ROOT / "continuous_nucleus_geometry.csv"
DEFAULT_TEMPLATE_SPACE = REPO_ROOT / "template_geometry_space.json"
DEFAULT_VALIDATION_LOG = REPO_ROOT / "template_mapping_validation_log.csv"


def rel(path: Path | str | None) -> str:
    if path is None:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def repo_path(text: str | None) -> Path | None:
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def ellipsoid_volume(semiaxes: list[float]) -> float:
    a, b, c = normalize_semiaxes(semiaxes)
    return 4.0 * math.pi * a * b * c / 3.0


def ellipsoid_surface_area(semiaxes: list[float]) -> float:
    a, b, c = normalize_semiaxes(semiaxes)
    p = 1.6075
    return 4.0 * math.pi * (((a**p * b**p) + (a**p * c**p) + (b**p * c**p)) / 3.0) ** (1.0 / p)


def normalize_semiaxes(semiaxes: list[Any] | None, rc_nm: float | None = None) -> list[float]:
    vals = [safe_float(x) for x in (semiaxes or [])]
    vals = [x for x in vals if x is not None and x > 0.0]
    if len(vals) >= 3:
        vals = sorted(vals[:3], reverse=True)
        return [float(vals[0]), float(vals[1]), float(vals[2])]
    r = float(rc_nm or 1.0)
    return [r, r, r]


def aspect_from_semiaxes(semiaxes: list[float]) -> float:
    a, _b, c = normalize_semiaxes(semiaxes)
    return a / max(c, 1.0e-12)


def feature_vector(rc_nm: float, semiaxes: list[float], shape_type: str) -> list[float]:
    a, b, c = normalize_semiaxes(semiaxes, rc_nm)
    aspect = a / max(c, 1.0e-12)
    volume = ellipsoid_volume([a, b, c])
    stv = ellipsoid_surface_area([a, b, c]) / max(volume, 1.0e-12)
    shape_code = {"spherical": 0.0, "ellipsoidal": 0.4, "anisotropic": 0.65, "faceted": 0.85, "planar": 1.0}.get(shape_type, 0.5)
    return [float(rc_nm), aspect, b / max(c, 1.0e-12), stv, shape_code]


@dataclass
class ContinuousNucleus:
    nucleus_id: str
    rc_nm: float
    shape_type: str
    aspect_ratio: float
    semiaxes_nm: list[float]
    volume_nm3: float
    surface_to_volume_nm_inv: float
    composition_profile: str
    source: str
    raw: dict[str, Any]


@dataclass
class TemplateGeometry:
    template_id: str
    shape_type: str
    aspect_ratios: list[float]
    semiaxes_nm: list[float]
    rc_range_validity: list[float]
    geometry_vector: list[float]
    profile_dir: str
    source_dyn_dir: str
    generated: bool = False


def entry_to_continuous(entry: dict[str, Any], source: str) -> ContinuousNucleus | None:
    rc = safe_float(entry.get("rc_nm"))
    if rc is None or rc <= 0.0:
        return None
    semiaxes = normalize_semiaxes(entry.get("semiaxes"), rc)
    aspect = safe_float(entry.get("aspect_ratio"), aspect_from_semiaxes(semiaxes)) or 1.0
    volume = ellipsoid_volume(semiaxes)
    stv = ellipsoid_surface_area(semiaxes) / max(volume, 1.0e-12)
    return ContinuousNucleus(
        nucleus_id=str(entry.get("id", "unknown_nucleus")),
        rc_nm=float(rc),
        shape_type=str(entry.get("shape_type") or "spherical"),
        aspect_ratio=float(aspect),
        semiaxes_nm=semiaxes,
        volume_nm3=volume,
        surface_to_volume_nm_inv=stv,
        composition_profile=str(entry.get("profile_dir") or entry.get("profile_dir_candidate") or ""),
        source=source,
        raw=entry,
    )


def extract_continuous_nuclei(
    selected_json: Path | None = None,
    catalog_json: Path = REPO_ROOT / "nucleus_catalog.json",
    output_csv: Path = DEFAULT_CONTINUOUS_CSV,
) -> list[ContinuousNucleus]:
    nuclei: list[ContinuousNucleus] = []
    if selected_json and selected_json.exists():
        payload = read_json(selected_json, {})
        for key in ("selected_nucleus", "predicted_lowest_energy_nucleus"):
            entry = payload.get(key)
            if isinstance(entry, dict):
                nucleus = entry_to_continuous(entry, f"{rel(selected_json)}:{key}")
                if nucleus:
                    nuclei.append(nucleus)

    catalog = read_json(catalog_json, {"entries": []})
    for entry in catalog.get("entries", []):
        if entry.get("origin") == "fallback":
            continue
        nucleus = entry_to_continuous(entry, rel(catalog_json))
        if nucleus:
            nuclei.append(nucleus)

    seen: set[str] = set()
    unique: list[ContinuousNucleus] = []
    for nucleus in nuclei:
        key = f"{nucleus.nucleus_id}:{nucleus.rc_nm:.6g}:{nucleus.shape_type}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(nucleus)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "nucleus_id", "rc_nm", "aspect_ratio", "shape_type", "semiaxes_nm",
            "volume_nm3", "surface_to_volume_nm_inv", "composition_profile", "source",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for nucleus in unique:
            writer.writerow({
                "nucleus_id": nucleus.nucleus_id,
                "rc_nm": nucleus.rc_nm,
                "aspect_ratio": nucleus.aspect_ratio,
                "shape_type": nucleus.shape_type,
                "semiaxes_nm": " ".join(f"{x:.8g}" for x in nucleus.semiaxes_nm),
                "volume_nm3": nucleus.volume_nm3,
                "surface_to_volume_nm_inv": nucleus.surface_to_volume_nm_inv,
                "composition_profile": nucleus.composition_profile,
                "source": nucleus.source,
            })
    return unique


def semiaxes_from_summary(summary_path: Path) -> list[float] | None:
    if not summary_path.exists():
        return None
    vals: dict[str, float] = {}
    for line in summary_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = [x.strip() for x in line.split(":", 1)]
        if key in {"L1_long", "L2_mid", "L3_short"}:
            vals[key] = float(value.split()[0])
    if all(k in vals for k in ("L1_long", "L2_mid", "L3_short")):
        return [0.5 * vals["L1_long"], 0.5 * vals["L2_mid"], 0.5 * vals["L3_short"]]
    return None


def classify_shape(aspect: float) -> str:
    if aspect < 1.1:
        return "spherical"
    if aspect < 1.8:
        return "anisotropic"
    return "faceted"


def build_template_space(
    template_root: Path = DEFAULT_TEMPLATE_ROOT,
    output_json: Path = DEFAULT_TEMPLATE_SPACE,
) -> list[TemplateGeometry]:
    templates: list[TemplateGeometry] = []
    for profile_csv in sorted(REPO_ROOT.rglob("faceted_family_profiles.csv")):
        profile_dir = profile_csv.parent
        source_dir = profile_dir.parent
        summary = source_dir / "summary.txt"
        semiaxes = semiaxes_from_summary(summary) or semiaxes_from_summary(profile_dir / "summary.txt")
        if semiaxes is None:
            continue
        rc = sum(semiaxes) / 3.0
        aspect = aspect_from_semiaxes(semiaxes)
        shape = classify_shape(aspect)
        templates.append(TemplateGeometry(
            template_id=rel(profile_dir),
            shape_type=shape,
            aspect_ratios=[aspect, semiaxes[1] / max(semiaxes[2], 1.0e-12)],
            semiaxes_nm=semiaxes,
            rc_range_validity=[0.8 * rc, 1.2 * rc],
            geometry_vector=feature_vector(rc, semiaxes, shape),
            profile_dir=rel(profile_dir),
            source_dyn_dir=rel(source_dir),
            generated="nucleus_templates/generated" in str(profile_dir),
        ))

    write_json(output_json, {
        "schema_version": 1,
        "template_count": len(templates),
        "templates": [asdict(t) for t in templates],
    })
    return templates


def shape_distance(nucleus: ContinuousNucleus, template: TemplateGeometry) -> dict[str, float]:
    nvec = feature_vector(nucleus.rc_nm, nucleus.semiaxes_nm, nucleus.shape_type)
    tvec = template.geometry_vector
    rc = abs(nvec[0] - tvec[0])
    aspect = abs(nvec[1] - tvec[1]) + abs(nvec[2] - tvec[2])
    feature_l2 = math.sqrt(sum((a - b) ** 2 for a, b in zip(nvec, tvec)))
    volume = abs(nucleus.volume_nm3 - ellipsoid_volume(template.semiaxes_nm)) / max(nucleus.volume_nm3, 1.0e-12)
    return {
        "rc_mismatch_nm": rc,
        "shape_distance": aspect,
        "feature_l2": feature_l2,
        "volume_relative_error": volume,
        "surface_energy_inconsistency": abs(nvec[3] - tvec[3]),
        "total": rc + 0.5 * aspect + 0.25 * feature_l2 + volume,
    }


def write_generated_template(nucleus: ContinuousNucleus, template_root: Path = DEFAULT_TEMPLATE_ROOT) -> TemplateGeometry:
    safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in nucleus.nucleus_id)
    template_id = f"generated_{safe_id}_rc{nucleus.rc_nm:.3f}".replace(".", "p")
    source_dir = template_root / "generated" / template_id / "source"
    profile_dir = template_root / "generated" / template_id / "profile"
    source_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    a, b, c = normalize_semiaxes(nucleus.semiaxes_nm, nucleus.rc_nm)
    summary = source_dir / "summary.txt"
    summary.write_text(
        "\n".join([
            f"phi_vtk_file: generated_template:{template_id}",
            "mode: generated_cuda_template",
            "grid_dimensions: (1, 1, 1)",
            "spacing_sim_units: (1, 1, 1)",
            "threshold_mode: phi > 0.5",
            "connected_components: 1",
            "chosen_component: 1",
            "voxel_count: 1",
            "center_of_mass: (0, 0, 0)",
            f"L1_long: {2.0 * a:.8g}",
            f"L2_mid: {2.0 * b:.8g}",
            f"L3_short: {2.0 * c:.8g}",
            f"L1/L3: {a / max(c, 1.0e-12):.8g}",
            f"L2/L3: {b / max(c, 1.0e-12):.8g}",
            f"L1/L2: {a / max(b, 1.0e-12):.8g}",
            "",
        ]),
        encoding="utf-8",
    )

    profile_csv = profile_dir / "faceted_family_profiles.csv"
    families = [f"{sx:+d}{sy:+d}{sz:+d}" for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    xb_inside = 1.0
    xb_outside = safe_float(nucleus.raw.get("xB"), 0.03) or 0.03
    interface_nm = max(0.2, 0.25 * nucleus.rc_nm)
    d_min = -max(2.5, 2.0 * interface_nm)
    d_max = max(2.5, 2.0 * interface_nm)
    n = 121
    with profile_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["family", "region", "u_nm", "phi_mean", "xB_mean"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for family in families:
            for i in range(n):
                d = d_min + (d_max - d_min) * i / (n - 1)
                phi = 0.5 * (1.0 - math.tanh(d / max(interface_nm, 1.0e-9)))
                xb = xb_outside + (xb_inside - xb_outside) * phi
                writer.writerow({
                    "family": family,
                    "region": "face",
                    "u_nm": f"{d:.8g}",
                    "phi_mean": f"{phi:.10g}",
                    "xB_mean": f"{xb:.10g}",
                })

    rc = sum([a, b, c]) / 3.0
    shape = nucleus.shape_type if nucleus.shape_type != "unknown" else classify_shape(aspect_from_semiaxes([a, b, c]))
    return TemplateGeometry(
        template_id=template_id,
        shape_type=shape,
        aspect_ratios=[a / max(c, 1.0e-12), b / max(c, 1.0e-12)],
        semiaxes_nm=[a, b, c],
        rc_range_validity=[0.5 * rc, 1.5 * rc],
        geometry_vector=feature_vector(rc, [a, b, c], shape),
        profile_dir=rel(profile_dir),
        source_dyn_dir=rel(source_dir),
        generated=True,
    )


def append_validation_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "nucleus_id", "rc_nm", "shape_type", "template_id", "profile_dir", "source_dyn_dir",
        "rc_mismatch_nm", "shape_distance", "feature_l2", "volume_relative_error",
        "surface_energy_inconsistency", "decision_reason", "fallback_used",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def map_continuous_nucleus_to_cuda_template(
    nucleus: ContinuousNucleus | dict[str, Any],
    template_space_json: Path = DEFAULT_TEMPLATE_SPACE,
    template_root: Path = DEFAULT_TEMPLATE_ROOT,
    validation_log: Path = DEFAULT_VALIDATION_LOG,
    max_total_mismatch: float = 1.5,
) -> dict[str, Any]:
    if isinstance(nucleus, dict):
        continuous = entry_to_continuous(nucleus, "dict")
        if continuous is None:
            continuous = ContinuousNucleus(
                nucleus_id="fallback_spherical",
                rc_nm=2.0,
                shape_type="spherical",
                aspect_ratio=1.0,
                semiaxes_nm=[2.0, 2.0, 2.0],
                volume_nm3=ellipsoid_volume([2.0, 2.0, 2.0]),
                surface_to_volume_nm_inv=ellipsoid_surface_area([2.0, 2.0, 2.0]) / ellipsoid_volume([2.0, 2.0, 2.0]),
                composition_profile="",
                source="fallback",
                raw=nucleus,
            )
    else:
        continuous = nucleus

    templates = [TemplateGeometry(**item) for item in read_json(template_space_json, {"templates": []}).get("templates", [])]
    best: tuple[TemplateGeometry, dict[str, float]] | None = None
    for template in templates:
        metrics = shape_distance(continuous, template)
        if best is None or metrics["total"] < best[1]["total"]:
            best = (template, metrics)

    fallback_used = False
    decision_reason = "nearest_existing_template"
    if best is None or best[1]["total"] > max_total_mismatch:
        generated = write_generated_template(continuous, template_root)
        templates.append(generated)
        write_json(template_space_json, {
            "schema_version": 1,
            "template_count": len(templates),
            "templates": [asdict(t) for t in templates],
        })
        best = (generated, shape_distance(continuous, generated))
        fallback_used = True
        decision_reason = "generated_intermediate_template"

    template, metrics = best
    row = {
        "nucleus_id": continuous.nucleus_id,
        "rc_nm": continuous.rc_nm,
        "shape_type": continuous.shape_type,
        "template_id": template.template_id,
        "profile_dir": template.profile_dir,
        "source_dyn_dir": template.source_dyn_dir,
        **metrics,
        "decision_reason": decision_reason,
        "fallback_used": int(fallback_used),
    }
    append_validation_log(validation_log, row)
    return {
        "continuous_nucleus": asdict(continuous),
        "mapped_template": asdict(template),
        "mismatch_metrics": metrics,
        "decision_reason": decision_reason,
        "fallback_used": fallback_used,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Map continuous CNT/minimized nucleus geometry to CUDA-compatible discrete insertion templates.")
    parser.add_argument("--selected-json", type=Path, default=REPO_ROOT / "selected_nucleus.json")
    parser.add_argument("--catalog-json", type=Path, default=REPO_ROOT / "nucleus_catalog.json")
    parser.add_argument("--continuous-csv", type=Path, default=DEFAULT_CONTINUOUS_CSV)
    parser.add_argument("--template-space-json", type=Path, default=DEFAULT_TEMPLATE_SPACE)
    parser.add_argument("--validation-log", type=Path, default=DEFAULT_VALIDATION_LOG)
    parser.add_argument("--template-root", type=Path, default=DEFAULT_TEMPLATE_ROOT)
    args = parser.parse_args()

    nuclei = extract_continuous_nuclei(args.selected_json, args.catalog_json, args.continuous_csv)
    build_template_space(args.template_root, args.template_space_json)
    selected_payload = read_json(args.selected_json, {})
    selected = selected_payload.get("predicted_lowest_energy_nucleus") or selected_payload.get("selected_nucleus")
    continuous = entry_to_continuous(selected, rel(args.selected_json)) if isinstance(selected, dict) else (nuclei[0] if nuclei else None)
    if continuous is None:
        raise SystemExit("No usable continuous nucleus found.")
    mapping = map_continuous_nucleus_to_cuda_template(
        continuous,
        template_space_json=args.template_space_json,
        template_root=args.template_root,
        validation_log=args.validation_log,
    )
    print("template_mapper_installed")
    print("continuous_to_discrete_mapping_active")
    print(f"geometry_loss_mean = {mapping['mismatch_metrics']['total']:.8g}")
    print(f"cuda_closed_loop_status = {str(bool(mapping['mapped_template'].get('profile_dir'))).lower()}")
    print("missing_template_regions = none_generated_fallback_used" if mapping["fallback_used"] else "missing_template_regions = none")
    print("next_stage_recommendation = validate generated profiles against minimized VTK-derived faceted profiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
