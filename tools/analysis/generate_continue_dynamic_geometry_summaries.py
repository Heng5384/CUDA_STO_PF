#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_repo_path(repo_root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return repo_root / p


def _parse_pf_params(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            out[key] = float(value)
        except ValueError:
            continue
    return out


def _read_legacy_scalar_vtk(path: Path) -> tuple[np.ndarray, tuple[int, int, int], tuple[float, float, float]]:
    dims: tuple[int, int, int] | None = None
    spacing = (1.0, 1.0, 1.0)

    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF before data in {path}")
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("DIMENSIONS"):
                _, sx, sy, sz = text.split()
                dims = (int(sx), int(sy), int(sz))
            elif text.startswith("SPACING"):
                _, sx, sy, sz = text.split()
                spacing = (float(sx), float(sy), float(sz))
            elif text.startswith("ASPECT_RATIO"):
                _, sx, sy, sz = text.split()
                spacing = (float(sx), float(sy), float(sz))
            elif text.startswith("LOOKUP_TABLE"):
                break

        if dims is None:
            raise ValueError(f"Missing DIMENSIONS in {path}")

        payload = f.read().decode("ascii", errors="replace")

    values = np.fromstring(payload, sep=" ", dtype=np.float64)
    expected = dims[0] * dims[1] * dims[2]
    if values.size != expected:
        raise ValueError(f"VTK scalar count mismatch in {path}: got {values.size}, expected {expected}")
    return values.reshape(dims, order="C"), dims, spacing


def _canonicalize_axis_sign(v: np.ndarray) -> np.ndarray:
    out = np.array(v, dtype=float, copy=True)
    for i in range(3):
        if abs(out[i]) > 1.0e-14:
            if out[i] < 0.0:
                out *= -1.0
            break
    return out


def _angle_deg_abs(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1.0e-14 or nb < 1.0e-14:
        return 0.0
    c = abs(float(np.dot(a, b) / (na * nb)))
    c = min(1.0, max(0.0, c))
    return math.degrees(math.acos(c))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    nx, ny, nz = mask.shape
    visited = np.zeros(mask.shape, dtype=np.uint8)
    largest: list[tuple[int, int, int]] = []
    neighbors = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]

    seeds = np.argwhere(mask)
    for sx, sy, sz in seeds:
        if visited[sx, sy, sz]:
            continue
        q: deque[tuple[int, int, int]] = deque()
        q.append((int(sx), int(sy), int(sz)))
        visited[sx, sy, sz] = 1
        comp: list[tuple[int, int, int]] = []
        while q:
            x, y, z = q.popleft()
            comp.append((x, y, z))
            for dx, dy, dz in neighbors:
                xn = x + dx
                yn = y + dy
                zn = z + dz
                if xn < 0 or yn < 0 or zn < 0 or xn >= nx or yn >= ny or zn >= nz:
                    continue
                if not mask[xn, yn, zn] or visited[xn, yn, zn]:
                    continue
                visited[xn, yn, zn] = 1
                q.append((xn, yn, zn))
        if len(comp) > len(largest):
            largest = comp

    selected = np.zeros(mask.shape, dtype=bool)
    if largest:
        idx = np.array(largest, dtype=int)
        selected[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return selected


def _compute_geometry(phi: np.ndarray, dx: float, dy: float, dz: float, threshold: float) -> dict[str, object] | None:
    phi_clamped = np.clip(phi, 0.0, 1.0)
    mask = phi_clamped > threshold
    if not mask.any():
        return None

    coords_all = np.argwhere(mask)
    mins = coords_all.min(axis=0)
    maxs = coords_all.max(axis=0)

    crop_phi = phi_clamped[mins[0]:maxs[0] + 1, mins[1]:maxs[1] + 1, mins[2]:maxs[2] + 1]
    crop_mask = mask[mins[0]:maxs[0] + 1, mins[1]:maxs[1] + 1, mins[2]:maxs[2] + 1]
    selected_crop = _largest_component(crop_mask)
    if not selected_crop.any():
        return None

    selected_local = np.argwhere(selected_crop)
    selected_global = selected_local + mins

    center = np.array([
        np.mean(selected_global[:, 0] * dx),
        np.mean(selected_global[:, 1] * dy),
        np.mean(selected_global[:, 2] * dz),
    ], dtype=float)

    rel = np.column_stack((
        selected_global[:, 0] * dx - center[0],
        selected_global[:, 1] * dy - center[1],
        selected_global[:, 2] * dz - center[2],
    ))
    cov = np.cov(rel, rowvar=False, bias=True)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    principal_axes = []
    full_axes = []
    for i in range(3):
        v = eigvecs[:, i]
        n = np.linalg.norm(v)
        if n > 1.0e-14:
            v = v / n
        v = _canonicalize_axis_sign(v)
        principal_axes.append(v)
        full_axes.append(2.0 * math.sqrt(max(5.0 * float(eigvals[i]), 0.0)))
    principal_axes_arr = np.array(principal_axes, dtype=float)
    full_axes_arr = np.array(full_axes, dtype=float)

    bbox_lengths = np.array([
        (maxs[0] - mins[0] + 1) * dx,
        (maxs[1] - mins[1] + 1) * dy,
        (maxs[2] - mins[2] + 1) * dz,
    ], dtype=float)

    gx, gy, gz = np.gradient(crop_phi, dx, dy, dz, edge_order=1)
    boundary_points: list[np.ndarray] = []
    boundary_normals: list[np.ndarray] = []
    nx, ny, nz = selected_crop.shape
    neighbors6 = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    for lx, ly, lz in selected_local:
        is_boundary = False
        for ddx, ddy, ddz in neighbors6:
            xn = lx + ddx
            yn = ly + ddy
            zn = lz + ddz
            if xn < 0 or yn < 0 or zn < 0 or xn >= nx or yn >= ny or zn >= nz or not selected_crop[xn, yn, zn]:
                is_boundary = True
                break
        if not is_boundary:
            continue
        normal = np.array([-gx[lx, ly, lz], -gy[lx, ly, lz], -gz[lx, ly, lz]], dtype=float)
        nrm = np.linalg.norm(normal)
        if nrm < 1.0e-14:
            continue
        normal /= nrm
        global_xyz = np.array([(lx + mins[0]) * dx, (ly + mins[1]) * dy, (lz + mins[2]) * dz], dtype=float)
        boundary_points.append(global_xyz)
        boundary_normals.append(normal)

    if not boundary_points:
        return None

    boundary_points_arr = np.array(boundary_points, dtype=float)
    boundary_normals_arr = np.array(boundary_normals, dtype=float)
    face_axis_ids = [0, 0, 1, 1, 2, 2]
    face_signs = [+1, -1, +1, -1, +1, -1]
    face_thickness_frac = 0.10
    face_radius_frac = 0.35
    face_valid = [False] * 6
    face_points = np.zeros((6, 3), dtype=float)
    face_normals = np.zeros((6, 3), dtype=float)

    rel_boundary = boundary_points_arr - center
    q_all = np.stack([rel_boundary @ principal_axes_arr[i] for i in range(3)], axis=1)
    for face in range(6):
        axis_id = face_axis_ids[face]
        face_sign = face_signs[face]
        q_axis = q_all[:, axis_id]
        q_ext = np.max(q_axis) if face_sign > 0 else np.min(q_axis)
        r_ref = face_radius_frac * min(full_axes_arr[(axis_id + 1) % 3], full_axes_arr[(axis_id + 2) % 3])

        if face_sign > 0:
            sel_axis = q_axis > (q_ext - face_thickness_frac * full_axes_arr[axis_id])
        else:
            sel_axis = q_axis < (q_ext + face_thickness_frac * full_axes_arr[axis_id])
        transverse = np.sqrt(q_all[:, (axis_id + 1) % 3] ** 2 + q_all[:, (axis_id + 2) % 3] ** 2)
        axial_candidates = np.where(sel_axis)[0]
        selected = np.where(sel_axis & (transverse < r_ref))[0]
        if selected.size < 20:
            selected = axial_candidates
        if selected.size < 5:
            continue

        mean_point = np.mean(boundary_points_arr[selected], axis=0)
        mean_normal = np.mean(boundary_normals_arr[selected], axis=0)
        nrm = np.linalg.norm(mean_normal)
        if nrm < 1.0e-14:
            continue
        mean_normal /= nrm
        face_valid[face] = True
        face_points[face] = mean_point
        face_normals[face] = mean_normal

    return {
        "threshold": threshold,
        "component_count": 1,
        "chosen_component": 1,
        "voxel_count": int(selected_local.shape[0]),
        "boundary_voxel_count": int(boundary_points_arr.shape[0]),
        "center": center,
        "bbox_lengths": bbox_lengths,
        "principal_axes": principal_axes_arr,
        "full_axes": full_axes_arr,
        "face_valid": face_valid,
        "face_points": face_points,
        "face_normals": face_normals,
        "grid": phi.shape,
        "spacing": (dx, dy, dz),
    }


def _pick_phi_vtk(cont_dir: Path, nsteps: int) -> Path | None:
    preferred = cont_dir / f"phi_{nsteps}.vtk"
    if preferred.exists():
        return preferred
    numeric = []
    for p in cont_dir.glob("phi_*.vtk"):
        stem = p.stem
        if stem == "phi_init":
            continue
        suffix = stem.removeprefix("phi_")
        if suffix.isdigit():
            numeric.append((int(suffix), p))
    if numeric:
        numeric.sort()
        return numeric[-1][1]
    init = cont_dir / "phi_init.vtk"
    return init if init.exists() else None


def _write_summary(path: Path, phi_vtk: Path, geom: dict[str, object], mode_label: str) -> None:
    ref_axes = np.eye(3, dtype=float)
    ref_names = ("x", "y", "z")
    face_names = ("+long face", "-long face", "+mid face", "-mid face", "+short face", "-short face")
    center = geom["center"]
    bbox = geom["bbox_lengths"]
    axes = geom["principal_axes"]
    full_axes = geom["full_axes"]
    face_valid = geom["face_valid"]
    face_points = geom["face_points"]
    face_normals = geom["face_normals"]
    nx, ny, nz = geom["grid"]
    dx, dy, dz = geom["spacing"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write("==================== Analysis Summary ====================\n")
        fp.write(f"summary_file              : {path}\n")
        fp.write(f"phi_vtk_file              : {phi_vtk}\n")
        fp.write(f"mode                      : {mode_label}\n")
        fp.write(f"grid_dimensions           : ({nx}, {ny}, {nz})\n")
        fp.write(f"spacing_sim_units         : ({dx:.6f}, {dy:.6f}, {dz:.6f})\n")
        fp.write(f"threshold_mode            : phi > {geom['threshold']:.3f}\n")
        fp.write(f"connected_components      : {geom['component_count']}\n")
        fp.write(f"chosen_component          : {geom['chosen_component']}\n")
        fp.write(f"voxel_count               : {geom['voxel_count']}\n")
        fp.write(f"boundary_voxel_count      : {geom['boundary_voxel_count']}\n")
        fp.write(f"center_of_mass            : [{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]\n")
        fp.write(f"bbox_length_xyz           : [{bbox[0]:.6f}, {bbox[1]:.6f}, {bbox[2]:.6f}]\n")
        fp.write("\n--- Principal axes (unit vectors) ---\n")
        fp.write(f"long_axis                 : [{axes[0][0]:.6f}, {axes[0][1]:.6f}, {axes[0][2]:.6f}]\n")
        fp.write(f"mid_axis                  : [{axes[1][0]:.6f}, {axes[1][1]:.6f}, {axes[1][2]:.6f}]\n")
        fp.write(f"short_axis                : [{axes[2][0]:.6f}, {axes[2][1]:.6f}, {axes[2][2]:.6f}]\n")
        fp.write("\n--- Principal lengths (covariance estimate) ---\n")
        fp.write(f"L1_long                   : {full_axes[0]:.6e}\n")
        fp.write(f"L2_mid                    : {full_axes[1]:.6e}\n")
        fp.write(f"L3_short                  : {full_axes[2]:.6e}\n")
        fp.write(f"L1/L3                     : {(full_axes[0] / full_axes[2]) if full_axes[2] > 0 else 0.0:.6f}\n")
        fp.write(f"L2/L3                     : {(full_axes[1] / full_axes[2]) if full_axes[2] > 0 else 0.0:.6f}\n")
        fp.write(f"L1/L2                     : {(full_axes[0] / full_axes[1]) if full_axes[1] > 0 else 0.0:.6f}\n")
        fp.write("\n--- Axis angles (deg, abs dot) ---\n")
        axis_names = ("long", "mid", "short")
        for axis_name, axis_vec in zip(axis_names, axes):
            for ref_name, ref_vec in zip(ref_names, ref_axes):
                fp.write(f"{axis_name}_vs_{ref_name}                 : {_angle_deg_abs(axis_vec, ref_vec):.3f}\n")
        fp.write("\n==================== Representative Face Normals ====================\n")
        for i, face_name in enumerate(face_names):
            fp.write(f"\n{face_name}\n")
            if not face_valid[i]:
                fp.write("  [warning] not enough boundary points selected.\n")
                continue
            fp.write(f"  mean point              : [{face_points[i][0]:.6f}, {face_points[i][1]:.6f}, {face_points[i][2]:.6f}]\n")
            fp.write(f"  mean normal             : [{face_normals[i][0]:.6f}, {face_normals[i][1]:.6f}, {face_normals[i][2]:.6f}]\n")
            fp.write(f"  angle with x            : {_angle_deg_abs(face_normals[i], ref_axes[0]):.3f}\n")
            fp.write(f"  angle with y            : {_angle_deg_abs(face_normals[i], ref_axes[1]):.3f}\n")
            fp.write(f"  angle with z            : {_angle_deg_abs(face_normals[i], ref_axes[2]):.3f}\n")
            fp.write(f"  angle with long axis    : {_angle_deg_abs(face_normals[i], axes[0]):.3f}\n")
            fp.write(f"  angle with mid axis     : {_angle_deg_abs(face_normals[i], axes[1]):.3f}\n")
            fp.write(f"  angle with short axis   : {_angle_deg_abs(face_normals[i], axes[2]):.3f}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate geometry summary.txt files for continue-dynamic results by reading VTK files from guide_continue_dynamic.csv.")
    parser.add_argument("--guide-csv", type=Path, required=True, help="guide_continue_dynamic.csv")
    parser.add_argument("--repo-root", "--results-root", dest="repo_root", type=Path, default=Path("."), help="repo root containing Results/")
    parser.add_argument("--threshold", type=float, default=0.5, help="phi threshold used for nucleus masking")
    parser.add_argument("--overwrite", action="store_true", help="rebuild summary.txt even if it already exists")
    args = parser.parse_args()

    guide_csv = args.guide_csv.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    rows = list(csv.DictReader(guide_csv.open(newline="", encoding="utf-8")))

    built = 0
    skipped = 0
    failed = 0
    for idx, row in enumerate(rows, start=1):
        summary_path = _resolve_repo_path(repo_root, row["continue_summary_rel"])
        cont_dir = summary_path.parent
        if summary_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            nsteps = int(row["nsteps"])
            phi_vtk = _pick_phi_vtk(cont_dir, nsteps)
            if phi_vtk is None or not phi_vtk.exists():
                raise FileNotFoundError(f"no continue dynamic phi vtk in {cont_dir}")

            pf_params = _parse_pf_params(cont_dir / "pf_input.params")
            phi, dims, vtk_spacing = _read_legacy_scalar_vtk(phi_vtk)
            dx = float(pf_params.get("dx", vtk_spacing[0]))
            dy = float(pf_params.get("dy", vtk_spacing[1]))
            dz = float(pf_params.get("dz", vtk_spacing[2]))

            geom = _compute_geometry(phi, dx, dy, dz, args.threshold)
            if geom is None:
                raise RuntimeError("no valid nucleus geometry found above threshold")

            _write_summary(summary_path, phi_vtk, geom, "dynamic")
            built += 1
            print(f"[built] row={idx} base={row['base_case_tag']} summary={summary_path}")
        except Exception as exc:
            failed += 1
            print(f"[failed] row={idx} base={row.get('base_case_tag', '')} error={exc}")

    print(f"built={built}")
    print(f"skipped={skipped}")
    print(f"failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
