#!/usr/bin/env python3
"""Independent small-grid FP64 oracle for the production spectral mechanics.

The implementation intentionally does not call CUDA_STO_PF mechanics code.  It
reconstructs the same discrete operator from the frozen parameter/input files:
periodic R2C wave numbers, strict component-wise 2/3 truncation, engineering
Voigt shear convention, fixed mean strain, h(phi) stiffness interpolation, and
the Green fixed-point iteration.  It is a diagnostic oracle, never a runtime
solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def h_phi(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def hp_phi(phi: np.ndarray) -> np.ndarray:
    return 30.0 * phi**2 * (phi - 1.0) ** 2


def read_params(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def param_float(params: dict[str, str], key: str) -> float:
    return float(params[key])


def voigt_matrix(params: dict[str, str], prefix: str) -> np.ndarray:
    matrix = np.zeros((6, 6), dtype=np.float64)
    for a in range(6):
        for b in range(a, 6):
            key = f"{prefix}_{a + 1}{b + 1}"
            matrix[a, b] = matrix[b, a] = param_float(params, key)
    return matrix


def voigt_to_tensor(matrix: np.ndarray) -> np.ndarray:
    tensor = np.zeros((3, 3, 3, 3), dtype=np.float64)
    for a, (i, j) in enumerate(VOIGT_PAIRS):
        for b, (k, l) in enumerate(VOIGT_PAIRS):
            value = matrix[a, b]
            tensor[i, j, k, l] = value
            tensor[j, i, k, l] = value
            tensor[i, j, l, k] = value
            tensor[j, i, l, k] = value
    return tensor


def wave_numbers(shape: tuple[int, int, int], spacing: tuple[float, float, float]):
    nx, ny, nz = shape
    dx, dy, dz = spacing
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kz = 2.0 * np.pi * np.fft.rfftfreq(nz, d=dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    mask = (
        (np.abs(KX) < np.pi * (2.0 / 3.0) / dx)
        & (np.abs(KY) < np.pi * (2.0 / 3.0) / dy)
        & (np.abs(KZ) < np.pi * (2.0 / 3.0) / dz)
    )
    return (KX, KY, KZ), mask


def dealias(field_k: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, field_k, 0.0)


def tensor_from_voigt_field(field: np.ndarray) -> np.ndarray:
    """Convert tensor-shear Voigt components (...,6) to (...,3,3)."""
    out = np.zeros(field.shape[:-1] + (3, 3), dtype=field.dtype)
    for a, (i, j) in enumerate(VOIGT_PAIRS):
        out[..., i, j] = field[..., a]
        out[..., j, i] = field[..., a]
    return out


def engineering_voigt(tensor: np.ndarray) -> np.ndarray:
    out = np.empty(tensor.shape[:-2] + (6,), dtype=tensor.dtype)
    out[..., 0] = tensor[..., 0, 0]
    out[..., 1] = tensor[..., 1, 1]
    out[..., 2] = tensor[..., 2, 2]
    out[..., 3] = 2.0 * tensor[..., 1, 2]
    out[..., 4] = 2.0 * tensor[..., 0, 2]
    out[..., 5] = 2.0 * tensor[..., 0, 1]
    return out


def stress_voigt_to_tensor(stress: np.ndarray) -> np.ndarray:
    return tensor_from_voigt_field(stress)


def strain_from_u_k(
    u_k: np.ndarray,
    wave: tuple[np.ndarray, np.ndarray, np.ndarray],
    mask: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    k = np.stack(wave, axis=-1)
    eps_k = 0.5j * (
        k[..., :, None] * u_k[..., None, :]
        + k[..., None, :] * u_k[..., :, None]
    )
    eps_k = np.where(mask[..., None, None], eps_k, 0.0)
    eps = np.empty(shape + (3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            eps[..., i, j] = np.fft.irfftn(
                eps_k[..., i, j], s=shape, axes=(0, 1, 2)
            ).real
    return eps


def green_inverse(C0_tensor: np.ndarray, wave, mask) -> np.ndarray:
    k = np.stack(wave, axis=-1)
    acoustic = np.einsum("...j,ijkl,...k->...il", k, C0_tensor, k)
    inverse = np.zeros_like(acoustic)
    active_indices = np.argwhere(mask & (np.sum(k * k, axis=-1) > 0.0))
    for index in active_indices:
        idx = tuple(int(v) for v in index)
        inverse[idx] = np.linalg.inv(acoustic[idx])
    return inverse


def force_from_tensor_k(tensor_k: np.ndarray, wave) -> np.ndarray:
    k = np.stack(wave, axis=-1)
    return np.einsum("...j,...ij->...i", k, tensor_k)


def rfftn_tensor(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    shape_k = field.shape[:2] + (field.shape[2] // 2 + 1,) + field.shape[3:]
    result = np.empty(shape_k, dtype=np.complex128)
    for suffix in np.ndindex(field.shape[3:]):
        source = field[(slice(None), slice(None), slice(None)) + suffix]
        result[(slice(None), slice(None), slice(None)) + suffix] = dealias(
            np.fft.rfftn(source), mask
        )
    return result


def solve(
    phi: np.ndarray,
    x_b: np.ndarray,
    params: dict[str, str],
    green_updates: int,
) -> dict[str, np.ndarray | float]:
    shape = tuple(int(v) for v in phi.shape)
    spacing = tuple(param_float(params, name) for name in ("dx", "dy", "dz"))
    wave, mask = wave_numbers(shape, spacing)
    kvec = np.stack(wave, axis=-1)
    kmax = float(np.max(np.linalg.norm(kvec[mask], axis=-1)))

    C0 = voigt_matrix(params, "S")
    Cp = voigt_matrix(params, "S_p")
    C0_tensor = voigt_to_tensor(C0)
    gamma = green_inverse(C0_tensor, wave, mask)
    h = h_phi(phi)
    hp = hp_phi(phi)
    x = np.clip(x_b, 0.0, 1.0)
    eps_iso = param_float(params, "eps_iso_over_vB")
    eps00 = np.array([
        param_float(params, "eps_xx00"),
        param_float(params, "eps_yy00"),
        param_float(params, "eps_zz00"),
        param_float(params, "eps_yz00"),
        param_float(params, "eps_xz00"),
        param_float(params, "eps_xy00"),
    ])
    eps0_voigt_tensor_shear = np.empty(shape + (6,), dtype=np.float64)
    eps_c = x * eps_iso
    eps0_voigt_tensor_shear[..., :3] = (
        (1.0 - h)[..., None] * eps_c[..., None]
        + h[..., None] * eps00[:3]
    )
    eps0_voigt_tensor_shear[..., 3:] = h[..., None] * eps00[3:]
    eps0_tensor = tensor_from_voigt_field(eps0_voigt_tensor_shear)
    eps0_engineering = engineering_voigt(eps0_tensor)

    eigenstress0_voigt = np.einsum("ab,...b->...a", C0, eps0_engineering)
    eigenstress0 = stress_voigt_to_tensor(eigenstress0_voigt)
    eigenstress0_k = rfftn_tensor(eigenstress0, mask)
    source0 = force_from_tensor_k(eigenstress0_k, wave)
    u_k = -1j * np.einsum("...ij,...j->...i", gamma, source0)
    u_k = np.where(mask[..., None], u_k, 0.0)

    Ceff = C0 + h[..., None, None] * Cp
    E0_voigt = np.array([
        param_float(params, "E0_xx"),
        param_float(params, "E0_yy"),
        param_float(params, "E0_zz"),
        2.0 * param_float(params, "E0_yz"),
        2.0 * param_float(params, "E0_xz"),
        2.0 * param_float(params, "E0_xy"),
    ])
    for _ in range(green_updates):
        strain = strain_from_u_k(u_k, wave, mask, shape)
        strain_eng = engineering_voigt(strain)
        hij_voigt = (
            -np.einsum("...ab,...b->...a", Ceff, strain_eng + E0_voigt)
            + np.einsum("ab,...b->...a", C0, strain_eng)
            + np.einsum("...ab,...b->...a", Ceff, eps0_engineering)
        )
        hij = stress_voigt_to_tensor(hij_voigt)
        hij_k = rfftn_tensor(hij, mask)
        source = force_from_tensor_k(hij_k, wave)
        u_k = -1j * np.einsum("...ij,...j->...i", gamma, source)
        u_k = np.where(mask[..., None], u_k, 0.0)

    strain = strain_from_u_k(u_k, wave, mask, shape)
    for a, (i, j) in enumerate(VOIGT_PAIRS):
        strain[..., i, j] += E0_voigt[a] / (2.0 if i != j else 1.0)
        if i != j:
            strain[..., j, i] = strain[..., i, j]
    strain_eng = engineering_voigt(strain)
    elastic_eng = strain_eng - eps0_engineering
    stress_voigt = np.einsum("...ab,...b->...a", Ceff, elastic_eng)
    stress = stress_voigt_to_tensor(stress_voigt)
    stress_k = rfftn_tensor(stress, mask)
    force_k = 1j * force_from_tensor_k(stress_k, wave)
    force = np.empty(shape + (3,), dtype=np.float64)
    for i in range(3):
        force[..., i] = np.fft.irfftn(
            force_k[..., i], s=shape, axes=(0, 1, 2)
        ).real

    stress_mag = np.sqrt(np.einsum("...ij,...ij->...", stress, stress))
    force_mag = np.linalg.norm(force, axis=-1)
    stress_l2 = float(np.sqrt(np.mean(stress_mag**2)))
    stress_linf = float(np.max(stress_mag))
    force_l2 = float(np.sqrt(np.mean(force_mag**2)))
    force_linf = float(np.max(force_mag))
    eta_l2 = force_l2 / max(kmax * stress_l2, np.finfo(float).tiny)
    eta_linf = force_linf / max(kmax * stress_linf, np.finfo(float).tiny)
    elastic_energy = 0.5 * np.einsum("...a,...a->...", stress_voigt, elastic_eng)

    deps0 = np.empty_like(eps0_engineering)
    deps0[..., :3] = hp[..., None] * (eps00[:3] - eps_iso)
    deps0[..., 3:] = hp[..., None] * (2.0 * eps00[3:])
    dgel_eigen = -np.einsum("...a,...a->...", stress_voigt, deps0)
    Q = np.einsum("...a,ab,...b->...", elastic_eng, Cp, elastic_eng)
    dgel = dgel_eigen + 0.5 * hp * Q
    u = np.empty(shape + (3,), dtype=np.float64)
    for i in range(3):
        u[..., i] = np.fft.irfftn(
            u_k[..., i], s=shape, axes=(0, 1, 2)
        ).real
    return {
        "u_k": u_k,
        "u": u,
        "strain": strain,
        "stress": stress,
        "force_k": force_k,
        "force": force,
        "dgel": dgel,
        "elastic_energy": elastic_energy,
        "stress_l2": stress_l2,
        "stress_linf": stress_linf,
        "force_l2": force_l2,
        "force_linf": force_linf,
        "eta_l2": eta_l2,
        "eta_linf": eta_linf,
        "kmax": kmax,
        "mask": mask,
    }


def load_fp32_dump(directory: Path, shape: tuple[int, int, int]):
    nz_c = shape[2] // 2 + 1
    kshape = (shape[0], shape[1], nz_c)

    def f32(name: str) -> np.ndarray:
        return np.fromfile(directory / name, dtype=np.float32).reshape(shape)

    def c64(name: str) -> np.ndarray:
        raw = np.fromfile(directory / name, dtype=np.float32).reshape(kshape + (2,))
        return raw[..., 0].astype(np.float64) + 1j * raw[..., 1].astype(np.float64)

    u_k = np.stack([c64(f"mechanics_u{axis}_k_f32.raw") for axis in "xyz"], axis=-1)
    strain_components = [f32(f"mechanics_e{name}_f32.raw")
                         for name in ("xx", "yy", "zz", "xy", "xz", "yz")]
    stress_components = [f32(f"mechanics_s{name}_f32.raw")
                         for name in ("xx", "yy", "zz", "xy", "xz", "yz")]
    strain = np.zeros(shape + (3, 3), dtype=np.float64)
    stress = np.zeros_like(strain)
    file_pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    for values, target in ((strain_components, strain), (stress_components, stress)):
        for component, (i, j) in zip(values, file_pairs):
            target[..., i, j] = component
            target[..., j, i] = component
    return {
        "u_k": u_k,
        "strain": strain,
        "stress": stress,
        "dgel": np.fromfile(
            directory / "mechanics_dgel_dphi_f64.raw", dtype=np.float64
        ).reshape(shape),
        "gel": np.fromfile(
            directory / "mechanics_gel_f64.raw", dtype=np.float64
        ).reshape(shape),
    }


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    numerator = float(np.linalg.norm((candidate - reference).ravel()))
    denominator = max(float(np.linalg.norm(reference.ravel())),
                      np.finfo(float).tiny)
    return numerator / denominator


def frozen_fp32_stress_residual(
    stress: np.ndarray,
    spacing: tuple[float, float, float],
) -> dict[str, float | np.ndarray]:
    shape = stress.shape[:3]
    wave, mask = wave_numbers(shape, spacing)
    kvec = np.stack(wave, axis=-1)
    kmax = float(np.max(np.linalg.norm(kvec[mask], axis=-1)))
    stress_k = rfftn_tensor(stress.astype(np.float64), mask)
    force_k = 1j * force_from_tensor_k(stress_k, wave)
    force = np.stack([
        np.fft.irfftn(force_k[..., i], s=shape, axes=(0, 1, 2)).real
        for i in range(3)
    ], axis=-1)
    stress_mag = np.sqrt(np.einsum("...ij,...ij->...", stress, stress))
    force_mag = np.linalg.norm(force, axis=-1)
    stress_l2 = float(np.sqrt(np.mean(stress_mag**2)))
    stress_linf = float(np.max(stress_mag))
    force_l2 = float(np.sqrt(np.mean(force_mag**2)))
    force_linf = float(np.max(force_mag))
    return {
        "force_k": force_k,
        "force_l2": force_l2,
        "force_linf": force_linf,
        "stress_l2": stress_l2,
        "stress_linf": stress_linf,
        "eta_l2": force_l2 / max(kmax * stress_l2, np.finfo(float).tiny),
        "eta_linf": force_linf / max(kmax * stress_linf, np.finfo(float).tiny),
        "kmax": kmax,
    }


def oracle_contract_hash() -> str:
    contract = {
        "schema": "mechanics_fp64_double_oracle_contract_v1",
        "gradient": "continuous_periodic_ik_on_r2c_grid",
        "dealias": "strict_componentwise_abs_k_lt_2pi_over_3dx",
        "nyquist": "removed_by_strict_two_thirds_mask",
        "strain": "symmetric_gradient_tensor_shear",
        "stiffness": "C0_plus_h_phi_Cp_engineering_voigt",
        "eigenstrain": "one_minus_h_xB_eps_iso_plus_h_eps00",
        "mean_strain": "fixed_E0_added_after_periodic_strain",
        "iteration": "same_unrelaxed_green_fixed_point",
        "residual": "real_ifft_of_i_kj_sigmaij",
    }
    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dump = args.dump_dir.resolve()
    params_path = args.params.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((dump / "mechanics_fp32_diagnostics.json").read_text())
    shape = (metadata["Nx"], metadata["Ny"], metadata["Nz"])
    phi = np.fromfile(dump / "mechanics_phi_f64.raw", dtype=np.float64).reshape(shape)
    x_b = np.fromfile(dump / "mechanics_xB_f64.raw", dtype=np.float64).reshape(shape)
    params = read_params(params_path)
    production = load_fp32_dump(dump, shape)
    oracle = solve(phi, x_b, params, int(metadata["green_iterations"]))
    production_u = np.stack([
        np.fft.irfftn(
            production["u_k"][..., i], s=shape, axes=(0, 1, 2)
        ).real
        for i in range(3)
    ], axis=-1)
    frozen = frozen_fp32_stress_residual(
        production["stress"],
        (float(params["dx"]), float(params["dy"]), float(params["dz"])),
    )
    result = {
        "schema": "mechanics_fp64_oracle_comparison_v1",
        "grid": list(shape),
        "green_iterations": int(metadata["green_iterations"]),
        "displacement_relative_l2": relative_l2(production_u, oracle["u"]),
        "strain_relative_l2": relative_l2(production["strain"], oracle["strain"]),
        "stress_relative_l2": relative_l2(production["stress"], oracle["stress"]),
        "elastic_energy_relative_error": abs(
            float(np.mean(production["gel"]))
            - float(np.mean(oracle["elastic_energy"]))
        ) / max(abs(float(np.mean(oracle["elastic_energy"]))), np.finfo(float).tiny),
        "elastic_phase_driving_relative_l2": relative_l2(
            production["dgel"], oracle["dgel"]
        ),
        "production_eta_linf_runtime": metadata["eta_Linf"],
        "production_eta_linf_host_double": frozen["eta_linf"],
        "production_eta_l2_runtime": metadata["eta_L2"],
        "production_eta_l2_host_double": frozen["eta_l2"],
        "oracle_eta_linf": oracle["eta_linf"],
        "oracle_eta_l2": oracle["eta_l2"],
        "oracle_contract_hash": oracle_contract_hash(),
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
