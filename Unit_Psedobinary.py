#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 17:19:07 2026

@author: heng
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
PF 参数转换子程序（集成版：含 6x6 弹性矩阵 + 完整经典成核分析 + 详细打印）
====================================================================================

本版本更新说明（本次最小侵入改动）：
- [新增] 输入一个弹性能密度 shift：gel_shift_Jm3 (J/m^3)
- [新增] 输出对应的无量纲弹性能密度 shift：gel_shift_hat = gel_shift_Jm3 / w
- 其余逻辑不变
"""

import argparse
import copy
import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple

# =========================================================
# 数值工具
# =========================================================

def _linspace(start: float, stop: float, num: int) -> List[float]:
    if num < 2:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def _cumulative_trapezoid(values: Iterable[float], dx: float) -> List[float]:
    vals = list(values)
    out = [0.0] * len(vals)
    for i in range(1, len(vals)):
        out[i] = out[i - 1] + 0.5 * (vals[i] + vals[i - 1]) * dx
    return out


def _trapezoid(values: Iterable[float], dx: float) -> float:
    vals = list(values)
    total = 0.0
    for i in range(1, len(vals)):
        total += 0.5 * (vals[i] + vals[i - 1]) * dx
    return total


def _validate_composition(x: float) -> None:
    if not (0.0 < x < 1.0):
        raise ValueError("x (x_Ag2Te) 必须在 (0,1) 之间")


def _validate_vector_length(name: str, values: List[float], n: int) -> None:
    if values is None or len(values) != n:
        raise ValueError(f"{name} 必须是长度为 {n} 的数组")


def _validate_matrix_shape(name: str, matrix: List[List[float]], rows: int, cols: int) -> None:
    if matrix is None or len(matrix) != rows:
        raise ValueError(f"{name} 必须是 {rows}x{cols} 矩阵")
    for row in matrix:
        if len(row) != cols:
            raise ValueError(f"{name} 必须是 {rows}x{cols} 矩阵")


def _transpose3(matrix: List[List[float]]) -> List[List[float]]:
    return [[matrix[j][i] for j in range(3)] for i in range(3)]


def _matmul3(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    out = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(3))
    return out


def _symmetrize3(matrix: List[List[float]]) -> List[List[float]]:
    return [
        [0.5 * (matrix[i][j] + matrix[j][i]) for j in range(3)]
        for i in range(3)
    ]


def _drop_none(obj):
    if isinstance(obj, dict):
        return {key: _drop_none(value) for key, value in obj.items() if value is not None}
    if isinstance(obj, list):
        return [_drop_none(value) for value in obj]
    return obj


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-30:
        return float("inf")
    return numerator / denominator


def _validate_rotation_matrix(name: str, rotation: List[List[float]], tol: float = 1e-8) -> None:
    _validate_matrix_shape(name, rotation, 3, 3)
    rt_r = _matmul3(_transpose3(rotation), rotation)
    for i in range(3):
        for j in range(3):
            target = 1.0 if i == j else 0.0
            if abs(rt_r[i][j] - target) > tol:
                raise ValueError(f"{name} 必须是正交旋转矩阵")


def resolve_eigenstrain_tensor(inputs: "PhysicalInputs") -> List[List[float]]:
    if inputs.eigenstrain_tensor is not None:
        _validate_matrix_shape("eigenstrain_tensor", inputs.eigenstrain_tensor, 3, 3)
        tensor = [[float(value) for value in row] for row in inputs.eigenstrain_tensor]
        for i in range(3):
            for j in range(3):
                if abs(tensor[i][j] - tensor[j][i]) > 1e-10:
                    raise ValueError("eigenstrain_tensor 必须是对称张量")
        return _symmetrize3(tensor)

    if inputs.eigenstrain_principal is None:
        raise ValueError("必须提供 eigenstrain_tensor 或 eigenstrain_principal")

    _validate_vector_length("eigenstrain_principal", inputs.eigenstrain_principal, 3)
    principal = [float(value) for value in inputs.eigenstrain_principal]

    if inputs.eigenstrain_rotation_matrix is None:
        rotation = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    else:
        rotation = [[float(value) for value in row] for row in inputs.eigenstrain_rotation_matrix]
        _validate_rotation_matrix("eigenstrain_rotation_matrix", rotation)

    diag = [
        [principal[0], 0.0, 0.0],
        [0.0, principal[1], 0.0],
        [0.0, 0.0, principal[2]],
    ]
    return _symmetrize3(_matmul3(_matmul3(rotation, diag), _transpose3(rotation)))


def tensor3_to_voigt_strain(tensor: List[List[float]]) -> Dict[str, float]:
    _validate_matrix_shape("eigenstrain_tensor", tensor, 3, 3)
    return {
        "eps_xx00": tensor[0][0],
        "eps_yy00": tensor[1][1],
        "eps_zz00": tensor[2][2],
        "eps_yz00": tensor[1][2],
        "eps_xz00": tensor[0][2],
        "eps_xy00": tensor[0][1],
    }


def resolve_pf_dx(inputs: "PhysicalInputs") -> float:
    if inputs.pf_dx is not None:
        return inputs.pf_dx
    return inputs.dx


def build_quality_checks(inputs: "PhysicalInputs") -> Dict[str, object]:
    pf_dx = resolve_pf_dx(inputs)
    lambda_over_pf_dx = inputs.lambda_sm / pf_dx if pf_dx > 0.0 else float("inf")
    warnings: List[str] = []

    if lambda_over_pf_dx <= 4.0:
        warnings.append(
            f"界面分辨率不足: lambda_sm/pf_dx = {lambda_over_pf_dx:.3f} <= 4，建议减小 pf_dx 或增大 lambda_sm。"
        )

    if inputs.phys_dx_ref is not None and inputs.phys_dx_ref > 0.0:
        refinement_ratio = _safe_ratio(inputs.phys_dx_ref, pf_dx)
        warnings.append(
            f"双dx说明: phys_dx_ref = {inputs.phys_dx_ref:.3e} m, pf_dx = {pf_dx:.3e} m, refinement = {refinement_ratio:.3f}x"
        )

    return {
        "pf_dx_m": pf_dx,
        "phys_dx_ref_m": inputs.phys_dx_ref,
        "lambda_over_pf_dx": lambda_over_pf_dx,
        "refinement_ratio_phys_over_pf": (
            _safe_ratio(inputs.phys_dx_ref, pf_dx) if inputs.phys_dx_ref is not None else None
        ),
        "interface_resolution_ok": (lambda_over_pf_dx > 4.0),
        "warnings": warnings,
    }


# -------------------------------
# 数据类定义
# -------------------------------

@dataclass
class PhysicalInputs:
    """
    物理输入参数
    """
    gamma: float
    lambda_sm: float
    dx: float
    dt: float
    v_A: float
    v_B: float
    temperature_C: float
    Vm_compound: float
    Vm_alpha_0: float
    pf_dx: float = None
    phys_dx_ref: float = None
    D_ratio: float = 0.1
    vf_init: float = 0.05
    vf_target: float = 0.20
    L_ref_factor: float = 1.0

    # === [新增] 弹性能密度 shift 输入 (J/m^3) ===
    gel_shift_Jm3: float = 0.0

    # 各向同性化学膨胀应变 epsilon_iso（无量纲）
    eps_iso: float = 0.0

    # 本征应变输入（二选一）
    # 1) 直接给出模拟坐标系下的 3x3 对称小应变张量
    eigenstrain_tensor: List[List[float]] = None
    # 2) 或给出主应变 + 从主轴到模拟坐标系的旋转矩阵
    eigenstrain_principal: List[float] = None
    eigenstrain_rotation_matrix: List[List[float]] = None

    # 完整 6x6 弹性刚度矩阵输入 (单位: GPa)
    C_tensor_PbTe_GPa: List[List[float]] = None
    C_tensor_Ag2Te_GPa: List[List[float]] = None


@dataclass
class PFParamSet:
    """
    对应 C 代码 PFParams 的关键字段（无量纲）
    """
    W: float
    kappa_phi: float
    L_phi: float
    D_alpha: float
    D_compound: float
    v_A: float
    v_B: float
    Vm_compound: float
    Vm_alpha_0: float
    dVm_alpha_dxB: float
    ic_xB_eq_matrix: float
    ic_vf_init_phi: float
    ic_vf_target_phi: float
    ic_phi_iface_w: float
    mu_reference: float

    # ===== 无量纲弹性矩阵 (6x6) =====
    C_tensor_matrix_hat: List[List[float]]
    C_tensor_precip_hat: List[List[float]]

    # === [新增] 无量纲弹性能密度 shift ===
    gel_shift_hat: float

    # 额外信息
    xAg2Te_eq: float
    temperature_C: float

    def to_pfparams_dict(self) -> Dict[str, object]:
        return self.__dict__


RUNTIME_GAP_FIELDS = [
    "Y_clip / xB_eps / xB_s_floor 等纯数值稳定参数仍沿用 C 侧默认值",
    "E0_xx..E0_xy 外加载荷暂未从真实物理输入自动推导",
    "minimize/dynamics 模式控制参数仍建议通过现有 CLI 传入",
]

EXAMPLE_INPUT_UNITS = {
    "gamma": "J/m^2",
    "lambda_sm": "m",
    "dx": "m (legacy alias of pf_dx; kept for backward compatibility)",
    "pf_dx": "m (actual physical spacing of one PF grid cell; used in nondimensionalization and main_cuda dx/dy/dz)",
    "phys_dx_ref": "m (optional coarse physical reference spacing; bookkeeping only, not used in nondimensionalization)",
    "dt": "dimensionless code time step",
    "v_A": "dimensionless stoichiometric coefficient",
    "v_B": "dimensionless stoichiometric coefficient",
    "temperature_C": "degC",
    "Vm_compound": "m^3/mol",
    "Vm_alpha_0": "m^3/mol",
    "D_ratio": "dimensionless (D_compound / D_alpha)",
    "vf_init": "dimensionless volume fraction",
    "vf_target": "dimensionless volume fraction",
    "L_ref_factor": "dimensionless (L_ref / lambda_sm)",
    "gel_shift_Jm3": "J/m^3",
    "eps_iso": "dimensionless isotropic chemical strain",
    "eigenstrain_tensor": "dimensionless strain tensor in simulation basis",
    "eigenstrain_principal": "dimensionless principal strain",
    "eigenstrain_rotation_matrix": "dimensionless rotation matrix from principal basis to simulation basis; identity means no rotation",
    "C_tensor_PbTe_GPa": "GPa",
    "C_tensor_Ag2Te_GPa": "GPa",
}


# ===== 用户配置区 =====

USER_PHYSICAL_INPUTS = PhysicalInputs(
    gamma=0.05,                      # J/m^2
    lambda_sm=6.0e-10,               # m
    dx=1.0e-10,                      # m (legacy alias; keep equal to pf_dx)
    pf_dx=1.0e-10,                   # m (actual PF grid spacing)
    phys_dx_ref=1.0e-9,              # m (optional coarse physical reference spacing)
    dt=1.0e-2,                      # code time step
    v_A=0.0,                        # 反应式 PbTe -> Ag2Te (A=0, B=1)
    v_B=1.0,                        # 纯 Ag2Te 析出相的 x_B = 1.0
    temperature_C=380.0,            # °C
    Vm_alpha_0=4.1009e-5,            # m^3/mol (PbTe)
    Vm_compound=4.1009e-5,           # m^3/mol (Ag2Te)
    D_ratio=0.01,                   # D_precip / D_matrix
    vf_init=0.00,                   # 初始相分数
    vf_target=0.04,                 # 目标相分数 (用于估算初始过饱和度)
    L_ref_factor=5.0,              # L_ref = 10 * lambda_sm

    # === [你填这里] PF 输出的弹性能密度 shift (J/m^3) ===
    gel_shift_Jm3=0.0,

    # 各向同性化学膨胀应变 epsilon_iso（默认与 main_cuda 当前默认值对应）
    eps_iso=0.00233,

    # --- 本征应变（默认与 main_cuda 当前默认值一致）---
    eigenstrain_principal=[0.046, -0.022, -0.017],
    eigenstrain_rotation_matrix=[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ],

    # --- 1. PbTe OR-frame stiffness tensor (GPa) ---
    C_tensor_PbTe_GPa=[
        [108.0,  6.0,  6.0,  0.0,  0.0,  0.0],
        [  6.0, 71.0, 43.0, -0.0,  0.0,  0.0],
        [  6.0, 43.0, 71.0,  0.0,  0.0,  0.0],
        [  0.0, -0.0, 0.0, 51.0,  0.0,  0.0],
        [  0.0,  0.0, 0.0,  0.0, 14.0,  0.0],
        [  0.0,  0.0, 0.0,  0.0,  0.0, 14.0]
    ],

    # --- 2. Ag2Te OR-frame stiffness tensor (GPa) ---
    C_tensor_Ag2Te_GPa=[
        [64.53932315, 42.71742108, 40.48003361,  2.17554086, -0.06871424,  1.43079278],
        [42.71742108, 90.00000000, 46.28257892,  2.50650732,  4.13187759,  1.64846021],
        [40.48003361, 46.28257892, 68.50060964,  2.83747377,  4.65968934,  1.86612765],
        [ 2.17554086,  2.50650732,  2.83747377, 11.50967838,  0.76715418, -2.29548755],
        [-0.06871424,  4.13187759,  4.65968934,  0.76715418, 17.48003361,  0.50453599],
        [ 1.43079278,  1.64846021,  1.86612765, -2.29548755,  0.50453599, 13.49032162]
    ]
)

OUTPUT_JSON_PATH = None
PRINT_SUMMARY = True


# -------------------------------
# 热力学 / 化学势 (xB = x_Ag2Te)
# -------------------------------

R_GAS = 8.31446261815324

def GHSER_Pb(T: float) -> float:
    if T < 600.61:
        return -7650.085 + 101.700244*T - 24.5242231*T*math.log(T) - 0.00365895*T**2 - 2.4395e-7*T**3
    else:
        return -10531.095 + 154.243182*T - 32.4913959*T*math.log(T) + 0.00154613*T**2 + 8.05448e25*T**(-9)

def GHSER_Ag(T: float) -> float:
    if T < 1234.93:
        return -7209.512 + 118.202013*T - 23.8463314*T*math.log(T) - 0.001790585*T**2 - 3.98587e-7*T**3 - 12011*T**(-1)
    else:
        return -15095.252 + 190.266404*T - 33.472*T*math.log(T) + 1.411773e29*T**(-9)

def GHSER_Te(T: float) -> float:
    if T < 722.66:
        return -10544.679 + 183.372894*T - 35.6687*T*math.log(T) + 0.01583435*T**2 - 5.240417e-6*T**3 + 155015*T**(-1)
    else:
        return 9160.595 - 129.265373*T + 13.004*T*math.log(T) - 0.0362361*T**2 + 5.006367e-6*T**3 - 1.28681e30*T**(-9)

def G_PbTe_Solid(T: float) -> float:
    base = -76063.2138 + 9.67716633 * T
    return base + GHSER_Pb(T) + GHSER_Te(T)

def G_Ag2Te_Solid(T: float) -> float:
    base_per_atom = -10128.93 - 12.645115 * T
    G_atom = base_per_atom + (2.0/3.0)*GHSER_Ag(T) + (1.0/3.0)*GHSER_Te(T)
    return 3.0 * G_atom

def L0_PseudoBinary(T: float) -> float:
    return 41212.9 - 18.05 * T

def mu_Ag2Te(T: float, x: float) -> float:
    _validate_composition(x)
    return (
        G_Ag2Te_Solid(T)
        + R_GAS * T * math.log(x)
        + L0_PseudoBinary(T) * (1.0 - x) ** 2
    )

def mu_PbTe(T: float, x: float) -> float:
    _validate_composition(x)
    return (
        G_PbTe_Solid(T)
        + R_GAS * T * math.log(1.0 - x)
        + L0_PseudoBinary(T) * x ** 2
    )

def xAg2Te_eq_from_T(T_K: float) -> float:
    L_val = L0_PseudoBinary(T_K)
    RT = R_GAS * T_K
    x = math.exp(-L_val / RT)
    x = max(1e-9, min(0.5, x))

    for _ in range(50):
        one_minus_x = 1.0 - x
        f = RT * math.log(x) + L_val * one_minus_x * one_minus_x
        df = RT / x - 2.0 * L_val * one_minus_x
        if abs(df) < 1e-30:
            break
        step = f / df
        x_new = x - step
        x_new = max(1e-9, min(0.99, x_new))
        if abs(x_new - x) < 1e-14:
            x = x_new
            break
        x = x_new

    return max(1e-9, min(0.99, x))

def linearize_mu(func, T_K: float, x_eq: float, dx: float = 1e-6) -> Tuple[float, float]:
    dx = min(dx, 0.25 * min(x_eq, 1.0 - x_eq))
    x_plus = min(1.0 - 1e-9, x_eq + dx)
    x_minus = max(1e-9, x_eq - dx)
    f0 = func(T_K, x_eq)
    slope = (func(T_K, x_plus) - func(T_K, x_minus)) / (x_plus - x_minus)
    intercept = f0 - slope * x_eq
    return intercept, slope

def D_Ag_in_PbTe_m2_per_s(T_K: float) -> float:
    D0_cm2_s = 4.251e-11
    Q_J_mol = 3.403e+04
    D_cm2_s = D0_cm2_s * math.exp(-Q_J_mol / (R_GAS * T_K))
    return D_cm2_s * 1.0e-4


# -------------------------------
# Diffusion-controlled 参数
# -------------------------------

def compute_interface_params(gamma: float, lambda_sm: float) -> Tuple[float, float, float]:
    if lambda_sm <= 0.0:
        raise ValueError("lambda_sm 必须为正")
    w = 12.0 * gamma / lambda_sm
    kappa = 1.5 * gamma * lambda_sm
    W_code = 1.0
    return w, kappa, W_code

def compute_zeta(muA_x: float, muB_x: float, vA: float, vB: float, x_eq: float) -> float:
    pref = vA * muA_x + vB * muB_x
    bracket = vB / (vA + vB) - x_eq
    return pref * bracket

def xi_eq_profile(x: float, w: float, kappa: float) -> float:
    s = math.sqrt(w / (2.0 * kappa))
    return 0.5 * (1.0 - math.tanh(s * x))

def dxi_dx_profile(x: float, w: float, kappa: float) -> float:
    s = math.sqrt(w / (2.0 * kappa))
    sech2 = 1.0 / math.cosh(s * x) ** 2
    return -0.5 * s * sech2

def h_switch(xi: float) -> float:
    return 6.0 * xi ** 5 - 15.0 * xi ** 4 + 10.0 * xi ** 3

def h_switch_derivative(xi: float) -> float:
    return 30.0 * xi ** 4 - 60.0 * xi ** 3 + 30.0 * xi ** 2

def compute_zeta0(lambda_sm: float, w: float, kappa: float, D_alpha: float, D_comp: float, n_quad: int = 2001) -> float:
    x_min = -0.5 * lambda_sm
    x_max =  0.5 * lambda_sm
    xs = _linspace(x_min, x_max, n_quad)
    dx = xs[1] - xs[0]
    xi_vals   = [xi_eq_profile(x, w, kappa) for x in xs]
    dxi_vals  = [dxi_dx_profile(x, w, kappa) for x in xs]
    h_vals    = [h_switch(xi) for xi in xi_vals]
    hprime    = [h_switch_derivative(xi) for xi in xi_vals]
    Dmix_vals = [(1.0 - h) * D_alpha + h * D_comp for h in h_vals]
    inner = [(1.0 - h_vals[i]) / max(Dmix_vals[i], 1e-32) for i in range(n_quad)]
    I_vals = _cumulative_trapezoid(inner, dx)
    outer = [hprime[i] * dxi_vals[i] * I_vals[i] for i in range(n_quad)]
    outer_int = _trapezoid(outer, dx)
    return -2.0 * D_alpha / lambda_sm * outer_int


# -------------------------------
# 主转换器
# -------------------------------

class PFParamConverter:
    def __init__(self):
        pass

    def _normalize_tensor(self, C_tensor_GPa: List[List[float]], w_scale: float) -> List[List[float]]:
        tensor_hat = [[0.0] * 6 for _ in range(6)]
        GPa_to_Pa = 1.0e9
        for i in range(6):
            for j in range(6):
                val_pa = C_tensor_GPa[i][j] * GPa_to_Pa
                tensor_hat[i][j] = val_pa / w_scale
        return tensor_hat

    def convert(self, inputs: PhysicalInputs) -> PFParamSet:
        T_K = inputs.temperature_C + 273.15
        pf_dx = resolve_pf_dx(inputs)

        # 1) 界面参数
        w, kappa, W_code = compute_interface_params(inputs.gamma, inputs.lambda_sm)
        f0_elastic = w  # [J/m^3] 能量标度

        # === [新增] 计算无量纲 gel shift ===
        gel_shift_hat = inputs.gel_shift_Jm3 / f0_elastic if abs(f0_elastic) > 1e-30 else 0.0

        # 2) 溶解度
        xAg2Te_eq = xAg2Te_eq_from_T(T_K)
        xB_eq = xAg2Te_eq

        # 3) μ 线性化
        _, muA_a1 = linearize_mu(mu_PbTe, T_K, xAg2Te_eq)
        _, muB_a1 = linearize_mu(mu_Ag2Te, T_K, xAg2Te_eq)

        # 4) 扩散
        D_alpha_phys = D_Ag_in_PbTe_m2_per_s(T_K)
        D_comp_phys = D_alpha_phys * inputs.D_ratio

        # 5) ζ 和 ζ0
        zeta = compute_zeta(muA_a1, muB_a1, inputs.v_A, inputs.v_B, xB_eq)
        zeta0 = compute_zeta0(inputs.lambda_sm, w, kappa, D_alpha_phys, D_comp_phys)
        zprod = zeta * zeta0
        if abs(zprod) < 1e-30:
            raise ValueError("zeta0*zeta≈0")

        c_tot_phys = 1.0 / inputs.Vm_alpha_0

        L_xi = (4.0 * (inputs.v_A + inputs.v_B) * D_alpha_phys
                / (3.0 * c_tot_phys * (inputs.lambda_sm) ** 2 * abs(zprod)))

        L_ref = inputs.L_ref_factor * inputs.lambda_sm
        t0_diff = L_ref ** 2 / D_alpha_phys

        kappa_code = kappa / (w * pf_dx ** 2)
        D_alpha_code = D_alpha_phys * t0_diff / (pf_dx ** 2)
        D_comp_code = D_comp_phys * t0_diff / (pf_dx ** 2)
        L_phi_code = L_xi * w * t0_diff

        mu_reference = w / c_tot_phys
        ic_phi_iface_w = (inputs.lambda_sm / pf_dx) / 2.0

        # Vm
        Vm_comp_hat = inputs.Vm_compound / inputs.Vm_alpha_0
        dVm_alpha_dxB_phys = (inputs.Vm_compound - inputs.Vm_alpha_0) / inputs.v_B
        dVm_hat = dVm_alpha_dxB_phys / inputs.Vm_alpha_0

        # 弹性矩阵无量纲化
        if inputs.C_tensor_PbTe_GPa is None or inputs.C_tensor_Ag2Te_GPa is None:
            raise ValueError("必须提供完整的 6x6 弹性刚度矩阵")

        C_matrix_hat = self._normalize_tensor(inputs.C_tensor_PbTe_GPa, f0_elastic)
        C_precip_hat = self._normalize_tensor(inputs.C_tensor_Ag2Te_GPa, f0_elastic)

        return PFParamSet(
            W=W_code,
            kappa_phi=kappa_code,
            L_phi=L_phi_code,
            D_alpha=D_alpha_code,
            D_compound=D_comp_code,
            v_A=inputs.v_A,
            v_B=inputs.v_B,
            Vm_compound=Vm_comp_hat,
            Vm_alpha_0=1.0,
            dVm_alpha_dxB=dVm_hat,
            ic_xB_eq_matrix=xB_eq,
            ic_vf_init_phi=inputs.vf_init,
            ic_vf_target_phi=inputs.vf_target,
            ic_phi_iface_w=ic_phi_iface_w,
            mu_reference=mu_reference,
            C_tensor_matrix_hat=C_matrix_hat,
            C_tensor_precip_hat=C_precip_hat,
            gel_shift_hat=gel_shift_hat,   # === [新增] ===
            xAg2Te_eq=xAg2Te_eq,
            temperature_C=inputs.temperature_C,
        )


def _voigt_independent_components(matrix: List[List[float]]) -> Dict[str, float]:
    return {
        "11": matrix[0][0],
        "12": matrix[0][1],
        "13": matrix[0][2],
        "14": matrix[0][3],
        "15": matrix[0][4],
        "16": matrix[0][5],
        "22": matrix[1][1],
        "23": matrix[1][2],
        "24": matrix[1][3],
        "25": matrix[1][4],
        "26": matrix[1][5],
        "33": matrix[2][2],
        "34": matrix[2][3],
        "35": matrix[2][4],
        "36": matrix[2][5],
        "44": matrix[3][3],
        "45": matrix[3][4],
        "46": matrix[3][5],
        "55": matrix[4][4],
        "56": matrix[4][5],
        "66": matrix[5][5],
    }


def build_main_cuda_overrides(inputs: PhysicalInputs, pfset: PFParamSet) -> Dict[str, float]:
    pf_dx = resolve_pf_dx(inputs)
    eigenstrain_voigt = tensor3_to_voigt_strain(resolve_eigenstrain_tensor(inputs))
    if abs(inputs.v_B) < 1e-30:
        raise ValueError("v_B 不能为 0，否则无法计算 eps_iso_over_vB = eps_iso / v_B")
    eps_iso_over_vB = inputs.eps_iso / inputs.v_B
    T_K = inputs.temperature_C + 273.15
    D_alpha_phys = D_Ag_in_PbTe_m2_per_s(T_K)
    L_ref = inputs.L_ref_factor * inputs.lambda_sm
    t_real_unit = L_ref ** 2 / D_alpha_phys
    overrides: Dict[str, float] = {
        "dx": pf_dx / 1.0e-9,
        "dy": pf_dx / 1.0e-9,
        "dz": pf_dx / 1.0e-9,
        "dt": inputs.dt,
        "t_real_unit": t_real_unit,
        "temperature_C": inputs.temperature_C,
        "mu_reference_scale": pfset.mu_reference,
        "W": pfset.W,
        "kappa_phi": pfset.kappa_phi,
        "L_phi": pfset.L_phi,
        "D_alpha": pfset.D_alpha,
        "D_compound": pfset.D_compound,
        "v_A": pfset.v_A,
        "v_B": pfset.v_B,
        "Vm_compound": pfset.Vm_compound,
        "Vm_alpha_0": pfset.Vm_alpha_0,
        "dVm_alpha_dxB": pfset.dVm_alpha_dxB,
        "ic_vf_init_phi": pfset.ic_vf_init_phi,
        "ic_vf_target_phi": pfset.ic_vf_target_phi,
        "ic_phi_iface_w": pfset.ic_phi_iface_w,
        "gamma_Jm2": inputs.gamma,
        "lambda_sm_m": inputs.lambda_sm,
        "Vm_alpha_0_phys_m3mol": inputs.Vm_alpha_0,
        "elastic_shift_dimless": pfset.gel_shift_hat,
        "eps_iso_over_vB": eps_iso_over_vB,
        "eps_xx00": eigenstrain_voigt["eps_xx00"],
        "eps_yy00": eigenstrain_voigt["eps_yy00"],
        "eps_zz00": eigenstrain_voigt["eps_zz00"],
        "eps_yz00": eigenstrain_voigt["eps_yz00"],
        "eps_xz00": eigenstrain_voigt["eps_xz00"],
        "eps_xy00": eigenstrain_voigt["eps_xy00"],
    }

    matrix_components = _voigt_independent_components(pfset.C_tensor_matrix_hat)
    precip_components = _voigt_independent_components(pfset.C_tensor_precip_hat)
    for suffix, value in matrix_components.items():
        overrides[f"S_{suffix}"] = value
    for suffix, value in precip_components.items():
        overrides[f"S_p_{suffix}"] = value - matrix_components[suffix]
    return overrides


def load_physical_inputs(input_json_path: str | None) -> PhysicalInputs:
    base = copy.deepcopy(asdict(USER_PHYSICAL_INPUTS))
    if input_json_path:
        with open(input_json_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        loaded = {key: value for key, value in loaded.items() if not key.startswith("_")}
        unknown = sorted(set(loaded.keys()) - set(base.keys()))
        if unknown:
            raise KeyError(f"未知输入字段: {', '.join(unknown)}")
        base.update(loaded)
    return PhysicalInputs(**base)


def write_main_cuda_override_file(path: str, overrides: Dict[str, float]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated by Unit_Psedobinary.py\n")
        f.write("# Format: key=value, consumable by ./main_cuda --pf-param-file <file>\n")
        for key in sorted(overrides.keys()):
            value = overrides[key]
            if isinstance(value, int):
                f.write(f"{key}={value}\n")
            else:
                f.write(f"{key}={value:.16e}\n")


def write_example_input_json(path: str) -> None:
    payload = _drop_none(asdict(USER_PHYSICAL_INPUTS))
    payload["_notes"] = {
        "dx": "兼容旧接口保留；若同时提供 pf_dx，脚本一律用 pf_dx 做无量纲化与 main_cuda dx/dy/dz。",
        "pf_dx": "PF 真正单个网格对应的物理长度。若每 1 nm 用 10 个网格解析，这里应填 1e-10。",
        "phys_dx_ref": "可选的粗物理参考长度，仅用于记录与人工对照，不参与无量纲化。",
        "eigenstrain_rotation_matrix": "单位矩阵表示无旋转；主应变方向与模拟坐标系一致。",
        "eigenstrain_tensor_priority": "若同时提供 eigenstrain_tensor 和 eigenstrain_principal，脚本优先使用 eigenstrain_tensor。",
        "eps_iso": "脚本会自动换算为 main_cuda 需要的 eps_iso_over_vB = eps_iso / v_B。",
    }
    payload["_units"] = EXAMPLE_INPUT_UNITS
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# -------------------------------
# 生成 JSON & 入口
# -------------------------------

def generate_payload(inputs: PhysicalInputs) -> Dict[str, object]:
    converter = PFParamConverter()
    pfset = converter.convert(inputs)
    pf_dx = resolve_pf_dx(inputs)
    main_cuda_overrides = build_main_cuda_overrides(inputs, pfset)
    eigenstrain_tensor = resolve_eigenstrain_tensor(inputs)
    eigenstrain_voigt = tensor3_to_voigt_strain(eigenstrain_tensor)
    quality_checks = build_quality_checks(inputs)

    T_K = inputs.temperature_C + 273.15
    D_alpha_phys = D_Ag_in_PbTe_m2_per_s(T_K)
    D_comp_phys = D_alpha_phys * inputs.D_ratio
    w_phys = 12.0 * inputs.gamma / inputs.lambda_sm
    c_tot_phys = 1.0 / inputs.Vm_alpha_0
    L_ref = inputs.L_ref_factor * inputs.lambda_sm
    t0_diff_s = L_ref ** 2 / D_alpha_phys
    dVm_phys = (inputs.Vm_compound - inputs.Vm_alpha_0) / inputs.v_B

    # === 利用目标体积分数 + Δμ_r 估算超临界半径（两种版本） ===
    xB_eq = pfset.ic_xB_eq_matrix
    vB = pfset.v_B
    vf_t = inputs.vf_target
    xB_tot0 = vf_t * vB + (1.0 - vf_t) * xB_eq
    xB_farfield = xB_tot0
    supersat_ratio = xB_farfield / xB_eq

    Vm_alpha_x_inf = inputs.Vm_alpha_0 + (inputs.Vm_compound - inputs.Vm_alpha_0) * (xB_farfield / vB)
    c_inf = 1.0 / Vm_alpha_x_inf

    muA_eq = mu_PbTe(T_K, xB_eq)
    muB_eq = mu_Ag2Te(T_K, xB_eq)
    mu_comp_eq = (1.0 - vB) * muA_eq + vB * muB_eq  # vB=1 -> muB_eq

    muA_inf = mu_PbTe(T_K, xB_farfield)
    muB_inf = mu_Ag2Te(T_K, xB_farfield)
    delta_mu_r_inf = mu_comp_eq - ((1.0 - vB) * muA_inf + vB * muB_inf)

    # 1) 无体积差
    DeltaG_v_no_vol = c_inf * delta_mu_r_inf
    rc_no_vol = float("inf") if abs(DeltaG_v_no_vol) < 1e-20 else 2.0 * inputs.gamma / abs(DeltaG_v_no_vol)

    # 2) 有体积差
    DeltaVm_eff = inputs.Vm_compound - Vm_alpha_x_inf + dVm_phys * (xB_farfield - vB)
    mu_tot_inf = (1.0 - xB_farfield) * muA_inf + xB_farfield * muB_inf
    DeltaG_v_with_vol = DeltaG_v_no_vol - (c_inf ** 2) * mu_tot_inf * DeltaVm_eff
    rc_with_vol = float("inf") if abs(DeltaG_v_with_vol) < 1e-20 else 2.0 * inputs.gamma / abs(DeltaG_v_with_vol)

    # 3) 无量纲验证
    mu_ref = pfset.mu_reference
    hat_delta_mu_r = delta_mu_r_inf / mu_ref if abs(mu_ref) > 1e-30 else 0.0
    if abs(hat_delta_mu_r) < 1e-30:
        rc_hat_over_lambda = float("inf")
        rc_hat_over_dx = float("inf")
    else:
        rc_hat_over_lambda = 1.0 / (6.0 * abs(hat_delta_mu_r))
        rc_hat_over_dx = (inputs.lambda_sm / pf_dx) * rc_hat_over_lambda

    payload = {
        "pf_params": pfset.to_pfparams_dict(),
        "metadata": {
            "temperature_C": pfset.temperature_C,
            "xAg2Te_eq": pfset.xAg2Te_eq,
            "solubility_AB": {
                "L(T)": "41212.9 - 18.05 * T",
                "form": "Pseudo-Binary Regular Solution",
                "component": "x = x_Ag2Te",
                "method": "fsolve (Exact)",
            },
            "scales": {
                "w_physical_J_per_m3": w_phys,
                "lambda_sm_m": inputs.lambda_sm,
                "dx_m_legacy": inputs.dx,
                "pf_dx_m": pf_dx,
                "phys_dx_ref_m": inputs.phys_dx_ref,
                "dx_main_cuda": pf_dx / 1.0e-9,
                "dt_code": inputs.dt,
                "eps_iso": inputs.eps_iso,
                "eps_iso_over_vB": inputs.eps_iso / inputs.v_B if abs(inputs.v_B) > 1e-30 else float("inf"),
                "mu_reference_J_per_mol": pfset.mu_reference,
                "t0_diff_s": t0_diff_s,
                "c_tot_mol_per_m3": c_tot_phys,
                "L_ref_m": L_ref,
                "L_ref_factor": inputs.L_ref_factor,
                "dVm_alpha_dxB_phys_m3_per_mol": dVm_phys,

                # === [新增] gel shift 输出 ===
                "gel_shift_input_J_per_m3": inputs.gel_shift_Jm3,
                "gel_shift_hat": pfset.gel_shift_hat,
            },
            "diffusivity_physical": {
                "D_alpha_m2_per_s": D_alpha_phys,
                "D_comp_m2_per_s": D_comp_phys,
                "D_ratio_Dcomp_over_Dalpha": inputs.D_ratio,
            },
            "critical_radius": {
                "xB_eq": xB_eq,
                "xB_farfield": xB_farfield,
                "supersaturation_ratio": supersat_ratio,
                "Delta_mu_r_farfield_J_per_mol": delta_mu_r_inf,
                "DeltaG_v_no_vol_J_per_m3": DeltaG_v_no_vol,
                "rc_no_vol_m": rc_no_vol,
                "rc_no_vol_over_pf_dx": rc_no_vol / pf_dx,
                "rc_no_vol_over_lambda": rc_no_vol / inputs.lambda_sm,
                "DeltaVm_eff_farfield_m3_per_mol": DeltaVm_eff,
                "mu_tot_farfield_J_per_mol": mu_tot_inf,
                "DeltaG_v_with_vol_J_per_m3": DeltaG_v_with_vol,
                "rc_with_vol_m": rc_with_vol,
                "rc_with_vol_over_pf_dx": rc_with_vol / pf_dx,
                "rc_with_vol_over_lambda": rc_with_vol / inputs.lambda_sm,
                "hat_delta_mu_r_farfield": hat_delta_mu_r,
                "rc_hat_over_lambda": rc_hat_over_lambda,
                "rc_hat_over_pf_dx": rc_hat_over_dx,
            },
            "quality_checks": quality_checks,
            "inputs_C_tensor_PbTe_GPa": inputs.C_tensor_PbTe_GPa,
            "inputs_C_tensor_Ag2Te_GPa": inputs.C_tensor_Ag2Te_GPa,
            "eigenstrain_input": {
                "input_mode": "tensor" if inputs.eigenstrain_tensor is not None else "principal+rotation",
                "tensor_simulation_basis": eigenstrain_tensor,
                "voigt_for_main_cuda": eigenstrain_voigt,
                "principal_values": inputs.eigenstrain_principal,
                "rotation_matrix": inputs.eigenstrain_rotation_matrix,
                "note": "剪切分量使用张量应变定义，而非工程剪切应变 gamma_ij",
            },
            "runtime_gap_fields": RUNTIME_GAP_FIELDS,
        },
        "main_cuda_overrides": main_cuda_overrides,
    }
    return payload


def print_matrix(name: str, matrix: List[List[float]]):
    print(f"\n{name} (6x6):")
    for row in matrix:
        print("  [ " + " ".join(f"{val:10.3e}" for val in row) + " ]")


def run_from_config() -> None:
    parser = argparse.ArgumentParser(
        description="将真实物理输入参数换算为 CUDA 相场程序可用的无量纲参数"
    )
    parser.add_argument("--input-json", type=str, default=None,
                        help="真实物理输入 JSON 文件；若省略则使用脚本内置 USER_PHYSICAL_INPUTS")
    parser.add_argument("--output-json", type=str, default=OUTPUT_JSON_PATH,
                        help="输出完整转换结果 JSON")
    parser.add_argument("--output-pf-param-file", type=str, default=None,
                        help="输出 main_cuda 可直接读取的 key=value 参数文件")
    parser.add_argument("--write-example-json", type=str, default=None,
                        help="写出一份 PhysicalInputs 示例 JSON 后退出")
    parser.add_argument("--print-main-cuda-cmd", action="store_true",
                        help="打印 main_cuda 的建议调用方式")
    parser.add_argument("--no-summary", action="store_true",
                        help="不打印终端摘要")
    args = parser.parse_args()

    if args.write_example_json:
        write_example_input_json(args.write_example_json)
        print(f"已写出示例输入文件: {args.write_example_json}")
        return

    inputs = load_physical_inputs(args.input_json)
    payload = generate_payload(inputs)
    main_cuda_overrides = payload["main_cuda_overrides"]

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"已写出完整转换结果 JSON: {args.output_json}")

    if args.output_pf_param_file:
        write_main_cuda_override_file(args.output_pf_param_file, main_cuda_overrides)
        print(f"已写出 main_cuda 参数文件: {args.output_pf_param_file}")

    if not args.no_summary and PRINT_SUMMARY:
        pf_params = payload["pf_params"]
        meta = payload["metadata"]
        scales = meta["scales"]
        D_phys = meta["diffusivity_physical"]
        crit = meta["critical_radius"]
        c_tot_phys = scales["c_tot_mol_per_m3"]
        t0 = scales["t0_diff_s"]
        L_ref = scales["L_ref_m"]
        L_factor = scales["L_ref_factor"]
        dVm_phys = scales["dVm_alpha_dxB_phys_m3_per_mol"]
        quality = meta["quality_checks"]

        print("\n--- 真实物理参数（带单位） ---")
        print(f"gamma          (J/m^2)   : {inputs.gamma:.6e}")
        print(f"lambda_sm      (m)       : {inputs.lambda_sm:.6e}")
        print(f"dx_legacy      (m)       : {inputs.dx:.6e}")
        print(f"pf_dx          (m)       : {scales['pf_dx_m']:.6e}")
        print(f"pf_dx          (nm)      : {scales['pf_dx_m'] * 1.0e9:.6e}")
        if scales["phys_dx_ref_m"] is not None:
            print(f"phys_dx_ref    (m)       : {scales['phys_dx_ref_m']:.6e}")
            print(f"phys_dx_ref    (nm)      : {scales['phys_dx_ref_m'] * 1.0e9:.6e}")
        print(f"dt             (code)    : {inputs.dt:.6e}")
        print(f"eps_iso        (-)       : {inputs.eps_iso:.6e}")
        print(f"Vm_alpha_0     (m^3/mol) : {inputs.Vm_alpha_0:.6e}")
        print(f"Vm_compound    (m^3/mol) : {inputs.Vm_compound:.6e}")
        print(f"dVm_alpha_dxB  (m^3/mol) : {dVm_phys:.6e}")
        print(f"c_tot          (mol/m^3) : {c_tot_phys:.6e}")
        print(f"T              (°C)      : {inputs.temperature_C:.2f}")
        print(f"D_alpha        (m^2/s)   : {D_phys['D_alpha_m2_per_s']:.6e}")
        print(f"D_comp         (m^2/s)   : {D_phys['D_comp_m2_per_s']:.6e}")
        print(f"D_ratio=Dc/Da  (-)       : {D_phys['D_ratio_Dcomp_over_Dalpha']:.3e}")
        print(f"L_ref          (m)       : {L_ref:.6e}")
        print(f"L_ref/lambda_sm         : {L_factor:.3f}")
        print(f"lambda_sm/pf_dx         : {quality['lambda_over_pf_dx']:.6f}")
        if quality["refinement_ratio_phys_over_pf"] is not None:
            print(f"phys_dx_ref/pf_dx       : {quality['refinement_ratio_phys_over_pf']:.6f}")
        print(f"t_hat = 1.0 对应 t_phys : {t0:.6e} s")

        print("\n--- 无量纲 PFParams 输出 ---")
        for key, value in pf_params.items():
            # 跳过矩阵类型的打印，避免报错
            if not isinstance(value, list):
                print(f"{key:22s}: {value:.6e}")

        print("\n--- main_cuda 可直接覆盖的参数 ---")
        print(f"可自动落地字段数        : {len(main_cuda_overrides)}")
        print("输出格式                : key=value")
        if args.output_pf_param_file:
            print(f"参数文件路径            : {args.output_pf_param_file}")
        for key in [
            "dt", "dx", "t_real_unit", "temperature_C", "mu_reference_scale", "kappa_phi", "L_phi",
            "D_alpha", "D_compound", "Vm_compound", "dVm_alpha_dxB",
            "eps_iso_over_vB",
            "elastic_shift_dimless", "eps_xx00", "eps_yy00", "eps_zz00"
        ]:
            if key in main_cuda_overrides:
                print(f"{key:22s}: {main_cuda_overrides[key]:.6e}")

        eig = meta["eigenstrain_input"]
        print("\n--- 自动换算的本征应变 eps^00_ij ---")
        print(f"输入模式                : {eig['input_mode']}")
        for key in ["eps_xx00", "eps_yy00", "eps_zz00", "eps_yz00", "eps_xz00", "eps_xy00"]:
            print(f"{key:22s}: {eig['voigt_for_main_cuda'][key]:.6e}")

        print("\n--- 当前尚未自动化的字段 ---")
        for item in meta["runtime_gap_fields"]:
            print(f"- {item}")

        print("\n--- 自动检查与警告 ---")
        if quality["warnings"]:
            for item in quality["warnings"]:
                print(f"[warn] {item}")
        else:
            print("界面分辨率检查          : 通过")

        print("\n--- 参考参数 ---")
        print(f"xAg2Te_eq ( = xB_eq )   : {meta['xAg2Te_eq']:.6e}")
        print(f"T (°C)                  : {meta['temperature_C']:.2f}")
        print(f"mu_reference (J/mol)    : {pf_params['mu_reference']:.6e}")
        print(f"L_phi (dimensionless)   : {pf_params['L_phi']:.6e}")
        print(f"ic_phi_iface_w (grid)   : {pf_params['ic_phi_iface_w']:.6e}")
        print(f"w_physical (J/m^3)      : {scales['w_physical_J_per_m3']:.6e}")

        # μ_Ag2Te, μ_PbTe 在 x_eq 处的数值
        T_K = inputs.temperature_C + 273.15
        xAg2Te_eq = meta["xAg2Te_eq"]
        mu_Ag2Te_eq = mu_Ag2Te(T_K, xAg2Te_eq)
        mu_PbTe_eq = mu_PbTe(T_K, xAg2Te_eq)
        print(f"mu_Ag2Te(x_eq) (J/mol)   : {mu_Ag2Te_eq:.6e}")
        print(f"mu_PbTe(x_eq)  (J/mol)   : {mu_PbTe_eq:.6e}")

        # [新增] 验证反应驱动力是否为 0
        G0_pure = G_Ag2Te_Solid(T_K)
        reaction_force_at_eq = G0_pure - mu_Ag2Te_eq
        print(f"Reaction Force @ x_eq    : {reaction_force_at_eq:.6e} (Should be ~0)")

        # 经典成核临界半径
        print("\n--- 经典成核临界半径（由 vf_target + Δμ_r 推出） ---")
        print(f"xB_eq (matrix eq)       : {crit['xB_eq']:.6e}")
        print(f"xB_farfield (initial)   : {crit['xB_farfield']:.6e}")
        print(f"supersaturation S       : {crit['supersaturation_ratio']:.6e}")
        print(f"Delta_mu_r_farfield(J/mol): {crit['Delta_mu_r_farfield_J_per_mol']:.6e}")

        print("\n[无体积差] ΔG_v = c_inf * Δμ_r")
        print(f"DeltaG_v_no_vol (J/m^3) : {crit['DeltaG_v_no_vol_J_per_m3']:.6e}")
        print(f"r_c_no_vol (m)          : {crit['rc_no_vol_m']:.6e}")
        print(f"r_c_no_vol / pf_dx      : {crit['rc_no_vol_over_pf_dx']:.6e}")
        print(f"r_c_no_vol / lambda_sm  : {crit['rc_no_vol_over_lambda']:.6e}")

        print("\n[考虑体积差] ΔG_v = c_inf Δμ_r - c_inf^2 μ_tot ΔV_eff")
        print(f"DeltaVm_eff_farfield(m^3/mol): {crit['DeltaVm_eff_farfield_m3_per_mol']:.6e}")
        print(f"mu_tot_farfield (J/mol) : {crit['mu_tot_farfield_J_per_mol']:.6e}")
        print(f"DeltaG_v_with_vol (J/m^3): {crit['DeltaG_v_with_vol_J_per_m3']:.6e}")
        print(f"r_c_with_vol (m)        : {crit['rc_with_vol_m']:.6e}")
        print(f"r_c_with_vol / pf_dx    : {crit['rc_with_vol_over_pf_dx']:.6e}")
        print(f"r_c_with_vol / lambda_sm: {crit['rc_with_vol_over_lambda']:.6e}")

        print("\n[无量纲验证] 由 hat{Δμ_r} 得到的 Rc")
        print(f"hat_delta_mu_r_farfield : {crit['hat_delta_mu_r_farfield']:.6e}")
        print(f"r_c_hat / lambda_sm     : {crit['rc_hat_over_lambda']:.6e}")
        print(f"r_c_hat / pf_dx         : {crit['rc_hat_over_pf_dx']:.6e}")
        print(f"(对比) r_c_no_vol / pf_dx: {crit['rc_no_vol_over_pf_dx']:.6e}")

        # 弹性常数输入与无量纲化结果 (适配 6x6 矩阵)
        print("\n--- 弹性常数 (Extracting Diagonals from 6x6 Tensor) ---")
        # 提取对角线元素模拟旧输出
        C_PbTe = inputs.C_tensor_PbTe_GPa
        C_Ag2Te = inputs.C_tensor_Ag2Te_GPa
        print(f"C11_PbTe (GPa)          : {C_PbTe[0][0]:.3f}")
        print(f"C12_PbTe (GPa)          : {C_PbTe[0][1]:.3f}")
        print(f"C44_PbTe (GPa)          : {C_PbTe[3][3]:.3f}")
        print(f"C11_Ag2Te (GPa)         : {C_Ag2Te[0][0]:.3f}")
        print(f"C12_Ag2Te (GPa)         : {C_Ag2Te[0][1]:.3f}")
        print(f"C44_Ag2Te (GPa)         : {C_Ag2Te[3][3]:.3f}")
        
        # 打印完整矩阵
        print_matrix("Dimensionless Matrix Stiffness (PbTe)", pf_params['C_tensor_matrix_hat'])
        print_matrix("Dimensionless Precipitate Stiffness (Ag2Te)", pf_params['C_tensor_precip_hat'])
        
        c11_mat = pf_params['C_tensor_matrix_hat'][0][0]
        print(f"\nMagnitude Check: PbTe C11_hat = {c11_mat:.2f}")

        print("\n--- [新增] 弹性能密度 shift（输入与无量纲化输出）---")
        print(f"gel_shift_input (J/m^3) : {scales['gel_shift_input_J_per_m3']:.6e}")
        print(f"w_scale (J/m^3)         : {scales['w_physical_J_per_m3']:.6e}")
        print(f"gel_shift_hat (=gel/w)  : {scales['gel_shift_hat']:.6e}")

        if args.print_main_cuda_cmd and args.output_pf_param_file:
            print("\n--- 建议运行命令 ---")
            print(
                f"./main_cuda 128 128 128 0.01 100 10 10 1 "
                f"--pf-param-file {args.output_pf_param_file}"
            )

        # 其余原有 summary 你可以保留/删除，这里不改动原结构
        # 如果你希望保持完全原样，把下面注释去掉即可（我这里省略了原长段打印）

        # === 原来那大段打印如果你要完全保留，直接把你原来的 run_from_config 内容粘回去即可 ===


if __name__ == "__main__":
    run_from_config()
