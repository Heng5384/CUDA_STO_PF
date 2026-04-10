#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Elastic anisotropy + predicted nucleus morphology
(three-case version)

Physical interpretation
-----------------------
Case 1: matrix-only constraint
    B_matrix(n) = Cp:e0:e0 - a^T Omega_m(n) a

Case 2: with stiffness perturbation / mismatch
    B_pert(n) = Cp:e0:e0 - a^T Omega_eff(n) a

Case 3: with stiffness perturbation + xB_out-controlled matrix chemical expansion
    This is a sharp-interface approximation of the current CUDA full-model idea:
    the matrix side carries an isotropic chemical swelling strain controlled by xB_out.

    In the CUDA code, the full-model eigenstrain is approximately
        eps0(x) = (1-h) * eps_c(xB) * I + h * eps00
    so for a far-field directional estimate we approximate the matrix-side eigenstrain by
        eps_m_chem = eps_c(xB_out) * I
    and use the effective misfit
        Delta_eps = eps_Ag2Te - eps_m_chem

    Then
        B_chem(n) = Cp:Delta_eps:Delta_eps - a^T Omega_eff(n) a

Notes
-----
- eps0 is treated as the transformation eigenstrain of Ag2Te itself.
- Therefore sigma0 = Cp : eps0, not Cm : eps0.
- The matrix (PbTe) enters through the elastic constraint operator Omega(n).
- Case 3 is not a full reproduction of the spatially varying CUDA full-model.
  It is a far-field xB_out-based approximation meant to mimic the matrix-side
  chemical swelling shift discussed in the CUDA code.

Outputs
-------
Three rows of figures:
  (1) 3D surface of log10(B(n))
  (2) Polar map of log10(B(n))
  (3) Predicted oblate nucleus morphology
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


# =========================================================
# 1. Input: stiffness tensors and Ag2Te eigenstrain
# =========================================================

# PbTe matrix stiffness tensor (GPa)
C_tensor_PbTe_GPa = np.array([
    [104.0,   6.0,   6.0,   0.0,   0.0,   0.0],
    [  6.0, 104.0,   6.0,   0.0,   0.0,   0.0],
    [  6.0,   6.0, 104.0,   0.0,   0.0,   0.0],
    [  0.0,   0.0,   0.0,  14.0,   0.0,   0.0],
    [  0.0,   0.0,   0.0,   0.0,  14.0,   0.0],
    [  0.0,   0.0,   0.0,   0.0,   0.0,  14.0]
], dtype=float)

# Ag2Te precipitate stiffness tensor (GPa)
C_tensor_Ag2Te_GPa = np.array([
    [80.0, 37.0, 49.0,  0.0,  4.0,  0.0],
    [37.0, 68.0, 40.0,  0.0,  2.0,  0.0],
    [49.0, 40.0, 90.0,  0.0,  4.0,  0.0],
    [ 0.0,  0.0,  0.0, 16.0,  0.0,  1.0],
    [ 4.0,  2.0,  4.0,  0.0, 10.0,  0.0],
    [ 0.0,  0.0,  0.0,  1.0,  0.0, 12.0]
], dtype=float)

# Convert GPa -> Pa
Cm_voigt = C_tensor_PbTe_GPa * 1.0e9
Cp_voigt = C_tensor_Ag2Te_GPa * 1.0e9

# Ag2Te eigenstrain (belongs to precipitate itself)
eps_diag = np.array([0.046, -0.022, -0.017], dtype=float)
eps0 = np.diag(eps_diag)

# stiffness perturbation strength
# eta = 0.0 -> pure matrix constraint
# eta = 1.0 -> full mismatch-included projected operator
eta = 1.0

# Predicted morphology parameters
# a,b are long axes; c is short axis
long_axis_a = 1.0
long_axis_b = 1.0
short_axis_c = 0.45

# xB_out-controlled matrix chemical expansion (matches current CUDA defaults)
v_B = 1.0
eps_iso_over_vB = 0.00233 / v_B
xB_out = 0.03


# =========================================================
# 2. Voigt -> full fourth-order tensor
# =========================================================

def voigt_to_tensor_4th(C6: np.ndarray) -> np.ndarray:
    """
    Convert 6x6 stiffness matrix in Voigt notation to full C_ijkl.
    Voigt order:
        0 -> 11
        1 -> 22
        2 -> 33
        3 -> 23
        4 -> 13
        5 -> 12
    """
    pairs = {
        0: (0, 0),
        1: (1, 1),
        2: (2, 2),
        3: (1, 2),
        4: (0, 2),
        5: (0, 1),
    }

    C4 = np.zeros((3, 3, 3, 3), dtype=float)

    for I in range(6):
        for J in range(6):
            i, j = pairs[I]
            k, l = pairs[J]
            val = C6[I, J]

            # minor symmetries
            C4[i, j, k, l] = val
            C4[j, i, k, l] = val
            C4[i, j, l, k] = val
            C4[j, i, l, k] = val

    return C4


Cm = voigt_to_tensor_4th(Cm_voigt)
Cp = voigt_to_tensor_4th(Cp_voigt)


# =========================================================
# 3. Elastic building blocks
# =========================================================

def sigma_from_C_and_eps(C4: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """sigma_ij = C_ijkl eps_kl"""
    return np.einsum("ijkl,kl->ij", C4, eps)


def M_matrix(C4: np.ndarray, n: np.ndarray) -> np.ndarray:
    """
    Projected stiffness matrix:
        M_ij(n) = C_iklj n_k n_l
    """
    return np.einsum("iklj,k,l->ij", C4, n, n)


def omega_from_M(M: np.ndarray) -> np.ndarray:
    """Omega = inv(M)"""
    return np.linalg.inv(M)


def make_case_tensors(eps_m_matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Build effective misfit quantities using
        Delta_eps = eps_precipitate - eps_matrix
    and evaluate sigma0, const with precipitate stiffness Cp.
    """
    delta_eps = eps0 - eps_m_matrix
    sigma0 = sigma_from_C_and_eps(Cp, delta_eps)
    const = np.einsum("ijkl,ij,kl->", Cp, delta_eps, delta_eps)
    return sigma0, const


# Case 1 and 2: no matrix-side chemical swelling
sigma0_p, const_p = make_case_tensors(np.zeros((3, 3), dtype=float))

# Case 3: matrix-side isotropic chemical swelling controlled by xB_out
# Mirrors the far-field matrix term eps_c(xB_out) * I in the CUDA full-model.
eps_c_out = xB_out * eps_iso_over_vB
eps_matrix_chem = np.eye(3, dtype=float) * eps_c_out
sigma0_chem, const_chem = make_case_tensors(eps_matrix_chem)


# =========================================================
# 4. B(n): definitions
# =========================================================

def B_matrix_only(n: np.ndarray) -> float:
    """
    Matrix-only constraint:
        B_matrix(n) = Cp:e0:e0 - a^T Omega_m a

    where
        a = sigma0_p^T n
        Omega_m = inv(M_m)
        M_m = projection of matrix stiffness along n
    """
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)

    M_m = M_matrix(Cm, n)
    Om_m = omega_from_M(M_m)

    a = sigma0_p.T @ n
    return const_p - a @ Om_m @ a


def B_with_perturbation(n: np.ndarray, eta: float = 1.0) -> float:
    """
    Include stiffness mismatch through an effective projected stiffness:
        M_eff = M_m + eta (M_p - M_m)
              = (1-eta) M_m + eta M_p

    Then:
        B_pert(n) = Cp:e0:e0 - a^T Omega_eff a
    """
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)

    M_m = M_matrix(Cm, n)
    M_p = M_matrix(Cp, n)
    M_eff = M_m + eta * (M_p - M_m)
    Om_eff = omega_from_M(M_eff)

    a = sigma0_p.T @ n
    return const_p - a @ Om_eff @ a


def B_with_xBout_chem_and_perturbation(n: np.ndarray, eta: float = 1.0) -> float:
    """
    Add the matrix-side isotropic chemical swelling controlled by xB_out.

    Sharp-interface approximation of the CUDA full-model idea:
        eps_matrix = eps_c(xB_out) * I
        Delta_eps  = eps_Ag2Te - eps_matrix

    Then use the same mismatch-aware projected operator Omega_eff.
    """
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)

    M_m = M_matrix(Cm, n)
    M_p = M_matrix(Cp, n)
    M_eff = M_m + eta * (M_p - M_m)
    Om_eff = omega_from_M(M_eff)

    a = sigma0_chem.T @ n
    return const_chem - a @ Om_eff @ a


# =========================================================
# 5. Sphere sampling
# =========================================================

n_theta = 181
n_phi = 361

theta = np.linspace(0.0, np.pi, n_theta)
phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
TH, PH = np.meshgrid(theta, phi, indexing="ij")

NX = np.sin(TH) * np.cos(PH)
NY = np.sin(TH) * np.sin(PH)
NZ = np.cos(TH)


def sample_B_grid(case: str, eta: float = 1.0) -> np.ndarray:
    Bvals = np.zeros_like(TH)
    for i in range(n_theta):
        for j in range(n_phi):
            nvec = np.array([NX[i, j], NY[i, j], NZ[i, j]])
            if case == "matrix":
                Bvals[i, j] = B_matrix_only(nvec)
            elif case == "pert":
                Bvals[i, j] = B_with_perturbation(nvec, eta=eta)
            elif case == "chem":
                Bvals[i, j] = B_with_xBout_chem_and_perturbation(nvec, eta=eta)
            else:
                raise ValueError(f"Unknown case: {case}")

    return np.maximum(Bvals, 1e-30)


# =========================================================
# 6. Find distinct minima
# =========================================================

def find_distinct_minima(Bvals: np.ndarray, theta: np.ndarray, phi: np.ndarray,
                         top_k: int = 40, dot_tol: float = 0.995):
    """
    Find several distinct low-energy directions from gridded B(n).
    Uses abs(dot) to merge opposite directions as equivalent if nearly collinear.
    """
    flat_idx = np.argsort(Bvals.ravel())[:top_k]
    distinct = []

    for idx in flat_idx:
        it, ip = np.unravel_index(idx, Bvals.shape)
        th = theta[it]
        ph = phi[ip]

        nvec = np.array([
            np.sin(th) * np.cos(ph),
            np.sin(th) * np.sin(ph),
            np.cos(th)
        ])

        is_new = True
        for old in distinct:
            if abs(np.dot(nvec, old["n"])) > dot_tol:
                is_new = False
                break

        if is_new:
            distinct.append({
                "B": Bvals[it, ip],
                "theta_deg": np.degrees(th),
                "phi_deg": np.degrees(ph),
                "n": nvec
            })

    return distinct


# =========================================================
# 7. Predicted morphology from lowest-energy normal
# =========================================================

def build_orthonormal_basis_from_normal(n_short: np.ndarray):
    """
    e3 = short axis direction (preferred normal)
    e1, e2 span the long-axis plane
    """
    e3 = np.asarray(n_short, dtype=float)
    e3 = e3 / np.linalg.norm(e3)

    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, e3)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(helper, e3)
    e1 = e1 / np.linalg.norm(e1)

    e2 = np.cross(e3, e1)
    e2 = e2 / np.linalg.norm(e2)

    return e1, e2, e3


def create_rotated_ellipsoid(e1, e2, e3, a=1.0, b=1.0, c=0.45, nu=120, nv=60):
    """
    Create a rotated oblate ellipsoid:
        local coordinates:
            x = a cos(u) sin(v)
            y = b sin(u) sin(v)
            z = c cos(v)
    then rotate with basis [e1 e2 e3].
    """
    u = np.linspace(0.0, 2.0 * np.pi, nu)
    v = np.linspace(0.0, np.pi, nv)
    U, V = np.meshgrid(u, v)

    x_local = a * np.cos(U) * np.sin(V)
    y_local = b * np.sin(U) * np.sin(V)
    z_local = c * np.cos(V)

    Rmat = np.column_stack([e1, e2, e3])
    pts_local = np.stack([x_local, y_local, z_local], axis=0).reshape(3, -1)
    pts_global = Rmat @ pts_local

    Xg = pts_global[0].reshape(x_local.shape)
    Yg = pts_global[1].reshape(y_local.shape)
    Zg = pts_global[2].reshape(z_local.shape)

    return Xg, Yg, Zg


# =========================================================
# 8. Compute all three cases
# =========================================================

B_matrix_vals = sample_B_grid(case="matrix")
B_pert_vals = sample_B_grid(case="pert", eta=eta)
B_chem_vals = sample_B_grid(case="chem", eta=eta)

logB_matrix = np.log10(B_matrix_vals)
logB_pert = np.log10(B_pert_vals)
logB_chem = np.log10(B_chem_vals)

mins_matrix = find_distinct_minima(B_matrix_vals, theta, phi, top_k=40)
mins_pert = find_distinct_minima(B_pert_vals, theta, phi, top_k=40)
mins_chem = find_distinct_minima(B_chem_vals, theta, phi, top_k=40)

print("\n=== Matrix-only minima (Ag2Te eigenstrain + PbTe matrix constraint) ===")
for item in mins_matrix[:8]:
    print(f"B = {item['B']:.6e} Pa, theta = {item['theta_deg']:.2f}°, phi = {item['phi_deg']:.2f}°, n = {item['n']}")

print("\n=== With perturbation minima (Ag2Te eigenstrain + stiffness mismatch) ===")
for item in mins_pert[:8]:
    print(f"B = {item['B']:.6e} Pa, theta = {item['theta_deg']:.2f}°, phi = {item['phi_deg']:.2f}°, n = {item['n']}")

print("\n=== With perturbation + xB_out-controlled matrix chemical expansion minima ===")
for item in mins_chem[:8]:
    print(f"B = {item['B']:.6e} Pa, theta = {item['theta_deg']:.2f}°, phi = {item['phi_deg']:.2f}°, n = {item['n']}")

print("\n=== Matrix chemical swelling parameters used in Case 3 ===")
print(f"xB_out = {xB_out:.6f}")
print(f"eps_iso_over_vB = {eps_iso_over_vB:.6e}")
print(f"eps_c_out = xB_out * eps_iso_over_vB = {eps_c_out:.6e}")
print("eps_matrix_chem =")
print(eps_matrix_chem)
print("effective Delta_eps = eps0 - eps_matrix_chem =")
print(eps0 - eps_matrix_chem)

# morphology basis for matrix-only
n_pred_matrix = mins_matrix[0]["n"]
e1_m, e2_m, e3_m = build_orthonormal_basis_from_normal(n_pred_matrix)
Xell_m, Yell_m, Zell_m = create_rotated_ellipsoid(
    e1_m, e2_m, e3_m,
    a=long_axis_a, b=long_axis_b, c=short_axis_c
)

# morphology basis for perturbation
n_pred_pert = mins_pert[0]["n"]
e1_p, e2_p, e3_p = build_orthonormal_basis_from_normal(n_pred_pert)
Xell_p, Yell_p, Zell_p = create_rotated_ellipsoid(
    e1_p, e2_p, e3_p,
    a=long_axis_a, b=long_axis_b, c=short_axis_c
)

# morphology basis for chemical-swelling case
n_pred_chem = mins_chem[0]["n"]
e1_c, e2_c, e3_c = build_orthonormal_basis_from_normal(n_pred_chem)
Xell_c, Yell_c, Zell_c = create_rotated_ellipsoid(
    e1_c, e2_c, e3_c,
    a=long_axis_a, b=long_axis_b, c=short_axis_c
)

print("\n=== Predicted morphology basis: matrix-only ===")
print("short axis e3 =", e3_m)
print("long axis e1  =", e1_m)
print("long axis e2  =", e2_m)

print("\n=== Predicted morphology basis: with perturbation ===")
print("short axis e3 =", e3_p)
print("long axis e1  =", e1_p)
print("long axis e2  =", e2_p)

print("\n=== Predicted morphology basis: with perturbation + xB_out-controlled matrix swelling ===")
print("short axis e3 =", e3_c)
print("long axis e1  =", e1_c)
print("long axis e2  =", e2_c)


# =========================================================
# 9. Plot helpers
# =========================================================

def plot_B_surface(ax, Bvals, logB, mins, title):
    Bmin = Bvals.min()
    Bmax = Bvals.max()

    R = 0.35 + 0.65 * (Bvals - Bmin) / (Bmax - Bmin + 1e-30)
    X = R * NX
    Y = R * NY
    Z = R * NZ

    norm = Normalize(vmin=logB.min(), vmax=logB.max())
    facecolors = cm.coolwarm(norm(logB))

    ax.plot_surface(
        X, Y, Z,
        facecolors=facecolors,
        rstride=2, cstride=2,
        linewidth=0.1,
        antialiased=True,
        shade=False
    )

    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    for item in mins[:6]:
        nvec = item["n"]
        th = np.radians(item["theta_deg"])
        ph = np.radians(item["phi_deg"])
        it = np.argmin(np.abs(theta - th))
        ip = np.argmin(np.abs(phi - ph))
        rr = R[it, ip]
        ax.scatter([rr * nvec[0]], [rr * nvec[1]], [rr * nvec[2]],
                   color="yellow", edgecolors="k", s=50)

    return Normalize(vmin=logB.min(), vmax=logB.max())


def plot_B_polar(ax, logB, mins, title):
    pcm = ax.pcolormesh(PH, TH, logB, shading="auto", cmap="coolwarm")
    ax.set_title(title)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_rticks([np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6, np.pi])
    ax.set_yticklabels(["30°", "60°", "90°", "120°", "150°", "180°"])

    for item in mins[:6]:
        th = np.radians(item["theta_deg"])
        ph = np.radians(item["phi_deg"])
        ax.scatter(ph, th, s=50, facecolor="yellow", edgecolor="k", zorder=5)

    return pcm


def plot_predicted_morphology(ax, Xell, Yell, Zell, e1, e2, e3, title):
    ax.plot_surface(Xell, Yell, Zell, color="lightsteelblue", alpha=0.88, linewidth=0)

    axis_scale = 1.4
    ax.quiver(0, 0, 0, axis_scale, 0, 0, color="k", linewidth=1.5)
    ax.quiver(0, 0, 0, 0, axis_scale, 0, color="k", linewidth=1.5)
    ax.quiver(0, 0, 0, 0, 0, axis_scale, color="k", linewidth=1.5)
    ax.text(axis_scale, 0, 0, "x", fontsize=10)
    ax.text(0, axis_scale, 0, "y", fontsize=10)
    ax.text(0, 0, axis_scale, "z", fontsize=10)

    # short axis
    ax.quiver(0, 0, 0, e3[0], e3[1], e3[2], color="red", linewidth=2.5)
    ax.quiver(0, 0, 0, -e3[0], -e3[1], -e3[2], color="red", linewidth=2.5)

    # long axes
    ax.quiver(0, 0, 0, e1[0], e1[1], e1[2], color="green", linewidth=2.0)
    ax.quiver(0, 0, 0, -e1[0], -e1[1], -e1[2], color="green", linewidth=2.0)
    ax.quiver(0, 0, 0, e2[0], e2[1], e2[2], color="orange", linewidth=2.0)
    ax.quiver(0, 0, 0, -e2[0], -e2[1], -e2[2], color="orange", linewidth=2.0)

    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


# =========================================================
# 10. Plot: three rows
# =========================================================

fig = plt.figure(figsize=(18, 18))

# ----- Row 1: matrix-only
ax11 = fig.add_subplot(3, 3, 1, projection="3d")
norm_m = plot_B_surface(
    ax11, B_matrix_vals, logB_matrix, mins_matrix,
    r"Matrix-only: 3D surface of $\log_{10} B_{\mathrm{matrix}}(\mathbf{n})$"
)

ax12 = fig.add_subplot(3, 3, 2, projection="polar")
pcm_m = plot_B_polar(
    ax12, logB_matrix, mins_matrix,
    r"Matrix-only: polar map of $\log_{10} B_{\mathrm{matrix}}(\mathbf{n})$"
)

ax13 = fig.add_subplot(3, 3, 3, projection="3d")
plot_predicted_morphology(
    ax13, Xell_m, Yell_m, Zell_m, e1_m, e2_m, e3_m,
    "Matrix-only: predicted nucleus morphology"
)

# ----- Row 2: with perturbation
ax21 = fig.add_subplot(3, 3, 4, projection="3d")
norm_p = plot_B_surface(
    ax21, B_pert_vals, logB_pert, mins_pert,
    r"With perturbation: 3D surface of $\log_{10} B_{\mathrm{pert}}(\mathbf{n})$"
)

ax22 = fig.add_subplot(3, 3, 5, projection="polar")
pcm_p = plot_B_polar(
    ax22, logB_pert, mins_pert,
    r"With perturbation: polar map of $\log_{10} B_{\mathrm{pert}}(\mathbf{n})$"
)

ax23 = fig.add_subplot(3, 3, 6, projection="3d")
plot_predicted_morphology(
    ax23, Xell_p, Yell_p, Zell_p, e1_p, e2_p, e3_p,
    "With perturbation: predicted nucleus morphology"
)

# ----- Row 3: with perturbation + matrix chemical swelling
ax31 = fig.add_subplot(3, 3, 7, projection="3d")
norm_c = plot_B_surface(
    ax31, B_chem_vals, logB_chem, mins_chem,
    r"With perturbation + matrix swelling: 3D surface of $\log_{10} B_{\mathrm{chem}}(\mathbf{n})$"
)

ax32 = fig.add_subplot(3, 3, 8, projection="polar")
pcm_c = plot_B_polar(
    ax32, logB_chem, mins_chem,
    r"With perturbation + matrix swelling: polar map of $\log_{10} B_{\mathrm{chem}}(\mathbf{n})$"
)

ax33 = fig.add_subplot(3, 3, 9, projection="3d")
plot_predicted_morphology(
    ax33, Xell_c, Yell_c, Zell_c, e1_c, e2_c, e3_c,
    "With perturbation + matrix swelling: predicted nucleus morphology"
)

# colorbars
cbar_m = fig.colorbar(
    cm.ScalarMappable(norm=norm_m, cmap=cm.coolwarm),
    ax=[ax11, ax12],
    shrink=0.72,
    pad=0.04
)
cbar_m.set_label(r"$\log_{10}[B_{\mathrm{matrix}}(\mathbf{n})]$ (Pa)")

cbar_p = fig.colorbar(
    cm.ScalarMappable(norm=norm_p, cmap=cm.coolwarm),
    ax=[ax21, ax22],
    shrink=0.72,
    pad=0.04
)
cbar_p.set_label(r"$\log_{10}[B_{\mathrm{pert}}(\mathbf{n})]$ (Pa)")

cbar_c = fig.colorbar(
    cm.ScalarMappable(norm=norm_c, cmap=cm.coolwarm),
    ax=[ax31, ax32],
    shrink=0.72,
    pad=0.04
)
cbar_c.set_label(r"$\log_{10}[B_{\mathrm{chem}}(\mathbf{n})]$ (Pa)")

plt.tight_layout()
plt.show()
