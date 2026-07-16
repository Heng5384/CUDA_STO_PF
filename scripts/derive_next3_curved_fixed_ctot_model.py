#!/usr/bin/env python3
"""Write the auditable fixed-Ctot curved inner/outer derivation artifacts."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from scipy.integrate import quad


ROOT = Path(__file__).resolve().parents[1]


def phi0(z: float) -> float:
    return 0.5 * (1.0 - math.tanh(2.0 * z))


def phi0_prime(z: float) -> float:
    phi = phi0(z)
    return -4.0 * phi * (1.0 - phi)


def h(phi: float) -> float:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def alpha(z: float) -> float:
    return 1.0 - h(phi0(z))


def integrate(function) -> float:
    return quad(
        function, -20.0, 20.0, epsabs=1.0e-14, epsrel=1.0e-13,
        points=[0.0], limit=500,
    )[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-dir", type=Path,
        default=ROOT / "reports/pf_ctot_production_candidate",
    )
    args = parser.parse_args()
    out = args.report_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    lambda_nm = 0.6
    kappa_phi_code = 0.045
    l_phi_code = 85.50541544691814
    a_phi = integrate(lambda z: phi0_prime(z) ** 2)
    i0_h = integrate(
        lambda z: h(phi0(z)) - (1.0 if z < 0.0 else 0.0)
    )
    i1_h = integrate(
        lambda z: z * (h(phi0(z)) - (1.0 if z < 0.0 else 0.0))
    )
    i0_alpha = integrate(
        lambda z: alpha(z) - (1.0 if z > 0.0 else 0.0)
    )
    i1_alpha = integrate(
        lambda z: z * (alpha(z) - (1.0 if z > 0.0 else 0.0))
    )
    k_at = integrate(lambda z: h(phi0(z)) * alpha(z))
    k_at_1 = integrate(lambda z: z * h(phi0(z)) * alpha(z))
    planar_jump = i0_alpha
    metric_current = integrate(
        lambda z: h(phi0(z)) * alpha(z)
        + z * (
            (30.0 * phi0(z) ** 2 * (1.0 - phi0(z)) ** 2)
            * phi0_prime(z) * (1.0 - 2.0 * h(phi0(z)))
        )
    )
    i1_nm2 = i1_h * lambda_nm**2
    gamma_code = kappa_phi_code * a_phi / lambda_nm
    beta_kinetic_code = a_phi / (l_phi_code * lambda_nm)

    coefficients = [
        {
            "coefficient_id": "phase_profile_norm_A_phi",
            "exact_integral": "integral (d_phi0/dz)^2 dz = 2/3",
            "numerical_value": a_phi,
            "units": "dimensionless",
            "sign_convention": "z points beta_to_matrix; phi0_prime<0",
            "planar_limit": "2/3",
            "cylindrical_factor": "1",
            "spherical_factor": "1",
            "status": "CLOSED",
            "meaning": "capillary and Allen-Cahn kinetic solvability norm",
        },
        {
            "coefficient_id": "storage_excess_I0_h",
            "exact_integral": "integral [h(phi0)-H(-z)] dz",
            "numerical_value": i0_h,
            "units": "dimensionless_z",
            "sign_convention": "positive excess is beta-side C excess",
            "planar_limit": "0 by h(-z)=1-h(z)",
            "cylindrical_factor": "kappa*lambda in first metric moment",
            "spherical_factor": "kappa*lambda in first metric moment",
            "status": "CLOSED_ROUNDOFF_ZERO",
            "meaning": "leading storage surface excess",
        },
        {
            "coefficient_id": "storage_first_moment_I1_h",
            "exact_integral": "integral z [h(phi0)-H(-z)] dz",
            "numerical_value": i1_h,
            "units": "dimensionless_z_squared",
            "sign_convention": "positive moves equimolar surface outward",
            "planar_limit": "does not enter planar jump",
            "cylindrical_factor": "R_e-R_phi=I1_phys/R+O(epsilon^3)",
            "spherical_factor": "R_e-R_phi=2*I1_phys/R+O(epsilon^3)",
            "status": "CLOSED",
            "meaning": "first nonzero curved storage moment",
        },
        {
            "coefficient_id": "storage_first_moment_I1_physical",
            "exact_integral": "lambda^2 I1_h",
            "numerical_value": i1_nm2,
            "units": "nm^2",
            "sign_convention": "positive",
            "planar_limit": "0 radius shift",
            "cylindrical_factor": "1/R",
            "spherical_factor": "2/R",
            "status": "CLOSED",
            "meaning": "surface-of-tension to C-equimolar radius map",
        },
        {
            "coefficient_id": "interface_stretching_Gamma_C_equimolar",
            "exact_integral": "Gamma_C(R_e)=0 by definition",
            "numerical_value": 0.0,
            "units": "xB_times_nm",
            "sign_convention": "total-C equimolar surface",
            "planar_limit": "0",
            "cylindrical_factor": "kappa*V*0",
            "spherical_factor": "kappa*V*0",
            "status": "CLOSED",
            "meaning": "no leading interface-stretching anomaly on R_e",
        },
        {
            "coefficient_id": "mobility_excess_I0_alpha",
            "exact_integral": "integral [(1-h)-H(z)] dz",
            "numerical_value": i0_alpha,
            "units": "dimensionless_z",
            "sign_convention": "matrix side is z>0",
            "planar_limit": "0 by symmetry",
            "cylindrical_factor": "1",
            "spherical_factor": "1",
            "status": "CLOSED_ROUNDOFF_ZERO",
            "meaning": "leading artificial surface-diffusion coefficient",
        },
        {
            "coefficient_id": "mobility_first_moment_I1_alpha",
            "exact_integral": "integral z [(1-h)-H(z)] dz = -I1_h",
            "numerical_value": i1_alpha,
            "units": "dimensionless_z_squared",
            "sign_convention": "matrix side is z>0",
            "planar_limit": "does not enter leading planar term",
            "cylindrical_factor": "kappa*lambda",
            "spherical_factor": "kappa*lambda",
            "status": "CLOSED",
            "meaning": "nonzero next-order curved mobility moment",
        },
        {
            "coefficient_id": "planar_antitrapping_current_K0",
            "exact_integral": "integral h(phi0)[1-h(phi0)] dz = 209/1680",
            "numerical_value": k_at,
            "units": "dimensionless_z",
            "sign_convention": "positive current follows signed V_n",
            "planar_limit": "209/1680",
            "cylindrical_factor": "metric expansion required",
            "spherical_factor": "metric expansion required",
            "status": "CLOSED",
            "meaning": "zeroth moment of current profile",
        },
        {
            "coefficient_id": "planar_antitrapping_current_K1",
            "exact_integral": "integral z h(phi0)[1-h(phi0)] dz",
            "numerical_value": k_at_1,
            "units": "dimensionless_z_squared",
            "sign_convention": "z points beta_to_matrix",
            "planar_limit": "0 because h(1-h) is even",
            "cylindrical_factor": "1",
            "spherical_factor": "1",
            "status": "CLOSED_ROUNDOFF_ZERO",
            "meaning": "first current moment",
        },
        {
            "coefficient_id": "normal_current_metric_net_O_epsilon",
            "exact_integral": "integral [j0 + z*dj0/dz] dz = [z*j0]_-inf^inf",
            "numerical_value": metric_current,
            "units": "dimensionless_z",
            "sign_convention": "Jacobian=1+kappa*lambda*z+...",
            "planar_limit": "0",
            "cylindrical_factor": "0 after Jacobian cancellation",
            "spherical_factor": "0 after Jacobian cancellation",
            "status": "CLOSED_ROUNDOFF_ZERO",
            "meaning": "kappa*j_at is not a standalone missing mass term",
        },
        {
            "coefficient_id": "planar_chemical_jump_F0",
            "exact_integral": "integral [(1-h)-H(z)] dz",
            "numerical_value": planar_jump,
            "units": "dimensionless_z",
            "sign_convention": "after a(-phi0_prime)=h(1-h)",
            "planar_limit": "0",
            "cylindrical_factor": "next-order matching unresolved",
            "spherical_factor": "next-order matching unresolved",
            "status": "PLANAR_CLOSED_CURVED_OPEN",
            "meaning": "planar anti-trapping chemical-jump condition",
        },
        {
            "coefficient_id": "capillary_gamma_code",
            "exact_integral": "kappa_phi/lambda * A_phi",
            "numerical_value": gamma_code,
            "units": "code_energy_times_code_length",
            "sign_convention": "positive convex beta raises matrix mu_B",
            "planar_limit": "curvature term absent",
            "cylindrical_factor": "1/R",
            "spherical_factor": "2/R",
            "status": "CLOSED",
            "meaning": "Gibbs-Thomson coefficient in accepted code units",
        },
        {
            "coefficient_id": "allen_cahn_kinetic_beta_code",
            "exact_integral": "A_phi/(L_phi*lambda)",
            "numerical_value": beta_kinetic_code,
            "units": "code_mu_per_code_velocity",
            "sign_convention": "positive V raises required matrix mu_B",
            "planar_limit": "beta_kinetic*V_n",
            "cylindrical_factor": "1",
            "spherical_factor": "1",
            "status": "CLOSED",
            "meaning": "finite selected-L_phi kinetic term",
        },
        {
            "coefficient_id": "curved_one_sided_chemical_companion",
            "exact_integral": "requires coupled phi1/x1 matching at alpha->0",
            "numerical_value": "",
            "units": "not_closed",
            "sign_convention": "not_assigned",
            "planar_limit": "must vanish",
            "cylindrical_factor": "not_closed",
            "spherical_factor": "not_closed",
            "status": "BLOCKER",
            "meaning": "no unique conservative curved operator can be emitted",
        },
    ]
    coefficient_path = out / "next3_surface_excess_coefficients.csv"
    with coefficient_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coefficients[0]))
        writer.writeheader()
        writer.writerows(coefficients)

    inner_outer = f"""# Next3 Curved Fixed-Ctot Inner/Outer Expansion

## Source-recovered model

For the accepted equal-volume, nonelastic T400 parameterization,
`v_A=0`, `v_B=1`, `Vm_alpha=Vm_beta=1`, and `dVm_alpha/dx=0`. The source at
`cuda_kernels.cu:3368-3402`, `thermo_utils.h:586-633`, and
`cuda_kernels.cu:3500-3547` therefore reduces exactly to

```text
C = h(phi) v_B + alpha(phi) x,       alpha=1-h,
mu = [mu_B(x)-mu_A(x)]/E0 = g_alpha'(x)/E0,
M = alpha D_alpha / [g_alpha''(x)/E0],
C_t = div[alpha D_alpha grad(x) + j_corr].
```

The chemical phase driving is evaluated at `cuda_kernels.cu:397-471`; the
accepted fixed-Ctot phase transaction invokes it and reconstructs the local
storage at `main_cuda.cu:31786-31852`, with the selected semismooth PDAS path
entered at `main_cuda.cu:31965-32051`. Its nonelastic continuum form is

```text
phi_t = -L_phi {{W g'(phi) + h'(phi)[mu0-mu_B(x)]
                 - kappa_phi laplacian(phi)}}
```

with local `C` fixed during the phase substep.

## Coordinates and operators

Let `n` point from beta to matrix, `z=n/lambda`, `V_n>0` denote beta growth,
and `kappa_s=div(n)>0` for a convex beta particle. Then

```text
partial_t = partial_t|surface - (V_n/lambda) partial_z + ...
laplacian = lambda^-2 partial_zz
          + kappa_s lambda^-1 partial_z + Delta_s + ...
div(J_n n) = lambda^-1 partial_z J_n + kappa_s J_n + ...
Jacobian = 1 + kappa_s lambda z + O((kappa_s lambda)^2).
```

The leading profile is

```text
phi0(z)=0.5[1-tanh(2z)],
phi0'=-4 phi0(1-phi0),
W g'(phi0) - (kappa_phi/lambda^2) phi0'' = 0.
```

## Phase solvability

Multiplication by `phi0'` gives, in the adopted signs,

```text
mu_B(x_i)-mu0 = gamma_code kappa_s + beta_kinetic_code V_n + higher order,
gamma_code = (kappa_phi/lambda) integral(phi0'^2 dz)
           = {gamma_code:.17e},
beta_kinetic_code = integral(phi0'^2 dz)/(L_phi lambda)
                  = {beta_kinetic_code:.17e}.
```

The selected finite `L_phi` therefore has a real kinetic term; the independent
sharp oracle used in Next2 contains capillarity but no explicit kinetic term.
This difference is small in absolute chemical potential but can be amplified
near a radius where the sharp velocity changes sign.

## Leading transport matching

Write `Delta c=v_B-x_i`. At `O(lambda^-1)`,

```text
-V_n Delta c h_0' = partial_z(J_d,0 + j_at,0),
J_d,0 + j_at,0 = V_n Delta c alpha_0.
```

The current in `cuda_kernels.cu:3625-3694` obeys

```text
a(phi0)(-phi0') = h_0 alpha_0,
j_at,0 = V_n Delta c h_0 alpha_0,
J_d,0 = V_n Delta c alpha_0^2,
partial_n x_1 = (V_n Delta c/D_alpha) alpha_0.
```

Since `integral[alpha_0-H(z)]dz=0`, the planar chemical-jump anomaly closes.

## Explicit order bookkeeping

Let `delta=kappa_s lambda` and expand the total normal flux `F=J_d+j_at` as

```text
phi = phi0 + delta phi1 + ...,
x = x_i + lambda x1 + delta lambda x2 + ...,
C = C0 + delta C1 + ...,
F = F0 + delta F1 + ...,
C0 = h0 v_B + alpha0 x_i.
```

The `O(1)` inner conservation equation and constitutive split are

```text
d_z(F0 + V_n C0) = 0,
F0 = V_n Delta c alpha0,
j_at,0 = V_n Delta c h0 alpha0,
J_d,0 = D_alpha alpha0 x1' = V_n Delta c alpha0^2.
```

At `O(delta)`, metric divergence gives the formal matched equation

```text
d_z(F1 + V_n C1) = -F0 + S1,time/gauge,
D_alpha alpha0 x2'
  = F1 - j_at,1 - D_alpha alpha1 x1'.
```

Here

```text
C1 = (v_B-x_i) h_phi(phi0) phi1 + alpha0 x1
     + dividing-surface gauge terms,
```

while `j_at,1` contains `phi1`, `x1`, the velocity correction, and the same
gauge. The curved chemical-jump coefficient is the matched integral of
`x2'-x2,outer'`. It therefore cannot be computed from `F0` or the metric
Jacobian alone.

## Curvature order and blocker

The surface-balance terms at `O(kappa_s lambda)` are not obtained by reading
`kappa_s*j_at` alone. Multiplication by the metric Jacobian produces

```text
integral (1+kappa_s lambda z)
  [partial_z j_at + kappa_s lambda j_at] dz
= [ (1+kappa_s lambda z) j_at ]_-inf^inf = 0.
```

On the total-C equimolar surface, `Gamma_C=0`; the leading stretching
coefficient is zero. The symmetric capacity interpolation also gives zero
leading surface-mobility excess. However, the next moments are nonzero:
`I1_h={i1_h:.17e}` and `I1_alpha={i1_alpha:.17e}`. At the same order the
one-sided capacity `alpha=1-h` vanishes cubically in the beta tail. More
explicitly, with `u=1-phi0`, `alpha0=10u^3+O(u^4)` as `z -> -infinity`.
Consequently the equation for `x2'` divides the still-undetermined numerator
`F1-j_at,1-D_alpha alpha1 x1'` by a cubically vanishing capacity. Regular
matching requires cancellation conditions supplied by the coupled `phi1`,
`x1`, outer-gradient, and moving-surface gauge boundary-value problem.

That problem does not determine a unique local companion-current shape from
the existing first-order conditions alone. Choosing one would add an
unproved physics operator. The expansion is therefore closed through leading
Stefan/Gibbs-Thomson and planar anti-trapping order, but not through the full
curved chemical-jump order required to implement `curved_matched_v1`.

`curved_asymptotic_derivation_status=BLOCKED_CURVED_MATCHED_ASYMPTOTIC_DERIVATION_INCOMPLETE`
"""
    (out / "next3_curved_inner_outer_expansion.md").write_text(inner_outer)

    jump = f"""# Next3 Sharp Jump-Condition Derivation

## Brackets and orientation

`n` points beta to matrix. Define `[J_n]=J_n^alpha-J_n^beta` and
`[C]=C_beta-C_alpha`; this mixed bracket convention makes a growing particle
have positive quantities on both sides of the Stefan equation.

For metric Jacobian `mathcal J(n)` and sharp reference on `R_s`, define

```text
Gamma_C = integral mathcal J(n) [C_PF-C_sharp(R_s)] dn,
J_s = integral mathcal J(n) [J_t,PF-J_t,sharp] dn.
```

Reynolds transport and the conservative PDE give

```text
[J_n] = V_n[C] + partial_t Gamma_C
        + kappa_s V_n Gamma_C + div_s(J_s).
```

## Audited coefficients

For `R_s=R_e`, the total-C equimolar/h-volume surface,

```text
Gamma_C = 0,
kappa_s V_n Gamma_C = 0,
J_n^beta = 0,
[J_n] = V_n(v_B-x_i)
```

for radial motion. The leading tangential mobility excess is proportional to
`integral[(1-h)-H(z)]dz={i0_alpha:.17e}`, hence is zero to quadrature
roundoff. The normal correction has no net far-field source: its apparent
`kappa_s*j_at` contribution cancels exactly against the first metric moment
of `partial_n j_at`.

At `phi=0.5`, by contrast,

```text
Gamma_C = (v_B-x_i) kappa_s I1_phys + higher order,
I1_phys = {i1_nm2:.17e} nm^2.
```

This is why the Stefan balance must use `R_e`, while Gibbs-Thomson written on
the surface of tension must be transformed to that radius convention.

## What is and is not closed

The following are closed without fitting: leading equilibrium, capillary and
finite-L_phi kinetic terms, one-sided Stefan balance, beta flux zero, surface
mapping, zero leading stretching, zero leading surface diffusion, and planar
anti-trapping chemical-jump cancellation. The full curved chemical-potential
jump coefficient is not closed because it needs the coupled next-order
one-sided inner solution. `next3_surface_excess_coefficients.csv` records the
exact integral, units, sign, geometry factor, and status of every coefficient.
"""
    (out / "next3_jump_condition_derivation.md").write_text(jump)

    correction = f"""# Next3 Audit of `planar_antitrapping_v1`

## Source operator

`cuda_kernels.cu:3625-3694` evaluates

```text
j_at = lambda a(phi) (x_alpha-v_B) phi_t n_beta,
a(phi)=h(phi)[1-h(phi)]/[4 phi(1-phi)].
```

Because `n_beta=-n` and `phi_t=-(V_n/lambda)phi0'`, its outward component is
`V_n(v_B-x_i)a(phi0)(-phi0')`. The exact planar identity
`a(phi0)(-phi0')=h_0(1-h_0)` shows that the coefficient normalization and sign
are internally consistent with the accepted planar derivation.

## What it removes

It converts the leading diffuse matrix gradient from a nonmatching one-sided
extension to `partial_n x proportional to (1-h)`. The remaining planar jump
integral is `integral[(1-h)-H(z)]dz={planar_jump:.17e}`, zero to roundoff.
This is the term the current removes; it is not a numerical-solver repair.

## Curved interpretation

The incidence divergence automatically contains the geometric divergence of
the current. The isolated `kappa_s*j_at` term has integral
`kappa_s lambda K0`, with `K0={k_at:.17e}`, but it cannot be interpreted alone:
the metric-weighted `z partial_z j_at` term contributes exactly the negative
amount. The net first-order mass-source coefficient is
`{metric_current:.17e}`. Therefore the prior statement that a missing
stretching term must cancel `kappa_s*j_at` was incomplete.

The current is odd in `V_n`. In the tested radial cases it adds flux in the
same signed direction as the current interface motion, so it increases both
growth and shrinkage magnitudes. That observation does not prove a sign error:
the planar matrix accepted the same sign. It demonstrates that the unclosed
curved chemical/profile matching changes kinetics in the wrong quantitative
direction for the Next2 matrix.

## Decision

- sign-convention mismatch: **not demonstrated**;
- planar coefficient-normalization mismatch: **not found**;
- missing leading equimolar stretching term: **no, coefficient is zero**;
- missing leading artificial surface-diffusion term: **no, coefficient is zero**;
- full curved one-sided chemical-jump closure: **missing**;
- overwrite or reinterpret the old mode: **forbidden**.

`current_planar_correction_status=REJECTED_FOR_CURVED_QUANTITATIVE_USE`
"""
    (out / "next3_current_correction_audit.md").write_text(correction)

    design = """# Next3 Curved Candidate Design Decision

No `curved_matched_v1` operator is emitted in this stage.

The leading curved surface-balance coefficients that might have supplied a
unique companion term are zero on the authoritative equimolar surface. The
first nonzero moments occur in the next-order one-sided chemical-matching
problem, where the beta-tail capacity degeneracy and the coupled phase-profile
correction must be solved together. A family of conservative curvature
currents can satisfy the same integrated jump, so conservation and planar
limit alone do not choose a unique member. Selecting a shape from that family
would be an unproved modeling choice, prohibited by this goal.

The reserved mode name is `curved_matched_v1`; it remains unavailable and
cannot be selected at runtime. `off` remains the default, and
`planar_antitrapping_v1` remains an explicit planar-benchmark-only mode.

`curved_candidate_mode=NOT_IMPLEMENTED_DERIVATION_GATE_FAILED`

`curved_candidate_conservation=NOT_EVALUATED`

`curved_candidate_energy_ledger=NOT_EVALUATED`

`curved_candidate_planar_limit=NOT_EVALUATED`
"""
    (out / "next3_curved_candidate_design.md").write_text(design)
    print(f"coefficients={coefficient_path}")
    print("curved_asymptotic_derivation_status=BLOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
