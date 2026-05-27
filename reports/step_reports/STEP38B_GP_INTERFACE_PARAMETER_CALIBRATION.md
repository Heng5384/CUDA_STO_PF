# Step 38B GP Interface Parameter Calibration

## Scope

This note derives candidate `gp_W_eta` and `gp_kappa_eta` values for the current GP `eta` equation using the code's actual double-well convention:

\[
g(\eta)=\eta^2(1-\eta)^2,\qquad
g'(\eta)=2\eta(1-\eta)(1-2\eta)
\]

and interfacial energy density:

\[
f_{int}=W_\eta g(\eta)+\frac{1}{2}\kappa_\eta |\nabla \eta|^2
\]

The code path is:

- explicit double-well: `gp_W_eta * g'(\eta)`
- semi-implicit gradient: `1 + L_\eta \, dt \, \kappa_\eta k^2`

## Width convention used here

For this calibration, `l_eta = 1.0 nm` is interpreted as the **10–90 interface width** of the equilibrium 1D profile.

For this `g(\eta)` convention, the equilibrium logistic/tanh-profile gives:

\[
l_{10-90}=\ln(9)\sqrt{\frac{2\kappa_\eta}{W_\eta}}
\]

The corresponding interfacial energy is:

\[
\gamma_{\alpha GP}=\frac{\sqrt{2\kappa_\eta W_\eta}}{6}
\]

Solving for `W_eta` and `kappa_eta`:

\[
W_\eta=\frac{6\gamma \ln 9}{l_\eta}
\]

\[
\kappa_\eta=\frac{3\gamma l_\eta}{\ln 9}
\]

with:

- `gamma` in `J/m^2`
- `l_eta` in `m`
- `W_eta` in `J/m^3`
- `kappa_eta` in `J/m`

## Candidate table

See:

- [gp_interface_parameter_candidates.csv](/Users/heng/Documents/GitHub/CUDA_STO_PF/gp_interface_parameter_candidates.csv)

For `l_eta = 1.0 nm` the candidates are:

| gamma (J/m²) | l_eta (nm) | W_eta (J/m³) | kappa_eta (J/m) |
|---|---:|---:|---:|
| 0.05 | 1.0 | 6.591673732e8 | 6.826794200e-11 |
| 0.10 | 1.0 | 1.318334746e9 | 1.365358840e-10 |
| 0.20 | 1.0 | 2.636669493e9 | 2.730717680e-10 |

The derived values exactly reproduce:

- target `gamma`
- target `l_eta = 1.0 nm`

within numerical roundoff, using the formulas above.

## Interpretation

These candidates are the minimum physically motivated nonzero interface terms to test before any further `mu_GP0` tuning:

1. `gp_W_eta` sets the double-well barrier between `eta=0` and `eta=1`
2. `gp_kappa_eta` sets the GP/matrix interface cost and width
3. with both currently zero, the present model has no energetic price for growing GP interface area

## Recommendation for Step 39

Use these three candidates directly as the first interface-only scan:

- `gamma = 0.05 J/m^2`
- `gamma = 0.10 J/m^2`
- `gamma = 0.20 J/m^2`

all at:

- `l_eta = 1.0 nm`
- `gp_elastic_enabled = 0`

This isolates whether interface physics alone can turn the current mechanical-mixture GP growth from precipitate-like to finite-size, GP-zone-like behavior.
