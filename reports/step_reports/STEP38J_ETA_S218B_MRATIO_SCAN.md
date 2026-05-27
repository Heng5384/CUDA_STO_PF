# Step38J ETA S2.18b M-ratio Scan

## Scope

This step adds `gp_L_eta_mode=sto_S218b` and computes the kernel-used `gp_L_eta` from the STO-SM Eq. S2.18b mixed-control expression:

\[
L_\eta =
\frac{2(v_A+v_B)}{3 c \lambda_\eta}
\left[
\frac{1}{M_\eta} + \frac{|\zeta_{0,\eta}\zeta_\eta|\lambda_\eta}{2D_\alpha}
\right]^{-1}
\]

with:

\[
M_{\mathrm{crit},\eta} = \frac{2D_\alpha}{|\zeta_{0,\eta}\zeta_\eta|\lambda_\eta}
\]

and asymptotic limits:

\[
L_{\eta,\mathrm{reaction}} \approx
\frac{2(v_A+v_B)}{3 c \lambda_\eta} M_\eta
\]

\[
L_{\eta,\mathrm{diff}} =
\frac{4(v_A+v_B)D_\alpha}{3 c \lambda_\eta^2 |\zeta_{0,\eta}\zeta_\eta|}
\]

The scan is run in `sto_S218b` mode, so `gp_L_eta` is no longer an arbitrary manual fraction of `L_eta_diff`; it is computed from `M_eta` or `M_eta / Mcrit_eta`.

## Conventions

- `W` / `kappa` use the Step38I document convention:
  - `W = 12 gamma / lambda`
  - `kappa = 1.5 gamma lambda`
- `eta` chemical potentials use the Step38H corrected dimensionless scale, aligned with `phi`.
- `Mcrit_eta`, `L_eta_full`, and `L_eta_diff` use `abs(zeta0_eta * zeta_eta)` for positive-definite regime diagnostics, matching the current diffusion-limit handling.

## Why scan M/Mcrit instead of absolute M_eta

Scanning `M_eta / Mcrit_eta` is better conditioned than guessing an absolute `M_eta` because it directly tells us which regime we are in:

- `M/Mcrit << 1`: reaction/interface-controlled
- `M/Mcrit ~ 1`: mixed-control
- `M/Mcrit >> 1`: diffusion-controlled

This makes the interpretation transferable even if `D_alpha`, `zeta_eta`, or `zeta0_eta` move when the thermodynamic setup changes.

## Smoke-scan summary (500 steps)

| M_ratio | Regime | L_full/L_diff | S_eta_total | xB_final_range | eta_far_final | Recommendation |
|---|---:|---:|---:|---|---:|---|
| `1e-4` | strongly reaction-controlled | `9.999e-05` | `3.964e-02` | `[0.02812, 0.05350]` | `2.64e-04` | too slow |
| `1e-3` | strongly reaction-controlled | `9.990e-04` | `3.961e-01` | `[0.02814, 0.05058]` | `8.69e-05` | baseline candidate |
| `1e-2` | reaction-controlled | `9.901e-03` | `3.926e+00` | `[0.02829, 0.03650]` | `1.96e-05` | baseline candidate |
| `1e-1` | mixed-control | `9.091e-02` | `3.604e+01` | `[0.02830, 0.03159]` | `6.36e-10` | aggressive |
| `1` | mixed-control | `5.000e-01` | `1.982e+02` | `[0.02874, 0.03093]` | `0` | aggressive |
| `10` | mixed-control | `9.091e-01` | `3.604e+02` | `[0.02879, 0.03083]` | `~0` | aggressive |

## Interpretation

1. Eq. S2.18b is wired correctly.
   - `L_eta_full / L_eta_diff` increases monotonically with `M/Mcrit`.
   - `M/Mcrit = 1` gives `L_eta_full / L_eta_diff = 0.5`, as expected from the mixed-resistance form.

2. The reaction-controlled side is usable.
   - `M/Mcrit = 1e-4` is numerically quiet but too slow.
   - `M/Mcrit = 1e-3` and `1e-2` both remain localized and do not clip `xB`.

3. The mixed/diffusion side becomes stiff quickly.
   - Already by `M/Mcrit = 1e-1`, `S_eta_total > 10`.
   - `M/Mcrit = 1, 10` push `S_eta_total` into clearly aggressive territory, even though this 500-step smoke did not yet crash.

4. This smoke scan is enough to choose the next full-scan window.
   - We do **not** need to start the 20000-step full scan across the whole `1e-4 ... 10` range.
   - The meaningful window is now much narrower.

## Recommended next baseline window

Recommended Step38J follow-up full scan:

- primary:
  - `M_ratio = 1e-3`
  - `M_ratio = 1e-2`
- optional conservative compare:
  - `M_ratio = 3e-3`

Current baseline candidate:

- **`M_ratio = 1e-2`**

Reason:

- still reaction-controlled
- `L_eta_full / L_eta_diff ≈ 0.0099`, comfortably below the diffusion limit
- visible eta kinetics without entering the `S_eta_total > 10` aggressive zone

Conservative compare:

- **`M_ratio = 1e-3`**

Reason:

- strongly reaction-controlled
- safer/stiffer-margin option if the full 20000-step run shows late-time mass-drift sensitivity at `1e-2`

## Note

This step changes only the eta kinetic-factor calculation in `gp_L_eta_mode=sto_S218b`, not the PDEs. Manual mode remains available and unchanged.
