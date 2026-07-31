# Yu 2024 equation contract

## Published conductivity integral

Main-text Eq. (3) is

\[
\kappa_{\mathrm{lat}}(T)=
\frac{k_B}{2\pi^2 v}
\left(\frac{k_B T}{\hbar}\right)^3
\int_0^{\Theta_D/T}
\tau_{\mathrm{tot}}(x,T)
\frac{x^4 e^x}{(e^x-1)^2}\,dx,
\qquad
x=\frac{\hbar\omega}{k_B T}.
\]

Thus

\[
\omega=\frac{x k_B T}{\hbar},
\qquad
0\le x\le\Theta_D/T,
\qquad
\omega_D=\frac{k_B\Theta_D}{\hbar}.
\]

The velocity is the SI Table S2 average sound velocity, `v = 1770 m/s`. The Debye temperature is `136 K`.

This is the only conductivity term published by Yu. It is a single-relaxation-time Debye integral. A hydrodynamic Callaway second term is absent, and there is no published first/second-term split to implement.

## Normal plus Umklapp process

SI Eq. (S1), checked visually on SI PDF page 14, is

\[
\tau_U^{-1}+\tau_N^{-1}
=
A_N\frac{2}{(6\pi^2)^{1/3}}
\frac{k_B\bar V^{1/3}\gamma^2\omega^2T}
{\bar M v^3}.
\]

`A_N = 1.5` is fitted by Yu. The source supplies only the combined rate. The implementation registers it as `phonon_phonon`; it never invents separate Normal and Umklapp rates.

## Resistive and defect rates

Main-text Eq. (4), expanded with the SI definitions, is implemented as

\[
\tau_{\mathrm{tot}}^{-1}
=
(\tau_U^{-1}+\tau_N^{-1})
+\tau_{GB}^{-1}
+\tau_{PD}^{-1}
+\tau_{Pre,s}^{-1}
+\tau_{Pre,b}^{-1}
+\tau_{DC}^{-1}
+\tau_{DS}^{-1}.
\]

All rates are combined by Matthiessen's rule.

### Grain boundary, SI Eq. S2

\[
\tau_{GB}^{-1}=v/d.
\]

### Point defects, SI Eqs. S3-S5

\[
\tau_{PD}^{-1}
=\frac{\bar V\omega^4}{4\pi v^3}\Gamma,
\qquad
\Gamma_i=x_i\left[
\left(\frac{\Delta M_i}{M}\right)^2+
\epsilon\left(\frac{\Delta\delta}{\delta}\right)^2
\right],
\qquad
\Gamma=\sum_i\Gamma_i.
\]

Yu retains only interstitial Ag and fixes `epsilon = 65` from the cited literature rather than evaluating SI Eq. S6. The calculation uses the literal Table S2 `Delta M_i = 107.87 g/mol`, `M = 207.2 g/mol`, and `(delta_i-delta)/delta = (160-180)/180`. The label of `Delta M_i` is source-ambiguous and is not silently corrected.

### Precipitates, SI Eqs. S7-S10

\[
\tau_{Pre}^{-1}
=vN_P\left(\sigma_S^{-1}+\sigma_l^{-1}\right)^{-1},
\]

\[
\sigma_S=2\pi R^2,
\qquad
\sigma_l=
\frac{4}{9}\pi R^2
\left(\frac{\Delta D}{D_M}\right)^2
\left(\frac{\omega R}{v}\right)^4,
\qquad
\Delta D=D_{\mathrm{PbTe}}-D_{\mathrm{Ag_2Te}}.
\]

The SI explicitly defines `R` as average radius and `N_P` as number density. `v` is the same average sound velocity used in Eq. (3). Small and big populations are evaluated separately and their rates are added. The same equations are used for AQ and 48 h; only Table S2 state parameters differ.

### Dislocation core, SI Eq. S11

\[
\tau_{DC}^{-1}
=N_D\frac{\bar V^{4/3}}{v^2}\omega^3.
\]

### Ag-decorated dislocation strain field, SI Eqs. S13-S16

\[
\tau_{DS}^{-1}
=CB_D^2N_D(\gamma+\gamma')^2\omega
\left[
\frac12+
\frac1{24}
\left(\frac{1-2r}{1-r}\right)^2
\left(1+\sqrt{2}\left(\frac{v_L}{v_T}\right)^2\right)^2
\right].
\]

\[
\gamma'=
\frac{V_m c_i B}{k_BT_a}
\left(\gamma\alpha^2-\alpha\beta\right),
\quad
\alpha=\frac{V_i-V_m}{V_m},
\quad
\beta=\frac12\frac{M_m-M_i}{M_m}.
\]

The production calculation uses the directly published `gamma_prime = 2.62`. An independent unit check from Eqs. S14-S16 gives `2.61778969265`, a relative difference of `0.0844%`.

## Numerical integration

- Production integral: deterministic 256-point Gauss-Legendre quadrature on `[0, Theta_D/T]`.
- Grid refinement: 512-point Gauss-Legendre.
- Independent integrator: adaptive SciPy quadrature with `epsabs = 1e-11`, `epsrel = 1e-10`, and `limit = 300`.
- Maximum 256/512 relative difference over both states and all Table S1 temperatures: `6.62e-14`.
- Maximum Gauss/adaptive relative difference: `6.81e-14`.

No fitted constant is hidden in the implementation. All scientific values are read from the two frozen JSON files.
