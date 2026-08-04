# Resolved-interface scattering literature contract

## Scope and identity

This contract concerns an **additional resolved PbTe/Ag2Te phase-interface channel**. It does not replace or refit the frozen Yu S7--S10 density-contrast precipitate cross section already present in M0, and it is not a dislocation model.

The local Sheskin/practical precipitate model and Yu Supporting Information use the Kim--Majumdar Rayleigh/geometrical interpolation for spherical density-contrast scatterers ([Kim and Majumdar, 2006](https://doi.org/10.1063/1.2188251)). That channel is already the qualified full-PSD direct sum. The Sheskin derivation explicitly assumes no precipitate-induced elastic strain, so it cannot supply the missing interface/strain channel.

## Literature candidates

### I0: structureless AMM/DMM

The acoustic mismatch model assigns reflection/transmission from the acoustic impedances and refraction angles; the diffuse mismatch model assigns transmission from the modal densities of states. Hanus, Garg, and Snyder summarize both as structureless planar-interface models and note that their low-frequency transmissivity is frequency independent ([Communications Physics 1, 78 (2018)](https://doi.org/10.1038/s42005-018-0070-z)).

For this task I0 is retained only as literature context. A grey or frequency-independent rate proportional to `v*Sv` is not an admissible formal dynamic model under the preregistration. Moreover, the registered PbTe/Ag2Te data do not contain polarization-resolved Ag2Te velocities or an independently measured interface transmission spectrum. Status: `REJECT_FORMAL_DYNAMIC_MODEL_FREQUENCY_INDEPENDENT_AND_INPUT_INCOMPLETE`.

### I1: spectral-transmissivity interface model

Hanus et al. report the experimentally motivated spectral form

\[
t(\omega)=\left(1+\alpha\,\omega/\omega_{\max}\right)^{-1},
\]

where \(\alpha\) is dimensionless and of order unity. They also give the Dames--Chen lifetime/transmissivity relation

\[
t(\omega)=\frac{v_g n_I\tau_I(\omega)}{3/4+v_g n_I\tau_I(\omega)}.
\]

For statistically isotropic closed interfaces, stereology gives the mean line-intersection density \(n_I=S_v/2\). Inverting the preceding equation therefore gives the preregistered PF-interface rate

\[
\boxed{
\tau_I^{-1}(\omega,t)=
\frac{2}{3}v_g S_v(t)\,\alpha\frac{\omega}{\omega_{\max}}
}
\]

and no unsourced `tau^-1 proportional to Sv` assumption has been introduced: the \(S_v\) dependence follows from the sourced transmissivity/lifetime relation plus the stereological mapping, while the required frequency dependence is retained.

Dimensional closure is

\[
[v_g S_v]=({\rm m\,s^{-1}})({\rm m^{-1}})={\rm s^{-1}},
\qquad [\alpha\omega/\omega_{\max}]=1.
\]

Required inputs are the PF true periodic marching-cubes \(S_v(t)\), the frozen Yu average sound velocity \(v_g=1770\ {\rm m\,s^{-1}}\), and \(\omega_{\max}=k_B\Theta_D/\hbar\) with the frozen \(\Theta_D=136\) K. The PF interface-normal and shape distributions are audited, but under the scalar isotropic Debye angular average they do not create another fitted parameter.

I1 is a semi-empirical extension from spectral grain/phase-interface transport to closed PbTe/Ag2Te interfaces, not an atomistic PbTe/Ag2Te transmission calculation. One common \(\alpha\) may be calibrated from the complete 6 h temperature curve only. Before calibration, the physicality interval is frozen as

\[
0.1\leq\alpha\leq10,
\]

the two-decade interval centered on the source statement “order unity.” It is common to A/B/C, temperature independent, and time independent. No 48 h datum may select or modify it. Status: `QUALIFIED_ONE_PARAMETER_SEMIEMPIRICAL_INTERFACE_CANDIDATE`.

## Applicability limits

- Elastic, scalar, isotropic Debye transport is assumed.
- The model resolves spectral loss but not mode conversion, coherent multiple scattering, or polarization-specific PbTe/Ag2Te transmission.
- `p(n)` and particle orientation are measured diagnostics; they cancel only because the formal transport approximation is isotropic.
- I1 must be rejected if the 6 h common fit requires \(\alpha\) outside the frozen interval.
- Yu S7--S10 remains in M0; I1 must not be relabeled as density contrast or used to refit `A_N`.

