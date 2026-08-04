# Frozen strain-model equation contract

Model S uses the accepted-field trace-strain spectrum and contains no calibrated dynamic coefficient:

\[
S_e(\mathbf q,t)=V|\widetilde{{\rm tr}\epsilon}(\mathbf q,t)|^2
=9S_{\epsilon_h}(\mathbf q,t),
\]

\[
\tau_S^{-1}(\omega,t)=
\frac{\gamma^2\omega^4}{4\pi^2v^3}
\int d\Omega\,S_e(\mathbf k-\mathbf k',t)(1-\cos\theta),
\]

or, after the frozen isotropic Debye angular reduction,

\[
\tau_S^{-1}(\omega,t)=
\frac{\gamma^2v}{4\pi}
\int_0^{\min(2\omega/v,\pi/\Delta x)}q^3S_e(q,t)\,dq.
\]

The implementation uses the exact piecewise-constant `q^3` bin integral, removes the zero mode, takes `Delta x=1 nm`, and sets unresolved power above `pi/Delta x` to zero. Inputs `gamma=1.96` and `v=1770 m/s` are frozen Yu values. The deviatoric spectrum remains a leverage diagnostic because no independently bounded shear Gruneisen coupling is available; it is not added with a fitted amplitude.

Assumptions are static weak Born scattering, one scalar acoustic speed, elastic scattering, and an isotropic radial spectrum. Yu S12/S13 is excluded because it is a dislocation-line model and PF does not predict dislocation density. No 6 h or 48 h strain coefficient is fitted.

