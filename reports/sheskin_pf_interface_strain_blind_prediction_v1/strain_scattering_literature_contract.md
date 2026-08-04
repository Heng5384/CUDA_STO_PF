# Coherent-strain scattering literature contract

## Rejected shortcuts and dislocation identity

Yu Supporting Information S12/S13 is a line-dislocation strain model containing dislocation density, Burgers vector, Poisson ratio, and a fitted prefactor. It is not a formula for a resolved coherent precipitate strain field. PF does not predict dislocation density, and the Yu dislocation scale is forbidden here. S12/S13 is therefore not used.

Rates directly proportional to mean elastic energy, total strain variance, or a selected q-band are diagnostic proxies only and are not formal models.

## S1: accepted-field strain-spectrum Born model

Klemens established perturbative phonon scattering by static imperfections ([Proc. Phys. Soc. A 68, 1113 (1955)](https://doi.org/10.1088/0370-1298/68/12/303)). Hanus, Garg, and Snyder write the elastic-defect matrix element as the Fourier transform of the spatial perturbation and explicitly use a strain perturbation of the form \(\widetilde V=\hbar\omega\gamma\widetilde\epsilon\), with transport weighting \(1-\hat{\mathbf k}\cdot\hat{\mathbf k}'\) ([Communications Physics 1, 78 (2018)](https://doi.org/10.1038/s42005-018-0070-z)). Interfacial strain fields between dissimilar lattices have also been treated as a phonon-scattering source by Meng, Wu, and Zhu ([Physical Review B 87, 064102 (2013)](https://doi.org/10.1103/PhysRevB.87.064102)).

For the periodic PF volume \(V\), define the volume-normalized transform and its correlation spectral density by

\[
\widetilde e(\mathbf q)=\frac{1}{V}\int_V e(\mathbf r)e^{-i\mathbf q\cdot\mathbf r}\,d^3r,
\qquad
S_e(\mathbf q)=V|\widetilde e(\mathbf q)|^2.
\]

The scalar dilation is \(e={\rm tr}(\epsilon)=3\epsilon_h\), so the measured hydrostatic spectrum obeys \(S_e=9S_{\epsilon_h}\). With \(\delta\omega/\omega=-\gamma e\), Fermi's golden rule in a one-speed isotropic Debye solid gives

\[
\boxed{
\tau_S^{-1}(\omega,t)=
\frac{\gamma^2\omega^4}{4\pi^2v^3}
\int d\Omega_{\mathbf k'}\,
S_e(\mathbf k-\mathbf k',t)
(1-\cos\theta)
}
\]

with elastic \(|\mathbf k'|=|\mathbf k|=k=\omega/v\) and

\[
q=|\mathbf k-\mathbf k'|=2k\sin(\theta/2).
\]

For the registered isotropic radial PF spectrum this reduces exactly to

\[
\boxed{
\tau_S^{-1}(\omega,t)=
\frac{\gamma^2v}{4\pi}
\int_0^{\min(2\omega/v,q_{\rm PF,max})}q^3S_e(q,t)\,dq
}
\]

where the PF-resolved cutoff is frozen at \(q_{\rm PF,max}=\pi/\Delta x\), \(\Delta x=1\) nm. The zero mode is removed. Spectrum above this cutoff is set to zero because it is not represented by the continuum PF field; it is not extrapolated or fitted.

Dimensional closure is

\[
[S_e]={\rm m^3},\quad
[q^3S_e\,dq]={\rm m^{-1}},\quad
[v\int q^3S_e\,dq]={\rm s^{-1}}.
\]

The required inputs are the accepted-field synchronized PF strain spectrum, \(v=1770\ {\rm m\,s^{-1}}\), and the independently frozen Yu Gruneisen parameter \(\gamma=1.96\). The model contains **no fitted dynamic prefactor**. Deviatoric strain spectra remain required diagnostics, but the scalar Gruneisen coupling does not provide an independently bounded shear coupling; they are not silently added to S1.

Status: `QUALIFIED_PARAMETER_FREE_SCALAR_BORN_STRAIN_CANDIDATE`.

## Assumptions and failure conditions

- Static, weak-scattering Born approximation and elastic scattering.
- One acoustic speed and scalar Gruneisen coupling; polarization conversion and anisotropic phonon dispersion are omitted.
- The PF field is a continuum low-pass representation; atomistic interface-core strain is outside its q support.
- A/B/C are independent finite-volume realizations and are propagated separately before ensemble statistics.
- If replay is not accepted-field synchronized, the model is blocked.
- If the rate depends materially on an invented high-q continuation or an unbounded shear prefactor, the model is blocked rather than fitted.

