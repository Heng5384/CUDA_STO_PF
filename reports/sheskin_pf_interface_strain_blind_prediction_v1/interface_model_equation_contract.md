# Frozen interface-model equation contract

Model I is the spectral-transmissivity candidate registered in `interface_scattering_literature_contract.md`:

\[
t(\omega)=\left(1+\alpha\omega/\omega_D\right)^{-1},
\qquad
\tau_I^{-1}(\omega,t)=\frac{2}{3}vS_v(t)\alpha\frac{\omega}{\omega_D}.
\]

The factor `2/3` follows by combining the Dames--Chen lifetime/transmissivity relation with the isotropic stereological intersection density `n_I=Sv/2`. Inputs are true periodic marching-cubes `Sv`, `v=1770 m/s`, and `omega_D=k_B*136 K/hbar`.

Only one dimensionless coefficient remains. It is common to A/B/C, all temperatures, and all times. The preregistered interval is `0.1 <= alpha <= 10`. It is calibrated by weighted least squares to the seven-point 6 h measured-total curve using the A/B/C ensemble mean, separately propagated through each of the 18 retained AQ-background envelope members. No 48 h value enters calibration.

Frozen outcome: 16 background members give an interior value; two conservative 50 nm-sphere diagnostic bounds reach the upper limit and are rejected for parameter physicality. The coefficient is not refitted for 48 h. Model MIS uses the already frozen MI coefficient plus the parameter-free strain rate; it is not jointly refitted.

This is a semi-empirical effective Debye reconstruction, not a polarization-resolved atomistic PbTe/Ag2Te transmission calculation and not an absolute experimental lattice-conductivity reproduction.

