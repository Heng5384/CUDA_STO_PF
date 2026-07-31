# Model scope and limitations

1. This work reproduces Yu's two-state effective model for AQ and 48 h only.
2. The precipitate rate uses Yu's effective average radii and number densities. It does not integrate a full particle-size distribution.
3. No PF-derived `N_v(t)`, `S_v(t)`, `M_6(t)`, particle field, or transport output is read.
4. No Sheskin 6-48 h data are used.
5. The implementation does not predict a continuous 6-48 h conductivity trajectory.
6. Yu's fitted/effective parameters are not assumed transferable to Sheskin specimens.
7. `A_N` and other effective parameters are not interpreted as unique microscopic mechanisms.
8. The spherical precipitate cross sections do not represent explicit plate/lath orientation or anisotropy.
9. The precipitate formula includes density contrast only. It does not explicitly contain interface transmission, force-constant mismatch, acoustic impedance, or the metavalent/iono-covalent bonding contrast emphasized in the article narrative.
10. The implementation preserves Yu's single-relaxation-time Eq. (3). It is not expanded into a textbook two-term Callaway model.
11. Normal and Umklapp processes cannot be separated from the published contract.
12. The Figure 6c/6d "THz" axis appears to be angular frequency divided by `1e12`, not cycles per second. Outputs retain both definitions.
13. The full 48 h published-parameter calculation does not reproduce the published curve. It must not be used as a validated prediction until the dislocation contract is clarified.
14. No interface-bonding difference, full size distribution, anisotropic cross section, or unreported parameter is added to repair the residual.
