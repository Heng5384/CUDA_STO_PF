# 10 Model limitations

- KWN is mean-field and spherical-equivalent; CUDA PF is spatial, elastic and diffuse-interface. A numerical agreement claim requires the gated trajectory comparison, which was not run.
- The 96³ calculation is a validation-only smoke, not a 400³ production or ensemble result.
- GP release, GP→beta conversion, beta birth, online KWN coupling, dislocation physics, matrix global reset and composition clamping remain disabled.
- No thermodynamic, diffusivity, mobility, gamma, lambda, elasticity, eigenstrain, fixture profile or physical initial PSD retuning was used to cure the KWN failure.
- The strict remaining KWN limitation is radius/size-space grid convergence, not a timestep, conservation or positivity-clamp issue.
