# 6–8 h particle shape-tensor audit

`PASS_PARTICLE_SHAPE_TENSOR_6_8H_V1`

Each particle uses periodic unwrapping and the diffuse `h(phi)` field as its
volume weight.  The shape covariance has units nm2.  The physical geometric
inertia tensor is `integral h(r) [r^2 I - r r] dV` and has units nm5.  Effective
ellipsoid semi-axes use `a_i=sqrt(5 lambda_i)`.

The major axis is an unoriented line: its sign is canonicalized only for stable
serialization.  Angles use absolute direction cosines relative to the box
axes, interpreted as [100], [010], [001].  Orientations with a major/intermediate
eigenvalue gap below 5% are marked ill-conditioned and should not be used as
crystallographic evidence.
