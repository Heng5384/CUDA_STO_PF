# PF400 periodic true-area validation

Final status: `PASS_PF400_PERIODIC_TRUE_AREA_V1`

The qualified object contract is periodic six-neighbour connectivity on
`h(phi)>1e-4`.  The interface contract is object-local periodic unwrapping and
watertight marching cubes at `phi=0.50`; `phi=0.45` and `0.55` are sensitivity
levels only.  Connected objects are not promoted to independent physical
particles, and a low-support object containing multiple `phi>=0.5` cores is
flagged as necked/merged.

The suite covers an interior sphere; spheres crossing one, two and three
periodic seams; two separated spheres; a necked pair; an ellipsoid; an
analytic sphere; four mesh resolutions; three level sets; and integer periodic
translation.  Seam-crossing areas agree with the interior representation to
`2.260e-09`.
The finest-grid analytic-sphere area error is
`0.081%` and the coarse-grid error is
`0.882%`.  Refinement acceptance is
`True`; integer-origin invariance is `2.260e-09`.

Mesh-volume versus diffuse `h`-volume is reported per object in production;
it is a representation comparison rather than an equality constraint because
the former encloses `phi=0.5` while the latter integrates the interpolation
function over the diffuse interface.
