# Beta-only KWN versus PF consistency

Status: `P0_CONTRACT_CONFLICT`

## Result

The PF thermodynamic freeze remains BLOCKED_CONTRACT_CONFLICT between legacy runtime and exact-candidate local source. The requested same-PF thermodynamic/mobility beta-only run cannot select either value silently.

The repository retains 246-cube A/B registered PSD at 12 h, but it does not retain the requested full individual-radius Broad and Narrow 400-cube PSD locally. Scalar moments are not substituted for a PSD. The listed 246 A/B routes are retained as future proxy routes only.

## Required before execution

1. Resolve and hash-bind the PF thermodynamic contract.
2. Export full 12 h resolved-beta PSD for one BROAD/REF and one NARROW case from authoritative checkpoints.
3. Re-run with nucleation and GP off, PF-consistent diffusivity, and no fitted `D_scale`.

The blocked state is not reported as a failed KWN numerical test; it is an authority/provenance gate.
