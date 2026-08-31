# KWN–PF MVP v1 environment inventory

Status: `INTERNAL_KWN_BACKEND_SELECTED`; PF-coupled production status remains
`P0_CONTRACT_CONFLICT` / `PARTIAL_PF_STATE_NOT_CLOSED`.

## Isolated delivery context

| item | observed value | consequence |
|---|---|---|
| worktree | `/Users/heng/Documents/GitHub/CUDA_STO_PF-kwn-pf-mvp-v1` | all MVP changes are isolated from the user's dirty production worktree |
| branch | `codex/kwn-pf-mvp-v1` | delivery branch only |
| base commit | `2f865360157ab877b8c811f03c92e667d9a1d366` | recorded before MVP edits |
| Python | 3.9.6 | the implementation uses the standard library, NumPy, and `unittest` |
| NumPy | 2.0.2 | available |
| matplotlib | available | used only for generated diagnostic figures when available |
| PyYAML | unavailable | KWN configuration files are JSON-subset YAML and use the in-repository deterministic parser |
| pytest | unavailable | qualification uses `python3 -m unittest` |
| Kawin | unavailable; no import, lockfile, or pinned version found | no Kawin result is represented as produced by this repository |

## Inputs and retained constraints

| source | role in MVP | handling |
|---|---|---|
| Sheskin et al. (2018) | primary 380 °C matrix-composition/GP timing constraints | used as the only quantitative primary target envelope |
| Yu et al. (2024) | secondary GP plausibility envelope | holdout-only; never jointly fitted |
| Grossfeld et al. (2017) | context | retained as non-fit context because its state/temperature mapping is not established |
| `data/kwn_constraints/*.yaml` and `experimental_constraints.csv` | machine-readable provenance | source, role, units, and warnings retained beside the KWN configurations |

## PF readiness inventory

The existing authoritative PF evidence contains resolved beta/matrix state and
registered beta PSDs, but does not supply a persistent subgrid-beta state that
can represent an arbitrary KWN beta population.  Current GP storage is a
legacy eta-based surrogate (`gp_xB_fixed`), and it is disabled by the qualified
production gates.  Therefore a four-bucket KWN ledger cannot be silently
collapsed into the PF matrix field.

The smallest existing conditional PF handoff fixture is the 96³, six-particle
library-assembled fixture in
`data/qualification/pf_mass_conserving_library_handoff_v1/six_particle_96cube_spec.json`.
It is `validation_only`; it is not an authorization to start a production PF
run.

## Decision

The MVP implements an internal finite-volume KWN backend and a versioned
handoff package.  It does not claim an executable KWN-to-current-PF production
coupling until the thermodynamic contract is frozen and the PF state can carry
all four ledger buckets without reassignment.
