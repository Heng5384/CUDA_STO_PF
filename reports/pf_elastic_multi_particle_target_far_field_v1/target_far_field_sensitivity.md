# Multi-particle target-far-field sensitivity

## Decision

```text
static_rebase_status=PASS_PF_ELASTIC_MULTI_PARTICLE_TARGET_FAR_FIELD_REBASE_V1
restart_status=PASS_BYTEWISE
mass_status=PASS
zero_mode_status=PASS
particle_identity_status=PASS_6_OF_6
dt_refinement_status=FAIL_XB_MAE_1.1170213602913285E-4_GT_5E-5
final_status=BLOCKED_MULTI_PARTICLE_COMMON_EQUILIBRIUM_NOT_RESTORED_BY_FAR_FIELD_REBASE
```

Pinning the target matrix far field and deriving the compatible global
inventory does not repair the multi-particle timestep-refinement failure.
The failure is therefore not caused by forcing the source fixture's global
mean canonical inventory to exactly 0.03.

## Scope

This was a validation-only 96³, T380, \(dx=1\) nm,
\(\lambda_{\rm sm}=4\) nm six-particle test. It reused only the exact
registered 8.0, 9.5 and 10.5 nm identity-orientation entries from the
qualified single-particle elastic target-profile library. GP, GP Birth, GP
release, external source and new beta nucleation remained disabled.

No production run, physical parameter retuning, cluster use, commit or push
occurred.

## Construction

The source V2 fixture already contained the deterministic bounded phase union
and phase-consistent portable local matrix correction:

\[
\phi_{\rm total}=1-\prod_j(1-\phi_j),
\qquad
\delta x_\alpha=\sum_j
\left(x^\alpha_{B,j}-x^\alpha_{B,\mathrm{far},j}\right).
\]

The sensitivity fixture retained `phi` and `delta_x_alpha_total` bytewise and
changed only the target-box baseline:

\[
x_B^\alpha=x_{B,\mathrm{matrix}}^{\mathrm{target}}+\delta x_\alpha,
\]

\[
C_B=h(\phi_{\rm total})+
\left(1-h(\phi_{\rm total})\right)x_B^\alpha.
\]

The pinned target was

\[
x_{B,\mathrm{matrix}}^{\mathrm{target}}
=0.006219279767278563,
\qquad
x_{\mathrm{Ag}}^{\mathrm{target}}=0.0062.
\]

The global inventory was not independently forced. It was derived as

\[
\left\langle C_B\right\rangle
=0.029996585491183673,
\]

which is \(3.4145088163\times10^{-6}\) below the source fixture's 0.03.
No clipping, scaling, normalization, profile interpolation or rotation was
used.

## Static audit

- six connected beta components were retained;
- `phi` and `delta_x_alpha_total` were byte-identical to the source V2
  fixture;
- \(x_B^\alpha\) remained in
  `[0.006013405410497348, 0.006363369045271839]`;
- the initial broad far-matrix mean at `h < 1e-4` was
  `0.006217993515080654`;
- all generated fields were finite and hash-pinned;
- the canonical inventory closed by construction.

The local V2 regression suite also passed 19/19 tests.

## Dynamic qualification

The ordinary selected conserved elastic PF path was run for:

- one-step handoff;
- continuous 64 steps at `dt_code=0.02`;
- 32 + checkpoint/restart + 32 steps at `dt_code=0.02`;
- 128 steps at `dt_code=0.01`;
- identical physical endpoint for timestep comparison.

| gate or metric | result |
|---|---:|
| continuous/restart checkpoint | byte-identical |
| component identity | PASS, 6/6 |
| one-step overlap | PASS |
| zero mode | PASS |
| mass relative error | \(2.716\times10^{-13}\) |
| dt \(\phi\) L1 | \(8.2746\times10^{-4}\), PASS |
| dt maximum axis relative difference | \(3.9308\times10^{-4}\), PASS |
| dt full-field \(x_B\) MAE | \(1.1170214\times10^{-4}\), FAIL |
| dt far-matrix \(x_B\) MAE | \(1.1524582\times10^{-4}\), FAIL |

Every stderr file was empty.

## Comparison with the fixed-global-inventory V2 source

| metric | source V2, mean \(C_B=0.03\) | target-far-field fixture | change |
|---|---:|---:|---:|
| dt full-field \(x_B\) MAE | \(1.1145355\times10^{-4}\) | \(1.1170214\times10^{-4}\) | +0.223% |
| dt far-matrix \(x_B\) MAE | \(1.1499033\times10^{-4}\) | \(1.1524582\times10^{-4}\) | +0.222% |
| dt \(\phi\) L1 | \(8.2632017\times10^{-4}\) | \(8.2745613\times10^{-4}\) | +0.137% |

The response is effectively unchanged. The small inventory/baseline
incompatibility is not the cause of the failed dynamic handoff.

## Interpretation

Each source entry is an isolated-particle constrained elastic equilibrium.
After several entries are translated into one periodic fixed-cell box, their
elastic fields interact nonlocally. A uniform matrix-baseline rebase preserves
local profile corrections and exact mass, but it cannot make the assembled
multi-particle field a stationary solution of the coupled multi-particle
elastic/composition problem.

The remaining problem is therefore a common constrained equilibrium problem,
not a far-field interpolation problem. The source V5e explicit
projection/minimization route also did not solve it: it preserved mass and
per-particle volumes but reached 24,000 iterations with
`rms_res=1.3420e-3` and did not satisfy the registered KKT/field/energy
convergence contract.

## Evidence identity

```text
source_fixture_manifest_sha256=bbac0b9ce0521fb525f76d7bea90da1fc6352cb5bac62cda86a06d27d5e54093
target_far_field_fixture_manifest_sha256=3df27d489abeb808f3e72e75d4910ee53a7f490d702af9f1ed17f5d20d03f82b
target_far_field_contract_sha256=0177d7f4de96e34508e71e623e9bf0be913f929ac7dc056013b83e88399c5fe0
dynamic_audit_sha256=80973341680ba3d1b62c39f202f025d25dbf0f1a2015d57e859206f3bdf67a93
continuous_checkpoint_sha256=e249fbfe3846f3cb5cb1b6e67adedb064d44717f679e5ce67aa7e64d6c089672
restart_checkpoint_sha256=e249fbfe3846f3cb5cb1b6e67adedb064d44717f679e5ce67aa7e64d6c089672
runtime_binary_sha256=87071d148f0579724046145a44bd00ea39d92c88346669af1741d558285fd811
dynamic_analyzer_sha256=1604a4d4aee83639704be68949b70f03615a5d512308817e60e60d7cadc72ca6
```

Workstation evidence roots:

```text
/home/zhiheng/tmp/pf_elastic_multi_particle_E2_target_far_field_v1_20260731
/home/zhiheng/tmp/pf_elastic_multi_particle_E2_target_far_field_dynamic_v1_20260731
```

## Recommended next numerical action

Do not continue tuning the uniform far-field baseline. If a zero-transient
elastic multi-particle start remains mandatory, replace the explicit repeated
volume projection with a genuinely convergent constrained optimizer, such as
an augmented-Lagrangian or primal-dual solve with one volume multiplier per
particle, while retaining the frozen physics, exact inventory and fail-closed
topology contract. Qualify that optimizer first on the same six-particle
validation fixture before applying it to an experimental 6 h PSD.
