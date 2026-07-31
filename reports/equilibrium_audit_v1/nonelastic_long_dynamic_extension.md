# Nonelastic R=8 nm long dynamic extension

An additional clean-branch workstation run was completed for the low
composition endpoint `xAg=0.0058` (`xB_far=0.005816868886`): 128^3, R=8 nm,
GP/birth/release/β-nucleation OFF, elasticity OFF, `dt=1e-4`, 8192 steps,
with zero-mode checkpoints at steps 4096 and 8192.

## Result

| accepted step | physical time (s) | vf_precip | R_avg (nm) |
|---:|---:|---:|---:|
| 0 | 0 | 1.060517e-3 | 7.851898 |
| 4096 | 20.29417 | 1.051352e-3 | 7.837267 |
| 8192 | 40.58833 | 1.044157e-3 | 7.837267 |

The post-checkpoint volume-fraction slope is approximately
`-3.54e-7 s^-1`.  The run therefore remains on the negative side of the
short bracket, but it is still a finite dynamic window, not an equilibrium
root: the radius output is discretely rounded and the interface transient is
not demonstrably exhausted.  The result strengthens the qualitative R=8
low-end sign observation; it does not replace the required long-time
composition bracket or claim a dynamic critical radius.

The zero-mode audit passed at the final step with mean mass error
`1.73472347597680709e-18`; the only stderr entries were the pre-existing
unknown legacy radius keys and the interface-resolution advisory.  Average
step wall time was `0.039052 s` and the zero-mode solver consumed
`26.267343 s` over the run.

## Provenance

- source branch: `codex/pf-zero-mode-restart-provenance-v1`
- commit: `6b69895af2d1b86b99c57c5479ff767349c61efe`
- binary SHA-256: `11d073a272a0b7fa668e40037bc1f96b8389909f95c74d6ad9a5db11236e01af`
- output root: `/home/zhiheng/tmp/codex_equilibrium_audit_v1_branch_20260729/long_R8_8192_low_20260729`
- checkpoint: `long.chk` (65 MiB, final accepted step 8192)
