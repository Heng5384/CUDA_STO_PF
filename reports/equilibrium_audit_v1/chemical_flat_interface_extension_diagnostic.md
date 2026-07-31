# Isolated zero-mode slab diagnostic (T380, GP OFF)

This is a read-only diagnostic on the isolated fresh-raw-profile extension
source, not a modification of the clean committed production branch.  It was
run after the clean branch rejected the same raw-field handoff.  The purpose
is to determine whether a conserved two-interface slab gives the expected
growth-sign bracket around the analytic chemical root.

## Contract

- source root: `/home/zhiheng/tmp/codex_equilibrium_audit_profile_extension_20260730`
- extension binary SHA-256: `9290225d44e987b73b8afe61cc60bea691bbe3965a00f5cd8788a2d318b17a1a`
- slab generator SHA-256: `3dfe736a5c0e1a5c3d07c493345f783bc01e87bd5a46b26424b0cb10eb1d7f84`
- grid: `128^3`, `dx=1 nm`, periodic, two planar interfaces, slab width 32 nm
- temperature: 380 °C; `dt=1e-4`; 256 steps; physical endpoint `1.2683854 s`
- elasticity, GP, birth, release, beta nucleation: OFF
- zero-mode: `PF_CONSERVED_Y_ZERO_MODE_V1`, all five runs PASS
- initial profile: analytic tanh slab; no fitting or parameter writeback

The experimental variable is converted with `xAg=2*xB/(2+xB)`.  The analytic
flat root is `xB=0.004664951821454188`, `xAg=0.004654096254055399`.

## Growth-sign bracket

The diagnostic uses the change in slab precipitate fraction,
`Delta_vf = vf(1.2683854 s)-vf(0)`, as a short-time sign indicator.  A sign
is not interpreted as an equilibrium root when it is close to the interface
relaxation noise floor.

| case | xB | xAg | vf(0) | vf(end) | Delta_vf | Delta_vf / s |
|---|---:|---:|---:|---:|---:|---:|
| low-wide | 0.003004506760 | 0.003000000 | 0.25000000 | 0.249880755 | -1.19245e-4 | -9.399e-5 s^-1 |
| low experimental | 0.005816868920 | 0.005800000 | 0.25000000 | 0.250069084 | +6.9084e-5 | +5.446e-5 s^-1 |
| analytic root | 0.004664951821 | 0.004654096 | 0.25000000 | 0.250003172 | +3.172e-7 | +2.500e-7 s^-1 |
| high experimental | 0.006621852112 | 0.006600000 | 0.25000000 | 0.250111067 | +1.11067e-4 | +8.754e-5 s^-1 |
| high-wide | 0.008032128514 | 0.008000000 | 0.25000000 | 0.250181109 | +1.81109e-4 | +1.4279e-4 s^-1 |

The wide pair gives a clean sign change: the xAg=0.003 case shrinks while
xAg=0.008 grows.  The analytic root lies close to the near-zero short-time
drift.  The two experimental-band points are on the positive-growth side in
this short analytic-profile window, but the result is not a converged
equilibrium concentration because the initial tanh profile relaxes during the
same window.

All runs retained the initial total inventory through the zero-mode audit. The
largest reported final mean mass error was `1.75803815949393538e-13`, which
passes the isolated extension's conservation check but is not evidence for a
clean-branch production qualification.

## Decision

`ISOLATED_EXTENSION_ROUTE_B_QUALITATIVE_BRACKET`: the fresh-raw extension
recovers the expected sign bracket and is qualitatively consistent with the
analytic root.  `BLOCKED_CLEAN_BRANCH_PROFILE_HANDOFF` remains: the clean
committed binary still rejects raw/VTK profile epochs under the frozen
zero-mode provenance contract.  A converged publication-level planar root
would require a separately reviewed zero-mode-compatible slab/profile
materialization path and a longer relaxation-separated bracket.

Raw-field, checkpoint and stdout hashes are preserved in the workstation root
under `slab_bracket_*_20260730`; the detailed per-step scalar traces are the
`vf_precip_vs_time_raw_fields_relax.csv` files in those case directories.

For reproducibility, the five xB raw-field hashes are:

```text
xAg=0.003000000  2afffc3db56a14ec4f2df89bb18e578be78b107b8db2fda18b3ee6063f9a2c16
xAg=0.005800000  903509efbb12bc2cd13b8de4e02782d5d85ad01672f324d7bd79be4acc950a0f
xAg=0.004654096  1a2bb9feaf57f9371e890396789d5f838ea6fc99cdf5e9513766c0c3a3a8b84b
xAg=0.006600000  dd141d54cd442a21069a5a4ceaf3675872992608d2bffe80d443dc868faf19c0
xAg=0.008000000  32baf4ea0a353c3e0850004e579f3467942a352d014f6dc14f90ca6c0dac0aa9
```
