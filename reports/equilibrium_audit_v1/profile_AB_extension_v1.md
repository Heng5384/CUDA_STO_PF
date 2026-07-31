# Isolated profile A/B/C extension (uncommitted source)

This report records a separate, non-production extension run.  It is not
evidence from the clean committed binary.  The extension copied the clean
branch at commit `6b69895af2d1b86b99c57c5479ff767349c61efe` and used only the
uncommitted fresh-raw-profile/zero-mode acceptance change.  The source and
binary hashes were respectively
`9addd9272e7826b52089a2f8b6f3768e7d615ff1a5f1d239d3177b2e6b787053` and
`9290225d44e987b73b8afe61cc60bea691bbe3965a00f5cd8788a2d318b17a1a`.

## Scope

The isolated workstation root was
`/home/zhiheng/tmp/codex_equilibrium_audit_profile_extension_20260730`.
All cases were 128^3, R=8 nm, T=380 °C, `dt=1e-4`, 128 steps, GP/birth/
release/nucleation OFF, and used `PF_CONSERVED_Y_ZERO_MODE_V1`.  Profile A is
the analytic tanh raw field; profile B is the locally minimized raw field;
profile C is the same B field with fixed-cell elasticity enabled.

| case | final `<h(phi)>` | final `<xB_tot>` | zero-mode mean mass error | wall time | provenance |
|---|---:|---:|---:|---:|---|
| A, elastic OFF | 1.05410630e-3 | 5.71119572e-3 | 5.20417043e-17 | 15.418 s | FRESH_RAW_FIELDS |
| B, local, elastic OFF | 1.01668881e-3 | 5.68294011e-3 | 4.16333634e-17 | 15.435 s | FRESH_RAW_FIELDS |
| C, local, elastic ON | 1.01270962e-3 | 5.68908474e-3 | 5.89805982e-17 | 40.145 s | FRESH_RAW_FIELDS |

The A/B mean absolute phi difference was `5.46993160e-5`, normalized by
`<h>_A` as `5.18916509e-2`; the B/C difference was `4.21144485e-6`, or
`4.14231456e-3` normalized.  These are a one-particle profile-method
diagnostic, not the required multi-particle V2 result.

## Restart evidence

The B and C 64-step checkpoint/restart pairs both passed
`RESTORED_AND_VALIDATED`.  B continuous/restart final raw hashes were:

```
phi   feacc98407538ceb30e8a6737368e691ff0cdb468e258432ba4b98795e090fdb
xB    4feab3e595ca952a7d7f75f3dd119e67114d391547f47721f32c8453bb7d201b
xBtot cb339afd57ef1871429b3c8b923eb750486c26eabb1142c90a49f147fffb0acd
```

C continuous/restart final raw hashes were:

```
phi   cdbdab9b7d2d49cca13cca00cddca40709f6c2489d26cea700e16ac1e9fc9f29
xB    4faf00869a60ca04ab8f4abb7b922a84f77793f3b5467a27be0a757e353be5dc
xBtot 356244f73be615904f3eabd7c42ae4807377d122c2873477ec7d4f257f4ed033
```

The extension therefore demonstrates that a separately materialized profile
can be run and restarted under the extension contract.  The corrected
three-particle V2 A/B/C extension subsequently closed the validation-only
multi-particle gate; its results are in
`multi_particle_v2_results_v1.md` and
`multi_particle_v2_restart_validation.md`.  Neither extension removes the
clean-branch method blocker.
