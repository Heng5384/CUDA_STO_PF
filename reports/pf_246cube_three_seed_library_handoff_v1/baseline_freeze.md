# Baseline freeze

This qualification preserves the PF-only post-nucleation scope in
`PROJECT_CORE_MEMORY.md`. It constructs three conditional 6 h handoff states
and does not claim to reproduce an experimental 6 h microstructure.

## Repository identity

- branch: `codex/pf-dynamic-microstructure-audit-v1`
- HEAD: `5bb0db2bb3321b70cb5f254ffd9ab738a3b3611d`
- worktree: dirty before this goal; pre-existing user and historical changes
  were preserved
- frozen runtime/source identity-list SHA-256:
  `19dcb258fc5428bd43676aa5455c9306e27885e4d1b236ddb45d34d21039bf0d`
- workstation binary SHA-256:
  `7581c169fb1d1c16ee60764f418ee9c33508682c8b9dd8cf76b89368c7990602`
- cluster `sm_80` binary SHA-256:
  `efb99c707acf7f22899425b8742c7dc06bb3231b9f21604cb5d75cf4565d8c94`
- production parameter SHA-256:
  `ecbdd0ac070bdf5e5d214322b5248a08f5ca5dd4e0670427513f5ef977ea977a`

## Frozen input identities

- historical PSD file:
  `f03ca43771200490cc0ff83646a86c8041fc65427b44ec2c3e71772f5b1d3f50`
- historical canonical manifest:
  `8ce52733e6755b23f4f7d4ba53f0414ef927bb863c3eb4ce9d43cb09d4d5de45`
- profile-library manifest:
  `58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe`
- library selection provenance:
  `56c44d8f72b27bb462dffe89b59cb2fcb2ff0bf8807dec9d9cac31d2bd7fcbe3`
- frozen library source tree:
  `f7855699addf98f9d5aed03af876d851d524fb62c98f5a48df78d1fe561a2a75`
- frozen library binary:
  `55cf917df94fcf01373d62f54f9ab99715975863ad5dcfb95baa8a517460cda1`

## Physical and numerical contract

`T=380 °C`, `246³`, `dx=1 nm`, `lambda_sm=4 nm`, fixed periodic
cell, identity/[100] orientation, eigenstrain
`(0.046,-0.022,-0.017,0,0,0)`, zero external strain/stress, and
`target_mean_C_Btot=0.03`.

The production timestep is `dt_code=0.02`, corresponding to
`0.9909260953431841 s` per macrostep. The selected fallback is
`dt_code=0.01`.

GP, GP Birth, GP release, external sources, and new beta nucleation are all
disabled. No multi-particle minimizer or common-equilibrium claim is used.
