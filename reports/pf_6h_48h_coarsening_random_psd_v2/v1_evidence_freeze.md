# V1 evidence freeze

V1 source provenance is the exact user-selected commit
`6b69895af2d1b86b99c57c5479ff767349c61efe` on
`codex/pf-zero-mode-restart-provenance-v1`. The V2 worktree is a new detached
worktree from that commit; V1 output and binary files are not used as V2
runtime state.

Frozen V1 pilot evidence:

- source commit: `6b69895af2d1b86b99c57c5479ff767349c61efe`;
- V1 fixture field hashes: phi `d934edb85036caf878b848980577945b399756ed65ca852b8ac05b2276ef5e76`,
  xB `6055b55f8c9b33a2a944626675e0bca3c6ff86849078e5bb4c8a5a7ba8147399`,
  meta `4f49683b8cac273bb43a9b73163e1c1f22900b6d729193d6ac201ddcdecde80a`;
- V1 fast-resume checkpoint count: 42;
- ordered checkpoint-list digest:
  `b54a1ec11351329d69e20af1c137aceccafd8cd5dc17d6c351af85ff2f3d1e46`;
- first checkpoint SHA-256:
  `576cd2715faa61e1ebb48a8022edbd407a9c9204d560f8c67fdafe81953b7e60`;
- 12 h restart checkpoint SHA-256:
  `4df5e73df52032515492e58fe327c501fbd9fd9ab2ddb57fe18b4c5f2e8c3e02`;
- final checkpoint SHA-256:
  `c0e8b53b0762df4793b817dff3b9f500efba97a091e5d4801d40655e8712b4ad`.

V1 reports, fixture manifest, source hash, converter hash, and the T400
qualification reports were preserved in the original worktree and are
referenced by `v1_evidence_manifest.json`.
