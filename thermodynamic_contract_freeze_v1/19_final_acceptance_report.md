# Final acceptance report

## Final status

```text
BLOCKED_AUTHORITATIVE_CONTRACT_CONFLICT
```

## Acceptance answers

1. Frozen thermodynamics: none; exact contract is not frozen. The legacy contract is documented but not promoted to publication authority.
2. Authoritative source: exact candidate evidence is now present in `/Users/heng/Desktop/1_Solubility_Calibration`; existing production source remains legacy `thermo_utils.h` / `Unit_Psedobinary.py`.
3. Exact candidate parameters: `DeltaH=41504.29119633958 J/mol`, `DeltaS=18.469276826409214 J/(mol K)`; candidate is reproducible but not deployed or publication-frozen.
4. Exact candidate 380 °C root: `T=653.15 K`, `xB=0.004649261005504821`, `xAg=0.004638478257461173` (0.46384782574611727 at.%).
5. Legacy 380 °C root: `T=653.15 K`, `xB=0.004664951821454188`, `xAg=0.004654096254055399`.
6. Local consistency: local runtime source is patched to the exact candidate and independently reproduces its root; no CUDA binary was rebuilt and no contract hash is deployed.
7. GitHub consistency: no requested branch or exact contract tag published.
8. Workstation consistency: no; dirty `main` checkout and different binary.
9. Cluster consistency: no; dirty `main` checkout, different binary, and non-Git campaign artifact directory.
10. Existing 246³ results: legacy or mixed/unknown; not exact-labeled.
11. Publication use: exact candidate may be cited as calibration evidence only; exact production/publication claim is blocked.
12. Rerun requirement: unresolved until residual/downstream-impact artifacts and matched sensitivity are available.
13. 400³ start: not allowed under this unresolved contract.
14. Remote dirty worktrees: yes, workstation and cluster source copies.
15. Undeployed environments: exact contract not deployed anywhere.
16. Blocking conflict: exact candidate and legacy runtime are both evidenced, but contract selection, residual/traceability closure, downstream sensitivity and cross-environment identity remain incomplete.

## Campaign continuity decision

Per the user's final instruction, the active workstation/cluster runtime is not
being migrated during the 21-case campaign. The campaign must finish with its
original legacy source and binary identity. Only the local Git branch carries
the exact-candidate source patch; that branch is not a production deployment.

## Safety statement

No legacy result was deleted or relabeled. The user-authorized local source
parameter patch was applied on the current branch; no remote file, binary,
running queue job, checkpoint, branch, tag or GitHub PR was modified. The local
CUDA binary was not rebuilt because `nvcc` is unavailable.
