# GitHub freeze report

Status: `PASS_LOCAL_AND_GITHUB_FROZEN_REMOTE_DEPLOYMENT_PENDING` is **not** claimed; current final status remains `BLOCKED_AUTHORITATIVE_CONTRACT_CONFLICT`.

- Repository: `git@github.com:Heng5384/CUDA_STO_PF.git`
- Remote reachable: yes.
- Remote default branch: `main`.
- Current requested branch `CUDA_STO_PF-transport-v3-polarization-anisotropic`: not present on `origin`.
- Thermodynamic exact-fit branch/tag: not found in remote refs.
- GitHub CLI: not installed; PR state was not queried through `gh`.
- The user subsequently authorized publishing the local exact-candidate source
  patch on branch `CUDA_STO_PF-transport-v3-polarization-anisotropic` while
  explicitly keeping workstation/cluster production on the legacy contract
  until the 21-case campaign finishes.
- Publishing this branch records a candidate source change only. It does not
  deploy, freeze, or relabel any production environment or historical result.

The calibration workspace is now registered as an external candidate source in
the local audit, but it is not a GitHub-tracked contract and has not been
deployed. The branch remains intentionally separated from the active legacy
campaign runtime.
