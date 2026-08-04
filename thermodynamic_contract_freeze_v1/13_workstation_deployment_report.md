# Workstation deployment report

Status: `NOT_DEPLOYED_READ_ONLY_AUDIT`

Host: `fuxin` via `workstation-tail`

Project copy audited: `/home/zhiheng/PF/CUDA_STO_PF`

- Branch: `main`
- Commit: `a9f4b6ef91d6974f8395c6fe3169bf53c6422891`
- Worktree: dirty with extensive deletions and untracked files.
- `thermo_utils.h` hash: `5257598d8bc54f4d6b401a538eb187d044965fec014bad6de44f308f99b97cea`.
- `Unit_Psedobinary.py` hash: `b666627e906544bdaad1f233065f541fcdfe88571b1173498635cc87770c1a43`.
- Binary `main_cuda` hash: `c96bb6f8c0a08c8ccfb155f02db4eae21ce1a103b607bd4b39d6569c36f2b58a`.
- No contract hash or exact-fit identity was found.
- No active `main_cuda` compute process was observed at audit time.

No pull, checkout, reset, build, binary replacement, or configuration edit was performed. Deployment requires a clean isolated worktree and an authoritative contract first.

## Final user decision

The temporary exact-candidate worktree created during deployment preparation
was removed on 2026-08-05. The production checkout and its binary remain on the
legacy formula. No workstation simulation queue was changed.
