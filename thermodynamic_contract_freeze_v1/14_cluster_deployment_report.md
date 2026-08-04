# Cluster deployment report

Status: `NOT_DEPLOYED_READ_ONLY_AUDIT`

Host: `login1` via `uvip-cluster`; scheduler: Slurm.

Source copy audited: `/data/home/luozhiheng/CUDA_STO_PF`

- Branch: `main`
- Commit: `9db5fb4572523b126e49b7152c1272f745bdd2aa`
- Worktree: dirty; runtime headers, CUDA sources, configs and scripts modified/untracked.
- `thermo_utils.h` hash: `5257598d8bc54f4d6b401a538eb187d044965fec014bad6de44f308f99b97cea`.
- `Unit_Psedobinary.py` hash: `b666627e906544bdaad1f233065f541fcdfe88571b1173498635cc87770c1a43`.
- Binary `main_cuda` hash: `d8f27009964df4cbe306bb9471463c02b33b80767fc8516849cb9d2c31a38f90`.
- No contract hash or exact-fit identity was found.

Existing Slurm jobs were observed and left untouched. At audit time, `74232_3` was RUNNING, `74232_[4-21%1]` was PENDING, and older `74017` was RUNNING. No queue cancellation or binary replacement was performed.

The campaign artifact directory `/data/home/luozhiheng/tmp/pf_400cube_psd_spatial_density_ladder_production_v1_20260803` is not a Git repository and has no deployable source identity. It cannot serve as a frozen environment.

## Final user decision

The separate build job was cancelled and the temporary exact-candidate
worktree was removed on 2026-08-05. No runtime guard, source patch, or binary
replacement was installed in the active campaign source root. The running and
pending 21-case campaign therefore continues consistently with:

```text
L(T)=41212.9-18.05*T J/mol
thermo_utils.h_sha256=5257598d8bc54f4d6b401a538eb187d044965fec014bad6de44f308f99b97cea
main_cuda_sha256=759956a89780db9a19ccd51b4115319de47463a3fd7b8d446a022a74c9beeb3b
```
