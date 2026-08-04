# Campaign continuity decision

Date: 2026-08-05

The user cancelled cross-environment deployment of the exact-candidate
thermodynamic parameters because changing a shared binary during a segmented
21-case campaign could mix contracts across checkpoints.

The binding operational decision is:

1. Workstation and cluster production remain on
   `L(T)=41212.9-18.05*T J/mol` until the existing 21 cases finish.
2. No running or pending simulation job is cancelled, modified, or relabeled.
3. No runtime guard or binary replacement is installed in the campaign source
   root.
4. Temporary isolated exact-candidate worktrees are removed.
5. The local branch may publish the exact-candidate source patch for later
   review, but that publication is not a deployment or production freeze.
6. Existing and in-flight results retain legacy provenance. A future switch to
   the exact candidate requires a new campaign boundary and fresh binary,
   parameter, fixture, checkpoint, and analysis provenance.

Verified active cluster campaign identity at decision time:

```text
thermo_utils.h_sha256=5257598d8bc54f4d6b401a538eb187d044965fec014bad6de44f308f99b97cea
main_cuda_sha256=759956a89780db9a19ccd51b4115319de47463a3fd7b8d446a022a74c9beeb3b
formula=L(T)=41212.9-18.05*T
```
