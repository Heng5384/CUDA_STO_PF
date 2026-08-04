# 246³ quarter-nm Method-1 A/B/C assembly and launch

Date: 2026-08-01

## Outcome

The 15-entry elastic target-profile library, exact integer Method-1 inventory
selection, and three non-overwriting 246³ fixtures passed their static gates.
The new A/B/C 6--48 h runs were then submitted to `gpu_uvip` before the
superseded jobs were cancelled.

```text
static_status=PASS_246CUBE_QUARTER_NM_HANDOFF_STATIC_V1
production_status=RUNNING_OR_QUEUED_NOT_YET_QUALIFIED
```

## Frozen identities

```text
library_manifest_sha256=de4142e0268e379f70fd1c860aab9e004421d07df4f8d872f4eaeb3dbef7af5b
inventory_selection_sha256=57e948762f97011116eeba61b87dcf19c16fffb755357ac228742419585e3076
runtime_binary_sha256=516489b3e4dbafd6ba5876beb2858df8309fbfcbd1065d455b73f1782f5fe8f5
parameter_sha256=ecbdd0ac070bdf5e5d214322b5248a08f5ca5dd4e0670427513f5ef977ea977a
dt_code=0.02
dt_physical_s=0.9909260953431841
final_step=152585
checkpoint_cadence=3633
```

## Static fixtures

| Case | Fixture SHA-256 | Initial matrix xAg | Inventory relative error |
|---|---|---:|---:|
| A | `f6cce1bd1abaf0e8767f52fc70009a1cce9c928a442bc3ebd38e7c8e7c26d7b2` | 0.006202336869744194 | 1.303e-16 |
| B | `8106091d92b2924f65a23ab5adf8e98d958dd6bac63f3fb099c4f85567c40389` | 0.006202284267465961 | 2.607e-16 |
| C | `2ad665a3b58d26c29fcfdbfc2c949be979e602147283fce25bc00d992acaac64` | 0.006202343009149754 | 6.517e-16 |

All three fixtures contain 96 particles, satisfy `mean_C_Btot=0.03`, and keep
the initial matrix Ag concentration inside the experimental 6 h band. No
single-particle profile was scaled or interpolated.

## Cluster launch

| Case | Slurm job | Partition/QoS | State at launch audit | Output root |
|---|---:|---|---|---|
| A | 73364 | `gpu_uvip` | RUNNING | `/data/home/luozhiheng/tmp/pf_246cube_6h48h_quarter_nm_method1_A_v1_20260801` |
| B | 73365 | `gpu_uvip` | PENDING | `/data/home/luozhiheng/tmp/pf_246cube_6h48h_quarter_nm_method1_B_v1_20260801` |
| C | 73366 | `gpu_uvip` | PENDING | `/data/home/luozhiheng/tmp/pf_246cube_6h48h_quarter_nm_method1_C_v1_20260801` |

The superseded jobs `73301`, `73302`, `73304`, and `73309` were cancelled only
after all three new jobs had been accepted by Slurm. Their existing output
roots were preserved.

## Initial performance observation

At the first live audit, A had advanced to approximately step 10,755 with an
empty driver stderr. The completed interval from step 3,633 to 7,266 took
about 564 seconds, or approximately 0.155 seconds per macro step, including
the checkpoint boundary. This projects to roughly 6.5--6.8 hours per case on
the current A100. Because the QoS currently admits one running job per user,
the three cases are expected to complete serially in roughly 19--21 hours if
the rate remains stable.

This is a throughput estimate, not a final production qualification. Final
scientific status requires all three complete chains and the registered
merge-aware, conservation, zero-mode, restart/provenance, and ensemble audits.
