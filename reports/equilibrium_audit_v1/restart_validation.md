# Workstation restart validation

## Authoritative branch result

The following result supersedes the exploratory table below. It was produced from `codex/pf-zero-mode-restart-provenance-v1` at commit `6b69895af2d1b86b99c57c5479ff767349c61efe`, compiled binary SHA-256 `11d073a272a0b7fa668e40037bc1f96b8389909f95c74d6ad9a5db11236e01af`, with `dt=1e-4`, 128³, R=8 nm, GP OFF and the zero-mode checkpoint contract.

The continuous 128-step run and a 64-step checkpoint + restart to absolute step 128 both printed `PF_ZERO_MODE_FINAL_AUDIT status=PASS` and `checkpoint_restart_provenance=RESTORED_AND_VALIDATED`. The final raw-field hashes were byte-identical:

| field | continuous SHA-256 | restart SHA-256 | result |
|---|---|---|---|
| phi_128.vtk | `af0c8e508b8ff0111edfeb4bf8797c8601907e27af59b16600d0fe690bf36600` | same | PASS |
| xB_128.vtk | `dab88c294ab94f03d687c5d822f7dc4f85ab82ce9a05a7f7220690dd0a435c0c` | same | PASS |
| xBtot_128.vtk | `7fdad1aeb9a421ee76802937f37daf700a5f5ee672f7020f27497b4091eb0b89` | same | PASS |

The branch run's final zero-mode mass error was `5.11743425413158093e-17`. The explicit CLI radius was `R=8 nm`; this is recorded because the parameter file does not define `ic_phi_seed_radius`.

## Elastic branch restart result

The same branch binary was run at 128³, R=8 nm, `dt=1e-4`, fixed-cell elasticity ON. A 64-step checkpoint followed by restart to absolute step 128 produced `PF_ZERO_MODE_FINAL_AUDIT status=PASS` and `checkpoint_restart_provenance=RESTORED_AND_VALIDATED`. Continuous and restarted endpoint hashes were identical:

| field | continuous SHA-256 | restart SHA-256 | result |
|---|---|---|---|
| phi_128.vtk | `dfe939fc3ccfe372b6ae0843b00d9556f1ec2746325a12abeb70918334b9d9f4` | same | PASS |
| xB_128.vtk | `1b16c6decd79bd049c8ca2c8f23d7dab788c1fd437eb30915b514394b28391cc` | same | PASS |
| xBtot_128.vtk | `42925e72318420dcc2281a01a28d52c6d26cfa3d6074c983ce2ef769aad40578` | same | PASS |

The elastic restart endpoint mean mass error was `7.63278329429795122e-17`.

## Non-authoritative exploratory result

The older table below came from the dirty current-worktree `main_cuda.cu`; it is kept for provenance only and must not be used as branch qualification evidence.

The GP-OFF T380 R=8 nm nonelastic case used the frozen zero-mode contract and a checkpoint at accepted step 128. A continuous 256-step run was compared with a restart run from that checkpoint through absolute step 256.

| field | continuous SHA-256 | restart SHA-256 | result |
|---|---|---|---|
| phi_256.vtk | `d367d2ba6dd11859121e19297ad5fb7d023340f8c7e9d16b2d398022ad71c2f7` | same | PASS |
| xB_256.vtk | `9472cc4a6e68d719b2ea35b168e16135e260a6c0e38a8d54f59b6a7b0ca1c871` | same | PASS |
| xBtot_256.vtk | `ee7f23e68b62d3192b0d16263ad5ebfe5a0018d97bca09f124920293edc5578d` | same | PASS |

The runtime printed `checkpoint_restart_provenance=RESTORED_AND_VALIDATED` and `PF_ZERO_MODE_FINAL_AUDIT status=PASS` with final mean mass error `3.13984949151802084e-16`.
