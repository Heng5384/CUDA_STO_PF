# Workstation short-run audit evidence

All cases below used the isolated source root `/home/zhiheng/tmp/codex_equilibrium_audit_v1_20260729`, T=380 °C, GP OFF, GP birth OFF, GP release OFF, beta nucleation OFF, no source, and `PF_CONSERVED_Y_ZERO_MODE_V1` unless noted.

| case | grid / R | elasticity | dt | steps | endpoint physical time | wall s/step | zero-mode mass error | result |
|---|---|---:|---:|---:|---:|---:|---:|---|
| nonelastic R8 | 128³ / 8 nm | off | 0.001 | 256 | 12.68385 s | 0.042673 | 3.14e-16 | PASS short conservation |
| elastic R8 | 128³ / 8 nm | on | 0.001 | 128 | 6.341927 s | 0.136887 | 1.73e-18 | PASS short conservation |
| nonelastic R8 restart | 128³ / 8 nm | off | 0.001 | 128→256 | 12.68385 s | 0.045449 (restart leg) | 3.14e-16 | PASS bytewise restart |
| dt coarse | 128³ / 8 nm | off | 2e-4 | 128 | 1.268385 s | 0.044836 | 3.83e-16 | PASS local dt |
| dt fine | 128³ / 8 nm | off | 1e-4 | 256 | 1.268385 s | 0.041840 | 3.12e-17 | PASS local dt |

The elastic/nonelastic wall-time ratio is about 3.21 for these small, fixed-cell runs. The short runs show evolution, not equilibrium: nonelastic R rises from 9.905864 to 9.989541 nm; elastic R falls to 9.815335 nm. The full R=6–20 nm equilibrium ladder, converged slab, and profile/V2 tests remain open.

## Branch-authoritative replacement

The rows above are exploratory results from the dirty current worktree and are not qualification evidence. The authoritative branch run uses an explicit CLI radius of 8 nm, 128³, GP OFF, `dt=1e-4`, and 128 steps (source commit `6b69895af2d1b86b99c57c5479ff767349c61efe`, binary SHA-256 `11d073a272a0b7fa668e40037bc1f96b8389909f95c74d6ad9a5db11236e01af`). It completed with zero-mode PASS, mean mass error `5.11743425413158093e-17`, wall time `0.046305 s/step`. The 64-step checkpoint/restart to step 128 completed with zero-mode PASS and byte-identical `phi`, `xB`, and `xBtot` VTK hashes; see `restart_validation.md`.

The matching authoritative elastic branch run used the same source/binary, 128³, explicit R=8 nm, `dt=1e-4`, 128 steps, and fixed-cell periodic elasticity. It completed with `PF_ZERO_MODE_FINAL_AUDIT status=PASS`, final mean mass error `7.63278329429795122e-17`, average step wall time `0.136422 s`, and endpoint physical time `0.6341927 s`. Endpoint hashes are `phi=dfe939fc3ccfe372b6ae0843b00d9556f1ec2746325a12abeb70918334b9d9f4`, `xB=1b16c6decd79bd049c8ca2c8f23d7dab788c1fd437eb30915b514394b28391cc`, `xBtot=42925e72318420dcc2281a01a28d52c6d26cfa3d6074c983ce2ef769aad40578`. This is a short dynamic conservation check, not an equilibrium result.

The elastic 64-step checkpoint/restart to step 128 also passed provenance validation and reproduced those three endpoint hashes bytewise.
