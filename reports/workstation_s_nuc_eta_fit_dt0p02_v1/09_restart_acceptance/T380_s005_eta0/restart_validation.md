# T380 s_nuc=0.05, eta_rel=0 restart validation

Date: 2026-07-27  
Temperature/box/seed: T380 / APPARENT_N4 (246^3) / SEED_A  
Registered code timestep: `dt_code=0.02`  
Physical prediction: `false`  
Fixture: non-accelerated, apparent-density validation scenario

## Runtime evidence

The continuous endpoint (`256` steps), the first-half endpoint (`128` steps),
and the resumed run (global endpoint `256`, restored from the first-half
checkpoint) all reported `PASS_CASE_RUNTIME` with empty stderr.  The exact
source binary SHA-256 was
`b24f62c1e987d1529c1043a1bbe49338342ed60bf3866e6c4c9a9be89b8cc397`.

| leg | accepted steps | elapsed (s) | seconds/accepted step | checkpoint SHA-256 |
|---|---:|---:|---:|---|
| continuous | 256 | 259.21731305122375 | 1.0125676291063428 | `819d5cb44ee62c2976c7a71a2be1653c44ec2fa162241f4b72552a8be22b5ecf` |
| first half | 128 | 63.98780536651611 | 0.4999047294259071 | `bb28e0ffab9e9e45013211ff456dc6837edcfaee19a4ff9846446837c21e12a3` |
| restarted to 256 | 256 global | 197.12975311279297 | 0.7700380980956749 | `819d5cb44ee62c2976c7a71a2be1653c44ec2fa162241f4b72552a8be22b5ecf` |

The restarted checkpoint is byte-identical to the continuous endpoint.  The
restart input checkpoint SHA-256 is the first-half value above.

## Bytewise gates

The final `raw_phi`, `raw_Y`, `raw_xB`, `raw_xBtot`, and `raw_dY_dt_prev`
files at step 256 are byte-identical between continuous and restarted runs.
The final checkpoint is also byte-identical.  The GP shadow ledger is written
from the resume point in the restarted leg; its common global-step interval
(steps 129--256) is byte-identical to the corresponding continuous ledger
rows.  The overlap SHA-256 is
`45de6089bafa79a8e7d542261c5646b6747885047ff264c8323cb14451333986`.

No physical or GP parameter was changed, no accelerated-event fixture was
used, and no cluster resource was used.

**restart_status = PASS_T380_RAW_FIELDS_CHECKPOINT_AND_LEDGER_OVERLAP_BYTEWISE**

