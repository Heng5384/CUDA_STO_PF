# Frozen production elastic contract

- Authority: `reports/pf_246cube_method1_production_authority_v1`.
- Runtime binary SHA-256: `516489b3e4dbafd6ba5876beb2858df8309fbfcbd1065d455b73f1782f5fe8f5`.
- Main CUDA source SHA-256: `76b09b9334e1dace77b39d21ca489061ae9e5aa104f45afca3f04b47e362f9d7`.
- CUDA kernels SHA-256: `0d36841845992e22ff4db3b4903200f49dc2b3ad1d42ab415b27477d72217f5a`.
- Checkpoint source SHA-256: `e9c65556ab68ba231493924b4c90b36cc0c3a69430930f2b540774f6c00c6003`.
- Parameter-file SHA-256: `ecbdd0ac070bdf5e5d214322b5248a08f5ca5dd4e0670427513f5ef977ea977a`.
- Fixed periodic cell, zero external strain/stress, identity orientation, and eigenstrain `[0.046, -0.022, -0.017, 0, 0, 0]`.
- Solver inputs named `S_ij`: S11=214.2857142857, S12=S13=11.9047619048, S22=S33=140.8730158730, S23=85.3174603175, S44=101.1904761905, S55=S66=27.7777777778 (frozen parameter-file units).
- Checkpoint V4 stores displacement warm state from source field `n-1`, while stored phi/Y/xB are accepted field `n`; matching n-1 fields are absent.
