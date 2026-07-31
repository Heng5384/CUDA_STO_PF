# Performance

| replicate | 256-step wall s/step | 6.07--8 h wall s/step | screening segment wall s | mean GPU util (%) | peak VRAM (MiB) |
|---|---|---|---|---|---|
| A | 1.27794921875 | 1.25492368046 | 8797.015 | 99.4536585366 | 4681 |
| B | 0.45626171875 | 0.421468473609 | 2954.494 | 97.1888182123 | 4411 |
| C | 0.45272265625 | 0.421601141227 | 2955.424 | 97.4426174497 | 4411 |

A ran on the registered workstation binary; B and C ran with the registered
cluster sm_80 binary.  The wall-clock values characterize those devices and
must not be interpreted as a hardware-independent physics difference.  All
runs used 246^3 cells, production `dt_code=0.02`, elasticity, the same
physical parameters, sparse checkpoints, and suppressed bulk VTK output.

Runtime analysis binary hashes:

```text
A=be44b81661f4902e1ef0c823ed45a91f5ed40e5904c573923686c8112fdc5336
B=1c5e1000f94c169bdc947a8ed5413f7fd84e3ba768caa042dbb12210df9eeb44
C=45c9a1436093bd1e028a7467f4a5d4b0f93a6530ed4e3482e3bdbbb720fdb4c3
```

SHA-256 identities of the copied runtime input-hash lists:

```text
A=9e5ba4cfc1fa4f4399639f3949243ebf68bc7d6962768a03e00b3aa10b5b84f5
B=be4dbfa00ab05b70717157e534b78bb8878563bd6d12e4004255b6e51305ec88
C=e61881ab9721ccf1a03e647637d9b8f156be1b0498bec0abc2bc9b54f5058b1a
```

Final report assembler SHA-256:
`24d619a6ef49b1329e85a2f2c742f26037ddcbc98e7a577c2c5e03e05101e361`.
