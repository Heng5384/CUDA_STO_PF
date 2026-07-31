# Multi-particle V2 restart validation

All three cases used a 64-step checkpoint and a restart to the common 128-step
endpoint.  The continuous and restarted final fields are byte-identical for
`phi`, `xB`, and `xBtot`.

| case | checkpoint provenance | `phi_128.vtk` SHA-256 | `xB_128.vtk` SHA-256 | `xBtot_128.vtk` SHA-256 |
|---|---|---|---|---|
| A tanh, elastic OFF | `RESTORED_AND_VALIDATED` | `b9de4d2d64d2bb50caf50f3e349c4d2c51638def9c113548bc1a54c21a3fe619` | `99703e3681408115523bcd0ea6d07cff205cf73c3db45c46f79dc1f20cdc2693` | `fa78a1b7267d885e74aca15747593073fb4cb538b0f9784e176452586bcc282c` |
| B local, elastic OFF | `RESTORED_AND_VALIDATED` | `a368531afc6be3da21df703f7990764c3d6ae626b4240caad8d059425323bab5` | `7e4d99225efee6c82661a71e07186c7df76cd5fa68b7582a1785507293b47910` | `1c959f4c06a874a2e82a692f548f0be3dbe22edf1167292c8f9d349b7a6e1d49` |
| C local, elastic ON | `RESTORED_AND_VALIDATED` | `cc7c5f2b8904da3cccd3925fdef207a61ee15bf882688910943280a29f64452a` | `646a74197ba98e5f6b2536c0553924f0a73b41dc9846c856fcfb7686e2d02807` | `032a42c56bc17565292981aada89bbe03b26bce30ae5937c26ecfda03636ad84` |

The corresponding restarted runs reported zero-mode mean mass errors of
`3.4663630e-15`, `3.2566542e-15`, and `6.0651073e-15` for A/B/C.  Their
checkpoint hashes are respectively:

```
A final checkpoint 4c6801e4fa5e4db64b6d47fc43a3b9cd2aa49698c904b781cf966a6322a0fd2e
B final checkpoint cf997258b0d89ffc89d1c8cc5338db3aa1ef3aee067367653b3b78020a491dae
C final checkpoint 22bf9d406955ee76db1bc37e26c586aa6d9ab8d950e790c8cf1b3e6f6d96c59b
```

The restart comparison is extension-only evidence; the clean committed branch
still rejects fresh raw profile epochs by design.
