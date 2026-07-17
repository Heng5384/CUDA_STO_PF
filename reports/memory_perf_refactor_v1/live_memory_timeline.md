# Live memory timeline

Static PF-only 400^3 source allocation falls from 19.579538 GiB to 12.182601 GiB.

| Stage | Dominant live groups | Refactor evidence |
|---|---|---|
| Initialization | authoritative state, histories, FFT/scratch | optional feature arrays absent |
| BDF2 context | Ctot/phi anchor/context | later reused by event rollback |
| Transport | one positive-face field, divergence, residual, derived context | y/z face fields absent |
| Phase PDAS | phase spectrum/PCG/scratch | Y spectrum aliases phase spectrum |
| Cold audit | transport scratch and one face field | no three-face persistence |
| Commit | authoritative/history fields | no outer-history fields |
| Event fallback | BDF2 anchor/context become macro rollback images | zero incremental full-grid allocation |
