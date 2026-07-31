# T380 long-window continuation — s_nuc=0.05, eta_rel=0

This run uses the latest workstation source and the non-accelerated apparent-
density fixture. It is the first registered long-window branch after the
T380 continuous/restart gate.

- temperature: T380
- grid: APPARENT_N4 (246^3); the user-authorized 309^3 case is not run
- seed: SEED_A
- `s_nuc=0.05`, `eta_rel=0.0`, `dt_code=0.02`
- registered endpoint: 21,798 accepted macrosteps (the registered 6-hour
  physical endpoint for this fixture)
- output cadence: 4,096 steps
- particle-analysis cadence: 256 steps
- checkpoint cadence: 4,096 steps, with the existing SIGUSR1 checkpoint hook
- physical_prediction: false
- accelerated_event_fixture: false
- workstation only; no cluster, commit, push, or parameter retuning

The run is serialized ahead of the eta=1 endpoint and will be preserved under
a new non-overwriting output root.

