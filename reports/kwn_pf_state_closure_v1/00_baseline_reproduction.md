# Baseline reproduction — KWN–PF state closure v1

## Frozen start point

- Branch: `codex/kwn-pf-state-closure-v1`
- Start commit: `f228cd45638279413ea62edd774d5b2315dbcbf6`
- Start configuration SHA-256: `2aa878559ca897fdbb80b384791280eac2d642dc6b175b116fb109abcfe8da43`
- Scope: reproduction only; no PF executable was launched.

## Commands and outcomes

| Command | Outcome |
|---|---|
| `PYTHONPATH=src python3 -m unittest discover -s tests/kwn -v` | PASS: 14 tests |
| `PYTHONPATH=src python3 -m unittest discover -s tests/coupling -v` | PASS: 9 tests |
| `PYTHONPATH=src python3 scripts/run_beta_only_consistency.py` | PASS: exited 0; retained P0 conflict status |
| `PYTHONPATH=src python3 scripts/run_gp_feasibility_sweep.py` | PASS: exited 0 |
| `PYTHONPATH=src python3 scripts/build_kwn_pf_handoff.py` | `PARTIAL_PF_STATE_NOT_CLOSED` as expected |
| `python3 scripts/run_pf_handoff_smoke.py` | `NOT_RUN_P0_CONTRACT_CONFLICT` as expected; PF executable not invoked |
| `PYTHONPATH=src python3 scripts/make_kwn_pf_report.py` | `PARTIAL_PF_STATE_NOT_CLOSED` as expected |

## Preservation note

The legacy scripts regenerate tracked files under `outputs/kwn_pf_mvp_v1/` and
`reports/kwn_pf_mvp_v1/` (including CSV line-ending-only rewrites). Those files
were restored exactly to `HEAD` immediately after recording the outcomes above.
The legacy v1 package remains unchanged; all new artifacts are written below
`outputs/kwn_pf_state_closure_v1/` and `reports/kwn_pf_state_closure_v1/`.

## Gate

`PASS_BASELINE_REPRODUCTION`
