# Preregistered Transport Residual Gate V1 Preservation

## Frozen contract

- Physics/solver baseline: commit `f440c0dcd4c02cd45d9079c35d3838ebfa9b37e2`.
- The formal binary also contains the default-off V1 trajectory observer used
  by this sweep. Its exact workstation source snapshot and hashes will be
  frozen after the running queue terminates; it is not represented falsely as
  byte-identical to the bare commit.
- Workstation formal-queue binary SHA-256: `3986440d157f29c5ab53387345f67bbedf7c264b1a5136b04461cec32a954b0b`.
- Original local-peak diagnostic:

  \[
  D_i=\sum_n \Delta t_n R_{C,i}^{(n)},\qquad
  g_D=\frac{\max_i|D_i^{\mathrm{continuation}}|/3}
             {\max_i|D_i^{\mathrm{common}}|}.
  \]

- Preregistered V1 hard gate: `g_D <= 2`.
- The V1 gate remains implemented by
  `scripts/analyze_transport_residual_gate_long_window.py`; its threshold and
  historical candidate statuses are not changed by V2.

## Preserved observed result

For `G10 + dt8`, the common-window maximum local cumulative defect was about
`1.56e-11` xB-cell units and the three-window continuation maximum was about
`1.84e-10` xB-cell units. Therefore `g_D` was approximately `3.93`, and the
preserved result is:

`FAIL_PREREGISTERED_LOCAL_PEAK_DEFECT_RATE_GATE`

This result is not renamed, deleted, relaxed, or converted to PASS. The V2
contract is an independent, versioned physical-normalization assessment and
will be reported beside V1.

## Evidence preservation policy

- Existing short-window and long-window outputs remain under
  `reports/transport_residual_gate_v1/` and
  `runs/transport_residual_gate_v1_long/`.
- The already-started long-window queue retains its original ordering, inputs,
  simulator binary, and hashes.
- No V2 field is written into a V1 result directory.
- A complete SHA-256 manifest of formal V1 outputs will be generated only after
  the unchanged queue finishes, so files still being written are not falsely
  described as frozen.

## Completed evidence freeze

- Formal long-window candidates terminal: `13/13`.
- V1 source snapshot: every root-level `.cu`, `.h`, and `Makefile` asset from
  the exact workstation formal repo.
- Hash manifest records: `2412`.
- Manifest: `preregistered_v1_hash_manifest.csv`.
- Manifest SHA-256:
  `773099c65ec7dab6707e280d4dc5d5f910f7fba1e4ca3311dbf61fdd1a3b639f`.
- Formal binary SHA-256 remained
  `3986440d157f29c5ab53387345f67bbedf7c264b1a5136b04461cec32a954b0b`
  throughout the queue.

Current preservation state: `V1_CONTRACT_AND_EVIDENCE_FROZEN_COMPLETE`.
