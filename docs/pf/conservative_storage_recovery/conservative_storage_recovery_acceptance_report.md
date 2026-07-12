# Conservative Storage Recovery Acceptance Report

## Scope

This recovery restores authoritative conservative-storage restart and final
checkpoint support without changing PF, CNT, GP, RSMD, or seed physics. Mode L,
Mode X, Mode Q, and S3 scientific acceptance states are unchanged.

## Provenance

- Exact patch group A supplied the `init_conservative_storage_raw_path` intent,
  CLI entry, device load, context reconstruction, and restart marker.
- Exact patch group B supplied final float64 raw/JSON output and
  `PF_CONSERVATIVE_CHECKPOINT_WRITTEN`.
- Both candidates were Codex `apply_patch` records rather than Git patches and
  were manually rebased onto the current source structure.
- The `dt=0.00025`, 2000-step analyzer case was reconstructed from task context,
  not recovered byte-for-byte. It preserves the existing equal-time protocol:
  `0.00025 * 2000 = 0.5` code-time.

## Semantic Rebase

The current implementation deliberately strengthens the recovered candidate:

- checkpoint storage is always read and written as float64;
- a same-basename JSON companion must identify the matching composition mode,
  grid shape, dtype, and authoritative status;
- missing, short, oversized, wrong-shape, and wrong-mode checkpoints fail;
- metadata records step, segment physical time, and raw filename;
- legacy/non-conservative mode cannot silently accept conservative storage.

## Validation

Workstation validation used CUDA 12.9 on an RTX 5080. `make main_cuda` passed.
Reference conservative and storage-exact tests passed, as did Python syntax,
focused checkpoint tests, CLI help, and analyzer dry-run generation.

An 8x8x8 nonuniform matrix case compared a four-step checkpoint plus four-step
restart against an uninterrupted eight-step run. Phi, xB, and authoritative
storage all matched with `Linf = 0`; the storage field remained nonuniform.
Segment physical times summed exactly to uninterrupted physical time. A separate
uniform round trip also matched exactly.

Negative runtime tests returned exit code 2 for missing metadata/raw input,
trailing bytes, wrong shape, and Ctot checkpoint loading into qalpha mode.
Positive Ctot and qalpha checkpoint/restart paths both emitted their load/write
markers.

## Remaining Boundary

The restart contract still requires exact phi and xB raw fields alongside the
authoritative storage checkpoint. Lossy VTK conversion is not an exact restart
source. The checkpoint metadata reports segment-local step/time; callers that
chain segments must maintain cumulative run metadata externally.

Overall status: **PASS**.
