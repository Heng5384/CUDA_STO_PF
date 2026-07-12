# Portable Dynamic-Continue Final Report

## Executive result

`PARTIAL_DYNAMIC_CONTINUE_BUNDLE_PASS_SOME_OVERLAYS_BLOCKED`

Commit A implements the explicit CUDA path contract and bundle integrity checks.
The formal `dynamic_continue_v1` bundle contains the three target entries and
passes offline mirror/checksum/negative tests. The five overlay files have been
rewritten to `bundle:` paths in the working tree, but remain untracked and are not
accepted because CUDA compile plus multi-cwd startup validation was unavailable.

## What changed

- No PF/CNT/GP/RSMD equations or physical values changed.
- No staging file was renamed into canonical status.
- No original Results tree was copied wholesale.
- The emitted bundle contains only profile arrays, metadata, source-dynamic
  summaries, library mirrors, manifest, and a status README.
- `physical_units_config.json` remains excluded.

## Remaining gate

Compile on a CUDA workstation, run the same complete base parameter plus each
overlay from repository/build/unrelated cwd with explicit roots, and capture the
resolved checksums/entry IDs. Only then may the five overlays become Commit C.
