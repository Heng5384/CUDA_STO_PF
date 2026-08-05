# PbTe–Ag2Te writing guidance package V3 — scope and authority

## Frozen purpose

This package reframes the manuscript around a finite-lifetime, interface-rich low-thermal-conductivity state and the separation between coarsening pathway and thermal consequence. It is a writing and evidence-organization freeze only. It does not modify simulation code, parameters, checkpoints, raw fields, transport data, or literature originals.

## Central story

Sheskin's 6–48 h measurements define an experimental low-κ trajectory. Because APT, SEM/TEM and phase-field segmentation do not observe identical object classes, the 6 h PF-resolved β population is not uniquely reconstructable. The model therefore propagates experimentally admissible resolved-β ensembles rather than claiming a unique real initial state. PSD defines the available size hierarchy, spatial organization selects the survival pathway, and—within the frozen V2 interface-scattering contract—loss of interfacial area sets the dominant modeled thermal recovery.

## Dataset boundary

Sheskin and Yu are used as complementary but non-interchangeable datasets: the former defines the 6–48 h microstructure–thermal-conductivity trajectory, whereas the latter resolves the broader defect redistribution associated with long-term annealing.

Grossfeld–Sheskin 2017 is not silently merged with Sheskin 2018. Yu's AQ→48 h endpoints are not spliced into Sheskin's 6→48 h trajectory.

## Evidence authority

Authority order is: immutable checkpoints/raw registered artifacts → hash-pinned production or conditional post-processing reports → reproducible derivation under a frozen contract → literature extracts → core memory → task specification. `PROJECT_CORE_MEMORY.md` routes evidence but cannot by itself support a manuscript number.

The Case 001–003 values in V3 are `COMPLETE_CHECKPOINT_CONDITIONAL_ANALYSIS`, not production authority. All three 48 h checkpoints exist and pass segment-level conservation/elastic gates; the old campaign closure failed at the 64-particle/schema anchor, and the later merge-aware pipeline remains blocked or incomplete. The 573.15 K Case 003 transport value is a read-only replay with the frozen V2 M0/MI contract and is retained only as a conditional evidence snapshot.

## Status vocabulary

- `SUPPORTED`: traceable artifact and appropriate identity exist.
- `SUPPORTED_WITHIN_FROZEN_CONTRACT`: direct model result, not a universal material law.
- `CONDITIONAL`: complete checkpoint or reproducible derivation exists, but production authority is not frozen.
- `PARTIAL`: only part of the claim has direct evidence.
- `PENDING_PRODUCTION_AUTHORITY`: required schema/merge-aware rerun is unfinished.
- `UNVERIFIED`: no traceable artifact was found.
- `BLOCKED`: manuscript-strength use is prohibited until the named dependency closes.

## Package location and predecessor

- V3 Git-tracked package: `/Users/heng/Documents/GitHub/CUDA_STO_PF/paper_writing_guidance_package_v3/`
- Preserved V2 package: `/Users/heng/Desktop/1_Solubility_Calibration/paper_writing_guidance_package_v2/`
- Branch: `codex/reframe-metastable-low-k-window`
- Parent HEAD at branch creation: `3551d07388aed02d4d6d896c20f2cc967a60a088`

The repository was dirty before this work. Pre-existing simulation/post-processing changes are excluded from the V3 writing commit.
