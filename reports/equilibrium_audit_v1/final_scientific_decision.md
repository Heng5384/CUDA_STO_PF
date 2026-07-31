# Final scientific decision

## What is established

The frozen branch source (`codex/pf-zero-mode-restart-provenance-v1`, commit `6b69895af2d1b86b99c57c5479ff767349c61efe`) has a uniquely reconstructed thermodynamic contract and composition conversion. The analytic chemical-flat root at 380 °C is `xB=0.004664951821454188`, `xAg=0.004654096254055399`, well below the experimental interval `0.0058–0.0066`. True branch runs with GP/birth/release/nucleation OFF pass the conserved-Y zero-mode audit for R=6, 8, 10, 12, 14, 16 and 20 nm short finite-particle observations. The representative R=8 continuous/restart raw fields are byte-identical, and the fixed-cell elastic short runs cover R=8, 10, 12, 14 and 20 nm.

## What is not established

The current branch has no supported slab initializer: `oneD_test_mode` is not consumed, and the clean zero-mode contract rejects raw/VTK continuation and minimize-mode profiles. The no-zero-mode raw slab fails the mass criterion. Consequently there is no clean-branch numerical chemical or coherent planar slab root and no clean-branch locally equilibrated profile-B handoff. An isolated uncommitted extension now has one-particle and three-particle A/B/C zero-mode/restart diagnostics plus a five-point two-interface slab sign bracket; the wide bracket shrinks at xAg=0.003, grows at xAg=0.008, and places the analytic root near zero short-time drift. These remain validation-only extension evidence rather than clean-branch evidence. The finite-particle observations are dynamic windows dominated partly by interface relaxation; the additional R=8 low-end 40.588 s run strengthens the negative growth-side observation but still does not establish an equilibrium root.

## Decision

`INCONCLUSIVE_MORE_EVIDENCE_REQUIRED`, with the concrete method blocker `BLOCKED_PROFILE_EQUILIBRATION_METHOD` (also blocking the numerical slab route). The analytic result does not support interpreting xAg=0.0062 as the pure chemical planar equilibrium. The diagnostic L-only shift required to reach xAg=0.0062 is −5.0613%, but no formal fit covariance or objective degradation is available, so no refit is justified and no production parameter was changed.

Recommended next action: add a separately validated, zero-mode-compatible slab/profile materialization path (without changing production thermodynamics), then repeat the planar and profile checks on the clean branch before making a publication-level equilibrium claim. The completed three-particle extension should be retained as bounded supporting evidence, not promoted as a fitted or production particle distribution.
