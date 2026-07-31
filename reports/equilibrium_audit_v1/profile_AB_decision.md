# Profile A/B decision

`BLOCKED_PROFILE_EQUILIBRATION_METHOD`: the branch's minimize mode can relax the standard seed, but `PF_CONSERVED_Y_ZERO_MODE_V1` explicitly rejects minimize and raw/VTK continuation. Consequently a locally equilibrated profile cannot be injected into a zero-mode dynamic run without changing the frozen runtime contract. The analytic-tanh path remains the only directly qualified initialization; no profile winner is selected.

A read-only workstation materialization diagnostic was nevertheless completed
for the same 128^3, R=8 nm geometry.  The minimized candidate differs from the
analytic seed by mean `|Δphi|=5.85536e-5`, mean `|ΔxB|=8.70993e-6`, and a
`2.50530e-3` voxel fraction with `|Δphi|>0.01`; its mean h-volume is 3.57039%
lower.  These values quantify the profile contrast but are not a zero-mode
dynamic A/B result.  See `profile_AB_materialization_diagnostic.md` and the
workstation JSON path recorded there.

An isolated uncommitted extension subsequently accepted fresh raw fields and
ran one-particle A/B/C dynamics with zero-mode and restart checks; the exact
numbers and hashes are in `profile_AB_extension_v1.md`.  The same extension
then completed the registered 96^3 three-particle A/B/C comparison, including
particle identity, no merge/split, and continuous/restart bytewise checks;
the results are in `multi_particle_v2_results_v1.md` and
`multi_particle_v2_restart_validation.md`.  This closes the validation-only
multi-particle comparison but does not change the clean-branch decision or
establish a production profile winner.  The fixture contract is specified in
`multiparticle_v2_fixture_contract.md`.
