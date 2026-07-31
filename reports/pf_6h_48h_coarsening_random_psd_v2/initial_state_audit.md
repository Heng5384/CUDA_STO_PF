# Initial-state audit

Status: `PASS_PF_RANDOM_PSD_STATIC_FIELDS_V2`

The hash-pinned materialized state has 96 resolved components in a `246^3`
periodic domain. The measured mean `h` is `0.023929544766323994`; the measured
mean conserved composition is `0.029999999999997168`. The field extrema are
`min(phi)=0` and `max(phi)=0.9999782981299825`. The static materializer reports
that no clipping, periodic overlap, or physical mass projection was required.

The radius CV is `0.0832655737`, with reference CV `0.25`; this is a continuous
set of 96 unique quantiles, not a three-class or monodisperse fixture. The
complete field and manifest hashes are recorded in `fixture_hashes.md`.

The dynamic solver's separate bound-safety counters must not be confused with
the static materializer result. In the 6--12 h preflight the reported
`phi_clip_count_*` is a numerical bound guard; the corresponding composition
mass increments remain at roundoff (`|delta_mass| <= 2.84e-15`). This is
reported explicitly rather than relabeled as a no-clipping dynamic audit.
