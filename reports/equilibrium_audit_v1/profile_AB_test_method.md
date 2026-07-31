# Profile A/B test method

A is the analytic tanh profile generated from the frozen single-particle profile function. B would be a local-equilibrium composition/profile generated with identical geometry, inventory, far-field composition and temperature.

The selected branch was tested with `--mode=minimize --minimize-full-model`; it
can generate a relaxed VTK profile, but the executable rejects
`--pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1` in minimize mode. Raw/VTK
continuation is also rejected by the provenance guard. A separately validated
uncommitted extension supplied the missing fresh-raw handoff and completed the
one-particle R=8 and multi-particle 96^3 A/B/C diagnostics, with restart
checks. Those results close the validation-only diagnostic gate but do not
change the clean-branch method blocker. The clean-branch requested
elasticity-OFF R=8/10/12 and R=10 elasticity-ON A/B comparisons therefore
remain unclaimed.
