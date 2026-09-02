# Minimal repair decision

Implemented: canonical direct cell initialization, a shared metric/closure module, explicit lower-bound number/volume/mol-B tallies, and literal face-Courant telemetry.

Not implemented: a new high-order transport solver, MUSCL/TVD, GP release, physical retuning, or an unverified boundary remap.  The next minimal numerical change must unify the Eulerian lower-face characteristic with the cohort Rmin event, then repeat the literal CFL ladder.
