# Common active-set event

## Result

Both trajectories reach the same beta-side matrix-capacity region.  The event
is the lower storage constraint `q_alpha=Ctot-h(phi)*v_B=0`, not a composition
upper bound, phi endpoint overshoot, mobility sign change, or mass loss.

| Case | Cell | q_n | q_anchor_E | Context invalid |
|---|---:|---:|---:|---:|
| dt/8 | 266 | 1.62936331093987974e-12 | -2.22927232229608308e-11 | 1 |
| dt/16 | 246 | 3.33066907387546962e-16 | -9.32587340685131494e-15 | 0 |

The affected indices span 246-266 on the two periodic beta-side interfaces.
The dt/8 extrapolated anchor crosses the hard context tolerance materially;
the dt/16 anchor is only a few ulps below zero and passes the existing 1e-12
context tolerance, but its endpoint trial reaches the same capacity branch.

`same_active_set_event=true`

`event_transition_type=FREE_TO_QALPHA_LOWER_CAPACITY`
