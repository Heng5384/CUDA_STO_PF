# Registered future 6–48 h plan

This file freezes a future production plan only. The present goal does not
start it.

| experimental age | absolute step | actual registered age |
|---:|---:|---:|
| 6 h | 0 | 6 h |
| 12 h | 21798 | 12.0000575073 h |
| 18 h | 43596 | 18.0001150146 h |
| 24 h | 65393 | 23.9998972647 h |
| 36 h | 108989 | 36.0000122793 h |
| 48 h | 152585 | 48.0001272939 h |

If and only if the three 6–8 h screenings pass, a future authorized
production run must:

- use three independent, non-overwriting output roots;
- retain the same source, binary, parameter, fixture, and analysis
  provenance;
- use periodic checkpoints and the frozen `dt_code=0.02`;
- keep GP, external sources, and new beta nucleation disabled;
- retain initial post-handoff reconstruction as physical time;
- fail closed on unresolved merge/split events;
- confirm available GPU time and disk space before submission.

Explicit user authorization is still required before any 6–48 h production
trajectory is launched.
