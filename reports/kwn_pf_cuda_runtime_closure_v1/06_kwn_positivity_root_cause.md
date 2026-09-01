# 06 KWN positivity root cause

The pre-repair failure is `REPRODUCED_STRICT_POSITIVITY_FAILURE` at `0.39317699499770825 h` (step `87`), beta radius-bin `18`.

It is classified as **donor-bin outgoing size-space face-flux overdraw**: proposed `dt=0.03271980767696527` s exceeded the donor bound `0.01400427765209778` s, with raw positivity utilization `2.33641523610209`. The candidate density changed from `0.1241169802731471` to `-0.1225814801775817` m⁻⁴.

This is not a lower-radius boundary failure, upper-radius boundary failure, source/nucleation overdraw, matrix inverse failure, or floating-point-roundoff-only event: beta and GP nucleation rates were zero, the lower-boundary flux was `0`, the upper-boundary flux was `1.412596025373013e-268`, and the ledger residual was `0`.
