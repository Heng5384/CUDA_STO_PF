# BDF2 history rebuild contract

An event-subcycled macro step never extrapolates across the nonsmooth
active-set transition.

After the macro transaction succeeds:

1. The event endpoint is committed once at the original macro time.
2. `bdf2_history_valid=0`.
3. `bdf2_restart_fallback_pending=1`.
4. The last fallback reason records the accepted subcycle depth.
5. The next macro step is forced to use Lie-BE.
6. That accepted BE step commits a clean same-dt `n/n-1` pair.
7. Only the following macro step may resume BDF2.

If the forced history-rebuild BE solve rejects, it is atomically retried with
the same 2/4/8 event subcycling ladder. Its endpoint still invalidates history,
so the following macro again performs the required clean BE rebuild.

Checkpoint metadata records both event selectors, the versioned event reason,
accepted subcycle depth, history-valid bit, pending rebuild bit, current and
previous dt, accepted macro step/time, and both history fields.

For event-enabled restart, a checkpoint with a complete valid history pair is
allowed to run the same preflight immediately.  The legacy default-off restart
continues to force its historical first-step BE fallback.  A V1 endpoint-energy
checkpoint may migrate to the V2 audit because the physical state/history
contract is unchanged; the migration changes only the diagnostic algebra.

The workstation `2+2` restart replay is bitwise identical to the uninterrupted
four-macro trajectory for `Ctot`, `Ctot_nm1`, `phi`, `phi_nm1`, and
`xB_alpha`. The dt/8 moving interface nevertheless produces a persistent
sequence of new capacity events after each rebuild. This is a method/step-size
qualification failure, not a history corruption: the implementation does not
reuse pre-event history and each occasional safe BDF2 resume is explicit in the
runtime log.
