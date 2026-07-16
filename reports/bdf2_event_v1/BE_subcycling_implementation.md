# Atomic BE subcycling implementation

## Contract

`BDF2_EVENT_BE_SUBCYCLING_V1=1` is default-off.  At the start of every macro
step, private device buffers snapshot authoritative `Ctot`, `phi`, and the
derived `Y` context.  An event first attempts two Lie-BE half steps.  If any
substep fails an unchanged transport, phase KKT, mass/storage/bounds,
energy/work, or mechanics gate, the whole macro state is restored and the
transaction retries with four quarter steps, then at most eight eighth steps.

Intermediate substeps may advance private trial/working fields, but they do
not commit:

- macro accepted time,
- macro accepted-step count,
- BDF2 `n-1` history,
- checkpoint/output state.

If depth 8 fails, the macro start is restored and the run hard-rejects.  There
is no partial commit, clipping, physical projection, or hidden tolerance
change.  The sum of successful substep times is exactly the original macro
`dt`.

The same mechanism is entered when preflight detects an event or when an
otherwise selected BDF2 attempt rejects. It also covers a rejected forced-BE
history-rebuild step: that step is part of the event recovery transaction and
must not escape the 2/4/8 rollback policy. All substeps use the existing
method-consistent Lie-BE transport-first operator with unchanged physical and
solver parameters.

## Diagnostics

Runtime markers include:

- `CTOT_BDF2_EVENT_PREFLIGHT`
- `CTOT_BDF2_EVENT_CELL`
- `CTOT_BDF2_EVENT_SUBSTEP_ACCEPT`
- `CTOT_BDF2_EVENT_SUBCYCLE_RETRY`
- `CTOT_BDF2_EVENT_MACRO_READY`
- `CTOT_BDF2_EVENT_SUBCYCLE_EXHAUSTED`

## Workstation validation

The frozen dt/8 and dt/16 failures both completed 502 accepted macros with no
hard reject. Maximum accepted depth was 2. The event transaction therefore
passes its correctness contract, but not its dt/8 production-efficiency goal:
dt/8 required 257 event-subcycled macros and only 6 BDF2 macros. See
`event_crossing_validation.md`; this limitation is not hidden as a PASS.
