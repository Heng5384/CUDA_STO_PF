# Internal reject forensics

Every old dt/4 and dt/8 reject is represented in `internal_reject_catalog.csv`; the compact rows retain predicate, frozen source line, integrator, depth, history state, full residual history, worst cell/face state, phase KKT, energy and recovery outcome.

### dt4

- Count: **15**.
- Categories: `{'J_OTHER_PROVEN_CAUSE': 1, 'C_TRANSPORT_LINE_SEARCH_STAGNATION': 9, 'B_TRANSPORT_NONLINEAR_ITERATION_LIMIT': 5}`.
- Worst-cell repetitions: `{'279': 1, '244': 3, '266': 3, '245': 3, '267': 3, '268': 1, '278': 1}`.
- Maximum consecutive rejected macros: **5**.
- Same macro recovered: **True**.
- History-valid rejects: **10/15**.

### dt8

- Count: **16**.
- Categories: `{'B_TRANSPORT_NONLINEAR_ITERATION_LIMIT': 12, 'C_TRANSPORT_LINE_SEARCH_STAGNATION': 4}`.
- Worst-cell repetitions: `{'244': 12, '267': 3, '243': 1}`.
- Maximum consecutive rejected macros: **2**.
- Same macro recovered: **True**.
- History-valid rejects: **15/16**.

The single dt/4 step 1283 row is `J_OTHER_PROVEN_CAUSE`: the transport and phase solves completed, but the method cold residual `1.00093e-12` marginally exceeded the unchanged outer cold gate. It is not a nonlinear iteration or line-search failure. All other rows are transport globalization failures; no nonfinite, bound, mobility, phase-KKT, mechanics, or energy-work predicate failed first.
