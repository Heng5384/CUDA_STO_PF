# Next Workorder Recommendation

## Option A: Fix phi/eta iteration asymmetry

- Risk: medium
- Workload: medium
- Verification complexity: medium

Why:

- The core semi-implicit update algebra is already symmetric.
- The remaining asymmetries are mostly intentional feature plumbing, so a deeper iteration refactor should be limited to a careful audit of the minimizing / projection loops and any places where phi gets extra correction logic.

## Option B: Clean diagnostic switches and archive STEP reports

- Risk: low
- Workload: medium
- Verification complexity: low

Why:

- The audit found a large cluster of audit/debug/test-only switches that are still live in `pf_params.h` and `main_cuda.cu`.
- That work reduces noise, improves discoverability, and gives a cleaner base before any further physics refactor.

## Recommendation

Do **B first**.

Reason:

The iteration-level physics is already structurally aligned, while the codebase still carries a lot of diagnostic surface area. Cleaning that up first lowers cognitive load and makes any later phi/eta refactor easier to verify.
