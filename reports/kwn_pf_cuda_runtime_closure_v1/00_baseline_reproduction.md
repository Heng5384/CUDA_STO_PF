# 00 Baseline reproduction

Baseline host evidence was reproduced before modifying the KWN transport solver. The frozen validation identity is `d0ff02973ab0f737043e1a40d4f69893a469cbfe2bc4cd22f9e6a410bd0b1333` on fixture `f1247cb66419af764b97de2f7843fc6de2049459d78550bc603edd8e88d9134f`.

- Host KWN suite: 22 tests passed.
- Coupling suite: 18 tests passed.
- Thermodynamic parity maximum relative error: `3.657021380996818e-12`; solvus maximum absolute error: `0`.
- Zero-aux remap difference: `4.336808689942018e-18`; conditioned four-bucket residual: `1.4878348096565177e-16`.
- Historical strict beta-only failure was reproduced at step `87`, `0.39317699499770825 h` with zero inventory residual.

The baseline failure is preserved as historical pre-repair evidence; it is not overwritten by the current solver qualification. Source artifacts are under `outputs/kwn_pf_cuda_runtime_closure_v1/baseline/` and the top-level failure diagnostic artifacts.
