# Hybrid L2/Linf merit validation

The far/near/very-near gate regions preserve the frozen merit contract, and
unit tests pass. Runtime V3 delays the dt/4 first terminal failure from step
1325 to step 1380 but does not complete the trajectory. V4/V5 can complete some
diagnostic trajectories only with fallback. The final cold Linf gate is never
relaxed. Status: `IMPLEMENTED_DEFAULT_OFF_NOT_SUFFICIENT`.
