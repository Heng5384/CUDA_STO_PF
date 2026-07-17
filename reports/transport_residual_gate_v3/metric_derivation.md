# V3 metric derivation

For each equal physical-time window, the frozen V2 observer provides `D_i=sum(dt*R_C,i)`, `A_i=sum(abs(Delta C_i))`, and the union-of-accepted-states interface mask. V3 adds:

`beta_global = abs(sum_i D_i) / max(sum_i A_i, A_floor)`

`beta_interface = abs(sum_{i in I_common} D_i) / max(sum_{i in I_common} A_i, A_floor)`

`beta_interface_excess = abs(S_candidate - S_strict) / max(sum_{i in I_common} A_i,strict, A_floor)`

The common mask is the union of strict and candidate frozen V2 masks. Own-mask values remain diagnostics. `b_signed` and `b_interface` are renamed sign-coherence diagnostics and are not hard gates. When `eta_interface <= 1e-6`, sign coherence is not materially interpretable.
