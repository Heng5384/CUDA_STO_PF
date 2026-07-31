# Operator order

1. Read accepted phi_n,Y_n, save rollback fields and history.
2. Evaluate explicit chemical/mobility coefficients at the accepted phase
   context and advance the non-zero Fourier modes.
3. Form the tangent reaction contribution from
   h'(phi_n)*(phi_(n+1)-phi_n)*(v_B-xB_alpha,n)/dt.
4. Apply the FP64 host Newton/bisection zero-mode shift to Y_star.
5. Validate bounds and finite state; reject/restore on failure.
6. Commit fields, history, diagnostics, and (at the registered cadence) an
   atomic checksummed checkpoint.

No physical global mass projection, clipping, sparse matrix, or implicit
spatial solve is used.
