# Physically Normalized Transport Defect Metrics

Contract: `PHYSICALLY_NORMALIZED_TRANSPORT_DEFECT_GATE_V2`.

For accepted method-context transport residuals only,

\[
D_i^{(k)}=\sum_{n\in k}\Delta t_nR_{C,i}^{(n)},\qquad
A_i^{(k)}=\sum_{n\in k}|C_i^{n+1}-C_i^n|.
\]

The observer also records

\[
B_i^{(k)}=\sum_{n\in k}|\Delta t_nR_{C,i}^{(n)}|,
\]

which is used only to form a reproducible floating-point summation error bound.
Rejected trials and rolled-back event subcycles contribute to none of these
arrays.

## Frozen interface contract

At every accepted final state, a cell belongs to the instantaneous interface
band when

\[
10^{-4}\le h(\phi_i)\le1-10^{-4},\qquad
h(\phi)=\phi^3(6\phi^2-15\phi+10).
\]

The window interface set is the union of this band over all accepted states in
the window. This captures a moving interface without selecting a mask after
seeing the result.

## Derived quantities

\[
D_{\rm peak}^{(k)}=\max_i|D_i^{(k)}|,
\quad
\eta_{\rm global}^{(k)}=
\frac{\sum_i|D_i^{(k)}|}{\max(\sum_iA_i^{(k)},A_{\rm floor})}.
\]

For the frozen interface union \(I_k\),

\[
\eta_{\rm interface}^{(k)}=
\frac{\sum_{i\in I_k}|D_i^{(k)}|}
     {\max(\sum_{i\in I_k}A_i^{(k)},A_{\rm floor})}.
\]

Let \(A_{\rm scale}^{(k)}=\max_iA_i^{(k)}\). Materially evolving cells obey
\(A_i^{(k)}\ge10^{-6}A_{\rm scale}^{(k)}\), and

\[
\eta_{\rm cell,max}^{(k)}=
\max_i\frac{|D_i^{(k)}|}
{\max(A_i^{(k)},10^{-6}A_{\rm scale}^{(k)})}.
\]

Signed fractions are

\[
b_{\rm signed}^{(k)}=
\frac{|\sum_iD_i^{(k)}|}{\max(\sum_i|D_i^{(k)}|,D_{\rm floor})},
\]

with the analogous expression over \(I_k\). The local peak-rate diagnostic is
\(r_{\rm peak}^{(k)}=D_{\rm peak}^{(k)}/\Delta T_k\).

For `N` cells, both declared denominator floors are fixed before holdout data:

\[
A_{\rm floor}=D_{\rm floor}=\max(10^{-30},100\epsilon_{64}N).
\]

They prevent zero-denominator classifications and do not alter any simulation
field.
