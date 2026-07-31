# Solver contract

The selected runtime state is
C_B_tot=(1-h(phi))*xB_alpha+h(phi)*v_B, xB_alpha=sigma(Y), v_B=1.

The phase interpolation is the quintic h(phi)=phi^3*(6*phi^2-15*phi+10).
The zero-mode branch uses SM_EXPLICIT_CONTEXT_N_V1 and SM_TANGENT_N_V1, with
all GP/source paths disabled. The scalar correction solves the total-mass
residual in FP64 and rejects non-finite or out-of-bound shifted fields.

This exact branch's zero-mode gate owns pf_composition_mode=legacy; the
ctot/qalpha experimental modes remain separate and are not silently claimed
as selected zero-mode evidence.
