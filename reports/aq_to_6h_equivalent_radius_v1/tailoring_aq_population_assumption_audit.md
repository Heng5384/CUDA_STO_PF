# Tailoring AQ population assumption audit

T-A uses the TEM/APT total density as one monodisperse diagnostic and scans its radius; it cannot yield a unique conductivity because Tailoring reports no unique AQ radius.  `HOST_YU_AQ_FIXED` retains the Yu AQ host including xAg=0.0069.  `HOST_YU_COMMON_WITH_TAILORING_MATRIX_UPDATED` changes only the point-defect xAg to the Tailoring AQ value 0.0078; all other host fields remain Yu inputs and are therefore explicitly cross-paper diagnostics.

T-B applies the 37/43 and 6/43 counts only as a single-APT-reconstruction fraction.  Large-object length is not treated as a sphere radius.  It is bracketed by a deliberately nonphysical L/2 sphere and a prolate-spheroid volume-equivalent radius for AR=2,5,10.

T-C is a separate micro-scale FIB diagnostic.  It is never added to the TEM/APT density, preventing double counting.
