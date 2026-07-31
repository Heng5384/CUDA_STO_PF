# PF elastic diagnostic restart audit

`PASS_PF_ELASTIC_DIAGNOSTIC_AND_FIELD_RESTART_V1`

The continuous diagnostic trajectory is compared to the concatenation of the
first-half and restarted legs at every registered absolute step.  Elastic
mean/max, hydrostatic-stress extrema, mass diagnostics, final phi/xB/xBtot VTK
fields and the complete final checkpoint are required to match exactly.
