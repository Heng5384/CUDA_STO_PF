# Runtime elastic-energy scalar audit

`PASS_RUNTIME_ELASTIC_SCALAR_EXTRACTED`

The scalar is taken from the solver's `mean_elastic_energy` and `max_elastic_energy` diagnostic columns. These are `gel_hat` values when `elastic_gel_is_dimless=1`; no zero is substituted. A physical J/m3 conversion is emitted only when the run's audited `12*gamma/lambda_sm` scale is supplied.

Rows in the 15--16 h physical window: 0.
