# Lower-bound number and B-inventory flux audit

Status: `FAIL_LOWER_BOUNDARY_OPERATOR_PARITY`.  The Eulerian FV face currently uses the cell-centre growth velocity, while the cohort continuum extrapolation follows the frozen physical Rmin characteristic.  Number/volume/mol-B flux relative errors are `1.000000e+00`, `1.000000e+00`, and `1.000000e+00` respectively.

The legacy fixed-pivot/Rmin volume-price difference is `2.513353e-03` and is retained only as a historical reporting diagnostic: the current beta flux tally uses the Rmin price on both sides.  The discrete cohort event stream is separate from this instantaneous continuum extrapolation. Number loss, beta-volume loss and mol-B return are reported as distinct quantities.
