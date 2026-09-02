# Analytic transport benchmarks

Status: `PASS_ANALYTIC_INTERIOR_TRANSPORT_BENCHMARKS_WITH_BOUNDARY_CROSSING_DIAGNOSTICS`.  The no-boundary positive translation and frozen-matrix controls run at low Courant with maximum interior metric error `3.098357e-09`.

The negative-velocity and -K/R controls use a unit-normalized first canonical cell, so every reference characteristic physically crosses the unchanged Rmin edge.  Their number loss, edge-volume return, mol-B return, event-time range, low-tail PSD/CDF and Wasserstein diagnostics are recorded in the CSV as boundary evidence; they are not mislabelled as a passing interior-transport gate.
