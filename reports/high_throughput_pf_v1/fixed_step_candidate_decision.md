# Fixed-step candidate decision

The selected fixed-step production candidate is:

```text
transport gate = G9 = 1e-9
dt_code = 7.8125e-4 (dt4)
dt_physical = 0.032132488612 s
```

It is the fastest tested candidate with a complete, unseen V3 holdout and all
registered equal-time QoIs. Across the long window it recorded retry and
fallback fractions of `2.5e-4`, retry wall overhead `2.1255e-4`, maximum one
consecutive fallback, and no registered persistent repeated failure cell.

The `G10/dt4` row is faster in the 1-D workload but is rejected because cell
246 recurs in the failure history. Other numerically acceptable rows are not
promoted because they lack the new V3 holdout required by the registered
contract. No result from a 512x1x1 run is used as a 400-cube timing estimate.

This decision qualifies the numerical interval for later 3-D benchmarking; it
does not itself establish 400-cube throughput.

`selected_fixed_dt_status=PASS_FIXED_G9_DT4_WITH_V3_HOLDOUT`
