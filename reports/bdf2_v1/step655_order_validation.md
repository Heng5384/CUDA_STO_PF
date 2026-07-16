
# Step655 Consistent-History BDF2 Order Validation

Every dt starts from the same frozen accepted state, independently performs
one BE startup, then eight coarse-step-equivalent BDF2 warmup steps. Startup is
excluded from the four-coarse-step measurement interval.

| Observable | ratio dt/2 to dt/4 | ratio dt/4 to dt/8 |
|---|---:|---:|
| Ctot field L2 | 3.974272 | 3.994950 |
| Ctot increment | 4.356008 | 4.126735 |
| phi field L2 | 3.956231 | 3.993339 |
| phi increment | 3.732925 | 3.971365 |
| h-volume | 4.008293 | 4.011069 |
| h-volume increment | 3.582408 | 3.970314 |

All registered ratios are within `[3,5]`; short-window errors are below the
1-2% gate and all method hard gates pass. The original dt was independently
rerun after the active-set fix and still rejected at step 7, so it is not an
eligible production candidate.

`step655_second_order_status=true`
