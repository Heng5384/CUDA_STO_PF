# Largest safe grid

PF-only `400^3` is the largest requested grid and is admitted: static source allocation
12.182601 GiB and measured peak 12.900000 GiB
(83.44% of the 16,303 MiB device).  This satisfies the hard 85% gate,
though not the preferred 80% engineering target.

Elastic and elastic+GP layouts were not run: the current accepted baseline is elasticity OFF and the goal
forbids opening GP/S3.  Their status is **GATED_NOT_MEASURED**, not inferred from PF-only data.
