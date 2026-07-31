# Runtime projection

The first diagnostic-every-step segment measured 0.122198 s/step, which
would project to about 5.2 h for the full chain. The validated diagnostic AB
test then measured 0.005819 s/step at interval 256 (0.005325 s/step with the
diagnostic disabled), and the 12 h→48 h continuation measured a mean
0.0055332 s/step. The completed accelerated continuation is therefore about
22.1x faster than the diagnostic-every-step path.

The acceleration changes only diagnostic cadence; the AB continuous/restart
checkpoint comparison is bytewise equal. The continuation used a unique
non-overwriting output root and wrote an exact checkpoint every physical hour.
