# Performance preflight

The first T380 pilot segment (3,633 steps, 128^3, elastic OFF) with a mass
diagnostic every step measured:
total_wall_s=443.946275
avg_wall_s_per_step=0.122198
zero_mode_wall_s=191.929654263
checkpoint_wall_s=0.110368550

The selected target is within the 24 h cluster walltime. GPU process and
checkpoint output were present during the run; no duplicate target was
submitted.

The later diagnostic AB qualification measured:

```
diag_off             0.005325 s/step
diag_interval_256    0.005819 s/step
diag_interval_1      0.122457 s/step
```

The interval-256 and interval-1 continuous/restart final checkpoints were
bytewise equal. The accelerated 12 h→48 h continuation used interval 256 and
measured 0.0055332 s/step mean (0.005612 s/step maximum).
