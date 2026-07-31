# Low-s GP-assisted beta smoke results

## Scope and frozen execution

The `s_GP=0.01` low-bound diagnostic was submitted as serial Slurm array
`71156` (T380 task 0, T400 task 1; one GPU task at a time) with aggregate
tail `71157`.  The exact task and tail markers are respectively
`PASS_LOW_S001_CLUSTER_TASK_V1` and `PASS_LOW_S001_CLUSTER_TAIL_V1`; all jobs
completed with exit code zero and no stderr.  The executable was the frozen
r22 binary:

```text
source: /data/home/luozhiheng/codex_jichen_fixed_dt_low_s_v1_20260722_r22
main_cuda SHA-256: 7305b50c0e77c7458b3d160bcbd62212efffea2b6050b490cc6e0f3df51d015d
source-manifest SHA-256: 19c270c1d45d8ce635f09b0b4cfc0da79c18861d50692cb5c5bbf4ed3880d5a0
output: /data/home/luozhiheng/jichen_low_s_smoke_r22c_20260722
```

Both cases used the real 15,729-object after-quench population, selected
`dt_code=0.02`, fixed global barrier reference `xB=0.03`, quiet mode off,
and release off.  Each ran a continuous 100-step leg plus a 50-step
checkpoint/restart leg.  The raw `phi`, `Y`, `xB`, `dY_dt_prev`, and `xBtot`
fields, final checkpoint, and final ledger were byte-identical across the
continuous/restarted comparison.

## Observed structural transaction

| T | physical dt (s) | handoff step | GP/site id | atomic GP debit (m³) | final relative system drift |
|---|---:|---:|---:|---:|---:|
| T380 | 0.9909260953431841 | 91 | 15533 | 8.689976500400955e-25 | 5.4304220748480144e-11 |
| T400 | 0.8225917084910954 | 91 | 15533 | 5.898725086352716e-25 | 9.621994329507444e-11 |

The independent atomic-event auditor passed at both temperatures.  It records
one resolved target handoff, exact GP debit equal to the hash-pinned target
requirement, a closed handoff inventory, a positive resolved-beta inventory,
and no authoritative `Ctot`, global mass projection, direct beta injection,
matrix reset, or runtime-local barrier composition.

## Required interpretation

The analytical prescreen classifies `s_GP=0.01` as `BURST` for the full
15,729-site population.  The frozen r22 CUDA entry point, however, explicitly
selects the lowest-threshold recovered site and initializes
`std::vector<Candidate>(1U, beta_candidate)`.  Thus this run is a successful
**one-candidate structural smoke** only:

```text
runtime_candidate_limit=1
real_GP_population_count=15729
quantitative_population_burst_verified=false
classification=ANALYTICAL_BURST_NOT_QUANTITATIVE_POPULATION_VALIDATION
```

It proves the selected existing hazard → embryo/bridge → atomic-target
handoff and restart path for ID 15533.  It does **not** prove simultaneous
multi-site burst dynamics, the full site-event distribution, or a physical
production result.  In addition, this r22c smoke used the legacy unbounded
deferred-queue selector (`gp_max_deferred_events=0`), so it is not the
explicit bounded committed-event safety-cap diagnostic required before a
population-burst case can be called Stage-6 complete.

The earlier r22/r22b submission roots are preserved as zero-step script
contract failures; they did not enter PF evolution and were not used here.

`low_s_smoke_status=PASS_STRUCTURAL_SINGLE_CANDIDATE_ONLY`

`stage6_population_burst_status=NOT_QUALIFIED_RUNTIME_SINGLE_CANDIDATE_AND_NO_EXPLICIT_BOUNDED_CAP`

`s001_physical_production_claim=false`
