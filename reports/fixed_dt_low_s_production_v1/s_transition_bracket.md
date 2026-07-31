# s_GP transition bracket after the low-s smoke

The fixed global-`xB=0.03` analytical prescreen remains the only valid
population-level bracket.  It uses the recovered 15,729-site population,
fixed historical thresholds, and no runtime-local composition in the barrier.
No traceable experimental onset time was recovered, so these are engineering
sensitivity brackets, not a fitted material calibration.

| T | burst side | analytical transition | no-event side |
|---|---:|---:|---:|
| T380 | `s=0.075` | `s=0.1125` | `s=0.16875` |
| T400 | `s=0.05` | `s=0.075` | `s=0.1125` |

At `s=0.01`, the full-population analytical expected events per selected
physical timestep are `2.2155e4` (T380) and `9.9414e3` (T400), with all
15,729 sites crossed over the registered engineering horizon.  It is therefore
too strong for a long PF production case.

The completed r22c low-s structural smoke confirms one first-threshold site
and one atomic handoff at each temperature, but it cannot refine this bracket:
the frozen CUDA setup retains exactly one candidate despite the full
population.  A quantitative common-T380/T400 bracket, bisection, and
Stage-9 ensemble must wait for a bounded, checkpointed multi-candidate
runtime.  It must preserve the existing event state machine and use the
existing `Runtime::initialize_candidates(vector<Candidate>)` capability;
creating a second beta-event pipeline is prohibited.

`experimental_comparison_time_recovered=false`

`common_quantitative_s_envelope=NOT_ESTABLISHED`

`next_runtime_gate=BOUNDED_MULTI_CANDIDATE_HAZARD_AND_ATOMIC_HANDOFF`
