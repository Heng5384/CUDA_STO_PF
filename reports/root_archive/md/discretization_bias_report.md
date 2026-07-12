# Discretization Bias Report

## Scope

This report quantifies the currently measurable bias introduced by:

```text
continuous CNT / minimization / selector nucleus
  -> discrete CUDA template mapping
```

The current audited CUDA case is:

```text
Results/orchestrator/T400_xB0p050
```

## Generated Data Products

- `continuous_physics_summary.csv`
- `cuda_execution_summary.csv`
- `physics_consistency_delta.csv`

## Available Measurements

### Continuous prediction

Selected continuous nucleus:

- case id: `cntcon_T400_xB0p050_strictref_eyy_sm0p01`
- `T`: 400 C
- `xB`: 0.05
- strain: -0.01
- continuous `rc`: 1.8245556046158986 nm
- continuous shape: spherical
- continuous barrier: 291.9433447129304 kBT

### Discrete CUDA template

Mapped template:

- template id: `generated_cntcon_T400_xB0p050_strictref_eyy_sm0p01_rc1p825`
- effective CUDA semiaxes mean radius: 1.8540789999999998 nm
- inserted shape: spherical
- source/profile: generated intermediate CUDA-compatible template

## Bias Metrics

| Metric | Value | Status |
| --- | ---: | --- |
| rc error | 0.029523395384101203 nm | measured |
| rc relative error | 1.6181143128447655% | measured |
| shape mismatch score | 0.0 | measured |
| mapping loss total | 0.036904244230126504 | measured |
| mean barrier shift | not available | no CUDA observed barrier |
| max ordering inversion rate | not available | only one discrete case, no observed ordering |
| shape projection loss | 0.0 | measured for selected case |
| nucleation rate distortion factor | not available | no CUDA observed rate proxy for mapped insertion |

## Bias Interpretation

Geometry bias is low for the selected case:

- shape is preserved;
- semiaxes are preserved by generated template;
- effective radius differs by only about 1.62%.

Full physics bias is not bounded:

- CUDA launch status is `not_requested`;
- no scheduled insertion trajectory was found for this mapped template;
- no observed CUDA barrier or rate proxy exists for this case;
- no continuous-vs-discrete ranking comparison can be performed.

## Dominant Error Source

The dominant uncertainty is not the radius projection. It is the lack of a dynamically validated CUDA barrier/rate measurement for the generated template.

Secondary uncertainty:

- generated profile uses an analytic tanh profile, not a relaxed VTK-derived faceted profile;
- current template library is sparse, so broader shape hierarchy preservation is not proven;
- GP-assisted vs non-GP preference is not tested on the same mapped-nucleus case.

## Bias Level

Geometry-only bias: **low**.

Full nucleation-physics bias risk: **high / unbounded with current data**.
