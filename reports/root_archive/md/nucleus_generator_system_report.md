# Dynamic-Continue Production Nucleus Generator Refactor

Audit/refactor date: 2026-06-22

Constraint status:

- CNT theory: unchanged
- PF evolution equations: unchanged
- CUDA kernels: unchanged
- simulations: not rerun
- new layer added: `nucleus_generator/`

## 1. Old vs New Role

### Old role

Before this refactor, `dynamic-continue` functioned as a post-critical PF continuation stage:

```text
minimized nucleus VTK
  -> main_cuda --mode=dynamics-continue
  -> phi_*.vtk / xB_*.vtk / summary.txt
  -> optional faceted profile extraction
```

Its practical role was mixed:

- early-stage PF relaxation: yes
- smoothing/conditioning: yes
- physical evolution: yes
- direct CUDA insertion object: no

It could change:

- nucleus topology: yes, because it runs PF dynamics
- effective radius `r_eff`: yes, growth/shrinkage can occur
- composition field `xB`: yes, dynamics evolves/redistributes composition
- inferred barrier `DeltaG*`: not recomputed directly, but the post-critical object no longer represents the exact CNT saddle point

It could output a CUDA-readable VTK field, but not a complete CUDA insertion object. Current scheduled insertion expects profile/template data, not arbitrary raw minimized VTK.

### New role

After this refactor, the dynamic-continue pipeline is redefined as:

```text
dynamic-continue = physical source evolution
                 + production nucleus generation
                 + CUDA insertion preparation
```

The new production layer is:

```text
nucleus_generator/nucleus_resampler.py
```

It converts minimized or dynamic-continued nuclei into:

- `phi_init.npy`
- `xB_init.npy`
- `phi_init.vtk`
- `xB_init.vtk`
- `nucleus_metadata.json`

These are CUDA/PF initial-condition objects, not only analysis artifacts.

## 2. Current Role Analysis Answers

1. Is existing dynamic-continue only smoothing?

No. It is a real PF dynamics continuation step. It also smooths/conditions the nucleus, but because it runs `P.mode = 0` dynamics from VTK fields, it is physical evolution rather than a pure numerical filter.

2. Does it change topology, `r_eff`, `DeltaG*`, and `xB`?

- topology: can change
- `r_eff`: can change
- `xB`: can change
- `DeltaG*`: not directly updated; the object may move away from the original CNT critical state

3. Can it directly output CUDA-compatible `phi`?

It can output VTK fields that CUDA tooling can read, but the old workflow did not guarantee:

- exact `r_eff` preservation
- mass-conserving insertion object
- sub-grid protection
- one-object metadata contract
- direct CUDA initial-condition package

The new generator layer supplies those guarantees.

## 3. Required Transformation Implemented

New module:

```text
nucleus_generator/
  cuda_nucleus_builder.h
  nucleus_generator.cpp
  nucleus_resampler.py
  nucleus_generation_report.md
```

### Input

Implemented input fields:

- minimized or dynamic `phi` VTK
- optional `xB` VTK
- `r_eff_nm`
- target CUDA grid spacing
- optional target grid dimensions
- interface width
- composition bounds

Strain is not modified here. It remains metadata/selection context, because this layer is a geometric/field generator and must not alter elastic/CNT/PF physics.

### Processing

Implemented:

- resample source nucleus onto target CUDA grid
- detect source centroid from `h(phi)`
- enforce interface-width-controlled smooth reconstruction
- remove sub-grid artifacts through diffuse reconstruction when needed
- preserve effective `h(phi)` volume corresponding to `r_eff`
- preserve total `xB` by local matrix-shell compensation
- validate connectedness, bounds, smoothness, and volume/mass errors

### Output

Generated output:

```text
phi_init.npy
xB_init.npy
phi_init.vtk
xB_init.vtk
nucleus_metadata.json
```

This is the new CUDA-ready nucleus object.

## 4. Scale Bridging Method

The scale bridge is explicit:

```text
if r_eff / dx >= 3:
    direct resampling is attempted
    if effective h(phi) volume exceeds tolerance:
        rebuild smooth phi with exact equivalent r_eff volume

if r_eff / dx < 3:
    diffuse kernel reconstruction is used
    footprint is widened to the minimum resolvable grid scale
    amplitude is solved so h(phi) volume still matches r_eff
```

This resolves the previous ambiguity for sub-grid CNT nuclei. A nucleus with `r_eff = 0.18 nm` on `dx = 0.1 nm` is no longer treated as a literal two-cell hard sphere. It becomes a diffuse, grid-resolvable representation whose equivalent `h(phi)` volume is tied to the CNT radius.

## 5. Mass Conservation

The generator preserves:

```text
sum(xB_before) = sum(xB_after)
```

Method:

1. Load or synthesize target-grid `xB`.
2. Generate/reconstruct `phi`.
3. Compute total `xB` before reconstruction.
4. Apply a weighted local compensation in matrix/interface shell.
5. Clip to configured composition bounds.
6. Iterate until mass error is below tolerance or compensation capacity is exhausted.

Recorded metadata:

```json
"mass_conservation": {
  "total_xB_before": ...,
  "total_xB_after": ...,
  "total_xB_error": ...,
  "compensation_active_fraction": ...
}
```

## 6. CUDA Insertion Readiness Proof

The generated `nucleus_metadata.json` contains a Boolean:

```json
"insertion_ready": true_or_false
```

It is true only when:

- `phi` and `xB` are finite
- `phi` is bounded in `[0, 1]`
- the nucleus is connected under the recorded component threshold
- equivalent `h(phi)` volume matches requested `r_eff`
- total `xB` mass error is within tolerance
- smoothness proxy is below the configured gradient limit

The C/CUDA-facing contract is defined in:

```text
nucleus_generator/cuda_nucleus_builder.h
```

The lightweight checker:

```text
nucleus_generator/nucleus_generator.cpp
```

loads metadata and applies compatibility checks without requiring CUDA kernel changes.

## 7. PF Consistency

This refactor is PF-consistent because it does not introduce a new nucleation equation or alter existing evolution equations. It only constructs initial/insertion fields that satisfy the PF representation constraints:

- smooth diffuse interface
- bounded order parameter
- grid-consistent size
- local composition compensation
- connected nucleus support

It is therefore a representation bridge, not a new physics model.

## 8. Production Readiness Status

Status: **production-ready generator layer installed, pending runtime adoption**.

What is production-ready now:

- deterministic field generation
- sub-grid scale bridging
- mass-conserving `xB` compensation
- CUDA-facing metadata contract
- VTK and NumPy output
- insertion readiness checks

What still requires integration:

- `main_cuda.cu` does not yet directly load `phi_init.npy`/`xB_init.npy` as scheduled insertion templates.
- Existing scheduled insertion still primarily consumes `faceted_family_profiles.csv`.
- A runtime loader path must be added if the project wants to insert generated full-field objects directly instead of profile-derived templates.

Therefore, the system definition is upgraded, but runtime default behavior is not silently changed.

## 9. Final System Definition Change

Old:

```text
dynamic-continue = smoothing / relaxation step
```

New:

```text
dynamic-continue = physical source evolution + nucleus generation + CUDA insertion preparation module
```

More precisely:

```text
minimize output
  -> optional dynamic-continue PF evolution
  -> nucleus_generator/nucleus_resampler.py
  -> CUDA-ready phi_init/xB_init object
  -> runtime insertion or initial-condition loader
```

## 10. Final Classification

Closed-loop nucleation system:

- **Partially yes.** The representation gap from continuous/minimized nucleus to CUDA-ready field object is now closed.
- Full runtime closed loop still requires wiring `main_cuda.cu` to consume the generated full-field object directly.

Production-ready:

- **Generator layer: yes.**
- **End-to-end runtime default path: not yet.**

PF-consistent:

- **Yes.** The new layer preserves PF field constraints and does not modify PF equations.

Bottom line:

```text
The project now has a production-ready nucleus generation layer.
It converts minimized/dynamic nuclei into grid-consistent, mass-conserving CUDA/PF initial-condition objects.
The remaining step is runtime loader integration if full-field insertion should replace profile-based scheduled insertion.
```
