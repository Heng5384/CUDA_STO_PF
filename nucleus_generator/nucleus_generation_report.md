# Nucleus Generator Module Report

This directory adds a production-oriented nucleus generation layer without changing CNT theory, PF evolution equations, or CUDA kernels.

## Added Files

- `nucleus_resampler.py`: builds CUDA-ready `phi_init` and `xB_init` fields from minimized or dynamic nucleus VTK fields.
- `cuda_nucleus_builder.h`: defines the C/CUDA-facing nucleus metadata contract.
- `nucleus_generator.cpp`: lightweight metadata compatibility checker for generated nucleus objects.

## Input Contract

Required:

- minimized or dynamic nucleus `phi` VTK
- effective radius `r_eff_nm`
- target CUDA grid spacing
- output directory

Optional:

- `xB` VTK
- target grid dimensions
- interface width
- matrix composition
- composition bounds

## Processing Contract

The generator performs:

1. source VTK loading
2. source centroid detection by `h(phi)` weight
3. target CUDA-grid reconstruction
4. explicit scale bridging:
   - `r_eff / dx >= 3`: direct trilinear resampling is attempted
   - `r_eff / dx < 3`: diffuse kernel reconstruction is used
5. equivalent `h(phi)` volume enforcement against `4/3*pi*r_eff^3`
6. `xB` mass compensation in a local matrix shell
7. smoothness, connectedness, bounds, and volume/mass checks

## Output Contract

Each generated object contains:

- `phi_init.npy`
- `xB_init.npy`
- `phi_init.vtk`
- `xB_init.vtk`
- `nucleus_metadata.json`

The metadata records:

- grid shape and spacing
- `r_eff_nm`
- `r_eff_over_dx`
- scale mode
- interface width
- target and generated effective volume
- mass conservation error
- field extrema
- smoothness proxy
- connected component count
- insertion readiness boolean

## CUDA Compatibility Definition

A generated nucleus is considered insertion-ready when:

- fields are finite
- `phi` remains in `[0, 1]`
- generated nucleus has one connected component under the recorded threshold
- effective `h(phi)` volume matches the requested `r_eff`
- total `xB` is conserved within tolerance
- maximum `phi` gradient is below the configured smoothness proxy

This is a preflight/template-generation check. It does not change runtime PF equations or CNT energetics.

## Example

```bash
python3 nucleus_generator/nucleus_resampler.py \
  --phi-vtk Results/workflows/T400_xB0p030/raw/.../phi_final_case.vtk \
  --xb-vtk Results/workflows/T400_xB0p030/raw/.../xB_final_case.vtk \
  --r-eff-nm 2.84 \
  --dx-nm 0.1 \
  --output-dir Results/generated_nuclei/T400_xB0p030_case
```

For a sub-grid nucleus:

```bash
python3 nucleus_generator/nucleus_resampler.py \
  --phi-vtk CNT_SCAN_WORKSTATION_RUN/.../phi_final_case.vtk \
  --r-eff-nm 0.18 \
  --dx-nm 0.1 \
  --output-dir Results/generated_nuclei/subgrid_case
```

This takes the diffuse reconstruction path because `0.18 / 0.1 < 3`.
