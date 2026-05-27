# Migration Notes for PF Params Schema 1

This repository now treats `Unit_Psedobinary.py` as the single source of truth for PF parameter derivation.

## What changed

- `pf_params_schema_version` is required in every generated `.params` file.
- GP eta inputs now use explicit `_code` and `_phys` fields:
  - `gp_W_eta_code`, `gp_W_eta_phys`
  - `gp_kappa_eta_code`, `gp_kappa_eta_phys`
  - `gp_L_eta_code`, `gp_L_eta_phys`
- Python now derives the eta interface parameters, kinetic reference values, and summary metadata before writing the override file.

## How to migrate

1. Regenerate the PF parameter file from physical inputs:

   ```bash
   python3 Unit_Psedobinary.py \
     --input-json physical_inputs.example.json \
     --output-pf-param-file /tmp/generated_pf.params
   ```

2. Pass the regenerated file to `main_cuda`:

   ```bash
   ./main_cuda --pf-param-file /tmp/generated_pf.params
   ```

3. Do not hand-edit old bare eta fields unless you are intentionally testing the transitional compatibility path.

## Notes

- Bare `gp_W_eta`, `gp_kappa_eta`, and `gp_L_eta` are transitional only.
- The C++ loader now rejects `.params` files that do not carry the schema version.
- If `t_real_unit`, `D_alpha`, or the eta code/phys pairs drift apart, rerun `Unit_Psedobinary.py` instead of patching the generated file by hand.
