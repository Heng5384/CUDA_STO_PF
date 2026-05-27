# Smoke Test: Unit Conversion

Run:

```bash
bash scripts/smoke_test_unit_conversion.sh
```

What it checks:

- a clean Python-generated `.params` run
- `t_real_unit` drift failure
- `gp_L_eta_code` / `gp_L_eta_phys` mismatch failure
- missing `pf_params_schema_version` failure
- legacy bare `gp_W_eta` compatibility path

If any case fails, paste the stderr block from that case back to me.

Expected runtime:

- each one-step case should finish in under 5 seconds on a normal GPU workstation
- the script itself should be short-lived once `main_cuda` is already built
