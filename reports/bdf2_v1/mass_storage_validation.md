
# BDF2 Mass and Storage Validation

For source-free transport,

```text
3 M_np1 - 4 M_n + M_nm1 = 0.
```

The selected dt/8 qualification emitted 999 BDF2 identity rows. Every row
passed; maximum absolute identity defect was `1.4495071809506044e-12` and
maximum relative defect was `1.043966997575569e-14`.

Across 1000 accepted selected steps:

- maximum source-free mass error: `4.83169060316868126e-13`;
- maximum transport residual: `9.99467884145846809e-13`;
- maximum phase projected KKT: `9.82295356166673628e-11`;
- clipping count: zero;
- physical projection count: zero;
- storage reconstruction and bounds: PASS;
- one transport and one phase solve per accepted step.

The phase substep holds final `Ctot` fixed. Conserved storage remains
`Ctot=h*v_B+(1-h)*xB_alpha`; no domain-wide mass correction is used.

`BDF2_mass_status=PASS`
`BDF2_storage_status=PASS`
`BDF2_KKT_status=PASS`
