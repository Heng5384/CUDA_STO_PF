
# BDF2 Endpoint Discrete-Work Contract

`BDF2_energy_contract_version=BDF2_ENDPOINT_DISCRETE_WORK_V1`

This is a method-consistent endpoint work identity, not a copied Lie-BE
monotonicity predicate. Define `dC_n=C_np1-C_n`, `dC_m=C_n-C_nm1`, and the
analogous phase increments. Multiplying the BDF2 transport equation by the
endpoint chemical potential gives

```text
<mu,dC_n> + (2/3)dt D_C
 - (1/3)<mu,dC_m> - (2/3)<mu,r_C> = 0,
D_C = <G mu, M_f G mu> >= 0.
```

Multiplying the fixed-final-C phase equation by its energy gradient gives

```text
<g,dphi_n> + (2/3)dt L_phi<g,g>
 - (1/3)<g,dphi_m> - (2/3)dt<g,r_phi> = 0.
```

The nonlinear endpoint energy is closed exactly with transport and phase
chain-rule remainders, explicit-context work
`<mu(C_np1,phi_n)-mu(C_np1,phi_E),dC_n>`, and final mechanics work. The hard
gate is the normalized residual of this complete identity plus nonnegative
transport/phase dissipation; unexplained energy increase cannot pass.

| Scenario | Grid | Elastic | BDF2 work rows | Max balance rel | Status |
|---|---:|---:|---:|---:|---|
| manufactured_smooth_elastic_off | 32x32x32 | False | 3/3 | 2.192e-22 | PASS |
| manufactured_weak_smooth_elastic_on | 16x16x16 | True | 3/3 | 1.062e-23 | PASS |
| stationary_interface_elastic_off | 32x32x32 | False | 3/3 | 8.097e-34 | PASS |
| planar_growth_elastic_off | 512x1x1 | False | 3/3 | 4.649e-24 | PASS |
| planar_dissolution_elastic_off | 512x1x1 | False | 3/3 | 1.001e-24 | PASS |

The elastic-on smooth case has nonzero mechanics work
(`4.806354012407667e-12` maximum absolute). A non-elastically-equilibrated
stationary slab was separately rejected by the unchanged final KKT/mechanics
gate and is not counted as accepted evidence.

`BDF2_energy_work_status=PASS_BDF2_ENERGY_WORK_SCENARIOS`
