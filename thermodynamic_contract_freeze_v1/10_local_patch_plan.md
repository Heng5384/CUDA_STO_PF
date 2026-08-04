# Local patch plan (not executed)

Status: BLOCKED until the exact-fit evidence conflict is resolved.

## Planned changes after authority is supplied

1. Add a single versioned exact contract source and canonical normalized hash.
2. Add explicit `thermodynamic_contract=exact_v1` and `legacy_246_v1` modes.
3. Route CUDA, host, Python, fixture builders, restart metadata, post-processing and transport adapters through the contract.
4. Remove duplicate literal defaults or make them fail-closed compatibility paths.
5. Print contract version/hash, `DeltaH`, `DeltaS`, temperature, root and conversion mode at startup.
6. Store the contract hash in checkpoints and reject mismatched restart unless an audited migration mode is explicit.
7. Add CPU/Python/GPU identity, four-point fit reproduction, conversion round-trip, derivative, driving-force, legacy sensitivity and cross-environment tests.
8. Reclassify old results without rewriting them; retain legacy outputs read-only.
9. Rebuild new binaries in isolated deployment directories; never overwrite running legacy binaries.

## Explicitly forbidden while blocked

- no production source edit;
- no exact numbers copied from the user prompt into runtime;
- no legacy-to-exact relabeling of old results;
- no 400³ production submission under an unresolved contract;
- no remote `pull`, `reset`, `clean`, overwrite or recompilation;
- no final GitHub push, tag or PR claiming freeze.

## User-authorized local parameter update (2026-08-05)

The user explicitly authorized applying the latest external exact-fit
parameters to the current local branch. The following source-level runtime and
analysis literals were updated in place to:

```text
Delta_H=41504.29119633958 J/mol
Delta_S=18.469276826409214 J/(mol K)
L(T)=Delta_H-Delta_S*T
```

Updated active paths include `thermo_utils.h`, `Unit_Psedobinary.py`,
`main_cuda.cu` GP host defaults, explicit-rate/Schur/GP analysis helpers and
the dynamic microstructure validation helpers. Existing reports, archived
results, checkpoints, workstation/cluster files and running jobs were not
modified. This is a local source patch only; it does not constitute a
cross-environment contract freeze, because the contract hash is still null and
the exact-vs-legacy sensitivity has not been run.
