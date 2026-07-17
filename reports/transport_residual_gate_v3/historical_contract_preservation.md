# Historical contract preservation

- V1 preserved: `true`
- V1 status remains `FAIL_PREREGISTERED_LOCAL_PEAK_DEFECT_RATE_GATE`.
- V2 status remains `FAIL_V2_MATERIAL_TRANSPORT_DEFECT`.
- Frozen V1 evidence manifest SHA-256: `773099c65ec7dab6707e280d4dc5d5f910f7fba1e4ca3311dbf61fdd1a3b639f`.
- V2 binary SHA-256: `7d76e43e262a99bc452efc008a1ea0e767745f423e508555eb13aed15fffc16d`.
- V2 simulations: `2`; observer neutrality is recorded as false for solver-state modification in both statuses.
- V1 diagnostics-on/off raw-field bitwise neutrality: `True`.
- V1 transactional observer raw-field bitwise neutrality: `True`.
- Window endpoints are read from checkpoint metadata, not inferred from names.
- BDF2 history provenance is read from each checkpoint metadata file.
- Changed historical paths: `NONE`.

V3 is independent. It does not rename or replace either historical failure.

## Strict `G12+dt32`

- Runtime params SHA-256: `29997f5dd6bbab329d53562aabfd0b934d6c463183b0c26827e128525acfc534`.
- Final Ctot SHA-256: `e42af4d48d89abd088c98708b15ad6b7c71566928ff7fe2403b618872bfeebbd`.
- Final phi SHA-256: `aa40ef9ccbb6f48d34b8bad24beb270e545b693173e6ea90f067e4b521c18107`.
- Final checkpoint metadata SHA-256: `2419a6002b2cab8a216b9f6d502b70d40a70f9ffa2e88314abecf5d047826a0c`.
- Full observer D SHA-256: `f39e227b86fed61ead559e51312f9ba2a8fde7550beacaa1affa30c0a145d9ab`.
- Full observer A SHA-256: `26cdd73b10db62d737c5960be7e2227a6f2f8b1c8c81c60d6b944c9ea8d4e18b`.
- Endpoint: `7.812500000011301` code time / `321.3248861297989` s.
- BDF2 history valid: `1`; accepted step `80000`.

## Candidate `G9+dt4`

- Runtime params SHA-256: `fb28a212f096c3e8acd3fe1a3eaa39c1215b6116b4d657b45afbf479fb87681c`.
- Final Ctot SHA-256: `5cf182c005fb35b359099fdb7a0f7b241b3d4815e934ae50586f05cb30d9b0d3`.
- Final phi SHA-256: `667b7f514c6cab39390100d49c59eb653a7498801964fb0b0702fa1e20ca7ce1`.
- Final checkpoint metadata SHA-256: `0289baac452f31052b9fe9467d2071f21ef3b05799e7261c9797369efe489ba9`.
- Full observer D SHA-256: `7e9168fd0c76631c4d805eb8e4dac3c26084f99dce2e02982836f2339ca747f2`.
- Full observer A SHA-256: `14daa38473a89bab53f29e2188bfc099a1f2d316771f5d16a7a04c378a2b954b`.
- Endpoint: `7.812500000001241` code time / `321.32488612938516` s.
- BDF2 history valid: `1`; accepted step `10000`.
