# Contract test status

The production exact-contract suite is intentionally not executed while the decision is `BLOCKED_CONTRACT_CONFLICT`.

The legacy static identity check in `legacy_static_identity.txt` independently reproduces the existing runtime legacy root and exact algebraic `xAg`/`xB` round trip. Audit revision 2 also independently reproduces the external calibration workspace's four-point candidate fit in `exact_fit_reproduction_v2.txt`; that validates candidate arithmetic, not cross-environment publication authority.

Required blocked tests before Stage B:

- schema and normalized YAML/JSON hash stability;
- four-point fit reproduction against an immutable pointwise residual artifact;
- exact root and CPU/Python/GPU identity;
- derivative and driving-force sign;
- legacy-vs-exact sensitivity;
- runtime identity and checkpoint compatibility;
- cross-platform hash/value reproducibility.
