# Mass inventory audit

The target global inventory is identical in A/B/C:

\[
\langle C_B^{tot}\rangle=0.03.
\]

The matrix baseline is a derived residual after retaining every selected
particle's actual \(h\)-volume and local relaxation inventory.

| replicate | derived matrix \(x_B\) | derived matrix \(x_{Ag}\) | field relative error |
|---|---:|---:|---:|
| A | 0.006810401095358129 | 0.006787289015086700 | 1.3033275375015027e-15 |
| B | 0.006810340069425079 | 0.006787228402649627 | 5.213310150006011e-16 |
| C | 0.006810406213610769 | 0.006787294098659213 | 5.213310150006011e-16 |

The derived value is not forced to the older approximately 0.0062 matrix
value. Doing so while retaining the 96 exact library profiles would violate
the frozen total inventory unless particle inventory were changed or a
case-specific global projection were applied. Both operations are prohibited.

All fields are finite, require no clipping/normalization, and close the
canonical inventory at machine precision.
