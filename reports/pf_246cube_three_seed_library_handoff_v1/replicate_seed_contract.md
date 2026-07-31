# Replicate seed contract

For replicate label \(L\), the deterministic seed material is:

```text
8ce52733e6755b23f4f7d4ba53f0414ef927bb863c3eb4ce9d43cb09d4d5de45|L
```

The unsigned 64-bit seed is the first 16 hexadecimal characters of the
SHA-256 digest.

| replicate | unsigned 64-bit seed |
|---|---:|
| replicate_A | 18278234711707939752 |
| replicate_B | 6256128897973916905 |
| replicate_C | 4209997954605651191 |

Placement uses descending registered radius followed by canonical particle
ID, with deterministic periodic hard-core rejection. Every pair satisfies

\[
d_{ij}^{periodic}>R_i+R_j+4\lambda_{sm}.
\]

The complete embedded `h(phi)>1e-4` supports were also checked directly and
do not overlap.
