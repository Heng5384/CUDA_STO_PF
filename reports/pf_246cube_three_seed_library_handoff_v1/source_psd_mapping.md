# Source PSD mapping

The 96 historical radii were re-read from the hash-pinned V2 manifest and
mapped independently to the nearest registered library radius. Ties select
the smaller radius.

| registered radius (nm) | count |
|---:|---:|
| 8.0 | 4 |
| 8.5 | 19 |
| 9.0 | 19 |
| 9.5 | 17 |
| 10.0 | 17 |
| 10.5 | 16 |
| 11.0 | 4 |
| 11.5 | 0 |
| total | 96 |

- historical radius range: `8.167822574685072–10.874406364067081 nm`
- historical mean \(R^3\): `863.4804248658662 nm³`
- discrete mean \(R^3\): `864.4661458333334 nm³`
- relative change of mean \(R^3\):
  `+0.0011415672423844539`

All three replicates use this exact discrete PSD. No profile interpolation,
profile scaling, analytic sphere replacement, or histogram retuning was used.
