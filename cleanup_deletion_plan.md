# Cleanup Deletion Plan

Only high-confidence, untracked/ignored, reproducible or unrelated files are approved for deletion in this pass.
No tracked source, accepted parameter, seed/library input, raw Results tree, or REVIEW_REQUIRED item is included.

## cache/editor artifacts

- count: 127
- scope: `.DS_Store`, Python bytecode/cache trees, and one ignored temporary postprocess helper.

Detailed per-file evidence is in `cleanup_delete_candidates.csv`.

The three tracked cache/editor entries are removed with `git rm`; all other entries are untracked or ignored and were unlinked only after this whitelist was generated.

## root scratch/render outputs

- count: 19
- scope: unreferenced `tmp*` render/seed-summary files and generated PDFs whose canonical Markdown/PDF evidence exists under `reports/`.

Detailed per-file evidence is in `cleanup_delete_candidates.csv`.

## unrelated personal files

- count: 2
- scope: two personal application PDFs unrelated to CUDA_STO_PF.

Detailed per-file evidence is in `cleanup_delete_candidates.csv`.

## Explicitly retained

- `Results/`, `Results_scan/`, and `outputs/`: ignored raw simulation output, retained on disk.
- `tmp_codex_ops/`, `.tmp_fitting_inputs/`, `.tmp_radius_table_local/`: `REVIEW_REQUIRED`; may contain unique scientific evidence.
- Large CSV/VTK/log artifacts under `reports/`: retained pending explicit raw-data policy and supersession review.
- All accepted params, seed/profile libraries, Mode L/X/Q audits, and S3/RSMD evidence.
