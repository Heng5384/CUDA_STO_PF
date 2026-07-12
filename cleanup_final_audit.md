# Cleanup Final Audit

## Executive summary

This pass cleaned and organized the repository without changing any physical formula, CUDA kernel algorithm, parameter calibration, composition mode, RSMD/S3 transaction, GP ledger, handoff path, or accepted runtime input. The cleanup commit is intentionally separated from the pre-existing functional worktree changes.

The repository remains scientifically dirty by design: large raw evidence and several thousand untracked functional/report/input files are retained pending explicit review. The correct final status is therefore `PARTIAL_PROJECT_CLEANED_BUT_REVIEW_REQUIRED_FILES_REMAIN`, not a false clean-tree PASS.

## Starting inventory

- filesystem files excluding `.git`: 9474
- tracked plus non-ignored candidates: 4135
- file-level dirty paths (`git status --short -uall`): 3728
- tracked modified functional/document paths: 14
- report Markdown files audited: 735
- duplicate checksum groups among files up to 10 MB: 515
- repository size drivers: `reports/` 4.3 GB, `tmp_codex_ops/` 4.2 GB, `Results/` 3.4 GB

The complete starting snapshot is `cleanup_repository_inventory.csv`.

## Files deleted

- tracked files deleted: 3 (two Python bytecode files and one `.DS_Store`)
- untracked/ignored files deleted: 145
- cache/editor artifacts: 127
- root scratch/render outputs: 19
- unrelated personal PDFs: 2
- duplicate members removed with the cache cleanup: 93

Deletion was driven only by the reviewed whitelist in `cleanup_delete_candidates.csv`. The three tracked cache/editor files are removed through `git rm`; no recursive repository cleaning command was used. Details and exclusions are documented in `cleanup_deletion_plan.md`.

## Files moved and archived

- 31 diagnostic profile-plot files were retained and normalized from the ignored `.tmp_profile_plots` location to `reports/root_archive/figures/profile_plots/`.
- `reports/root_archive/` now contains 58 compact archived report/table/figure files.
- Three archived Python cache trees were deleted instead of being preserved as evidence.
- No source, parameter, seed, profile library, Mode L/X/Q evidence, or RSMD/S3 evidence was moved or deleted in this pass.

## Reports and provenance

`reports/report_supersession_ledger.csv` inventories 735 Markdown reports:

- `CURRENT_ACCEPTED`: 243
- `CURRENT_PARTIAL`: 66
- `CURRENT_FAILED_BUT_DIAGNOSTICALLY_IMPORTANT`: 129
- `SUPERSEDED_RETAIN_FOR_PROVENANCE`: 5
- `UNVERIFIED`: 292

All ledger entries are retained. `FAIL` was never treated as a deletion criterion. Mode L/X/Q comparison evidence, capacity-bound and projection failures, PF-only gates, S3 source-only validation, GP ledgers, and handoff reports remain available.

## Structure and documentation

- `.gitignore` now covers reproducible CUDA/CMake products, Python/Jupyter/test caches, editor files, VTK/RAW dumps, and named scratch trees.
- `docs/project_structure.md` records directory ownership and raw/generated boundaries.
- `docs/current_model_status.md` records Mode L/X/Q as non-production and keeps `S3_source_component_frozen=true` / `S3_reintegration_allowed=false` explicit.
- README links to the new structure/status documents and no longer contains its local absolute link to `physical_inputs.example.json`.
- 17 obvious local absolute Markdown links were converted to repository-relative links in active workflow documentation.

## Absolute paths remaining

There are 41 code/workflow files outside `reports/` that still mention the local macOS or workstation repository path. Most shell launchers expose the path through an overridable `ROOT`/`ROOT_DIR` default and are not broken. Two untracked Python workflow files use a hard-coded local `Path`; they belong to the pre-existing functional nucleation work and are retained for a dedicated functional commit rather than mixed into cleanup. Historical report provenance paths are intentionally retained.

## Validation

- Python syntax scan for repository workflow code: PASS
- Key analysis `--help` entry points: PASS
- `Unit_Psedobinary.py` parameter generation: PASS
- storage-exact formula test: PASS
- X/Q operator diagnostic: process PASS; four intentional Mode X failure rows retained as the audited gap
- GP mass-budget initialization dry-run: PASS
- GP-assisted beta event and release-kernel mass-conservation dry-run: PASS
- local/workstation MD5 for `main_cuda.cu`, `cuda_kernels.cu`, `cuda_kernels.h`, `pf_params.h`, `thermo_utils.h`: MATCH
- workstation CUDA build: PASS (`BUILD_EXIT=0`)
- workstation unit-conversion smoke: PASS 5/5 (`SMOKE_EXIT=0`)

The build retains three warnings unrelated to this cleanup: one unused GP post-birth label, one unused profile helper, and the known `thermo_utils.h:270` device-constant declaration warning.

## Git and scope audit

The full `git diff --check` reports trailing whitespace in pre-existing `main_cuda.cu` functional changes. This cleanup does not edit those lines. The cleanup staged diff is checked separately and must pass before commit.

Pre-existing functional changes intentionally excluded from the cleanup commit include `main_cuda.cu`, `cuda_kernels.cu`, `cuda_kernels.h`, `pf_params.h`, `thermo_utils.h`, `physical_inputs.example.json`, CNT/dynamic-continue analysis changes, cluster environment changes, and untracked nucleation workflow code.

## Remaining review-required material

- `cleanup_final_inventory.csv` marks 2777 files `REVIEW_REQUIRED`.
- Large raw CSV/VTK/log evidence under `reports/` is retained until accepted summaries and reproducibility are explicitly confirmed.
- `tmp_codex_ops/`, `.tmp_fitting_inputs/`, and `.tmp_radius_table_local/` are ignored or retained on disk because they may contain unique scientific evidence.
- `Results/`, `Results_scan/`, and `outputs/` remain local and ignored; no raw result tree was deleted.
- No remote push was performed.

## Final decision

`PARTIAL_PROJECT_CLEANED_BUT_REVIEW_REQUIRED_FILES_REMAIN`

The next safe action is a separate functional packaging audit for the current nucleation/CUDA/dynamic-continue changes, followed by an explicit raw-data retention decision for the 2777 review-required files.
