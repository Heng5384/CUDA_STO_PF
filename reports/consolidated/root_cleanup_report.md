# Root Cleanup Report

This pass archived safe root-level reports, CSVs, and temporary folders into `reports/root_archive/` while leaving code, inputs, pipeline directories, and hardcoded runtime dependencies in place.

## Summary
- root_md_total: 26
- root_md_moved: 19
- root_md_retained: 7
- root_csv_total: 28
- root_csv_moved: 7
- root_csv_retained: 21
- root_folder_total: 4
- root_folder_moved: 4
- root_folder_retained: 0

## Archive destinations
- `reports/root_archive/md/`
- `reports/root_archive/csv/`
- `reports/root_archive/misc/`
- `reports/root_archive/figures/profile_plots/` (diagnostic figures retained)

## Notes
- Files that remain in the root are either code, configuration, runtime inputs, or report artifacts still referenced by code/script/README defaults.
- The first pass moved all four folders. The final cleanup retained the profile plots under the canonical figures path and removed the three reproducible Python cache trees.
