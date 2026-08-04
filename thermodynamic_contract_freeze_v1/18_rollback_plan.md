# Rollback plan

No deployment or production modification occurred in this run. If a later exact freeze is accepted:

1. Preserve all legacy binaries, configs, checkpoints and results in their original directories.
2. Restore legacy reproduction only by checking out the recorded legacy commit/tag and using an explicit `legacy_246_v1` mode.
3. Never use `git reset --hard` or `git clean` on the current dirty local/workstation/cluster trees.
4. Install exact binaries under a versioned isolated directory; do not overwrite a binary referenced by running Slurm jobs.
5. Reject legacy checkpoints in exact mode unless a reviewed migration tool writes a new checkpoint with the exact contract hash.
6. If a GitHub PR/tag is created later, close/revert it without rewriting old result files; preserve the audit and source hashes.
7. Files that must never be deleted: old result directories, checkpoint libraries, source manifests, binary hashes, terminal logs, and legacy provenance reports.
