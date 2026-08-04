# Checkpoint compatibility report

Status: `NOT_VERIFIED_BLOCKED`

- Existing checkpoints and restart libraries do not carry a verified thermodynamic contract hash.
- Existing runtime uses legacy thermodynamics or has unknown source identity.
- No exact binary exists locally, and no contract-aware restart guard was found in the audited source.
- Therefore an exact binary must not read an old checkpoint silently.
- Required future behavior: reject missing/mismatched contract hash by default; allow only an explicitly named, audited migration/sensitivity mode.
- A legacy reproduction path must remain explicit and print `WARNING_LEGACY_THERMODYNAMIC_CONTRACT` and `NOT_FOR_NEW_PRODUCTION`.

No checkpoint was edited, migrated, or deleted.
