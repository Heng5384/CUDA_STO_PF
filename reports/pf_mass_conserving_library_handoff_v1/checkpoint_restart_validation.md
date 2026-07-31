# Checkpoint and restart validation

Checkpoint format V3 stores all zero-mode runtime fields plus the conditional
initial-state identity.  The 64-step continuous run and 32+32 restart run
produced the same complete checkpoint bytes:

```text
continuous_checkpoint_sha256=1998b319bfb20709111b610a794dde64d5bcf8136cfd37e305cb451e39e69962
restart_checkpoint_sha256=1998b319bfb20709111b610a794dde64d5bcf8136cfd37e305cb451e39e69962
continuous_restart_checkpoint_bytewise_equal=true
```

Both headers contain:

```text
magic=PFZMCHK3
version=3
accepted_step=64
initial_state_class=MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1
fixture_manifest_sha256=a87e76405bd1b901b6848b6c39d788ff3fe537ca0acea93f99b64ac67e6ced81
profile_library_manifest_sha256=58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe
payload_checksum=16278730450631557460
```

Negative runtime tests changed only one hash at a time.  Both were rejected
before state restoration with `checkpoint provenance mismatch`.

```text
fixture_mismatch_rejection=PASS
library_mismatch_rejection=PASS
normal_run_stderr_empty=true
restart_status=PASS
```

Historical V2 checkpoint parsing is retained only for callers presenting the
legacy sentinel identity.  A conditional-handoff run cannot accept a legacy
checkpoint because it lacks fixture/library provenance.

