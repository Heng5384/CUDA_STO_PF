
# BDF2 Startup, Restart, and Rollback Validation

The current workstation binary was used for this rerun.

| Check | Result |
|---|---|
| BE startup establishes one legal history pair | PASS |
| normal BDF2 continuation | PASS |
| checkpoint after startup then restart | PASS |
| active BDF2 checkpoint restart, duplicate runs | PASS bitwise |
| rejected BDF2 attempt leaves no history commit | PASS |
| retry rebuilds history through BE then resumes BDF2 | PASS |
| forced/control state and provenance comparison | PASS bitwise |

Restart hashes are recorded in `evidence/startup_restart_summary.json`; rollback
hashes and metadata are in `evidence/rollback_summary.json`.

`BDF2_startup_status=PASS`
`BDF2_restart_status=PASS_BITWISE`
`BDF2_rollback_status=PASS_BITWISE`
