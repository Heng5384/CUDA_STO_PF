# Portable Parameter Acceptance Report

## Outcome

`PARTIAL`: static parsing and dependency inventory passed for all five overlays;
portable acceptance failed for all five. No parameter, physical value, seed data,
barrier value, selector setting, or runtime model flag was changed.

## Acceptance matrix

| Check | Result | Reason |
|---|---|---|
| Commit 3 scope | PASS | `b190ddb` contains the ten audited Commit 3 files only |
| Overlay key/value parse | PASS (static) | all five files are valid `key=value` inventories |
| Staging/canonical equivalence | FAIL | 16 shared IDs differ; two required dynamic rows are staging-only |
| CWD-independent C++ resolution | FAIL | `ifstream` receives raw relative paths |
| Missing required input behavior | NOT RUN | no safe portable overlay exists to test |
| Runtime cache independence | FAIL | dynamic-continue constructs profile paths from cache root |
| Physical-units config | BLOCKED | provenance contains machine-specific absolute references |
| Large CUDA simulation | NOT RUN | prohibited by task |

## Acceptance boundary

The five files remain untracked and are not production inputs. The committed
canonical library remains `VALIDATION_REFERENCE_ONLY`. This report does not promote
GP release/growth or change the PF-only composition acceptance boundary.
