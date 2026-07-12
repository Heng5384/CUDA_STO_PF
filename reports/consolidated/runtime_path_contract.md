# Runtime Path Contract

## Current contract observed in source

| Path class | Current base | Source evidence | Required behavior |
|---|---|---|---|
| Runtime barrier CSV | process current working directory | `main_cuda.cu:5353-5355` | required when GP runtime library is enabled |
| Runtime nucleus CSV | process current working directory | `main_cuda.cu:5587-5591` | required when runtime nucleus library is enabled |
| Dynamic-continue profile root | process current working directory | `main_cuda.cu:5662-5668` | nonempty when runtime library plus bridge is enabled |
| Python selector/catalog paths | repository root | `nucleus_selector.py:15-49` | stable for Python selector only |
| Orchestrator workflow outputs | repository root plus workflow root | `nucleus_orchestrator.py:127-145` | generated output, not portable input |

## Portable target contract

Authoritative CSV/JSON inputs should be repository-relative after a single explicit
resolver is introduced. Runtime cache should be regenerable and selected through
an output/cache-root CLI or environment contract. Restart inputs should remain
explicit and checksum-bearing. A missing required input must be a hard error;
missing cache must not silently select a different nucleus.

The present five overlays do not satisfy this contract. Their `Results/...` paths
are not merely output labels: the C++ runtime uses the cache root to construct
profile, source-dynamic, and metadata paths. Treating the cache as optional would
change dynamic-continue behavior.
