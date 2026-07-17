# Internal homotopy validation

The fixed theta sequence 0.25, 0.5, 1.0 uses the same macro anchor/history and
does not advance time or commit history at intermediate levels. Unit contract
tests pass. Runtime qualification fails: V5 dt/4 triggered homotopy 9 times and
two levels failed, leaving two fallback macros; V5 dt/16 accumulated
129 triggers and
84 failed levels. V5 dt/8 and dt/2
eventually exhausted their safety path. Status: `IMPLEMENTED_DEFAULT_OFF_NOT_PRODUCTION_QUALIFIED`.
