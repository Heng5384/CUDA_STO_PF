# Production candidate decision

## Decision

**PASS_400CUBE_PF_ONLY_ELASTIC_MEMORY_BLOCKED**

Selected engineering mask: `187` = M1+M2+M4+M5+M6+M8.

- 400^3 static PF-only source: 12.182601 GiB (target <=12.5 GiB: PASS).
- Reduction: 37.779% (preferred >=35%: PASS).
- Actual 400^3 peak: 12.900000 GiB / 83.44% (hard <=85%: PASS).
- 8000-step easy-path wall regression: 0.341% (<=3%: PASS).
- Endpoint, rollback and restart: PASS_BITWISE_ENDPOINT_ROLLBACK_RESTART.
- Physics, transport/phase equations and hard gates: unchanged.

The candidate qualifies PF-only memory admission. Elastic and elastic+GP 400^3 remain separately gated and
unmeasured. M3, M7, M9 and M10 are not part of the selected default.
