# Long-window validation

| Run | Completed | Internal rejects | Fallbacks | Result |
|---|---:|---:|---:|---|
| V0 dt/16 | 8000/8000 | 0 | 0 | baseline zero-fallback pass |
| V5 dt/16 | 8000/8000 | 84 | 82 | production fail |
| V5 dt/8 | 3464/4000 | 0 | 0 | exhausted at step 3465 |
| V5 dt/2 | 830/1000 | 0 | 0 | exhausted at step 831 |

V5 recovers many individual difficult solves, but it does not satisfy the
pre-registered zero internal retry and zero fallback contract. The old retry
safety net remains bitwise restorative where exercised.
