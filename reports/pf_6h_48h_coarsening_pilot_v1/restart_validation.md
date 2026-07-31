# Restart validation

Preflight job 72668 wrote a 64-step checkpoint and restarted to step 128.
The continuous and restarted final checkpoint files compare byte-for-byte.
The 42 h pilot uses the same exact restart mechanism at every one-hour
boundary, with serialized (non-concurrent) legs.
