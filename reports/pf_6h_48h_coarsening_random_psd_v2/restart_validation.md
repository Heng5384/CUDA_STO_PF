# Restart validation

The official 6--12 h preflight reports bytewise-equal continuous and restarted
final checkpoints. Both hashes are
`5c4d857f9cec7e7a11946c44fc4ee87fe54b623292549ddeedca81581b638761`.

This validates checkpoint/restart for the preflight only. The 12--48 h
continuation remains pending in job 72915 and must produce its own final
comparison before the V2 goal can be closed.
