# 400³ campaign readiness

Final readiness: `READY_FOR_400_PRODUCTION=NO`

Blocking status: `BLOCKED_CONTRACT_CONFLICT`

The current 400³ campaign has existing static/short qualification artifacts and live Slurm jobs, but no frozen exact contract hash. Those artifacts and jobs were not altered or canceled. No new 400³ submission was made by this freeze audit.

Before any new exact-contract production submission, all of the following are required:

1. accept and hash-register the external exact-fit source data/report/CSV outputs;
2. freeze normalized YAML/JSON contract and SHA-256;
3. route fixture, runtime, restart and post-processing through the same hash;
4. build an isolated binary and run CPU/Python/GPU identity tests;
5. run a one-step smoke, short restart, mass audit and cross-node test;
6. classify existing 400³ artifacts as legacy/mixed/unknown without rewriting them;
7. obtain explicit user authorization before submitting the complete campaign.
