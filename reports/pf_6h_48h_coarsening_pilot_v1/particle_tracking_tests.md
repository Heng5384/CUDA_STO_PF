# Particle tracking tests

The fixture materializer finds 32 periodic components at t=0. The cluster
pilot analyzer consumes only immutable checkpoint fields and writes
particle_trajectories.csv. Unexpected component merges/splits set the analyzer
status to BLOCKED_UNEXPECTED_PARTICLE_MERGE_SPLIT rather than relabeling or
volume-sorting particles.
