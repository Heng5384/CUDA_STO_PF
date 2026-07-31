# Particle observation contract

Particles are identified from the read-only structural mask phi > 0.5.
Connected components use six-neighbor periodic connectivity. Component IDs are
assigned deterministically, and later outputs use maximum voxel overlap first
with periodic centroid distance as the tie-breaker. A merge or split is
reported fail-closed.

Each row contains component cell count, h-volume, equivalent radius, periodic
centroid, parent overlap count, and the checkpoint step. Global rows contain
resolved count, resolved number density, beta fraction, interface area density,
matrix xB and converted matrix xAg.
