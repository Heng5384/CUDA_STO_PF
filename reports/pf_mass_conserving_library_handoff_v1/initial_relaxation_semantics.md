# Initial relaxation semantics

The historical V5/V5e attempts asked whether six particles could first be
driven to a common constrained multi-particle equilibrium.  Their convergence
failures remain valid diagnostics, but that equilibrium is no longer an
admission condition for a conditional experimental handoff.

For V1:

```text
common_multi_particle_equilibrium_required=false
common_multi_particle_equilibrium_claim=false
common_multi_particle_pre_relaxation_run=false
initial_relaxation_is_physical_evolution=true
particle_profiles_locked_after_t0=false
```

The one-step dynamic probe proved that `phi(t=6 h + dt)` is not byte-identical
to `phi(t=6 h)`, while component identity and overlap remain valid:

```text
initial_profile_free_after_t0=true
component_identity_status=true
unexpected_merge_or_split=false
```

Therefore the engine does not silently freeze the library shapes.  Elastic
restructuring begins at 6 h and consumes physical simulated time.  No
pre-relaxed state may later be relabelled as the 6 h initial condition.

