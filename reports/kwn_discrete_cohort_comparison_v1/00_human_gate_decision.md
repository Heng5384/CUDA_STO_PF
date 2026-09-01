# Human gate decision — discrete-event adjudication

Source commit audited before this decision: `2f67a34751af1312eaf63f635f4a5be9166bf03a` (`Close KWN radius-grid convergence audit v1`).  This is the clean commit that adds the latest radius-grid runner and reports.  Its runtime analysis provenance records `944b8ca40270df945d9ab07c28a63c9be2267d5b` as the source baseline used while the audit was executed; that historical runtime field does not replace the report-producing clean commit.

```text
LEGACY_SIX_PARTICLE_EULERIAN_P5: FAIL_RETAINED
SMOOTH_POPULATION_EULERIAN_P5: ACCEPTED_AS_KWN_POPULATION_BACKEND_QUALIFICATION
EXACT_SIX_PARTICLE_FIXTURE: REQUIRES_DISCRETE_COHORT_CHARACTERISTIC_COMPARATOR
TWO_PERCENT_GATE: NOT_RELAXED
EULERIAN_SIX_PARTICLE_OUTPUT: NOT_AUTHORITY_FOR_BETA_ONLY_PF_COMPARISON
LOCAL_GP_RELEASE: NOT_AUTHORIZED_UNTIL_COHORT_COMPARISON_PASSES
```

The legacy exact-six-particle P5 failure is retained as evidence: no threshold is relaxed, overwritten, or re-labelled as a pass.  The smooth-PSD result qualifies the Eulerian KWN backend only in its intended high-number-density continuous-population domain.  The exact six-particle PF fixture instead contains finite dissolution events, for which an event-aware characteristic/cohort representation is the model object aligned with the comparison question.  Choosing that representation is therefore a model-object alignment decision, not a post-hoc relaxation of the 2% gate.

No GP release, GP-to-beta pathway, or online coupling is authorized by this decision.  A separately gated local GP release prototype remains unavailable until the cohort comparison has passed every stated gate.
