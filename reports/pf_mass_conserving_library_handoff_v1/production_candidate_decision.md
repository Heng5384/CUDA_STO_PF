# Conditional handoff production-candidate decision

## Decision

The initial-state generation, global inventory, deterministic materialization,
checkpoint provenance, zero mode, and short dynamic/restart functions all
pass.

The accepted scientific claim is limited to:

> a mass-conserving, profile-library-assembled conditional 6 h handoff state.

It is not a common multi-particle equilibrium and not a unique reconstruction
of the experimental 6 h microstructure.

## Evidence roots

```text
runtime_source=/home/zhiheng/tmp/codex_pf_mass_conserving_library_handoff_v1_20260731
fixture=/home/zhiheng/tmp/pf_mass_conserving_library_handoff_fixture_v1_20260731
qualification=/home/zhiheng/tmp/pf_mass_conserving_library_handoff_qualification_v2_20260731
runtime_source_tree_sha256=480e6d8c0e25fd9d13ecce51b35498b0b3726b91e08f91f6837628d0bffa3e93
runtime_binary_sha256=7581c169fb1d1c16ee60764f418ee9c33508682c8b9dd8cf76b89368c7990602
fixture_manifest_sha256=a87e76405bd1b901b6848b6c39d788ff3fe537ca0acea93f99b64ac67e6ced81
qualification_summary_sha256=02dfc09e83a6af41ec80f9f95473dd72a1451ff05a3331e67b854e4893e6e840
```

The first non-overwriting qualification root (`...qualification_v1_...`) is
preserved as a preflight-only failure caused by a missing copied top-level
physical-input JSON.  It launched no CUDA steps.  The successful evidence is
the V2 qualification root listed above.

## Scope

No thermodynamics, interface energy, eigenstrain, mobility, or diffusivity was
retuned.  No GP or beta-nucleation path was enabled.  No cluster resource,
commit, push, or full 6–48 h production was used.

```text
qualified_conditional_handoff_fixture=true
recommended_next_action=select production dt with registered observable convergence, then obtain explicit authorization before any 6-48 h ensemble
final_status=PASS_MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1
```

