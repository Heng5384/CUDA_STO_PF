# Allocation call graph

`main -> parameter validation -> PF candidate selector -> core state allocations -> BDF2 history/context -> transport/phase work fields -> shared FFT plans -> accepted-step loop -> teardown`.

The frozen PF-only path is detailed in `allocation_inventory.csv`; every syntactic device allocate/free, pinned-host allocate/free, cuFFT plan/create/size/work-area/destroy call in the audited sources is indexed in `all_cuda_allocation_calls.csv`. Conditional GP, elastic, minimize, observer, and legacy paths remain classified rather than counted as simultaneously live. The table is an asset map, not a peak-memory assertion.

Authoritative state is `Ctot + phi` with BDF2 histories. `Y`, `xB_alpha`, and `q_alpha` are derived. The selected path removes the persistent `Y^n` image and rebuilds it from the authoritative rollback image. Event macro images alias dead BDF2 context buffers during BE subcycling.
