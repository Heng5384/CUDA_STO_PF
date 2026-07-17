# Device-memory dependency graph

`Ctot accepted` feeds BDF2 history/anchor, then transport work/trial/residual. A rejected attempt requires an intact accepted state and histories. `phi` follows the same accepted/history/context chain. This is why pointer rotation must rotate authoritative pointers atomically and cannot merely delete rollback storage.

Transport reconstruction produces `q_alpha`, `xB_alpha`, `Y`, active code and chemical potential. The three shared-face arrays feed one divergence field and are simultaneously needed by the existing outer face-change audit. Face streaming is therefore conditional on preserving shared-face antisymmetry and replacing that audit without losing its prior-outer reference.

The two-slot real scratch arena stores the preconditioned direction/snapshot and scalar-reduction inputs. The complex scratch and shared double FFT plans serve transport and phase serially. No low-memory globalization feature adds a full-grid field.

The current runtime `print_memory_ledger` omits several candidate allocations and is not an admission oracle. `device_buffer_lifetime.csv` is the source allocation ledger; `memory_scaling.csv` remains incomplete until target-GPU cuFFT workspaces and observed peak loss are measured.

Static GP reservoirs are host-side `GpAssistedSite` vectors in this code path. The Ctot candidate currently rejects GP/S3 at validation, so the combined PF+elastic+GP row is a union ledger, not an executable integrated production claim.
