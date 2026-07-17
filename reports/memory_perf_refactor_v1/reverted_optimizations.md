# Reverted or not-retained optimizations

- **M3**: NOT_RETAINED. memory target already met; coordinated ABI/reduction risk not justified
- **M7**: NOT_RETAINED_ZERO_BENEFIT. no memory benefit
- **M9**: NOT_IMPLEMENTED. top kernels have distinct lifetimes; fusion risk unsupported
- **M10**: NOT_IMPLEMENTED. variable loops and fail-closed branches dominate
