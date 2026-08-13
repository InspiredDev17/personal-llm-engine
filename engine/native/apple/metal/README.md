# Apple Metal backend boundary

Status: **PLANNED / DISABLED**.

This directory is reserved for a first-party Apple Metal system-API backend.
No Metal bindings, shaders, dynamic-library discovery, kernels, or unsafe FFI
exist in the foundation stage. Future work must isolate FFI and unsafe code,
document invariants, select kernels only from engine-owned code, and never load
model-supplied executable code.

The separate host-only `engine-arm64` crate does not change this status. Its
Advanced SIMD CPU kernels do not call Metal or provide GPU execution.
