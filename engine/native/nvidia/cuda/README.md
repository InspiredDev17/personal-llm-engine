# NVIDIA CUDA backend boundary

Status: **PLANNED / DISABLED**.

This directory is reserved for a first-party NVIDIA CUDA system-API backend.
No CUDA bindings, kernels, runtime dynamic-library discovery, or unsafe FFI
exist in the foundation stage. Future work must isolate FFI and unsafe code,
document invariants, select kernels only from engine-owned code, and never load
model-supplied executable code.
