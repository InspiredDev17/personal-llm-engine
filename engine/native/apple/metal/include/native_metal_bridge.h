#ifndef PERSONALAI_NATIVE_METAL_BRIDGE_H
#define PERSONALAI_NATIVE_METAL_BRIDGE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct PaiMetalContext PaiMetalContext;
typedef struct PaiMetalBuffer PaiMetalBuffer;
typedef enum PaiMetalStatus { PAI_METAL_OK=0, PAI_METAL_INVALID_ARGUMENT=1,
  PAI_METAL_UNSUPPORTED=2, PAI_METAL_ALLOCATION_FAILURE=3, PAI_METAL_LIBRARY_FAILURE=4,
  PAI_METAL_PIPELINE_FAILURE=5, PAI_METAL_COMMAND_FAILURE=6, PAI_METAL_INTERNAL_FAILURE=7 } PaiMetalStatus;
typedef enum PaiMetalStorageMode { PAI_METAL_STORAGE_SHARED=0, PAI_METAL_STORAGE_PRIVATE=1 } PaiMetalStorageMode;
typedef enum PaiMetalKernel { PAI_METAL_KERNEL_BUFFER_COPY=0, PAI_METAL_KERNEL_VECTOR_ADD=1,
  PAI_METAL_KERNEL_VECTOR_MULTIPLY=2, PAI_METAL_KERNEL_SCALAR_MULTIPLY=3,
  PAI_METAL_KERNEL_F32_MATVEC=4, PAI_METAL_KERNEL_F32_MATMUL=5,
  PAI_METAL_KERNEL_F16_MATVEC=6, PAI_METAL_KERNEL_F16_MATMUL=7,
  PAI_METAL_KERNEL_RMS_NORM=8, PAI_METAL_KERNEL_SOFTMAX=9, PAI_METAL_KERNEL_SILU=10,
  PAI_METAL_KERNEL_SWIGLU=11, PAI_METAL_KERNEL_ROPE=12, PAI_METAL_KERNEL_EMBEDDING=13,
  PAI_METAL_KERNEL_ATTENTION_SCORES=14, PAI_METAL_KERNEL_CAUSAL_MASK=15,
  PAI_METAL_KERNEL_ATTENTION_VALUES=16, PAI_METAL_KERNEL_AIPQ8_MATVEC=17,
  PAI_METAL_KERNEL_AIPQ4_MATVEC=18, PAI_METAL_KERNEL_COUNT=19 } PaiMetalKernel;
typedef struct PaiMetalKernelParams { uint32_t count, rows, columns, inner, position,
  head_dimension, head_count, kv_head_count, sequence_length, query_count,
  first_query_position, block_count; float scalar0, scalar1, scalar2, scalar3; } PaiMetalKernelParams;
typedef struct PaiMetalDeviceInfo { uint64_t registry_id, recommended_working_set_bytes,
  maximum_buffer_bytes; uint32_t maximum_threads_per_group; uint8_t unified_memory,
  supports_family_apple7, supports_f16, reserved; char name[128]; } PaiMetalDeviceInfo;
typedef struct PaiMetalStatistics { uint64_t current_allocated_bytes, peak_allocated_bytes,
  allocation_count, host_to_device_bytes, device_to_host_bytes, dispatch_count,
  command_failures; } PaiMetalStatistics;
size_t pai_metal_device_count(void);
PaiMetalStatus pai_metal_device_info(size_t, PaiMetalDeviceInfo*);
PaiMetalStatus pai_metal_context_create(size_t, const uint8_t*, size_t, PaiMetalContext**);
void pai_metal_context_destroy(PaiMetalContext*);
PaiMetalStatus pai_metal_buffer_create(PaiMetalContext*, size_t, size_t, PaiMetalStorageMode, PaiMetalBuffer**);
void pai_metal_buffer_destroy(PaiMetalBuffer*);
size_t pai_metal_buffer_length(const PaiMetalBuffer*);
void* pai_metal_buffer_contents(PaiMetalBuffer*);
PaiMetalStatus pai_metal_upload(PaiMetalContext*, const uint8_t*, size_t, PaiMetalBuffer*);
PaiMetalStatus pai_metal_download(PaiMetalContext*, const PaiMetalBuffer*, uint8_t*, size_t);
PaiMetalStatus pai_metal_upload_range(PaiMetalContext*, const uint8_t*, size_t, size_t, PaiMetalBuffer*);
PaiMetalStatus pai_metal_download_range(PaiMetalContext*, const PaiMetalBuffer*, size_t, uint8_t*, size_t);
PaiMetalStatus pai_metal_buffer_clear(PaiMetalContext*, PaiMetalBuffer*);
PaiMetalStatus pai_metal_dispatch(PaiMetalContext*, PaiMetalKernel, PaiMetalBuffer* const*,
  const size_t*, size_t, const PaiMetalKernelParams*, uint32_t, uint32_t, uint32_t);
PaiMetalStatus pai_metal_synchronize(PaiMetalContext*);
PaiMetalStatus pai_metal_statistics(PaiMetalContext*, PaiMetalStatistics*);
size_t pai_metal_last_error(PaiMetalContext*, char*, size_t);
#ifdef __cplusplus
}
#endif
#endif
