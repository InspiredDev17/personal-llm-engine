#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "native_metal_bridge.h"
#include <algorithm>
#include <cstring>
#include <mutex>
#include <new>
#include <string>

struct PaiMetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    NSArray<id<MTLComputePipelineState>> *pipelines;
    std::mutex mutex;
    std::string error;
    PaiMetalStatistics statistics{};
};
struct PaiMetalBuffer { PaiMetalContext *owner; id<MTLBuffer> buffer; size_t length; PaiMetalStorageMode mode; };

static const char *const kKernelNames[PAI_METAL_KERNEL_COUNT] = {
 "pai_buffer_copy","pai_vector_add","pai_vector_multiply","pai_scalar_multiply",
 "pai_f32_matvec","pai_f32_matmul","pai_f16_matvec","pai_f16_matmul","pai_rms_norm",
 "pai_softmax","pai_silu","pai_swiglu","pai_rope","pai_embedding","pai_attention_scores",
 "pai_causal_mask","pai_attention_values","pai_aipq8_matvec","pai_aipq4_matvec"};

static NSArray<id<MTLDevice>> *available_devices(void) {
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    if (devices.count != 0) return devices;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device == nil ? @[] : @[device];
}

static PaiMetalStatus fail(PaiMetalContext *context, PaiMetalStatus status, const char *message) {
    if (context != nullptr) { std::lock_guard<std::mutex> lock(context->mutex); context->error = message; }
    return status;
}
static bool multiply(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return false; *out = a * b; return true;
}
static bool add(size_t a, size_t b, size_t *out) {
    if (b > SIZE_MAX - a) return false; *out = a + b; return true;
}
static size_t expected_buffers(PaiMetalKernel kernel) {
    switch (kernel) {
      case PAI_METAL_KERNEL_BUFFER_COPY: case PAI_METAL_KERNEL_SCALAR_MULTIPLY:
      case PAI_METAL_KERNEL_SOFTMAX: case PAI_METAL_KERNEL_SILU: case PAI_METAL_KERNEL_ROPE:
      case PAI_METAL_KERNEL_EMBEDDING: case PAI_METAL_KERNEL_CAUSAL_MASK: return 2;
      case PAI_METAL_KERNEL_AIPQ8_MATVEC: case PAI_METAL_KERNEL_AIPQ4_MATVEC: return 4;
      default: return 3;
    }
}
static bool validate_lengths(PaiMetalKernel kernel, PaiMetalBuffer *const *buffers,
                             const size_t *offsets, size_t count, const PaiMetalKernelParams &p) {
    size_t elements=0, bytes=0; size_t required[4]={0,0,0,0};
    switch (kernel) {
      case PAI_METAL_KERNEL_BUFFER_COPY: required[0]=required[1]=p.count; break;
      case PAI_METAL_KERNEL_VECTOR_ADD: case PAI_METAL_KERNEL_VECTOR_MULTIPLY:
      case PAI_METAL_KERNEL_SWIGLU:
        if(!multiply(p.count,4,&bytes))return false;required[0]=required[1]=required[2]=bytes;break;
      case PAI_METAL_KERNEL_SCALAR_MULTIPLY: case PAI_METAL_KERNEL_SILU:
        if(!multiply(p.count,4,&bytes))return false;required[0]=required[1]=bytes;break;
      case PAI_METAL_KERNEL_F32_MATVEC:
        if(!multiply(p.rows,p.columns,&elements)||!multiply(elements,4,&required[0])||
           !multiply(p.columns,4,&required[1])||!multiply(p.rows,4,&required[2])) return false;
        break;
      case PAI_METAL_KERNEL_F16_MATVEC:
        if(!multiply(p.rows,p.columns,&elements)||!multiply(elements,2,&required[0])||
           !multiply(p.columns,2,&required[1])||!multiply(p.rows,4,&required[2])) return false;
        break;
      case PAI_METAL_KERNEL_F32_MATMUL: case PAI_METAL_KERNEL_F16_MATMUL: {
        size_t width=kernel==PAI_METAL_KERNEL_F32_MATMUL?4:2;
        if(!multiply(p.rows,p.inner,&elements)||!multiply(elements,width,&required[0])||
           !multiply(p.inner,p.columns,&elements)||!multiply(elements,width,&required[1])||
           !multiply(p.rows,p.columns,&elements)||!multiply(elements,4,&required[2])) return false;
        break; }
      case PAI_METAL_KERNEL_RMS_NORM:
        if(!multiply(p.rows,p.columns,&elements)||!multiply(elements,4,&required[0])||
           !multiply(p.columns,4,&required[1])) return false;
        required[2]=required[0];break;
      case PAI_METAL_KERNEL_SOFTMAX: case PAI_METAL_KERNEL_CAUSAL_MASK:
        if(!multiply(p.rows,p.columns,&elements)||!multiply(elements,4,&bytes))return false;
        required[0]=required[1]=bytes;break;
      case PAI_METAL_KERNEL_ROPE:
        if(!multiply(p.head_count,p.head_dimension,&elements)||!multiply(elements,4,&bytes))return false;
        required[0]=required[1]=bytes;break;
      case PAI_METAL_KERNEL_EMBEDDING:
        if(!add(p.position,1,&elements)||!multiply(elements,p.columns,&elements)||!multiply(elements,4,&required[0])||
           !multiply(p.columns,4,&required[1])) return false;
        break;
      case PAI_METAL_KERNEL_ATTENTION_SCORES:
        if(!multiply(p.head_count,p.head_dimension,&elements)||!multiply(elements,4,&required[0])||
           !multiply(p.sequence_length,p.kv_head_count,&elements)||!multiply(elements,p.head_dimension,&elements)||
           !multiply(elements,4,&required[1])||!multiply(p.head_count,p.sequence_length,&elements)||
           !multiply(elements,4,&required[2])) return false;
        break;
      case PAI_METAL_KERNEL_ATTENTION_VALUES:
        if(!multiply(p.head_count,p.sequence_length,&elements)||!multiply(elements,4,&required[0])||
           !multiply(p.sequence_length,p.kv_head_count,&elements)||!multiply(elements,p.head_dimension,&elements)||
           !multiply(elements,4,&required[1])||!multiply(p.head_count,p.head_dimension,&elements)||
           !multiply(elements,4,&required[2])) return false;
        break;
      case PAI_METAL_KERNEL_AIPQ8_MATVEC: case PAI_METAL_KERNEL_AIPQ4_MATVEC: {
        size_t bpr=(size_t(p.columns)+31)/32, blocks=0;if(!multiply(p.rows,bpr,&blocks))return false;
        if(!multiply(blocks,4,&required[0])||!multiply(blocks,kernel==PAI_METAL_KERNEL_AIPQ8_MATVEC?32:16,&required[1])||
           !multiply(p.columns,4,&required[2])||!multiply(p.rows,4,&required[3])) return false;
        break; }
      default: return false;
    }
    for(size_t i=0;i<count;i++){
        if(buffers[i]==nullptr||offsets[i]>buffers[i]->length||required[i]>buffers[i]->length-offsets[i])return false;
    }
    return true;
}

size_t pai_metal_device_count(void) { @autoreleasepool { return available_devices().count; } }
PaiMetalStatus pai_metal_device_info(size_t index, PaiMetalDeviceInfo *out) { @autoreleasepool { @try {
    if(out==nullptr)return PAI_METAL_INVALID_ARGUMENT; NSArray<id<MTLDevice>> *devices=available_devices();
    if(index>=devices.count)return PAI_METAL_INVALID_ARGUMENT; id<MTLDevice>d=devices[index];std::memset(out,0,sizeof(*out));
    out->registry_id=d.registryID;out->recommended_working_set_bytes=d.recommendedMaxWorkingSetSize;
    out->maximum_buffer_bytes=d.maxBufferLength;out->maximum_threads_per_group=1024;
    out->unified_memory=d.hasUnifiedMemory?1:0;out->supports_family_apple7=[d supportsFamily:MTLGPUFamilyApple7]?1:0;out->supports_f16=1;
    NSData *name=[d.name dataUsingEncoding:NSUTF8StringEncoding];size_t n=std::min<size_t>(name.length,sizeof(out->name)-1);std::memcpy(out->name,name.bytes,n);
    return PAI_METAL_OK; } @catch(NSException*) { return PAI_METAL_INTERNAL_FAILURE; } } }

PaiMetalStatus pai_metal_context_create(size_t index,const uint8_t*bytes,size_t length,PaiMetalContext**out){@autoreleasepool{@try{
    if(bytes==nullptr||length==0||out==nullptr)return PAI_METAL_INVALID_ARGUMENT;*out=nullptr;NSArray<id<MTLDevice>>*devices=available_devices();
    if(index>=devices.count)return PAI_METAL_INVALID_ARGUMENT;auto *context=new(std::nothrow) PaiMetalContext();if(!context)return PAI_METAL_ALLOCATION_FAILURE;
    context->device=devices[index];context->queue=[context->device newCommandQueue];if(!context->queue){delete context;return PAI_METAL_ALLOCATION_FAILURE;}
    void*copy=std::malloc(length);if(!copy){delete context;return PAI_METAL_ALLOCATION_FAILURE;}std::memcpy(copy,bytes,length);
    dispatch_data_t data=dispatch_data_create(copy,length,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),DISPATCH_DATA_DESTRUCTOR_FREE);
    NSError*error=nil;id<MTLLibrary>library=[context->device newLibraryWithData:data error:&error];if(!library){delete context;return PAI_METAL_LIBRARY_FAILURE;}
    NSMutableArray *pipelines=[NSMutableArray arrayWithCapacity:PAI_METAL_KERNEL_COUNT];
    for(size_t i=0;i<PAI_METAL_KERNEL_COUNT;i++){NSString*n=[NSString stringWithUTF8String:kKernelNames[i]];id<MTLFunction>f=[library newFunctionWithName:n];
      if(!f){delete context;return PAI_METAL_LIBRARY_FAILURE;}id<MTLComputePipelineState>p=[context->device newComputePipelineStateWithFunction:f error:&error];
      if(!p){delete context;return PAI_METAL_PIPELINE_FAILURE;}[pipelines addObject:p];}
    context->pipelines=[pipelines copy];*out=context;return PAI_METAL_OK;
 }@catch(NSException*){return PAI_METAL_INTERNAL_FAILURE;}}}
void pai_metal_context_destroy(PaiMetalContext*context){if(context)delete context;}

PaiMetalStatus pai_metal_buffer_create(PaiMetalContext*c,size_t length,size_t alignment,PaiMetalStorageMode mode,PaiMetalBuffer**out){@autoreleasepool{@try{
 if(!c||!out||length==0||alignment==0||(alignment&(alignment-1))||alignment>4096||
   (mode!=PAI_METAL_STORAGE_SHARED&&mode!=PAI_METAL_STORAGE_PRIVATE)) return fail(c,PAI_METAL_INVALID_ARGUMENT,"invalid Metal allocation");
 *out=nullptr;
 MTLResourceOptions options=mode==PAI_METAL_STORAGE_SHARED?MTLResourceStorageModeShared:MTLResourceStorageModePrivate;
 id<MTLBuffer>b=[c->device newBufferWithLength:length options:options];if(!b)return fail(c,PAI_METAL_ALLOCATION_FAILURE,"Metal buffer allocation failed");
 if(mode==PAI_METAL_STORAGE_SHARED&&(reinterpret_cast<uintptr_t>(b.contents)%alignment)!=0)return fail(c,PAI_METAL_ALLOCATION_FAILURE,"Metal buffer alignment failed");
 if(mode==PAI_METAL_STORAGE_SHARED)std::memset(b.contents,0,length);
 auto*wrapper=new(std::nothrow) PaiMetalBuffer{c,b,length,mode};if(!wrapper)return fail(c,PAI_METAL_ALLOCATION_FAILURE,"Metal buffer wrapper allocation failed");
 {std::lock_guard<std::mutex>lock(c->mutex);if(UINT64_MAX-c->statistics.current_allocated_bytes<length){delete wrapper;return PAI_METAL_ALLOCATION_FAILURE;}
  c->statistics.current_allocated_bytes+=length;c->statistics.peak_allocated_bytes=std::max(c->statistics.peak_allocated_bytes,c->statistics.current_allocated_bytes);c->statistics.allocation_count++;}
 *out=wrapper;return PAI_METAL_OK;}@catch(NSException*){return fail(c,PAI_METAL_INTERNAL_FAILURE,"Metal allocation exception");}}}
void pai_metal_buffer_destroy(PaiMetalBuffer*b){if(!b)return;PaiMetalContext*c=b->owner;{std::lock_guard<std::mutex>lock(c->mutex);c->statistics.current_allocated_bytes-=b->length;}delete b;}
size_t pai_metal_buffer_length(const PaiMetalBuffer*b){return b?b->length:0;}
void*pai_metal_buffer_contents(PaiMetalBuffer*b){return b&&b->mode==PAI_METAL_STORAGE_SHARED?b->buffer.contents:nullptr;}

static PaiMetalStatus wait(PaiMetalContext*c,id<MTLCommandBuffer>command){[command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted){
 std::lock_guard<std::mutex>lock(c->mutex);c->statistics.command_failures++;c->error="Metal command failed";return PAI_METAL_COMMAND_FAILURE;}return PAI_METAL_OK;}
PaiMetalStatus pai_metal_upload(PaiMetalContext*c,const uint8_t*src,size_t length,PaiMetalBuffer*dst){@autoreleasepool{@try{
 if(!c||!src||!dst||dst->owner!=c||length!=dst->length)return fail(c,PAI_METAL_INVALID_ARGUMENT,"invalid host-to-device transfer");
 if(dst->mode==PAI_METAL_STORAGE_SHARED)std::memcpy(dst->buffer.contents,src,length);else{id<MTLBuffer>staging=[c->device newBufferWithBytes:src length:length options:MTLResourceStorageModeShared];
  if(!staging)return fail(c,PAI_METAL_ALLOCATION_FAILURE,"Metal upload staging failed");id<MTLCommandBuffer>cmd=[c->queue commandBuffer];id<MTLBlitCommandEncoder>e=[cmd blitCommandEncoder];
  [e copyFromBuffer:staging sourceOffset:0 toBuffer:dst->buffer destinationOffset:0 size:length];[e endEncoding];PaiMetalStatus s=wait(c,cmd);if(s!=PAI_METAL_OK)return s;}
 {std::lock_guard<std::mutex>lock(c->mutex);c->statistics.host_to_device_bytes+=length;}return PAI_METAL_OK;}@catch(NSException*){return fail(c,PAI_METAL_INTERNAL_FAILURE,"Metal upload exception");}}}
PaiMetalStatus pai_metal_download(PaiMetalContext*c,const PaiMetalBuffer*src,uint8_t*dst,size_t length){@autoreleasepool{@try{
 if(!c||!src||!dst||src->owner!=c||length!=src->length)return fail(c,PAI_METAL_INVALID_ARGUMENT,"invalid device-to-host transfer");
 if(src->mode==PAI_METAL_STORAGE_SHARED)std::memcpy(dst,src->buffer.contents,length);else{id<MTLBuffer>staging=[c->device newBufferWithLength:length options:MTLResourceStorageModeShared];
  if(!staging)return fail(c,PAI_METAL_ALLOCATION_FAILURE,"Metal download staging failed");id<MTLCommandBuffer>cmd=[c->queue commandBuffer];id<MTLBlitCommandEncoder>e=[cmd blitCommandEncoder];
  [e copyFromBuffer:src->buffer sourceOffset:0 toBuffer:staging destinationOffset:0 size:length];[e endEncoding];PaiMetalStatus s=wait(c,cmd);if(s!=PAI_METAL_OK)return s;std::memcpy(dst,staging.contents,length);}
 {std::lock_guard<std::mutex>lock(c->mutex);c->statistics.device_to_host_bytes+=length;}return PAI_METAL_OK;}@catch(NSException*){return fail(c,PAI_METAL_INTERNAL_FAILURE,"Metal download exception");}}}
PaiMetalStatus pai_metal_upload_range(PaiMetalContext*c,const uint8_t*src,size_t offset,size_t length,PaiMetalBuffer*dst){@autoreleasepool{@try{
 if(!c||!src||!dst||dst->owner!=c||offset>dst->length||length>dst->length-offset||dst->mode!=PAI_METAL_STORAGE_SHARED)return fail(c,PAI_METAL_INVALID_ARGUMENT,"invalid ranged Metal upload");
 std::memcpy(static_cast<uint8_t*>(dst->buffer.contents)+offset,src,length);{std::lock_guard<std::mutex>lock(c->mutex);c->statistics.host_to_device_bytes+=length;}return PAI_METAL_OK;
 }@catch(NSException*){return fail(c,PAI_METAL_INTERNAL_FAILURE,"Metal ranged upload exception");}}}
PaiMetalStatus pai_metal_download_range(PaiMetalContext*c,const PaiMetalBuffer*src,size_t offset,uint8_t*dst,size_t length){@autoreleasepool{@try{
 if(!c||!src||!dst||src->owner!=c||offset>src->length||length>src->length-offset||src->mode!=PAI_METAL_STORAGE_SHARED)return fail(c,PAI_METAL_INVALID_ARGUMENT,"invalid ranged Metal download");
 std::memcpy(dst,static_cast<const uint8_t*>(src->buffer.contents)+offset,length);{std::lock_guard<std::mutex>lock(c->mutex);c->statistics.device_to_host_bytes+=length;}return PAI_METAL_OK;
 }@catch(NSException*){return fail(c,PAI_METAL_INTERNAL_FAILURE,"Metal ranged download exception");}}}
PaiMetalStatus pai_metal_buffer_clear(PaiMetalContext*c,PaiMetalBuffer*b){@autoreleasepool{@try{
 if(!c||!b||b->owner!=c)return fail(c,PAI_METAL_INVALID_ARGUMENT,"invalid Metal clear");if(b->mode==PAI_METAL_STORAGE_SHARED)std::memset(b->buffer.contents,0,b->length);else{
  id<MTLCommandBuffer>cmd=[c->queue commandBuffer];id<MTLBlitCommandEncoder>e=[cmd blitCommandEncoder];[e fillBuffer:b->buffer range:NSMakeRange(0,b->length) value:0];[e endEncoding];return wait(c,cmd);}return PAI_METAL_OK;
 }@catch(NSException*){return fail(c,PAI_METAL_INTERNAL_FAILURE,"Metal clear exception");}}}

PaiMetalStatus pai_metal_dispatch(PaiMetalContext*c,PaiMetalKernel kernel,PaiMetalBuffer*const*buffers,const size_t*offsets,size_t count,
 const PaiMetalKernelParams*p,uint32_t gx,uint32_t gy,uint32_t gz){@autoreleasepool{@try{
 if(!c||!buffers||!offsets||!p||kernel<0||kernel>=PAI_METAL_KERNEL_COUNT||count!=expected_buffers(kernel)||gx==0||gy==0||gz==0)
  return fail(c,PAI_METAL_INVALID_ARGUMENT,"invalid Metal dispatch");
 for(size_t i=0;i<count;i++)if(!buffers[i]||buffers[i]->owner!=c)return fail(c,PAI_METAL_INVALID_ARGUMENT,"foreign Metal buffer");
 if(!validate_lengths(kernel,buffers,offsets,count,*p))return fail(c,PAI_METAL_INVALID_ARGUMENT,"Metal dispatch range is invalid");
 id<MTLCommandBuffer>cmd=[c->queue commandBuffer];id<MTLComputeCommandEncoder>e=[cmd computeCommandEncoder];id<MTLComputePipelineState>pipeline=c->pipelines[(NSUInteger)kernel];[e setComputePipelineState:pipeline];
 for(size_t i=0;i<count;i++)[e setBuffer:buffers[i]->buffer offset:offsets[i] atIndex:i];[e setBytes:p length:sizeof(*p) atIndex:4];
 MTLSize grid=MTLSizeMake(gx,gy,gz);MTLSize group=gy>1?MTLSizeMake(std::min<uint32_t>(8,gx),std::min<uint32_t>(8,gy),1):MTLSizeMake(std::min<uint32_t>(64,gx),1,1);
 [e dispatchThreads:grid threadsPerThreadgroup:group];[e endEncoding];PaiMetalStatus s=wait(c,cmd);if(s!=PAI_METAL_OK)return s;
 {std::lock_guard<std::mutex>lock(c->mutex);c->statistics.dispatch_count++;}return PAI_METAL_OK;}@catch(NSException*){return fail(c,PAI_METAL_INTERNAL_FAILURE,"Metal dispatch exception");}}}
PaiMetalStatus pai_metal_synchronize(PaiMetalContext*c){return c?PAI_METAL_OK:PAI_METAL_INVALID_ARGUMENT;}
PaiMetalStatus pai_metal_statistics(PaiMetalContext*c,PaiMetalStatistics*out){if(!c||!out)return PAI_METAL_INVALID_ARGUMENT;std::lock_guard<std::mutex>lock(c->mutex);*out=c->statistics;return PAI_METAL_OK;}
size_t pai_metal_last_error(PaiMetalContext*c,char*dst,size_t cap){if(!c)return 0;std::lock_guard<std::mutex>lock(c->mutex);size_t n=c->error.size();if(dst&&cap){size_t copy=std::min(n,cap-1);std::memcpy(dst,c->error.data(),copy);dst[copy]=0;}return n;}
