kernel void pai_buffer_copy(device const uchar *a [[buffer(0)]], device uchar *o [[buffer(1)]],
 constant PaiMetalKernelParams &p [[buffer(4)]], uint i [[thread_position_in_grid]]) { if(i<p.count)o[i]=a[i]; }
kernel void pai_vector_add(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
 device float *o [[buffer(2)]], constant PaiMetalKernelParams &p [[buffer(4)]],
 uint i [[thread_position_in_grid]]) { if(i<p.count)o[i]=a[i]+b[i]; }
kernel void pai_vector_multiply(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
 device float *o [[buffer(2)]], constant PaiMetalKernelParams &p [[buffer(4)]],
 uint i [[thread_position_in_grid]]) { if(i<p.count)o[i]=a[i]*b[i]; }
kernel void pai_scalar_multiply(device const float *a [[buffer(0)]], device float *o [[buffer(1)]],
 constant PaiMetalKernelParams &p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
 if(i<p.count)o[i]=a[i]*p.scalar0; }
