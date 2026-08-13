kernel void pai_silu(device const float *a [[buffer(0)]], device float *o [[buffer(1)]],
 constant PaiMetalKernelParams &p [[buffer(4)]], uint i [[thread_position_in_grid]]) {if(i<p.count)o[i]=a[i]/(1+exp(-a[i]));}
kernel void pai_swiglu(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
 device float *o [[buffer(2)]], constant PaiMetalKernelParams &p [[buffer(4)]],
 uint i [[thread_position_in_grid]]) {if(i<p.count)o[i]=(a[i]/(1+exp(-a[i])))*b[i];}
