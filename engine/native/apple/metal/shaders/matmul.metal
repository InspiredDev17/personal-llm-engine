kernel void pai_f32_matvec(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
 device float *o [[buffer(2)]], constant PaiMetalKernelParams &p [[buffer(4)]],
 uint r [[thread_position_in_grid]]) { if(r>=p.rows)return; float s=0; for(uint k=0;k<p.columns;k++)s=fma(a[r*p.columns+k],b[k],s);o[r]=s; }
kernel void pai_f32_matmul(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
 device float *o [[buffer(2)]], constant PaiMetalKernelParams &p [[buffer(4)]],
 uint2 i [[thread_position_in_grid]]) { if(i.y>=p.rows||i.x>=p.columns)return; float s=0;
 for(uint k=0;k<p.inner;k++)s=fma(a[i.y*p.inner+k],b[k*p.columns+i.x],s);o[i.y*p.columns+i.x]=s; }
kernel void pai_f16_matvec(device const half *a [[buffer(0)]], device const half *b [[buffer(1)]],
 device float *o [[buffer(2)]], constant PaiMetalKernelParams &p [[buffer(4)]],
 uint r [[thread_position_in_grid]]) { if(r>=p.rows)return; float s=0; for(uint k=0;k<p.columns;k++)s=fma(float(a[r*p.columns+k]),float(b[k]),s);o[r]=s; }
kernel void pai_f16_matmul(device const half *a [[buffer(0)]], device const half *b [[buffer(1)]],
 device float *o [[buffer(2)]], constant PaiMetalKernelParams &p [[buffer(4)]],
 uint2 i [[thread_position_in_grid]]) { if(i.y>=p.rows||i.x>=p.columns)return; float s=0;
 for(uint k=0;k<p.inner;k++)s=fma(float(a[i.y*p.inner+k]),float(b[k*p.columns+i.x]),s);o[i.y*p.columns+i.x]=s; }
