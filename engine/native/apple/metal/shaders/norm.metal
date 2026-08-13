kernel void pai_rms_norm(device const float *a [[buffer(0)]], device const float *w [[buffer(1)]],
 device float *o [[buffer(2)]], constant PaiMetalKernelParams &p [[buffer(4)]],
 uint r [[thread_position_in_grid]]) { if(r>=p.rows)return;uint b=r*p.columns;float s=0;
 for(uint i=0;i<p.columns;i++)s=fma(a[b+i],a[b+i],s);float inv=rsqrt(s/float(p.columns)+p.scalar0);
 for(uint i=0;i<p.columns;i++)o[b+i]=a[b+i]*inv*w[i]; }
