kernel void pai_softmax(device const float *a [[buffer(0)]], device float *o [[buffer(1)]],
 constant PaiMetalKernelParams &p [[buffer(4)]], uint r [[thread_position_in_grid]]) {
 if(r>=p.rows)return;uint b=r*p.columns;float m=-INFINITY;for(uint i=0;i<p.columns;i++)m=max(m,a[b+i]);
 float s=0;for(uint i=0;i<p.columns;i++){o[b+i]=exp(a[b+i]-m);s+=o[b+i];}for(uint i=0;i<p.columns;i++)o[b+i]/=s; }
