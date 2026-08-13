kernel void pai_aipq8_matvec(device const float*s [[buffer(0)]],device const char*c [[buffer(1)]],device const float*v [[buffer(2)]],device float*o [[buffer(3)]],
 constant PaiMetalKernelParams&p [[buffer(4)]],uint r [[thread_position_in_grid]]){if(r>=p.rows)return;uint bpr=(p.columns+31)/32;float sum=0;
 for(uint x=0;x<p.columns;x++){uint b=r*bpr+x/32;sum=fma(s[b]*float(c[b*32+x%32]),v[x],sum);}o[r]=sum;}
kernel void pai_aipq4_matvec(device const float*s [[buffer(0)]],device const uchar*c [[buffer(1)]],device const float*v [[buffer(2)]],device float*o [[buffer(3)]],
 constant PaiMetalKernelParams&p [[buffer(4)]],uint r [[thread_position_in_grid]]){if(r>=p.rows)return;uint bpr=(p.columns+31)/32;float sum=0;
 for(uint x=0;x<p.columns;x++){uint b=r*bpr;uchar z=c[b*16+(x%32)/2],n=(x&1)?z>>4:z&15;sum=fma(s[b+x/32]*float(pai_q4_decode(n)),v[x],sum);}o[r]=sum;}
