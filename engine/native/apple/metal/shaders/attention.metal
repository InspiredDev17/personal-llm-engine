kernel void pai_embedding(device const float *a [[buffer(0)]], device float *o [[buffer(1)]],
 constant PaiMetalKernelParams &p [[buffer(4)]], uint i [[thread_position_in_grid]]) {if(i<p.columns)o[i]=a[p.position*p.columns+i];}
kernel void pai_attention_scores(device const float *q [[buffer(0)]],device const float *k [[buffer(1)]],device float *o [[buffer(2)]],
 constant PaiMetalKernelParams&p [[buffer(4)]],uint2 i [[thread_position_in_grid]]){uint pos=i.x,qh=i.y;if(pos>=p.sequence_length||qh>=p.head_count)return;
 uint kh=qh/(p.head_count/p.kv_head_count);float s=0;for(uint d=0;d<p.head_dimension;d++)s=fma(q[qh*p.head_dimension+d],k[(pos*p.kv_head_count+kh)*p.head_dimension+d],s);
 o[qh*p.sequence_length+pos]=s*p.scalar0;}
kernel void pai_causal_mask(device const float*a [[buffer(0)]],device float*o [[buffer(1)]],constant PaiMetalKernelParams&p [[buffer(4)]],uint2 i [[thread_position_in_grid]]){
 if(i.y>=p.query_count||i.x>=p.columns)return;uint n=i.y*p.columns+i.x;o[n]=i.x>p.first_query_position+i.y?-INFINITY:a[n];}
kernel void pai_attention_values(device const float*prob [[buffer(0)]],device const float*v [[buffer(1)]],device float*o [[buffer(2)]],
 constant PaiMetalKernelParams&p [[buffer(4)]],uint2 i [[thread_position_in_grid]]){uint d=i.x,qh=i.y;if(d>=p.head_dimension||qh>=p.head_count)return;
 uint kh=qh/(p.head_count/p.kv_head_count);float s=0;for(uint pos=0;pos<p.sequence_length;pos++)s=fma(prob[qh*p.sequence_length+pos],v[(pos*p.kv_head_count+kh)*p.head_dimension+d],s);o[qh*p.head_dimension+d]=s;}
