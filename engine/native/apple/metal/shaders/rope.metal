kernel void pai_rope(device const float *a [[buffer(0)]], device float *o [[buffer(1)]],
 constant PaiMetalKernelParams &p [[buffer(4)]], uint pair [[thread_position_in_grid]]) {
 uint ph=p.head_dimension/2,total=p.head_count*ph;if(pair>=total)return;uint d=pair%ph,h=pair/ph,x=h*p.head_dimension+d*2;
 float angle=float(p.position)*pow(p.scalar0,-2*float(d)/float(p.head_dimension)),c=cos(angle),s=sin(angle);
 float l=a[x],r=a[x+1];o[x]=l*c-r*s;o[x+1]=l*s+r*c; }
