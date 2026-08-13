#include <metal_stdlib>
using namespace metal;
struct PaiMetalKernelParams { uint count, rows, columns, inner, position, head_dimension,
  head_count, kv_head_count, sequence_length, query_count, first_query_position, block_count;
  float scalar0, scalar1, scalar2, scalar3; };
inline int pai_q4_decode(uchar nibble) { return nibble <= 7 ? int(nibble) : int(nibble) - 15; }
