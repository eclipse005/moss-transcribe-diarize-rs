// GPU element-wise kernels for the MOSS-Transcribe-Diarize decoder + encoder.
// All arithmetic accumulates in f32 but storage is BF16 (matches the Python
// original's bfloat16 compute, for node-by-node token-level alignment).
// Targets sm_61+ (bf16 has no native compute units on sm_61; cuBLAS uses an
// emulated path, and these kernels convert bf16→f32 for all math).
//
// Implementation note: we remap the __half* intrinsics to __nv_bfloat16*
// equivalents via macros below, so the kernel bodies read identically to the
// f16 originals while operating on bf16 storage. Vectorized __half2 loads are
// preserved as __nv_bfloat162 (still a valid 32-bit memory transaction even
// without bf16 ALU support).

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

// ── BF16 remap macros ────────────────────────────────────────────────
// Map the f16 intrinsic names used throughout this file to bf16 equivalents.
#define __half          __nv_bfloat16
#define __half2         __nv_bfloat162
#define __half2float    __bfloat162float
#define __float2half    __float2bfloat16
#define __halves2half2  __halves2bfloat162

// ─── RMS norm: x [outer, last] * weight[last] → out, with f32 accumulation ──
// One block per row; block_size threads cooperate over `last`.
// Uses __half2 vectorized loads for 2x memory throughput.
// Shared mem: block_size * sizeof(float).
extern "C" __global__ void __launch_bounds__(1024, 2)
rms_norm_f16(
    __half* __restrict__ out,
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    int last,
    int outer,
    float eps
) {
    int row = blockIdx.x;
    if (row >= outer) return;
    int tid = threadIdx.x;
    int bs = blockDim.x;

    extern __shared__ float sdata[];

    int last2 = last / 2;
    const __half2* x2 = (const __half2*)(x + row * last);
    const __half2* w2 = (const __half2*)w;

    float local = 0.0f;
    for (int j = tid; j < last2; j += bs) {
        __half2 v = x2[j];
        float vx = __half2float(v.x), vy = __half2float(v.y);
        local += vx * vx + vy * vy;
    }
    if (last & 1) {
        // Handle odd trailing element (unlikely for model dims but safe)
        float v = __half2float(x[row * last + last - 1]);
        local += v * v;
    }
    sdata[tid] = local;
    __syncthreads();

    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sdata[0] / (float)last + eps);

    __half2* o2 = (__half2*)(out + row * last);
    for (int j = tid; j < last2; j += bs) {
        __half2 xv = x2[j];
        __half2 wv = w2[j];
        float vx = __half2float(xv.x) * inv_rms * __half2float(wv.x);
        float vy = __half2float(xv.y) * inv_rms * __half2float(wv.y);
        o2[j] = __halves2half2(__float2half(vx), __float2half(vy));
    }
    if (last & 1) {
        float v = __half2float(x[row * last + last - 1]) * inv_rms * __half2float(w[last - 1]);
        out[row * last + last - 1] = __float2half(v);
    }
}

// ─── Fused: residual_inplace = residual + add_in; out = rms_norm(residual_inplace, w) ──
// Writes `residual_inplace` with the residual sum AND `out` with the normed result.
// Uses __half2 vectorized loads for 2x memory throughput.
// Shared mem: bs * sizeof(float).
extern "C" __global__ void __launch_bounds__(1024, 2)
add_residual_rms_norm_f16(
    __half* __restrict__ residual_inplace,   // r' = r + a
    __half* __restrict__ out,                // out = rms_norm(r', w)
    const __half* __restrict__ add_in,
    const __half* __restrict__ w,
    int last,
    int outer,
    float eps
) {
    int row = blockIdx.x;
    if (row >= outer) return;
    int tid = threadIdx.x;
    int bs = blockDim.x;

    extern __shared__ float sdata[];

    int last2 = last / 2;
    const __half2* ri2 = (const __half2*)(residual_inplace + row * last);
    const __half2* ai2 = (const __half2*)(add_in + row * last);
    __half2* o2 = (__half2*)(residual_inplace + row * last);

    // Pass 1: residual sum + sum of squares (vectorized)
    float local = 0.0f;
    for (int j = tid; j < last2; j += bs) {
        __half2 rv = ri2[j];
        __half2 av = ai2[j];
        float rx = __half2float(rv.x), ry = __half2float(rv.y);
        float ax = __half2float(av.x), ay = __half2float(av.y);
        float vx = rx + ax, vy = ry + ay;
        o2[j] = __halves2half2(__float2half(vx), __float2half(vy));
        local += vx * vx + vy * vy;
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sdata[0] / (float)last + eps);

    // Pass 2: normalize (vectorized)
    const __half2* norm2 = (const __half2*)(residual_inplace + row * last);
    const __half2* w2 = (const __half2*)w;
    __half2* out2 = (__half2*)(out + row * last);
    for (int j = tid; j < last2; j += bs) {
        __half2 nv = norm2[j];
        __half2 wv = w2[j];
        out2[j] = __halves2half2(
            __float2half(__half2float(nv.x) * inv_rms * __half2float(wv.x)),
            __float2half(__half2float(nv.y) * inv_rms * __half2float(wv.y))
        );
    }
}

// ─── Element-wise add: out = a + b ─────────────────────────────────
extern "C" __global__ void __launch_bounds__(1024, 2)
add_f16(
    __half* __restrict__ out,
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

// ─── In-place add: a += b ──────────────────────────────────────────
extern "C" __global__ void __launch_bounds__(1024, 2)
add_inplace_f16(
    __half* __restrict__ a,
    const __half* __restrict__ b,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

// ─── SiLU(gate) * up (SwiGLU activation) ───────────────────────────
extern "C" __global__ void __launch_bounds__(1024, 2)
silu_mul_f16(
    __half* __restrict__ out,
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]);
    float sig = 1.0f / (1.0f + expf(-g));
    out[i] = __float2half(g * sig * __half2float(up[i]));
}

// ─── Fused gate-up split + SiLU(gate)*up ──────────────────────────
// gu: [outer, 2*inter] in row-major (gate first half of last dim, up second half).
// Writes activated [outer, inter]. Uses __half2 vectorized loads.
extern "C" __global__ void __launch_bounds__(1024, 2)
silu_mul_split_f16(
    __half* __restrict__ out,
    const __half* __restrict__ gu,
    int outer,
    int inter
) {
    int inter2 = inter / 2;
    int total2 = outer * inter2;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total2) return;
    int o = i / inter2;
    int c = i % inter2;
    int base = o * 2 * inter;
    const __half2* g2 = (const __half2*)(gu + base);
    const __half2* u2 = (const __half2*)(gu + base + inter);
    __half2 gv = g2[c];
    __half2 uv = u2[c];
    float gx = __half2float(gv.x), gy = __half2float(gv.y);
    float ux = __half2float(uv.x), uy = __half2float(uv.y);
    float sx = 1.0f / (1.0f + __expf(-gx));
    float sy = 1.0f / (1.0f + __expf(-gy));
    ((__half2*)out)[i] = __halves2half2(__float2half(gx * sx * ux), __float2half(gy * sy * uy));
    if (inter & 1 && i == 0) {
        // Handle odd trailing element
        int last_c = inter - 1;
        float g = __half2float(gu[base + last_c]);
        float u = __half2float(gu[base + inter + last_c]);
        float sig = 1.0f / (1.0f + __expf(-g));
        out[o * inter + last_c] = __float2half(g * sig * u);
    }
}

// ─── In-place SiLU (VQAdaptor activation): x = x * sigmoid(x) ──────
extern "C" __global__ void __launch_bounds__(1024, 2)
silu_inplace_f16(
    __half* __restrict__ x,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(x[i]);
    float sig = 1.0f / (1.0f + __expf(-v));
    x[i] = __float2half(v * sig);
}

// ─── Softmax with scale + optional causal mask ─────────────────────
// x shape (logical [bh, m, n]) launched as grid = (bh * m,). One block per row.
// is_causal != 0 keeps positions [0..row_in_m+1) only.
extern "C" __global__ void __launch_bounds__(1024, 1)
softmax_scaled_causal_f16(
    __half* __restrict__ out,
    const __half* __restrict__ x,
    int m,
    int n,
    float scale,
    int is_causal
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int bs = blockDim.x;

    int m_idx = row % m;
    int valid_n = (is_causal != 0) ? (m_idx + 1) : n;

    extern __shared__ float sdata[];

    // Max
    float local_max = -INFINITY;
    for (int j = tid; j < valid_n; j += bs) {
        float v = __half2float(x[row * n + j]) * scale;
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // Sum
    float local_sum = 0.0f;
    for (int j = tid; j < valid_n; j += bs) {
        local_sum += expf(__half2float(x[row * n + j]) * scale - row_max);
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];

    for (int j = tid; j < n; j += bs) {
        if (j < valid_n) {
            float v = __half2float(x[row * n + j]) * scale - row_max;
            out[row * n + j] = __float2half(expf(v) * inv_sum);
        } else {
            out[row * n + j] = __float2half(0.0f);
        }
    }
}

// ─── Rotary embedding ──────────────────────────────────────────────
// x [b, h, s, d], cos/sin [total_s, d] (broadcast over b, h, indexed at pos_offset+is).
// rotate_half: for i<half → -x[i+half], for i>=half → x[i-half].
extern "C" __global__ void __launch_bounds__(1024, 2)
rotary_emb_f16(
    __half* __restrict__ out,
    const __half* __restrict__ x,
    const __half* __restrict__ cos,
    const __half* __restrict__ sin,
    int b, int h, int s, int d, int pos_offset
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * h * s * d;
    if (tot >= total) return;

    int id = tot % d;
    int is = (tot / d) % s;
    int half_ = d >> 1;

    float x_val = __half2float(x[tot]);
    float pair_val = (id < half_)
        ? -__half2float(x[tot + half_])
        :  __half2float(x[tot - half_]);

    int row = pos_offset + is;
    float c = __half2float(cos[row * d + id]);
    float si = __half2float(sin[row * d + id]);
    out[tot] = __float2half(x_val * c + pair_val * si);
}

// ─── Fused RMSNorm + rotary embedding (Q/K path) ───────────────────
// Per-row norm over head_dim, then rotary on the same row.
// x [b, h, s, d], weight [d], cos/sin [total_s, d] indexed at pos_offset+is.
// Grid = (b*h*s,); one block per (head, seq) row.
extern "C" __global__ void __launch_bounds__(128, 8)
rms_norm_rotary_f16(
    __half* __restrict__ out,
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    const __half* __restrict__ cos,
    const __half* __restrict__ sin,
    int b, int h, int s, int d, int pos_offset, float eps
) {
    int row = blockIdx.x;          // global row index over b*h*s
    int tid = threadIdx.x;
    int bs = blockDim.x;
    int total_rows = b * h * s;
    if (row >= total_rows) return;
    int is = row % s;
    int cs_row = pos_offset + is;

    extern __shared__ float sdata[];

    // Compute sum of squares
    float local = 0.0f;
    for (int j = tid; j < d; j += bs) {
        float v = __half2float(x[row * d + j]);
        local += v * v;
    }
    sdata[tid] = local;
    __syncthreads();
    for (int sh = bs >> 1; sh > 0; sh >>= 1) {
        if (tid < sh) sdata[tid] += sdata[tid + sh];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sdata[0] / (float)d + eps);

    // x_normed[j] computed in registers per j; need x_normed at pj for rotary.
    // Cheap workaround: load x[pj], scale by inv_rms * w[pj].
    int half_ = d >> 1;
    for (int j = tid; j < d; j += bs) {
        float x_val_j  = __half2float(x[row * d + j])  * inv_rms * __half2float(w[j]);
        int pj = (j < half_) ? (j + half_) : (j - half_);
        float x_pair   = __half2float(x[row * d + pj]) * inv_rms * __half2float(w[pj]);
        float pair_val = (j < half_) ? -x_pair : x_pair;
        float c = __half2float(cos[cs_row * d + j]);
        float si = __half2float(sin[cs_row * d + j]);
        out[row * d + j] = __float2half(x_val_j * c + pair_val * si);
    }
}

// ─── Repeat KV from sparse cache ───────────────────────────────────
// cache layout (per-layer): [b, nkvh, max_seq, d]; valid rows [0..cur_len).
// Output dst [b, nqh, cur_len, d] contiguous (cur_len-major within head).
extern "C" __global__ void __launch_bounds__(1024, 2)
repeat_kv_from_cache_f16(
    __half* __restrict__ dst,
    const __half* __restrict__ cache,
    int b, int nkvh, int max_seq, int d,
    int n_rep, int cur_len
) {
    int nqh = nkvh * n_rep;
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * nqh * cur_len * d;
    if (tot >= total) return;

    int id = tot % d;
    int is = (tot / d) % cur_len;
    int iq = (tot / (d * cur_len)) % nqh;
    int ib = tot / (d * cur_len * nqh);

    int ikv = iq / n_rep;
    int src_idx = ((ib * nkvh + ikv) * max_seq + is) * d + id;
    dst[tot] = cache[src_idx];
}

// ─── Embedding lookup ─────────────────────────────────────────────
extern "C" __global__ void __launch_bounds__(256, 4)
embed_lookup_f16(
    __half* __restrict__ out,
    const __half* __restrict__ table,
    const long long* __restrict__ ids,
    int n, int d
) {
    int i = blockIdx.x;
    if (i >= n) return;
    int tid = threadIdx.x;
    int bs = blockDim.x;
    long long id = ids[i];
    for (int j = tid; j < d; j += bs) {
        out[i * d + j] = table[id * d + j];
    }
}

// ─── Single-token embed lookup, id read from GPU i32 buffer ──────
// Used by the decode hot loop to chain argmax (writes ids[slot]) → embed (reads ids[slot])
// without an htod round-trip.
extern "C" __global__ void __launch_bounds__(1024, 1)
embed_lookup_single_i32_f16(
    __half* __restrict__ out,
    const __half* __restrict__ table,
    const int* __restrict__ ids,
    int slot,
    int d
) {
    int tid = threadIdx.x;
    int bs = blockDim.x;
    long long id = (long long)ids[slot];
    for (int j = tid; j < d; j += bs) {
        out[j] = table[id * (long long)d + j];
    }
}

// ─── Argmax over a vector of length n (single block) ───────────────
extern "C" __global__ void __launch_bounds__(1024, 1)
argmax_f16(
    int* __restrict__ out_idx,
    const __half* __restrict__ x,
    int n
) {
    int tid = threadIdx.x;
    int bs = blockDim.x;
    __shared__ float smax[1024];
    __shared__ int sidx[1024];

    float local_max = -INFINITY;
    int local_idx = 0;
    for (int i = tid; i < n; i += bs) {
        float v = __half2float(x[i]);
        if (v > local_max) { local_max = v; local_idx = i; }
    }
    smax[tid] = local_max;
    sidx[tid] = local_idx;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            if (smax[tid + s] > smax[tid]) {
                smax[tid] = smax[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) *out_idx = sidx[0];
}

// Same as argmax_f16 but writes into out_idx[slot] instead of out_idx[0].
// Uses __half2 vectorized loads for 2x memory throughput.
extern "C" __global__ void __launch_bounds__(1024, 1)
argmax_into_slot_f16(
    int* __restrict__ out_idx,
    const __half* __restrict__ x,
    int n,
    int slot
) {
    int tid = threadIdx.x;
    int bs = blockDim.x;
    __shared__ float smax[1024];
    __shared__ int sidx[1024];

    int n2 = n / 2;
    const __half2* x2 = (const __half2*)x;

    float local_max = -INFINITY;
    int local_idx = 0;
    for (int i = tid; i < n2; i += bs) {
        __half2 v = x2[i];
        float vx = __half2float(v.x);
        float vy = __half2float(v.y);
        int ix = i * 2;
        if (vx > local_max) { local_max = vx; local_idx = ix; }
        if (vy > local_max) { local_max = vy; local_idx = ix + 1; }
    }
    if ((n & 1) && tid == 0) {
        float v = __half2float(x[n - 1]);
        if (v > local_max) { local_max = v; local_idx = n - 1; }
    }
    smax[tid] = local_max;
    sidx[tid] = local_idx;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            if (smax[tid + s] > smax[tid]) {
                smax[tid] = smax[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) out_idx[slot] = sidx[0];
}

// ─── Swap dims 1 and 2 of a 4-D tensor ─────────────────────────────
// src [d0, d1, d2, d3] (contig) → dst [d0, d2, d1, d3] (contig)
extern "C" __global__ void __launch_bounds__(1024, 2)
swap_dims_12_f16(
    __half* __restrict__ dst,
    const __half* __restrict__ src,
    int d0, int d1, int d2, int d3
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d0 * d1 * d2 * d3;
    if (tot >= total) return;
    int i3 = tot % d3;
    int i1 = (tot / d3) % d1;
    int i2 = (tot / (d3 * d1)) % d2;
    int i0 = tot / (d3 * d1 * d2);
    int src_idx = ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
    dst[tot] = src[src_idx];
}

// ─── Split fused QKV into a single head group ──────────────────────
// qkv [b, s, total_cols], offset selects start column, h*d contiguous columns.
// dst [b, h, s, d] (transposed layout from src).
extern "C" __global__ void __launch_bounds__(1024, 2)
qkv_split_f16(
    __half* __restrict__ dst,
    const __half* __restrict__ qkv,
    int b, int s, int h, int d, int total_cols, int offset
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_total = b * h * s * d;
    if (tot >= dst_total) return;
    int id = tot % d;
    int is = (tot / d) % s;
    int ih = (tot / (d * s)) % h;
    int ib = tot / (d * s * h);
    int col = offset + ih * d + id;
    int src_idx = (ib * s + is) * total_cols + col;
    dst[tot] = qkv[src_idx];
}

// ─── Fused Q path: extract Q from QKV, RMSNorm per head row, apply rotary ──
// qkv [b, s, total_cols]; Q occupies columns [0..q_dim) with q_dim = nqh*d.
// Output q_out [b, nqh, s, d]; weight [d]; cos/sin [total_s, d] indexed at pos_offset+is.
// Grid = (b*nqh*s,) one block per (head, seq) row, threads cooperate over d.
extern "C" __global__ void __launch_bounds__(128, 8)
qkv_extract_q_norm_rotary_f16(
    __half* __restrict__ q_out,
    const __half* __restrict__ qkv,
    const __half* __restrict__ w,
    const __half* __restrict__ cos,
    const __half* __restrict__ sin,
    int b, int nqh, int s, int d, int total_cols, int pos_offset, float eps
) {
    int row = blockIdx.x;          // global row index over b*nqh*s
    int tid = threadIdx.x;
    int bs = blockDim.x;
    int total_rows = b * nqh * s;
    if (row >= total_rows) return;
    int is = row % s;
    int ih = (row / s) % nqh;
    int ib = row / (s * nqh);
    int cs_row = pos_offset + is;

    // qkv row pointer: Q is at columns [ih*d .. ih*d + d) in row (ib, is).
    const __half* src = qkv + (ib * s + is) * total_cols + ih * d;

    extern __shared__ float sdata[];

    // Sum of squares (load Q values inline; we read them twice but cheap vs DRAM trip).
    float local = 0.0f;
    for (int j = tid; j < d; j += bs) {
        float v = __half2float(src[j]);
        local += v * v;
    }
    sdata[tid] = local;
    __syncthreads();
    for (int sh = bs >> 1; sh > 0; sh >>= 1) {
        if (tid < sh) sdata[tid] += sdata[tid + sh];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sdata[0] / (float)d + eps);

    int half_ = d >> 1;
    for (int j = tid; j < d; j += bs) {
        float x_val_j  = __half2float(src[j])  * inv_rms * __half2float(w[j]);
        int pj = (j < half_) ? (j + half_) : (j - half_);
        float x_pair   = __half2float(src[pj]) * inv_rms * __half2float(w[pj]);
        float pair_val = (j < half_) ? -x_pair : x_pair;
        float c = __half2float(cos[cs_row * d + j]);
        float si = __half2float(sin[cs_row * d + j]);
        q_out[row * d + j] = __float2half(x_val_j * c + pair_val * si);
    }
}

// ─── Single-kernel QKV: extract Q, K, V from fused projection; norm+rotary on Q and K;
//     write Q to q_out, K to k_cache, V to v_cache.  One launch replaces two.
// Grid: x = b*s,  y = nqh + nkvh  (head slot)
// Block: d threads cooperate over one row of head_dim.
extern "C" __global__ void __launch_bounds__(128, 8)
qkv_extract_qkv_norm_rotary_cache_f16(
    __half* __restrict__ q_out,
    __half* __restrict__ k_cache,
    __half* __restrict__ v_cache,
    const __half* __restrict__ qkv,
    const __half* __restrict__ qn_w,
    const __half* __restrict__ kn_w,
    const __half* __restrict__ cos,
    const __half* __restrict__ sin,
    int b, int nqh, int nkvh, int s, int d, int total_cols,
    int q_dim, int kv_dim, int max_seq, int start, int pos_offset, float eps
) {
    int bx = blockIdx.x;
    int hy = blockIdx.y;
    int ib = bx / s;
    int is = bx - ib * s;
    int tid = threadIdx.x;
    int bs = blockDim.x;
    int cs_row = pos_offset + is;
    int half_ = d >> 1;

    extern __shared__ float sdata[];

    if (hy < nqh) {
        // Q path
        int ih = hy;
        const __half* src = qkv + (ib * s + is) * total_cols + ih * d;
        __half* dst = q_out + ((ib * nqh + ih) * s + is) * d;
        float local = 0.0f;
        for (int j = tid; j < d; j += bs) {
            float v = __half2float(src[j]);
            local += v * v;
        }
        sdata[tid] = local;
        __syncthreads();
        for (int sh = bs >> 1; sh > 0; sh >>= 1) {
            if (tid < sh) sdata[tid] += sdata[tid + sh];
            __syncthreads();
        }
        float inv_rms = rsqrtf(sdata[0] / (float)d + eps);
        for (int j = tid; j < d; j += bs) {
            float x_val_j = __half2float(src[j]) * inv_rms * __half2float(qn_w[j]);
            int pj = (j < half_) ? (j + half_) : (j - half_);
            float x_pair = __half2float(src[pj]) * inv_rms * __half2float(qn_w[pj]);
            float pair_val = (j < half_) ? -x_pair : x_pair;
            float c = __half2float(cos[cs_row * d + j]);
            float si = __half2float(sin[cs_row * d + j]);
            dst[j] = __float2half(x_val_j * c + pair_val * si);
        }
    } else {
        // K+V path
        int ih = hy - nqh;
        if (ih >= nkvh) return;
        const __half* k_src = qkv + (ib * s + is) * total_cols + q_dim + ih * d;
        const __half* v_src = qkv + (ib * s + is) * total_cols + q_dim + kv_dim + ih * d;
        int cache_idx = ((ib * nkvh + ih) * max_seq + (start + is)) * d;
        float local = 0.0f;
        for (int j = tid; j < d; j += bs) {
            float v = __half2float(k_src[j]);
            local += v * v;
        }
        sdata[tid] = local;
        __syncthreads();
        for (int sh = bs >> 1; sh > 0; sh >>= 1) {
            if (tid < sh) sdata[tid] += sdata[tid + sh];
            __syncthreads();
        }
        float inv_rms = rsqrtf(sdata[0] / (float)d + eps);
        for (int j = tid; j < d; j += bs) {
            float k_val_j = __half2float(k_src[j]) * inv_rms * __half2float(kn_w[j]);
            int pj = (j < half_) ? (j + half_) : (j - half_);
            float k_pair = __half2float(k_src[pj]) * inv_rms * __half2float(kn_w[pj]);
            float pair_val = (j < half_) ? -k_pair : k_pair;
            float c = __half2float(cos[cs_row * d + j]);
            float si = __half2float(sin[cs_row * d + j]);
            k_cache[cache_idx + j] = __float2half(k_val_j * c + pair_val * si);
            v_cache[cache_idx + j] = v_src[j];
        }
    }
}

// ─── Fused KV path: extract K (with norm+rotary) and V from QKV, write both into KV cache ──
// qkv [b, s, total_cols]; K at cols [q_dim..q_dim+kv_dim), V at [q_dim+kv_dim..q_dim+2*kv_dim).
// Writes k_cache [b, nkvh, max_seq, d] and v_cache [b, nkvh, max_seq, d] at rows [start..start+s).
// Grid = (b*nkvh*s,) one block per (kv_head, seq) row.
extern "C" __global__ void __launch_bounds__(128, 8)
qkv_extract_kv_norm_rotary_cache_f16(
    __half* __restrict__ k_cache,
    __half* __restrict__ v_cache,
    const __half* __restrict__ qkv,
    const __half* __restrict__ kn_w,
    const __half* __restrict__ cos,
    const __half* __restrict__ sin,
    int b, int nkvh, int s, int d, int total_cols,
    int q_dim, int kv_dim, int max_seq, int start, int pos_offset, float eps
) {
    int row = blockIdx.x;          // global row index over b*nkvh*s
    int tid = threadIdx.x;
    int bs = blockDim.x;
    int total_rows = b * nkvh * s;
    if (row >= total_rows) return;
    int is = row % s;
    int ih = (row / s) % nkvh;
    int ib = row / (s * nkvh);
    int cs_row = pos_offset + is;

    const __half* k_src = qkv + (ib * s + is) * total_cols + q_dim + ih * d;
    const __half* v_src = qkv + (ib * s + is) * total_cols + q_dim + kv_dim + ih * d;
    int cache_idx = ((ib * nkvh + ih) * max_seq + (start + is)) * d;

    extern __shared__ float sdata[];

    // K: sum of squares (norm), then rotary + cache write
    float local = 0.0f;
    for (int j = tid; j < d; j += bs) {
        float v = __half2float(k_src[j]);
        local += v * v;
    }
    sdata[tid] = local;
    __syncthreads();
    for (int sh = bs >> 1; sh > 0; sh >>= 1) {
        if (tid < sh) sdata[tid] += sdata[tid + sh];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sdata[0] / (float)d + eps);

    int half_ = d >> 1;
    for (int j = tid; j < d; j += bs) {
        float k_val_j  = __half2float(k_src[j])  * inv_rms * __half2float(kn_w[j]);
        int pj = (j < half_) ? (j + half_) : (j - half_);
        float k_pair   = __half2float(k_src[pj]) * inv_rms * __half2float(kn_w[pj]);
        float pair_val = (j < half_) ? -k_pair : k_pair;
        float c = __half2float(cos[cs_row * d + j]);
        float si = __half2float(sin[cs_row * d + j]);
        k_cache[cache_idx + j] = __float2half(k_val_j * c + pair_val * si);
        // V: direct copy (no norm, no rotary).
        v_cache[cache_idx + j] = v_src[j];
    }
}

// ─── KV cache write ───────────────────────────────────────────────
// k_new [b, nkvh, s_new, d] (contig) → cache [b, nkvh, max_seq, d] at rows [start..start+s_new).
extern "C" __global__ void __launch_bounds__(1024, 2)
kv_cache_write_f16(
    __half* __restrict__ cache,
    const __half* __restrict__ k_new,
    int b, int nkvh, int max_seq, int d,
    int start, int s_new
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * nkvh * s_new * d;
    if (tot >= total) return;
    int id = tot % d;
    int isn = (tot / d) % s_new;
    int ih = (tot / (d * s_new)) % nkvh;
    int ib = tot / (d * s_new * nkvh);
    int dst_idx = ((ib * nkvh + ih) * max_seq + (start + isn)) * d + id;
    cache[dst_idx] = k_new[tot];
}

// ─── Fused KV cache write (K + V in one launch) ───────────────────
extern "C" __global__ void __launch_bounds__(1024, 2)
kv_cache_write_pair_f16(
    __half* __restrict__ k_cache,
    __half* __restrict__ v_cache,
    const __half* __restrict__ k_new,
    const __half* __restrict__ v_new,
    int b, int nkvh, int max_seq, int d,
    int start, int s_new
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * nkvh * s_new * d;
    if (tot >= total) return;
    int id = tot % d;
    int isn = (tot / d) % s_new;
    int ih = (tot / (d * s_new)) % nkvh;
    int ib = tot / (d * s_new * nkvh);
    int dst_idx = ((ib * nkvh + ih) * max_seq + (start + isn)) * d + id;
    k_cache[dst_idx] = k_new[tot];
    v_cache[dst_idx] = v_new[tot];
}

// ─── Element-wise GELU (audio encoder activation) ─────────────────
// Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
extern "C" __global__ void __launch_bounds__(1024, 2)
gelu_f16(
    __half* __restrict__ out,
    const __half* __restrict__ x,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(x[i]);
    float g = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
    out[i] = __float2half(g);
}

// ─── In-place GELU ─────────────────────────────────────────────────
extern "C" __global__ void __launch_bounds__(1024, 2)
gelu_inplace_f16(
    __half* __restrict__ x,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(x[i]);
    float g = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
    x[i] = __float2half(g);
}

// ─── LayerNorm (with bias) ─────────────────────────────────────────
// x [outer, last] * weight[last] + bias[last]; mean/var per row.
extern "C" __global__ void __launch_bounds__(1024, 2)
layer_norm_f16(
    __half* __restrict__ out,
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    const __half* __restrict__ bias,
    int last,
    int outer,
    float eps
) {
    int row = blockIdx.x;
    if (row >= outer) return;
    int tid = threadIdx.x;
    int bs = blockDim.x;
    extern __shared__ float sdata[];

    // Sum and sum of squares
    float l_sum = 0.0f, l_sq = 0.0f;
    for (int j = tid; j < last; j += bs) {
        float v = __half2float(x[row * last + j]);
        l_sum += v;
        l_sq  += v * v;
    }
    sdata[tid] = l_sum;
    sdata[tid + bs] = l_sq;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid]     += sdata[tid + s];
            sdata[tid + bs] += sdata[tid + bs + s];
        }
        __syncthreads();
    }
    float mean = sdata[0] / (float)last;
    float var  = sdata[bs] / (float)last - mean * mean;
    float inv_std = rsqrtf(var + eps);

    for (int j = tid; j < last; j += bs) {
        float v = (__half2float(x[row * last + j]) - mean) * inv_std;
        out[row * last + j] = __float2half(v * __half2float(w[j]) + __half2float(bias[j]));
    }
}

// ─── Add bias broadcast (for Linear with bias) ─────────────────────
// x [outer, last] += bias[last]
extern "C" __global__ void __launch_bounds__(1024, 2)
add_bias_inplace_f16(
    __half* __restrict__ x,
    const __half* __restrict__ bias,
    int outer, int last
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    if (tot >= outer * last) return;
    int j = tot % last;
    x[tot] = __float2half(__half2float(x[tot]) + __half2float(bias[j]));
}

// ─── Fused GQA attention (decode-only: s_q = 1) ────────────────────
// Q  [b, nqh, 1, d]  (current query token)
// Kc [b, nkvh, max_seq, d]  (KV cache, valid rows [0..cur_len))
// Vc [b, nkvh, max_seq, d]
// out [b, nqh, 1, d]
// Computes scores = Q · K^T * scale → softmax → attn · V, all in one kernel.
// Each block handles one (b, q_head, head_dim) — but we collapse to one block per
// (b, q_head); threads cooperate over head_dim and (then) over cur_len.
// Uses two reduction passes (max, sum) in shared mem.
//
// Required shared mem: cur_len * sizeof(float) + d * sizeof(float).
#define MAX_SEQ_FUSED 4096
extern "C" __global__ void __launch_bounds__(1024, 1)
fused_gqa_decode_f16(
    __half* __restrict__ out,           // [b, nqh, 1, d]
    const __half* __restrict__ q,       // [b, nqh, 1, d]
    const __half* __restrict__ k_cache, // [b, nkvh, max_seq, d]
    const __half* __restrict__ v_cache, // [b, nkvh, max_seq, d]
    int b, int nqh, int nkvh, int max_seq, int d, int cur_len, float scale
) {
    int qh_global = blockIdx.x;
    int ib = qh_global / nqh;
    int qh = qh_global % nqh;
    int kh = qh / (nqh / nkvh);
    int tid = threadIdx.x;
    int bs = blockDim.x;

    extern __shared__ float smem[];   // scores[cur_len], then partial_out[d * t_chunks]
    float* partial = smem + cur_len;

    const __half* q_row  = q       + (ib * nqh  + qh) * d;
    const __half* k_base = k_cache + (ib * nkvh + kh) * max_seq * d;
    const __half* v_base = v_cache + (ib * nkvh + kh) * max_seq * d;
    __half* out_row = out + (ib * nqh + qh) * d;

    // --- Stage 1: scores[t] = (Q · K[t]) * scale ---
    for (int t = tid; t < cur_len; t += bs) {
        float dot = 0.0f;
        for (int j = 0; j < d; j += 2) {
            __half2 q2 = *(const __half2*)(q_row + j);
            __half2 k2 = *(const __half2*)(k_base + t * d + j);
            dot += __half2float(q2.x) * __half2float(k2.x)
                 + __half2float(q2.y) * __half2float(k2.y);
        }
        smem[t] = dot * scale;
    }
    __syncthreads();

    // --- Stage 2: row max (reduction over cur_len) ---
    float local_max = -INFINITY;
    for (int t = tid; t < cur_len; t += bs) {
        if (smem[t] > local_max) local_max = smem[t];
    }
    __shared__ float reduce_buf[1024];
    reduce_buf[tid] = local_max;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + s]);
        __syncthreads();
    }
    float row_max = reduce_buf[0];

    // --- Stage 3: smem[t] = exp(smem[t] - row_max); sum-reduce ---
    float local_sum = 0.0f;
    for (int t = tid; t < cur_len; t += bs) {
        float v = expf(smem[t] - row_max);
        smem[t] = v;
        local_sum += v;
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / reduce_buf[0];

    // --- Stage 4: out[j] = sum_t attn[t] * V[t, j]
    // Reuse the original t-split layout: tid = t_idx * d + j_idx (2 t-chunks for bs=256, d=128).
    // Each thread sums a t-stride for one j, then we reduce in shared mem.
    int t_chunks = bs / d;
    if (t_chunks < 1) t_chunks = 1;
    int j_idx = tid % d;
    int t_idx = tid / d;
    if (j_idx < d && t_idx < t_chunks) {
        float acc = 0.0f;
        for (int t = t_idx; t < cur_len; t += t_chunks) {
            acc += smem[t] * __half2float(v_base[t * d + j_idx]);
        }
        partial[t_idx * d + j_idx] = acc;
    }
    __syncthreads();
    if (tid < d) {
        float acc = 0.0f;
        for (int ti = 0; ti < t_chunks; ti++) {
            acc += partial[ti * d + tid];
        }
        out_row[tid] = __float2half(acc * inv_sum);
    }
}

// ─── Split-K fused GQA decode (for long context) ───────────────────
// 2-kernel flash-attention style.  Used when cur_len is large enough that single-block
// fused_gqa_decode_f16 becomes the bottleneck.
//
// Phase 1: each block handles one (b, nqh, t_chunk) — computes partial scores for its
// chunk, then partial numerator (sum of exp(scores) * V) and partial max + sum_exp.
// Writes to per-chunk buffers in global memory.
//
// Phase 2: each block handles one (b, nqh) — reads N chunks of (max, sum, partial_out),
// merges with online-softmax correction, writes final out.

extern "C" __global__ void
fused_gqa_decode_split_p1_f16(
    float* __restrict__ part_out,       // [b, nqh, NCHUNKS, d]   (float for accumulation)
    float* __restrict__ part_max,       // [b, nqh, NCHUNKS]
    float* __restrict__ part_sum,       // [b, nqh, NCHUNKS]
    const __half* __restrict__ q,       // [b, nqh, d]
    const __half* __restrict__ k_cache, // [b, nkvh, max_seq, d]
    const __half* __restrict__ v_cache, // [b, nkvh, max_seq, d]
    int b, int nqh, int nkvh, int max_seq, int d,
    int cur_len, int chunk_size, int n_chunks, float scale
) {
    int bx = blockIdx.x;
    int by = blockIdx.y;        // chunk index in [0, n_chunks)
    int ib = bx / nqh;
    int qh = bx % nqh;
    int kh = qh / (nqh / nkvh);
    int tid = threadIdx.x;
    int bs = blockDim.x;

    int t_start = by * chunk_size;
    int t_end = min(t_start + chunk_size, cur_len);
    int meta_idx = (ib * nqh + qh) * n_chunks + by;
    if (t_start >= cur_len) {
        if (tid == 0) {
            part_max[meta_idx] = -INFINITY;
            part_sum[meta_idx] = 0.0f;
        }
        if (tid < d) part_out[meta_idx * d + tid] = 0.0f;
        return;
    }
    int chunk_len = t_end - t_start;

    extern __shared__ float smem[];   // scores[chunk_size]
    __shared__ __half q_smem[128];    // d=128, Q cached in shared mem

    const __half* q_row  = q       + (ib * nqh  + qh) * d;
    const __half* k_base = k_cache + (ib * nkvh + kh) * max_seq * d + t_start * d;
    const __half* v_base = v_cache + (ib * nkvh + kh) * max_seq * d + t_start * d;

    // Cooperatively load Q into shared memory
    for (int j = tid; j < d; j += bs) q_smem[j] = q_row[j];
    __syncthreads();

    // Stage 1: scores = Q · K^T * scale
    for (int t = tid; t < chunk_len; t += bs) {
        float dot = 0.0f;
        for (int j = 0; j < d; j += 2) {
            __half2 q2 = *(const __half2*)(q_smem + j);
            __half2 k2 = *(const __half2*)(k_base + t * d + j);
            dot += __half2float(q2.x) * __half2float(k2.x)
                 + __half2float(q2.y) * __half2float(k2.y);
        }
        smem[t] = dot * scale;
    }
    __syncthreads();

    // Stage 2: chunk max (parallel reduce)
    float local_max = -INFINITY;
    for (int t = tid; t < chunk_len; t += bs) {
        if (smem[t] > local_max) local_max = smem[t];
    }
    __shared__ float reduce_buf[256];
    reduce_buf[tid] = local_max;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + s]);
        __syncthreads();
    }
    float chunk_max = reduce_buf[0];

    // Stage 3: exp(score - chunk_max) → smem, sum reduce
    float local_sum = 0.0f;
    for (int t = tid; t < chunk_len; t += bs) {
        float v = expf(smem[t] - chunk_max);
        smem[t] = v;
        local_sum += v;
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }
    float chunk_sum = reduce_buf[0];

    // Stage 4: partial_out[j] = Σ_t attn[t] * V[t, j]  (attn not yet globally normalized).
    // Each thread tid accumulates a strided sum over t for output dim j=tid%d,
    // then we reduce the t_stride partials within each j-group via shared mem.
    // We use bs threads (=256), d=128, so t_stride = bs/d = 2 threads per output dim.
    int t_stride = bs / d;
    if (t_stride < 1) t_stride = 1;
    int j = tid % d;            // output dim this thread contributes to
    int t_lane = tid / d;       // which stride-lane (0..t_stride-1)
    float my_acc = 0.0f;
    if (t_lane < t_stride) {
        for (int t = t_lane; t < chunk_len; t += t_stride) {
            my_acc += smem[t] * __half2float(v_base[t * d + j]);
        }
    }
    // Reduce t_stride lanes per j into part_out. Use reduce_buf[d * t_stride].
    // Layout: reduce_buf[t_lane * d + j].
    __shared__ float partial_out[256]; // covers up to d=128, t_stride=2
    if (t_lane < t_stride) {
        partial_out[t_lane * d + j] = my_acc;
    }
    __syncthreads();
    if (tid < d) {
        float acc = 0.0f;
        for (int ti = 0; ti < t_stride; ti++) {
            acc += partial_out[ti * d + tid];
        }
        part_out[meta_idx * d + tid] = acc;
    }

    // Write chunk metadata (after stage 4 so smem is no longer needed by this block).
    if (tid == 0) {
        part_max[meta_idx] = chunk_max;
        part_sum[meta_idx] = chunk_sum;
    }
}

// Phase 2: merge chunks via online softmax correction.
// Grid = (b * nqh,), block_dim = d.
extern "C" __global__ void __launch_bounds__(128, 8)
fused_gqa_decode_split_p2_f16(
    __half* __restrict__ out,           // [b, nqh, d]
    const float* __restrict__ part_out, // [b, nqh, n_chunks, d]
    const float* __restrict__ part_max, // [b, nqh, n_chunks]
    const float* __restrict__ part_sum, // [b, nqh, n_chunks]
    int b, int nqh, int n_chunks, int d
) {
    int qh_global = blockIdx.x;
    int ib = qh_global / nqh;
    int qh = qh_global % nqh;
    int tid = threadIdx.x;

    // Find global max across chunks (one thread reads all maxes since n_chunks is small).
    const float* maxes = part_max + (ib * nqh + qh) * n_chunks;
    const float* sums  = part_sum + (ib * nqh + qh) * n_chunks;
    float g_max = -INFINITY;
    for (int c = 0; c < n_chunks; c++) {
        if (maxes[c] > g_max) g_max = maxes[c];
    }
    // Renormalized total sum.
    float g_sum = 0.0f;
    for (int c = 0; c < n_chunks; c++) {
        if (maxes[c] > -INFINITY) {
            g_sum += sums[c] * expf(maxes[c] - g_max);
        }
    }
    float inv_g_sum = 1.0f / g_sum;

    // Accumulate per-chunk partial outputs scaled by exp(chunk_max - g_max) * inv_g_sum.
    if (tid < d) {
        float acc = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            if (maxes[c] > -INFINITY) {
                float w = expf(maxes[c] - g_max) * inv_g_sum;
                acc += w * part_out[((ib * nqh + qh) * n_chunks + c) * d + tid];
            }
        }
        out[(ib * nqh + qh) * d + tid] = __float2half(acc);
    }
}

// ─── Slice along dim 2 of [b, h, s, d] ─────────────────────────────
// out [b, h, len, d] = src[..., start:start+len, ...]
extern "C" __global__ void __launch_bounds__(1024, 2)
slice_dim2_f16(
    __half* __restrict__ out,
    const __half* __restrict__ src,
    int b, int h, int s, int d, int start, int len
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * h * len * d;
    if (tot >= total) return;
    int id = tot % d;
    int isl = (tot / d) % len;
    int ih = (tot / (d * len)) % h;
    int ib = tot / (d * len * h);
    int src_idx = ((ib * h + ih) * s + (start + isl)) * d + id;
    out[tot] = src[src_idx];
}

// ─── Concat along dim 2: write a chunk [b, h, len, d] into dst [b, h, s, d] at offset ──
extern "C" __global__ void __launch_bounds__(1024, 2)
concat_dim2_write_f16(
    __half* __restrict__ dst,
    const __half* __restrict__ src,
    int b, int h, int s, int d, int dst_offset, int len
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * h * len * d;
    if (tot >= total) return;
    int id = tot % d;
    int isl = (tot / d) % len;
    int ih = (tot / (d * len)) % h;
    int ib = tot / (d * len * h);
    int dst_idx = ((ib * h + ih) * s + (dst_offset + isl)) * d + id;
    dst[dst_idx] = src[tot];
}

// ─── im2col for conv2d with stride=2, pad=1, kernel=3×3 ────────────
// input  [b, c_in, h, w]
// out    [b * h_out * w_out, c_in * 3 * 3]   row-major
//   where h_out = (h + 2 - 3) / 2 + 1, w_out similarly
// One thread per output element across the full unfolded matrix.
extern "C" __global__ void __launch_bounds__(1024, 2)
im2col_3x3_s2p1_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int b, int c_in, int h, int w, int h_out, int w_out
) {
    int total = b * h_out * w_out * c_in * 9;
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    if (tot >= total) return;
    // tot layout: [b * h_out * w_out] [c_in * 9]
    int kk = tot % 9;
    int ci = (tot / 9) % c_in;
    int spatial = tot / (9 * c_in);
    int iw = spatial % w_out;
    int ih = (spatial / w_out) % h_out;
    int ib = spatial / (w_out * h_out);

    int ky = kk / 3;
    int kx = kk % 3;
    int in_y = ih * 2 + ky - 1;
    int in_x = iw * 2 + kx - 1;
    __half v;
    if (in_y < 0 || in_y >= h || in_x < 0 || in_x >= w) {
        v = __float2half(0.0f);
    } else {
        int in_idx = ((ib * c_in + ci) * h + in_y) * w + in_x;
        v = in[in_idx];
    }
    out[tot] = v;
}

// ─── Fused GELU + bias add + reshape from GEMM output to [b, c_out, h_out, w_out] ──
// gemm_out [b * h_out * w_out, c_out] (row-major, no bias) → conv_out [b, c_out, h_out, w_out]
// with bias[c_out] added and GELU applied.
extern "C" __global__ void __launch_bounds__(1024, 2)
conv_postprocess_f16(
    __half* __restrict__ out,
    const __half* __restrict__ gemm_out,
    const __half* __restrict__ bias,
    int b, int c_out, int h_out, int w_out
) {
    int total = b * c_out * h_out * w_out;
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    if (tot >= total) return;
    int iw = tot % w_out;
    int ih = (tot / w_out) % h_out;
    int co = (tot / (w_out * h_out)) % c_out;
    int ib = tot / (w_out * h_out * c_out);
    int gemm_idx = ((ib * h_out + ih) * w_out + iw) * c_out + co;
    float v = __half2float(gemm_out[gemm_idx]) + __half2float(bias[co]);
    // GELU
    float g = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
    out[tot] = __float2half(g);
}

// ─── Permute [b, c, f, t] → [b, t, c, f] ──────────────────────────────
// Each thread copies one element. Grid covers b*t*c*f total elements.
extern "C" __global__ void __launch_bounds__(512, 4)
permute_bcft_to_btcf_f16(
    __half* __restrict__ dst,
    const __half* __restrict__ src,
    int b, int c, int f, int t
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * c * f * t;
    if (tot >= total) return;

    // Decompose linear index into (ib, ic, if_, it) in source layout [b,c,f,t]
    int it = tot % t;
    int rem = tot / t;
    int if_ = rem % f;
    rem /= f;
    int ic = rem % c;
    int ib = rem / c;

    // Destination layout [b, t, c, f]
    int dst_idx = ((ib * t + it) * c + ic) * f + if_;
    dst[dst_idx] = src[tot];
}

// ─── Broadcast PE add: out[b, t, d] = x[b, t, d] + pe[t, d] ──────────
// Grid: (b*t, 1, 1), Block: min(d, 1024), shared mem: block_size * sizeof(float)
extern "C" __global__ void __launch_bounds__(512, 4)
add_pe_f16(
    __half* __restrict__ out,
    const __half* __restrict__ x,
    const __half* __restrict__ pe,
    int d,       // d_model
    int t,       // t2 (number of time steps actually used)
    int bt       // b * t2 total rows
) {
    int row = blockIdx.x;
    if (row >= bt) return;
    int tid = threadIdx.x;
    int bs = blockDim.x;
    int t_step = row % t;  // which time step (for PE lookup)

    for (int j = tid; j < d; j += bs) {
        float v = __half2float(x[row * d + j]) + __half2float(pe[t_step * d + j]);
        out[row * d + j] = __float2half(v);
    }
}

// ─── Conv1d (kernel=3, padding=1, configurable stride) for the Whisper stem ──
// input x: [b, in_ch, in_t], weight w: [out_ch, in_ch, 3] (row-major), bias: [out_ch].
// out: [b, out_ch, out_t] where out_t = (in_t + 2 - 3)/stride + 1, with in-place GELU.
// One thread per (b, out_ch, out_t) output element.
extern "C" __global__ void __launch_bounds__(256, 4)
conv1d_k3_gelu_f16(
    __half* __restrict__ out,
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    const __half* __restrict__ bias,
    int b, int in_ch, int in_t, int out_ch, int out_t, int stride
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * out_ch * out_t;
    if (tot >= total) return;
    int ot = tot % out_t;
    int oc = (tot / out_t) % out_ch;
    int ib = tot / (out_t * out_ch);
    float acc = __half2float(bias[oc]);
    int center = ot * stride;
    for (int ic = 0; ic < in_ch; ic++) {
        const __half* w_row = w + oc * in_ch * 3 + ic * 3;
        const __half* x_row = x + (ib * in_ch + ic) * in_t;
        for (int k = 0; k < 3; k++) {
            int it = center + k - 1;
            if (it >= 0 && it < in_t) {
                acc += __half2float(x_row[it]) * __half2float(w_row[k]);
            }
        }
    }
    // GELU
    float g = 0.5f * acc * (1.0f + erff(acc * 0.70710678118654752440f));
    out[tot] = __float2half(g);
}

// ─── Permute [b, c, t] → [b, t, c] (Whisper stem → transformer layout) ──
extern "C" __global__ void __launch_bounds__(256, 4)
permute_bct_to_btc_f16(
    __half* __restrict__ dst,
    const __half* __restrict__ src,
    int b, int c, int t
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * c * t;
    if (tot >= total) return;
    // src [b, c, t]: element (ib, ic, it) at ((ib*c + ic)*t + it)
    int it = tot % t;
    int ic = (tot / t) % c;
    int ib = tot / (t * c);
    // dst [b, t, c]: element (ib, it, ic) at ((ib*t + it)*c + ic)
    dst[(ib * t + it) * c + ic] = src[tot];
}
