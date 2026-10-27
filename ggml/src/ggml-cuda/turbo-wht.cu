#include "turbo-quant.cuh"
#include "turbo-wht.cuh"

// One CTA transforms one group; tails pass through unchanged.
template <int direction, int group_size>
static __global__ void k_turbo_wht_f32(const float * __restrict__ src,
                                        float * __restrict__ dst,
                                        const float * __restrict__ scale_inv,
                                        int64_t n_groups,
                                        int64_t head_dim,
                                        int64_t groups_per_head) {
    static_assert(group_size == 128 || group_size == 64 || group_size == 32, "group_size must be 128, 64, or 32");

    const int64_t g = blockIdx.x;
    if (g >= n_groups) return;

    const int t = threadIdx.x;  // 0 .. group_size-1

    const int64_t head_idx     = g / groups_per_head;
    const int64_t grp_in_head  = g % groups_per_head;
    const int64_t base         = head_idx * head_dim + grp_in_head * group_size;

    __shared__ float x[group_size];

    x[t] = src[base + t];
    __syncthreads();

    if (direction == 0 && scale_inv != nullptr) {
        x[t] *= scale_inv[t % group_size];
        __syncthreads();
    }

    if (group_size == 128) {
        x[t] *= (direction == 0) ? TURBO_WHT_SIGNS1[t] : TURBO_WHT_SIGNS2[t];
    } else if (group_size == 64) {
        x[t] *= (direction == 0) ? TURBO_WHT_SIGNS1_64[t] : TURBO_WHT_SIGNS2_64[t];
    } else {
        x[t] *= (direction == 0) ? TURBO_WHT_SIGNS1[t] : TURBO_WHT_SIGNS2[t];
    }
    __syncthreads();

    const int lane = t & 31;
    float val = x[t];

#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, h, WARP_SIZE);
        val = (lane & h) ? (o - val) : (val + o);
    }

    x[t] = val;
    __syncthreads();

    if (group_size >= 64) {
        if (t % 64 < 32) {
            float a = x[t], b = x[t + 32];
            x[t] = a + b;
            x[t + 32] = a - b;
        }
        __syncthreads();
    }

    if (group_size == 128) {
        if (t % 128 < 64) {
            float a = x[t], b = x[t + 64];
            x[t] = a + b;
            x[t + 64] = a - b;
        }
        __syncthreads();
    }

    constexpr float inv_sqrt = (group_size == 128) ? 0.08838834764831845f :
                               (group_size == 64)  ? 0.125f :
                                                     0.17677669529663688f; // 1/sqrt(32)
    float result;
    if (group_size == 128) {
        result = x[t] * inv_sqrt *
            ((direction == 0) ? TURBO_WHT_SIGNS2[t] : TURBO_WHT_SIGNS1[t]);
    } else if (group_size == 64) {
        result = x[t] * inv_sqrt *
            ((direction == 0) ? TURBO_WHT_SIGNS2_64[t] : TURBO_WHT_SIGNS1_64[t]);
    } else {
        result = x[t] * inv_sqrt *
            ((direction == 0) ? TURBO_WHT_SIGNS2[t] : TURBO_WHT_SIGNS1[t]);
    }

    if (direction == 1 && scale_inv != nullptr) {
        result *= scale_inv[t % group_size];
    }

    dst[base + t] = result;
}

static __device__ __forceinline__ float turbo_wht_sign_flip(float x, unsigned bit) {
    return __uint_as_float(__float_as_uint(x) ^ (bit << 31));
}

// Vectorized 128-element transform: one group per warp.
template <int direction, int warps_per_block>
static __global__ void k_turbo_wht_f32_fast(const float * __restrict__ src,
                                            float * __restrict__ dst,
                                            const float * __restrict__ scale_inv,
                                            int64_t n_groups,
                                            int64_t head_dim,
                                            int64_t groups_per_head) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const int64_t g = (int64_t) blockIdx.x * warps_per_block + warp;
    if (g >= n_groups) return;

    const int64_t head_idx    = g / groups_per_head;
    const int64_t grp_in_head = g % groups_per_head;
    const int64_t base        = head_idx * head_dim + grp_in_head * 128;

    float4 v = *((const float4 *) (src + base) + lane);

    if (direction == 0 && scale_inv != nullptr) {
        const float4 s = *((const float4 *) scale_inv + lane);
        v.x *= s.x; v.y *= s.y; v.z *= s.z; v.w *= s.w;
    }

    {
        const unsigned * s = (direction == 0) ? TURBO_WHT_SIGNBITS1 : TURBO_WHT_SIGNBITS2;
        const unsigned nib = s[lane >> 3] >> (4 * (lane & 7));
        v.x = turbo_wht_sign_flip(v.x, (nib     ) & 1u);
        v.y = turbo_wht_sign_flip(v.y, (nib >> 1) & 1u);
        v.z = turbo_wht_sign_flip(v.z, (nib >> 2) & 1u);
        v.w = turbo_wht_sign_flip(v.w, (nib >> 3) & 1u);
    }

    { float a = v.x, b = v.y; v.x = a + b; v.y = a - b;
      float c = v.z, d = v.w; v.z = c + d; v.w = c - d; }

    { float a = v.x, b = v.z; v.x = a + b; v.z = a - b;
      float c = v.y, d = v.w; v.y = c + d; v.w = c - d; }

#define WHT_SHFL_STAGE(m)                                       \
    {                                                           \
        const float px = __shfl_xor_sync(0xffffffff, v.x, (m), 32); \
        const float py = __shfl_xor_sync(0xffffffff, v.y, (m), 32); \
        const float pz = __shfl_xor_sync(0xffffffff, v.z, (m), 32); \
        const float pw = __shfl_xor_sync(0xffffffff, v.w, (m), 32); \
        const bool hi = (lane & (m)) != 0;                      \
        v.x = hi ? px - v.x : v.x + px;                         \
        v.y = hi ? py - v.y : v.y + py;                         \
        v.z = hi ? pz - v.z : v.z + pz;                         \
        v.w = hi ? pw - v.w : v.w + pw;                         \
    }

    WHT_SHFL_STAGE(1)    // h = 4
    WHT_SHFL_STAGE(2)    // h = 8
    WHT_SHFL_STAGE(4)    // h = 16
    WHT_SHFL_STAGE(8)    // h = 32
    WHT_SHFL_STAGE(16)   // h = 64
#undef WHT_SHFL_STAGE

    {
        constexpr float inv_sqrt = 0.08838834764831845f;  // 1/sqrt(128)
        const unsigned * s = (direction == 0) ? TURBO_WHT_SIGNBITS2 : TURBO_WHT_SIGNBITS1;
        const unsigned nib = s[lane >> 3] >> (4 * (lane & 7));
        v.x = turbo_wht_sign_flip(v.x * inv_sqrt, (nib     ) & 1u);
        v.y = turbo_wht_sign_flip(v.y * inv_sqrt, (nib >> 1) & 1u);
        v.z = turbo_wht_sign_flip(v.z * inv_sqrt, (nib >> 2) & 1u);
        v.w = turbo_wht_sign_flip(v.w * inv_sqrt, (nib >> 3) & 1u);
    }

    if (direction == 1 && scale_inv != nullptr) {
        const float4 s = *((const float4 *) scale_inv + lane);
        v.x *= s.x; v.y *= s.y; v.z *= s.z; v.w *= s.w;
    }

    *((float4 *) (dst + base) + lane) = v;
}

static __global__ void k_turbo_wht_copy_tail(const float * __restrict__ src,
                                              float * __restrict__ dst,
                                              int64_t n_heads,
                                              int64_t head_dim,
                                              int64_t tail_offset,
                                              int tail_size) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_heads * tail_size) return;

    const int64_t head_idx  = i / tail_size;
    const int64_t tail_elem = i % tail_size;
    const int64_t offset    = head_idx * head_dim + tail_offset + tail_elem;
    dst[offset] = src[offset];
}

void ggml_cuda_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * scale_tensor = dst->src[1];  // InnerQ scale_inv (may be NULL)

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int direction = ggml_get_op_params_i32(dst, 0);
    const int group_size = ggml_get_op_params_i32(dst, 1);

    const int64_t head_dim        = src->ne[0];
    const int64_t n_heads         = ggml_nelements(src) / head_dim;

    GGML_ASSERT(direction == 0 || direction == 1);
    GGML_ASSERT(group_size == 32 || group_size == 64 || group_size == 128);
    if (scale_tensor) {
        GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(scale_tensor));
        GGML_ASSERT(ggml_nelements(scale_tensor) >= group_size);
    }
    const int64_t groups_per_head = head_dim / group_size;
    const int     tail_size       = (int)(head_dim % group_size);
    const int64_t n_groups        = groups_per_head * n_heads;

    const float * src_ptr = (const float *) src->data;
    float       * dst_ptr = (float       *) dst->data;
    const float * scale_inv_ptr = scale_tensor ? (const float *) scale_tensor->data : nullptr;

    cudaStream_t stream = ctx.stream();

    if (n_groups > 0) {
        const bool fast_ok =
            group_size == 128 &&
            (head_dim % 4) == 0 &&                                          // float4 indexing
            (((uintptr_t) src_ptr | (uintptr_t) dst_ptr) % 16) == 0 &&      // float4 alignment
            (scale_inv_ptr == nullptr || ((uintptr_t) scale_inv_ptr % 16) == 0);

        dim3 blocks(n_groups);

        if (fast_ok) {
            constexpr int warps = 4;
            const int64_t n_blocks = (n_groups + warps - 1) / warps;
            if (direction == 0) {
                k_turbo_wht_f32_fast<0, warps><<<(int) n_blocks, warps*32, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            } else {
                k_turbo_wht_f32_fast<1, warps><<<(int) n_blocks, warps*32, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            }
        } else if (group_size == 128) {
            dim3 threads(128);
            if (direction == 0) {
                k_turbo_wht_f32<0, 128><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            } else {
                k_turbo_wht_f32<1, 128><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            }
        } else if (group_size == 64) {
            dim3 threads(64);
            if (direction == 0) {
                k_turbo_wht_f32<0, 64><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            } else {
                k_turbo_wht_f32<1, 64><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            }
        } else {
            dim3 threads(32);
            if (direction == 0) {
                k_turbo_wht_f32<0, 32><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            } else {
                k_turbo_wht_f32<1, 32><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            }
        }
    }

    if (tail_size > 0) {
        const int64_t total_tail = n_heads * tail_size;
        const int block_sz = 256;
        const int n_blocks = (int)((total_tail + block_sz - 1) / block_sz);
        k_turbo_wht_copy_tail<<<n_blocks, block_sz, 0, stream>>>(
            src_ptr, dst_ptr, n_heads, head_dim, groups_per_head * group_size, tail_size);
    }
}
