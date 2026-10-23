#include "common.cuh"
#include "convrot.cuh"

static __device__ __forceinline__ void convrot_transform_group(
        const float * src,
        float * rd,
        float * wr,
        const bool active,
        const int lane) {
    // normalized Kronecker-Hadamard base matrix, no all-ones column
    constexpr float h4_cr[4][4] = {
        { 1.0f,  1.0f,  1.0f, -1.0f},
        { 1.0f,  1.0f, -1.0f,  1.0f},
        { 1.0f, -1.0f,  1.0f,  1.0f},
        {-1.0f,  1.0f,  1.0f,  1.0f},
    };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int j = 0; j < QK8_CR/32; ++j) {
        rd[lane + 32*j] = active ? src[lane + 32*j] : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int len = 4, shift = 0; len <= QK8_CR; len *= 4, shift += 2) {
        const int half = len >> 2;
#pragma unroll
        for (int k = 0; k < QK8_CR/32; ++k) {
            const int i   = lane + 32*k;
            const int blk = i & ~(len - 1);
            const int r   = (i >> shift) & 3;
            const int t   = i & (half - 1);

            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc += h4_cr[r][j] * rd[blk + j*half + t];
            }
            wr[i] = acc;
        }
        __syncthreads();
        float * tmp = rd; rd = wr; wr = tmp;
    }
}

// One warp transforms one QK8_CR group with the radix-4 butterfly from ggml-quants.c.
template <int warps_per_block>
__global__ void convrot_rotate_cuda(
        const float * src_ptr,
        float * dst_ptr,
        const int64_t n_groups) {
    const float * GGML_CUDA_RESTRICT src = src_ptr;
    float * GGML_CUDA_RESTRICT dst = dst_ptr;

    const int lane   = threadIdx.x;
    const int warp   = threadIdx.y;
    const int64_t g  = (int64_t) blockIdx.x * warps_per_block + warp;

    const bool active = g < n_groups;
    if (active) {
        src += g * QK8_CR;
        dst += g * QK8_CR;
    }

    __shared__ float buf[warps_per_block][2][QK8_CR];

    float * rd = buf[warp][0];
    float * wr = buf[warp][1];

    convrot_transform_group(src, rd, wr, active, lane);

    // the product of the four stages has unit norm, scale once at the end
    constexpr float scale = 1.0f / 16.0f; // 1 / sqrt(QK8_CR)
    if (active) {
#pragma unroll
        for (int j = 0; j < QK8_CR/32; ++j) {
            dst[lane + 32*j] = rd[lane + 32*j] * scale;
        }
    }
}

void ggml_cuda_convrot_rotate(ggml_backend_cuda_context & ctx, const ggml_tensor * src, float * dst) {
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_rows(src));
    GGML_ASSERT(src->ne[0] % QK8_CR == 0);

    const int64_t n_groups = ggml_nrows(src) * (src->ne[0] / QK8_CR);
    if (n_groups == 0) {
        return;
    }

    constexpr int warps_per_block = 4;
    const int64_t num_blocks = (n_groups + warps_per_block - 1) / warps_per_block;

    cudaStream_t                         stream = ctx.stream();
    dim3                                 grid_dims(num_blocks, 1, 1);
    dim3                                 block_dims(32, warps_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    ggml_cuda_kernel_launch(convrot_rotate_cuda<warps_per_block>, launch_params,
        (const float *) src->data, dst, n_groups);
}

