#include "set-rows.cuh"
#include "cpy-utils.cuh"
#include "turbo-quant.cuh"

// One on-disk KV block is one 128-element rotation group.
static_assert(QK_TURBO2 == 128 && QK_TURBO3 == 128 && QK_TURBO4 == 128,
              "TurboQuant SET_ROWS requires 128-element blocks");

template <typename block_t> struct turbo_kv_codec;

template <> struct turbo_kv_codec<block_turbo2_0> {
    static __device__ uint8_t index(float v) { return turbo_nearest_centroid_2bit(v); }
    static __device__ float centroid(uint8_t i) { return TURBO_CENTROIDS_2BIT[i]; }
    static __device__ void pack(block_turbo2_0 * dst, const uint8_t * idx, int j) {
        if (j % 4 == 0) {
            dst->qs[j / 4] = idx[j] | (idx[j + 1] << 2) | (idx[j + 2] << 4) | (idx[j + 3] << 6);
        }
    }
};

template <> struct turbo_kv_codec<block_turbo3_0> {
    static __device__ uint8_t index(float v) { return turbo_nearest_centroid_3bit(v); }
    static __device__ float centroid(uint8_t i) { return TURBO_CENTROIDS_3BIT[i]; }
    static __device__ void pack(block_turbo3_0 * dst, const uint8_t * idx, int j) {
        if (j % 4 == 0) {
            dst->qs[j / 4] = (idx[j] & 3) | ((idx[j + 1] & 3) << 2) |
                            ((idx[j + 2] & 3) << 4) | ((idx[j + 3] & 3) << 6);
        }
        if (j % 8 == 0) {
            uint8_t signs = 0;
            for (int k = 0; k < 8; ++k) {
                signs |= ((idx[j + k] >> 2) & 1) << k;
            }
            dst->signs[j / 8] = signs;
        }
    }
};

#if TURBO4_USE_4BIT
template <> struct turbo_kv_codec<block_turbo4_0> {
    static __device__ uint8_t index(float v) { return turbo_nearest_centroid_4bit(v); }
    static __device__ float centroid(uint8_t i) { return TURBO_CENTROIDS_4BIT[i]; }
    static __device__ void pack(block_turbo4_0 * dst, const uint8_t * idx, int j) {
        if (j % 2 == 0) {
            dst->qs[j / 2] = idx[j] | (idx[j + 1] << 4);
        }
    }
};
#endif

template <typename src_t, typename idx_t, typename block_t>
static __global__ void k_set_rows_turbo(
        const src_t * src, const idx_t * indices, char * dst,
        int64_t blocks_per_row, int64_t ne01, int64_t ne02, int64_t ne11, int64_t ne12,
        size_t nb01, size_t nb02, size_t nb03, size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3) {
    ggml_cuda_pdl_sync();
    const int j = threadIdx.x;
    int64_t row = blockIdx.x / blocks_per_row;
    const int64_t b = blockIdx.x % blocks_per_row;
    const int64_t i01 = row % ne01;
    row /= ne01;
    const int64_t i02 = row % ne02;
    const int64_t i03 = row / ne02;
    const int64_t dst_row = *(const idx_t *) ((const char *) indices +
            i01 * nb10 + (i02 % ne11) * nb11 + (i03 % ne12) * nb12);
    const src_t * src_row = (const src_t *) ((const char *) src + i01 * nb01 + i02 * nb02 + i03 * nb03);
    block_t * out = (block_t *) (dst + dst_row * nb1 + i02 * nb2 + i03 * nb3) + b;

    __shared__ float x[128];
    __shared__ uint8_t idx[128];
    __shared__ float norm;
    __shared__ float inv_norm;
    x[j] = ggml_cuda_cast<float>(src_row[b * 128 + j]);
    __syncthreads();
    if (j == 0) {
        float sum = 0.0f;
        for (int k = 0; k < 128; ++k) { sum += x[k] * x[k]; }
        norm = sqrtf(sum);
        inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    }
    __syncthreads();
    x[j] = (x[j] * inv_norm) * TURBO_WHT_SIGNS1[j];
    __syncthreads();

    // All threads read a stage before any thread writes the next stage.
    for (int h = 1; h < 128; h <<= 1) {
        const float a = x[j & ~h];
        const float b = x[j | h];
        __syncthreads();
        x[j] = (j & h) ? a - b : a + b;
        __syncthreads();
    }
    idx[j] = turbo_kv_codec<block_t>::index(x[j] * (0.08838834764831845f * TURBO_WHT_SIGNS2[j]));
    __syncthreads();
    turbo_kv_codec<block_t>::pack(out, idx, j);
    if (j == 0) {
        float recon_sq = 0.0f;
        for (int k = 0; k < 128; ++k) {
            const float c = turbo_kv_codec<block_t>::centroid(idx[k]);
            recon_sq += c * c;
        }
        const float recon_norm = sqrtf(recon_sq);
        out->norm = __float2half(recon_norm > 1e-10f ? norm / recon_norm : norm);
    }
}

template <typename src_t, typename idx_t, typename block_t>
static void set_rows_cuda_turbo(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src, const ggml_tensor * indices, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_supports_turbo_set_rows(dst));
    const int64_t blocks_per_row = src->ne[0] / 128;
    const int64_t n_blocks = blocks_per_row * src->ne[1] * src->ne[2] * src->ne[3];
    if (n_blocks == 0) { return; }
    k_set_rows_turbo<src_t, idx_t, block_t><<<n_blocks, 128, 0, ctx.stream()>>>(
        (const src_t *) src->data, (const idx_t *) indices->data, (char *) dst->data,
        blocks_per_row, src->ne[1], src->ne[2], indices->ne[1], indices->ne[2],
        src->nb[1], src->nb[2], src->nb[3], indices->nb[0], indices->nb[1], indices->nb[2],
        dst->nb[1], dst->nb[2], dst->nb[3]);
}

template <typename src_t, typename idx_t>
static void set_rows_cuda_turbo(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->type) {
        case GGML_TYPE_TURBO2_0:
            set_rows_cuda_turbo<src_t, idx_t, block_turbo2_0>(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_TYPE_TURBO3_0:
            set_rows_cuda_turbo<src_t, idx_t, block_turbo3_0>(ctx, dst->src[0], dst->src[1], dst);
            break;
#if TURBO4_USE_4BIT
        case GGML_TYPE_TURBO4_0:
            set_rows_cuda_turbo<src_t, idx_t, block_turbo4_0>(ctx, dst->src[0], dst->src[1], dst);
            break;
#endif
        default: GGML_ABORT("unsupported TurboQuant SET_ROWS type");
    }
}

bool ggml_cuda_supports_turbo_set_rows(const ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * idx = dst->src[1];
    const int group_size = ggml_get_op_params_i32(dst, 0);
    return src && idx &&
        (src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16) &&
        (idx->type == GGML_TYPE_I32 || idx->type == GGML_TYPE_I64) &&
        src->nb[0] == ggml_type_size(src->type) && src->ne[0] % 128 == 0 &&
        idx->ne[1] > 0 && idx->ne[2] > 0 &&
        (group_size == 0 || group_size == 128) &&
        (dst->type == GGML_TYPE_TURBO2_0 || dst->type == GGML_TYPE_TURBO3_0
#if TURBO4_USE_4BIT
         || dst->type == GGML_TYPE_TURBO4_0
#endif
        );
}

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * src0_ptr,
                                  const idx_t * src1_ptr,
                                  dst_t * dst_ptr,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const src_t * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const idx_t * GGML_CUDA_RESTRICT src1 = src1_ptr;
    dst_t       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    ggml_cuda_pdl_lc();

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_size, block_size, 0, stream);
        ggml_cuda_kernel_launch(k_set_rows<src_t, idx_t, dst_t>, launch_params,
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
            s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
            ne11_fd, ne12_fd);
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}

template<>
void set_rows_cuda<half, int32_t>(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const half    * src0_d = (const half *)src0->data;
    const int32_t * src1_d = (const int32_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}

template<>
void set_rows_cuda<half, int64_t>(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const half    * src0_d = (const half *)src0->data;
    const int64_t * src1_d = (const int64_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (dst->type == GGML_TYPE_TURBO2_0 || dst->type == GGML_TYPE_TURBO3_0 || dst->type == GGML_TYPE_TURBO4_0) {
        GGML_ASSERT(ggml_cuda_supports_turbo_set_rows(dst));
        if (src0->type == GGML_TYPE_F32) {
            if (src1->type == GGML_TYPE_I64) {
                set_rows_cuda_turbo<float, int64_t>(ctx, dst);
            } else {
                set_rows_cuda_turbo<float, int32_t>(ctx, dst);
            }
        } else {
            if (src1->type == GGML_TYPE_I64) {
                set_rows_cuda_turbo<half, int64_t>(ctx, dst);
            } else {
                set_rows_cuda_turbo<half, int32_t>(ctx, dst);
            }
        }
        return;
    }

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16));
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src0->type == GGML_TYPE_F32) {
        if (src1->type == GGML_TYPE_I64) {
            set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
        } else {
            set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
        }
    } else if (src0->type == GGML_TYPE_F16) {
        if (src1->type == GGML_TYPE_I64) {
            set_rows_cuda<half, int64_t>(ctx, src0, src1, dst);
        } else {
            set_rows_cuda<half, int32_t>(ctx, src0, src1, dst);
        }
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(src0->type));
    }
}
