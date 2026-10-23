// TQ4_1S decode kernels. Prefill and TQ3_1S use dequantization + cuBLAS.
#include "mmvq-tq.cuh"
#include "turbo-quant.cuh"
#include "vecdotq.cuh"

#define MMVQ_TQ_NWARPS 4

bool ggml_cuda_tq_fast_path(const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, const ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_TQ4_1S || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) ||
            src0->ne[0] != src1->ne[0] || src0->ne[0] % QK_TQ4_1S != 0 ||
            src0->ne[3] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1) {
        return false;
    }
    if (!ids) {
        return src0->ne[2] == 1 && src1->ne[2] == 1 && dst->ne[2] == 1 &&
            src1->ne[1] >= 1 && src1->ne[1] <= 8 && ggml_is_contiguous(dst);
    }
    return ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t) &&
        ids->nb[1] % sizeof(int32_t) == 0 && ids->ne[2] == 1 && ids->ne[3] == 1 &&
        dst->nb[0] == sizeof(float) && dst->nb[1] % sizeof(float) == 0 && dst->nb[2] % sizeof(float) == 0 &&
        dst->ne[2] >= 1 && dst->ne[2] <= 8 &&
        src1->ne[1] > 0 && dst->ne[1] % src1->ne[1] == 0 &&
        src1->ne[2] == dst->ne[2] && ids->ne[0] == dst->ne[1] && ids->ne[1] == dst->ne[2];
}

static __global__ void tq_prerotate_q8_1(
        const float * __restrict__ src,
        block_q8_1  * __restrict__ dst,
        const int n_elements) {

    const int block_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    const int offset = block_idx * 32 + lane;
    if (offset >= n_elements) return;

    float val = src[offset];
    val *= TQ_WEIGHT_SIGNS[lane];

    #pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, h);
        val = (lane & h) ? (o - val) : (val + o);
    }
    val *= 0.17677669529663688f;

    float amax = fabsf(val);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));

    float sum = val;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, off);

    const float d = amax / 127.0f;
    const float id = (d > 0.0f) ? 127.0f / amax : 0.0f;

    dst[block_idx].qs[lane] = (int8_t)roundf(val * id);
    if (lane == 0) {
        dst[block_idx].ds = make_half2(__float2half(d), __float2half(sum));
    }
}

// ============================================================================
// TQ4_1S: dp4a path with fixed int8 centroid LUT + q8_1 activation
// ============================================================================

// Fixed int8 centroid table: centroid_i8[i] = round(TQ4_CENTROIDS_WEIGHT[i] * 127 / 2.733)
// Rescale factor to recover float centroids: 2.733 / 127
static constexpr float TQ4_CENTROID_I8_RESCALE = 2.733f / 127.0f;

// Register-based centroid lookup: maps 4 qs bytes (1 uint32) to 2 packed 4× centroid_i8 for dp4a.
// Processes a full uint32 at once, sharing nibble extraction across both byte pairs.
__device__ __forceinline__ void tq4_cents8_reg(uint32_t four_bytes, int &c0, int &c1) {
    // 4 qs bytes -> 8 nibble indices -> 8 int8 centroids, in natural element order
    // (element 2k = low nibble of byte k, element 2k+1 = high nibble of byte k).
    //
    // This uses the hardware byte-permute (v_perm_b32) through get_int_from_table_16, the same
    // path IQ4_NL/MXFP4 use on ROCm. NOTE: do NOT express this with HIP's __byte_perm() wrapper
    // -- with a runtime selector its generic fallback lowers through a dynamically indexed byte
    // union that PromoteAlloca turns into a 32KB LDS staging area plus ds_read_u8 chains, which
    // caps occupancy and costs ~3x. __builtin_amdgcn_perm maps 1:1 to the instruction.
    //
    // The previous shift-based fallback avoided that bug but cost ~100 ALU ops per call (~400
    // per 32-weight block), which made the TQ decode kernels ALU-bound rather than
    // bandwidth-bound on CDNA.
    const int2 v = get_int_from_table_16((int) four_bytes, kvalues_tq4);

    // v.x = centroids for even elements (0,2,4,6), v.y = odd elements (1,3,5,7).
    // Interleave back to natural order: c0 = [e0,e1,e2,e3], c1 = [e4,e5,e6,e7].
#if defined(GGML_USE_HIP)
    c0 = (int) __builtin_amdgcn_perm(v.y, v.x, 0x05010400);
    c1 = (int) __builtin_amdgcn_perm(v.y, v.x, 0x07030602);
#else
    // nvcc / MUSA: prmt selector nibbles pick bytes 0-3 from the first operand and 4-7 from
    // the second. Same byte order as the HIP form above.
    c0 = __byte_perm(v.x, v.y, 0x5140);
    c1 = __byte_perm(v.x, v.y, 0x7362);
#endif
}

template <int ncols_dst>
static __global__ void mul_mat_tq4_1s_dp4a_multi(
        const void       * __restrict__ vx,
        const block_q8_1 * __restrict__ vy_q8,
        float            * __restrict__ dst,
        const int ncols_x,
        const int nrows_x,
        const int stride_col_y,
        const int stride_col_dst) {

    const int row = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TQ4_1S;
    const block_tq4_1s * x_row = ((const block_tq4_1s *) vx) + (int64_t)row * blocks_per_row;

    float sumf[ncols_dst] = {};

    for (int ib = lane; ib < blocks_per_row; ib += WARP_SIZE) {
        const block_tq4_1s * blk = &x_row[ib];
        const float fd0 = __half2float(blk->d0);
        const float fd1 = __half2float(blk->d1);

        // Load weight once, reuse across all tokens
        const uint32_t * qs32 = (const uint32_t *)(blk->qs);
        const uint32_t w0 = qs32[0], w1 = qs32[1], w2 = qs32[2], w3 = qs32[3];

        int c0_0, c1_0, c0_1, c1_1, c0_2, c1_2, c0_3, c1_3;
        tq4_cents8_reg(w0, c0_0, c1_0);
        tq4_cents8_reg(w1, c0_1, c1_1);
        tq4_cents8_reg(w2, c0_2, c1_2);
        tq4_cents8_reg(w3, c0_3, c1_3);

        #pragma unroll
        for (int j = 0; j < ncols_dst; j++) {
            const block_q8_1 * a_blk = &vy_q8[j * stride_col_y + ib];
            const float d_act = __half2float((__half)a_blk->ds.x);
            const int * a_qs = (const int *)(a_blk->qs);

            const int s0 = ggml_cuda_dp4a(c0_0, a_qs[0], ggml_cuda_dp4a(c1_0, a_qs[1],
                           ggml_cuda_dp4a(c0_1, a_qs[2], ggml_cuda_dp4a(c1_1, a_qs[3], 0))));
            const int s1 = ggml_cuda_dp4a(c0_2, a_qs[4], ggml_cuda_dp4a(c1_2, a_qs[5],
                           ggml_cuda_dp4a(c0_3, a_qs[6], ggml_cuda_dp4a(c1_3, a_qs[7], 0))));

            sumf[j] += d_act * (fd0 * (float)s0 + fd1 * (float)s1);
        }
    }

    // Apply centroid int8→float rescale + warp reduction
    #pragma unroll
    for (int j = 0; j < ncols_dst; j++)
        sumf[j] *= TQ4_CENTROID_I8_RESCALE;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++)
            sumf[j] += __shfl_xor_sync(0xffffffff, sumf[j], offset);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++)
            dst[j * stride_col_dst + row] = sumf[j];
    }
}

template <int ncols_dst>
static void launch_tq4_1s_multi(
        const void * src0_d, const block_q8_1 * q8_buf,
        float * dst_d, int ncols_x, int nrows_x,
        int stride_col_y, int stride_col_dst, cudaStream_t stream) {
    const dim3 block(WARP_SIZE, MMVQ_TQ_NWARPS);
    const dim3 grid((nrows_x + MMVQ_TQ_NWARPS - 1) / MMVQ_TQ_NWARPS);
    mul_mat_tq4_1s_dp4a_multi<ncols_dst><<<grid, block, 0, stream>>>(
        src0_d, q8_buf, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst);
}

template <int LPR>   // LPR: lanes cooperating on one output row (must divide WARP_SIZE)
static __global__ void mul_mat_tq4_1s_dp4a_moe(
        const void       * __restrict__ vx,
        const block_q8_1 * __restrict__ vy_q8,      // pre-rotated activations (q8_1)
        float            * __restrict__ dst,
        const int32_t    * __restrict__ ids,
        const int     ncols_x,
        const int     nrows_x,
        const int64_t nb_expert,
        const int     ids_stride,
        const int     nchannels_y,
        const int64_t stride_channel_y,             // q8_1 blocks to next y-channel
        const int64_t stride_sample_y,              // q8_1 blocks to next sample
        const int64_t stride_channel_dst,
        const int64_t stride_sample_dst) {

    // One row per LPR lanes: each lane covers blocks_per_row/LPR blocks and the cross-lane
    // reduction is log2(LPR) rounds instead of 5. With LPR == WARP_SIZE this is the classic
    // one-row-per-warp mapping.
    constexpr int rows_per_warp = WARP_SIZE / LPR;

    const int lane_in_row = threadIdx.x % LPR;
    const int row_in_warp = threadIdx.x / LPR;

    const int row = (blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y) * rows_per_warp + row_in_warp;
    if (row >= nrows_x) return;

    const int expert_slot = blockIdx.y;
    const int sample      = blockIdx.z;
    const int expert      = ids[sample * ids_stride + expert_slot];
    const int channel_y   = expert_slot % nchannels_y;

    const int lane = lane_in_row;
    const int blocks_per_row = ncols_x / QK_TQ4_1S;
    const block_tq4_1s * x_row =
        (const block_tq4_1s *) ((const char *) vx + (int64_t) expert * nb_expert)
        + (int64_t) row * blocks_per_row;
    const block_q8_1 * a_base = vy_q8 + sample * stride_sample_y + (int64_t) channel_y * stride_channel_y;

    float sumf = 0.0f;
    for (int ib = lane; ib < blocks_per_row; ib += LPR) {
        const block_tq4_1s * blk = &x_row[ib];
        const float fd0 = __half2float(blk->d0);
        const float fd1 = __half2float(blk->d1);

        const uint32_t * qs32 = (const uint32_t *)(blk->qs);
        const uint32_t w0 = qs32[0], w1 = qs32[1], w2 = qs32[2], w3 = qs32[3];

        int c0_0, c1_0, c0_1, c1_1, c0_2, c1_2, c0_3, c1_3;
        tq4_cents8_reg(w0, c0_0, c1_0);
        tq4_cents8_reg(w1, c0_1, c1_1);
        tq4_cents8_reg(w2, c0_2, c1_2);
        tq4_cents8_reg(w3, c0_3, c1_3);

        const block_q8_1 * a_blk = &a_base[ib];
        const float d_act = __half2float((__half)a_blk->ds.x);
        const int * a_qs = (const int *)(a_blk->qs);

        const int s0 = ggml_cuda_dp4a(c0_0, a_qs[0], ggml_cuda_dp4a(c1_0, a_qs[1],
                       ggml_cuda_dp4a(c0_1, a_qs[2], ggml_cuda_dp4a(c1_1, a_qs[3], 0))));
        const int s1 = ggml_cuda_dp4a(c0_2, a_qs[4], ggml_cuda_dp4a(c1_2, a_qs[5],
                       ggml_cuda_dp4a(c0_3, a_qs[6], ggml_cuda_dp4a(c1_3, a_qs[7], 0))));

        sumf += d_act * (fd0 * (float)s0 + fd1 * (float)s1);
    }

    sumf *= TQ4_CENTROID_I8_RESCALE;
    #pragma unroll
    for (int offset = LPR / 2; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset, LPR);
    }

    if (lane == 0) {
        dst[sample * stride_sample_dst + (int64_t) expert_slot * stride_channel_dst + row] = sumf;
    }
}

void ggml_cuda_mul_mat_tq(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_tq_fast_path(src0, src1, nullptr, dst));
    GGML_ASSERT(src0->type == GGML_TYPE_TQ4_1S);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst));
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1);
    const int k = src0->ne[0];
    const int m = src0->ne[1];
    const int n = src1->ne[1];
    GGML_ASSERT(k == src1->ne[0] && k % QK_TQ4_1S == 0 && n >= 1 && n <= 8);

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<block_q8_1> act(ctx.pool(), int64_t(k)*n/QK_TQ4_1S);
    const int64_t n_blocks = int64_t(k)*n/QK_TQ4_1S;
    const dim3 block(WARP_SIZE, MMVQ_TQ_NWARPS);
    const dim3 grid((n_blocks + MMVQ_TQ_NWARPS - 1)/MMVQ_TQ_NWARPS);
    tq_prerotate_q8_1<<<grid, block, 0, stream>>>((const float *) src1->data, act.get(), k*n);

#define LAUNCH_TQ4(N) launch_tq4_1s_multi<N>(src0->data, act.get(), (float *) dst->data, k, m, k/32, m, stream)
    switch (n) {
        case 1: LAUNCH_TQ4(1); break;
        case 2: LAUNCH_TQ4(2); break;
        case 3: LAUNCH_TQ4(3); break;
        case 4: LAUNCH_TQ4(4); break;
        case 5: LAUNCH_TQ4(5); break;
        case 6: LAUNCH_TQ4(6); break;
        case 7: LAUNCH_TQ4(7); break;
        case 8: LAUNCH_TQ4(8); break;
    }
#undef LAUNCH_TQ4
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_id_tq(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_tq_fast_path(src0, src1, ids, dst));
    GGML_ASSERT(src0->type == GGML_TYPE_TQ4_1S);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(src1));
    GGML_ASSERT(ids->nb[0] == sizeof(int32_t) && dst->nb[0] == sizeof(float));
    GGML_ASSERT(src0->ne[3] == 1 && src1->ne[3] == 1 && dst->ne[3] == 1);
    const int k = src0->ne[0];
    const int m = src0->ne[1];
    const int n_tokens = dst->ne[2];
    const int n_expert_used = dst->ne[1];
    const int nchannels_y = src1->ne[1];
    GGML_ASSERT(k == src1->ne[0] && k % QK_TQ4_1S == 0);
    GGML_ASSERT(n_tokens >= 1 && n_tokens <= 8);
    GGML_ASSERT(nchannels_y > 0 && n_expert_used % nchannels_y == 0);
    GGML_ASSERT(src1->ne[2] == n_tokens && ids->ne[0] == n_expert_used && ids->ne[1] == n_tokens);

    cudaStream_t stream = ctx.stream();
    const int64_t n_blocks = int64_t(k)*nchannels_y*n_tokens/QK_TQ4_1S;
    ggml_cuda_pool_alloc<block_q8_1> act(ctx.pool(), n_blocks);
    const dim3 block(WARP_SIZE, MMVQ_TQ_NWARPS);
    const dim3 pgrid((n_blocks + MMVQ_TQ_NWARPS - 1)/MMVQ_TQ_NWARPS);
    tq_prerotate_q8_1<<<pgrid, block, 0, stream>>>((const float *) src1->data, act.get(), k*nchannels_y*n_tokens);

    // Device-side ids keep routing capture-safe without persistent address tables or host reads.
    const dim3 grid((m + MMVQ_TQ_NWARPS - 1)/MMVQ_TQ_NWARPS, n_expert_used, n_tokens);
    mul_mat_tq4_1s_dp4a_moe<WARP_SIZE><<<grid, block, 0, stream>>>(
        src0->data, act.get(), (float *) dst->data, (const int32_t *) ids->data, k, m,
        src0->nb[2], ids->nb[1]/sizeof(int32_t), nchannels_y,
        k/QK_TQ4_1S, int64_t(nchannels_y)*(k/QK_TQ4_1S),
        dst->nb[1]/sizeof(float), dst->nb[2]/sizeof(float));
    CUDA_CHECK(cudaGetLastError());
}
