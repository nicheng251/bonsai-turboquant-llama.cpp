#pragma once

#include "common.cuh"

// Shared by dispatch and graph synchronization checks.
bool ggml_cuda_tq_fast_path(const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, const ggml_tensor * dst);

// Contiguous 2D TQ4_1S decode, 1-8 columns. TQ3_1S uses dequantization + cuBLAS.
void ggml_cuda_mul_mat_tq(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Contiguous TQ4_1S experts/activations, 1-8 tokens; ids are read only on-device.
void ggml_cuda_mul_mat_id_tq(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);
