/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/preQuantScaleKernel.h"
#include <cub/cub.cuh>

namespace tensorrt_llm
{
namespace kernels
{
namespace
{
template <typename T>
struct Vec2Type;

template <>
struct Vec2Type<half>
{
    using type = half2;
};
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_BF16))
template <>
struct Vec2Type<__nv_bfloat16>
{
    using type = __nv_bfloat162;
};
#endif
}; // namespace

template <typename T_in, typename T_out, int kProcessRows, typename AccessType>
__global__ void apply_per_channel_scale(T_out* smoothed_act, T_in const* act, T_in const* per_channel_scale, int rows,
    int cols, int64_t const* num_valid_tokens_ptr)
{
    static constexpr int kElems = sizeof(AccessType) / sizeof(T_in);
    T_in scale[kElems], act_vec[kElems];
    int col_offset = blockIdx.y * blockDim.x + threadIdx.x;
    int row_offset = blockIdx.x;
    if (col_offset * kElems >= cols || row_offset * kProcessRows >= rows)
        return;
    if (num_valid_tokens_ptr && (row_offset * kProcessRows >= *num_valid_tokens_ptr))
        return;
    act += row_offset * kProcessRows * cols;
    smoothed_act += row_offset * kProcessRows * cols;
    *reinterpret_cast<AccessType*>(scale) = reinterpret_cast<AccessType const*>(per_channel_scale)[col_offset];
#pragma unroll
    for (int i = 0; i < kProcessRows; ++i)
    {
        *reinterpret_cast<AccessType*>(act_vec) = reinterpret_cast<AccessType const*>(act + i * cols)[col_offset];
        if constexpr ((std::is_same_v<T_in, half>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_BF16))
                          || std::is_same_v<T_in, __nv_bfloat16>
#endif
                          ) &&(kElems % 2 == 0))
        {
            using Vec2 = typename Vec2Type<T_in>::type;
#pragma unroll
            for (int j = 0; j < kElems; j += 2)
            {
                *reinterpret_cast<Vec2*>(act_vec + j)
                    = __hmul2(*reinterpret_cast<Vec2*>(act_vec + j), *reinterpret_cast<Vec2*>(scale + j));
            }
        }
        else
        {
#pragma unroll
            for (int j = 0; j < kElems; ++j)
            {
                act_vec[j] = static_cast<T_in>(static_cast<float>(act_vec[j]) * static_cast<float>(scale[j]));
            }
        }
        if constexpr (std::is_same_v<T_in, T_out>)
        {
            reinterpret_cast<AccessType*>(smoothed_act + i * cols)[col_offset]
                = *reinterpret_cast<AccessType*>(act_vec);
        }
        else
        {
#pragma unroll
            for (int j = 0; j < kElems; ++j)
            {
                (smoothed_act + i * cols)[col_offset * kElems + j] = static_cast<T_out>(act_vec[j]);
            }
        }
    }
}

template <typename T_in, typename T_out, typename AccessType>
__global__ void apply_per_channel_scale_low_latency(T_out* smoothed_act, T_in const* act, T_in const* per_channel_scale,
    int rows, int cols, int const* valid_tokens, int max_tokens_per_expert, int num_experts)
{
    using WarpScan = cub::WarpScan<int>;
    extern __shared__ char shared_mem[];
    int* valid_tokens_cumsum = reinterpret_cast<int*>(shared_mem);
    T_in* per_channel_scale_shared = reinterpret_cast<T_in*>(valid_tokens_cumsum + 32);
    WarpScan::TempStorage* temp_storage = reinterpret_cast<WarpScan::TempStorage*>(per_channel_scale_shared + cols);
    static constexpr int kElems = sizeof(AccessType) / sizeof(T_in);
    T_in scale[kElems], act_vec[kElems];
    AccessType* scale_access = reinterpret_cast<AccessType*>(scale);
    AccessType* act_vec_access = reinterpret_cast<AccessType*>(act_vec);
    AccessType* per_channel_scale_shared_access = reinterpret_cast<AccessType*>(per_channel_scale_shared);
    AccessType const* per_channel_scale_access = reinterpret_cast<AccessType const*>(per_channel_scale);
    for (int i = threadIdx.x; i < cols / kElems; i += blockDim.x)
    {
        per_channel_scale_shared_access[i] = per_channel_scale_access[i];
    }
    if (threadIdx.x < 32)
    {
        int valid_token_val = threadIdx.x < num_experts ? valid_tokens[threadIdx.x] : 0;
        WarpScan(*temp_storage).InclusiveSum(valid_token_val, valid_token_val);
        valid_tokens_cumsum[threadIdx.x] = valid_token_val;
    }
    __syncthreads();
    int const num_valid_tokens = valid_tokens_cumsum[31];

    int expert = 0;
    int valid_token_min = 0;
    int valid_token_max = valid_tokens_cumsum[0];
    for (int idx_token = blockIdx.x; idx_token < num_valid_tokens; idx_token += gridDim.x)
    {
        while (idx_token >= valid_token_max)
        {
            ++expert;
            valid_token_min = valid_token_max;
            valid_token_max = valid_tokens_cumsum[expert];
        }
        int offset = (expert * max_tokens_per_expert + idx_token - valid_token_min) * cols;
        AccessType const* act_access = reinterpret_cast<AccessType const*>(act + offset);
        T_out* smoothed_act_out = smoothed_act + offset;
        AccessType* smoothed_act_access = reinterpret_cast<AccessType*>(smoothed_act_out);
        for (int col_offset = threadIdx.x; col_offset < cols / kElems; col_offset += blockDim.x)
        {
            *scale_access = per_channel_scale_shared_access[col_offset];
            *act_vec_access = act_access[col_offset];
            if constexpr ((std::is_same_v<T_in, half>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_BF16))
                              || std::is_same_v<T_in, __nv_bfloat16>
#endif
                              ) &&(kElems % 2 == 0))
            {
                using Vec2 = typename Vec2Type<T_in>::type;
#pragma unroll
                for (int j = 0; j < kElems; j += 2)
                {
                    *reinterpret_cast<Vec2*>(act_vec + j)
                        = __hmul2(*reinterpret_cast<Vec2*>(act_vec + j), *reinterpret_cast<Vec2*>(scale + j));
                }
            }
            else
            {
#pragma unroll
                for (int j = 0; j < kElems; ++j)
                {
                    act_vec[j] = static_cast<T_in>(static_cast<float>(act_vec[j]) * static_cast<float>(scale[j]));
                }
            }
            if constexpr (std::is_same_v<T_in, T_out>)
            {
                smoothed_act_access[col_offset] = *reinterpret_cast<AccessType*>(act_vec);
            }
            else
            {
#pragma unroll
                for (int j = 0; j < kElems; ++j)
                {
                    smoothed_act_out[col_offset * kElems + j] = static_cast<T_out>(act_vec[j]);
                }
            }
        }
    }
}

template <typename T_in, typename T_out, int kProcessRows, typename AccessType = float4>
void apply_per_channel_scale_kernel_launcher_(T_out* smoothed_act, T_in const* act, T_in const* per_channel_scale,
    int rows, int cols, int64_t const* num_valid_tokens_ptr = nullptr, cudaStream_t stream = 0)
{
    static constexpr int kElems = sizeof(AccessType) / sizeof(T_in);
    dim3 block(128);
    dim3 grid((rows + kProcessRows - 1) / kProcessRows, (cols / kElems + block.x - 1) / block.x);
    apply_per_channel_scale<T_in, T_out, kProcessRows, AccessType>
        <<<grid, block, 0, stream>>>(smoothed_act, act, per_channel_scale, rows, cols, num_valid_tokens_ptr);
}

template <typename T_in, typename T_out, typename AccessType = float4>
void apply_per_channel_scale_low_latency_kernel_launcher_(T_out* smoothed_act, T_in const* act,
    T_in const* per_channel_scale, int rows, int cols, int64_t const* num_valid_tokens_ptr = nullptr,
    int32_t const* valid_tokens = nullptr, int64_t max_tokens_per_expert = 0, cudaStream_t stream = 0)
{
    int num_experts = rows / max_tokens_per_expert;
    TLLM_CHECK(num_experts <= 32);
    using WarpScan = cub::WarpScan<int32_t>;
    int smem_size = 32 * sizeof(int32_t) + sizeof(WarpScan::TempStorage) + cols * sizeof(T_in);
    static int const smCount = tensorrt_llm::common::getMultiProcessorCount();
    // Note: Launching 8 blocks per SM can fully leverage the memory bandwidth (tested on B200).
    dim3 grid(std::min(smCount * 8, rows));
    dim3 block(256);
    apply_per_channel_scale_low_latency<T_in, T_out, AccessType><<<grid, block, smem_size, stream>>>(smoothed_act, act,
        per_channel_scale, rows, cols, valid_tokens, static_cast<int32_t>(max_tokens_per_expert), num_experts);
}

template <typename T_in, typename T_out>
void apply_per_channel_scale_kernel_launcher(T_out* smoothed_act, T_in const* act, T_in const* per_channel_scale,
    int rows, int cols, int64_t const* num_valid_tokens_ptr, cudaStream_t stream)
{
    uint64_t elems = static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols);
    if (elems < 2048 * 2048)
    {
        apply_per_channel_scale_kernel_launcher_<T_in, T_out, 1, float4>(
            smoothed_act, act, per_channel_scale, rows, cols, num_valid_tokens_ptr, stream);
    }
    else if (elems < 4096 * 4096)
    {
        apply_per_channel_scale_kernel_launcher_<T_in, T_out, 4, float4>(
            smoothed_act, act, per_channel_scale, rows, cols, num_valid_tokens_ptr, stream);
    }
    else if (elems < 8192 * 8192)
    {
        apply_per_channel_scale_kernel_launcher_<T_in, T_out, 8, float4>(
            smoothed_act, act, per_channel_scale, rows, cols, num_valid_tokens_ptr, stream);
    }
    else
    {
        apply_per_channel_scale_kernel_launcher_<T_in, T_out, 16, float4>(
            smoothed_act, act, per_channel_scale, rows, cols, num_valid_tokens_ptr, stream);
    }
}

template <typename T_in, typename T_out>
void apply_per_channel_scale_low_latency_kernel_launcher(T_out* smoothed_act, T_in const* act,
    T_in const* per_channel_scale, int rows, int cols, int64_t const* num_valid_tokens_ptr, int32_t const* valid_tokens,
    int64_t max_tokens_per_expert, cudaStream_t stream)
{
    if (cols % (sizeof(float4) / sizeof(T_in)) == 0)
    {
        apply_per_channel_scale_low_latency_kernel_launcher_<T_in, T_out, float4>(smoothed_act, act, per_channel_scale,
            rows, cols, num_valid_tokens_ptr, valid_tokens, max_tokens_per_expert, stream);
    }
    else if (cols % (sizeof(float2) / sizeof(T_in)) == 0)
    {
        apply_per_channel_scale_low_latency_kernel_launcher_<T_in, T_out, float2>(smoothed_act, act, per_channel_scale,
            rows, cols, num_valid_tokens_ptr, valid_tokens, max_tokens_per_expert, stream);
    }
    else if (cols % (sizeof(float) / sizeof(T_in)) == 0)
    {
        apply_per_channel_scale_low_latency_kernel_launcher_<T_in, T_out, float>(smoothed_act, act, per_channel_scale,
            rows, cols, num_valid_tokens_ptr, valid_tokens, max_tokens_per_expert, stream);
    }
    else
    {
        apply_per_channel_scale_low_latency_kernel_launcher_<T_in, T_out, T_in>(smoothed_act, act, per_channel_scale,
            rows, cols, num_valid_tokens_ptr, valid_tokens, max_tokens_per_expert, stream);
    }
}

#define INSTANTIATE_PREQUANT_SCALE(T_in, T_out)                                                                        \
    template void apply_per_channel_scale_kernel_launcher<T_in, T_out>(T_out * smoothed_act, const T_in* act,          \
        const T_in* per_channel_scale, int rows, int cols, int64_t const* num_valid_tokens_ptr, cudaStream_t stream)

#define INSTANTIATE_PREQUANT_SCALE_EXTRA(T_in, T_out)                                                                  \
    template void apply_per_channel_scale_low_latency_kernel_launcher<T_in, T_out>(T_out * smoothed_act,               \
        const T_in* act, const T_in* per_channel_scale, int rows, int cols, int64_t const* num_valid_tokens_ptr,       \
        int32_t const* valid_tokens, int64_t max_tokens_per_expert, cudaStream_t stream)

INSTANTIATE_PREQUANT_SCALE(half, half);
INSTANTIATE_PREQUANT_SCALE_EXTRA(half, half);
#if defined(ENABLE_FP8)
INSTANTIATE_PREQUANT_SCALE(half, __nv_fp8_e4m3);
INSTANTIATE_PREQUANT_SCALE_EXTRA(half, __nv_fp8_e4m3);
#endif

#if defined(ENABLE_BF16)
INSTANTIATE_PREQUANT_SCALE(__nv_bfloat16, __nv_bfloat16);
INSTANTIATE_PREQUANT_SCALE_EXTRA(__nv_bfloat16, __nv_bfloat16);
#if defined(ENABLE_FP8)
INSTANTIATE_PREQUANT_SCALE(__nv_bfloat16, __nv_fp8_e4m3);
INSTANTIATE_PREQUANT_SCALE_EXTRA(__nv_bfloat16, __nv_fp8_e4m3);
#endif
#endif

} // namespace kernels
} // namespace tensorrt_llm
