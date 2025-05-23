/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#pragma once

#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"
#include <iostream>

namespace tensorrt_llm
{
namespace kernels
{
struct TrtllmGenBlockScaleGemmKernelParams
{
    //
    // Gemm parameters.
    //

    // TMA descriptor for A.
    CUtensorMap tmaA;
    // TMA descriptor for B.
    CUtensorMap tmaB;
    // TMA descriptor for C, (when useTmaStore is true)
    CUtensorMap tmaC;

    // The output matrix C. The shape is m x n. Layout is row-major (contiguous in the n dimension).
    // (when useTmaStore is false)
    void* ptrC;

    // The scaling factors to dequantize A. It is used when the DeepSeek Fp8 recipe is enabled.
    float const* ptrDqSfsA;
    // The scaling factors to dequantize B. It is used when the DeepSeek Fp8 recipe is enabled.
    float const* ptrDqSfsB;
    // The scaling factors calculated when quantizing C, for MxFp{4,8} and NvFp4 formats.
    void* ptrSfC;

    // TMA descriptor for the block scale factors of A, for MxFp{4,8} and NvFp4 formats.
    CUtensorMap tmaSfA;
    // TMA descriptor for the block scale factors of B, for MxFp{4,8} and NvFp4 formats.
    CUtensorMap tmaSfB;

    // The device output scale for FP8 quantization. It can either be a static value passed to the
    // kernel or it can be computed by the kernel. TensorRT-LLM fp8 kernels expect a single scaling
    // factor on the device.
    //
    // When DeepSeek FP8 recipe is used, the array is filled with dequantization factors to later
    // dequantize the C values.
    float* ptrScaleC;

    // The M dimension. It is the total number of tokens if A is the activation matrix.
    int32_t m;
    // The N dimension. It is the number of output channels if B is the weight matrix.
    int32_t n;
    // The K dimension. It is the hidden dimension of the input matrices.
    int32_t k;

    //
    // All-reduce parameters.
    //

    // The rank id.
    int rank;
    // The number of peer devices in tensor-parallel group.
    int tpGrpSize;
    // Pointer for output with multicast mapping. It is used by the "reduce" op (LDGMC.ADD) of the
    // two-shot reduce-scatter phase.
    void* multimemC;
    // Pointer for partial sums for split-k computation.
    void* ptrPartialSumsForSplitK;

    // The barriers in global memory.
    //
    // The kernel arrives on (with release ordering) the multicast mapping of the barrier to broadcast
    // amongst peer devices. It then waits (with acquire ordering) on the unicast mapping of the
    // barrier.
    //
    // Flags in global memory that sync on "entrance" of reduce-scatter phase in two-shot all-reduce.
    void* ptrTileBars;
    void* multimemTileBars;

    // Flags in global memory that sync on "exit" after the all-reduce finishes.
    void* ptrCompletionBars;
    void* multimemCompletionBars;

    // The barriers in global memory for split k reduction.
    // The kernel arrives on the barrier and CtaIdx.z == 0 waits
    // on the barrier to flip to perform a reduction.
    void* ptrSplitKCompletionBars;

    //
    // Methods.
    //

    enum class MatrixType
    {
        MatrixA = 0,
        MatrixB
    };

    // Create the TMA shape/stride for A/B.
    template <class GemmOptions>
    static auto makeTmaShapeStrideAb(GemmOptions const& options, MatrixType matrixType)
    {
        // The outer dimension.
        auto numTokens = (matrixType == MatrixType::MatrixA) ? options.mM : options.mN;
        // The inner dimension.
        auto hiddenSize = options.mK;
        // The cute tensor shape for A/B: (numTokens, hiddenSize).
        // Note that TMA descriptor expects the first dimension's stride to be
        // 1, so swap the first two dimension so that the hiddenSize dimension comes first.
        auto shape = std::vector<uint64_t>{static_cast<uint64_t>(hiddenSize), static_cast<uint64_t>(numTokens)};

        // Assemble the stride (strideTokens, 1).
        // Swap the first two dimension as mentioned before.
        auto stride = std::vector<uint64_t>{1, static_cast<uint64_t>(hiddenSize)};

        return std::make_tuple(shape, stride);
    }

    // Create the TMA shape/stride for C.
    template <class GemmOptions>
    static auto makeTmaShapeStrideC(GemmOptions const& options)
    {
        // The number of tokens.
        auto numTokens = options.mTransposeMmaOutput ? options.mN : options.mM;
        // The hidden dimension.
        auto hiddenSize = options.mTransposeMmaOutput ? options.mM : options.mN;
        // Note that TMA descriptor expects the first dimension's stride to be
        // 1, so swap the first two dimension so that the hiddenSize dimension comes first.
        auto shape = std::vector<uint64_t>{static_cast<uint64_t>(hiddenSize), static_cast<uint64_t>(numTokens)};

        // Assemble the stride (strideTokens, 1).
        // Swap the first two dimension as mentioned before.
        auto stride = std::vector<uint64_t>{1, static_cast<uint64_t>(hiddenSize)};

        return std::make_tuple(shape, stride);
    }

    // Create the TMA shape/stride for A/B block scale factors.
    template <class GemmOptions>
    static auto makeTmaShapeStrideSfAb(GemmOptions const& options, MatrixType matrixType)
    {
        // Note: the scale factor tensor packs 128x4 tiles into contiguous 512B blocks.
        // The 512B block maps to a 32x16B (32x128b) block in TMEM.
        // See https://nvbugspro.nvidia.com/bug/4165523
        //
        // Additionally, we have to meet constraints of TMA that the box dimensions are less
        // than 256 and boxDim[0] is a multiple of 16B.
        //
        // The "logical" tensor is:      [outer,        inner / numEltsPerSf]
        // The aforementioned format is: [outer / 128,  inner / numEltsPerSf / 4,    512]
        // The shape we use for TMA is:  [outer / 128,  inner / numEltsPerSf / 4, 2, 256]

        // The outer dimension.
        auto numTokens = matrixType == MatrixType::MatrixA ? options.mM : options.mN;
        // The inner dimension.
        auto hiddenSize = options.mK;

        const int32_t numEltsPerSf = 16;

        auto shape = std::vector<uint64_t>{
            256, 2, static_cast<uint64_t>((hiddenSize / numEltsPerSf / 4)), static_cast<uint64_t>(numTokens / 128)};

        std::vector<uint64_t> stride(shape.size());
        stride[0] = 1;
        for (size_t i = 1; i < shape.size(); i++)
        {
            stride[i] = shape[i - 1] * stride[i - 1];
        }

        return std::make_tuple(shape, stride);
    }

    static CUtensorMap buildNdTmaDescriptor(Data_type dtype, std::vector<uint64_t> const& shapes,
        std::vector<uint64_t> const& strides, int32_t tileSizeMn, int32_t tileSizeK, void* gmemAddr,
        bool doSwizzle = true)
    {
        CUtensorMap desc{};
        // The data type.
        CUtensorMapDataType tmaDataFormat;
        if (dtype == Data_type::DATA_TYPE_E4M3)
        {
            tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_UINT8;
        }
        else if (dtype == Data_type::DATA_TYPE_FP16)
        {
            tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
        }
        else if (dtype == Data_type::DATA_TYPE_BF16)
        {
            tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        }
        else if (dtype == Data_type::DATA_TYPE_E2M1)
        {
            tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
        }
        else if (dtype == Data_type::DATA_TYPE_FP32)
        {
            tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
        }
        else
        {
            std::cerr << "Unexpected dtype " << static_cast<int32_t>(dtype) << std::endl;
            TLLM_CHECK(false);
        }

        // The swizzle type.
        CUtensorMapSwizzle swizzleType{CU_TENSOR_MAP_SWIZZLE_NONE};
        int32_t tileKSizeInBytes = (tileSizeK * get_size_in_bits(dtype)) / /* bits */ 8;
        if (doSwizzle)
        {
            if ((tileKSizeInBytes % 128) == 0)
            {
                swizzleType = CU_TENSOR_MAP_SWIZZLE_128B;
            }
            else if ((tileKSizeInBytes % 64) == 0)
            {
                swizzleType = CU_TENSOR_MAP_SWIZZLE_64B;
            }
            else if ((tileKSizeInBytes % 32) == 0)
            {
                swizzleType = CU_TENSOR_MAP_SWIZZLE_32B;
            }
            else
            {
                std::cerr << "Unexpected tileKSizeInBytes " << tileKSizeInBytes << std::endl;
                TLLM_CHECK(false);
            }
        }

        // Check gmem address must be 16B-aligned
        TLLM_CHECK((reinterpret_cast<uint64_t>(gmemAddr) & 0b1111) == 0); //

        // Check shape must be in range [1, 2^32]
        int32_t dim = shapes.size();
        // Expect 2 dimensions.
        TLLM_CHECK(dim == 2 || dim == 3);
        // Check shape range.
        for (int32_t ii = 0; ii < dim; ++ii)
        {
            TLLM_CHECK(shapes[ii] >= (uint64_t(1)));       // Size must be min 1
            TLLM_CHECK(shapes[ii] <= (uint64_t(1) << 32)); // Size must be max 2^32
        }

        // TMA descriptor does not store the zeroth stride and assumes it is 1.
        TLLM_CHECK(static_cast<int32_t>(strides.size()) == dim);
        TLLM_CHECK(strides[0] == 1);

        // Build strides in bytes.
        // cuTensorMapEncodeTiled ignores the stride of the first dimension (implicitly 1).
        std::vector<uint64_t> stridesInBytes(dim - 1);
        for (int32_t ii = 0; ii < dim - 1; ++ii)
        {
            stridesInBytes[ii] = (strides[ii + 1] * get_size_in_bits(dtype)) / /* bits */ 8;
        }

        // Set the number of elements in the packed uint32_t element.
        auto const numEltsPerUInt32 = 4 * /* bits */ 8 / get_size_in_bits(dtype);
        // The number of elements in 128B.
        auto const numEltsIn128B = numEltsPerUInt32 /*4B*/ * 32;
        // The number of tile K hidden size (per token) in each block of shared memory.
        auto const numEltsInClampedTileKSize = std::min((int32_t) numEltsIn128B, tileSizeK);

        // Build tile shapes.
        std::vector<uint32_t> tileShapes(dim, 1);
        tileShapes[0] = numEltsInClampedTileKSize; // tileSizeK
        tileShapes[1] = tileSizeMn;                // tileSizeMn

        // Set tile strides to 1;
        std::vector<uint32_t> tileStrides(dim, 1);

        // Build the descriptor.
        CUresult result = cuTensorMapEncodeTiled(&desc, tmaDataFormat,
            /*tensorRank=*/dim, gmemAddr, shapes.data(), stridesInBytes.data(), tileShapes.data(), tileStrides.data(),
            /*interleave=*/CU_TENSOR_MAP_INTERLEAVE_NONE, swizzleType,
            /*l2Promotion=*/CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            /*oobFill=*/CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

        if (result != CUDA_SUCCESS)
        {
            std::cerr << "Error: Failed to initialize the TMA descriptor " << result << std::endl;

            std::cerr << "tmaFormat: " << static_cast<int>(tmaDataFormat) << " dim: " << dim << " gmem: " << gmemAddr
                      << std::endl;

            std::cerr << "Shape: ";
            for (int ii = 0; ii < dim; ++ii)
            {
                std::cerr << shapes[ii] << " ";
            }
            std::cerr << std::endl;

            std::cerr << "Stride: ";
            for (int ii = 0; ii < dim - 1; ++ii)
            {
                std::cerr << stridesInBytes[ii] << " ";
            }
            std::cerr << std::endl;

            std::cerr << "tileShapes: ";
            for (int ii = 0; ii < dim; ++ii)
            {
                std::cerr << tileShapes[ii] << " ";
            }
            std::cerr << std::endl;

            std::cerr << "tileStrides: ";
            for (int ii = 0; ii < dim; ++ii)
            {
                std::cerr << tileStrides[ii] << " ";
            }
            std::cerr << std::endl;
            std::cerr << "swizzleType: " << int(swizzleType) << std::endl;
            TLLM_CHECK(false);
        }

        return desc;
    }

    static CUtensorMap buildSfTmaDescriptor(Data_type dtype, std::vector<uint64_t> const& shapes,
        std::vector<uint64_t> const& strides, std::vector<uint32_t> const& tileShapes, void* gmemAddr)
    {
        CUtensorMap desc{};
        CUtensorMapDataType tmaDataFormat;
        if (dtype == Data_type::DATA_TYPE_E4M3)
        {
            tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_UINT8;
        }
        else
        {
            std::cerr << "Unexpected dtype " << static_cast<int32_t>(dtype) << std::endl;
            TLLM_CHECK(false);
        }

        // No swizzle for scaling factors.
        CUtensorMapSwizzle swizzleType = CU_TENSOR_MAP_SWIZZLE_NONE;

        // Check gmem address must be 16B-aligned
        TLLM_CHECK((reinterpret_cast<uint64_t>(gmemAddr) & 0b1111) == 0); //

        // Check shape must be in range [1, 2^32]
        int32_t dim = shapes.size();
        // Check shape range.
        for (int32_t ii = 0; ii < dim; ++ii)
        {
            TLLM_CHECK(shapes[ii] >= (uint64_t(1)));       // Size must be min 1
            TLLM_CHECK(shapes[ii] <= (uint64_t(1) << 32)); // Size must be max 2^32
        }

        // TMA descriptor does not store the zeroth stride and assumes it is 1.
        TLLM_CHECK(static_cast<int32_t>(strides.size()) == dim);
        TLLM_CHECK(strides[0] == 1);

        // Build strides in bytes.
        // cuTensorMapEncodeTiled ignores the stride of the first dimension (implicitly 1).
        std::vector<uint64_t> stridesInBytes(dim - 1);
        for (int32_t ii = 0; ii < dim - 1; ++ii)
        {
            stridesInBytes[ii] = (strides[ii + 1] * get_size_in_bits(dtype)) / /* bits */ 8;
        }

        // Set tile strides to 1;
        std::vector<uint32_t> tileStrides(dim, 1);

        // Build the descriptor.
        CUresult result = cuTensorMapEncodeTiled(/*tensorMap=*/&desc,
            /*tensorDataType=*/tmaDataFormat,
            /*tensorRank=*/dim,
            /*globalAddress=*/gmemAddr,
            /*globalDim=*/shapes.data(),
            /*globalStrides=*/stridesInBytes.data(),
            /*boxDim=*/tileShapes.data(),
            /*elementStrides=*/tileStrides.data(),
            /*interleave=*/CU_TENSOR_MAP_INTERLEAVE_NONE,
            /*swizzle=*/swizzleType,
            /*l2Promotion=*/CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            /*oobFill=*/CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

        if (result != CUDA_SUCCESS)
        {
            std::cerr << "Error: Failed to initialize the TMA descriptor for SF " << result << std::endl;

            std::cerr << "tmaFormat: " << static_cast<int>(tmaDataFormat) << " dim: " << dim << " gmem: " << gmemAddr
                      << std::endl;

            std::cerr << "shape:";
            for (uint32_t shape_i : shapes)
            {
                std::cerr << " " << shape_i;
            }
            std::cerr << std::endl;

            std::cerr << "stridesInBytes:";
            for (uint32_t stride_i : stridesInBytes)
            {
                std::cerr << " " << stride_i;
            }
            std::cerr << std::endl;

            std::cerr << "tileShapes:";
            for (uint32_t tileShape_i : tileShapes)
            {
                std::cerr << " " << tileShape_i;
            }
            std::cerr << std::endl;

            std::cerr << "tileStrides:";
            for (uint32_t tileStride_i : tileStrides)
            {
                std::cerr << " " << tileStride_i;
            }
            std::cerr << std::endl;

            std::cerr << "swizzleType: " << int(swizzleType) << std::endl;
            TLLM_CHECK(false);
        }

        return desc;
    }

    enum class AllReduceAlgo
    {
        None = 0,
        OneShot,
        TwoShot,
    };

    // Setup the kernel parameters.
    template <class GemmOptions_>
    static TrtllmGenBlockScaleGemmKernelParams setKernelParams(GemmOptions_ const& options, void const* ptrA,
        void const* ptrSfA, void const* ptrB, void const* ptrSfB, void* ptrC, void* ptrSfC, void* multimemC,
        float* ptrScaleC, void* ptrPartialSumsForSplitK, void* ptrTileBars, void* multimemTileBars,
        void* ptrCompletionBars, void* multimemCompletionBars, void* ptrSplitKCompletionBars, int rank, int tpGrpSize)
    {

        // Is one-shot all-reduce?
        bool const oneShotAr{options.mAllReduceAlgo == AllReduceAlgo::OneShot};
        // Is two-shot all-reduce?
        bool const twoShotAr{options.mAllReduceAlgo == AllReduceAlgo::TwoShot};
        // Are there peer devices?
        bool const multiDevice{tpGrpSize > 1};

        // Create the return struct.
        TrtllmGenBlockScaleGemmKernelParams params;

        // Shape/stride for gmem tensor A.
        auto [shapeA, strideA] = makeTmaShapeStrideAb(options, MatrixType::MatrixA);
        // Build tma descriptor for A.
        params.tmaA = buildNdTmaDescriptor(
            options.mDtypeElt, shapeA, strideA, options.mTileM, options.mTileK, const_cast<void*>(ptrA));

        // Shape/stride for gmem tensor B.
        auto [shapeB, strideB] = makeTmaShapeStrideAb(options, MatrixType::MatrixB);
        // Build tma descriptor for B.
        params.tmaB = buildNdTmaDescriptor(options.mDtypeElt, shapeB, strideB, options.mTileN, options.mTileK,
            const_cast<void*>(ptrB),
            /* swizzle */ !options.mSliceK);

        if (options.mDtypeElt == Data_type::DATA_TYPE_E2M1)
        {
            const Data_type dTypeSf = Data_type::DATA_TYPE_E4M3;

            const int32_t numEltsPerSf = 16;

            // Build TMA descriptor for gmem A block scale factors.
            auto [shapeSfA, strideSfA] = makeTmaShapeStrideSfAb(options, MatrixType::MatrixA);
            auto tileShapesSfA = std::vector<uint32_t>{256, 2, static_cast<uint32_t>(options.mTileK / numEltsPerSf / 4),
                static_cast<uint32_t>(options.mTileM / 128)};
            params.tmaSfA
                = buildSfTmaDescriptor(dTypeSf, shapeSfA, strideSfA, tileShapesSfA, const_cast<void*>(ptrSfA));

            // Build TMA descriptor for gmem B block scale factors.
            auto [shapeSfB, strideSfB] = makeTmaShapeStrideSfAb(options, MatrixType::MatrixB);
            auto tileShapesSfB = std::vector<uint32_t>{256, 2, static_cast<uint32_t>(options.mTileK / numEltsPerSf / 4),
                static_cast<uint32_t>(options.mTileN / 128)};
            params.tmaSfB
                = buildSfTmaDescriptor(dTypeSf, shapeSfB, strideSfB, tileShapesSfB, const_cast<void*>(ptrSfB));
        }

        if (options.mUseTmaStore)
        {
            // Shape/stride for gmem tensor C.
            auto [shapeC, strideC] = makeTmaShapeStrideC(options);

            // Swap M and N tiles for the M-major epilogue.
            auto outputTileM = options.mTransposeMmaOutput ? options.mEpilogueTileN : options.mEpilogueTileM;
            auto outputTileN = options.mTransposeMmaOutput ? options.mEpilogueTileM : options.mEpilogueTileN;

            // One-shot performs TMA reduction on multicast mapping of the output buffer directly.
            // Two-shot performs TMA store on unicast mapping of the output buffer. The reduction happens
            // in the next phase.
            void* ptrTmaC{oneShotAr && multiDevice ? multimemC : ptrC};
            auto dtypeC{options.mDtypeC};
            // Regardless of output dtype, two-shot all-reduce store partial
            // accumulation results to global memory in float32 precision.
            if (twoShotAr && multiDevice)
            {
                dtypeC = options.mDtypeAcc;
            }

            // Build tma descriptor for C.
            params.tmaC
                = buildNdTmaDescriptor(dtypeC, shapeC, strideC, outputTileM, outputTileN, const_cast<void*>(ptrTmaC));
        }

        // Set the dequantization factors for A and B when DeepSeek FP8 recipe is used.
        params.ptrDqSfsA = (float const*) ptrSfA;
        params.ptrDqSfsB = (float const*) ptrSfB;

        // Also set ptrC (it may be used by the NCCL reduction code in "layers/Llama").
        params.ptrC = ptrC;
        params.ptrScaleC = ptrScaleC;

        // The block scale factors of C for MxFp{4,8} and NvFp4 formats.
        // (not to be confused with the tensor-level scale factor stored in ptrScaleC)
        params.ptrSfC = ptrSfC;

        params.m = options.mM;
        params.n = options.mN;
        params.k = options.mK;

        params.rank = rank;
        params.tpGrpSize = tpGrpSize;

        params.multimemC = multimemC;
        params.ptrPartialSumsForSplitK = ptrPartialSumsForSplitK;
        params.ptrTileBars = ptrTileBars;
        params.multimemTileBars = multimemTileBars;
        params.ptrCompletionBars = ptrCompletionBars;
        params.multimemCompletionBars = multimemCompletionBars;

        params.ptrSplitKCompletionBars = ptrSplitKCompletionBars;

        return params;
    }

    // Setup the kernel parameters.
    template <class GemmOptions_>
    static TrtllmGenBlockScaleGemmKernelParams setKernelParams(GemmOptions_ const& options, void const* ptrA,
        void const* ptrB, void* ptrC, void* multimemC, float const* ptrScaleC, void* ptrTileBars,
        void* multimemTileBars, void* ptrCompletionBars, void* multimemCompletionBars, int rank, int tpGrpSize)
    {
        return setKernelParams(options, ptrA, nullptr, ptrB, nullptr, ptrC, multimemC, ptrScaleC, ptrTileBars,
            multimemTileBars, ptrCompletionBars, multimemCompletionBars, rank, tpGrpSize);
    }
};
} // namespace kernels
} // namespace tensorrt_llm
