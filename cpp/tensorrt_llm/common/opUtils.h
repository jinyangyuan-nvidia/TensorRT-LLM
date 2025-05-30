/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/workspace.h"

#include <NvInferRuntime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif // ENABLE_MULTI_DEVICE

#include <cstring>
#include <map>
#include <memory>
#include <nvml.h>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>

namespace tensorrt_llm::common::op
{

// Write values into buffer
template <typename T>
void write(char*& buffer, T const& val)
{
    std::memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
void read(char const*& buffer, T& val)
{
    auto* valPtr = reinterpret_cast<char*>(&val);
    std::memcpy(valPtr, buffer, sizeof(T));
    buffer += sizeof(T);
}

inline cudaDataType_t trtToCublasDtype(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT: return CUDA_R_32F;
    case nvinfer1::DataType::kHALF: return CUDA_R_16F;
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 9
    case nvinfer1::DataType::kBF16: return CUDA_R_16BF;
#endif
    default: TLLM_THROW("Not supported data type for cuBLAS");
    }
}

// Like std::unique_ptr, but does not prevent generation of default copy constructor when used as class members.
// The copy constructor produces nullptr. So the plugin default copy constructor will not really copy this, and
// your clone() implementation is responsible for initializing such data members.
// With this we can simplify clone() implementation when there are many data members including at least one unique_ptr.
template <typename T, typename Del = std::default_delete<T>>
class UniqPtrWNullCopy : public std::unique_ptr<T, Del>
{
public:
    using std::unique_ptr<T, Del>::unique_ptr;

    // for compatibility with std::make_unique
    explicit UniqPtrWNullCopy(std::unique_ptr<T, Del>&& src)
        : std::unique_ptr<T, Del>::unique_ptr{std::move(src)}
    {
    }

    // copy constructor produces nullptr
    UniqPtrWNullCopy(UniqPtrWNullCopy const&)
        : std::unique_ptr<T, Del>::unique_ptr{}
    {
    }

    // copy assignment copies nothing
    UniqPtrWNullCopy& operator=(UniqPtrWNullCopy const&)
    {
        return *this;
    }
};

namespace
{

template <typename T>
struct hash_helper;

// Base case: use std::hash for basic types
template <typename T>
struct hash_helper
{
    size_t operator()(T const& v) const
    {
        return std::hash<T>{}(v);
    }
};

// Specialization for std::set
template <typename T>
struct hash_helper<std::set<T>>
{
    size_t operator()(std::set<T> const& s) const
    {
        size_t hash_value = 0;
        for (auto const& item : s)
        {
            // Recursively hash each element
            hash_value ^= hash_helper<T>{}(item) + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
        }
        return hash_value;
    }
};

// Helper for tuple hashing
template <typename Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
struct tuple_hash_helper
{
    static size_t hash(Tuple const& tuple)
    {
        size_t hash_value = tuple_hash_helper<Tuple, Index - 1>::hash(tuple);
        return hash_value
            ^ (hash_helper<typename std::tuple_element<Index, Tuple>::type>{}(std::get<Index>(tuple)) + 0x9e3779b9
                + (hash_value << 6) + (hash_value >> 2));
    }
};

// Base case for tuple hashing
template <typename Tuple>
struct tuple_hash_helper<Tuple, 0>
{
    static size_t hash(Tuple const& tuple)
    {
        return hash_helper<typename std::tuple_element<0, Tuple>::type>{}(std::get<0>(tuple));
    }
};

// Specialization for std::tuple
template <typename... Args>
struct hash_helper<std::tuple<Args...>>
{
    size_t operator()(std::tuple<Args...> const& t) const
    {
        return tuple_hash_helper<std::tuple<Args...>>::hash(t);
    }
};

} // namespace

// Main hash struct to be used
template <typename T>
struct hash
{
    size_t operator()(T const& v) const
    {
        return hash_helper<T>{}(v);
    }
};

// for testing only
void const* getCommSessionHandle();
} // namespace tensorrt_llm::common::op

inline bool isBuilding()
{
    auto constexpr key = "IS_BUILDING";
    auto const val = getenv(key);
    return val != nullptr && std::string(val) == "1";
}

#if ENABLE_MULTI_DEVICE
#define NCCLCHECK(cmd)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        ncclResult_t r = cmd;                                                                                          \
        if (r != ncclSuccess)                                                                                          \
        {                                                                                                              \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));                      \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define NCCLCHECK_THROW(cmd)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        ncclResult_t r = cmd;                                                                                          \
        if (TLLM_UNLIKELY(r != ncclSuccess))                                                                           \
        {                                                                                                              \
            TLLM_THROW("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));                  \
        }                                                                                                              \
    } while (0)

std::unordered_map<nvinfer1::DataType, ncclDataType_t>* getDtypeMap();

std::shared_ptr<ncclComm_t> getComm(std::set<int> const& group);

#endif // ENABLE_MULTI_DEVICE

//! To save GPU memory, all the plugins share the same cublas and cublasLt handle globally.
//! Get cublas and cublasLt handle for current cuda context
std::shared_ptr<cublasHandle_t> getCublasHandle();
std::shared_ptr<cublasLtHandle_t> getCublasLtHandle();

#ifndef DEBUG

#define PLUGIN_CHECK(status)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        if (status != 0)                                                                                               \
            abort();                                                                                                   \
    } while (0)

#define ASSERT_PARAM(exp)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
            return STATUS_BAD_PARAM;                                                                                   \
    } while (0)

#define ASSERT_FAILURE(exp)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
            return STATUS_FAILURE;                                                                                     \
    } while (0)

#define CSC(call, err)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t cudaStatus = call;                                                                                 \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            return err;                                                                                                \
        }                                                                                                              \
    } while (0)

#define DEBUG_PRINTF(...)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)

#else

#define ASSERT_PARAM(exp)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
        {                                                                                                              \
            fprintf(stderr, "Bad param - " #exp ", %s:%d\n", __FILE__, __LINE__);                                      \
            return STATUS_BAD_PARAM;                                                                                   \
        }                                                                                                              \
    } while (0)

#define ASSERT_FAILURE(exp)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
        {                                                                                                              \
            fprintf(stderr, "Failure - " #exp ", %s:%d\n", __FILE__, __LINE__);                                        \
            return STATUS_FAILURE;                                                                                     \
        }                                                                                                              \
    } while (0)

#define CSC(call, err)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t cudaStatus = call;                                                                                 \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            printf("%s %d CUDA FAIL %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus));                        \
            return err;                                                                                                \
        }                                                                                                              \
    } while (0)

#define PLUGIN_CHECK(status)                                                                                           \
    {                                                                                                                  \
        if (status != 0)                                                                                               \
        {                                                                                                              \
            DEBUG_PRINTF("%s %d CUDA FAIL %s\n", __FILE__, __LINE__, cudaGetErrorString(status));                      \
            abort();                                                                                                   \
        }                                                                                                              \
    }

#define DEBUG_PRINTF(...)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        printf(__VA_ARGS__);                                                                                           \
    } while (0)

#endif // DEBUG

#define NVML_CHECK(cmd)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        nvmlReturn_t r = cmd;                                                                                          \
        if (r != NVML_SUCCESS)                                                                                         \
        {                                                                                                              \
            printf("Failed, NVML error %s:%d '%s'\n", __FILE__, __LINE__, nvmlErrorString(r));                         \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define NVML_CHECK_THROW(cmd)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        nvmlReturn_t r = cmd;                                                                                          \
        if (TLLM_UNLIKELY(r != NVML_SUCCESS))                                                                          \
        {                                                                                                              \
            TLLM_THROW("Failed, NVML error %s:%d '%s'\n", __FILE__, __LINE__, nvmlErrorString(r));                     \
        }                                                                                                              \
    } while (0)
