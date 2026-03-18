/*
 * RasterDB CUDA runtime compatibility shim.
 * Provides stub types so existing code compiles without CUDA toolkit.
 * All CUDA runtime calls become no-ops or throw at runtime.
 */
#pragma once

#include <cstddef>
#include <cstring>
#include <cstdint>
#include <stdexcept>

// CUDA attribute macros — no-ops in CPU compilation
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __shared__
#define __shared__
#endif
#ifndef __constant__
#define __constant__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif

// Basic CUDA types
typedef int cudaError_t;
typedef void* cudaStream_t;
typedef void* cudaEvent_t;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

// cudaError constants
static constexpr cudaError_t cudaSuccess = 0;
static constexpr cudaError_t cudaErrorMemoryAllocation = 2;

// Stub CUDA runtime functions
inline cudaError_t cudaStreamCreate(cudaStream_t*) { return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }

inline cudaError_t cudaMalloc(void** ptr, size_t size) {
    *ptr = ::malloc(size);
    return *ptr ? cudaSuccess : cudaErrorMemoryAllocation;
}
inline cudaError_t cudaFree(void* ptr) { ::free(ptr); return cudaSuccess; }

inline cudaError_t cudaMallocHost(void** ptr, size_t size) {
    *ptr = ::malloc(size);
    return *ptr ? cudaSuccess : cudaErrorMemoryAllocation;
}
inline cudaError_t cudaFreeHost(void* ptr) { ::free(ptr); return cudaSuccess; }

inline cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int) {
    *ptr = ::malloc(size);
    return *ptr ? cudaSuccess : cudaErrorMemoryAllocation;
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind) {
    std::memcpy(dst, src, count);
    return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                                    cudaMemcpyKind, cudaStream_t = nullptr) {
    std::memcpy(dst, src, count);
    return cudaSuccess;
}
inline cudaError_t cudaMemset(void* ptr, int value, size_t count) {
    std::memset(ptr, value, count);
    return cudaSuccess;
}
inline cudaError_t cudaMemsetAsync(void* ptr, int value, size_t count, cudaStream_t = nullptr) {
    std::memset(ptr, value, count);
    return cudaSuccess;
}

inline cudaError_t cudaGetDevice(int* device) { *device = 0; return cudaSuccess; }
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }

inline const char* cudaGetErrorString(cudaError_t) { return "RasterDB: CUDA shim (no real GPU)"; }
inline const char* cudaGetErrorName(cudaError_t) { return "cudaSuccess"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaPeekAtLastError() { return cudaSuccess; }

// Event flags
static constexpr unsigned int cudaEventDefault = 0;
static constexpr unsigned int cudaEventBlockingSync = 1;
static constexpr unsigned int cudaEventDisableTiming = 2;

// Event stubs
inline cudaError_t cudaEventCreate(cudaEvent_t*) { return cudaSuccess; }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned int) { return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = nullptr) { return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) { *ms = 0; return cudaSuccess; }

// Device properties stub
struct cudaDeviceProp {
    char name[256] = "RasterDB Vulkan Device";
    size_t totalGlobalMem = 0;
    int major = 0, minor = 0;
    int multiProcessorCount = 0;
    int maxThreadsPerBlock = 256;
    int warpSize = 32;
};

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int) {
    *prop = cudaDeviceProp{};
    return cudaSuccess;
}

// Memory info
inline cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
    *free = 0; *total = 0;
    return cudaSuccess;
}

// Host register/unregister
static constexpr unsigned int cudaHostAllocDefault = 0;
static constexpr unsigned int cudaHostAllocPortable = 1;
static constexpr unsigned int cudaHostAllocMapped = 2;
static constexpr unsigned int cudaHostAllocWriteCombined = 4;
static constexpr unsigned int cudaHostRegisterDefault = 0;

inline cudaError_t cudaHostRegister(void*, size_t, unsigned int) { return cudaSuccess; }
inline cudaError_t cudaHostUnregister(void*) { return cudaSuccess; }

// dim3 type
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1) : x(x_), y(y_), z(z_) {}
};
