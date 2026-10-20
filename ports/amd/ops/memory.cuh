#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ninfer::ops {
enum class Cache { ca, cg };
template <class V, class T>
__device__ __forceinline__ V load_vec(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 || sizeof(V) == 16);
    return *reinterpret_cast<const V*>(ptr);
}
template <class V, class T>
__device__ __forceinline__ V load_ldg(const T* ptr) {
    return __ldg(reinterpret_cast<const V*>(ptr));
}
template <class T, class V>
__device__ __forceinline__ void store_vec(T* ptr, V value) { *reinterpret_cast<V*>(ptr) = value; }

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
    // LLVM address space 3 is a 32-bit workgroup/LDS offset. The explicit
    // address-space cast also works when the generic LDS aperture is nonzero.
    auto* local = (const unsigned char __attribute__((address_space(3)))*)ptr;
    return static_cast<unsigned>(reinterpret_cast<std::uintptr_t>(local));
#else
    return 0;
#endif
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* dst, const void* src, int source_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16);
    if constexpr (Policy == Cache::cg) { static_assert(Bytes == 16); }
    auto* to = static_cast<unsigned char*>(dst);
    const auto* from = static_cast<const unsigned char*>(src);
    // This qualified baseline completes each copy in the issuing thread. It is
    // stronger than cp.async ordering, with no promise of NVIDIA-style overlap.
    // The owning kernels retain their existing workgroup barriers.
#pragma unroll
    for (int i = 0; i < Bytes; ++i) { to[i] = i < source_bytes ? from[i] : 0; }
}
template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* dst, const void* src) {
    cp_async_zfill<Bytes, Policy>(dst, src, Bytes);
}
__device__ __forceinline__ void cp_commit() { asm volatile("" ::: "memory"); }
template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7);
    asm volatile("" ::: "memory");
}
template <int Bytes>
__device__ __forceinline__ void pipe_copy(void* dst, const void* src) { cp_async<Bytes>(dst, src); }
__device__ __forceinline__ void pipe_commit() { cp_commit(); }
template <int Groups>
__device__ __forceinline__ void pipe_wait() { cp_wait<Groups>(); }
} // namespace ninfer::ops
