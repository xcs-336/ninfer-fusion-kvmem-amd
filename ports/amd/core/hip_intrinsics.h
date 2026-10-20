#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
template <class... Args>
inline hipError_t ninfer_hip_func_set_attribute(void (*kernel)(Args...), hipFuncAttribute attribute,
                                               int value) {
    return hipFuncSetAttribute(reinterpret_cast<const void*>(kernel), attribute, value);
}
// HIP exposes scalar RNE conversion as __float2bfloat16 and the pair operation
// with a float2 argument. Preserve the original rounding boundary at each call.
__host__ __device__ inline __hip_bfloat162 ninfer_hip_floats2bfloat162_rn(float lo, float hi) {
    return __float22bfloat162_rn(make_float2(lo, hi));
}
__host__ __device__ inline __hip_bfloat162 ninfer_hip_float2bfloat162_rn(float value) {
    return __float22bfloat162_rn(make_float2(value, value));
}
// The production kernels have a 32-lane contract. Native HIP sync intrinsics
// require an explicitly 64-bit mask even on gfx1201 wave32; widen the mask, not
// the logical wave or the lane numbering.
template <class T>
__device__ inline T ninfer_hip_shfl_sync(unsigned mask, T value, int source, int width = 32) {
    return __shfl_sync(static_cast<unsigned long long>(mask), value, source, width);
}
template <class T>
__device__ inline T ninfer_hip_shfl_xor_sync(unsigned mask, T value, int lane_mask, int width = 32) {
    return __shfl_xor_sync(static_cast<unsigned long long>(mask), value, lane_mask, width);
}
template <class T>
__device__ inline T ninfer_hip_shfl_down_sync(unsigned mask, T value, unsigned delta, int width = 32) {
    return __shfl_down_sync(static_cast<unsigned long long>(mask), value, delta, width);
}
template <class T>
__device__ inline T ninfer_hip_shfl_up_sync(unsigned mask, T value, unsigned delta, int width = 32) {
    return __shfl_up_sync(static_cast<unsigned long long>(mask), value, delta, width);
}
__device__ inline unsigned ninfer_hip_ballot_sync(unsigned mask, int predicate) {
    return static_cast<unsigned>(__ballot_sync(static_cast<unsigned long long>(mask), predicate));
}
__device__ inline int ninfer_hip_all_sync(unsigned mask, int predicate) {
    return __all_sync(static_cast<unsigned long long>(mask), predicate);
}
__device__ inline int ninfer_hip_any_sync(unsigned mask, int predicate) {
    return __any_sync(static_cast<unsigned long long>(mask), predicate);
}
__device__ inline void ninfer_hip_syncwarp(unsigned mask = 0xffffffffu) {
    __syncwarp(static_cast<unsigned long long>(mask));
}
template <class T>
__device__ inline T ninfer_hip_reduce_max_sync(unsigned mask, T value) {
    return __reduce_max_sync(static_cast<unsigned long long>(mask), value);
}
template <class T>
__device__ inline T ninfer_hip_reduce_min_sync(unsigned mask, T value) {
    return __reduce_min_sync(static_cast<unsigned long long>(mask), value);
}
template <class T>
__device__ inline T ninfer_hip_reduce_add_sync(unsigned mask, T value) {
    return __reduce_add_sync(static_cast<unsigned long long>(mask), value);
}
template <class T>
__device__ inline T ninfer_hip_reduce_and_sync(unsigned mask, T value) {
    return __reduce_and_sync(static_cast<unsigned long long>(mask), value);
}
template <class T>
__device__ inline T ninfer_hip_reduce_or_sync(unsigned mask, T value) {
    return __reduce_or_sync(static_cast<unsigned long long>(mask), value);
}
template <class T>
__device__ inline T ninfer_hip_reduce_xor_sync(unsigned mask, T value) {
    return __reduce_xor_sync(static_cast<unsigned long long>(mask), value);
}
