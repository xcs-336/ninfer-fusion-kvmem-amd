#pragma once

// SOURCE ONLY candidate: these wrappers instantiate the copied ACTUAL native
// append/small-T/prompt kernel bodies. No alternative Attention implementation,
// weight/cache repack, hidden allocation, affine old-view library, or Engine hook.
// Generate the three private bodies with segmented_bf16_ops_overlay.py first.
#include "core/paged_kv_cache.h"
#ifndef NINFER_AMD_SEGMENTED_POOL_ABI
#error "Requires isolated rebuilt Core view, not the published old-view Op archives"
#endif
#include "amd-memory/segmented_bf16_view.cuh"
#include "amd-memory/segmented_bf16_append.cuh"
#include "amd-memory/segmented_bf16_small_t.cuh"
#include "amd-memory/segmented_bf16_prompt.cuh"
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::amd::memory {

namespace segmented_bf16_detail {
inline void check(hipError_t value) {
    if (value != hipSuccess) throw std::runtime_error(hipGetErrorString(value));
}
inline void demand(bool value, const char* text) {
    if (!value) throw std::invalid_argument(text);
}
inline void tensor(const Tensor& value, DType dtype, int d0, int d1, int d2) {
    demand(value.dtype == dtype && value.data != nullptr && value.is_contiguous() &&
           value.ne[0] == d0 && value.ne[1] == d1 && value.ne[2] == d2 && value.ne[3] == 1,
           "segmented BF16 candidate: tensor shape/dtype/contiguity is unsupported");
    demand((reinterpret_cast<std::uintptr_t>(value.data) & 15u) == 0,
           "segmented BF16 candidate: tensor is not 16-byte aligned");
}
inline int cache(const PagedKVLayerView& value, PagedKVPlaneOrder order) {
    demand(order == PagedKVPlaneOrder::PageMajor && value.head_dim == 256 &&
           (value.num_kv_heads == 2 || value.num_kv_heads == 4) &&
           value.storage == KvCacheStorage::BFloat16 && value.selected_blocks == nullptr &&
           value.k_scale_pages.data == nullptr && value.v_scale_pages.data == nullptr,
           "candidate accepts only dense, single-request, PageMajor D256 BF16 without scales");
    const int groups = value.k_pages.ne[3];
    demand(groups > 0 && value.v_pages.ne[3] == groups,
           "segmented BF16 candidate: invalid physical group inventory");
    for (const Tensor* plane : {&value.k_pages, &value.v_pages}) {
        demand(plane->data && plane->dtype == DType::BF16 && plane->is_contiguous() &&
               plane->ne[0] == 256 && plane->ne[1] == 64 &&
               plane->ne[2] == value.num_kv_heads && plane->ne[3] == groups &&
               (reinterpret_cast<std::uintptr_t>(plane->data) & 15u) == 0,
               "segmented BF16 candidate: code plane must be PageMajor [256,64,KVHeads,groups]");
    }
    const int pages = value.block_table.ne[0];
    tensor(value.block_table, DType::I32, pages, 1, 1);
    demand(pages > 0 && std::int64_t{pages} * 64 <= std::numeric_limits<int>::max(),
           "segmented BF16 candidate: logical capacity overflow");
    demand((value.k_page_bases == nullptr) == (value.v_page_bases == nullptr),
           "segmented BF16 candidate: both semantic planes must use the same addressing mode");
    for (const auto* table : {value.k_page_bases, value.v_page_bases}) {
        demand((reinterpret_cast<std::uintptr_t>(table) & 7u) == 0,
               "segmented BF16 candidate: U64 table pointer is unaligned");
    }
    return pages * 64;
}
inline SegmentedBF16Tables tables(const PagedKVLayerView& value) {
    return {value.k_page_bases, value.v_page_bases};
}
inline void query(const Tensor& q, const Tensor& positions, const Tensor& out,
                  const PagedKVLayerView& value, float scale) {
    const int width = q.ne[2];
    demand(width > 0 && width <= std::int64_t{value.block_table.ne[0]} * 64 &&
           std::isfinite(scale) && scale > 0,
           "segmented BF16 candidate: invalid query width or softmax scale");
    const int heads = value.num_kv_heads == 4 ? 24 : 16;
    tensor(q, DType::BF16, 256, heads, width);
    tensor(out, DType::BF16, 256, heads, width);
    tensor(positions, DType::I32, width, 1, 1);
}
} // namespace segmented_bf16_detail

// Positions are the original device contract: a contiguous nonnegative run within
// the declared logical capacity, block-table entries are valid minted live group
// IDs, and each U64 table has exactly the physical group inventory. No host readback
// is hidden in these launchers. A future public wrapper must preserve/check its
// original execution envelope rather than infer bounds from a captured position.
// Dense valid history through positions[0]+width-1 is required; prompt staging
// computes the page base before its per-row tail predicate. The host shape checks
// here do not inspect or certify any device position/block-table/table-entry value.
// The caller owns all partials; capacity covers the maximum original BF16 split
// policy, so Graph replay can change positions within the declared logical cache.
struct SegmentedBF16Partials {
    float* acc = nullptr;
    float* maximum = nullptr;
    float* denominator = nullptr;
    std::uint64_t acc_elements = 0, stat_elements = 0;
};

inline void segmented_bf16_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                                  const PagedKVLayerView& cache, PagedKVPlaneOrder order,
                                  hipStream_t stream) {
    using namespace segmented_bf16_detail;
    (void)segmented_bf16_detail::cache(cache, order);
    const int width = k.ne[2];
    demand(width > 0 && width <= std::int64_t{cache.block_table.ne[0]} * 64,
           "segmented BF16 candidate: append width exceeds logical capacity");
    tensor(k, DType::BF16, 256, cache.num_kv_heads, width);
    tensor(v, DType::BF16, 256, cache.num_kv_heads, width);
    tensor(positions, DType::I32, width, 1, 1);
    const auto metadata = ops::PagedKVDirectMetadata{
        static_cast<const std::int32_t*>(cache.block_table.data), nullptr};
    const auto launch = [&]<typename Geometry>() {
        const auto vectors = std::int64_t{width} * Geometry::KVHeads * 32;
        demand(vectors <= std::numeric_limits<int>::max(), "candidate append grid overflow");
        ops::segmented_kv_cache_append_full_bf16_kernel<Geometry><<<
            static_cast<unsigned>((vectors + 255) / 256), 256, 0, stream>>>(
            static_cast<const __hip_bfloat16*>(k.data), static_cast<const __hip_bfloat16*>(v.data),
            static_cast<const std::int32_t*>(positions.data), metadata,
            static_cast<__hip_bfloat16*>(cache.k_pages.data),
            static_cast<__hip_bfloat16*>(cache.v_pages.data), width, tables(cache));
    };
    if (cache.num_kv_heads == 4) launch.template operator()<ops::KVCacheAppendD256Kv4>();
    else launch.template operator()<ops::KVCacheAppendD256Kv2>();
    check(hipGetLastError());
}

template <typename CacheInput>
inline void segmented_bf16_small(const Tensor& q, CacheInput input, const Tensor& positions,
                                 float scale, const PagedKVLayerView& cache,
                                 PagedKVPlaneOrder order, SegmentedBF16Partials partials,
                                 Tensor& out, hipStream_t stream) {
    using namespace segmented_bf16_detail;
    const int capacity = segmented_bf16_detail::cache(cache, order);
    query(q, positions, out, cache, scale);
    const int width = q.ne[2];
    demand(width <= (cache.num_kv_heads == 4 ? 8 : 6), "unsupported original small-T geometry");
    const auto launch = [&]<typename Geometry>() {
        constexpr int splits = Geometry::SmallTMaximumSplits;
        const auto stats = std::uint64_t{Geometry::QHeads} * width * splits;
        demand(partials.acc && partials.maximum && partials.denominator &&
               partials.stat_elements >= stats && partials.acc_elements >= stats * 256,
               "segmented BF16 candidate: caller partial capacity is insufficient");
        const auto tile = [&]<int Tokens, int Warps>() {
            ops::segmented_causal_attention_small_t_tc_partial_bf16_kernel<
                Geometry, Tokens, Warps, false, false><<<dim3(Geometry::KVHeads, splits),
                Warps * 32, 0, stream>>>(
                static_cast<const __hip_bfloat16*>(q.data), input,
                static_cast<const std::int32_t*>(positions.data),
                static_cast<__hip_bfloat16*>(cache.k_pages.data),
                static_cast<__hip_bfloat16*>(cache.v_pages.data),
                static_cast<const std::int32_t*>(cache.block_table.data), nullptr, nullptr,
                cache.block_table.ne[0], width, width, 0, capacity, scale,
                partials.acc, partials.maximum, partials.denominator, nullptr, tables(cache));
        };
        switch (width) {
        case 1: tile.template operator()<1, 2>(); break;
        case 2: tile.template operator()<2, 4>(); break;
        case 3: tile.template operator()<3, 4>(); break;
        case 4: tile.template operator()<4, 4>(); break;
        case 5: tile.template operator()<5, 4>(); break;
        case 6: tile.template operator()<6, 4>(); break;
        case 7:
            if constexpr (Geometry::QHeads == 24) tile.template operator()<7, 4>();
            break;
        case 8:
            if constexpr (Geometry::QHeads == 24) tile.template operator()<8, 4>();
            break;
        default: throw std::invalid_argument("unsupported small-T width");
        }
        check(hipGetLastError());
        constexpr int chunk = Geometry::QHeads == 24 ? 256 : 64;
        ops::causal_attention_small_t_reduce_output_kernel<
            Geometry, chunk, false, false, false, false><<<
            dim3(Geometry::QHeads, 256 / chunk, width), 256, 0, stream>>>(
            partials.acc, partials.maximum, partials.denominator,
            static_cast<const std::int32_t*>(positions.data), nullptr,
            width, width, 0, 1, splits, 0, static_cast<__hip_bfloat16*>(out.data), nullptr);
        check(hipGetLastError());
    };
    if (cache.num_kv_heads == 4) launch.template operator()<ops::CausalD256H24Kv4>();
    else launch.template operator()<ops::CausalD256H16Kv2>();
}

inline void segmented_bf16_small_cached(const Tensor& q, const Tensor& positions, float scale,
    const PagedKVLayerView& cache, PagedKVPlaneOrder order, SegmentedBF16Partials partials,
    Tensor& out, hipStream_t stream) {
    segmented_bf16_small(q, ops::CausalCachedInput{}, positions, scale, cache, order,
                         partials, out, stream);
}

inline void segmented_bf16_small_append(const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& positions, float scale, const PagedKVLayerView& cache,
    PagedKVPlaneOrder order, SegmentedBF16Partials partials, Tensor& out, hipStream_t stream) {
    segmented_bf16_detail::tensor(k, DType::BF16, 256, cache.num_kv_heads, q.ne[2]);
    segmented_bf16_detail::tensor(v, DType::BF16, 256, cache.num_kv_heads, q.ne[2]);
    segmented_bf16_small(q, ops::CausalAppendInput{static_cast<const __hip_bfloat16*>(k.data),
                         static_cast<const __hip_bfloat16*>(v.data)}, positions, scale,
                         cache, order, partials, out, stream);
}

inline void segmented_bf16_prompt_cached(const Tensor& q, const Tensor& positions, float scale,
    const PagedKVLayerView& cache, PagedKVPlaneOrder order, Tensor& out, hipStream_t stream) {
    using namespace segmented_bf16_detail;
    (void)segmented_bf16_detail::cache(cache, order);
    query(q, positions, out, cache, scale);
    static_assert(ops::kCausalPromptBc == 32 && ops::kCausalPromptBr == 32 &&
                  ops::kCausalPromptThreads == 64 && ops::kCausalPromptSmemBytes == 49152);
    const auto metadata = ops::PagedKVDirectMetadata{
        static_cast<const std::int32_t*>(cache.block_table.data), nullptr};
    const auto launch = [&]<typename Geometry>() {
        ops::segmented_causal_attention_prompt_bf16_kernel<Geometry><<<
            dim3(static_cast<unsigned>((std::int64_t{q.ne[2]} + 31) / 32), Geometry::QHeads),
            64, 49152, stream>>>(
            static_cast<const __hip_bfloat16*>(q.data),
            static_cast<const __hip_bfloat16*>(cache.k_pages.data),
            static_cast<const __hip_bfloat16*>(cache.v_pages.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__hip_bfloat16*>(out.data), q.ne[2], tables(cache));
    };
    if (cache.num_kv_heads == 4) launch.template operator()<ops::CausalD256H24Kv4>();
    else launch.template operator()<ops::CausalD256H16Kv2>();
    check(hipGetLastError());
}

// Call before capture, exactly as the original launcher configures the actual
// kernel's dynamic LDS limit. This is host preparation, not a captured operation.
inline void segmented_bf16_prepare_prompt(int kv_heads, hipStream_t stream) {
    using namespace segmented_bf16_detail;
    demand(kv_heads == 4 || kv_heads == 2, "unsupported BF16 prompt head geometry");
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    check(hipStreamIsCapturing(stream, &status));
    demand(status == hipStreamCaptureStatusNone, "configure prompt LDS before capture");
    const auto prepare = [&]<typename Geometry>() {
        check(ninfer_hip_func_set_attribute(
            ops::segmented_causal_attention_prompt_bf16_kernel<Geometry, ops::PagedKVDirectMetadata>,
            hipFuncAttributeMaxDynamicSharedMemorySize, ops::kCausalPromptSmemBytes));
    };
    if (kv_heads == 4) prepare.template operator()<ops::CausalD256H24Kv4>();
    else prepare.template operator()<ops::CausalD256H16Kv2>();
}

inline void segmented_bf16_prompt_append(const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& positions, float scale, const PagedKVLayerView& cache,
    PagedKVPlaneOrder order, Tensor& out, hipStream_t stream) {
    segmented_bf16_append(k, v, positions, cache, order, stream);
    segmented_bf16_prompt_cached(q, positions, scale, cache, order, out, stream);
}

} // namespace ninfer::amd::memory
