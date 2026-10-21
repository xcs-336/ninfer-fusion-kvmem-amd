#pragma once

// Experimental device addressing only. Not used by published Attention or Engine.
// A table entry is the TRUE start of one PageMajor [256,64,KVHeads] BF16 cell.
// The allocation containing the U64 entries must stay at one stable device address.
// Core publishes/finishes table updates before a consumer starts or replays; volatile
// below is a compiler load requirement, NOT a substitute for that stream dependency.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <type_traits>

namespace ninfer::amd::memory {

struct SegmentedBF16Tables {
    const std::uint64_t* k = nullptr;
    const std::uint64_t* v = nullptr;
};

template <int KVHeads, typename Element>
__device__ __forceinline__ Element* segmented_bf16_address(
    Element* affine_base, const std::uint64_t* page_bases, std::int64_t original_element_offset) {
    static_assert(KVHeads == 2 || KVHeads == 4);
    static_assert(sizeof(std::remove_cv_t<Element>) == 2);
    // Keep the original expression exactly on the affine path. In particular this
    // does not reinterpret a D128 HeadMajor offset as a D256 PageMajor offset.
    if (page_bases == nullptr) return affine_base + original_element_offset;
    constexpr std::int64_t cell_elements = std::int64_t{256} * 64 * KVHeads;
    const auto cell = original_element_offset / cell_elements;
    const auto within = original_element_offset % cell_elements;
    const auto address = static_cast<const volatile std::uint64_t*>(page_bases)[cell];
    return reinterpret_cast<Element*>(static_cast<std::uintptr_t>(address)) + within;
}

} // namespace ninfer::amd::memory
