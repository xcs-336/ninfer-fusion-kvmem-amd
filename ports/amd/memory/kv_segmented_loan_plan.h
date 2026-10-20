#pragma once

// Experimental Core byte-placement planner. SOURCE ONLY / NOT BUILT / NOT QUALIFIED.
// No HIP calls, no mutation, no production integration. One homogeneous page-slice class:
// first intended consumer is a single-rank, page-major BF16 Main pool, with equal-size
// K/V slices in every plane. Models choose the use; Ops must resolve every plane address.
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::amd::memory {

struct KVSegmentedPlacementSnapshot {
    std::uint32_t planes = 0;
    std::uint32_t page_groups = 0;
    std::uint64_t slot_bytes = 0;
    // Cell (plane,page) has index plane*page_groups+page. This is a permutation of
    // ALL physical slots across this class, allowing equal-size cross-plane relocation.
    std::span<const std::uint32_t> slot_of_cell;
    // The existing pool uses vector<bool>/free runs: materialized snapshots additionally
    // require page_groups bytes + 4*free_count bytes, beyond placement/planning metadata.
    std::span<const std::uint8_t> page_allocated; // exactly page_groups booleans
    std::span<const std::uint32_t> free_pages;   // sorted, distinct, excludes lent IDs
    // Free IDs can also back outstanding reservations. Only this many may be lent.
    std::uint32_t available_pages = 0;
    // Echoed only. A future commit binds the owning pool and freezes this entire state,
    // or revalidates an epoch changing on allocation/free/placement, not capacity alone.
    std::uint64_t resource_revision = 0;
};

struct KVSegmentedSwap {
    std::uint32_t owner_cell = 0;
    std::uint32_t loan_cell = 0;
    std::uint32_t source_slot = 0;
    std::uint32_t destination_slot = 0;
    bool copy_live_bytes = false;
};

struct KVSegmentedLoanPlan {
    std::uint64_t resource_revision = 0;
    std::uint64_t window_offset = 0; // relative to the homogeneous class backing
    std::uint64_t window_bytes = 0;
    std::uint64_t slot_bytes = 0;
    std::uint64_t copied_live_bytes = 0;
    std::vector<std::uint32_t> loan_page_ids;
    // Each pair swaps its cell pointers after copying live owner bytes to the donor slot.
    // Sources lie inside the window, destinations outside; distinct donors mean no cycle,
    // overlap, scratch page or second device workspace is required.
    std::vector<KVSegmentedSwap> swaps;
};

// Pure host plan: invalid snapshots throw; insufficient UNPROMISED capacity returns nullopt.
// Call only with no earlier loan open. The owner must revalidate revision, lend all selected
// IDs transactionally, freeze/revalidate allocation AND placement until publication,
// wait all readers/producers, copy, publish the stable device table,
// then expose ready-event-ordered raw memory. This function performs NONE of those steps.
[[nodiscard]] std::optional<KVSegmentedLoanPlan>
plan_kv_segmented_loan(const KVSegmentedPlacementSnapshot& snapshot,
                       std::uint64_t window_offset, std::uint64_t window_bytes);

} // namespace ninfer::amd::memory
