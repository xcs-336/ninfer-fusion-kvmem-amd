#include "kv_segmented_loan_plan.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::amd::memory {
namespace {

std::uint64_t multiply(std::uint64_t a, std::uint64_t b) {
    if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b)
        throw std::overflow_error("segmented KV class byte extent overflow");
    return a * b;
}

void require(bool value, const char* message) {
    if (!value) throw std::invalid_argument(message);
}

} // namespace

std::optional<KVSegmentedLoanPlan>
plan_kv_segmented_loan(const KVSegmentedPlacementSnapshot& input,
                       std::uint64_t window_offset, std::uint64_t window_bytes) {
    require(input.planes != 0 && input.page_groups != 0 && input.slot_bytes != 0,
            "segmented KV needs a nonempty homogeneous physical class");
    const auto cells64 = multiply(input.planes, input.page_groups);
    require(cells64 <= std::numeric_limits<std::uint32_t>::max(),
            "segmented KV class has too many slots for the explicit index format");
    const auto cells = static_cast<std::uint32_t>(cells64);
    const auto bytes = multiply(cells64, input.slot_bytes);
    require(window_bytes != 0 && window_offset < bytes && window_bytes <= bytes - window_offset,
            "segmented KV window is empty or outside its physical class");
    require(input.slot_of_cell.size() == cells &&
            input.page_allocated.size() == input.page_groups,
            "segmented KV placement snapshot has an inconsistent extent");
    require(input.available_pages <= input.free_pages.size(),
            "segmented KV available capacity exceeds the actual free IDs");

    const auto missing = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> cell_at_slot(cells, missing);
    for (std::uint32_t cell = 0; cell < cells; ++cell) {
        const auto slot = input.slot_of_cell[cell];
        require(slot < cells && cell_at_slot[slot] == missing,
                "segmented KV placement must be a bijection, not an alias");
        cell_at_slot[slot] = cell;
    }
    std::uint64_t allocated = 0;
    for (const auto state : input.page_allocated) {
        require(state <= 1, "segmented KV allocation states must be booleans");
        allocated += state;
    }
    require(allocated + input.free_pages.size() == input.page_groups,
            "segmented KV planning requires a closed pool with every ID accounted for");
    std::uint32_t previous = 0;
    bool first = true;
    for (const auto page : input.free_pages) {
        require(page < input.page_groups && input.page_allocated[page] == 0 &&
                (first || previous < page),
                "segmented KV free IDs must be distinct sorted and genuinely free");
        previous = page;
        first = false;
    }

    const auto first_slot = static_cast<std::uint32_t>(window_offset / input.slot_bytes);
    const auto last_slot = static_cast<std::uint32_t>(
        (window_offset + window_bytes - 1) / input.slot_bytes);
    const std::uint64_t touched = std::uint64_t(last_slot) - first_slot + 1;
    const auto wanted = static_cast<std::uint32_t>(1 + (touched - 1) / input.planes);
    if (wanted > input.available_pages) return std::nullopt;

    KVSegmentedLoanPlan out;
    out.resource_revision = input.resource_revision;
    out.window_offset = window_offset;
    out.window_bytes = window_bytes;
    out.slot_bytes = input.slot_bytes;
    // Highest free IDs match the existing loan preference. Selecting IDs does not mutate
    // allocator capacity or consume reservations; only the future owner can do that.
    out.loan_page_ids.assign(input.free_pages.end() - wanted, input.free_pages.end());
    std::vector<std::uint8_t> is_loan(input.page_groups, 0);
    for (const auto page : out.loan_page_ids) is_loan[page] = 1;
    std::vector<std::uint32_t> outside_donors;
    outside_donors.reserve(static_cast<std::size_t>(input.planes) * wanted);
    for (std::uint32_t plane = 0; plane < input.planes; ++plane) {
        for (const auto page : out.loan_page_ids) {
            const auto cell = plane * input.page_groups + page;
            const auto slot = input.slot_of_cell[cell];
            if (slot < first_slot || slot > last_slot) outside_donors.push_back(cell);
        }
    }
    out.swaps.reserve(static_cast<std::size_t>(touched));
    std::size_t donor_index = 0;
    for (std::uint64_t slot = first_slot; slot <= last_slot; ++slot) {
        const auto owner = cell_at_slot[static_cast<std::size_t>(slot)];
        if (is_loan[owner % input.page_groups]) continue;
        // P*wanted >= touched ensures at least one outside loan cell for every touched
        // non-loan owner. No source or donor is ever reused by another pair.
        if (donor_index >= outside_donors.size())
            throw std::logic_error("segmented KV donor cardinality invariant failed");
        const auto donor = outside_donors[donor_index++];
        const bool live = input.page_allocated[owner % input.page_groups] != 0;
        out.swaps.push_back({owner, donor, static_cast<std::uint32_t>(slot),
                             input.slot_of_cell[donor], live});
        if (live) out.copied_live_bytes += input.slot_bytes; // bounded by class extent
    }
    return out;
}

} // namespace ninfer::amd::memory
