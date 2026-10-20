#pragma once

// Experimental physical Core transport/transaction. SOURCE ONLY, not built/qualified.
// No adapter to the existing DeviceKVPagePool is supplied or implied by these interfaces.
#include "kv_segmented_loan_plan.h"
#include <hip/hip_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::amd::memory {

class KVSegmentedAllocatorOwner;

struct KVSegmentedFrozenAllocation {
    const KVSegmentedAllocatorOwner* owner = nullptr;
    std::uint64_t mutation_epoch = 0; // alloc/free/placement/lend/return, NOT capacity only
    std::uint32_t page_groups = 0;
    // Value-owned: a later allocator query cannot silently mutate the earlier snapshot.
    std::vector<std::uint8_t> allocated;
    std::vector<std::uint32_t> free_pages;
    std::uint32_t available_pages = 0;
};

// An opaque, unique allocator-issued capability. Its destructor MUST NOT return IDs:
// physical completion is established separately, and an abandoned loan must stay excluded.
class KVSegmentedLentPages {
public:
    virtual ~KVSegmentedLentPages() = default;
    [[nodiscard]] virtual const KVSegmentedAllocatorOwner& owner() const noexcept = 0;
    [[nodiscard]] virtual std::span<const std::uint32_t> pages() const noexcept = 0;
    KVSegmentedLentPages(const KVSegmentedLentPages&) = delete;
protected:
    KVSegmentedLentPages() = default;
};

// Integration must implement real exclusion of ALL allocator/placement mutations while
// this RAII object lives, and materialize value-owned U8/free-ID arrays counted in its budget.
// A separate mutex which raw DeviceKVPagePool users ignore would NOT satisfy this contract.
class KVSegmentedAllocatorFreeze {
public:
    virtual ~KVSegmentedAllocatorFreeze() = default;
    [[nodiscard]] virtual const KVSegmentedAllocatorOwner& owner() const noexcept = 0;
    [[nodiscard]] virtual KVSegmentedFrozenAllocation snapshot() const = 0;
    // Atomic: failure leaves capacity/ownership unchanged. Must enforce available quota,
    // no existing loan, genuine free IDs and a new owner-issued unique grant/epoch.
    [[nodiscard]] virtual std::unique_ptr<KVSegmentedLentPages>
    lend_selected(std::span<const std::uint32_t> pages) = 0;
    // Verify owner/grant uniqueness/current lent state. Atomic success invalidates the
    // grant in allocator bookkeeping; any failure MUST preserve excluded IDs atomically.
    virtual void return_selected(KVSegmentedLentPages& grant) = 0;
    KVSegmentedAllocatorFreeze(const KVSegmentedAllocatorFreeze&) = delete;
protected:
    KVSegmentedAllocatorFreeze() = default;
};

class KVSegmentedAllocatorOwner {
public:
    virtual ~KVSegmentedAllocatorOwner() = default;
    [[nodiscard]] virtual std::unique_ptr<KVSegmentedAllocatorFreeze> freeze() = 0;
};

struct KVSegmentedSpan { void* data = nullptr; std::uint64_t bytes = 0; };

struct KVSegmentedStorageFootprint {
    // This storage OWNS its payload. Future Program integration must replace/exclude the
    // old KV class allocation; constructing both would double-count/allocate its payload.
    std::uint64_t payload_device_bytes = 0;
    std::uint64_t pointer_table_device_bytes = 0;
    std::uint64_t pinned_pointer_shadow_bytes = 0;
    std::uint64_t current_and_next_slot_maps_host_bytes = 0;
    // Planner temporary/input/opaque allocator-grant storage is separate and must also
    // be charged. Stream/event driver overhead is unmeasured, not asserted to be zero.
    std::uint32_t private_streams = 0, events = 0;
};

// One host control thread (or equivalent external serialization) owns this storage and
// GPU submission boundaries. No new old-map reader/producer may be submitted after the
// recorded prior-user fences until begin_loan returns ready. The allocator RAII protocol
// does not itself freeze an unrelated scheduler or unregistered GPU stream.
// These synchronous begin/wait/close boundaries reject actively capturing streams.
class KVSegmentedStorage {
public:
    struct Config {
        int device = 0;
        std::uint32_t planes = 0, page_groups = 0;
        std::uint64_t slot_bytes = 0;
        std::uint64_t window_offset = 0, max_window_bytes = 0;
        std::uint32_t borrower_alignment = 256;
        std::uint32_t max_user_streams = 4; // actual recorded prior/borrower fences, <=8
    };

    class Transaction {
    public:
        Transaction() noexcept = default;
        ~Transaction(); // An open, unclosed ticket poisons; it NEVER silently returns IDs.
        Transaction(Transaction&&) noexcept;
        Transaction& operator=(Transaction&&) noexcept;
        Transaction(const Transaction&) = delete;
        Transaction& operator=(const Transaction&) = delete;
        [[nodiscard]] bool open() const noexcept { return storage_ != nullptr; }
        [[nodiscard]] KVSegmentedSpan leased() const noexcept { return span_; }
        [[nodiscard]] std::uint64_t copied_live_bytes() const noexcept { return copied_; }
        // Ready was completed before exposure; retain an explicit GPU dependency for users.
        void wait_ready(hipStream_t stream) const;
        // Record OWN events after all supplied stream users, then synchronize every event
        // before returning IDs. Empty/incomplete stream inventory is not a valid close.
        // Raw-span Graph replays must finish in that inventory and may never replay after
        // this ticket returns; a persistent KV table-consumer graph has a different lifetime.
        [[nodiscard]] bool close_after_streams(std::span<const hipStream_t> streams) noexcept;
    private:
        friend class KVSegmentedStorage;
        Transaction(KVSegmentedStorage&,std::uint64_t serial,KVSegmentedSpan,
                    std::uint64_t copied) noexcept;
        void abandon() noexcept;
        KVSegmentedStorage* storage_ = nullptr; // storage/allocator MUST outlive this ticket
        std::uint64_t serial_ = 0, copied_ = 0;
        KVSegmentedSpan span_{};
    };

    // The allocator owner outlives the storage; the storage outlives every ticket/consumer.
    KVSegmentedStorage(const Config&,KVSegmentedAllocatorOwner&);
    ~KVSegmentedStorage();
    KVSegmentedStorage(const KVSegmentedStorage&) = delete;
    KVSegmentedStorage& operator=(const KVSegmentedStorage&) = delete;
    [[nodiscard]] KVSegmentedStorageFootprint footprint() const noexcept;
    [[nodiscard]] bool poisoned() const noexcept;
    [[nodiscard]] bool lease_open() const noexcept;
    [[nodiscard]] std::string_view poison_reason() const noexcept;
    // Geometry is still logical plane-major; each entry is a true device page pointer.
    // This allocation is separate from and NEVER part of the raw loan span.
    [[nodiscard]] const std::uint64_t* device_plane_page_bases(std::uint32_t plane) const;
    // For explicit Core byte upload/tests only; existing affine KV Ops cannot consume it.
    [[nodiscard]] KVSegmentedSpan raw_payload_for_initialization() const;
    // Owner snapshot remains frozen through real copy/table completion. No device-wide
    // synchronize is used in ordinary begin/close. Other streams may continue disjoint work.
    [[nodiscard]] Transaction begin_loan(std::uint64_t requested_bytes,
                                       std::span<const hipStream_t> all_prior_users);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    void wait_ready(std::uint64_t serial,hipStream_t) const;
    bool close(std::uint64_t serial,std::span<const hipStream_t>) noexcept;
    void abandon(std::uint64_t serial) noexcept;
};

} // namespace ninfer::amd::memory
