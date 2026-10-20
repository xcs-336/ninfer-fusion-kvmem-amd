#pragma once

#include "core/arena.h"
#include "core/device.h"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <memory>

namespace ninfer {

// Native HIP weight borrowing without virtual-memory aliases. One device allocation backs the
// stable weight arena. An exclusive transaction lends its actual chunk-aligned suffix in place;
// close restores just those weight bytes from a pinned host mirror. There are no device
// allocations, remaps, or alias mappings in capture_window_mirror(), evict(), or close().
//
// Weight/home pointers never move. The leased pointer is arena_reserved - leased.bytes: it is
// stable for a particular rounded extent, and deliberately changes when the rounded extent does.
// Callers must bind workspace from this transaction's leased() span, as VisionOverlaySession does.
// A graph capturing a leased workspace can be replayed only inside a live transaction of that same
// extent. A graph reading weights can be replayed only after the exclusive transaction closes.
//
// evict() drains the two owning DeviceContext streams before lending the suffix. No other stream
// may access arena weights while the transaction is open. All writers of the leased span on other
// streams must be drained/ordered before close(); work on the supplied stream is ordered naturally
// before the restore copy. The pool must outlive its transactions and outstanding GPU work.
class EvictableWeightPool {
public:
    static constexpr std::size_t kChunkBytes = 16ULL * 1024ULL * 1024ULL;

    struct Config {
        std::size_t arena_bytes = 0;
        std::size_t evictable_tail_bytes = 0;
    };

    struct TransactionStats {
        double evict_seconds = 0.0;
        double restore_seconds = 0.0;
        // Existing caller/report field: bytes borrowed, although this backend never remaps.
        std::size_t mapped_bytes = 0;
    };

    class Transaction {
    public:
        Transaction() noexcept = default;
        ~Transaction();

        Transaction(const Transaction&) = delete;
        Transaction& operator=(const Transaction&) = delete;
        Transaction(Transaction&& other) noexcept;
        Transaction& operator=(Transaction&& other) noexcept;

        [[nodiscard]] bool open() const noexcept { return pool_ != nullptr; }
        [[nodiscard]] DeviceSpan leased() const noexcept { return leased_; }
        [[nodiscard]] const TransactionStats& stats() const noexcept { return stats_; }

        // Idempotent and nonthrowing. A failed restore poisons the pool and forbids another loan.
        void close() noexcept;

    private:
        friend class EvictableWeightPool;

        Transaction(EvictableWeightPool& pool, DeviceSpan leased, hipStream_t stream,
                    double evict_seconds) noexcept;

        EvictableWeightPool* pool_ = nullptr;
        DeviceSpan leased_{};
        hipStream_t stream_ = nullptr;
        TransactionStats stats_{};
    };

    // Requires a usable HIP device, not VMM support.
    [[nodiscard]] static bool supported(const DeviceContext& device);

    EvictableWeightPool(DeviceContext& device, const Config& config);
    ~EvictableWeightPool();

    EvictableWeightPool(const EvictableWeightPool&) = delete;
    EvictableWeightPool& operator=(const EvictableWeightPool&) = delete;

    [[nodiscard]] DeviceSpan arena() const noexcept;
    [[nodiscard]] std::size_t evictable_tail_bytes() const noexcept;
    [[nodiscard]] std::size_t window_capacity_bytes() const noexcept;
    [[nodiscard]] std::size_t mirror_bytes() const noexcept;
    [[nodiscard]] bool mirror_captured() const noexcept;
    [[nodiscard]] bool transaction_open() const noexcept;
    [[nodiscard]] bool poisoned() const noexcept;

    // Called once after upload. Pins precisely the weight bytes that the maximum rounded suffix
    // can dirty. Arena padding is device backing but does not hold weights or consume mirror RAM.
    void capture_window_mirror(std::size_t window_capacity_bytes, hipStream_t stream);

    // Lends exactly ceil(bytes / kChunkBytes) chunks at the arena END. Rejects zero, oversized,
    // nested, pre-mirror and poisoned requests before changing transaction state.
    [[nodiscard]] Transaction evict(std::size_t bytes, hipStream_t stream);

private:
    struct Impl;
    void restore(Transaction& transaction) noexcept;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
