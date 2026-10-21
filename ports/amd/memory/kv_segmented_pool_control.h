#pragma once
// SOURCE ONLY. Included ONLY by the isolated experimental Core source view.
// Adds real per-pool exclusion; not a registry or an external ignored mutex.
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace ninfer::amd::memory {
class KVSegmentedPoolOwner;
class KVSegmentedStorage;

struct KVSegmentedPoolControl {
    mutable std::recursive_mutex mutex;
    std::uint64_t mutation_epoch=1;
    KVSegmentedPoolOwner* owner=nullptr;
    KVSegmentedStorage* storage=nullptr;
    std::thread::id submission_thread{};
    void require_submission_thread()const {
        if(owner&&submission_thread!=std::this_thread::get_id())
            throw std::logic_error("segmented GPU submission requires its startup-bound controller thread");
    }
};

// Conservative invalidation: even an unsuccessful attempted mutation advances the
// epoch. Nested original helpers/lease destruction may acquire the same lock.
class KVSegmentedPoolMutation {
public:
    explicit KVSegmentedPoolMutation(KVSegmentedPoolControl& control):lock(control.mutex) {
        if(control.mutation_epoch==std::numeric_limits<std::uint64_t>::max())std::terminate();
        ++control.mutation_epoch;
    }
private:std::unique_lock<std::recursive_mutex> lock;
};
} // namespace ninfer::amd::memory
