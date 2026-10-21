#pragma once

// Experimental one-success startup->worker capability. No HIP/API calls here.
#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>

namespace ninfer::amd::memory {
class KVSegmentedPoolOwner;

enum class KVControllerPhase : std::uint8_t {
    Startup, Quiescing, Released, Worker, Retired, Failed
};

struct KVControllerStatus {
    KVControllerPhase phase = KVControllerPhase::Startup;
    bool disabled = false;
    std::uint64_t mutation_epoch = 0, handoff_generation = 0;
    std::uint32_t completed_cold_device_syncs = 0;
};

// Shared allocation identity prevents an old capability from matching a different
// factory constructed at a reused address. It never contains a Core/payload pointer.
struct KVControllerIdentity { std::atomic<bool> alive{true}; };

class KVControllerHandoffToken {
public:
    KVControllerHandoffToken() noexcept = default;
    ~KVControllerHandoffToken() = default; // Loss leaves the released owner DISABLED.
    KVControllerHandoffToken(const KVControllerHandoffToken&) = delete;
    KVControllerHandoffToken& operator=(const KVControllerHandoffToken&) = delete;
    KVControllerHandoffToken(KVControllerHandoffToken&& other) noexcept
        :identity_(std::move(other.identity_)),generation_(std::exchange(other.generation_,0)),
         device_(std::exchange(other.device_,-1)) {}
    KVControllerHandoffToken& operator=(KVControllerHandoffToken&& other) noexcept {
        if(this!=&other) {
            // Dropping an older capability does not silently reactivate its owner.
            identity_=std::move(other.identity_);
            generation_=std::exchange(other.generation_,0);
            device_=std::exchange(other.device_,-1);
        }
        return *this;
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return identity_ && identity_->alive.load(std::memory_order_acquire) && generation_!=0;
    }
private:
    friend class KVSegmentedPoolOwner;
    KVControllerHandoffToken(std::shared_ptr<const KVControllerIdentity> identity,
                             std::uint64_t generation,int device) noexcept
        :identity_(std::move(identity)),generation_(generation),device_(device) {}
    void consumed() noexcept {identity_.reset();generation_=0;device_=-1;}
    std::shared_ptr<const KVControllerIdentity> identity_;
    std::uint64_t generation_=0;
    int device_=-1;
};
} // namespace ninfer::amd::memory
