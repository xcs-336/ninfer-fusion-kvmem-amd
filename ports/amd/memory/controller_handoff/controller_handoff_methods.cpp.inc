// Included into the NEW cloned owner TU within namespace ninfer::amd::memory.
namespace controller_handoff_detail {
void require(bool value,const char* message) {
    if(!value)throw std::logic_error(message);
}
void checked(hipError_t status,const char* where) {
    if(status!=hipSuccess)throw std::runtime_error(std::string(where)+": "+hipGetErrorString(status));
}
void device_and_inventory(int expected,std::span<const hipStream_t> all_streams) {
    require(!all_streams.empty() && all_streams.size()<=8,
            "controller handoff needs the complete explicit Program stream inventory (1..8)");
    int actual=-1;checked(hipGetDevice(&actual),"handoff query current device");
    require(actual==expected,"controller must explicitly bind the pool's actual HIP device");
    for(std::size_t i=0;i<all_streams.size();++i) {
        for(std::size_t j=0;j<i;++j)
            require(all_streams[i]!=all_streams[j],"controller stream inventory contains duplicates");
        hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
        checked(hipStreamIsCapturing(all_streams[i],&capture),"handoff query capture state");
        require(capture==hipStreamCaptureStatusNone,"controller handoff cannot cross an active capture");
    }
}
void next_epoch(KVSegmentedPoolControl& control) {
    require(control.mutation_epoch!=std::numeric_limits<std::uint64_t>::max(),
            "controller handoff cannot advance exhausted allocator epoch");
    ++control.mutation_epoch;
}
} // namespace controller_handoff_detail

KVControllerStatus KVSegmentedPoolOwner::controller_status() const {
    using namespace controller_handoff_detail;
    require(pool_!=nullptr,"unbound controller has no status");
    auto& c=pool_->segmented_native_;std::lock_guard lock(c.mutex);
    return {c.controller_phase,c.controller_disabled,c.mutation_epoch,c.handoff_generation,
            c.completed_cold_device_syncs};
}

KVControllerHandoffToken KVSegmentedPoolOwner::release_controller_for_worker(
    std::span<const hipStream_t> all_program_streams) {
    using namespace controller_handoff_detail;
    require(pool_!=nullptr,"unbound controller cannot be handed off");
    auto& c=pool_->segmented_native_;std::unique_lock lock(c.mutex);
    // All decidable preconditions precede disabling a legitimate controller.
    // A wrong thread, already-used transfer or open grant cannot revoke its rights.
    c.require_submission_thread();
    require(c.owner==this && c.storage && c.controller_phase==KVControllerPhase::Startup &&
            c.handoff_generation==0 && !c.controller_disabled,"only the startup controller may release once");
    require(c.active_controller_freezes==0,"release cannot cross a live allocator Freeze capability");
    require(active_grant_==nullptr && pool_->lent_pages_==0 && !c.storage->lease_open() &&
            !c.storage->poisoned(),"controller release requires no grant, open loan or poison");
    device_and_inventory(config_.device,all_program_streams);
    next_epoch(c);
    c.controller_disabled=true;c.controller_phase=KVControllerPhase::Quiescing;
    c.handoff_generation=1;
    try {
        // This cold one-time boundary covers prior work on THIS current device.
        // It is deliberately not used in view/transfer/loan or decode hot paths.
        checked(hipDeviceSynchronize(),"startup controller quiescent device synchronization");
        ++c.completed_cold_device_syncs;
        c.controller_phase=KVControllerPhase::Released;
        return KVControllerHandoffToken(controller_identity_,c.handoff_generation,config_.device);
    }catch(...) {
        c.controller_phase=KVControllerPhase::Failed;
        // No token, no rebind API, no automatic recovery; the old owner stays disabled.
        throw;
    }
}

void KVSegmentedPoolOwner::consume_controller_for_worker(KVControllerHandoffToken&& token,
    std::span<const hipStream_t> all_worker_streams) {
    using namespace controller_handoff_detail;
    require(pool_!=nullptr,"unbound worker controller");
    auto& c=pool_->segmented_native_;std::unique_lock lock(c.mutex);
    require(c.owner==this && c.storage && c.controller_phase==KVControllerPhase::Released &&
            c.controller_disabled,"controller is not awaiting its one startup token");
    require(token && token.identity_.get()==controller_identity_.get() &&
            token.generation_==c.handoff_generation && token.device_==config_.device,
            "controller token has wrong owner/lifetime/generation/device or was consumed");
    require(c.submission_thread!=std::this_thread::get_id(),"startup controller cannot consume its own transfer");
    require(c.active_controller_freezes==0,"accept cannot cross a live allocator Freeze capability");
    require(active_grant_==nullptr && pool_->lent_pages_==0 && !c.storage->lease_open() &&
            !c.storage->poisoned(),"worker cannot accept a grant/loan/poisoned controller");
    device_and_inventory(config_.device,all_worker_streams);
    // CPU-only reserve/materialize/RAII release is still mutex-serialized while
    // released. It may legitimately change mutation_epoch. That epoch invalidates
    // allocator snapshots, NOT this identity/generation capability.
    next_epoch(c);
    c.submission_thread=std::this_thread::get_id();
    c.controller_phase=KVControllerPhase::Worker;c.controller_disabled=false;
    token.consumed(); // noexcept; no failure point after enabling the new thread.
}

void KVSegmentedPoolOwner::retire_controller_for_teardown(
    std::span<const hipStream_t> all_worker_streams) {
    using namespace controller_handoff_detail;
    require(pool_!=nullptr,"unbound controller cannot retire");
    auto& c=pool_->segmented_native_;std::unique_lock lock(c.mutex);
    c.require_submission_thread();
    require(c.owner==this && c.storage && c.controller_phase==KVControllerPhase::Worker &&
            !c.controller_disabled,"only the accepted worker may retire");
    require(c.active_controller_freezes==0,"retirement cannot cross a live allocator Freeze capability");
    require(active_grant_==nullptr && pool_->lent_pages_==0 && !c.storage->lease_open() &&
            !c.storage->poisoned(),"retirement needs closed loans/grants and usable storage");
    device_and_inventory(config_.device,all_worker_streams);
    next_epoch(c);c.controller_disabled=true;c.controller_phase=KVControllerPhase::Quiescing;
    try {
        checked(hipDeviceSynchronize(),"worker controller final quiescent device synchronization");
        ++c.completed_cold_device_syncs;c.controller_phase=KVControllerPhase::Retired;
    }catch(...) {c.controller_phase=KVControllerPhase::Failed;throw;}
    // This is permanent disable, not a transfer back to the parent. Join follows.
}
