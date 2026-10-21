// Added methods only. Original extracted methods are transformed reversibly.
namespace {
using SegmentedStorage = ::ninfer::amd::memory::KVSegmentedStorage;
using BrokerClock = std::chrono::steady_clock;
double broker_seconds(BrokerClock::time_point start) noexcept {
    return std::chrono::duration<double>(BrokerClock::now()-start).count();
}
std::size_t broker_inventory(std::span<const hipStream_t> source,
                            std::array<hipStream_t,4>& destination) {
    if(source.empty()||source.size()>destination.size())
        throw std::invalid_argument("segmented Vision requires a bounded complete stream inventory");
    std::size_t count=0;
    for(const auto stream:source) {
        if(std::find(destination.begin(),destination.begin()+count,stream)==destination.begin()+count)
            destination[count++]=stream;
    }
    return count;
}
bool broker_contains(std::span<const hipStream_t> inventory,hipStream_t stream) noexcept {
    return std::find(inventory.begin(),inventory.end(),stream)!=inventory.end();
}
} // namespace

void VisionResidencyBroker::enable_segmented_tier(
    SegmentedStorage& storage,std::size_t maximum_request_bytes,
    std::span<const hipStream_t> all_prior_users,
    std::span<const hipStream_t> all_borrower_users,
    std::function<bool()> can_lend,std::function<void()> on_change) {
    require_closed();
    if(kv_arena_||kv_pages_||segmented_storage_)
        throw std::logic_error("Vision KV tier was already selected");
    if(device_.size()!=1||maximum_request_bytes==0||
       maximum_request_bytes>storage.footprint().payload_device_bytes)
        throw std::invalid_argument("segmented Vision requires one device and a nonempty configured window bound");
    std::array<hipStream_t,4> prior{},borrowers{};
    const auto prior_count=broker_inventory(all_prior_users,prior);
    const auto borrower_count=broker_inventory(all_borrower_users,borrowers);
    const std::span<const hipStream_t> prior_view(prior.data(),prior_count);
    const std::span<const hipStream_t> borrower_view(borrowers.data(),borrower_count);
    // These are the actual Context users plus default-stream Core operations.
    // Any extra producer/reader must also fit the declared complete inventory.
    const std::array<hipStream_t,4> required{
        device_.stream,device_.transfer_stream,device_.vision_stream,nullptr};
    for(auto stream:required) {
        if(!broker_contains(prior_view,stream)||!broker_contains(borrower_view,stream))
            throw std::invalid_argument("segmented Vision inventory omits a Context/default stream");
    }
    // Uses the actual owner/controller guard. This does not prove an external
    // scheduler's completeness or the caller's maximum <= storage fixed window.
    (void)storage.device_plane_page_bases(0);
    segmented_storage_=&storage;segmented_max_request_=maximum_request_bytes;
    segmented_prior_=prior;segmented_prior_count_=prior_count;
    segmented_borrowers_=borrowers;segmented_borrower_count_=borrower_count;
    can_lend_=std::move(can_lend);on_change_=std::move(on_change);
}

std::optional<VisionWindow> VisionResidencyBroker::try_acquire_segmented(std::size_t bytes) {
    // require_closed was executed by the real public try_acquire_kv entry.
    if((can_lend_&&!can_lend_())||bytes>segmented_max_request_)return std::nullopt;
    const auto start=BrokerClock::now();
    auto loan=segmented_storage_->try_begin_loan(bytes,
        std::span<const hipStream_t>(segmented_prior_.data(),segmented_prior_count_));
    if(!loan)return std::nullopt; // ONLY actual typed planner capacity refusal here.
    VisionWindow window;
    window.broker_=this;window.tier_=VisionWindow::Tier::KvSegmented;
    window.segmented_=std::move(*loan); // unique capability, never legacy runs.
    window.segmented_users_=segmented_borrowers_;
    window.segmented_user_count_=segmented_borrower_count_;
    window.segmented_bytes_=static_cast<std::size_t>(window.segmented_.leased().bytes);
    window.segmented_open_seconds_=broker_seconds(start);
    try {
        if(on_change_)on_change_(); // original Program revision contract; no fake mutation.
    }catch(...) {
        remember_segmented_failure("segmented lend succeeded but Program revision callback failed",
                                    std::current_exception());
        throw; // local window still runs the real checked return while unwinding.
    }
    return window;
}

void VisionWindow::wait_ready(hipStream_t stream) const {
    if(!broker_||close_failed_)throw std::logic_error("closed/failed Vision window cannot submit users");
    if(tier_!=Tier::KvSegmented)
        throw std::logic_error("segmented wait_ready is not a legacy WeightTail completion proof");
    if(!broker_contains(std::span<const hipStream_t>(segmented_users_.data(),segmented_user_count_),stream))
        throw std::invalid_argument("Vision user stream is absent from the immutable completion inventory");
    segmented_.wait_ready(stream);
}

void VisionResidencyBroker::remember_segmented_failure(const char* reason,std::exception_ptr error) noexcept {
    if(!sticky_failure_) {
        sticky_failure_=true;failure_reason_=reason;failure_exception_=std::move(error);
    }
}

bool VisionWindow::close_checked() noexcept {
    if(close_failed_)return false; // no retry of close/events/revision after any failure.
    if(!broker_)return true;
    VisionResidencyBroker* const broker=broker_;
    if(tier_!=Tier::KvSegmented) {
        // This is exactly the legacy close body, not a newly proved checked
        // restore. Original Session/caller must finish every WeightTail user.
        close_legacy();
        if(broker->poisoned()) {
            close_failed_=true;broker_=broker;
            broker->remember_segmented_failure("legacy WeightTail/pool reports poison after best-effort close");
            return false;
        }
        return true;
    }
    if(!runs_.empty()) {
        close_failed_=true;
        broker->remember_segmented_failure("segmented Vision must not own legacy raw return runs");
        return false; // ticket destruction poisons instead of returning guessed IDs.
    }
    const auto start=BrokerClock::now();
    const bool returned=segmented_.close_after_streams(
        std::span<const hipStream_t>(segmented_users_.data(),segmented_user_count_));
    segmented_close_seconds_=broker_seconds(start);
    if(!returned) {
        close_failed_=true;
        broker->remember_segmented_failure("segmented Vision completion/return failed; grant remains excluded");
        return false; // broker_ stays nonnull, storage keeps real lent/poison.
    }
    try {
        if(broker->on_change_)broker->on_change_(); // only AFTER the real return succeeded.
    }catch(...) {
        close_failed_=true;
        broker->remember_segmented_failure("segmented capacity returned but Program revision callback failed",
                                            std::current_exception());
        return false; // capacity is returned; do not falsely claim it remains lent.
    }
    broker_=nullptr;
    return true;
}
