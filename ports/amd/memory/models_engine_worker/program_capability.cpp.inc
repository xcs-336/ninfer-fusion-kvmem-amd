// All incomplete State ownership operations are out of line, including default ctor.
struct Program::StartupControllerCapability::State {
    detail::ProgramImpl* origin=nullptr; // identity comparison only; never dereferenced by destructor
    amd::memory::KVControllerHandoffToken core;
};
Program::StartupControllerCapability::StartupControllerCapability() noexcept=default;
Program::StartupControllerCapability::~StartupControllerCapability()=default;
Program::StartupControllerCapability::StartupControllerCapability(StartupControllerCapability&&) noexcept=default;
Program::StartupControllerCapability& Program::StartupControllerCapability::operator=(StartupControllerCapability&&) noexcept=default;
Program::StartupControllerCapability::operator bool() const noexcept {
    return state_ && static_cast<bool>(state_->core);
}
Program::StartupControllerCapability Program::release_startup_controller() {
    StartupControllerCapability result;
    if(!impl_->segmented_native_pool)return result; // original Program: no allocation or HIP query
    auto state=std::make_unique<StartupControllerCapability::State>();
    state->origin=impl_.get();
    const auto streams=impl_->segmented_controller_streams();
    state->core=impl_->segmented_native_pool->release_controller_for_worker(streams);
    result.state_=std::move(state);
    return result;
}
void Program::accept_startup_controller(StartupControllerCapability&& capability) {
    if(!impl_->segmented_native_pool) {
        if(capability.state_)throw std::logic_error("ordinary Program rejects a foreign controller capability");
        return;
    }
    if(!capability.state_ || capability.state_->origin!=impl_.get() || !capability)
        throw std::logic_error("Program startup capability is empty, consumed or belongs to another Program");
    if(impl_->segmented_startup_test==SegmentedWorkerStartupTest::BeforeConsume)
        throw std::runtime_error("segmented fixture startup failure before consume");
    const auto streams=impl_->segmented_controller_streams();
    impl_->segmented_native_pool->consume_controller_for_worker(std::move(capability.state_->core),streams);
    capability.state_.reset();
    // Executed on the bound worker, never the owner/test thread. Only this Program
    // can clear its TLS slot; callback cannot allocate, throw, launch or synchronize.
    amd::models_engine_worker::install_graph_observer(impl_.get(),
        [](void* context,std::uintptr_t exec,std::uint64_t generation,std::uint64_t revision) noexcept {
            static_cast<detail::ProgramImpl*>(context)->record_segmented_graph(exec,generation,revision);
        });
}
void Program::retire_worker_controller() {
    if(!impl_->segmented_native_pool)return;
    // worker_loop has ended: there can be no later legitimate host GraphLaunch.
    // Clear before even forming the inventory, so bad_alloc/precondition errors
    // cannot leave a callback installed through the worker's exception exit.
    amd::models_engine_worker::clear_graph_observer(impl_.get());
    const auto streams=impl_->segmented_controller_streams();
    impl_->segmented_native_pool->retire_controller_for_teardown(streams);
    segmented_worker_counters::checked_retires.fetch_add(1);
}
void Program::set_segmented_startup_test(SegmentedWorkerStartupTest test) {
    if(!impl_->segmented_native_pool)throw std::logic_error("ordinary Program has no segmented startup-test control");
    auto controller_freeze=impl_->segmented_native_pool->freeze_allocator();
    if(
       impl_->segmented_native_pool->controller_status().handoff_generation!=0 ||
       (test!=SegmentedWorkerStartupTest::None && test!=SegmentedWorkerStartupTest::BeforeConsume &&
        test!=SegmentedWorkerStartupTest::AfterConsume))
        throw std::logic_error("startup test control is legal only before candidate handoff");
    impl_->segmented_startup_test=test;
}
void Program::worker_startup_ready_test() {
    if(impl_->segmented_native_pool && impl_->segmented_startup_test==SegmentedWorkerStartupTest::AfterConsume)
        throw std::runtime_error("segmented fixture startup failure after consume");
}
Program::SegmentedStorageAccounting Program::segmented_storage_accounting() const {
    if(!impl_->segmented_native_pool)return {};
    SegmentedStorageAccounting out;
    {std::lock_guard lock(impl_->segmented_graph_mutex);out=impl_->segmented_graph_observation;}
    const auto f=impl_->segmented_native_pool->storage().footprint();
    const auto controller=impl_->segmented_native_pool->controller_status();
    out.enabled=true;out.payload_device_bytes=static_cast<std::size_t>(f.payload_device_bytes);
    out.pointer_table_device_bytes=static_cast<std::size_t>(f.pointer_table_device_bytes);
    out.pinned_pointer_shadow_bytes=static_cast<std::size_t>(f.pinned_pointer_shadow_bytes);
    out.slot_maps_host_bytes=static_cast<std::size_t>(f.current_and_next_slot_maps_host_bytes);
    out.controller_disabled=controller.disabled;
    out.controller_phase=static_cast<std::uint8_t>(controller.phase);
    out.handoff_generation=controller.handoff_generation;
    out.cold_device_syncs=controller.completed_cold_device_syncs;
    return out;
}
