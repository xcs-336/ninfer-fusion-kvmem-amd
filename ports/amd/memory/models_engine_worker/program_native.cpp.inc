// Inside detail's original anonymous namespace, before the first resident allocation.
std::unique_ptr<amd::memory::KVSegmentedNativePool> make_segmented_program_storage(
    const execution::Parameters& parameters,const SequencePlanImpl& plan,DeviceContext& device) {
    if(!plan.segmented_text_storage)return nullptr;
    validate_segmented_sequence_plan(parameters,plan,device);
    const auto& layout=plan.persistent.decoder.text_kv.pages;
    const auto& cache=plan.persistent.decoder.text_kv;
    const auto planes=layout.planes.size();
    const auto groups=layout.spec.page_group_count;
    const auto slot=static_cast<std::uint64_t>(256)*64*plan.persistent.decoder.text_kv.kv_heads*2;
    const auto payload=plan.persistent.segmented_payload_bytes;
    const auto table=plan.persistent.segmented_pointer_table_bytes;
    if(cache.layers==0 || cache.layers!=parameters.model.config().text.full_attention_layers ||
       cache.head_dim!=256 || (cache.kv_heads!=2 && cache.kv_heads!=4) ||
       cache.max_context!=plan.capacity || cache.storage!=KvCacheStorage::BFloat16 ||
       groups!=plan.main_page_groups || layout.spec.allow_logical_reservation ||
       layout.spec.geometry.page_tokens!=64 || layout.spec.geometry.device_plane_order!=PagedKVPlaneOrder::PageMajor ||
       planes!=2ULL*cache.layers || layout.spec.geometry.planes.size()!=planes ||
       cache.execution_tables.size()!=1 || cache.execution_tables.front().rank!=0 ||
       cache.execution_tables.front().spec.table_rows!=1 ||
       planes==0 || planes>std::numeric_limits<std::uint32_t>::max() || groups==0 ||
       slot>std::numeric_limits<std::size_t>::max()/groups/planes ||
       payload!=slot*groups*planes || table!=8ULL*groups*planes ||
       payload>std::numeric_limits<std::size_t>::max()-table ||
       payload+table>std::numeric_limits<std::size_t>::max()-plan.persistent.bytes)
        throw std::invalid_argument("segmented Program resident inventory is invalid or overflows");
    // Validate the entire fresh homogeneous layout before the factory can allocate.
    for(std::size_t p=0;p<planes;++p) {
        const auto& plane=layout.planes[p];
        if(plane.rank!=0 || layout.spec.geometry.planes[p]!=plane.geometry ||
           plane.storage.dtype!=DType::BF16 || plane.geometry!=KVPlaneGeometry{DType::BF16,256,
                plan.persistent.decoder.text_kv.kv_heads,256} ||
           plane.storage.region.offset!=p*groups*slot || plane.storage.region.bytes!=groups*slot)
            throw std::invalid_argument("segmented Program payload is not its exact fresh PageMajor layout");
    }
    // No Vision borrower is exposed in this finite candidate. A one-cell legal
    // window configures the real backend; it is not a claimed Engine memory saving.
    amd::memory::KVSegmentedStorage::Config config{.device=plan.device,
        .planes=static_cast<std::uint32_t>(planes),.page_groups=groups,.slot_bytes=slot,
        .window_offset=0,.max_window_bytes=slot,.borrower_alignment=256,.max_user_streams=4};
    auto native=std::make_unique<amd::memory::KVSegmentedNativePool>(config,layout);
    const auto f=native->storage().footprint();
    if(f.payload_device_bytes!=payload || f.pointer_table_device_bytes!=table)
        throw std::logic_error("native factory disagrees with the frozen Program resident bill");
    return native;
}
std::size_t segmented_program_device_bytes(amd::memory::KVSegmentedNativePool* native) noexcept {
    if(!native)return 0;
    const auto f=native->storage().footprint();
    // make_segmented_program_storage checked representability and the full sum.
    return static_cast<std::size_t>(f.payload_device_bytes+f.pointer_table_device_bytes);
}
