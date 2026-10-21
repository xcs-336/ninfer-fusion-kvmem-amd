// Spliced into the NEW decoder_state.cpp, after its original planner functions.
// Both ordinary and tagged constructors use the actual original table pool.
namespace {
void segmented_storage_require(bool value,const char* message) {
    if(!value)throw std::invalid_argument(message);
}
void require_segmented_spec(const DecoderStateSpec& spec) {
    segmented_storage_require(spec.full_attention_layers>0 && spec.capacity>0 && spec.capacity<=512 &&
        spec.attention_head_dim==256 && (spec.kv_heads==2||spec.kv_heads==4) &&
        spec.kv_storage==KvCacheStorage::BFloat16 && spec.kv_table_rows==1,
        "segmented Models storage requires dense B1 BF16 D256 KV2/4 context1..512");
    segmented_storage_require(!spec.enable_mtp && spec.mtp_layers==0 && spec.mtp_physical_page_groups==0,
        "segmented Models storage does not admit MTP");
    segmented_storage_require(spec.text_physical_page_groups>=page_count(spec.capacity),
        "segmented Models storage does not admit a ring or selected history");
    segmented_storage_require(spec.text_layer_rank.empty() ||
        (spec.text_layer_rank.size()==spec.full_attention_layers &&
         std::all_of(spec.text_layer_rank.begin(),spec.text_layer_rank.end(),[](auto rank){return rank==0;})),
        "segmented Models storage is single rank");
}
const DeviceKVPagePoolLayout& require_owned_layout(const PagedKVCacheLayout& layout) {
    segmented_storage_require(!layout.external_segmented_payload,
        "separate segmented payload layout requires its explicit native-pool constructor");
    return layout.pages;
}
const PagedKVCacheLayout& require_no_mtp(const DecoderStateLayout& layout) {
    segmented_storage_require(!layout.mtp_kv,"segmented DecoderState cannot contain MTP storage");
    return layout.text_kv;
}
DeviceKVPagePool* require_segmented_pool(amd::memory::KVSegmentedNativePool& native,
    DeviceSpan metadata,const PagedKVCacheLayout& layout) {
    segmented_storage_require(layout.external_segmented_payload && layout.layers>0 &&
        layout.max_context>0 && layout.max_context<=512 && layout.head_dim==256 &&
        (layout.kv_heads==2||layout.kv_heads==4) && layout.storage==KvCacheStorage::BFloat16,
        "tagged Models storage requires its separately planned BF16 layout");
    const auto& planned=layout.pages;
    const auto& geometry=planned.spec.geometry;
    const auto logical=page_count(layout.max_context);
    segmented_storage_require(!planned.spec.allow_logical_reservation &&
        planned.spec.page_group_count>=logical &&
        planned.spec.page_group_count<=static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) &&
        geometry.page_tokens==64 && geometry.device_plane_order==PagedKVPlaneOrder::PageMajor &&
        static_cast<std::uint64_t>(layout.layers)*2==geometry.planes.size() &&
        planned.planes.size()==geometry.planes.size(),
        "tagged Models storage rejects scales, ring, HeadMajor and incomplete planes");
    auto& pool=native.page_pool();
    auto freeze=native.freeze_allocator(); // original controller/mutex, startup binding only
    segmented_storage_require(!native.storage().poisoned() && !native.storage().lease_open() && pool.lent_pages()==0,
        "Models storage cannot bind during a native grant/loan or after poison");
    segmented_storage_require(pool.rank_count()==1 && pool.capacity_pages()==planned.spec.page_group_count &&
        pool.usable_pages()>=logical && pool.geometry()==geometry && pool.plane_count()==geometry.planes.size(),
        "tagged Models storage native pool does not match the exact planned geometry");
    segmented_storage_require(layout.execution_tables.size()==1 && metadata.data!=nullptr,
        "tagged Models storage needs one real metadata backing/table copy");
    const auto& table=layout.execution_tables.front();
    segmented_storage_require(table.rank==0 && table.spec.table_rows==1 &&
        table.spec.logical_page_capacity==logical && table.block_tables.dtype==DType::I32 &&
        table.block_tables.shape==std::array<std::int32_t,4>{static_cast<std::int32_t>(logical),1,1,1},
        "tagged Models storage rejects batch/mismatched execution-table layouts");
    (void)table.block_tables.bind(metadata); // original bounds/alignment checking, no allocation
    const auto metadata_begin=reinterpret_cast<std::uintptr_t>(metadata.data);
    segmented_storage_require(metadata.bytes<=std::numeric_limits<std::uintptr_t>::max()-metadata_begin,
        "segmented metadata backing address overflow");
    const auto metadata_end=metadata_begin+metadata.bytes;
    const auto slot_bytes=static_cast<std::uint64_t>(256)*64*layout.kv_heads*2;
    std::uint64_t payload_bytes=0;
    for(std::size_t p=0;p<planned.planes.size();++p) {
        const auto& plane=planned.planes[p];
        segmented_storage_require(geometry.planes[p]==KVPlaneGeometry{DType::BF16,256,layout.kv_heads,256} &&
            plane.geometry==geometry.planes[p] && plane.rank==0 && pool.plane_rank(p)==0 &&
            plane.storage.dtype==DType::BF16 &&
            plane.storage.shape==std::array<std::int32_t,4>{256,64,layout.kv_heads,
                static_cast<std::int32_t>(planned.spec.page_group_count)} &&
            plane.storage.region.offset==payload_bytes &&
            plane.storage.region.bytes==slot_bytes*planned.spec.page_group_count,
            "segmented payload must be the exact fresh zero-based homogeneous layout");
        segmented_storage_require(plane.storage.region.bytes<=std::numeric_limits<std::uint64_t>::max()-payload_bytes,
            "segmented payload inventory overflow");
        payload_bytes+=plane.storage.region.bytes;
        const auto view=pool.plane_view(p); // actual controller check + stable device table
        segmented_storage_require(view.page_bases!=nullptr && view.storage.dtype==DType::BF16 &&
            view.storage.ne[0]==256 && view.storage.ne[1]==64 && view.storage.ne[2]==layout.kv_heads &&
            view.storage.ne[3]==static_cast<std::int32_t>(planned.spec.page_group_count) &&
            view.storage.nb[0]==2 && view.storage.nb[1]==512 &&
            view.storage.nb[2]==256*64*2 && view.storage.nb[3]==static_cast<std::int64_t>(slot_bytes),
            "native pool must expose the actual PageMajor BF16 view/table ABI");
        const auto begin=reinterpret_cast<std::uintptr_t>(view.storage.data);
        segmented_storage_require(begin!=0 && plane.storage.region.bytes<=std::numeric_limits<std::uintptr_t>::max()-begin,
            "segmented payload address overflow");
        const auto end=begin+plane.storage.region.bytes;
        segmented_storage_require(metadata_end<=begin || end<=metadata_begin,
            "ordinary metadata backing must not alias the native KV payload");
        const auto table_begin=reinterpret_cast<std::uintptr_t>(view.page_bases);
        const auto table_bytes=8ULL*planned.spec.page_group_count;
        segmented_storage_require(table_bytes<=std::numeric_limits<std::uintptr_t>::max()-table_begin,
            "segmented pointer table address overflow");
        segmented_storage_require(metadata_end<=table_begin || table_begin+table_bytes<=metadata_begin,
            "ordinary metadata backing must not alias the native device page table");
    }
    const auto footprint=native.storage().footprint();
    segmented_storage_require(footprint.payload_device_bytes==payload_bytes &&
        footprint.pointer_table_device_bytes==8ULL*planned.planes.size()*planned.spec.page_group_count,
        "native payload/table inventory differs from the Models storage plan");
    return &pool; // no new DeviceKVPagePool, DeviceArena or payload allocation
}
} // namespace

DecoderStateLayout plan_segmented_decoder_state(LayoutBuilder& fresh_payload,
    LayoutBuilder& ordinary_metadata,const DecoderStateSpec& spec) {
    require_segmented_spec(spec);
    segmented_storage_require(&fresh_payload!=&ordinary_metadata && fresh_payload.finish()==0,
        "segmented KV payload requires a separate fresh zero-based builder");
    LayoutBuilder* const metadata[]={&ordinary_metadata};
    DecoderStateLayout result;
    result.text_kv=plan_cache(metadata,spec.text_layer_rank,spec.full_attention_layers,spec.capacity,
        spec.kv_heads,spec.attention_head_dim,spec.kv_storage,spec.kv_table_rows,
        spec.text_physical_page_groups,&fresh_payload);
    return result; // MTP absent; all other Program state still requires explicit replanning
}

PagedKVCache::PagedKVCache(SegmentedDecoderStorageTag,DeviceSpan metadata,
    const PagedKVCacheLayout& layout,amd::memory::KVSegmentedNativePool& native)
    :owned_pages_(),pages_(require_segmented_pool(native,metadata,layout)),
     execution_tables_(metadata,layout.execution_tables.front(),*pages_),layers_(layout.layers),
     max_context_(layout.max_context),kv_heads_(layout.kv_heads),head_dim_(layout.head_dim),
     storage_(layout.storage),external_segmented_pool_(true) {}

DecoderState::DecoderState(SegmentedDecoderStorageTag tag,DeviceSpan metadata,
    const DecoderStateLayout& layout,amd::memory::KVSegmentedNativePool& native)
    :text_kv(tag,metadata,require_no_mtp(layout),native) {}
