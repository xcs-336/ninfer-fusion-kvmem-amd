// Candidate-only pure planning admission. No device allocation or HIP call.
void require_segmented_engine_scope(const execution::Parameters& parameters,
    std::uint32_t capacity,std::uint32_t concurrency,KvCacheStorage storage,
    const models::LoadOptions& features,const ContextCacheOptions& cache) {
    const auto& text=parameters.model.config().text;
    if(capacity==0||capacity>512||concurrency!=1||parameters.text.rank_count!=1||features.ranks!=1||
       storage!=KvCacheStorage::BFloat16||!text.attention||text.full_attention_layers==0||
       text.attention->head_dim!=256||
       !((text.attention->num_attention_heads==16&&text.attention->num_key_value_heads==2)||
         (text.attention->num_attention_heads==24&&text.attention->num_key_value_heads==4))||
       features.vision||features.speculative!=SpeculativeBackend::None||features.gdn_state_fp16)
        throw std::invalid_argument("segmented planner requires B1 dense512 BF16 D256 H16/KV2 or H24/KV4, FP32 GDN, no Vision/speculation/ranks");
    // Narrow first cold candidate: no retained-prefix/Host/disk scheduler or hidden
    // HybridPrefixCache restore_stream. This does not silently change user options.
    if(cache.enabled||!cache.disk_kv_path.empty()||cache.disk_kv_capacity_bytes!=0||
       cache.disk_kv_restore||cache.disk_kv_directstorage)
        throw std::invalid_argument("initial segmented Engine planner requires explicitly disabled context cache and disk tier");
    if(ring_requested()||reuse_hostbacked_enabled()||ops::detail::kvmem_score_enabled()||
       ops::detail::kvmem_shadow_enabled())
        throw std::invalid_argument("segmented planner does not admit ring/retrieval/score/shadow policy");
}

std::size_t segmented_persistent_resident_bytes(const PersistentLayout& persistent) {
    return checked_add(checked_add(persistent.bytes,persistent.segmented_payload_bytes,
                                  "segmented resident payload"),
                       persistent.segmented_pointer_table_bytes,"segmented resident table");
}
