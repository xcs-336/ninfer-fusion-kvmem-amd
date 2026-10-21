// Within original planner detail namespace, outside the anonymous helper namespace.
void validate_segmented_runtime_policy() {
    const char* selected=std::getenv("NINFER_PROTO_SELECT");
    if(ring_requested() || reuse_hostbacked_enabled() || ops::detail::kvmem_score_enabled() ||
       ops::detail::kvmem_shadow_enabled() || (selected && std::string_view(selected)!="all"))
        throw std::invalid_argument("segmented Engine candidate rejects ring/reuse/score/shadow/selection before allocation");
    if(const char* mode=std::getenv("NINFER_KV_DTYPE_AUTO");mode && mode[0] && std::strtol(mode,nullptr,10)>0)
        throw std::invalid_argument("segmented Engine candidate rejects automatic KV precision selection");
}
void validate_segmented_preallocation(const Config& config,const EngineOptions& options,
                                     DeviceContext& device) {
    const auto features=models::load_options(options);
    const auto& text=config.text;
    if(options.max_context==0 || options.max_context>512 || options.max_concurrency!=1 ||
       device.size()!=1 || device.rank(0).device!=options.device || features.ranks!=1 ||
       options.kv_cache!=KvCacheStorage::BFloat16 || !text.attention ||
       text.full_attention_layers==0 || text.attention->head_dim!=256 ||
       !((text.attention->num_attention_heads==16 && text.attention->num_key_value_heads==2) ||
         (text.attention->num_attention_heads==24 && text.attention->num_key_value_heads==4)) ||
       features.vision || features.speculative!=SpeculativeBackend::None || features.gdn_state_fp16 ||
       options.gdn_state_storage!=GdnStateStorage::FP32 || options.speculative.ngram_archive_bytes!=0 ||
       options.speculative.mtp_attention_window!=0 ||
       options.speculative.draft_tokens!=0 || options.speculative.ngram_draft_tokens!=0 ||
       options.speculative.lookup_ngram!=0 || options.context_cache.enabled ||
       !options.context_cache.disk_kv_path.empty() || options.context_cache.disk_kv_capacity_bytes!=0 ||
       options.context_cache.disk_kv_restore || options.context_cache.disk_kv_directstorage ||
       options.device_profile!="off")
        throw std::invalid_argument("segmented Engine candidate requires explicit profile-off, B1 dense512 BF16 D256 KV2/4 FP32 GDN, no prefix/disk/Vision/draft/ranks");
    validate_segmented_runtime_policy();
    int actual=-1;CUDA_CHECK(hipGetDevice(&actual));
    if(actual!=options.device)throw std::logic_error("startup thread must explicitly bind the candidate's actual device");
}
void validate_segmented_sequence_plan(const execution::Parameters& parameters,
                                    const SequencePlanImpl& plan,DeviceContext& device) {
    if(!plan.segmented_text_storage)return;
    if(plan.parameters!=&parameters || parameters.model.options()!=plan.features)
        throw std::invalid_argument("segmented Program parameters differ from its frozen plan");
    require_segmented_engine_scope(parameters,plan.capacity,plan.max_concurrency,
        plan.kv_storage,plan.features,plan.context_cache);
    validate_segmented_runtime_policy();
    if(device.size()!=1 || device.rank(0).device!=plan.device ||
       plan.speculative_backend!=SpeculativeBackend::None || plan.draft_window!=0 ||
       plan.ngram_draft_window!=0 || plan.lookup_ngram!=0 ||
       plan.main_page_groups<(plan.capacity+63U)/64U ||
       !plan.persistent.extra_rank_bytes.empty() || !plan.extra_rank_reservation_bytes.empty() ||
       !plan.persistent.decoder.text_kv.external_segmented_payload || plan.persistent.decoder.mtp_kv)
        throw std::invalid_argument("segmented Program rejects non-dense, MTP or split resident layout");
    int actual=-1;CUDA_CHECK(hipGetDevice(&actual));
    if(actual!=plan.device)throw std::logic_error("native factory needs the explicitly bound planned device");
}
