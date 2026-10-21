// SOURCE ONLY / NOT_BUILT / NOT_RUN. This is a new public-policy candidate
// fixture, not a replacement for the frozen six-case fixture or original suite.
#include "fixture_common.cuh"
#include <memory>

namespace {
struct HostStage {
    int row = 0, base = 0;
    std::vector<std::int32_t> positions;
    std::vector<std::uint16_t> query, gate, logical_k, logical_v;
    std::vector<double> ideal;
};

void one_public_case(int kv_heads, int width, std::uint64_t& checks, std::uint64_t& replays) {
    const int q_heads = kv_heads == 4 ? 24 : 16;
    const auto slot_bytes = std::uint64_t{D} * 64 * kv_heads * 2;
    const auto slot_words = static_cast<std::size_t>(slot_bytes / 2);
    const ops::AttentionHeadGeometry geometry{D, q_heads, kv_heads};
    const ops::CausalAttentionExecutionEnvelope envelope{1, 512};
    // Validate finite public candidate scope and obtain ACTUAL original resource
    // planning before this case allocates payload/partials. No fixed max split.
    const auto capacity = ops::segmented_b1_causal_softmax_attention_workspace_capacity_bytes(
        geometry, KvCacheStorage::BFloat16, envelope, 1, width, width);
    const int family = ops::segmented_b1_causal_softmax_attention_route_family(
        geometry, KvCacheStorage::BFloat16, envelope, 1, width);
    need(family == (width == 3 ? 0 : 2), "actual original B1 resolver selected unexpected family");
    if (family == 2) need(capacity == 0, "actual prompt workspace must be zero");

    DeviceKVPagePoolSpec spec;
    spec.page_group_count = Groups;
    spec.geometry.device_plane_order = PagedKVPlaneOrder::PageMajor;
    spec.geometry.planes = {{DType::BF16, D, kv_heads, 256}, {DType::BF16, D, kv_heads, 256}};
    LayoutBuilder payload_layout;
    const auto layout = plan_device_kv_page_pool(payload_layout, spec);
    KVSegmentedStorage::Config config;
    config.device = 0; config.planes = 2; config.page_groups = Groups; config.slot_bytes = slot_bytes;
    config.window_offset = 0; config.max_window_bytes = Groups * slot_bytes; config.max_user_streams = 2;
    KVSegmentedNativePool native(config, layout);
    auto& pool = native.page_pool();
    auto& storage = native.storage();
    Stream consumer, borrower;
    auto reservation = pool.reserve(LogicalPages);
    need(reservation.has_value(), "public fixture reserve refused");
    std::vector<DeviceKVPageLease> pages;
    pages.reserve(LogicalPages);
    pool.materialize(*reservation, LogicalPages, pages);
    need(pages.size() == LogicalPages, "public fixture did not materialize real handles");
    LayoutBuilder table_builder;
    const auto table_layout = plan_kv_execution_tables(table_builder, {LogicalPages, 2});
    Buffer block_buffer(table_builder.finish());
    KVExecutionTablePool execution(DeviceSpan{block_buffer.data, block_buffer.bytes}, table_layout, pool);
    auto row0 = execution.acquire(0), row1 = execution.acquire(1);
    std::array<std::array<DeviceKVPageHandle, LogicalPages>, 2> row_handles{};
    for (int i = 0; i < LogicalPages; ++i) {
        row_handles[0][i] = pages[i].handle();
        row_handles[1][i] = pages[(i + 3) % LogicalPages].handle();
    }
    execution.publish(row0.handle(), 0, row_handles[0], RankStreams{consumer.value});
    execution.publish(row1.handle(), 0, row_handles[1], RankStreams{consumer.value});
    const auto published = block_buffer.read<std::int32_t>(consumer.value);
    need(published.size() == 2 * LogicalPages, "unexpected execution-table padding");
    const auto kp = pool.plane_view(0), vp = pool.plane_view(1);
    PagedKVBatchLayerView cache;
    cache.k_pages = kp.storage; cache.v_pages = vp.storage; cache.block_tables = execution.matrix();
    cache.head_dim = D; cache.num_kv_heads = kv_heads; cache.storage = KvCacheStorage::BFloat16;
    cache.k_page_bases = kp.page_bases; cache.v_page_bases = vp.page_bases;
    need(cache.k_page_bases && cache.v_page_bases, "missing stable page tables");
    std::vector<std::uint64_t> identity(2*Groups);
    check(hipMemcpyAsync(identity.data(),kp.page_bases,identity.size()*8,hipMemcpyDeviceToHost,consumer.value),"read initial identity map");
    check(hipStreamSynchronize(consumer.value),"finish initial identity proof");
    for(int plane=0;plane<2;++plane) for(int group=0;group<Groups;++group)
        need(identity[plane*Groups+group]==reinterpret_cast<std::uintptr_t>((plane==0?kp:vp).storage.data)+group*slot_bytes,
             "affine control requires explicit initial identity placement");
    const auto canonical_k = pattern(static_cast<std::size_t>(D) * kv_heads * LogicalTokens, 5, 128);
    const auto canonical_v = pattern(canonical_k.size(), 7, 32);
    const auto new_k = pattern(static_cast<std::size_t>(D) * kv_heads * width, 17, 128);
    const auto new_v = pattern(new_k.size(), 19, 32);
    std::array<HostStage, 2> stages;
    for (int s = 0; s < 2; ++s) {
        auto& stage = stages[s];
        stage.row = s; stage.base = s == 0 ? 63 : 512-width;
        stage.positions.resize(width);
        for (int t = 0; t < width; ++t) stage.positions[t] = stage.base + t;
        stage.query = pattern(static_cast<std::size_t>(D) * q_heads * width, 11 + s * 29, 128);
        stage.gate = pattern(stage.query.size(), 23 + s * 31, 16);
        stage.logical_k.resize(canonical_k.size()); stage.logical_v.resize(canonical_v.size());
        for (int p = 0; p < LogicalPages; ++p) {
            const int source_page = s == 0 ? p : (p + 3) % LogicalPages;
            for (int h = 0; h < kv_heads; ++h) for (int t = 0; t < 64; ++t) for (int d = 0; d < D; ++d) {
                stage.logical_k[logical_index(kv_heads,h,p*64+t,d)] = canonical_k[logical_index(kv_heads,h,source_page*64+t,d)];
                stage.logical_v[logical_index(kv_heads,h,p*64+t,d)] = canonical_v[logical_index(kv_heads,h,source_page*64+t,d)];
            }
        }
        for (int t = 0; t < width; ++t) for (int h = 0; h < kv_heads; ++h) for (int d = 0; d < D; ++d) {
            stage.logical_k[logical_index(kv_heads,h,stage.base+t,d)] = new_k[logical_index(kv_heads,h,t,d)];
            stage.logical_v[logical_index(kv_heads,h,stage.base+t,d)] = new_v[logical_index(kv_heads,h,t,d)];
        }
        stage.ideal = oracle(stage.query, stage.logical_k, stage.logical_v, kv_heads, stage.base, width);
    }
    Buffer q(stages[0].query.size()*2), gate(stages[0].gate.size()*2), k(new_k.size()*2), v(new_v.size()*2);
    Buffer positions(width*4), row_selector(4), out(q.bytes), plain(q.bytes), split_gate(q.bytes);
    // Prompt has zero live workspace. A guarded 256B sentinel supplies a legal
    // DeviceArena backing; its peak must stay zero. It is never hidden scratch.
    Buffer scratch(std::max<std::size_t>(capacity, 256));
    WorkspaceArena workspace(DeviceSpan{scratch.data, scratch.bytes});
    Tensor tq(q.data,DType::BF16,{D,q_heads,width,1}), tk(k.data,DType::BF16,{D,kv_heads,width,1});
    Tensor tv(v.data,DType::BF16,{D,kv_heads,width,1}), tg(gate.data,DType::BF16,{D,q_heads,width,1});
    Tensor tp(positions.data,DType::I32,{width,1,1,1}), tr(row_selector.data,DType::I32,{1,1,1,1});
    Tensor to(out.data,DType::BF16,{D,q_heads,width,1}), tu(plain.data,DType::BF16,{D,q_heads,width,1});
    Tensor ts(split_gate.data,DType::BF16,{D,q_heads,width,1});
    const Tensor valid{};
    k.upload(new_k.data(),consumer.value); v.upload(new_v.data(),consumer.value);
    const auto launch = [&](const PagedKVBatchLayerView& selected, bool gated_arm) {
        Tensor& target=gated_arm?to:tu;
        ops::segmented_b1_causal_softmax_attention(tq,tk,tv,tp,valid,tr,geometry,Scale,selected,envelope,workspace,target,consumer.value,gated_arm?&tg:nullptr);
    };
    const auto host_layout = plan_host_kv_page_layout(spec.geometry);
    PinnedHostBuffer records(host_layout.page_stride * LogicalPages);
    std::vector<const std::byte*> record_addresses(LogicalPages);
    std::vector<std::uint32_t> record_groups(LogicalPages,0);
    std::array<DeviceKVPageHandle, LogicalPages> destinations{};
    for (int i=0;i<LogicalPages;++i) destinations[i]=pages[i].handle();
    auto seed = [&](int selected_stage) {
        const auto& stage = stages[selected_stage];
        for (int p=0;p<LogicalPages;++p) {
            auto* record=static_cast<std::byte*>(records.data())+p*host_layout.page_stride;
            record_addresses[p]=record;
            for (int plane=0;plane<2;++plane) {
                auto* page=reinterpret_cast<std::uint16_t*>(record+host_layout.planes[plane].offset);
                const auto& canonical=plane==0?canonical_k:canonical_v;
                for(int h=0;h<kv_heads;++h) for(int t=0;t<64;++t) for(int d=0;d<D;++d)
                    page[d+D*(t+64*h)]=canonical[logical_index(kv_heads,h,p*64+t,d)];
            }
        }
        // Resolve through the independently read published row, never a guessed
        // physical ID or host-baked selector used by the kernel.
        for(int t=0;t<width;++t) {
            const int logical=stage.base+t;
            const int page_index=(logical/64+(selected_stage?3:0))%LogicalPages;
            auto* record=static_cast<std::byte*>(records.data())+page_index*host_layout.page_stride;
            for(int plane=0;plane<2;++plane) {
                auto* page=reinterpret_cast<std::uint16_t*>(record+host_layout.planes[plane].offset);
                for(int h=0;h<kv_heads;++h) for(int d=0;d<D;++d) page[d+D*(logical%64+64*h)]=0xffff;
            }
        }
        pool.copy_from_host_records(record_addresses,record_groups,destinations,host_layout,RankStreams{consumer.value});
        q.upload(stage.query.data(),consumer.value); gate.upload(stage.gate.data(),consumer.value);
        positions.upload(stage.positions.data(),consumer.value); row_selector.upload(&stage.row,consumer.value);
        check(hipMemsetAsync(out.data,0xff,out.bytes,consumer.value),"poison gated output");
        check(hipMemsetAsync(plain.data,0xff,plain.bytes,consumer.value),"poison ungated output");
        need(workspace.used()==0,"cannot poison a live workspace scope");
        check(hipMemsetAsync(scratch.data,0xff,scratch.bytes,consumer.value),"poison previous arm partials");
        check(hipStreamSynchronize(consumer.value),"finish changed metadata and mapped poison");
        std::vector<std::uint64_t> mapping(2*Groups);
        check(hipMemcpyAsync(mapping.data(),kp.page_bases,mapping.size()*8,hipMemcpyDeviceToHost,consumer.value),"read current seed table");
        check(hipStreamSynchronize(consumer.value),"finish seed table");
        std::array<std::uint16_t,D> bits{};
        for(int plane=0;plane<2;++plane) for(int t=0;t<width;++t) for(int h=0;h<kv_heads;++h) {
            const int logical=stage.base+t, group=published[selected_stage*LogicalPages+logical/64];
            const auto* page=reinterpret_cast<const std::uint16_t*>(static_cast<std::uintptr_t>(mapping[plane*Groups+group]));
            check(hipMemcpyAsync(bits.data(),page+D*(logical%64+64*h),D*2,hipMemcpyDeviceToHost,consumer.value),"read poisoned actual append row");
            check(hipStreamSynchronize(consumer.value),"finish append poison proof");
            need(std::all_of(bits.begin(),bits.end(),[](auto x){return x==0xffff;}),"fresh mapped append poison missing");
        }
    };
    KVSegmentedSpan borrowed{};
    std::array<std::vector<std::uint16_t>,2> affine_output, affine_plain;
    std::vector<std::uint16_t> verified_plain;
    int verified_plain_stage=-1;
    auto verify = [&](int selected_stage, bool gated_arm, bool compare_affine) {
        const auto& stage=stages[selected_stage];
        const auto got=(gated_arm?out:plain).read<std::uint16_t>(consumer.value);
        if(!gated_arm) {
            std::vector<double> actual(got.size());
            std::transform(got.begin(),got.end(),actual.begin(),fp64);
            const test::ReductionCriterion attention{2.8e-3,1.0e-3,test::kBf16GrossRelativeFloor};
            need(test::reduction_passes(test::compute_reduction_stats(actual.data(),stage.ideal.data(),actual.size()),actual.size(),attention),"original independent FP64 attention criterion failed");
            verified_plain=got; verified_plain_stage=selected_stage;
        } else {
            need(verified_plain_stage==selected_stage && verified_plain.size()==got.size(),"gated oracle needs this phase's independently verified BF16 boundary");
            std::vector<double> gated(got.size()),gate_ref(got.size());
            for(std::size_t i=0;i<got.size();++i) {
                gated[i]=fp64(got[i]);
                // This host checkpoint passed the independent FP64 attention
                // oracle in the separate ungated arm. Its DEVICE output and all
                // partials were poisoned again before the gated arm replayed.
                gate_ref[i]=fp64(verified_plain[i])/(1.0+std::exp(-fp64(stage.gate[i])));
            }
            const test::PointwiseCriterion gate_criterion{2.0e-5,4.05e-3};
            need(test::pointwise_passes(test::compute_pointwise_stats(gated.data(),gate_ref.data(),gated.size(),gate_criterion),gated.size()),"independent FP64 sigmoid gate criterion failed");
            split_gate.upload(verified_plain.data(),consumer.value);
            ops::sigmoid_mul(tg,ts,consumer.value);
            need(split_gate.read<std::uint16_t>(consumer.value)==got,"public gated route differs bitwise from original BF16 split sigmoid_mul");
            verified_plain_stage=-1;
        }
        if(compare_affine) need(got==(gated_arm?affine_output:affine_plain)[selected_stage],"same arm Graph differs from stage-matched affine control");
        need(q.read<std::uint16_t>(consumer.value)==stage.query && gate.read<std::uint16_t>(consumer.value)==stage.gate &&
             k.read<std::uint16_t>(consumer.value)==new_k && v.read<std::uint16_t>(consumer.value)==new_v &&
             positions.read<std::int32_t>(consumer.value)==stage.positions && row_selector.read<std::int32_t>(consumer.value)==std::vector<std::int32_t>{stage.row},"public wrapper modified input");
        need(block_buffer.read<std::int32_t>(consumer.value)==published,"execution-table matrix was modified");
        std::vector<std::uint64_t> mapping(2*Groups);
        check(hipMemcpyAsync(mapping.data(),kp.page_bases,mapping.size()*8,hipMemcpyDeviceToHost,consumer.value),"read verified current mapping");
        check(hipStreamSynchronize(consumer.value),"finish map read");
        std::vector<std::uint16_t> page(slot_words);
        for(int plane=0;plane<2;++plane) for(int logical=0;logical<LogicalPages;++logical) {
            const int group=published[selected_stage*LogicalPages+logical];
            check(hipMemcpyAsync(page.data(),reinterpret_cast<const void*>(static_cast<std::uintptr_t>(mapping[plane*Groups+group])),slot_bytes,hipMemcpyDeviceToHost,consumer.value),"read exact current cache cell");
            check(hipStreamSynchronize(consumer.value),"finish cache byte read");
            const auto& expected=plane==0?stage.logical_k:stage.logical_v;
            for(int h=0;h<kv_heads;++h) for(int t=0;t<64;++t) for(int d=0;d<D;++d)
                need(page[d+D*(t+64*h)]==expected[logical_index(kv_heads,h,logical*64+t,d)],"full cache/history byte oracle failed");
        }
        for(const Buffer* buffer:{&q,&gate,&k,&v,&positions,&row_selector,&out,&plain,&split_gate,&scratch,&block_buffer}) buffer->guards(consumer.value);
        need(workspace.used()==0 && workspace.peak_used()<=capacity,"actual original workspace scope/capacity mismatch");
        if(borrowed.data) {
            std::vector<std::uint8_t> bytes(static_cast<std::size_t>(borrowed.bytes));
            check(hipMemcpyAsync(bytes.data(),borrowed.data,bytes.size(),hipMemcpyDeviceToHost,consumer.value),"read active loan canary");
            check(hipStreamSynchronize(consumer.value),"finish loan canary");
            need(std::all_of(bytes.begin(),bytes.end(),[](auto x){return x==0xd3;}),"public append overwrote lent window");
        }
        checks+=got.size();
    };
    auto affine=cache; affine.k_page_bases=nullptr; affine.v_page_bases=nullptr;
    for(int s=0;s<2;++s) {
        verified_plain_stage=-1;
        seed(s); launch(affine,false); check(hipStreamSynchronize(consumer.value),"finish original ungated affine control");
        verify(s,false,false); affine_plain[s]=verified_plain;
        seed(s); launch(affine,true); check(hipStreamSynchronize(consumer.value),"finish original gated affine control");
        verify(s,true,false); affine_output[s]=out.read<std::uint16_t>(consumer.value);
    }
    Graph ungated_graph,gated_graph;
    ungated_graph.capture(consumer.value,[&]{launch(cache,false);});
    gated_graph.capture(consumer.value,[&]{launch(cache,true);});
    const std::array<hipGraphExec_t,2> captured{ungated_graph.exec,gated_graph.exec};
    const auto replay_stage=[&](int s) {
        verified_plain_stage=-1;
        seed(s); ungated_graph.replay(consumer.value); ++replays; verify(s,false,true);
        seed(s); gated_graph.replay(consumer.value); ++replays; verify(s,true,true);
        need(ungated_graph.exec==captured[0] && gated_graph.exec==captured[1] &&
             cache.k_page_bases==kp.page_bases && cache.v_page_bases==vp.page_bases,
             "arm Graph or table allocation changed instead of device contents");
    };
    replay_stage(0);
    std::vector<std::uint64_t> before(2*Groups),after(2*Groups);
    check(hipMemcpyAsync(before.data(),kp.page_bases,before.size()*8,hipMemcpyDeviceToHost,consumer.value),"before mapping");
    check(hipStreamSynchronize(consumer.value),"finish before mapping");
    const std::array<hipStream_t,2> users{consumer.value,borrower.value};
    auto loan=storage.begin_loan((Groups-1)*slot_bytes+1,users);
    need(loan.open() && loan.copied_live_bytes()>=LogicalPages*slot_bytes,"real loan did not move live cells");
    loan.wait_ready(consumer.value); loan.wait_ready(borrower.value);
    check(hipMemcpyAsync(after.data(),kp.page_bases,after.size()*8,hipMemcpyDeviceToHost,consumer.value),"after mapping");
    check(hipStreamSynchronize(consumer.value),"finish after mapping");
    need(before!=after,"mapping did not change");
    bool cross_plane=false;
    for(int i=0;i<LogicalPages;++i) cross_plane|=after[published[i]]>=reinterpret_cast<std::uintptr_t>(vp.storage.data);
    need(cross_plane,"no actual K cell crossed semantic plane slabs");
    borrowed=loan.leased();
    poison_borrowed_span<<<64,256,0,borrower.value>>>(static_cast<std::uint8_t*>(borrowed.data),borrowed.bytes);
    check(hipGetLastError(),"launch raw borrower canary");
    check(hipStreamSynchronize(borrower.value),"finish borrower canary");
    replay_stage(1);
    need(loan.close_after_streams(users),"typed loan close failed"); borrowed={};
    replay_stage(0);
    need(pool.lent_pages()==0 && !storage.poisoned(),"excluded capacity was not returned");
}
} // namespace

int main(int argc,char** argv) {
    if(argc==1) {
        std::cout<<"{\"status\":\"PLAN_ONLY_NOT_QUALIFIED\",\"gpu_called\":false,\"scope\":\"actual public B1 BF16 dense <=512; dynamic device row and gate\"}\n";
        return 0;
    }
    if(argc!=2 || std::string_view(argv[1])!="--run") return 2;
    try {
        const auto* pageable=std::getenv("NINFER_HOST_PAGEABLE");
        need(!pageable || pageable[0]=='\0' || pageable[0]=='0',"requires real pinned Host-record path");
        // Mandatory host preflight rejects unsupported scope before allocation.
        bool rejected=false;
        try { (void)ops::segmented_b1_causal_softmax_attention_workspace_capacity_bytes({256,24,4},KvCacheStorage::BFloat16,{1,513},1,1,1); }
        catch(const std::invalid_argument&) { rejected=true; }
        need(rejected,"unsupported >512 profile was not rejected before allocation");
        check(hipSetDevice(0),"select native device");
        hipDeviceProp_t props{}; check(hipGetDeviceProperties(&props,0),"device properties");
        need(std::string_view(props.gcnArchName).starts_with("gfx1201"),"candidate requires gfx1201");
        std::uint64_t checks=0,replays=0;
        for(int kv:{4,2}) for(int width:{3,35}) one_public_case(kv,width,checks,replays);
        std::cout<<"{\"status\":\"FUNCTIONAL_ONLY_PASS\",\"cases\":4,\"same_graph_replays\":"<<replays
                 <<",\"output_checks\":"<<checks<<",\"independent_arm_graphs\":true,\"dynamic_rows\":true,\"changed_positions\":true,\"fused_and_post_gate\":true,\"fp64_attention_and_gate\":true,\"full_cache_bytes\":true,\"fresh_nan_append_output_and_partials_per_arm\":true,\"engine_qualified\":false,\"rotation_qualified\":false,\"performance_sample\":false}\n";
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
