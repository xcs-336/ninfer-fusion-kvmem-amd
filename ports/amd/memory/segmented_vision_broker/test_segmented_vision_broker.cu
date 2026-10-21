// SOURCE ONLY. Actual extracted production Broker + actual new full Core.
#ifndef NINFER_BROKER_ORIGINAL_CORE_FIXTURE
#error "Exact new-Core original-pool helper snapshot required"
#endif
#define main unused_segmented_vision_original_pool_fixture_main
#include NINFER_BROKER_ORIGINAL_CORE_FIXTURE
#undef main
#include "amd-segmented-vision/vision_broker.h"

namespace {
using ninfer::models::qwen3_5::execution::VisionResidencyBroker;
using ninfer::models::qwen3_5::execution::VisionWindow;
struct BrokerDrain {
    DeviceContext& device;
    ~BrokerDrain(){for(auto s:std::array<hipStream_t,4>{device.stream,device.transfer_stream,device.vision_stream,nullptr})
        (void)hipStreamSynchronize(s);}
};
class BrokerWorld {
public:
    BrokerWorld():device(0),native(config(),layout()),
        weights(device,{EvictableWeightPool::kChunkBytes,EvictableWeightPool::kChunkBytes}),broker(device,weights) {
        const auto home=weights.arena();need(home.bytes==EvictableWeightPool::kChunkBytes,"broker actual WeightTail arena bound");
        check(hipMemsetAsync(home.data,0x5c,home.bytes,device.stream),"broker real weight pattern");
        weights.capture_window_mirror(EvictableWeightPool::kChunkBytes,device.stream);
        need(weights.mirror_captured()&&weights.mirror_bytes()==home.bytes,"broker real native pinned mirror absent");
        auto& pool=native.page_pool();auto entitlement=pool.reserve(LIVE);
        need(entitlement.has_value(),"broker original reserve failed");
        leases.reserve(LIVE);pool.materialize(*entitlement,LIVE,leases);ids=handles(leases);
        host_layout=plan_host_kv_page_layout(pool.geometry());
        const std::array<HostKVPageLayout,1> supported{host_layout};
        arena=std::make_unique<HostKVArena>(LIVE*host_layout.page_stride,supported);
        input=arena->allocate(host_layout,LIVE);need(input.has_value(),"broker real Host KV allocation failed");
        expected=pattern(18031);auto host=arena->writable_view(*input);
        pack(host.data(),host_layout,expected,host_layout.page_stride);
        pool.copy_from_host(arena->view(*input),ids,RankStreams{device.stream});
        LayoutBuilder builder;auto planned=plan_kv_execution_tables(builder,{LIVE,1});
        metadata=std::make_unique<DeviceArena>(builder.finish(256));need(metadata->capacity()<=4096,"broker metadata bound");
        tables=std::make_unique<KVExecutionTablePool>(DeviceSpan{metadata->base(),metadata->capacity()},planned,pool);
        check(hipStreamSynchronize(nullptr),"broker default metadata initialization completion");
        row.emplace(tables->acquire(0));tables->publish(row->handle(),0,ids,RankStreams{device.stream});
        check(hipStreamSynchronize(device.stream),"broker actual upload/table ready");
        table=pool.plane_view(0).page_bases;observed=DeviceBuffer(P*LIVE*S+2*GUARD);
        check(hipStreamBeginCapture(device.stream,hipStreamCaptureModeGlobal),"broker independent observer capture");
        hipLaunchKernelGGL(observe,dim3(64),dim3(256),0,device.stream,table,
            static_cast<const std::int32_t*>(tables->row(row->handle()).data),
            reinterpret_cast<std::uint16_t*>(static_cast<std::byte*>(observed.p)+GUARD));
        check(hipGetLastError(),"broker independent observer capture kernel");
        check(hipStreamEndCapture(device.stream,&graph.graph),"broker observer capture end");
        check(hipGraphInstantiate(&graph.exec,graph.graph,nullptr,nullptr,0),"broker observer instantiate");
        original_exec=graph.exec;const auto inventory=users();
        broker.enable_segmented_tier(native.storage(),4*S,inventory,inventory,
            [this]{return can_lend;},[this]{++changes;});
    }
    std::array<hipStream_t,4> users() const {return {device.stream,device.transfer_stream,device.vision_stream,nullptr};}
    void finish_users() {for(auto s:users())check(hipStreamSynchronize(s),"broker explicit normal user completion");}
    void verify() {
        need(graph.exec==original_exec&&native.page_pool().plane_view(0).page_bases==table,"broker replaced observer/table allocation");
        check(hipMemsetAsync(observed.p,0xa5,observed.bytes,device.stream),"broker fresh complete output/guards");
        check(hipGraphLaunch(graph.exec,device.stream),"broker same Graph execution");
        check(hipStreamSynchronize(device.stream),"broker observer complete");
        std::vector<std::byte> actual(observed.bytes);observed.copy_to_host(actual.data(),actual.size());
        need(std::all_of(actual.begin(),actual.begin()+GUARD,[](auto x){return x==std::byte{0xa5};})&&
             std::all_of(actual.end()-GUARD,actual.end(),[](auto x){return x==std::byte{0xa5};}),"broker observer guards");
        need(std::equal(actual.begin()+GUARD,actual.end()-GUARD,reinterpret_cast<const std::byte*>(expected.data())),
             "broker live KV full-byte oracle");++replays;
    }
    void verify_weights() {
        std::vector<std::byte> piece(1024*1024);const auto home=weights.arena();
        for(std::size_t offset=0;offset<home.bytes;offset+=piece.size()) {
            const auto bytes=std::min(piece.size(),home.bytes-offset);
            check(hipMemcpy(piece.data(),static_cast<const std::byte*>(home.data)+offset,bytes,hipMemcpyDeviceToHost),
                  "broker original native WeightTail actual restore readback");
            need(std::all_of(piece.begin(),piece.begin()+bytes,[](auto x){return x==std::byte{0x5c};}),
                 "broker normal WeightTail restore changed actual weight bytes");
        }
    }
    DeviceContext device;KVSegmentedNativePool native;EvictableWeightPool weights;VisionResidencyBroker broker;
    std::vector<DeviceKVPageLease> leases;std::vector<DeviceKVPageHandle> ids;
    HostKVPageLayout host_layout;std::unique_ptr<HostKVArena> arena;std::optional<HostKVAllocation> input;
    std::unique_ptr<DeviceArena> metadata;std::unique_ptr<KVExecutionTablePool> tables;std::optional<KVExecutionRowLease> row;
    std::vector<std::uint16_t> expected;DeviceBuffer observed;Graph graph;
    const std::uint64_t* table=nullptr;hipGraphExec_t original_exec=nullptr;
    bool can_lend=true;unsigned changes=0,replays=0;
    BrokerDrain drain_before_every_resource{device};
};

unsigned normal_fallback_and_unique_grant() {
    BrokerWorld world;auto& pool=world.native.page_pool();world.verify();
    auto promised=pool.reserve(3);need(promised.has_value()&&pool.available_pages()==1,"broker real capacity refusal setup");
    const auto refused_epoch=world.native.controller_status().mutation_epoch;
    need(!world.broker.try_acquire_kv(4*S)&&world.changes==0&&pool.lent_pages()==0,
         "typed capacity empty changed broker revision/capacity");
    need(world.native.controller_status().mutation_epoch==refused_epoch,"broker capacity query mutated allocator epoch");
    bool original_threw=false;
    try {const auto users=world.users();auto invalid=world.native.storage().begin_loan(4*S,users);}
    catch(const std::runtime_error& error) {
        need(std::string_view(error.what())=="insufficient unpromised segmented KV capacity",
             "broker fixture linked a different original throwing Core contract");original_threw=true;
    }
    need(original_threw&&world.native.controller_status().mutation_epoch==refused_epoch&&world.changes==0,
         "original throwing wrapper changed real capacity/revision");
    auto fallback=world.broker.acquire(4*S);
    need(fallback.open()&&fallback.tier()==VisionWindow::Tier::WeightTail&&
         fallback.memory().bytes==EvictableWeightPool::kChunkBytes&&world.changes==0&&pool.lent_pages()==0,
         "broker did not use actual original WeightTail fallback after typed empty");
    check(hipMemsetAsync(fallback.memory().data,0xd9,fallback.memory().bytes,world.device.transfer_stream),"actual fallback dirty bytes");
    world.finish_users(); // Legacy obligation, not a new checked-restore theorem.
    need(fallback.close_checked()&&!world.broker.poisoned(),"normal legacy WeightTail close reports poison");
    world.verify_weights();world.verify();
    promised->clear();world.can_lend=false;
    const auto policy_epoch=world.native.controller_status().mutation_epoch;
    need(!world.broker.try_acquire_kv(4*S)&&world.changes==0&&pool.lent_pages()==0&&
         world.native.controller_status().mutation_epoch==policy_epoch,"original can_lend policy was ignored");
    world.can_lend=true;auto window=world.broker.acquire(4*S);
    need(window.tier()==VisionWindow::Tier::KvSegmented&&pool.lent_pages()==2&&world.changes==1,
         "broker segmented tier failed actual unique grant/revision publication");
    VisionWindow moved(std::move(window));need(!window.open()&&moved.open(),"broker window move duplicated ownership");
    moved.wait_ready(world.device.transfer_stream);moved.wait_ready(world.device.vision_stream);
    check(hipMemsetAsync(moved.memory().data,0xd3,moved.memory().bytes,world.device.transfer_stream),"actual segmented borrower dirty bytes");
    world.finish_users();world.verify();
    need(moved.close_checked()&&!moved.open()&&!world.broker.window_open()&&
         pool.lent_pages()==0&&pool.available_pages()==B-LIVE&&world.changes==2,
         "broker checked completion failed actual single return/revision");
    const auto closed_epoch=world.native.controller_status().mutation_epoch;
    need(closed_epoch==policy_epoch+2,"broker called legacy return or duplicated grant mutation");
    need(moved.close_checked()&&world.changes==2&&world.native.controller_status().mutation_epoch==closed_epoch,
         "broker idempotent close republished/reissued a grant");world.verify();return world.replays;
}

void semantic_close_failure_stays_lent() {
    BrokerWorld world;auto& pool=world.native.page_pool();auto window=world.broker.acquire(4*S);
    need(window.tier()==VisionWindow::Tier::KvSegmented&&world.changes==1&&pool.lent_pages()==2,"broker close negative actual grant absent");
    const auto epoch=world.native.controller_status().mutation_epoch;
    hipGraph_t empty=nullptr;std::exception_ptr unexpected;bool returned=true;
    check(hipStreamBeginCapture(world.device.stream,hipStreamCaptureModeGlobal),"broker finite semantic close negative capture");
    try{returned=window.close_checked();}catch(...){unexpected=std::current_exception();}
    const auto ended=hipStreamEndCapture(world.device.stream,&empty);
    if(empty)check(hipGraphDestroy(empty),"broker discard never-launched negative capture");
    check(ended,"broker close negative capture end");if(unexpected)std::rethrow_exception(unexpected);
    need(!returned&&window.open()&&window.close_failed()&&world.broker.window_open()&&world.broker.poisoned()&&
         world.broker.sticky_failure()&&world.native.storage().poisoned()&&world.native.storage().lease_open()&&
         pool.lent_pages()==2&&world.changes==1,"broker semantic failure falsely reported return/closed/revision");
    need(!window.close_checked()&&world.changes==1&&pool.lent_pages()==2&&
         world.native.controller_status().mutation_epoch==epoch,"broker retried a failed close or returned IDs");
    bool refused=false;try{auto invalid=world.broker.acquire(S);}catch(const std::runtime_error&){refused=true;}
    need(refused&&pool.lent_pages()==2&&world.changes==1,"broker poison fell back to healthy WeightTail");
    // Backend retains the grant; destructor does not retry or notify. This is a
    // valid-handle capture-precondition refusal, never a fabricated driver fault.
}
} // namespace

int main(int argc,char** argv) {
    try {
        if(argc==1){std::cout<<"{\"status\":\"PLAN_ONLY\",\"main_explicit_hip_calls\":false,"
            "\"pre_main_registration_not_excluded\":true,\"Session_Engine_connected\":false}\n";return 0;}
        need(argc==2&&std::string_view(argv[1])=="--run","usage: ninfer_amd_segmented_vision_broker_test [--run]");
        const auto replays=normal_fallback_and_unique_grant();semantic_close_failure_stays_lent();
        std::cout<<"{\"status\":\"REAL_BROKER_COMPONENT_CHECKS_PASSED\",\"sameGraph_replays\":"<<replays<<","
            "\"actual_typed_capacity_and_unique_grant\":true,\"normal_WeightTail_bytes\":true,"
            "\"semantic_close_retains_lent_poison\":true,\"checked_WeightTail_restore_qualified\":false,"
            "\"pending_ordering_qualified\":false,\"Session_Engine_connected\":false,\"production_gate_changed\":false}\n";return 0;
    }catch(const std::exception& error){std::cerr<<"REAL_BROKER_COMPONENT_CHECKS_FAILED: "<<error.what()<<'\n';return 1;}
}
