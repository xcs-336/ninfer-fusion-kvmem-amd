// SOURCE ONLY. Reuse the pinned original-Core byte observer/helpers, not a new
// allocator model. The copied original fixture's main is never called here.
#ifndef NINFER_CONTROLLER_ORIGINAL_CORE_FIXTURE
#error "Requires the pinned isolated original-Core fixture snapshot"
#endif
#define main unused_frozen_original_pool_fixture_main
#include NINFER_CONTROLLER_ORIGINAL_CORE_FIXTURE
#undef main
#ifndef NINFER_AMD_SEGMENTED_CONTROLLER_HANDOFF_ABI
#error "Never compile this test with old Core/old candidate class layouts"
#endif
#include <optional>
#include <type_traits>

static_assert(!std::is_copy_constructible_v<ninfer::amd::memory::KVControllerHandoffToken>);
static_assert(!std::is_copy_assignable_v<ninfer::amd::memory::KVControllerHandoffToken>);
static_assert(std::is_nothrow_move_constructible_v<ninfer::amd::memory::KVControllerHandoffToken>);

namespace {
class HandoffWorld {
public:
    HandoffWorld():native(config(),layout()) {
        auto& pool=native.page_pool();
        entitlement=pool.reserve(LIVE+1);need(entitlement.has_value(),"handoff real allocator reserve failed");
        leases.reserve(LIVE+1);pool.materialize(*entitlement,LIVE+1,leases);
        for(std::size_t i=0;i<LIVE;++i)ids.push_back(leases[i].handle());
        extra=std::move(leases.back());leases.pop_back(); // not referenced by the observer Graph
        host_layout=plan_host_kv_page_layout(pool.geometry());
        const std::array<HostKVPageLayout,1> supported{host_layout};
        arena=std::make_unique<HostKVArena>(LIVE*host_layout.page_stride,supported);
        image=arena->allocate(host_layout,LIVE);need(image.has_value(),"handoff Host image allocation failed");
        expected=pattern(717);
        auto host=arena->writable_view(*image);pack(host.data(),host_layout,expected,host_layout.page_stride);
        pool.copy_from_host(arena->view(*image),ids,RankStreams{streams.s[0]});
        LayoutBuilder builder;const auto planned=plan_kv_execution_tables(builder,{LIVE,1});
        metadata=std::make_unique<DeviceArena>(builder.finish(256));
        tables=std::make_unique<KVExecutionTablePool>(DeviceSpan{metadata->base(),metadata->capacity()},planned,pool);
        row.emplace(tables->acquire(0));tables->publish(row->handle(),0,ids,RankStreams{streams.s[0]});
        check(hipStreamSynchronize(streams.s[0]),"startup upload/table publication");
        table_pointer=pool.plane_view(0).page_bases;
        need(table_pointer&&pool.plane_view(1).page_bases==table_pointer+B,"handoff plane table inventory");
        observed=DeviceBuffer(P*LIVE*S+2*GUARD);
        check(hipStreamBeginCapture(streams.s[0],hipStreamCaptureModeGlobal),"handoff graph capture");
        hipLaunchKernelGGL(observe,dim3(64),dim3(256),0,streams.s[0],table_pointer,
            static_cast<const std::int32_t*>(tables->row(row->handle()).data),
            reinterpret_cast<std::uint16_t*>(static_cast<std::byte*>(observed.p)+GUARD));
        check(hipGetLastError(),"handoff observer capture");
        check(hipStreamEndCapture(streams.s[0],&graph.graph),"handoff capture end");
        check(hipGraphInstantiate(&graph.exec,graph.graph,nullptr,nullptr,0),"handoff graph instantiate");
        original_exec=graph.exec;
    }
    ~HandoffWorld() {
        // Normal test protocol has already retired+joined the worker. Conservative
        // failure cleanup is not evidence of successful retire or device-fault recovery.
        for(auto stream:streams.s)if(stream)(void)hipStreamSynchronize(stream);
    }
    void verify() {
        need(graph.exec==original_exec&&native.page_pool().plane_view(0).page_bases==table_pointer,
             "handoff replaced the Graph or stable device table allocation");
        check(hipMemsetAsync(observed.p,0xa5,observed.bytes,streams.s[0]),"fresh handoff output/guards");
        check(hipGraphLaunch(graph.exec,streams.s[0]),"same Graph on accepted controller");
        check(hipStreamSynchronize(streams.s[0]),"handoff observer finish");
        std::vector<std::byte> actual(observed.bytes);observed.copy_to_host(actual.data(),actual.size());
        need(std::all_of(actual.begin(),actual.begin()+GUARD,[](auto x){return x==std::byte{0xa5};})&&
             std::all_of(actual.end()-GUARD,actual.end(),[](auto x){return x==std::byte{0xa5};}),"handoff observer guards");
        need(std::equal(actual.begin()+GUARD,actual.end()-GUARD,reinterpret_cast<const std::byte*>(expected.data())),
             "same Graph logical bytes changed across controller or physical transfer");
        ++replays;
    }
    // Factory outlives leases/Host/execution table/Graph users. All are destroyed
    // on the joined parent after explicit worker retirement in the success path.
    KVSegmentedNativePool native;
    Streams streams;
    std::optional<DeviceKVPageReservation> entitlement;
    std::vector<DeviceKVPageLease> leases;
    DeviceKVPageLease extra;
    HostKVPageLayout host_layout;
    std::unique_ptr<HostKVArena> arena;
    std::optional<HostKVAllocation> image;
    std::unique_ptr<DeviceArena> metadata;
    std::unique_ptr<KVExecutionTablePool> tables;
    std::optional<KVExecutionRowLease> row;
    std::vector<DeviceKVPageHandle> ids;
    std::vector<std::uint16_t> expected;
    DeviceBuffer observed;
    Graph graph;
    const std::uint64_t* table_pointer=nullptr;
    hipGraphExec_t original_exec=nullptr;
    unsigned replays=0;
};

class JoinAndRelease {
public:
    JoinAndRelease(std::thread& worker,std::promise<void>& release):worker(worker),release(release){}
    ~JoinAndRelease() {
        if(!sent)try{release.set_value();}catch(const std::future_error&){}
        if(worker.joinable())worker.join();
    }
    void resume(){release.set_value();sent=true;}
private:std::thread& worker;std::promise<void>& release;bool sent=false;
};

std::string describe_failure(const std::exception_ptr& error) {
    if(!error)return "none";
    try{std::rethrow_exception(error);}
    catch(const std::exception& value){return value.what();}
    catch(...){return "non-standard exception";}
}

unsigned successful_once_handoff() {
    HandoffWorld world;world.verify();auto& native=world.native;auto& pool=native.page_pool();
    const auto startup=native.controller_status();
    need(startup.phase==KVControllerPhase::Startup&&!startup.disabled,"startup controller not enabled");
    const std::array<hipStream_t,2> inventory{world.streams.s[0],world.streams.s[1]};
    // An unauthorized worker's early call cannot disable the legal startup thread.
    std::exception_ptr early_error;
    std::thread early([&]{try{check(hipSetDevice(0),"early worker bind");
        reject([&]{auto invalid=native.release_controller_for_worker(inventory);},"wrong thread released controller");}
        catch(...){early_error=std::current_exception();}});
    early.join();if(early_error)std::rethrow_exception(early_error);
    need(native.controller_status().phase==KVControllerPhase::Startup&&!native.controller_status().disabled,
         "wrong-thread request revoked legal startup control");
    auto token=native.release_controller_for_worker(inventory);
    const auto released=native.controller_status();
    need(token&&released.phase==KVControllerPhase::Released&&released.disabled&&
         released.completed_cold_device_syncs==1,"cold startup release did not disable after one sync");
    reject([&]{(void)pool.plane_view(0);},"old controller retained Core view permission");
    reject([&]{(void)native.storage().device_plane_page_bases(0);},"old controller bypassed through direct storage view");
    reject([&]{auto freeze=native.freeze_allocator();},"old controller retained freeze permission");
    reject([&]{pool.zero_pages(world.ids,RankStreams{world.streams.s[0]});},"old controller retained transfer permission");
    // CPU-only RAII release stays legal and advances snapshots without invalidating
    // the independent handoff generation/token. Its page is absent from the Graph.
    need(world.extra.release(),"transit CPU RAII release failed");
    const auto after_cpu=native.controller_status();
    need(after_cpu.mutation_epoch!=released.mutation_epoch&&after_cpu.handoff_generation==released.handoff_generation&&
         token&&pool.allocated_pages()==LIVE,"CPU RAII epoch incorrectly invalidated handoff or allocator");
    std::promise<void> ready,resume;auto ready_future=ready.get_future();auto resumed=resume.get_future();
    std::exception_ptr worker_error,cleanup_error;
    std::thread worker([&,capability=std::move(token)]()mutable {
        bool signaled=false;
        try {
            check(hipSetDevice(0),"worker explicit device bind");int current=-1;
            check(hipGetDevice(&current),"worker actual device query");need(current==0,"worker did not bind device0");
            native.consume_controller_for_worker(std::move(capability),inventory);
            need(!capability&&native.controller_status().phase==KVControllerPhase::Worker,"capability not consumed once");
            reject([&]{auto again=native.release_controller_for_worker(inventory);},"second handoff accepted");
            reject([&]{native.consume_controller_for_worker(std::move(capability),inventory);},"consumed token reused");
            world.verify();ready.set_value();signaled=true;resumed.get();
            // The same Graph can also follow real physical relocation on the new
            // controller. The old host never captures/rebinds a page base.
            auto loan=native.storage().begin_loan(4*S,inventory);
            loan.wait_ready(world.streams.s[1]);
            check(hipMemsetAsync(loan.leased().data,0x5a,loan.leased().bytes,world.streams.s[1]),"worker borrower overwrite");
            check(hipStreamSynchronize(world.streams.s[1]),"worker borrower finish");world.verify();
            need(loan.close_after_streams(inventory),"worker typed return failed");world.verify();
            need(pool.lent_pages()==0&&pool.allocated_pages()==LIVE,"worker ledger after return wrong");
            native.retire_controller_for_teardown(inventory);
            const auto retired=native.controller_status();
            need(retired.phase==KVControllerPhase::Retired&&retired.disabled&&retired.completed_cold_device_syncs==2,
                 "worker did not permanently retire after explicit final sync");
        }catch(...) {
            worker_error=std::current_exception();
            // Ordinary successful retirement is the only positive shutdown claim.
            // Preserve both failures; never turn conservative destructor cleanup
            // into an accepted handoff/return/device-synchronization result.
            try{if(native.controller_status().phase==KVControllerPhase::Worker)
                native.retire_controller_for_teardown(inventory);}catch(...){cleanup_error=std::current_exception();}
            if(!signaled)try{ready.set_exception(worker_error);}catch(const std::future_error&){}
        }
    });
    JoinAndRelease joined(worker,resume);
    std::exception_ptr startup_notification_error;
    try{ready_future.get();}catch(...){startup_notification_error=std::current_exception();}
    if(!startup_notification_error) {
        reject([&]{(void)pool.plane_view(0);},"startup thread stole accepted worker view");
        reject([&]{auto freeze=native.freeze_allocator();},"startup thread stole accepted worker freeze");
        reject([&]{pool.zero_pages(world.ids,RankStreams{world.streams.s[0]});},"startup thread stole accepted worker transfer");
    }
    joined.resume();worker.join();
    if(worker_error||cleanup_error)
        throw std::runtime_error("worker failure: "+describe_failure(worker_error)+
                                 "; retirement cleanup failure: "+describe_failure(cleanup_error));
    if(startup_notification_error)std::rethrow_exception(startup_notification_error);
    need(native.controller_status().phase==KVControllerPhase::Retired&&pool.allocated_pages()==LIVE&&
         pool.lent_pages()==0&&pool.available_pages()==B-LIVE,"joined controller ledger/retirement wrong");
    reject([&]{(void)pool.plane_view(0);},"retired state automatically rebound to parent");
    need(world.replays==4,"same Graph before/after/during/after loan domain incomplete");
    return world.replays;
}

void preconditions_and_lost_tokens() {
    Streams streams;const std::array<hipStream_t,2> inventory{streams.s[0],streams.s[1]};
    {
        KVSegmentedNativePool native(config(),layout());
        auto freeze=native.freeze_allocator();
        reject([&]{auto invalid=native.release_controller_for_worker(inventory);},"live recursive Freeze crossed handoff");
        need(native.controller_status().phase==KVControllerPhase::Startup&&!native.controller_status().disabled&&
             native.controller_status().completed_cold_device_syncs==0,"live-Freeze refusal revoked the controller");
        freeze.reset();
    }
    {
        KVSegmentedNativePool native(config(),layout());
        auto ticket=native.storage().begin_loan(S,inventory);
        reject([&]{auto invalid=native.release_controller_for_worker(inventory);},"open grant/loan handoff accepted");
        need(native.controller_status().phase==KVControllerPhase::Startup&&!native.controller_status().disabled&&
             native.controller_status().completed_cold_device_syncs==0,"open-loan refusal revoked valid old controller");
        need(ticket.close_after_streams(inventory),"old controller could not close refused open loan");
    }
    {
        KVSegmentedNativePool native(config(),layout());
        {auto abandoned=native.storage().begin_loan(S,inventory);}
        need(native.storage().poisoned(),"poison negative control absent");
        reject([&]{auto invalid=native.release_controller_for_worker(inventory);},"poisoned controller handoff accepted");
        need(native.controller_status().phase==KVControllerPhase::Startup&&native.page_pool().lent_pages()==1,
             "poison refusal altered the excluded grant or controller phase");
    }
    {
        KVSegmentedNativePool native(config(),layout());hipGraph_t empty=nullptr;
        check(hipStreamBeginCapture(streams.s[0],hipStreamCaptureModeGlobal),"handoff capture negative begin");
        bool refused=false;std::exception_ptr error;
        try{auto invalid=native.release_controller_for_worker(inventory);}catch(const std::logic_error&){refused=true;}
        catch(...){error=std::current_exception();}
        const auto ended=hipStreamEndCapture(streams.s[0],&empty);
        if(empty)check(hipGraphDestroy(empty),"discard never-launched handoff negative capture");
        check(ended,"handoff negative capture finish");if(error)std::rethrow_exception(error);
        need(refused&&native.controller_status().phase==KVControllerPhase::Startup&&
             !native.controller_status().disabled,"capture refusal revoked valid startup controller");
    }
    {
        KVSegmentedNativePool a(config(),layout()),b(config(),layout());
        auto token_a=a.release_controller_for_worker(inventory);auto token_b=b.release_controller_for_worker(inventory);
        std::exception_ptr error;
        std::thread wrong([&]{try{check(hipSetDevice(0),"foreign token worker bind");
            reject([&]{b.consume_controller_for_worker(std::move(token_a),inventory);},"foreign owner token accepted");}
            catch(...){error=std::current_exception();}});
        wrong.join();if(error)std::rethrow_exception(error);
        need(token_a&&token_b&&a.controller_status().disabled&&b.controller_status().disabled&&
             b.controller_status().phase==KVControllerPhase::Released,"foreign token granted/reactivated a controller");
        // Both tokens are intentionally lost here; no destructor returns control.
    }
    std::optional<KVControllerHandoffToken> outliving;
    {
        KVSegmentedNativePool native(config(),layout());
        outliving.emplace(native.release_controller_for_worker(inventory));
        need(static_cast<bool>(*outliving),"fresh lifetime token invalid");
    }
    need(!static_cast<bool>(*outliving),"token retained a destroyed factory identity");
    outliving.reset(); // Safe: marker only, no dereference of destroyed Core/owner.
}
} // namespace

int main(int argc,char** argv) {
    if(argc==1) {
        std::cout<<"{\"status\":\"PLAN_ONLY_NOT_QUALIFIED\",\"main_explicit_hip_calls\":false,"
            "\"scope\":\"isolated once-only Core startup-to-worker handoff; no Engine hook\"}\n";
        return 0;
    }
    if(argc!=2||std::string_view(argv[1])!="--run")return 2;
    try {
        check(hipSetDevice(0),"handoff fixture startup device bind");
        const auto replays=successful_once_handoff();preconditions_and_lost_tokens();
        std::cout<<"{\"status\":\"FUNCTIONAL_CONTROLLER_HANDOFF_PASS\",\"same_graph_replays\":"<<replays
            <<",\"cpu_raii_during_transit\":true,\"wrong_owner_thread_and_reuse_rejected\":true,"
            "\"live_freeze_open_grant_poison_capture_rejected\":true,\"copy_compile_rejected\":true,"
            "\"explicit_device0_bind\":true,\"wrong_device_runtime_tested\":false,"
            "\"device_sync_failure_injected\":false,\"full_engine_supported\":false,\"production_gate\":false}\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<"HANDOFF_NOT_QUALIFIED: "<<error.what()<<'\n';return 1;}
}
