// SOURCE ONLY: a real new-Core physical acceptance fixture, not a provider/model.
// The helpers and byte observer compile against this target's full new Core ABI.
#ifndef NINFER_CAPACITY_ORIGINAL_CORE_FIXTURE
#error "Requires the new capacity view's exact original-pool fixture snapshot"
#endif
#define main unused_capacity_original_pool_fixture_main
#include NINFER_CAPACITY_ORIGINAL_CORE_FIXTURE
#undef main
#ifndef NINFER_AMD_SEGMENTED_CONTROLLER_HANDOFF_ABI
#error "Requires the full controller Core derivative, never old Core"
#endif
#include <optional>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace {
static_assert(std::is_same_v<decltype(std::declval<KVSegmentedStorage&>().try_begin_loan(
    std::uint64_t{},std::declval<std::span<const hipStream_t>>())),
    std::optional<KVSegmentedStorage::Transaction>>);

// Tags the entire valid allocation, including free cells, before live pages are
// uploaded through the real Core H2D path. No invalid or unallocated VA is used.
__global__ void capacity_tag_all(const std::uint64_t* table) {
    for(std::uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<P*B*WORDS;i+=gridDim.x*blockDim.x) {
        const auto cell=i/WORDS,word=i%WORDS;
        auto* destination=reinterpret_cast<std::uint16_t*>(table[cell]);
        destination[word]=static_cast<std::uint16_t>(cell*131U+word*17U+29U);
    }
}

struct CapacityState {
    KVSegmentedFrozenAllocation allocation;
    std::uint32_t allocated=0,reserved=0,lent=0,available=0,usable=0;
    const std::uint64_t* table_allocation=nullptr;
    std::vector<std::uint64_t> table_values,host_addresses;
    std::vector<std::byte> every_cell;
};

bool same_state(const CapacityState& a,const CapacityState& b) {
    return a.allocation.owner==b.allocation.owner&&
        a.allocation.mutation_epoch==b.allocation.mutation_epoch&&
        a.allocation.page_groups==b.allocation.page_groups&&
        a.allocation.available_pages==b.allocation.available_pages&&
        a.allocation.allocated==b.allocation.allocated&&a.allocation.free_pages==b.allocation.free_pages&&
        a.allocated==b.allocated&&a.reserved==b.reserved&&a.lent==b.lent&&
        a.available==b.available&&a.usable==b.usable&&
        a.table_allocation==b.table_allocation&&a.table_values==b.table_values&&
        a.host_addresses==b.host_addresses&&a.every_cell==b.every_cell;
}

class CapacityWorld {
public:
    CapacityWorld():native(config(),layout()) {
        auto& pool=native.page_pool();
        const auto budget=native.storage().footprint();
        need(budget.payload_device_bytes==P*B*S&&budget.pointer_table_device_bytes==P*B*sizeof(std::uint64_t)&&
             budget.pinned_pointer_shadow_bytes==P*B*sizeof(std::uint64_t)&&
             budget.private_streams==1&&budget.events==9,"capacity actual single-payload/table/event budget differs");
        auto entitlement=pool.reserve(LIVE);need(entitlement.has_value(),"capacity initial real reserve failed");
        leases.reserve(LIVE);pool.materialize(*entitlement,LIVE,leases);ids=handles(leases);
        table=pool.plane_view(0).page_bases;
        need(table&&pool.plane_view(1).page_bases==table+B,"capacity per-plane stable table layout");
        hipLaunchKernelGGL(capacity_tag_all,dim3(64),dim3(256),0,streams.s[0],table);
        check(hipGetLastError(),"capacity full allocation tags");
        host_layout=plan_host_kv_page_layout(pool.geometry());
        const std::array<HostKVPageLayout,1> supported{host_layout};
        arena=std::make_unique<HostKVArena>(LIVE*host_layout.page_stride,supported);
        input=arena->allocate(host_layout,LIVE);need(input.has_value(),"capacity real Host allocation failed");
        expected=pattern(9021);auto host=arena->writable_view(*input);
        pack(host.data(),host_layout,expected,host_layout.page_stride);
        pool.copy_from_host(arena->view(*input),ids,RankStreams{streams.s[0]});
        LayoutBuilder builder;const auto planned=plan_kv_execution_tables(builder,{LIVE,1});
        metadata=std::make_unique<DeviceArena>(builder.finish(256));
        need(metadata->capacity()<=4096,"capacity actual execution-table metadata exceeds fixture bound");
        tables=std::make_unique<KVExecutionTablePool>(DeviceSpan{metadata->base(),metadata->capacity()},planned,pool);
        // Metadata may initialize on default stream; explicitly establish its
        // completion before publishing through our nonblocking user stream.
        check(hipStreamSynchronize(nullptr),"capacity default metadata initialization drain");
        row.emplace(tables->acquire(0));tables->publish(row->handle(),0,ids,RankStreams{streams.s[0]});
        check(hipStreamSynchronize(streams.s[0]),"capacity original H2D/table publication complete");
        observed=DeviceBuffer(P*LIVE*S+2*GUARD);
        check(hipStreamBeginCapture(streams.s[0],hipStreamCaptureModeGlobal),"capacity observer capture");
        hipLaunchKernelGGL(observe,dim3(64),dim3(256),0,streams.s[0],table,
            static_cast<const std::int32_t*>(tables->row(row->handle()).data),
            reinterpret_cast<std::uint16_t*>(static_cast<std::byte*>(observed.p)+GUARD));
        check(hipGetLastError(),"capacity actual table observer capture");
        check(hipStreamEndCapture(streams.s[0],&graph.graph),"capacity capture end");
        check(hipGraphInstantiate(&graph.exec,graph.graph,nullptr,nullptr,0),"capacity observer instantiate");
        original_exec=graph.exec;
    }

    std::array<hipStream_t,2> inventory() const {return streams.s;}

    void verify_live() {
        need(table==native.page_pool().plane_view(0).page_bases&&graph.exec==original_exec,
             "capacity test replaced the stable pointer allocation or observer Graph");
        check(hipMemsetAsync(observed.p,0xa5,observed.bytes,streams.s[0]),"capacity fresh observer payload/guards");
        check(hipGraphLaunch(graph.exec,streams.s[0]),"capacity same original observer Graph replay");
        check(hipStreamSynchronize(streams.s[0]),"capacity observer completion");
        std::vector<std::byte> actual(observed.bytes);observed.copy_to_host(actual.data(),actual.size());
        need(std::all_of(actual.begin(),actual.begin()+GUARD,[](auto x){return x==std::byte{0xa5};})&&
             std::all_of(actual.end()-GUARD,actual.end(),[](auto x){return x==std::byte{0xa5};}),"capacity observer guards changed");
        need(std::equal(actual.begin()+GUARD,actual.end()-GUARD,
             reinterpret_cast<const std::byte*>(expected.data())),"capacity full live KV bytes changed");
        ++replays;
    }

    CapacityState snapshot_all() {
        for(auto stream:streams.s)check(hipStreamSynchronize(stream),"capacity snapshot registered users complete");
        auto frozen=native.freeze_allocator();CapacityState result;
        result.allocation=frozen->snapshot();auto& pool=native.page_pool();
        result.allocated=pool.allocated_pages();result.reserved=pool.reserved_pages();
        result.lent=pool.lent_pages();result.available=pool.available_pages();result.usable=pool.usable_pages();
        result.table_allocation=pool.plane_view(0).page_bases;
        result.table_values.resize(P*B);result.host_addresses.reserve(P*B);
        result.every_cell.resize(P*B*S);
        check(hipMemcpy(result.table_values.data(),result.table_allocation,P*B*sizeof(std::uint64_t),
                        hipMemcpyDeviceToHost),"capacity stable device table value readback");
        auto physical=result.table_values;std::sort(physical.begin(),physical.end());
        for(std::size_t cell=0;cell<physical.size();++cell)
            need(physical[cell]==physical.front()+cell*S,"capacity table does not cover every unique physical cell");
        for(std::uint32_t cell=0;cell<P*B;++cell) {
            const auto* address=native.storage().host_page_address(cell/B,cell%B);
            const auto bits=reinterpret_cast<std::uintptr_t>(address);
            result.host_addresses.push_back(bits);
            need(bits==result.table_values[cell],"capacity host mapping/device pointer table disagree");
            // Independent full-cell read by the actual table value; no Core
            // affine resolver or transfer implementation supplies the oracle.
            check(hipMemcpy(result.every_cell.data()+cell*S,reinterpret_cast<const void*>(bits),S,
                            hipMemcpyDeviceToHost),"capacity complete physical cell readback");
        }
        return result;
    }

    std::vector<std::uint64_t> table_values() const {
        std::vector<std::uint64_t> result(P*B);
        check(hipMemcpy(result.data(),table,P*B*sizeof(std::uint64_t),hipMemcpyDeviceToHost),
              "capacity placement-only readback");return result;
    }

    KVSegmentedNativePool native; // outlives every original lease and GPU/Host object
    Streams streams;
    std::vector<DeviceKVPageLease> leases;
    std::vector<DeviceKVPageHandle> ids;
    HostKVPageLayout host_layout;
    std::unique_ptr<HostKVArena> arena;
    std::optional<HostKVAllocation> input;
    std::unique_ptr<DeviceArena> metadata;
    std::unique_ptr<KVExecutionTablePool> tables;
    std::optional<KVExecutionRowLease> row;
    std::vector<std::uint16_t> expected;
    DeviceBuffer observed;
    Graph graph;
    const std::uint64_t* table=nullptr;
    hipGraphExec_t original_exec=nullptr;
    unsigned replays=0;
    DrainStreams drain_before_every_resource{streams}; // also runs on constructor exceptions
};

template<class Exception,class Fn>
void capacity_exact_exception(Fn&& fn,const char* message,const char* why) {
    bool refused=false;
    try{fn();}
    catch(const Exception& error) {
        need(typeid(error)==typeid(Exception),"capacity refusal had a different dynamic exception type");
        if(message)need(std::string_view(error.what())==message,"capacity legacy exception text changed");
        refused=true;
    }
    need(refused,why); // No exception or an empty optional must not silently pass.
}

unsigned capacity_refusal_then_real_grant() {
    CapacityWorld world;auto& native=world.native;auto& pool=native.page_pool();
    const auto users=world.inventory();world.verify_live();
    auto promised=pool.reserve(3);need(promised.has_value(),"capacity real pending reservation failed");
    need(pool.allocated_pages()==LIVE&&pool.reserved_pages()==3&&pool.available_pages()==1,
         "capacity fixture is not genuinely reservation-limited");
    const auto before=world.snapshot_all();
    for(unsigned attempt=0;attempt<2;++attempt) {
        auto loan=native.storage().try_begin_loan(4*S,users);
        need(!loan.has_value(),"capacity optional entry consumed promised pages");
        need(!native.storage().lease_open()&&!native.storage().poisoned(),"capacity refusal opened/poisoned storage");
        need(same_state(before,world.snapshot_all()),"capacity nullopt changed real epoch/ledger/reservation/mapping/table/bytes");
        world.verify_live();
    }
    capacity_exact_exception<std::runtime_error>([&]{auto wrong=native.storage().begin_loan(4*S,users);},
        "insufficient unpromised segmented KV capacity","legacy throwing capacity entry returned normally");
    need(same_state(before,world.snapshot_all()),"legacy capacity exception changed real state");
    world.verify_live();

    // Free genuine reservations through the original RAII API, not a synthetic
    // planner override. Enough capacity now means the original physical path.
    promised->clear();need(pool.available_pages()==4&&pool.reserved_pages()==0,"cleared reservation did not restore available quota");
    const auto ready=world.snapshot_all();
    auto admitted=native.storage().try_begin_loan(4*S,users);
    need(admitted&&admitted->open()&&native.storage().lease_open(),"available physical loan was refused");
    need(pool.lent_pages()==2&&pool.available_pages()==2&&pool.allocated_pages()==LIVE,
         "accepted optional result did not own exactly one two-ID real grant");
    need(admitted->copied_live_bytes()==4*S&&admitted->leased().bytes==4*S,"real optional relocation/capacity extent wrong");
    const auto moved=world.table_values();need(moved!=ready.table_values,"physical admission failed to change placement");
    admitted->wait_ready(world.streams.s[1]);
    check(hipMemsetAsync(admitted->leased().data,0xd7,admitted->leased().bytes,world.streams.s[1]),"capacity borrower full window poison");
    check(hipStreamSynchronize(world.streams.s[1]),"capacity borrower poison complete");
    world.verify_live();
    std::vector<std::byte> borrower(admitted->leased().bytes);
    check(hipMemcpy(borrower.data(),admitted->leased().data,borrower.size(),hipMemcpyDeviceToHost),"capacity borrower preservation");
    need(std::all_of(borrower.begin(),borrower.end(),[](auto x){return x==std::byte{0xd7};}),"table consumer damaged borrower window");
    need(admitted->close_after_streams(users),"optional true grant could not return after actual users");
    need(!admitted->open()&&!native.storage().lease_open()&&!native.storage().poisoned()&&
         pool.lent_pages()==0&&pool.available_pages()==4,"optional close leaked or double-returned original capacity");
    world.verify_live();
    const auto closed=world.snapshot_all();
    need(closed.allocation.mutation_epoch==ready.allocation.mutation_epoch+2&&closed.table_values==moved,
         "single lend/return epoch or retained physical placement wrong");
    need(admitted->close_after_streams(users),"idempotent closed ticket unexpectedly failed");
    need(same_state(closed,world.snapshot_all()),"second close returned IDs or changed epoch/table/bytes again");
    world.verify_live();
    return world.replays;
}

unsigned noncapacity_errors_are_not_empty() {
    CapacityWorld world;auto& storage=world.native.storage();const auto users=world.inventory();
    const auto before=world.snapshot_all();unsigned rejected=0;
    capacity_exact_exception<std::invalid_argument>([&]{auto invalid=storage.try_begin_loan(0,users);},nullptr,
        "zero size became empty/success");++rejected;
    capacity_exact_exception<std::invalid_argument>([&]{auto invalid=storage.try_begin_loan(4*S+1,users);},nullptr,
        "out-of-window size became empty/success");++rejected;
    capacity_exact_exception<std::invalid_argument>([&]{auto invalid=storage.try_begin_loan(S,{});},nullptr,
        "missing actual stream inventory became empty/success");++rejected;
    const std::array<hipStream_t,5> oversized{users[0],users[1],users[0],users[1],users[0]};
    capacity_exact_exception<std::invalid_argument>([&]{auto invalid=storage.try_begin_loan(S,oversized);},nullptr,
        "oversized legal-handle stream inventory became empty/success");++rejected;
    need(same_state(before,world.snapshot_all()),"noncapacity argument refusal changed full state");

    hipGraph_t empty=nullptr;std::exception_ptr capture_error;
    check(hipStreamBeginCapture(users[0],hipStreamCaptureModeGlobal),"capacity capture refusal begin");
    try{capacity_exact_exception<std::invalid_argument>([&]{auto invalid=storage.try_begin_loan(S,users);},nullptr,
        "actively capturing stream became capacity-empty/success");}
    catch(...){capture_error=std::current_exception();}
    const auto ended=hipStreamEndCapture(users[0],&empty);
    if(empty)check(hipGraphDestroy(empty),"capacity never-launched negative graph destroy");
    check(ended,"capacity negative capture end");if(capture_error)std::rethrow_exception(capture_error);++rejected;
    need(same_state(before,world.snapshot_all()),"capture refusal changed real state");world.verify_live();

    // Actual invalid construction, before payload allocation; not a made-up HIP
    // handle or injected driver error. An unbound real owner is never asked to freeze.
    auto invalid_cfg=config();invalid_cfg.borrower_alignment=3;
    capacity_exact_exception<std::invalid_argument>([&]{KVSegmentedPoolOwner owner(invalid_cfg);
        KVSegmentedStorage invalid(invalid_cfg,owner);},nullptr,"invalid class geometry was constructed");++rejected;
    need(same_state(before,world.snapshot_all()),"independent invalid construction altered the valid pool");
    return rejected;
}

void genuine_poison_is_not_capacity_refusal() {
    CapacityWorld world;auto& storage=world.native.storage();auto& pool=world.native.page_pool();
    const auto users=world.inventory();
    {auto abandoned=storage.try_begin_loan(S,users);need(abandoned&&abandoned->open(),"poison control never acquired a real grant");}
    need(storage.poisoned()&&storage.lease_open()&&pool.lent_pages()==1,"real abandonment did not poison/retain the real grant");
    const auto epoch=world.native.controller_status().mutation_epoch;
    const auto allocated=pool.allocated_pages(),reserved=pool.reserved_pages(),available=pool.available_pages();
    capacity_exact_exception<std::runtime_error>([&]{auto invalid=storage.try_begin_loan(S,users);},nullptr,
        "real poison became an empty optional");
    need(storage.poisoned()&&storage.lease_open()&&pool.lent_pages()==1&&pool.allocated_pages()==allocated&&
         pool.reserved_pages()==reserved&&pool.available_pages()==available&&
         world.native.controller_status().mutation_epoch==epoch,"poison refusal changed excluded capacity/epoch");
    // No return, retry, controller retirement success, or driver-fault recovery is
    // claimed for this deliberately poisoned instance. Normal teardown drains GPU
    // users and frees its private backing; logical IDs are not reissued.
}
} // namespace

int main(int argc,char** argv) {
    try {
        if(argc==1) {
            std::cout<<"{\"status\":\"PLAN_ONLY\",\"main_explicit_hip_calls\":false,"
                "\"scope\":\"real capacity Core ledger/table/full-byte fixture\","
                "\"pre_main_hip_registration_not_excluded\":true}\n";return 0;
        }
        need(argc==2&&std::string_view(argv[1])=="--run","usage: ninfer_amd_vision_segmented_capacity_test [--run]");
        check(hipSetDevice(0),"capacity real device bind");
        const auto replays=capacity_refusal_then_real_grant();
        const auto errors=noncapacity_errors_are_not_empty();genuine_poison_is_not_capacity_refusal();
        std::cout<<"{\"status\":\"REAL_CORE_CAPACITY_FIXTURE_PASSED\",\"capacity_empty_attempts\":2,"
            "\"legacy_capacity_exception_exact\":true,\"complete_all_cell_refusal_bytes\":"<<P*B*S<<","
            "\"noncapacity_refusals\":"<<errors+1<<",\"real_unique_grant\":true,"
            "\"copied_live_bytes\":"<<4*S<<",\"capacity_world_same_graph_replays\":"<<replays<<","
            "\"pending_ordering_qualified\":false,\"hip_fault_recovery_qualified\":false,"
            "\"Vision_or_Engine_connected\":false,\"production_gate_changed\":false,"
            "\"link_map_requires_external_gate\":true}\n";return 0;
    }catch(const std::exception& error){std::cerr<<"REAL_CORE_CAPACITY_FIXTURE_FAILED: "<<error.what()<<'\n';return 1;}
}
