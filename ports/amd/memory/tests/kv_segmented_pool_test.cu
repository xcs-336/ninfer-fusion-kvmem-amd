// SOURCE ONLY. Actual original DeviceKVPagePool allocator/transfers, isolated ABI.
// No replacement allocation model, attention math, model load or production gate.
#include "kv_segmented_pool_owner.h"
#include "core/host_kv_arena.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::amd::memory;
constexpr std::uint32_t P=2,B=8,LIVE=4,WORDS=256*64;
constexpr std::size_t S=WORDS*2,GUARD=256;
void need(bool value,const char* why){if(!value)throw std::runtime_error(why);}
void check(hipError_t status,const char* why){if(status!=hipSuccess)throw std::runtime_error(std::string(why)+": "+hipGetErrorString(status));}
template<class Fn>void reject(Fn&& fn,const char* why){bool rejected=false;try{fn();}catch(const std::logic_error&){rejected=true;}need(rejected,why);}
KVSegmentedStorage::Config config(){return {0,P,B,S,0,4*S,256,4};}
DeviceKVPagePoolLayout layout() {
    DeviceKVPagePoolSpec spec;spec.page_group_count=B;
    spec.geometry.page_tokens=64;spec.geometry.device_plane_order=PagedKVPlaneOrder::PageMajor;
    for(std::uint32_t p=0;p<P;++p)spec.geometry.planes.push_back({DType::BF16,256,1});
    LayoutBuilder builder;auto result=plan_device_kv_page_pool(builder,spec);
    need(builder.finish(256)==P*B*S,"original planned pool is not exactly one payload");return result;
}
struct Streams {
    std::array<hipStream_t,2> s{};
    Streams(){try{for(auto& stream:s)check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking),"Core test stream");}
        catch(...){for(auto stream:s)if(stream)(void)hipStreamDestroy(stream);throw;}}
    ~Streams(){for(auto stream:s)if(stream){(void)hipStreamSynchronize(stream);(void)hipStreamDestroy(stream);}}
};
struct Pinned {
    void* data=nullptr;std::size_t bytes=0;
    explicit Pinned(std::size_t count):bytes(count){check(hipHostMalloc(&data,bytes,hipHostMallocDefault),"Core records pinned allocation");}
    ~Pinned(){if(data){(void)hipDeviceSynchronize();(void)hipHostFree(data);}}
};
struct Graph {
    hipGraph_t graph=nullptr;hipGraphExec_t exec=nullptr;
    ~Graph(){(void)hipDeviceSynchronize();if(exec)(void)hipGraphExecDestroy(exec);if(graph)(void)hipGraphDestroy(graph);}
};
struct DrainStreams {
    Streams& streams;
    ~DrainStreams(){for(auto stream:streams.s)if(stream)(void)hipStreamSynchronize(stream);}
};
// Independent byte observer: reads true GPU table entries and original published IDs,
// never calls the Core address resolver/H2D/D2H implementation under test.
__global__ void observe(const std::uint64_t* pointers,const std::int32_t* ids,std::uint16_t* out) {
    constexpr std::uint32_t total=P*LIVE*WORDS;
    for(std::uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<total;i+=gridDim.x*blockDim.x) {
        const auto plane=i/(LIVE*WORDS),row=(i/WORDS)%LIVE,word=i%WORDS;
        const auto* page=reinterpret_cast<const std::uint16_t*>(pointers[plane*B+ids[row]]);
        out[i]=page[word];
    }
}
std::vector<std::uint16_t> pattern(unsigned seed) {
    std::mt19937 random(seed);std::vector<std::uint16_t> data(P*LIVE*WORDS);
    for(auto& value:data)value=static_cast<std::uint16_t>(random());return data;
}
void pack(std::byte* destination,const HostKVPageLayout& host,const std::vector<std::uint16_t>& expected,std::size_t pitch) {
    for(std::size_t row=0;row<LIVE;++row)for(std::size_t p=0;p<P;++p)
        std::copy_n(reinterpret_cast<const std::byte*>(expected.data()+(p*LIVE+row)*WORDS),S,
                    destination+row*pitch+host.planes[p].offset);
}
void unpack_check(const std::byte* source,const HostKVPageLayout& host,const std::vector<std::uint16_t>& expected,std::size_t pitch) {
    for(std::size_t row=0;row<LIVE;++row)for(std::size_t p=0;p<P;++p)
        need(std::equal(source+row*pitch+host.planes[p].offset,source+row*pitch+host.planes[p].offset+S,
              reinterpret_cast<const std::byte*>(expected.data()+(p*LIVE+row)*WORDS)),"Core host copy logical bytes differ");
}
std::vector<DeviceKVPageHandle> handles(const std::vector<DeviceKVPageLease>& leases) {
    std::vector<DeviceKVPageHandle> result;for(const auto& lease:leases)result.push_back(lease.handle());return result;
}

void allocator_contract() {
    auto cfg=config();auto planned=layout();KVSegmentedPoolOwner owner(cfg);
    reject([&]{auto invalid=owner.freeze();},"unbound owner freeze did not fail closed");
    // Declaration order is intentional: storage dies before original pool and owner.
    std::unique_ptr<DeviceKVPagePool> pool;
    KVSegmentedStorage storage(cfg,owner);
    const auto raw=storage.raw_payload_for_initialization();
    pool=std::make_unique<DeviceKVPagePool>(DeviceSpan{raw.data,static_cast<std::size_t>(raw.bytes)},planned);
    owner.bind(*pool,storage);
    const auto snapshot=[&]{auto frozen=owner.freeze();return frozen->snapshot();};
    auto reservation=pool->reserve(2);need(reservation.has_value(),"original reserve failed");
    const auto before=snapshot();std::vector<DeviceKVPageLease> pages;pages.reserve(2);
    pool->materialize(*reservation,2,pages);const auto after=snapshot();
    need(before.available_pages==after.available_pages&&before.mutation_epoch!=after.mutation_epoch&&
         before.allocated!=after.allocated,"reservation materialization failed unchanged-available epoch negative control");
    need(pool->allocated_pages()==2&&pool->reserved_pages()==0,"original materialize ledger wrong");
    const auto old=pages[0].handle();pages[0]=std::move(pages[1]);pages.pop_back();
    need(pool->allocated_pages()==1,"original lease move assignment did not release old group");
    reject([&]{const std::array<DeviceKVPageHandle,1> stale{old};pool->zero_pages(stale);},"old generation handle accepted after actual release");
    pool->dematerialize_one(*reservation,std::move(pages[0]));pages.clear();
    need(pool->allocated_pages()==0&&pool->reserved_pages()==1,"original single dematerialize ledger wrong");
    const auto before_single=snapshot();
    auto single=pool->materialize_one(*reservation,"segmented-Core-contract");
    need(single.valid()&&pool->allocated_pages()==1&&pool->reserved_pages()==0,"original single materialize failed");
    const auto after_single=snapshot();
    need(before_single.available_pages==after_single.available_pages&&
         before_single.mutation_epoch!=after_single.mutation_epoch,"single materialize missed unchanged-available epoch");

    // The real RAII release must not mutate the original allocator while Freeze lives.
    std::unique_ptr<KVSegmentedAllocatorFreeze> frozen=owner.freeze();
    const auto held_state=frozen->snapshot();
    std::promise<void> entered,done;auto entered_future=entered.get_future(),done_future=done.get_future();
    std::thread worker;
    try {
        worker=std::thread([page=std::move(single),&entered,&done]()mutable{entered.set_value();(void)page.release();done.set_value();});
        need(entered_future.wait_for(std::chrono::seconds(1))==std::future_status::ready,"release worker did not enter");
        need(done_future.wait_for(std::chrono::milliseconds(50))==std::future_status::timeout,
             "original lease release bypassed the internal Freeze mutex");
        const auto while_held=frozen->snapshot();
        need(while_held.allocated==held_state.allocated&&while_held.mutation_epoch==held_state.mutation_epoch,
             "original state/epoch changed inside held Freeze");
        frozen.reset();worker.join();
    }catch(...){frozen.reset();if(worker.joinable())worker.join();throw;}
    need(pool->allocated_pages()==0&&pool->available_pages()==B,"RAII release did not restore original capacity");
    pool->resize_reservation(*reservation,2);pool->materialize(*reservation,2,pages);
    const auto before_drop=snapshot();pool->dematerialize(*reservation,0,pages);const auto after_drop=snapshot();
    need(before_drop.available_pages==after_drop.available_pages&&before_drop.mutation_epoch!=after_drop.mutation_epoch&&
         pool->allocated_pages()==0&&pool->reserved_pages()==2,"batch drop missed actual release/entitlement epoch");
    reservation->clear();
    {auto r=pool->reserve(2),other=pool->reserve(1);need(r.has_value()&&other.has_value(),"temporary reservation failed");
        DeviceKVPageReservation moved=std::move(*r);moved=std::move(*other);
        need(pool->reserved_pages()==1,"reservation move assignment did not return previous entitlement");
        moved.clear();need(pool->reserved_pages()==0,"clear leaked reservation");
        pool->resize_reservation(moved,1);need(pool->reserved_pages()==1,"resize empty retained reservation failed");}
    need(pool->reserved_pages()==0&&pool->available_pages()==B,"reservation destructor leaked original capacity");
}

unsigned mapped_transfers() {
    auto planned=layout();KVSegmentedNativePool native(config(),planned);auto& pool=native.page_pool();
    auto entitlement=pool.reserve(LIVE);need(entitlement.has_value(),"Core transfer reservation failed");
    std::vector<DeviceKVPageLease> leases;leases.reserve(LIVE);pool.materialize(*entitlement,LIVE,leases);
    const auto ids=handles(leases);auto reserved=pool.reserve(1);need(reserved.has_value(),"promised capacity reservation failed");
    Streams streams;const RankStreams core_stream(streams.s[0]);
    reject([&]{(void)native.storage().raw_payload_for_initialization();},"bound startup span remained available");
    reject([&]{(void)pool.plane(0);},"bound affine plane escaped");
    reject([&]{(void)pool.free_runs();},"bound free-runs span escaped");
    reject([&]{(void)pool.plane_page_range(0,0,2);},"bound affine raw range escaped");
    reject([&]{pool.lend_pages(7,1);},"raw untyped lend bypassed capability");
    reject([&]{pool.return_pages(7,1);},"raw untyped return bypassed capability");

    const auto host_layout=plan_host_kv_page_layout(pool.geometry());
    const std::array<HostKVPageLayout,1> supported{host_layout};
    HostKVArena arena(2*LIVE*host_layout.page_stride,supported);
    auto input=arena.allocate(host_layout,LIVE),output=arena.allocate(host_layout,LIVE);
    need(input&&output,"original pinned HostKVArena allocation failed");
    DrainStreams drain_before_arena_release{streams};
    auto expected=pattern(101);auto writable=arena.writable_view(*input);
    pack(writable.data(),host_layout,expected,host_layout.page_stride);
    pool.copy_from_host(arena.view(*input),ids,core_stream);
    check(hipStreamSynchronize(streams.s[0]),"initial original plain H2D completion");

    LayoutBuilder table_builder;auto table_layout=plan_kv_execution_tables(table_builder,{LIVE,1});
    DeviceArena table_arena(table_builder.finish(256));
    KVExecutionTablePool tables(DeviceSpan{table_arena.base(),table_arena.capacity()},table_layout,pool);
    auto row=tables.acquire(0);tables.publish(row.handle(),0,ids,core_stream);
    check(hipStreamSynchronize(streams.s[0]),"actual Core table publish completion");
    const auto* table=pool.plane_view(0).page_bases;
    need(table&&pool.plane_view(1).page_bases==table+B,"real Core views do not expose stable per-plane tables");
    DeviceBuffer observed(P*LIVE*S+2*GUARD);Graph graph;
    check(hipStreamBeginCapture(streams.s[0],hipStreamCaptureModeGlobal),"Core observer capture");
    hipLaunchKernelGGL(observe,dim3(64),dim3(256),0,streams.s[0],table,
        static_cast<const std::int32_t*>(tables.row(row.handle()).data),
        reinterpret_cast<std::uint16_t*>(static_cast<std::byte*>(observed.p)+GUARD));
    check(hipGetLastError(),"Core observer capture kernel");
    check(hipStreamEndCapture(streams.s[0],&graph.graph),"Core observer end capture");
    check(hipGraphInstantiate(&graph.exec,graph.graph,nullptr,nullptr,0),"Core observer instantiate");
    unsigned replays=0;
    const auto verify=[&] {
        check(hipMemsetAsync(observed.p,0xa5,observed.bytes,streams.s[0]),"Core output guards");
        check(hipGraphLaunch(graph.exec,streams.s[0]),"same original-pool observer replay");
        check(hipStreamSynchronize(streams.s[0]),"Core output completion");
        std::vector<std::byte> actual(observed.bytes);observed.copy_to_host(actual.data(),actual.size());
        need(std::all_of(actual.begin(),actual.begin()+GUARD,[](auto b){return b==std::byte{0xa5};})&&
             std::all_of(actual.end()-GUARD,actual.end(),[](auto b){return b==std::byte{0xa5};}),"Core observer guard changed");
        need(std::equal(actual.begin()+GUARD,actual.end()-GUARD,reinterpret_cast<const std::byte*>(expected.data())),
             "actual original-Core transfer disagrees with independent full logical-byte oracle");++replays;
    };
    const auto plain_readback=[&] {
        auto destination=arena.writable_view(*output);
        std::fill_n(destination.data(),LIVE*host_layout.page_stride,std::byte{0xcc});
        pool.copy_to_host(ids,destination,core_stream);
        check(hipStreamSynchronize(streams.s[0]),"original plain D2H completion");
        unpack_check(destination.data(),host_layout,expected,host_layout.page_stride);
    };
    verify();plain_readback();need(pool.contiguous_run_count(ids)==1,"initial true physical run should be contiguous");
    // Reject host-resolved addresses in capture without invalid GPU work or launch.
    hipGraph_t forbidden_graph=nullptr;
    check(hipStreamBeginCapture(streams.s[0],hipStreamCaptureModeGlobal),"Core transfer rejection capture");
    bool capture_rejected=false;
    std::exception_ptr capture_error;
    try{pool.zero_pages(ids,core_stream);}catch(const std::logic_error&){capture_rejected=true;}
    catch(...){capture_error=std::current_exception();}
    const auto ended=hipStreamEndCapture(streams.s[0],&forbidden_graph);
    if(forbidden_graph)check(hipGraphDestroy(forbidden_graph),"discard unlaunched rejection graph");
    check(ended,"Core rejection capture completion");if(capture_error)std::rethrow_exception(capture_error);
    need(capture_rejected,"raw-address Core transfer was capture-admitted");
    const std::array<hipStream_t,2> prior{streams.s[0],streams.s[1]};
    auto loan=native.storage().begin_loan(4*S,prior);
    need(pool.lent_pages()==2&&pool.usable_pages()==B-2&&pool.available_pages()==1,
         "typed original grant consumed promised/incorrect IDs");
    need(loan.copied_live_bytes()==4*S,"original four live source slices were not moved");
    reject([&]{pool.return_pages(6,2);},"raw return stole IDs from an active unique grant");
    need(pool.lent_pages()==2,"rejected raw return mutated actual grant capacity");
    loan.wait_ready(streams.s[1]);check(hipMemsetAsync(loan.leased().data,0x5a,loan.leased().bytes,streams.s[1]),"borrowed window poison");
    check(hipStreamSynchronize(streams.s[1]),"borrower poison completion");
    need(pool.contiguous_run_count(ids)==2,"mapped physical fragment/coalescing boundary was not exercised");
    verify();plain_readback();
    const auto preserve_loan=[&] {
        std::vector<std::byte> actual(static_cast<std::size_t>(loan.leased().bytes));
        check(hipMemcpy(actual.data(),loan.leased().data,actual.size(),hipMemcpyDeviceToHost),"raw borrower preservation readback");
        need(std::all_of(actual.begin(),actual.end(),[](auto b){return b==std::byte{0x5a};}),"Core writer damaged borrowed physical bytes");
    };
    pool.zero_pages(ids,core_stream);std::fill(expected.begin(),expected.end(),0);verify();plain_readback();preserve_loan();
    expected=pattern(202);pack(writable.data(),host_layout,expected,host_layout.page_stride);
    pool.copy_from_host(arena.view(*input),ids,core_stream);verify();plain_readback();preserve_loan();
    pool.copy_page(ids[0],ids[2],core_stream);
    for(std::uint32_t p=0;p<P;++p)std::copy_n(expected.begin()+(p*LIVE)*WORDS,WORDS,expected.begin()+(p*LIVE+2)*WORDS);
    verify();plain_readback();preserve_loan();

    // Constant pitched records in one pinned allocation, spanning both actual device
    // contiguous pairs and their physical gap. A second pass has separate allocations.
    const auto pitch=host_layout.page_stride+256;
    Pinned records(LIVE*pitch+2*GUARD);
    auto* record_base=static_cast<std::byte*>(records.data)+GUARD;
    std::array<std::byte*,LIVE> mutable_records{};
    std::array<const std::byte*,LIVE> const_records{};
    std::array<std::uint32_t,LIVE> record_groups{};
    for(std::uint32_t i=0;i<LIVE;++i){mutable_records[i]=record_base+i*pitch;const_records[i]=mutable_records[i];}
    std::fill_n(static_cast<std::byte*>(records.data),records.bytes,std::byte{0xcc});
    expected=pattern(303);pack(record_base,host_layout,expected,pitch);
    pool.copy_from_host_records(const_records,record_groups,ids,host_layout,core_stream);
    verify();preserve_loan();
    std::fill_n(static_cast<std::byte*>(records.data),records.bytes,std::byte{0xcc});
    pool.copy_to_host_records(ids,mutable_records,record_groups,host_layout,core_stream);
    check(hipStreamSynchronize(streams.s[0]),"original pitched records D2H completion");
    unpack_check(record_base,host_layout,expected,pitch);
    for(std::size_t byte=0;byte<records.bytes;++byte) {
        bool payload=false;
        for(std::size_t r=0;r<LIVE;++r)for(const auto& plane:host_layout.planes) {
            const auto first=GUARD+r*pitch+plane.offset;
            if(byte>=first&&byte<first+plane.page_payload_bytes)payload=true;
        }
        if(!payload)need(static_cast<std::byte*>(records.data)[byte]==std::byte{0xcc},"record D2H overwrote host gap/guard");
    }
    // Restricted H2D plane range must leave every other plane unchanged.
    auto changed=pattern(404);pack(record_base,host_layout,changed,pitch);
    pool.copy_from_host_records(const_records,record_groups,ids,host_layout,1,2,core_stream);
    std::copy(changed.begin()+LIVE*WORDS,changed.end(),expected.begin()+LIVE*WORDS);
    verify();plain_readback();preserve_loan();
    // Different allocations/groups force host-record splitting independently of ID order.
    std::array<std::unique_ptr<Pinned>,LIVE> separate;
    expected=pattern(505);
    for(std::uint32_t i=0;i<LIVE;++i) {
        separate[i]=std::make_unique<Pinned>(host_layout.page_stride+2*GUARD);
        std::fill_n(static_cast<std::byte*>(separate[i]->data),separate[i]->bytes,std::byte{0xdd});
        mutable_records[i]=static_cast<std::byte*>(separate[i]->data)+GUARD;
        const_records[i]=mutable_records[i];record_groups[i]=i+1;
        for(std::size_t p=0;p<P;++p)std::copy_n(reinterpret_cast<const std::byte*>(expected.data()+(p*LIVE+i)*WORDS),S,
            mutable_records[i]+host_layout.planes[p].offset);
    }
    pool.copy_from_host_records(const_records,record_groups,ids,host_layout,core_stream);verify();preserve_loan();
    for(const auto& allocation:separate)
        std::fill_n(static_cast<std::byte*>(allocation->data),allocation->bytes,std::byte{0xdd});
    pool.copy_to_host_records(ids,mutable_records,record_groups,host_layout,core_stream);
    check(hipStreamSynchronize(streams.s[0]),"separate host record completion");
    for(std::size_t r=0;r<LIVE;++r)for(std::size_t p=0;p<P;++p)
        need(std::equal(mutable_records[r]+host_layout.planes[p].offset,mutable_records[r]+host_layout.planes[p].offset+S,
             reinterpret_cast<const std::byte*>(expected.data()+(p*LIVE+r)*WORDS)),"separate pinned record bytes wrong");
    for(const auto& allocation:separate) {
        const auto* data=static_cast<const std::byte*>(allocation->data);
        need(std::all_of(data,data+GUARD,[](auto b){return b==std::byte{0xdd};})&&
             std::all_of(data+allocation->bytes-GUARD,data+allocation->bytes,[](auto b){return b==std::byte{0xdd};}),
             "separate record D2H overwrote allocation guards");
    }
    need(loan.close_after_streams(prior)&&pool.lent_pages()==0&&pool.available_pages()==B-LIVE-1,"typed original IDs were not returned");
    verify();plain_readback();need(replays==9,"Core branches did not all reach independent observer");
    return replays;
}

void abandoned_grant() {
    auto planned=layout();KVSegmentedNativePool native(config(),planned);auto& pool=native.page_pool();
    {const std::array<hipStream_t,1> prior{nullptr};auto ticket=native.storage().begin_loan(S,prior);}
    need(native.storage().poisoned()&&pool.lent_pages()==1&&pool.available_pages()==B-1,"abandoned original grant released IDs");
    auto reservation=pool.reserve(B-1);need(reservation.has_value(),"remaining original capacity unavailable");
    std::vector<DeviceKVPageLease> live;live.reserve(B-1);pool.materialize(*reservation,B-1,live);
    need(pool.available_pages()==0&&pool.allocated_pages()==B-1,"excluded IDs leaked into original allocation");
    need(!pool.reserve(1).has_value(),"original allocator could promise abandoned IDs");
    LayoutBuilder builder;auto table_layout=plan_kv_execution_tables(builder,{B-1,1});
    DeviceArena metadata(builder.finish(256));
    KVExecutionTablePool tables(DeviceSpan{metadata.base(),metadata.capacity()},table_layout,pool);
    auto row=tables.acquire(0);tables.publish(row.handle(),0,live);
    check(hipStreamSynchronize(nullptr),"abandoned-grant original table publication");
    std::array<std::int32_t,B-1> actual{};
    check(hipMemcpy(actual.data(),tables.row(row.handle()).data,sizeof(actual),hipMemcpyDeviceToHost),"actual remaining ID readback");
    std::sort(actual.begin(),actual.end());
    for(std::size_t i=0;i<actual.size();++i)need(actual[i]==static_cast<std::int32_t>(i),"original allocator reused excluded highest free grant ID");
}
} // namespace

int main(int argc,char** argv) {
    try {
        if(argc!=2||std::string_view(argv[1])!="--run") {
            need(argc==1,"usage: ninfer_amd_segmented_pool_test [--run]");
            std::cout<<"{\"status\":\"PLAN_ONLY\",\"scope\":\"isolated actual Core allocator/transfers\",\"hip_called\":false}\n";return 0;
        }
        check(hipSetDevice(0),"realpool device");allocator_contract();const auto replays=mapped_transfers();abandoned_grant();
        std::cout<<"{\"status\":\"ORIGINAL_CORE_SEGMENTED_POOL_CHECKS_PASSED\",\"original_allocator\":true,"
            "\"full_byte_oracle\":true,\"fragmented_plain_and_record_transfers\":true,\"epoch_negative_control\":true,"
            "\"observer_captures\":1,\"observer_same_instance_replays\":"<<replays<<","
            "\"production_gate_changed\":false,\"attention_math_qualified\":false,\"link_map_requires_external_gate\":true}\n";return 0;
    }catch(const std::exception& error){std::cerr<<"ORIGINAL_CORE_SEGMENTED_POOL_CHECKS_FAILED: "<<error.what()<<'\n';return 1;}
}
