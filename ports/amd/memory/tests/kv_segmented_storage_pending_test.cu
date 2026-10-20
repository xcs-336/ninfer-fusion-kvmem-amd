// SOURCE ONLY / NOT BUILT / NOT EXECUTED. New v2 pending-ordering candidate.
// Forked from frozen v1 SHA256 18d027342ed4db7983ed836830b55041c47536f7a2eb43f34634afaf782c0de9.
// The consumer resolves stable page pointers, not existing affine Attention. No model,
// quantizer, codec, illegal GPU address or driver-failure injection is involved.
// Independent events must be NotReady immediately before begin/close, then Success
// before exposure/ledger release. CPU callbacks never call HIP or wait for GPU work.
// Concurrent allocation/free/generation mutations remain outside this fixture.
#include "kv_segmented_storage.h"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {
using namespace ninfer::amd::memory;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void checked(hipError_t value,const char* where) {
    if(value!=hipSuccess)throw std::runtime_error(std::string(where)+": "+hipGetErrorString(value));
}

// Real count-reservation allocator used ONLY by this isolated Core test. Its mutex is
// the actual exclusion used by every mutation here. It is not an original-pool adapter.
class TestOwner final:public KVSegmentedAllocatorOwner {
    class Grant final:public KVSegmentedLentPages {
    public:
        Grant(TestOwner& owner,std::uint64_t nonce,std::span<const std::uint32_t> pages)
            :actor(owner),nonce(nonce),ids(pages.begin(),pages.end()){}
        const KVSegmentedAllocatorOwner& owner()const noexcept override{return actor;}
        std::span<const std::uint32_t> pages()const noexcept override{return ids;}
        TestOwner& actor;std::uint64_t nonce;std::vector<std::uint32_t> ids;
        // Destructor intentionally does not release anything.
    };
    class Freeze final:public KVSegmentedAllocatorFreeze {
    public:
        explicit Freeze(TestOwner& owner):actor(owner),lock(owner.mutex){}
        const KVSegmentedAllocatorOwner& owner()const noexcept override{return actor;}
        KVSegmentedFrozenAllocation snapshot()const override {
            std::vector<std::uint32_t> free;
            for(std::uint32_t i=0;i<actor.live.size();++i)
                if(!actor.live[i]&&!actor.lent[i])free.push_back(i);
            return {&actor,actor.epoch,static_cast<std::uint32_t>(actor.live.size()),
                    actor.live,std::move(free),actor.available_unlocked()};
        }
        std::unique_ptr<KVSegmentedLentPages> lend_selected(std::span<const std::uint32_t> ids)override {
            require(!actor.active_nonce&&!ids.empty()&&ids.size()<=actor.available_unlocked(),
                    "test allocator cannot promise those loan IDs");
            require(actor.epoch!=std::numeric_limits<std::uint64_t>::max()&&
                    actor.next_nonce!=std::numeric_limits<std::uint64_t>::max(),"test owner epoch overflow");
            std::uint32_t previous=0;bool first=true;
            for(auto id:ids){require(id<actor.live.size()&&!actor.live[id]&&!actor.lent[id]&&
                (first||previous<id),"test selected IDs not genuinely free/distinct");previous=id;first=false;}
            // Allocate the complete opaque capability BEFORE changing any capacity.
            auto grant=std::make_unique<Grant>(actor,actor.next_nonce,ids);
            actor.active_nonce=actor.next_nonce++;
            for(auto id:ids)actor.lent[id]=1;
            ++actor.epoch;++actor.lend_calls;
            return grant;
        }
        void return_selected(KVSegmentedLentPages& opaque)override {
            auto* grant=dynamic_cast<Grant*>(&opaque);
            require(grant && &grant->actor==&actor && grant->nonce==actor.active_nonce,
                    "test return has wrong owner/grant nonce");
            for(auto id:grant->ids)require(id<actor.lent.size()&&actor.lent[id]&&!actor.live[id],
                                          "test returned ID is not excluded/free");
            ++actor.return_attempts;
            if(actor.fail_return){actor.fail_return=false;throw std::runtime_error("intentional host return refusal");}
            // Independent witness BEFORE changing the actual capacity ledger. These
            // are test-owned events recorded after borrower writes, not storage's fences.
            for(std::uint32_t i=0;i<actor.return_event_count;++i) {
                checked(hipEventQuery(actor.return_events[i]),"v2 owner must not return pending borrower IDs");
                ++actor.return_completion_checks;
            }
            require(actor.epoch!=std::numeric_limits<std::uint64_t>::max(),"test owner epoch overflow");
            for(auto id:grant->ids)actor.lent[id]=0;
            actor.active_nonce=0;++actor.epoch;++actor.return_successes;
        }
    private:TestOwner& actor;std::unique_lock<std::mutex> lock;
    };
public:
    explicit TestOwner(std::uint32_t groups,std::span<const std::uint32_t> initial_live,
                       std::uint32_t reservation=0)
        :live(groups,0),lent(groups,0),generations(groups,17),reserved(reservation) {
        for(auto id:initial_live){require(id<groups&&!live[id],"test initial allocation invalid");live[id]=1;}
        require(reserved<=groups-initial_live.size(),"test reservation exceeds capacity");
    }
    std::unique_ptr<KVSegmentedAllocatorFreeze> freeze()override{return std::make_unique<Freeze>(*this);}
    std::uint32_t available(){std::lock_guard lock(mutex);return available_unlocked();}
    std::uint32_t lent_count(){std::lock_guard lock(mutex);return std::accumulate(lent.begin(),lent.end(),0U);}
    std::vector<std::uint8_t> mask(){std::lock_guard lock(mutex);return live;}
    std::vector<std::uint32_t> generation_copy(){std::lock_guard lock(mutex);return generations;}
    void refuse_next_return(){std::lock_guard lock(mutex);fail_return=true;}
    std::uint64_t successful_returns(){std::lock_guard lock(mutex);return return_successes;}
    void require_completion_at_return(std::span<const hipEvent_t> events) {
        std::lock_guard lock(mutex);
        require(events.size()<=return_events.size(),"too many independent return events");
        return_event_count=static_cast<std::uint32_t>(events.size());
        std::copy(events.begin(),events.end(),return_events.begin());
    }
    std::uint64_t completion_checks(){std::lock_guard lock(mutex);return return_completion_checks;}
private:
    std::uint32_t available_unlocked()const {
        const auto allocated=std::accumulate(live.begin(),live.end(),0U);
        const auto borrowed=std::accumulate(lent.begin(),lent.end(),0U);
        require(allocated+borrowed+reserved<=live.size(),"test owner overcommitted");
        return static_cast<std::uint32_t>(live.size())-allocated-borrowed-reserved;
    }
    std::mutex mutex;
    std::vector<std::uint8_t> live,lent;
    std::vector<std::uint32_t> generations;
    std::uint32_t reserved=0;
    std::uint64_t epoch=1,next_nonce=1,active_nonce=0,lend_calls=0,return_attempts=0,return_successes=0;
    bool fail_return=false;
    std::array<hipEvent_t,2> return_events{};
    std::uint32_t return_event_count=0;
    std::uint64_t return_completion_checks=0;
};

class Events {
public:
    Events() {
        try {for(auto& event:events)checked(hipEventCreateWithFlags(&event,hipEventDisableTiming),"v2 independent event");}
        catch(...) {for(auto event:events)if(event)(void)hipEventDestroy(event);throw;}
    }
    ~Events(){for(auto event:events)if(event){(void)hipEventSynchronize(event);(void)hipEventDestroy(event);}}
    std::array<hipEvent_t,4> events{};
};

// Finite CPU-only callback gate. It never polls a device, calls HIP, or waits for
// subsequently enqueued work. Its helper releases independently after a fixed hold;
// a 2-second callback fallback also prevents an unarmed/failed host path hanging.
class CallbackGate {
public:
    explicit CallbackGate(hipStream_t stream):stream(stream) {}
    void enqueue() {
        require(!submitted,"v2 callback gate submitted twice");
        submitted=true; // cleanup drains even if an API reports an uncertain error
        checked(hipLaunchHostFunc(stream,&CallbackGate::callback,this),"v2 enqueue bounded CPU gate");
    }
    ~CallbackGate() {
        release_now();
        if(helper.joinable())helper.join();
        if(submitted)(void)hipStreamSynchronize(stream); // userdata outlives callback
    }
    CallbackGate(const CallbackGate&)=delete;
    void arm(std::chrono::milliseconds hold) {
        require(!helper.joinable(),"v2 callback gate armed twice");
        helper=std::thread([this,hold] {
            std::unique_lock lock(mutex);
            changed.wait_for(lock,hold,[this]{return released;});
            released=true;changed.notify_all();
        });
    }
    void verify_completed() {
        std::lock_guard lock(mutex);
        require(entered&&exited&&!fallback_timeout,"v2 CPU gate failed or reached timeout; ordering not admitted");
    }
private:
    static void callback(void* opaque) noexcept {
        auto& gate=*static_cast<CallbackGate*>(opaque);
        try {
            std::unique_lock lock(gate.mutex);
            gate.entered=true;gate.changed.notify_all();
            if(!gate.changed.wait_for(lock,std::chrono::seconds(2),[&gate]{return gate.released;})) {
                gate.fallback_timeout=true;gate.released=true;
            }
            gate.exited=true;gate.changed.notify_all();
        }catch(...) {
            // A callback may not propagate an exception through the runtime. There are
            // no throwing user actions here; a host synchronization failure is fatal.
            std::terminate();
        }
    }
    void release_now() noexcept {
        try {std::lock_guard lock(mutex);released=true;changed.notify_all();}
        catch(...) {std::terminate();}
    }
    hipStream_t stream=nullptr;
    bool submitted=false,entered=false,exited=false,released=false,fallback_timeout=false;
    std::mutex mutex;std::condition_variable changed;std::thread helper;
};

void require_pending(hipEvent_t event) {
    require(hipEventQuery(event)==hipErrorNotReady,"v2 event was not actually pending at boundary; not admitted");
}

class Buffer {
public:
    explicit Buffer(std::size_t bytes):bytes(bytes){checked(hipMalloc(&data,bytes),"test buffer allocation");}
    ~Buffer(){if(data){(void)hipDeviceSynchronize();(void)hipFree(data);}}
    Buffer(const Buffer&)=delete;
    void* data=nullptr;std::size_t bytes=0;
};
class Streams {
public:
    Streams(){for(auto& s:streams)checked(hipStreamCreateWithFlags(&s,hipStreamNonBlocking),"test user stream");}
    ~Streams(){(void)hipDeviceSynchronize();for(auto s:streams)if(s)(void)hipStreamDestroy(s);}
    std::array<hipStream_t,5> streams{};
};

__global__ void read_live(const std::uint64_t* table,const std::uint8_t* allocated,
                          std::uint16_t* output,std::uint32_t cells,std::uint32_t groups,
                          std::uint32_t words_per_slot) {
    const std::uint64_t total=static_cast<std::uint64_t>(cells)*words_per_slot;
    for(std::uint64_t index=blockIdx.x*blockDim.x+threadIdx.x;index<total;
        index+=static_cast<std::uint64_t>(gridDim.x)*blockDim.x) {
        const auto cell=static_cast<std::uint32_t>(index/words_per_slot);
        if(!allocated[cell%groups]){output[index]=0;continue;}
        const auto* page=reinterpret_cast<const std::uint16_t*>(table[cell]);
        output[index]=page[index%words_per_slot];
    }
}
// Prior writers bind exact raw addresses BEFORE the loan. They cannot follow a newly
// published pointer table and accidentally hide a missing prior-user ordering edge.
__global__ void xor_raw_four(std::uint16_t* page0,std::uint16_t* page1,
                             std::uint16_t* page2,std::uint16_t* page3,
                             std::uint32_t words,std::uint16_t bits) {
    const std::uint64_t total=static_cast<std::uint64_t>(4)*words;
    for(std::uint64_t i=blockIdx.x*blockDim.x+threadIdx.x;i<total;
        i+=static_cast<std::uint64_t>(gridDim.x)*blockDim.x) {
        const auto plane=static_cast<std::uint32_t>(i/words);
        auto* page=plane==0?page0:plane==1?page1:plane==2?page2:page3;
        page[i%words]^=bits;
    }
}
__global__ void poison_bytes(std::uint8_t* data,std::uint64_t count,std::uint8_t bits) {
    for(std::uint64_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;
        i+=static_cast<std::uint64_t>(gridDim.x)*blockDim.x)data[i]=bits;
}

class Graph {
public:
    ~Graph(){(void)hipDeviceSynchronize();if(exec)(void)hipGraphExecDestroy(exec);if(graph)(void)hipGraphDestroy(graph);}
    void capture(hipStream_t stream,const std::uint64_t* table,const std::uint8_t* mask,
                 std::uint16_t* out,std::uint32_t cells,std::uint32_t groups,std::uint32_t words) {
        checked(hipStreamBeginCapture(stream,hipStreamCaptureModeGlobal),"test real begin capture");
        hipLaunchKernelGGL(read_live,dim3(128),dim3(256),0,stream,table,mask,out,cells,groups,words);
        checked(hipGetLastError(),"test captured page-table consumer submission");
        checked(hipStreamEndCapture(stream,&graph),"test real end capture");
        checked(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0),"test real graph instantiate");
        ++captures;
    }
    void launch(hipStream_t stream) {
        checked(hipGraphLaunch(exec,stream),"test same graph instance replay");++replays;
    }
    hipGraph_t graph=nullptr;hipGraphExec_t exec=nullptr;
    unsigned captures=0,replays=0;
};

std::vector<std::uint16_t> tags(std::uint32_t cells,std::uint32_t words) {
    std::mt19937 generator(20261021);
    std::vector<std::uint16_t> result(static_cast<std::size_t>(cells)*words);
    for(auto& bits:result)bits=static_cast<std::uint16_t>(generator());
    // Explicit distinct cell headers supplement full arbitrary-bit payload preservation.
    for(std::uint32_t cell=0;cell<cells;++cell){result[cell*words]=static_cast<std::uint16_t>(cell);
        result[cell*words+1]=static_cast<std::uint16_t>(~cell);}
    return result;
}
void upload_initial(KVSegmentedStorage& storage,const std::vector<std::uint16_t>& expected) {
    const auto backing=storage.raw_payload_for_initialization();
    require(backing.bytes==expected.size()*2,"test input geometry mismatch");
    checked(hipMemcpy(backing.data,expected.data(),static_cast<std::size_t>(backing.bytes),hipMemcpyHostToDevice),
            "test initial tagged payload upload");
    checked(hipStreamSynchronize(nullptr),"test initial upload/default completion");
}
std::vector<std::uint64_t> read_table(const KVSegmentedStorage& storage,std::uint32_t cells) {
    std::vector<std::uint64_t> table(cells);
    checked(hipMemcpy(table.data(),storage.device_plane_page_bases(0),cells*sizeof(std::uint64_t),
                      hipMemcpyDeviceToHost),"test actual pointer-table readback");
    return table;
}
void compare_output(Buffer& guarded,const std::vector<std::uint16_t>& expected,
                    const std::vector<std::uint8_t>& mask,std::uint32_t groups,
                    std::uint32_t words,hipStream_t stream) {
    checked(hipStreamSynchronize(stream),"test real graph completion");
    std::vector<std::uint8_t> readback(guarded.bytes);
    checked(hipMemcpy(readback.data(),guarded.data,guarded.bytes,hipMemcpyDeviceToHost),"test complete output readback");
    constexpr std::size_t guard=256;
    require(std::all_of(readback.begin(),readback.begin()+guard,[](auto b){return b==0xa5;})&&
            std::all_of(readback.end()-guard,readback.end(),[](auto b){return b==0xa5;}),"test output guard modified");
    for(std::size_t index=0;index<expected.size();++index) {
        const auto actual=static_cast<std::uint16_t>(readback[guard+2*index]|
                         (static_cast<std::uint16_t>(readback[guard+2*index+1])<<8));
        const auto cell=index/words;
        require(actual==(mask[cell%groups]?expected[index]:0),"actual graph consumer read wrong cell/byte");
    }
}
void inspect_table(const std::vector<std::uint64_t>& table,KVSegmentedSpan backing,std::uint64_t slot) {
    const auto begin=reinterpret_cast<std::uintptr_t>(backing.data);
    std::vector<std::uint64_t> order;
    for(auto pointer:table){require(pointer>=begin&&pointer-begin<backing.bytes&&
        (pointer-begin)%slot==0,"actual pointer table points outside/misaligned");order.push_back((pointer-begin)/slot);}
    std::sort(order.begin(),order.end());
    for(std::size_t i=0;i<order.size();++i)require(order[i]==i,"actual pointer table aliases/loses a slot");
}

struct HappyEvidence {
    unsigned captures=0,replays=0;
    std::uint64_t payload=0,table=0,pinned=0,maps=0,first_copy=0;
    unsigned prior_not_ready=0,borrower_not_ready=0,owner_completion_checks=0;
};
HappyEvidence happy_case() {
    constexpr std::uint32_t P=4,B=8,words=256*64,C=P*B;
    constexpr std::uint64_t S=words*2; // actual BF16 D256/Hkv1 64-token plane slice
    const std::array<std::uint32_t,2> live{0,1};
    TestOwner owner(B,live,1);const auto initial_available=owner.available();
    const auto generations=owner.generation_copy();
    Streams users;
    KVSegmentedStorage storage({0,P,B,S,0,4*S,256,8},owner);
    const auto budget=storage.footprint();
    require(budget.payload_device_bytes==C*S&&budget.pointer_table_device_bytes==8*C&&
            budget.pinned_pointer_shadow_bytes==8*C&&budget.current_and_next_slot_maps_host_bytes>=8*C&&
            budget.private_streams==1&&budget.events==17,"actual storage request budget differs");
    auto expected=tags(C,words);upload_initial(storage,expected);
    const auto backing=storage.raw_payload_for_initialization();
    const auto* stable_table=storage.device_plane_page_bases(0);
    const auto original_table=read_table(storage,C);
    const auto mask=owner.mask();Buffer device_mask(B),output(expected.size()*2+512);
    checked(hipMemcpy(device_mask.data,mask.data(),B,hipMemcpyHostToDevice),"test mask upload");
    checked(hipStreamSynchronize(nullptr),"test mask/default completion");
    auto* out=reinterpret_cast<std::uint16_t*>(static_cast<std::byte*>(output.data)+256);
    const auto a=users.streams[0],b=users.streams[1],compute=users.streams[2];
    const auto borrower0=users.streams[3],borrower1=users.streams[4];
    Events independent;
    const std::array<hipEvent_t,2> borrower_done{independent.events[2],independent.events[3]};
    owner.require_completion_at_return(borrower_done);
    Graph graph;graph.capture(compute,stable_table,static_cast<const std::uint8_t*>(device_mask.data),out,C,B,words);
    const auto replay=[&] {
        checked(hipMemsetAsync(output.data,0xa5,output.bytes,compute),"test output poison");
        graph.launch(compute);compare_output(output,expected,mask,B,words,compute);
    };
    replay(); // same real instance, captured before any placement change

    // Complete ALL CPU oracle work before queuing the deliberately pending writers.
    for(std::uint32_t plane=0;plane<P;++plane)for(std::uint32_t word=0;word<words;++word) {
        expected[(plane*B+0)*words+word]^=1;
        expected[(plane*B+1)*words+word]^=0x80;
    }
    // Default-stream initialization/readback was explicitly completed before these
    // gates. Do not add a new null-stream fence which could hide the independent
    // nonblocking-user ordering requirement via an incidental global dependency.
    const std::array<hipStream_t,5> prior{a,b,compute,borrower0,borrower1};
    KVSegmentedStorage::Transaction ticket;
    unsigned prior_pending_witnesses=0;
    {
        CallbackGate gate0(a),gate1(b);
        gate0.enqueue();gate1.enqueue();
        const auto raw_pointer=[&](std::uint32_t plane,std::uint32_t group) {
            return reinterpret_cast<std::uint16_t*>(original_table[plane*B+group]);
        };
        hipLaunchKernelGGL(xor_raw_four,dim3(64),dim3(256),0,a,
            raw_pointer(0,0),raw_pointer(1,0),raw_pointer(2,0),raw_pointer(3,0),words,std::uint16_t{1});
        checked(hipGetLastError(),"v2 prior0 exact raw-address write");
        checked(hipEventRecord(independent.events[0],a),"v2 independent prior0 completion record");
        hipLaunchKernelGGL(xor_raw_four,dim3(64),dim3(256),0,b,
            raw_pointer(0,1),raw_pointer(1,1),raw_pointer(2,1),raw_pointer(3,1),words,std::uint16_t{0x80});
        checked(hipGetLastError(),"v2 prior1 exact raw-address write");
        checked(hipEventRecord(independent.events[1],b),"v2 independent prior1 completion record");
        gate0.arm(std::chrono::milliseconds(200));gate1.arm(std::chrono::milliseconds(300));
        require_pending(independent.events[0]);require_pending(independent.events[1]);
        prior_pending_witnesses=2;
        // No readback, synchronization, CPU oracle loop or helper join before begin.
        ticket=storage.begin_loan(4*S,prior);
        checked(hipEventQuery(independent.events[0]),"v2 begin exposed span before raw prior0 completion");
        checked(hipEventQuery(independent.events[1]),"v2 begin exposed span before raw prior1 completion");
        gate0.verify_completed();gate1.verify_completed();
    }
    require(ticket.open()&&owner.lent_count()==1&&owner.available()+1==initial_available,
            "actual loan did not exclude the promised capacity");
    require(ticket.copied_live_bytes()==2*S,"expected two actual live slices were not copied");
    const auto first_copy=ticket.copied_live_bytes();
    auto moved=std::move(ticket);require(!ticket.open()&&moved.open(),"ticket move duplicated ownership");
    moved.wait_ready(compute);moved.wait_ready(borrower0);moved.wait_ready(borrower1);
    const auto loan=moved.leased();require(loan.data==backing.data&&loan.bytes==4*S,"fixed raw loan span changed");
    replay(); // full post-relocation oracle BEFORE pending-close witnesses are queued
    const auto during=read_table(storage,C);
    require(during!=original_table&&storage.device_plane_page_bases(0)==stable_table,
            "actual table contents/address failed relocation/stability proof");
    inspect_table(during,backing,S);
    const std::array<hipStream_t,2> borrowers{borrower0,borrower1};
    unsigned borrower_pending_witnesses=0;
    {
        CallbackGate gate0(borrower0),gate1(borrower1);
        gate0.enqueue();gate1.enqueue();
        hipLaunchKernelGGL(poison_bytes,dim3(64),dim3(256),0,borrower0,
            static_cast<std::uint8_t*>(loan.data),loan.bytes/2,std::uint8_t{0x5a});
        checked(hipGetLastError(),"v2 pending borrower0 poison");
        checked(hipEventRecord(borrower_done[0],borrower0),"v2 independent borrower0 record");
        hipLaunchKernelGGL(poison_bytes,dim3(64),dim3(256),0,borrower1,
            static_cast<std::uint8_t*>(loan.data)+loan.bytes/2,loan.bytes/2,std::uint8_t{0xa6});
        checked(hipGetLastError(),"v2 pending borrower1 poison");
        checked(hipEventRecord(borrower_done[1],borrower1),"v2 independent borrower1 record");
        gate0.arm(std::chrono::milliseconds(250));gate1.arm(std::chrono::milliseconds(350));
        require_pending(borrower_done[0]);require_pending(borrower_done[1]);
        borrower_pending_witnesses=2;
        // Owner queries both independent events BEFORE any ledger release. No graph
        // replay/readback/join is permitted between these NotReady checks and close.
        require(moved.close_after_streams(borrowers)&&!moved.open(),"v2 pending completed close failed");
        checked(hipEventQuery(borrower_done[0]),"v2 close returned before independent borrower0 event");
        checked(hipEventQuery(borrower_done[1]),"v2 close returned before independent borrower1 event");
        gate0.verify_completed();gate1.verify_completed();
    }
    require(owner.completion_checks()==2,"v2 owner did not query completion before its ledger return");
    owner.require_completion_at_return({}); // repeat-loan path has no new pending work
    require(moved.close_after_streams(borrowers),"closed ticket not idempotent");
    require(owner.lent_count()==0&&owner.available()==initial_available&&owner.successful_returns()==1&&
            owner.generation_copy()==generations,"capacity/generation not preserved after return");
    require(read_table(storage,C)==during,"close silently restored/changed placement");
    std::vector<std::uint8_t> poison(static_cast<std::size_t>(loan.bytes));
    checked(hipMemcpy(poison.data(),loan.data,poison.size(),hipMemcpyDeviceToHost),"test completed borrowers readback");
    require(std::all_of(poison.begin(),poison.begin()+poison.size()/2,[](auto x){return x==0x5a;})&&
            std::all_of(poison.begin()+poison.size()/2,poison.end(),[](auto x){return x==0xa6;}),
            "close returned before both actual borrower writes completed");
    replay();
    auto again=storage.begin_loan(S+1,prior); // alignment rounds safely, same fixed base
    require(again.leased().data==loan.data&&again.leased().bytes==S+256&&
            again.copied_live_bytes()==0,"repeated clear window required hidden restore/copy");
    again.wait_ready(compute);replay();
    require(again.close_after_streams(borrowers)&&owner.lent_count()==0&&owner.successful_returns()==2,
            "repeated window did not return actual IDs");
    require(!storage.poisoned()&&!storage.lease_open(),"normal closed storage poisoned/open");
    return {graph.captures,graph.replays,budget.payload_device_bytes,budget.pointer_table_device_bytes,
            budget.pinned_pointer_shadow_bytes,budget.current_and_next_slot_maps_host_bytes,first_copy,
            prior_pending_witnesses,borrower_pending_witnesses,static_cast<unsigned>(owner.completion_checks())};
}

void retained_failure_case(bool return_failure) {
    constexpr std::uint32_t P=2,B=4,words=256*64,C=P*B;
    constexpr std::uint64_t S=words*2;
    const std::array<std::uint32_t,1> live{0};TestOwner owner(B,live);
    {
        KVSegmentedStorage storage({0,P,B,S,0,2*S,256,2},owner);
        upload_initial(storage,tags(C,words));
        const std::array<hipStream_t,1> prior{nullptr};
        {
            auto ticket=storage.begin_loan(2*S,prior);
            if(return_failure) {
                owner.refuse_next_return();
                require(!ticket.close_after_streams(prior),"owner return refusal incorrectly succeeded");
            } // otherwise deliberate host-side abandon, no invalid GPU/driver operation
        }
        require(storage.poisoned()&&storage.lease_open()&&owner.lent_count()==1&&
                owner.successful_returns()==0,"uncertain/abandoned loan returned excluded IDs");
        bool rejected=false;try{auto invalid=storage.begin_loan(S,prior);(void)invalid;}
        catch(const std::exception&){rejected=true;}
        require(rejected,"poisoned storage accepted another loan");
    }
    require(owner.lent_count()==1,"shutdown grant destructor silently returned retained IDs");
}
} // namespace

int main(int argc,char** argv) {
    try {
        if(argc!=2||std::string_view(argv[1])!="--run") {
            if(argc!=1)throw std::invalid_argument("usage: ninfer-kv-segmented-storage-pending-test [--run]");
            std::cout<<"{\"status\":\"PLAN_ONLY\",\"scope\":\"v2 isolated Core pending-ordering witness; no Attention\","
                         "\"execution_requested\":false,\"not_qualified\":true}\n";return 0;
        }
        checked(hipSetDevice(0),"test device binding");
        const auto result=happy_case();retained_failure_case(false);retained_failure_case(true);
        require(result.captures==1&&result.replays>=4,"same captured instance was not actually exercised");
        require(result.prior_not_ready==2&&result.borrower_not_ready==2&&result.owner_completion_checks==2,
                "v2 required pending-boundary/owner-ledger witnesses absent");
        std::cout<<"{\"status\":\"HIP_CORE_SEGMENTED_STORAGE_PENDING_CHECKS_PASSED\",\"graph_captures\":"<<result.captures
            <<",\"same_instance_replays\":"<<result.replays<<",\"payload_bytes\":"<<result.payload
            <<",\"device_table_bytes\":"<<result.table<<",\"pinned_shadow_bytes\":"<<result.pinned
            <<",\"host_slot_map_capacity_bytes\":"<<result.maps<<",\"first_live_copy_bytes\":"<<result.first_copy
            <<",\"two_prior_streams\":true,\"two_completed_borrower_streams\":true,"
              "\"pending_ordering_witnessed\":true,\"prior_boundary_not_ready_events\":"<<result.prior_not_ready
            <<",\"borrower_boundary_not_ready_events\":"<<result.borrower_not_ready
            <<",\"independent_owner_return_success_checks\":"<<result.owner_completion_checks
            <<",\"independent_test_events\":4,\"callback_fallback_timeouts\":0,\"max_helper_hold_ms\":350,"
              "\"callback_hard_wait_ms\":2000,\"concurrent_allocator_mutation_tested\":false,"
              "\"abandon_retains_ids\":true,\"owner_return_exception_retains_ids\":true,"
              "\"production_gate_changed\":false,\"attention_math_qualified\":false}\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<"HIP_CORE_SEGMENTED_STORAGE_PENDING_CHECKS_FAILED_OR_NOT_ADMITTED: "<<error.what()<<'\n';return 1;}
}
