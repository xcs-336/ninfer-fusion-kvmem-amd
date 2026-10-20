// SOURCE ONLY / NOT BUILT / NOT EXECUTED. This is a real isolated HIP Core byte test.
// The consumer resolves stable page pointers, not existing affine Attention. No model,
// quantizer, codec, illegal GPU address or driver-failure injection is involved.
// Two-stream use is functional coverage: this fixture does not force work to remain
// pending at a boundary, or test concurrent allocation/free/generation mutations.
#include "kv_segmented_storage.h"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
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
};

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
__global__ void xor_group(const std::uint64_t* table,std::uint32_t planes,
                          std::uint32_t groups,std::uint32_t group,std::uint32_t words,
                          std::uint16_t bits) {
    const std::uint64_t total=static_cast<std::uint64_t>(planes)*words;
    for(std::uint64_t i=blockIdx.x*blockDim.x+threadIdx.x;i<total;
        i+=static_cast<std::uint64_t>(gridDim.x)*blockDim.x) {
        const auto cell=static_cast<std::uint32_t>(i/words)*groups+group;
        auto* page=reinterpret_cast<std::uint16_t*>(table[cell]);
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

struct HappyEvidence {unsigned captures=0,replays=0;std::uint64_t payload=0,table=0,pinned=0,maps=0,first_copy=0;};
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
    Graph graph;graph.capture(compute,stable_table,static_cast<const std::uint8_t*>(device_mask.data),out,C,B,words);
    const auto replay=[&] {
        checked(hipMemsetAsync(output.data,0xa5,output.bytes,compute),"test output poison");
        graph.launch(compute);compare_output(output,expected,mask,B,words,compute);
    };
    replay(); // same real instance, captured before any placement change

    hipLaunchKernelGGL(xor_group,dim3(64),dim3(256),0,a,stable_table,P,B,0,words,std::uint16_t{1});
    checked(hipGetLastError(),"test prior-user0 write");
    hipLaunchKernelGGL(xor_group,dim3(64),dim3(256),0,b,stable_table,P,B,1,words,std::uint16_t{0x80});
    checked(hipGetLastError(),"test prior-user1 write");
    for(std::uint32_t plane=0;plane<P;++plane)for(std::uint32_t word=0;word<words;++word) {
        expected[(plane*B+0)*words+word]^=1;
        expected[(plane*B+1)*words+word]^=0x80;
    }
    const std::array<hipStream_t,6> prior{nullptr,a,b,compute,borrower0,borrower1};
    auto ticket=storage.begin_loan(4*S,prior);
    require(ticket.open()&&owner.lent_count()==1&&owner.available()+1==initial_available,
            "actual loan did not exclude the promised capacity");
    require(ticket.copied_live_bytes()==2*S,"expected two actual live slices were not copied");
    const auto first_copy=ticket.copied_live_bytes();
    auto moved=std::move(ticket);require(!ticket.open()&&moved.open(),"ticket move duplicated ownership");
    moved.wait_ready(compute);moved.wait_ready(borrower0);moved.wait_ready(borrower1);
    const auto loan=moved.leased();require(loan.data==backing.data&&loan.bytes==4*S,"fixed raw loan span changed");
    hipLaunchKernelGGL(poison_bytes,dim3(64),dim3(256),0,borrower0,
        static_cast<std::uint8_t*>(loan.data),loan.bytes/2,std::uint8_t{0x5a});
    checked(hipGetLastError(),"test borrower0 poison");
    hipLaunchKernelGGL(poison_bytes,dim3(64),dim3(256),0,borrower1,
        static_cast<std::uint8_t*>(loan.data)+loan.bytes/2,loan.bytes/2,std::uint8_t{0xa6});
    checked(hipGetLastError(),"test borrower1 poison");
    replay(); // current pointers resolve moved live data while borrower uses disjoint bytes
    const auto during=read_table(storage,C);
    require(during!=original_table&&storage.device_plane_page_bases(0)==stable_table,
            "actual table contents/address failed relocation/stability proof");
    inspect_table(during,backing,S);
    const std::array<hipStream_t,2> borrowers{borrower0,borrower1};
    require(moved.close_after_streams(borrowers)&&!moved.open(),"actual completed close failed");
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
            budget.pinned_pointer_shadow_bytes,budget.current_and_next_slot_maps_host_bytes,first_copy};
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
            if(argc!=1)throw std::invalid_argument("usage: ninfer-kv-segmented-storage-test [--run]");
            std::cout<<"{\"status\":\"PLAN_ONLY\",\"scope\":\"isolated real HIP Core bytes; no Attention\","
                         "\"execution_requested\":false,\"not_qualified\":true}\n";return 0;
        }
        checked(hipSetDevice(0),"test device binding");
        const auto result=happy_case();retained_failure_case(false);retained_failure_case(true);
        require(result.captures==1&&result.replays>=4,"same captured instance was not actually exercised");
        std::cout<<"{\"status\":\"HIP_CORE_SEGMENTED_STORAGE_CHECKS_PASSED\",\"graph_captures\":"<<result.captures
            <<",\"same_instance_replays\":"<<result.replays<<",\"payload_bytes\":"<<result.payload
            <<",\"device_table_bytes\":"<<result.table<<",\"pinned_shadow_bytes\":"<<result.pinned
            <<",\"host_slot_map_capacity_bytes\":"<<result.maps<<",\"first_live_copy_bytes\":"<<result.first_copy
            <<",\"two_prior_streams\":true,\"two_completed_borrower_streams\":true,"
              "\"pending_ordering_witnessed\":false,\"concurrent_allocator_mutation_tested\":false,"
              "\"abandon_retains_ids\":true,\"owner_return_exception_retains_ids\":true,"
              "\"production_gate_changed\":false,\"attention_math_qualified\":false}\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<"HIP_CORE_SEGMENTED_STORAGE_CHECKS_FAILED: "<<error.what()<<'\n';return 1;}
}
