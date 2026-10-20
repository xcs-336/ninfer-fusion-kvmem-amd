// Calls the actual unmodified planner. SOURCE ONLY: not compiled/executed at delivery.
// Independent observable checks: capacity lower bound, a tagged-byte storage oracle,
// permutation/borrowed-window ownership, and a deliberately unsafe stale-plan example.
#include "kv_segmented_loan_plan.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ninfer::amd::memory;
std::uint64_t checks=0, admitted=0, refused=0, unaligned=0, invalid=0;

void require(bool value,const char* message) {
    if (!value) throw std::runtime_error(message);
}

struct Fixture {
    std::uint32_t p,b;
    std::uint64_t s;
    std::vector<std::uint32_t> slots,free;
    std::vector<std::uint8_t> allocated;
    KVSegmentedPlacementSnapshot snapshot(std::uint32_t available,std::uint64_t epoch=13) const {
        return {p,b,s,slots,allocated,free,available,epoch};
    }
    void rebuild_free() {
        free.clear();for(std::uint32_t g=0;g<b;++g)if(!allocated[g])free.push_back(g);
    }
};

std::uint64_t tag(std::uint32_t cell,std::uint64_t byte) {
    return (static_cast<std::uint64_t>(cell)+1)*1000003U+byte;
}

// Simulate only the plan's claimed physical operations; expected live contents are generated
// independently from logical (plane,group,byte) tags. The oracle never calls a production codec.
bool tagged_bytes_survive(const Fixture& f,const KVSegmentedLoanPlan& plan,bool strict=true) {
    const std::size_t cells=f.slots.size();
    std::vector<std::uint32_t> map=f.slots;
    std::vector<std::uint64_t> memory(cells*static_cast<std::size_t>(f.s));
    for(std::uint32_t cell=0;cell<cells;++cell)
        for(std::uint64_t byte=0;byte<f.s;++byte)
            memory[static_cast<std::size_t>(map[cell]*f.s+byte)]=tag(cell,byte);
    std::vector<bool> selected(f.b,false),used_source(cells,false),used_donor(cells,false);
    for(auto id:plan.loan_page_ids) {
        require(id<f.b&&!selected[id]&&!f.allocated[id],"selected ID not distinct/free");
        selected[id]=true;
    }
    const auto last=plan.window_offset+plan.window_bytes-1;
    const auto inside=[&](std::uint32_t slot) {
        return slot>=plan.window_offset/f.s && slot<=last/f.s;
    };
    std::uint64_t copied=0;
    for(const auto& move:plan.swaps) {
        require(move.owner_cell<cells&&move.loan_cell<cells,"swap cell out of bounds");
        require(!used_source[move.owner_cell]&&!used_donor[move.loan_cell],"reused owner/donor");
        used_source[move.owner_cell]=true;used_donor[move.loan_cell]=true;
        require(!selected[move.owner_cell%f.b]&&selected[move.loan_cell%f.b],"wrong swap ownership");
        require(map[move.owner_cell]==move.source_slot&&map[move.loan_cell]==move.destination_slot,
                "swap does not match physical placement");
        require(inside(move.source_slot)&&!inside(move.destination_slot),"copy cycle/inside donor");
        if(strict) require(move.copy_live_bytes==(f.allocated[move.owner_cell%f.b]!=0),"wrong live-copy flag");
        if(move.copy_live_bytes) {
            std::copy_n(memory.begin()+static_cast<std::size_t>(move.source_slot*f.s),
                        static_cast<std::size_t>(f.s),
                        memory.begin()+static_cast<std::size_t>(move.destination_slot*f.s));
            copied+=f.s;
        }
        std::swap(map[move.owner_cell],map[move.loan_cell]);
    }
    require(copied==plan.copied_live_bytes,"copy byte accounting differs from actual records");
    auto sorted=map;std::sort(sorted.begin(),sorted.end());
    for(std::uint32_t i=0;i<cells;++i)require(sorted[i]==i,"final mapping aliases or loses a slot");
    // The borrower poisons EVERY requested byte, including unaligned window edges.
    for(std::uint64_t byte=plan.window_offset;byte<=last;++byte)
        memory[static_cast<std::size_t>(byte)]=std::numeric_limits<std::uint64_t>::max();
    bool preserved=true;
    for(std::uint32_t cell=0;cell<cells;++cell) {
        if(inside(map[cell]))require(selected[cell%f.b],"window still holds an unselected cell");
        if(f.allocated[cell%f.b])for(std::uint64_t byte=0;byte<f.s;++byte)
            preserved &= memory[static_cast<std::size_t>(map[cell]*f.s+byte)]==tag(cell,byte);
    }
    return preserved;
}

void run_case(const Fixture& f,std::uint32_t available,std::uint64_t offset,std::uint64_t bytes) {
    ++checks;
    const auto original=f.slots;
    const auto original_free=f.free;
    const auto original_allocated=f.allocated;
    const auto input=f.snapshot(available);
    const auto plan=plan_kv_segmented_loan(input,offset,bytes);
    require(f.slots==original,"pure planner mutated the supplied placement");
    require(f.free==original_free&&f.allocated==original_allocated,
            "pure planner mutated allocation/free input snapshots");
    const auto touched=(offset+bytes-1)/f.s-offset/f.s+1;
    // Independent cardinality lower bound: L groups own only L*P physical slices.
    const auto minimum=(touched+f.p-1)/f.p;
    if(minimum>available) {++refused;require(!plan,"promised capacity stolen");return;}
    ++admitted;if(offset%f.s||(offset+bytes)%f.s)++unaligned;
    require(plan.has_value(),"sufficient genuinely free group capacity refused");
    require(plan->resource_revision==input.resource_revision,"epoch not preserved");
    require(plan->window_offset==offset&&plan->window_bytes==bytes&&plan->slot_bytes==f.s,
            "plan changed requested byte interval");
    require(plan->loan_page_ids.size()==minimum,"loan group count exceeds minimum");
    require(plan->copied_live_bytes<=touched*f.s,"live-copy upper bound exceeded");
    // Only small fixtures use this expression; the u64 tests below deliberately avoid it.
    require(plan->copied_live_bytes<bytes+2*f.s,"unaligned evacuation byte bound exceeded");
    require(tagged_bytes_survive(f,*plan),"independent allocated-byte oracle failed");
}

template<class Function>void rejects(Function&& function) {
    ++invalid;
    try {function();}
    catch(const std::invalid_argument&){return;}
    catch(const std::overflow_error&){return;}
    throw std::runtime_error("invalid snapshot/extent accepted");
}

void invalid_and_arithmetic_cases() {
    Fixture good{2,2,3,{0,1,2,3},{0,1},{0,0}};
    const auto call=[&](const Fixture& f){(void)plan_kv_segmented_loan(f.snapshot(2),0,1);};
    auto bad=good;bad.slots[1]=0;rejects([&]{call(bad);});
    bad=good;bad.slots[1]=4;rejects([&]{call(bad);});
    bad=good;bad.slots.pop_back();rejects([&]{call(bad);});
    bad=good;bad.free={1,0};rejects([&]{call(bad);});
    bad=good;bad.free={0,0};rejects([&]{call(bad);});
    bad=good;bad.allocated[0]=1;rejects([&]{call(bad);});
    bad=good;bad.allocated[0]=2;rejects([&]{call(bad);});
    bad=good;bad.free.pop_back();rejects([&]{call(bad);});
    rejects([&]{(void)plan_kv_segmented_loan(good.snapshot(3),0,1);});
    for(auto member:{0,1,2}) {
        bad=good;if(member==0)bad.p=0;else if(member==1)bad.b=0;else bad.s=0;
        rejects([&]{call(bad);});
    }
    rejects([&]{(void)plan_kv_segmented_loan(good.snapshot(2),0,0);});
    rejects([&]{(void)plan_kv_segmented_loan(good.snapshot(2),12,1);});
    rejects([&]{(void)plan_kv_segmented_loan(good.snapshot(2),11,2);});
    auto huge=good.snapshot(2);huge.planes=65536;huge.page_groups=65536;
    rejects([&]{(void)plan_kv_segmented_loan(huge,0,1);});
    huge=good.snapshot(2);huge.slot_bytes=std::uint64_t{1}<<63;
    rejects([&]{(void)plan_kv_segmented_loan(huge,0,1);});
    huge=good.snapshot(2);huge.planes=1;huge.page_groups=std::numeric_limits<std::uint32_t>::max();
    rejects([&]{(void)plan_kv_segmented_loan(huge,0,1);}); // extent guard before giant allocation

    Fixture single{1,1,std::numeric_limits<std::uint64_t>::max(),{0},{0},{0}};
    const auto max=single.s;
    for(const auto range:std::array<std::array<std::uint64_t,2>,3>{{{0,max},{max-1,1},{1,max-1}}}) {
        const auto plan=plan_kv_segmented_loan(single.snapshot(1),range[0],range[1]);
        require(plan&&plan->loan_page_ids.size()==1&&plan->copied_live_bytes==0,
                "valid u64 edge failed");
    }
    rejects([&]{(void)plan_kv_segmented_loan(single.snapshot(1),max-1,2);});
}

void stale_snapshot_contract() {
    Fixture f{1,3,1,{0,1,2},{0,1,2},{0,0,0}};
    const auto old=plan_kv_segmented_loan(f.snapshot(2,40),0,1);
    require(old&&old->swaps.size()==1&&!old->swaps[0].copy_live_bytes,"stale example setup wrong");
    // reserved1 -> materialize page0 -> reserved0; available stays2 but old plan is unsafe.
    f.allocated[0]=1;f.rebuild_free();
    require(!tagged_bytes_survive(f,*old,false),"stale skip-copy counterexample was not detected");
    const auto fresh=plan_kv_segmented_loan(f.snapshot(2,41),0,1);
    require(fresh&&fresh->swaps[0].copy_live_bytes&&tagged_bytes_survive(f,*fresh),
            "fresh full-state plan did not preserve newly live page");
    // This fixture demonstrates a future commit obligation; the planner itself does not
    // authenticate/freeze a real allocator, and there is no physical transaction tested here.
}
} // namespace

int main() {
    try {
        for(const auto geometry:std::array<std::array<std::uint32_t,2>,3>{{{1,3},{2,2},{3,1}}}) {
            Fixture f{geometry[0],geometry[1],3,{},{},{}};
            f.slots.resize(f.p*f.b);std::iota(f.slots.begin(),f.slots.end(),0U);
            do {
                for(std::uint32_t mask=0;mask<(1U<<f.b);++mask) {
                    f.allocated.resize(f.b);
                    for(std::uint32_t g=0;g<f.b;++g)f.allocated[g]=(mask>>g)&1U;
                    f.rebuild_free();
                    for(std::uint32_t available=0;available<=f.free.size();++available)
                        for(std::uint64_t offset=0;offset<f.slots.size()*f.s;++offset)
                            for(std::uint64_t bytes=1;bytes<=f.slots.size()*f.s-offset;++bytes)
                                run_case(f,available,offset,bytes);
                }
            }while(std::next_permutation(f.slots.begin(),f.slots.end()));
        }
        std::mt19937 rng(20261020);
        for(int trial=0;trial<512;++trial) {
            Fixture f{static_cast<std::uint32_t>(1U+rng()%8),
                static_cast<std::uint32_t>(1U+rng()%24),1U+rng()%32,{},{},{}};
            f.slots.resize(f.p*f.b);std::iota(f.slots.begin(),f.slots.end(),0U);
            std::shuffle(f.slots.begin(),f.slots.end(),rng);
            f.allocated.resize(f.b);for(auto& bit:f.allocated)bit=rng()%2;
            f.rebuild_free();const auto total=f.slots.size()*f.s;
            const auto offset=rng()%total,bytes=1+rng()%(total-offset);
            run_case(f,rng()%(f.free.size()+1),offset,bytes);
        }
        invalid_and_arithmetic_cases();stale_snapshot_contract();
        std::cout<<"{\"status\":\"CPU_PLANNER_CHECKS_PASSED\",\"cases\":"<<checks
            <<",\"admitted\":"<<admitted<<",\"capacity_refusals\":"<<refused
            <<",\"unaligned_admitted\":"<<unaligned<<",\"invalid_rejections\":"<<invalid
            <<",\"stale_contract_counterexample_observed\":true,\"hip_called\":false,"
              "\"physical_backend_qualified\":false}\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"CPU_PLANNER_CHECKS_FAILED: "<<e.what()<<'\n';return 1;
    }
}
