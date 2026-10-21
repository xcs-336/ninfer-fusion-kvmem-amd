#include "kv_segmented_pool_owner.h"
#include "kv_segmented_pool_control.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::amd::memory {
namespace {
void need(bool value,const char* why){if(!value)throw std::logic_error(why);}
bool contains(const std::vector<KVPageRun>& runs,std::uint32_t id) {
    for(const auto& run:runs)if(run.begin>=0&&id>=static_cast<std::uint32_t>(run.begin)&&
        static_cast<std::uint64_t>(id)<static_cast<std::uint64_t>(run.begin)+run.count)return true;
    return false;
}
// Work only on a private, fully allocated vector; commit is one noexcept swap.
void remove_id(std::vector<KVPageRun>& runs,std::uint32_t id) {
    for(std::size_t i=0;i<runs.size();++i) {
        const auto run=runs[i];const auto start=static_cast<std::uint32_t>(run.begin);
        if(id<start||static_cast<std::uint64_t>(id)>=static_cast<std::uint64_t>(start)+run.count)continue;
        const auto left=id-start,right=run.count-left-1;
        if(left&&right){runs[i].count=left;runs.insert(runs.begin()+i+1,{static_cast<std::int32_t>(id+1),right});}
        else if(left)runs[i].count=left;
        else if(right){runs[i].begin=static_cast<std::int32_t>(id+1);runs[i].count=right;}
        else runs.erase(runs.begin()+i);
        return;
    }
    throw std::logic_error("segmented selected page disappeared from original free runs");
}
void insert_id(std::vector<KVPageRun>& runs,std::uint32_t id) {
    const auto next=std::lower_bound(runs.begin(),runs.end(),id,
        [](const KVPageRun& run,std::uint32_t value){return static_cast<std::uint32_t>(run.begin)<value;});
    const bool right=next!=runs.end()&&static_cast<std::uint64_t>(id)+1==static_cast<std::uint32_t>(next->begin);
    const bool left=next!=runs.begin()&&static_cast<std::uint64_t>((next-1)->begin)+(next-1)->count==id;
    if(left&&right){auto& previous=*(next-1);previous.count+=1+next->count;runs.erase(next);}
    else if(left)++(next-1)->count;
    else if(right){--next->begin;++next->count;}
    else runs.insert(next,{static_cast<std::int32_t>(id),1});
}
} // namespace

class KVSegmentedPoolOwner::Grant final:public KVSegmentedLentPages {
public:
    Grant(KVSegmentedPoolOwner& owner,std::uint64_t nonce,std::span<const std::uint32_t> ids)
        :actor(owner),nonce(nonce),ids(ids.begin(),ids.end()){}
    const KVSegmentedAllocatorOwner& owner()const noexcept override{return actor;}
    std::span<const std::uint32_t> pages()const noexcept override{return ids;}
    KVSegmentedPoolOwner& actor;std::uint64_t nonce;
    std::vector<std::uint32_t> ids;
    // Never implicitly returns IDs, including poisoned storage shutdown.
};
class KVSegmentedPoolOwner::Freeze final:public KVSegmentedAllocatorFreeze {
public:
    explicit Freeze(KVSegmentedPoolOwner& owner):actor(owner),lock(owner.control_mutex()){}
    const KVSegmentedAllocatorOwner& owner()const noexcept override{return actor;}
    KVSegmentedFrozenAllocation snapshot()const override{return actor.snapshot_locked();}
    std::unique_ptr<KVSegmentedLentPages> lend_selected(std::span<const std::uint32_t> ids)override{return actor.lend_locked(ids);}
    void return_selected(KVSegmentedLentPages& grant)override{actor.return_locked(grant);}
private:KVSegmentedPoolOwner& actor;std::unique_lock<std::recursive_mutex> lock;
};

KVSegmentedPoolOwner::KVSegmentedPoolOwner(const KVSegmentedStorage::Config& config):config_(config){}
KVSegmentedPoolOwner::~KVSegmentedPoolOwner()=default;
void KVSegmentedPoolOwner::bind(DeviceKVPagePool& pool,KVSegmentedStorage& storage) {
    need(pool_==nullptr,"segmented owner already bound");
    std::lock_guard lock(pool.segmented_native_.mutex);
    need(!pool.segmented_native_.owner&&!pool.segmented_native_.storage&&pool.lent_pages_==0,
         "original pool already bound/lent");
    need(pool.rank_count_==1&&pool.capacity_pages()==config_.page_groups&&
         pool.planes_.size()==config_.planes&&pool.geometry().device_plane_order==PagedKVPlaneOrder::PageMajor,
         "segmented native pool requires one rank and exact plane/group geometry");
    const auto backing=storage.raw_payload_for_initialization();
    for(std::size_t p=0;p<pool.planes_.size();++p) {
        const auto& plane=pool.planes_[p];
        need(plane.dtype==DType::BF16&&plane.ne[0]==256&&plane.ne[1]==64&&
             plane.nb[3]==config_.slot_bytes&&pool.plane_ranks_[p]==0,
             "first native segmented class is homogeneous page-major BF16 D256");
        need(plane.data==static_cast<std::byte*>(backing.data)+p*config_.page_groups*config_.slot_bytes,
             "original pool views do not bind the sole segmented payload");
    }
    need(pool.segmented_native_.mutation_epoch!=std::numeric_limits<std::uint64_t>::max(),"segmented binding epoch exhausted");
    pool_=&pool;pool.segmented_native_.owner=this;pool.segmented_native_.storage=&storage;
    pool.segmented_native_.submission_thread=std::this_thread::get_id();
    ++pool.segmented_native_.mutation_epoch;
    storage.seal_initialization();
}
std::recursive_mutex& KVSegmentedPoolOwner::control_mutex()const {
    pool_->segmented_native_.require_submission_thread();return pool_->segmented_native_.mutex;
}
std::unique_ptr<KVSegmentedAllocatorFreeze> KVSegmentedPoolOwner::freeze() {
    need(pool_!=nullptr,"segmented original allocator is not bound");
    return std::make_unique<Freeze>(*this);
}
KVSegmentedFrozenAllocation KVSegmentedPoolOwner::snapshot_locked()const {
    need(!active_grant_&&pool_->lent_pages_==0,"cannot plan a second original-pool loan");
    KVSegmentedFrozenAllocation out;
    out.owner=this;out.mutation_epoch=pool_->segmented_native_.mutation_epoch;
    out.page_groups=pool_->capacity_pages();out.available_pages=pool_->available_pages();
    out.allocated.reserve(out.page_groups);out.free_pages.reserve(out.page_groups);
    for(bool allocated:pool_->page_allocated_)out.allocated.push_back(allocated?1:0);
    for(const auto run:pool_->free_page_runs_)for(std::uint32_t i=0;i<run.count;++i)
        out.free_pages.push_back(static_cast<std::uint32_t>(run.begin)+i);
    return out;
}
std::unique_ptr<KVSegmentedLentPages> KVSegmentedPoolOwner::lend_locked(std::span<const std::uint32_t> ids) {
    need(!active_grant_&&pool_->lent_pages_==0&&!ids.empty()&&ids.size()<=pool_->available_pages(),
         "original allocator cannot grant selected unpromised capacity");
    need(next_nonce_!=std::numeric_limits<std::uint64_t>::max()&&
         pool_->segmented_native_.mutation_epoch!=std::numeric_limits<std::uint64_t>::max(),
         "segmented grant/epoch exhausted");
    for(std::size_t i=0;i<ids.size();++i)need(ids[i]<pool_->capacity_pages()&&
        !pool_->page_allocated_[ids[i]]&&(i==0||ids[i-1]<ids[i])&&contains(pool_->free_page_runs_,ids[i]),
        "selected IDs are not sorted distinct original free pages");
    auto grant=std::make_unique<Grant>(*this,next_nonce_,ids);
    std::vector<KVPageRun> next;next.reserve(pool_->capacity_pages());
    next.assign(pool_->free_page_runs_.begin(),pool_->free_page_runs_.end());
    for(auto id:ids)remove_id(next,id);
    // All throwing work precedes the actual original allocator mutation.
    pool_->free_page_runs_.swap(next);pool_->lent_pages_+=static_cast<std::uint32_t>(ids.size());
    ++pool_->segmented_native_.mutation_epoch;++next_nonce_;active_grant_=grant.get();
    return grant;
}
void KVSegmentedPoolOwner::return_locked(KVSegmentedLentPages& opaque) {
    auto* grant=dynamic_cast<Grant*>(&opaque);
    need(grant && grant==active_grant_ && &grant->actor==this && grant->nonce+1==next_nonce_,
         "segmented return does not own the unique active original-pool grant");
    need(grant->ids.size()==pool_->lent_pages_&&
         pool_->segmented_native_.mutation_epoch!=std::numeric_limits<std::uint64_t>::max(),
         "segmented return counter/epoch mismatch");
    for(auto id:grant->ids)need(id<pool_->capacity_pages()&&!pool_->page_allocated_[id]&&
        !contains(pool_->free_page_runs_,id),"lent original ID was allocated or already returned");
    std::vector<KVPageRun> next;next.reserve(pool_->capacity_pages());
    next.assign(pool_->free_page_runs_.begin(),pool_->free_page_runs_.end());
    for(auto id:grant->ids)insert_id(next,id);
    // Atomic on host exceptions: excluded IDs remain excluded until this commit.
    pool_->free_page_runs_.swap(next);pool_->lent_pages_=0;
    ++pool_->segmented_native_.mutation_epoch;active_grant_=nullptr;
}

KVSegmentedNativePool::KVSegmentedNativePool(const KVSegmentedStorage::Config& config,
                                             const DeviceKVPagePoolLayout& layout):owner_(config) {
    need(layout.planes.size()==config.planes&&layout.spec.page_group_count==config.page_groups,
         "segmented layout inventory differs from storage configuration");
    for(std::size_t p=0;p<layout.planes.size();++p)need(layout.planes[p].rank==0&&
        layout.planes[p].storage.region.offset==p*config.page_groups*config.slot_bytes&&
        layout.planes[p].storage.region.bytes==config.page_groups*config.slot_bytes,
        "segmented isolated layout must start at zero and contain only the one payload");
    storage_=std::make_unique<KVSegmentedStorage>(config,owner_);
    const auto raw=storage_->raw_payload_for_initialization();
    pool_=std::make_unique<DeviceKVPagePool>(DeviceSpan{raw.data,static_cast<std::size_t>(raw.bytes)},layout);
    owner_.bind(*pool_,*storage_);
}
KVSegmentedNativePool::~KVSegmentedNativePool()=default;
} // namespace ninfer::amd::memory
