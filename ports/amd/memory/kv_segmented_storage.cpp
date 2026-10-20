#include "kv_segmented_storage.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::amd::memory {
namespace {
void check(hipError_t value,const char* where) {
    if(value!=hipSuccess)
        throw std::runtime_error(std::string(where)+": "+hipGetErrorString(value));
}
void require(bool value,const char* message) {
    if(!value)throw std::invalid_argument(message);
}
void require_not_capturing(hipStream_t stream) {
    hipStreamCaptureStatus status=hipStreamCaptureStatusNone;
    check(hipStreamIsCapturing(stream,&status),"segmented query stream capture state");
    require(status==hipStreamCaptureStatusNone,"segmented synchronous boundary cannot run inside stream capture");
}
std::uint64_t product(std::uint64_t a,std::uint64_t b) {
    if(b && a>std::numeric_limits<std::uint64_t>::max()/b)
        throw std::overflow_error("segmented storage byte count overflow");
    return a*b;
}
class DeviceBinding {
public:
    explicit DeviceBinding(int device) {
        check(hipGetDevice(&old_),"segmented get device");
        if(old_!=device) {check(hipSetDevice(device),"segmented bind device");changed_=true;}
    }
    ~DeviceBinding(){if(changed_)(void)hipSetDevice(old_);}
private:int old_=0;bool changed_=false;
};
} // namespace

struct KVSegmentedStorage::Impl {
    Config config;
    KVSegmentedAllocatorOwner& owner;
    std::uint32_t cells=0;
    KVSegmentedStorageFootprint bytes{};
    void* payload=nullptr;
    std::uint64_t* device_table=nullptr;
    std::uint64_t* pinned_table=nullptr;
    std::vector<std::uint32_t> current,next;
    hipStream_t transfer=nullptr;
    hipEvent_t ready=nullptr;
    std::array<hipEvent_t,8> prior{},finished{};
    std::unique_ptr<KVSegmentedLentPages> grant;
    std::uint64_t serial=0;
    bool active=false,bad=false;
    std::array<char,256> reason{};

    Impl(const Config& c,KVSegmentedAllocatorOwner& o):config(c),owner(o) {}
    void poison(const char* message) noexcept {
        bad=true;std::snprintf(reason.data(),reason.size(),"%s",message);
    }
    void check_usable() const {
        if(bad)throw std::runtime_error(std::string("segmented KV poisoned: ")+reason.data());
    }
    void fill_table(const std::vector<std::uint32_t>& map) {
        const auto base=reinterpret_cast<std::uintptr_t>(payload);
        for(std::uint32_t cell=0;cell<cells;++cell)
            pinned_table[cell]=base+static_cast<std::uint64_t>(map[cell])*config.slot_bytes;
    }
    void initialize() {
        require(config.device>=0&&config.planes&&config.page_groups&&config.slot_bytes,
                "invalid segmented storage geometry/device");
        const auto count=product(config.planes,config.page_groups);
        require(count<=std::numeric_limits<std::uint32_t>::max(),"too many segmented storage cells");
        cells=static_cast<std::uint32_t>(count);
        bytes.payload_device_bytes=product(count,config.slot_bytes);
        bytes.pointer_table_device_bytes=product(count,sizeof(std::uint64_t));
        bytes.pinned_pointer_shadow_bytes=bytes.pointer_table_device_bytes;
        bytes.current_and_next_slot_maps_host_bytes=product(count,2*sizeof(std::uint32_t));
        require(bytes.payload_device_bytes<=std::numeric_limits<std::size_t>::max()&&
                bytes.pointer_table_device_bytes<=std::numeric_limits<std::size_t>::max(),
                "segmented storage exceeds host size_t");
        const auto alignment=config.borrower_alignment;
        require(alignment&&alignment<=256&&(alignment&(alignment-1))==0&&
                config.slot_bytes%alignment==0&&config.window_offset%alignment==0&&
                config.max_window_bytes&&config.max_window_bytes%alignment==0,
                "segmented backing/borrower alignment invalid");
        require(config.window_offset<bytes.payload_device_bytes&&
                config.max_window_bytes<=bytes.payload_device_bytes-config.window_offset,
                "segmented fixed window outside payload");
        require(config.max_user_streams>=1&&config.max_user_streams<=prior.size(),
                "segmented stream inventory must be in [1,8]");
        current.resize(cells);next.resize(cells);
        bytes.current_and_next_slot_maps_host_bytes=product(
            static_cast<std::uint64_t>(current.capacity())+next.capacity(),sizeof(std::uint32_t));
        std::iota(current.begin(),current.end(),0U);
        DeviceBinding binding(config.device);
        check(hipMalloc(&payload,static_cast<std::size_t>(bytes.payload_device_bytes)),"segmented payload allocation");
        require(reinterpret_cast<std::uintptr_t>(payload)%alignment==0,
                "actual payload allocation fails borrower alignment");
        check(hipMalloc(reinterpret_cast<void**>(&device_table),
                        static_cast<std::size_t>(bytes.pointer_table_device_bytes)),"segmented nonlendable table allocation");
        check(hipHostMalloc(reinterpret_cast<void**>(&pinned_table),
                            static_cast<std::size_t>(bytes.pinned_pointer_shadow_bytes),hipHostMallocDefault),
              "segmented pinned table allocation");
        check(hipStreamCreateWithFlags(&transfer,hipStreamNonBlocking),"segmented private transfer stream");
        bytes.private_streams=1;
        check(hipEventCreateWithFlags(&ready,hipEventDisableTiming),"segmented ready event");
        ++bytes.events;
        for(std::uint32_t i=0;i<config.max_user_streams;++i) {
            check(hipEventCreateWithFlags(&prior[i],hipEventDisableTiming),"segmented reader fence");++bytes.events;
            check(hipEventCreateWithFlags(&finished[i],hipEventDisableTiming),"segmented borrower fence");++bytes.events;
        }
        fill_table(current);
        check(hipMemcpyAsync(device_table,pinned_table,static_cast<std::size_t>(bytes.pointer_table_device_bytes),
                             hipMemcpyHostToDevice,transfer),"segmented initial table upload");
        check(hipStreamSynchronize(transfer),"segmented initial table completion");
    }
    ~Impl() {
        // Storage must outlive its users. Shutdown is conservative, including the fatal
        // abandoned-ticket path; ordinary begin/close never performs device-wide sync.
        if(!payload&&!device_table&&!pinned_table&&!transfer&&!ready)return;
        int old=0;const bool have_old=hipGetDevice(&old)==hipSuccess;
        (void)hipSetDevice(config.device);
        if(payload)(void)hipDeviceSynchronize();
        if(transfer)(void)hipStreamSynchronize(transfer);
        // A poisoned/abandoned grant is intentionally NOT returned to allocator capacity.
        // Its protocol destructor cannot return IDs implicitly.
        grant.reset();
        for(auto e:prior)if(e)(void)hipEventDestroy(e);
        for(auto e:finished)if(e)(void)hipEventDestroy(e);
        if(ready)(void)hipEventDestroy(ready);
        if(transfer)(void)hipStreamDestroy(transfer);
        if(pinned_table)(void)hipHostFree(pinned_table);
        if(device_table)(void)hipFree(device_table);
        if(payload)(void)hipFree(payload);
        if(have_old&&old!=config.device)(void)hipSetDevice(old);
    }
    std::unique_ptr<KVSegmentedAllocatorFreeze> freeze() {
        auto frozen=owner.freeze();
        require(frozen && &frozen->owner()==&owner,"allocator freeze belongs to another owner");
        return frozen;
    }
    KVSegmentedFrozenAllocation snapshot(KVSegmentedAllocatorFreeze& frozen) {
        auto state=frozen.snapshot();
        require(state.owner==&owner&&state.page_groups==config.page_groups,
                "frozen allocation snapshot owner/geometry mismatch");
        return state;
    }
    void validate_grant(std::span<const std::uint32_t> expected) {
        require(grant && &grant->owner()==&owner,"allocator grant belongs to another owner");
        const auto actual=grant->pages();
        require(actual.size()==expected.size()&&std::equal(actual.begin(),actual.end(),expected.begin()),
                "allocator grant differs from selected pages");
    }
    void rollback(KVSegmentedAllocatorFreeze& frozen) {
        // Old source slots have not been exposed to the borrower. Drain any partial upload
        // BEFORE reconstructing/reusing its pinned source buffer from the unchanged map.
        check(hipStreamSynchronize(transfer),"segmented rollback drain");
        fill_table(current);
        check(hipMemcpyAsync(device_table,pinned_table,static_cast<std::size_t>(bytes.pointer_table_device_bytes),
                             hipMemcpyHostToDevice,transfer),"segmented rollback table upload");
        check(hipStreamSynchronize(transfer),"segmented rollback table completion");
        frozen.return_selected(*grant);
        grant.reset();active=false;
    }
};

KVSegmentedStorage::KVSegmentedStorage(const Config& config,KVSegmentedAllocatorOwner& owner)
    :impl_(std::make_unique<Impl>(config,owner)) {impl_->initialize();}
KVSegmentedStorage::~KVSegmentedStorage()=default;
KVSegmentedStorageFootprint KVSegmentedStorage::footprint() const noexcept{return impl_->bytes;}
bool KVSegmentedStorage::poisoned() const noexcept{return impl_->bad;}
bool KVSegmentedStorage::lease_open() const noexcept{return impl_->active;}
std::string_view KVSegmentedStorage::poison_reason() const noexcept{return impl_->reason.data();}

const std::uint64_t* KVSegmentedStorage::device_plane_page_bases(std::uint32_t plane) const {
    impl_->check_usable();require(plane<impl_->config.planes,"segmented plane out of bounds");
    return impl_->device_table+static_cast<std::size_t>(plane)*impl_->config.page_groups;
}
KVSegmentedSpan KVSegmentedStorage::raw_payload_for_initialization() const {
    impl_->check_usable();require(!impl_->active,"cannot initialize a borrowed segmented payload");
    return {impl_->payload,impl_->bytes.payload_device_bytes};
}

KVSegmentedStorage::Transaction KVSegmentedStorage::begin_loan(
    std::uint64_t requested,std::span<const hipStream_t> users) {
    auto& s=*impl_;s.check_usable();
    require(!s.active&&!s.grant,"segmented storage already has a loan");
    require(!users.empty()&&users.size()<=s.config.max_user_streams,"incomplete/oversized prior-user stream inventory");
    require(requested!=0&&requested<=s.config.max_window_bytes,"invalid segmented request size");
    const auto align=s.config.borrower_alignment;
    require(requested<=std::numeric_limits<std::uint64_t>::max()-(align-1),"segmented request rounding overflow");
    const auto rounded=(requested+align-1)/align*align;
    require(rounded<=s.config.max_window_bytes,"rounded segmented request exceeds fixed window");
    require(s.serial!=std::numeric_limits<std::uint64_t>::max(),"segmented loan serial exhausted");
    DeviceBinding binding(s.config.device);
    for(const auto stream:users)require_not_capturing(stream); // before allocator mutation
    auto frozen=s.freeze();
    const auto allocation=s.snapshot(*frozen);
    const KVSegmentedPlacementSnapshot input{s.config.planes,s.config.page_groups,s.config.slot_bytes,
        s.current,allocation.allocated,allocation.free_pages,allocation.available_pages,allocation.mutation_epoch};
    const auto planned=plan_kv_segmented_loan(input,s.config.window_offset,rounded);
    if(!planned)throw std::runtime_error("insufficient unpromised segmented KV capacity");
    // Complete all CPU allocation/planning before allocator mutation and GPU publication.
    s.next=s.current;
    for(const auto& move:planned->swaps) {
        require(s.next[move.owner_cell]==move.source_slot&&s.next[move.loan_cell]==move.destination_slot,
                "segmented plan does not match current placement");
        std::swap(s.next[move.owner_cell],s.next[move.loan_cell]);
    }
    const auto revalidated=s.snapshot(*frozen);
    require(revalidated.mutation_epoch==allocation.mutation_epoch&&
            revalidated.available_pages==allocation.available_pages&&
            revalidated.allocated==allocation.allocated&&revalidated.free_pages==allocation.free_pages,
            "allocator state/epoch mutated while holding its purported freeze");
    // Value-owned snapshots permit complete comparison, even if an adapter wrongly omits
    // an epoch update. Correct exclusion must still protect the check-to-lend interval.
    s.grant=frozen->lend_selected(planned->loan_page_ids);
    s.active=true;
    bool grant_validated=false;
    bool prior_fences_complete=false;
    try {
        s.validate_grant(planned->loan_page_ids);
        grant_validated=true;
        for(std::size_t i=0;i<users.size();++i) {
            check(hipEventRecord(s.prior[i],users[i]),"segmented record actual prior-user completion");
            check(hipStreamWaitEvent(s.transfer,s.prior[i],0),"segmented wait prior-user completion");
        }
        prior_fences_complete=true;
        auto* raw=static_cast<std::byte*>(s.payload);
        for(std::size_t begin=0;begin<planned->swaps.size();) {
            const auto& move=planned->swaps[begin];
            if(!move.copy_live_bytes){++begin;continue;}
            std::size_t end=begin+1;
            while(end<planned->swaps.size()) {
                const auto& following=planned->swaps[end];
                const auto distance=static_cast<std::uint64_t>(end-begin);
                if(!following.copy_live_bytes||
                   following.source_slot!=static_cast<std::uint64_t>(move.source_slot)+distance||
                   following.destination_slot!=static_cast<std::uint64_t>(move.destination_slot)+distance)break;
                ++end;
            }
            // Coalesce ONLY genuinely consecutive live source AND free-donor destinations;
            // never include an unproved gap or overwrite another mapping's bytes.
            const auto copy_bytes=product(end-begin,s.config.slot_bytes);
            check(hipMemcpyAsync(raw+static_cast<std::uint64_t>(move.destination_slot)*s.config.slot_bytes,
                                 raw+static_cast<std::uint64_t>(move.source_slot)*s.config.slot_bytes,
                                 static_cast<std::size_t>(copy_bytes),hipMemcpyDeviceToDevice,s.transfer),
                  "segmented exact live-slice relocation");
            begin=end;
        }
        s.fill_table(s.next);
        check(hipMemcpyAsync(s.device_table,s.pinned_table,static_cast<std::size_t>(s.bytes.pointer_table_device_bytes),
                             hipMemcpyHostToDevice,s.transfer),"segmented stable-table publication");
        check(hipEventRecord(s.ready,s.transfer),"segmented record ready");
        // Establish completion/errors before exposing raw bytes or releasing owner exclusion.
        check(hipEventSynchronize(s.ready),"segmented publication completion");
        s.current.swap(s.next);
        ++s.serial;
    } catch(...) {
        const auto original=std::current_exception();
        if(!grant_validated) {
            // A null/foreign/mismatched capability cannot authorize a return. In particular
            // never dereference a null grant or return guessed IDs after adapter failure.
            s.poison("allocator returned a null/foreign/mismatched segmented grant");
        } else if(!prior_fences_complete) {
            // A record may have succeeded just before a wait/other record failed. Do not
            // make the storage reusable and re-record an event which may still be recording.
            s.poison("segmented prior-user fencing failed; event/loan reuse is forbidden");
        } else {
            try {s.rollback(*frozen);}
            catch(const std::exception& error){s.poison(error.what());}
            catch(...){s.poison("unknown segmented rollback failure");}
        }
        std::rethrow_exception(original);
    }
    return Transaction(*this,s.serial,
        {static_cast<std::byte*>(s.payload)+s.config.window_offset,rounded},planned->copied_live_bytes);
}

void KVSegmentedStorage::wait_ready(std::uint64_t serial,hipStream_t stream) const {
    auto& s=*impl_;s.check_usable();
    require(s.active&&serial==s.serial,"stale segmented loan ticket");
    DeviceBinding binding(s.config.device);
    require_not_capturing(stream);
    check(hipStreamWaitEvent(stream,s.ready,0),"segmented user wait ready");
}

bool KVSegmentedStorage::close(std::uint64_t serial,std::span<const hipStream_t> streams) noexcept {
    auto& s=*impl_;
    try {
        s.check_usable();require(s.active&&s.grant&&serial==s.serial,"stale segmented close ticket");
        require(!streams.empty()&&streams.size()<=s.config.max_user_streams,"invalid borrower completion stream inventory");
        DeviceBinding binding(s.config.device);
        for(const auto stream:streams)require_not_capturing(stream);
        for(std::size_t i=0;i<streams.size();++i)
            check(hipEventRecord(s.finished[i],streams[i]),"segmented record actual borrower completion");
        for(std::size_t i=0;i<streams.size();++i)
            check(hipEventSynchronize(s.finished[i]),"segmented wait actual borrower completion");
        auto frozen=s.freeze();
        frozen->return_selected(*s.grant);
        s.grant.reset();s.active=false;
        return true;
    }catch(const std::exception& error){s.poison(error.what());}
    catch(...){s.poison("unknown segmented close failure");}
    return false; // grant and IDs remain excluded; no successful return is reported
}

void KVSegmentedStorage::abandon(std::uint64_t serial) noexcept {
    auto& s=*impl_;
    if(s.active&&serial==s.serial)s.poison("segmented transaction abandoned without completed borrower streams");
}
KVSegmentedStorage::Transaction::Transaction(KVSegmentedStorage& storage,std::uint64_t serial,
    KVSegmentedSpan span,std::uint64_t copied) noexcept
    :storage_(&storage),serial_(serial),copied_(copied),span_(span){}
KVSegmentedStorage::Transaction::~Transaction(){abandon();}
void KVSegmentedStorage::Transaction::abandon() noexcept {
    if(storage_)storage_->abandon(serial_);
    storage_=nullptr;span_={};
}
KVSegmentedStorage::Transaction::Transaction(Transaction&& other) noexcept
    :storage_(std::exchange(other.storage_,nullptr)),serial_(other.serial_),copied_(other.copied_),
     span_(std::exchange(other.span_,KVSegmentedSpan{})){}
KVSegmentedStorage::Transaction& KVSegmentedStorage::Transaction::operator=(Transaction&& other) noexcept {
    if(this!=&other){abandon();storage_=std::exchange(other.storage_,nullptr);serial_=other.serial_;
        copied_=other.copied_;span_=std::exchange(other.span_,KVSegmentedSpan{});}
    return *this;
}
void KVSegmentedStorage::Transaction::wait_ready(hipStream_t stream) const {
    require(storage_!=nullptr,"closed segmented ticket cannot wait ready");storage_->wait_ready(serial_,stream);
}
bool KVSegmentedStorage::Transaction::close_after_streams(std::span<const hipStream_t> streams) noexcept {
    if(!storage_)return true;
    const bool result=storage_->close(serial_,streams);
    storage_=nullptr;span_={};return result;
}

} // namespace ninfer::amd::memory
