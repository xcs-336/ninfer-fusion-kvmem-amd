#pragma once
// SOURCE ONLY: requires prepare_segmented_pool.py's isolated Core ABI.
#include "core/paged_kv_cache.h"
#include "kv_segmented_storage.h"
#include <memory>
#include <mutex>

#ifndef NINFER_AMD_SEGMENTED_POOL_ABI
#error "Compile only with the isolated prepare_segmented_pool.py Core view; production Core ABI is incompatible"
#endif

namespace ninfer::amd::memory {

// The referenced original Core pool is the allocator; no second allocation ledger is
// maintained. Every original pool mutation, including RAII release callbacks, must
// have the isolated source view's INTERNAL guard. Raw production Core is incompatible.
class KVSegmentedPoolOwner final:public KVSegmentedAllocatorOwner {
public:
    explicit KVSegmentedPoolOwner(const KVSegmentedStorage::Config& config);
    ~KVSegmentedPoolOwner() override;
    KVSegmentedPoolOwner(const KVSegmentedPoolOwner&)=delete;
    // The owner outlives BOTH the bound pool and storage. Binding is startup only.
    void bind(DeviceKVPagePool&,KVSegmentedStorage&);
    [[nodiscard]] std::unique_ptr<KVSegmentedAllocatorFreeze> freeze() override;
private:
    class Freeze;
    class Grant;
    [[nodiscard]] KVSegmentedFrozenAllocation snapshot_locked() const;
    [[nodiscard]] std::unique_ptr<KVSegmentedLentPages> lend_locked(std::span<const std::uint32_t>);
    void return_locked(KVSegmentedLentPages&);
    [[nodiscard]] std::recursive_mutex& control_mutex()const;
    KVSegmentedStorage::Config config_;
    DeviceKVPagePool* pool_=nullptr;
    std::uint64_t next_nonce_=1;
    Grant* active_grant_=nullptr; // non-owning unique capability held by storage
};

// Exactly one physical payload: original pool's Tensor views bind the storage's
// existing allocation. No DeviceArena or old persistent KV payload is created here.
// Its lifetime requires every original lease/reservation and GPU user already ended.
class KVSegmentedNativePool {
public:
    KVSegmentedNativePool(const KVSegmentedStorage::Config&,const DeviceKVPagePoolLayout&);
    ~KVSegmentedNativePool();
    KVSegmentedNativePool(const KVSegmentedNativePool&)=delete;
    [[nodiscard]] DeviceKVPagePool& page_pool() noexcept{return *pool_;}
    [[nodiscard]] KVSegmentedStorage& storage() noexcept{return *storage_;}
private:
    KVSegmentedPoolOwner owner_; // outlives the two following members
    std::unique_ptr<DeviceKVPagePool> pool_;
    std::unique_ptr<KVSegmentedStorage> storage_; // destroyed before original pool
};
} // namespace ninfer::amd::memory
