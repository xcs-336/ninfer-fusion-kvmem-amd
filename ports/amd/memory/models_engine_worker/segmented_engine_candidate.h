#pragma once
// Public only inside the opt-in candidate include root. No production ABI change.
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <utility>
#include "ninfer/types.h"
namespace ninfer {
struct SegmentedEngineStorageTag {explicit SegmentedEngineStorageTag()=default;};
inline constexpr SegmentedEngineStorageTag segmented_engine_storage{};
// Explicit SOURCE-only fixture controls, never environment variables or ordinary
// Engine defaults. They exercise real promise/retirement exception paths.
enum class SegmentedWorkerStartupTest : std::uint8_t {None,BeforeConsume,AfterConsume};
class SegmentedWorkerStartupFailure final:public std::exception {
    std::exception_ptr startup_,retirement_;
public:
    SegmentedWorkerStartupFailure(std::exception_ptr startup,std::exception_ptr retirement) noexcept
        :startup_(std::move(startup)),retirement_(std::move(retirement)){}
    [[nodiscard]] const char* what() const noexcept override {
        return "segmented worker startup and checked retirement both failed";
    }
    [[nodiscard]] std::exception_ptr startup_error() const noexcept{return startup_;}
    [[nodiscard]] std::exception_ptr retirement_error() const noexcept{return retirement_;}
};
struct SegmentedWorkerLifecycleCounts {std::uint64_t starts=0,exits=0,checked_retires=0;};
namespace segmented_worker_counters {
inline std::atomic<std::uint64_t> starts{0},exits{0},checked_retires{0};
}
inline SegmentedWorkerLifecycleCounts segmented_worker_lifecycle_counts() noexcept {
    return {segmented_worker_counters::starts.load(),segmented_worker_counters::exits.load(),
        segmented_worker_counters::checked_retires.load()};
}
// Pure early option admission, shared by the real public constructor and CPU
// boundary tests. Geometry still requires actual LoadPlan::config before upload.
inline void validate_segmented_engine_options(const EngineOptions& options) {
    if(options.max_context==0 || options.max_context>512 || options.max_concurrency!=1 ||
       options.devices.size()>1 || (!options.devices.empty() && options.devices.front()!=options.device) ||
       (options.kv_capacity.mode==KvCapacityMode::Explicit && options.kv_capacity.explicit_tokens<options.max_context) ||
       options.kv_cache!=KvCacheStorage::BFloat16 ||
       options.enable_vision || options.gdn_state_fp16 || options.gdn_state_storage!=GdnStateStorage::FP32 ||
       options.speculative.backend!=SpeculativeBackend::None || options.speculative.draft_tokens ||
       options.speculative.lookup_ngram || options.speculative.ngram_draft_tokens ||
       options.speculative.ngram_archive_bytes || options.speculative.mtp_attention_window ||
       options.context_cache.enabled || !options.context_cache.disk_kv_path.empty() ||
       options.context_cache.disk_kv_capacity_bytes || options.context_cache.disk_kv_restore ||
       options.context_cache.disk_kv_directstorage || options.device_profile!="off")
        throw std::invalid_argument("segmented public Engine requires explicit bounded B1 BF16 FP32-GDN no-prefix/no-spec/no-Vision single-device options");
}
inline EngineOptions validated_segmented_engine_options(EngineOptions options) {
    validate_segmented_engine_options(options);return options;
}
struct SegmentedEngineStorageSnapshot {
    bool enabled=false,controller_disabled=false,graph_observation_overflow=false;
    std::uint8_t controller_phase=0;
    std::uint32_t cold_device_syncs=0,graph_exec_count=0;
    std::uint64_t handoff_generation=0;
    std::size_t payload_device_bytes=0,pointer_table_device_bytes=0;
    std::size_t pinned_pointer_shadow_bytes=0,slot_maps_host_bytes=0;
    std::array<std::uintptr_t,64> graph_execs{};
    std::array<std::uint64_t,64> graph_instance_generations{},graph_update_revisions{};
    std::array<std::uint64_t,64> successful_graph_launches{};
    // Arrays/driver/control metadata are explicit diagnostics, not allocator savings.
};
} // namespace ninfer
