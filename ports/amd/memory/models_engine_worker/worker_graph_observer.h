#pragma once
// Host-only successful GraphLaunch witness. No GPU payload or synchronization.
#include <hip/hip_runtime_api.h>
#include <array>
#include <cstdint>
#include <limits>
#include <mutex>
namespace ninfer::amd::models_engine_worker {
using GraphObserver=void(*)(void*,std::uintptr_t,std::uint64_t,std::uint64_t) noexcept;
struct GraphIdentity {std::uintptr_t exec=0;std::uint64_t generation=0,revision=0;};
// Captures/instantiations happen on startup; workers do not inherit owner TLS.
// This bounded process registry records REAL successful instantiate/update/destroy
// events. Generation rejects address reuse; revision rejects intervening updates.
inline std::mutex registry_mutex;
inline std::array<GraphIdentity,256> registry{};
inline std::uint64_t next_generation=1;
inline bool registry_failed=false;
inline void registered_graph_exec(hipGraphExec_t exec) noexcept {
    std::lock_guard lock(registry_mutex);
    const auto address=reinterpret_cast<std::uintptr_t>(exec);
    for(const auto& entry:registry)if(entry.exec==address){registry_failed=true;return;}
    if(next_generation==std::numeric_limits<std::uint64_t>::max()){registry_failed=true;return;}
    for(auto& entry:registry)if(!entry.exec){entry={address,next_generation++,0};return;}
    registry_failed=true;
}
inline void updated_graph_exec(hipGraphExec_t exec) noexcept {
    std::lock_guard lock(registry_mutex);
    for(auto& entry:registry)if(entry.exec==reinterpret_cast<std::uintptr_t>(exec)) {
        if(entry.revision==std::numeric_limits<std::uint64_t>::max())registry_failed=true;
        else ++entry.revision;
        return;
    }
    registry_failed=true;
}
inline hipError_t observed_graph_destroy(hipGraphExec_t exec) noexcept {
    const auto status=::hipGraphExecDestroy(exec);
    std::lock_guard lock(registry_mutex);
    if(status!=hipSuccess){registry_failed=true;return status;}
    for(auto& entry:registry)if(entry.exec==reinterpret_cast<std::uintptr_t>(exec)){entry={};return status;}
    registry_failed=true;return status;
}
inline thread_local GraphObserver observer=nullptr;
inline thread_local void* observer_context=nullptr;
inline void install_graph_observer(void* context,GraphObserver callback) noexcept {
    observer_context=context;observer=callback;
}
inline void clear_graph_observer(void* context) noexcept {
    if(observer_context==context){observer=nullptr;observer_context=nullptr;}
}
inline hipError_t observed_graph_launch(hipGraphExec_t exec,hipStream_t stream) {
    const auto status=::hipGraphLaunch(exec,stream);
    if(status==hipSuccess && observer) {
        GraphIdentity identity;
        {std::lock_guard lock(registry_mutex);
         if(!registry_failed)for(const auto& entry:registry)
            if(entry.exec==reinterpret_cast<std::uintptr_t>(exec)){identity=entry;break;}}
        observer(observer_context,reinterpret_cast<std::uintptr_t>(exec),identity.generation,identity.revision);
    }
    return status;
}
} // namespace ninfer::amd::models_engine_worker
