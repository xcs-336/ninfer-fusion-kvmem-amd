#include "core/nvtx.h"
#include <windows.h>
#include <evntrace.h>
#include <TraceLoggingProvider.h>
#include <atomic>

TRACELOGGING_DEFINE_PROVIDER(g_ninfer_hip_provider, "NInfer.HIP",
    (0xea169b61, 0x0ed8, 0x47a6, 0xa6, 0x27, 0x44, 0x79, 0x36, 0x50, 0x0e, 0x39));
namespace {
struct Provider {
    Provider() { TraceLoggingRegister(g_ninfer_hip_provider); }
    ~Provider() { TraceLoggingUnregister(g_ninfer_hip_provider); }
} provider;
std::atomic<std::uint64_t> next_range{1};
}
namespace ninfer::nvtx {
std::uint64_t trace_begin(const char* name, Category category, std::uint64_t payload) noexcept {
    if (!TraceLoggingProviderEnabled(g_ninfer_hip_provider, TRACE_LEVEL_VERBOSE, 0)) { return 0; }
    const auto id = next_range.fetch_add(1, std::memory_order_relaxed);
    TraceLoggingWrite(g_ninfer_hip_provider, "RangeBegin", TraceLoggingLevel(TRACE_LEVEL_VERBOSE),
        TraceLoggingString(name, "Name"), TraceLoggingUInt32(static_cast<unsigned>(category), "Category"),
        TraceLoggingUInt64(payload, "Payload"), TraceLoggingUInt64(id, "RangeId"));
    return id;
}
void trace_end(std::uint64_t id) noexcept {
    if (id) {
        TraceLoggingWrite(g_ninfer_hip_provider, "RangeEnd", TraceLoggingLevel(TRACE_LEVEL_VERBOSE),
            TraceLoggingUInt64(id, "RangeId"));
    }
}
}
