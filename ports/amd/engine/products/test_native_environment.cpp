#include "product/amd_environment.h"
#include "ninfer_bench_support.h"
#include "serve/request_log.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <string>

namespace {
using Json = nlohmann::json;
int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::fprintf(stderr, "native environment: %s\n", message);
    return 1;
}
int native_record(const Json& record) {
    int failures = 0;
    failures += check(record.at("backend") == "hip", "actual backend missing");
    failures += check(record.at("gcn_arch_name") == "gfx1201", "actual GCN architecture missing");
    failures += check(record.at("hip_compile_version") == "7.15.26333", "HIP compile version differs");
    failures += check(record.at("hip_runtime_version") == "7.15.26334", "HIP runtime version differs");
    failures += check(record.at("hip_driver_version") == "7.15.26335", "HIP driver version differs");
    for (const char* key : {"cuda_compile_version", "cuda_runtime_version", "cuda_driver_version",
                           "compute_capability_major", "compute_capability_minor"}) {
        failures += check(!record.contains(key), "HIP record contains CUDA-specific metadata");
    }
    return failures;
}
} // namespace

int main() {
    try {
        int failures = 0;
        failures += check(ninfer::amd::hip_version_string(71526333) == "7.15.26333",
                          "HIP encoded major/minor/patch decoded incorrectly");
        failures += check(ninfer::amd::hip_version_string(100000000) == "10.0.0",
                          "HIP two-digit major decoded incorrectly");
        failures += check(ninfer::amd::hip_version_string(0).empty(), "unknown HIP version is not empty");
        ninfer::bench::BenchEnvironment benchmark{};
        benchmark.backend = "hip";
        benchmark.gpu_name = "AMD Radeon RX 9070";
        benchmark.gcn_arch_name = "gfx1201";
        benchmark.hip_compile_version = "7.15.26333";
        benchmark.hip_runtime_version = "7.15.26334";
        benchmark.hip_driver_version = "7.15.26335";
        // Deliberate hostile values must never leak into a HIP report.
        benchmark.cuda_runtime_version = "wrong-CUDA-version";
        benchmark.cuda_driver_version = "wrong-CUDA-driver";
        const Json report = Json::parse(ninfer::bench::format_json(benchmark, "ninfer_bench", {}));
        failures += native_record(report.at("environment"));
        const auto table = ninfer::bench::format_table(benchmark, {});
        failures += check(table.find("hip:") != std::string::npos && table.find("gfx1201") != std::string::npos,
                          "human report lacks actual HIP platform");
        failures += check(table.find("wrong-CUDA") == std::string::npos, "human HIP report leaks CUDA versions");

        ninfer::serve::ServerLogEnvironment environment{};
        environment.backend = "hip";
        environment.gpu_name = benchmark.gpu_name;
        environment.gcn_arch_name = benchmark.gcn_arch_name;
        environment.hip_compile_version = benchmark.hip_compile_version;
        environment.hip_runtime_version = benchmark.hip_runtime_version;
        environment.hip_driver_version = benchmark.hip_driver_version;
        environment.compute_capability_major = 120;
        environment.compute_capability_minor = 99;
        environment.cuda_compile_version = "wrong-CUDA-compile";
        // The production formatter accepts resolved Engine options, as in the
        // original request-log fixture. Empty optionals are not its contract.
        ninfer::EngineOptions engine_options{};
        engine_options.context_cache.device_state_slots = 2;
        engine_options.context_cache.host_state_slots = 3;
        engine_options.context_cache.host_kv_capacity_bytes = 64ULL << 20;
        engine_options.context_cache.max_private_continuations = 4;
        engine_options.context_cache.max_shared_prefixes = 2;
        engine_options.context_cache.max_long_anchors_per_continuation = 2;
        engine_options.context_cache.max_cache_markers_per_request = 4;
        const Json server = Json::parse(ninfer::serve::format_server_start_json(
            "native-test", 0, {}, engine_options, {}, "native-model", {}, {}, environment, {}));
        failures += native_record(server.at("environment"));
        failures += check(server.at("environment").at("gpu_name") == benchmark.gpu_name,
                          "native GPU identity changed");
        std::printf("native HIP product environment format: %s (versions, architecture, omitted CUDA metadata)\n",
                    failures == 0 ? "PASS" : "FAIL");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "native product metadata: %s\n", error.what());
        return 1;
    }
}
