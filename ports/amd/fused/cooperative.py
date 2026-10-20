"""Native HIP cooperative launch and compiled-kernel residency policy."""
from __future__ import annotations

import re


def transform(relative: str, text: str, helpers) -> str:
    if relative.endswith("bf16_gdn_gating_proj_gemm_mma.cuh"):
        if text.count("#include <cooperative_groups.h>") != 1:
            raise ValueError("missing GDN cooperative-groups include")
        return text.replace("#include <cooperative_groups.h>",
                            "#include <hip/hip_cooperative_groups.h>")
    if relative.endswith("bf16_gdn_gating_proj_kernels.h"):
        anchor = "namespace ninfer::ops::detail {\n"
        return text.replace(anchor, anchor + """
// Current device's actual native cooperative capacity for both token variants.
// Zero requests the existing ordinary MMA route without submitting GPU work.
std::int32_t native_gdn_gating_ctas_per_cu(std::int32_t heads, std::int32_t split_k) noexcept;
""", 1)
    if relative.endswith("bf16_gdn_gating_proj_kernels.cu"):
        text = text.replace("#include <algorithm>", "#include <algorithm>\n#include <array>\n#include <atomic>")
        first = text.index("template <class Geometry, int SplitK>\nconstexpr std::int32_t cooperative_resident_ctas_per_sm()")
        last = text.index("template <class Geometry, int SplitK, int Warps =", first)
        text = text[:first] + """template <class Geometry, int SplitK, int Warps = 8,
          bool NormalizeInput = false, int NormTokenCapacity = 0>
std::int32_t cooperative_resident_ctas_per_sm() noexcept {
    static_assert(SplitK > 1);
    static std::array<std::atomic<int>, 64> cache{};
    int device = 0, cooperative = 0, full = 0, predicated = 0;
    if (hipGetDevice(&device) != hipSuccess || device < 0 || device >= 64) return 0;
    const int known = cache[device].load(std::memory_order_relaxed);
    if (known != 0) return known < 0 ? 0 : known;
    if (hipDeviceGetAttribute(&cooperative, hipDeviceAttributeCooperativeLaunch, device) != hipSuccess)
        return 0;
    if (cooperative == 0) {
        cache[device].store(-1, std::memory_order_relaxed);
        return 0;
    }
    constexpr int shared = kBf16GdnSmemBytes<Geometry::kBlockN>;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&full,
        bf16_gdn_gating_proj_gemm_mma_kernel<Geometry, SplitK, true, Warps,
            NormalizeInput, NormTokenCapacity>, Warps * 32, shared) != hipSuccess ||
        hipOccupancyMaxActiveBlocksPerMultiprocessor(&predicated,
        bf16_gdn_gating_proj_gemm_mma_kernel<Geometry, SplitK, false, Warps,
            NormalizeInput, NormTokenCapacity>, Warps * 32, shared) != hipSuccess) return 0;
    const int result = std::min(full, predicated);
    cache[device].store(result == 0 ? -1 : result, std::memory_order_relaxed);
    return result;
}

""" + text[last:]
        first = text.index("                hipLaunchConfig_t config{};")
        last = text.index("            } else {", first)
        text = text[:first] + """                const auto* kernel_x = static_cast<const __hip_bfloat16*>(launch_x.data);
                const auto* kernel_norm = norm_weight != nullptr
                    ? static_cast<const __hip_bfloat16*>(norm_weight->data) : nullptr;
                auto* kernel_normalized = launch_normalized_x != nullptr
                    ? static_cast<__hip_bfloat16*>(launch_normalized_x->data) : nullptr;
                float kernel_eps = norm_eps;
                const auto* kernel_a = static_cast<const __hip_bfloat16*>(a_weight.qdata);
                const auto* kernel_b = static_cast<const __hip_bfloat16*>(b_weight.qdata);
                const auto* kernel_alog = static_cast<const float*>(A_log.data);
                const auto* kernel_bias = static_cast<const float*>(dt_bias.data);
                auto* kernel_partial = static_cast<float*>(workspace);
                auto* kernel_g = static_cast<float*>(launch_g.data);
                auto* kernel_beta = static_cast<float*>(launch_beta.data);
                std::int32_t kernel_tokens = launch_t;
                void* arguments[] = {&kernel_x, &kernel_norm, &kernel_normalized, &kernel_eps,
                    &kernel_a, &kernel_b, &kernel_alog, &kernel_bias, &kernel_partial,
                    &kernel_g, &kernel_beta, &kernel_tokens};
                CUDA_CHECK(hipLaunchCooperativeKernel(
                    bf16_gdn_gating_proj_gemm_mma_kernel<Geometry, SplitK, FullTokens, Warps,
                        NormalizeInput, NormTokenCapacity>, grid, block, arguments, kSmemBytes, stream));
""" + text[last:]
        before = """        constexpr std::int32_t kResidentCtasPerSm =
            cooperative_resident_ctas_per_sm<Geometry, SplitK>();"""
        after = """        const std::int32_t kResidentCtasPerSm =
            cooperative_resident_ctas_per_sm<Geometry, SplitK, Warps, NormalizeInput, NormTokenCapacity>();
        int native_device = 0, native_cus = 0;
        if (hipGetDevice(&native_device) != hipSuccess ||
            hipDeviceGetAttribute(&native_cus, hipDeviceAttributeMultiprocessorCount, native_device) != hipSuccess)
            return false;
        multiprocessor_count = native_cus;"""
        if text.count(before) != 1: raise ValueError("missing cooperative runtime occupancy use")
        text = text.replace(before, after)
        anchor = "} // namespace\n\nvoid bf16_gdn_gating_proj_gemv_launch"
        export = """} // namespace

std::int32_t native_gdn_gating_ctas_per_cu(std::int32_t heads, std::int32_t split_k) noexcept {
    if (heads == 48) {
        switch (split_k) {
        case 8: return cooperative_resident_ctas_per_sm<Bf16Gdn27Geometry, 8>();
        case 4: return cooperative_resident_ctas_per_sm<Bf16Gdn27Geometry, 4>();
        case 2: return cooperative_resident_ctas_per_sm<Bf16Gdn27Geometry, 2>();
        default: return 0;
        }
    }
    if (heads == 32) {
        switch (split_k) {
        case 32: return cooperative_resident_ctas_per_sm<Bf16Gdn35Geometry, 32>();
        case 16: return cooperative_resident_ctas_per_sm<Bf16Gdn35Geometry, 16>();
        case 8: return cooperative_resident_ctas_per_sm<Bf16Gdn35Geometry, 8>();
        case 4: return cooperative_resident_ctas_per_sm<Bf16Gdn35Geometry, 4>();
        case 2: return cooperative_resident_ctas_per_sm<Bf16Gdn35Geometry, 2>();
        default: return 0;
        }
    }
    return 0;
}

void bf16_gdn_gating_proj_gemv_launch"""
        if text.count(anchor) != 1: raise ValueError("missing native cooperative export anchor")
        return text.replace(anchor, export)
    if relative.endswith("bf16_gdn_gating_proj_plan.cpp"):
        first = text.index("// Per-SM resident-CTA occupancy")
        last = text.index("bool is_27(", first)
        text = text[:first] + """// The route intervals remain scheduling preferences. Native residency is a
// runtime property of the compiled gfx1201 kernel and the current HIP device.
std::int32_t device_sm_count() noexcept {
    int device = 0, count = 0;
    if (hipGetDevice(&device) != hipSuccess ||
        hipDeviceGetAttribute(&count, hipDeviceAttributeMultiprocessorCount, device) != hipSuccess)
        return 0;
    return count;
}

std::int32_t native_split_k(Bf16GdnGatingScheduleId schedule) noexcept {
    switch (schedule) {
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit32: return 32;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit16: return 16;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit8: return 8;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit4: return 4;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit2: return 2;
    default: return 0;
    }
}

std::int32_t resident_ctas_27(Bf16GdnGatingScheduleId schedule, std::int32_t cu_count) noexcept {
    return native_gdn_gating_ctas_per_cu(48, native_split_k(schedule)) * cu_count;
}

std::int32_t resident_ctas_35(Bf16GdnGatingScheduleId schedule, std::int32_t cu_count) noexcept {
    return native_gdn_gating_ctas_per_cu(32, native_split_k(schedule)) * cu_count;
}

""" + text[last:]
        for shape, tile, rows in ((27, 128, 3), (35, 64, 2)):
            text = helpers.body(text, rf"bool cooperative_{shape}_grid_is_resident\([^{{}}]+\)\s*noexcept\s*\{{",
                f"    return cooperative_grid_is_resident(schedule, cols, {tile}, {rows}, resident_ctas_{shape}(schedule, sm_count));")
        text = re.sub(r"    // sm_86 has[^\n]+\n(?:    //[^\n]*\n){3}",
                      "    // Original interval preferences; native admission queries actual kernel occupancy.\n", text)
        text = re.sub(r"    // Same progression, clamped[^\n]+\n(?:    //[^\n]*\n){2}",
                      "    // Native capability/occupancy may select a later complete ordinary MMA route.\n", text)
        text = re.sub(r"    // The route table's static_asserts[^\n]+\n(?:    //[^\n]*\n){3}",
                      "    // Recheck physical residency at launch; the ordinary MMA route preserves the mathematical contract.\n", text)
        before = "    if (is_35(problem) && problem.cols <= 16) {"
        if text.count(before) != 1: raise ValueError("missing norm/control cooperative admission")
        text = text.replace(before, """    if (is_35(problem) && problem.cols <= 16 &&
        candidate_is_legal(Bf16GdnGatingScheduleId::MmaCooperativeSplit32,
                           problem, device_sm_count())) {""")
    if relative.endswith("tests/ops/test_gdn_gating_proj.cpp"):
        # Only the NVIDIA-specific physical-policy oracle changes. All primal
        # FP64/FP32 outputs, tolerances, guards, domains and workspace checks stay.
        anchor = '#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h"'
        text = text.replace(anchor, anchor + '\n#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"')
        text = helpers.body(text, r"std::int32_t ctas_per_sm\([^{}]+\)\s*\{", """
    return ninfer::ops::detail::native_gdn_gating_ctas_per_cu(g.heads, split_of(schedule));""")
        before = """            check(plan.schedule == tuned(g, cols),
                  std::string(g.label) + ": 82 SMs changed the tuned route at " +
                      std::to_string(cols) + " columns");"""
        after = """            const Id preferred = tuned(g, cols);
            const std::string where = std::string(g.label) + ": 82 CUs, " + std::to_string(cols);
            check(grid_fits(g, plan.schedule, cols, 82), where + " exceeded native resident budget");
            if (grid_fits(g, preferred, cols, 82))
                check(plan.schedule == preferred, where + " left a preferred route that fits");
            else
                check(split_of(plan.schedule) < split_of(preferred), where + " failed to reduce splitting");"""
        if text.count(before) != 1: raise ValueError("missing hardware-specific planner oracle")
        text = text.replace(before, after)
    return text
