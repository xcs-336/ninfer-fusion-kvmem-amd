"""Retile the real Flash-Next NVFP4 gate/up schedules for 64KiB native LDS.

Stored packing, global scale strides, K64 accumulation order, activation
quantization, expert/token/path maps and epilogue casts are unchanged.
"""


def replace(source, before, after, count=1):
    actual = source.count(before)
    if actual != count:
        raise ValueError(f'Expected {count} MoE anchors, found {actual}: {before[:100]!r}')
    return source.replace(before, after)


def legacy_and_async(source):
    source = replace(source, '__launch_bounds__(128, 4)', '__launch_bounds__(32, 1)')
    source = replace(source, 'static_cast<int>(blockIdx.x) * 16', 'static_cast<int>(blockIdx.x) * 8')
    start = source.index('    const int warp_m = warp >> 1;')
    end = source.index('    const auto* exp_codes', start)
    source = source[:start] + '''    const int warp_m = 0; // native CTA owns one 16-row (8 gate/up pairs) tile
    const int warp_n = 0; // one wave owns eight query tokens
    const int lane   = tid & 31;

    extern __shared__ __align__(16) std::uint8_t s_dyn_mem[];
    auto& lds = *reinterpret_cast<NativeGateUpLds<8, kAsyncStaging ? 2 : 1>*>(s_dyn_mem);
    auto* s_w_codes = lds.weight_codes;
    auto* s_w_scales = lds.weight_scales;
    auto* s_a_codes_0 = lds.activation[0].codes;
    auto* s_a_scales_0 = lds.activation[0].scales;
    auto* s_a_codes_1 = lds.activation[kAsyncStaging ? 1 : 0].codes;
    auto* s_a_scales_1 = lds.activation[kAsyncStaging ? 1 : 0].scales;

''' + source[end:]
    # Mark original M=32 staging loops before reducing N=16 staging loops: this
    # avoids applying the N substitution to the newly halved M loop.
    source = replace(source, 'i < 32 * 1280', 'i < NativeWeightRows * 1280', 2)
    source = replace(source, 'i < 32 * 40', 'i < NativeWeightRows * 40', 2)
    source = source.replace('i < 16 * 1280', 'i < 8 * 1280')
    source = source.replace('i < 16 * 40', 'i < 8 * 40')
    source = source.replace('NativeWeightRows', '16')
    source = source.replace('i += 128 * 16', 'i += 32 * 16')
    source = source.replace('i += 128)', 'i += 32)')
    source = replace(source, 't_chunk += 16', 't_chunk += 8', 2)
    source = replace(source, '(t_chunk >> 4) & 1', '(t_chunk >> 3) & 1')
    source = replace(source, 'next_chunk = t_chunk + 16', 'next_chunk = t_chunk + 8')
    return source


def v2(source):
    source = replace(source, '__launch_bounds__(256, 1)', '__launch_bounds__(64, 1)')
    source = replace(source, 'static_cast<int>(blockIdx.x) * 16', 'static_cast<int>(blockIdx.x) * 8')
    start = source.index('    const int warp_n_grp = warp >> 2;')
    end = source.index('    const auto* exp_codes', start)
    source = source[:start] + '''    const int warp_m = 0; // one 16-row gate/up tile per CTA
    const int warp_n = warp; // two waves own token groups 0..7 and 8..15
    const int lane = tid & 31;

    extern __shared__ __align__(16) std::uint8_t s_dyn_mem[];
    auto& lds = *reinterpret_cast<NativeGateUpLds<16, 1>*>(s_dyn_mem);
    auto* s_w_codes = lds.weight_codes;
    auto* s_w_scales = lds.weight_scales;
    auto* s_a_codes = lds.activation[0].codes;
    auto* s_a_scales = lds.activation[0].scales;

''' + source[end:]
    # Both weight rows and activation rows are halved; each original cooperative
    # load and each token iteration keeps its original global source address.
    source = source.replace('i < 32 * 1280', 'i < 16 * 1280')
    source = source.replace('i < 32 * 40', 'i < 16 * 40')
    source = source.replace('i += 256 * 16', 'i += 64 * 16')
    source = source.replace('i += 256)', 'i += 64)')
    source = replace(source, 't_chunk += 32', 't_chunk += 16')
    source = replace(source, 't_chunk + 32 < count', 't_chunk + 16 < count')
    return source


def launch_function():
    arguments = '''
            static_cast<const std::uint8_t*>(workspace.act_codes.data),
            static_cast<const std::uint8_t*>(workspace.act_scales.data),
            static_cast<const std::int32_t*>(workspace.expert_offsets.data),
            static_cast<const std::int32_t*>(workspace.expert_counts.data),
            static_cast<const std::int32_t*>(workspace.active_expert_ids.data),
            static_cast<const std::int32_t*>(workspace.active_count.data),
            static_cast<const std::int32_t*>(workspace.grouped_tokens.data),
            static_cast<const std::int32_t*>(workspace.grouped_paths.data),
            reinterpret_cast<const std::uint8_t*>(weights.expert_gate_up.codes),
            reinterpret_cast<const std::uint8_t*>(weights.expert_gate_up.scales),
            weights.expert_gate_up.weight_scale_divisors,
            weights.expert_gate_up.code_bytes_per_expert,
            weights.expert_gate_up.scale_bytes_per_expert,
            static_cast<__hip_bfloat16*>(workspace.activations.data)'''
    text = '''void native_flash_next_moe_gate_up_launch(NativeGateUpSchedule schedule,
                                         const MoeWeights& weights,
                                         const FlashNextMoeWorkspace& workspace,
                                         hipStream_t stream) {
    static const bool initialized = [] {
        CUDA_CHECK(ninfer_hip_func_set_attribute(flash_next_moe_prefill_gate_up_mma_kernel<false>,
            hipFuncAttributeMaxDynamicSharedMemorySize, sizeof(NativeGateUpLds<8, 1>)));
        CUDA_CHECK(ninfer_hip_func_set_attribute(flash_next_moe_prefill_gate_up_mma_kernel<true>,
            hipFuncAttributeMaxDynamicSharedMemorySize, sizeof(NativeGateUpLds<8, 2>)));
        CUDA_CHECK(ninfer_hip_func_set_attribute(flash_next_moe_prefill_gate_up_mma_v2_kernel,
            hipFuncAttributeMaxDynamicSharedMemorySize, sizeof(NativeGateUpLds<16, 1>)));
        return true;
    }();
    (void)initialized;
    const dim3 grid(kIntermediate / 8, 512);
'''
    variants = [('Legacy', 'flash_next_moe_prefill_gate_up_mma_kernel<false>', 32, '8, 1'),
                ('Async', 'flash_next_moe_prefill_gate_up_mma_kernel<true>', 32, '8, 2'),
                ('V2', 'flash_next_moe_prefill_gate_up_mma_v2_kernel', 64, '16, 1')]
    for index, (name, kernel, threads, layout) in enumerate(variants):
        text += ('    if' if index == 0 else '    else if') + f' (schedule == NativeGateUpSchedule::{name}) {{\n'
        text += f'        {kernel}<<<grid, {threads}, sizeof(NativeGateUpLds<{layout}>), stream>>>('
        text += arguments + ');\n    }\n'
    text += '''    else { throw std::invalid_argument("invalid native MoE gate/up schedule"); }
    CUDA_CHECK(hipGetLastError());
}

'''
    return text


def transform(source):
    # Keep every original TU, every NVFP4 quantizer and the complete down path.
    start = source.index('// Step 2b: Grouped Expert Gate & Up Projection (Native NVFP4 Tensor Core MMA)')
    end = source.index('template <bool kAsyncStaging = false>', start)
    source = source[:start] + '''// Native gate/up: 16 stored rows (8 gate + 8 up), one wave and eight
// tokens per chunk. Actual aligned LDS is 34,656 bytes, or 46,208 bytes
// for the original double-buffered staging algorithm. Grid M is 640/8.
''' + source[end:]
    start = source.index('// Step 2c:')
    end = source.index('__global__ __launch_bounds__(256, 1)', start)
    source = source[:start] + '''// Step 2c: Native two-wave schedule: the same 16 weight rows and two
// eight-token groups. Actual aligned LDS is 46,208 bytes. The complete
// forty-step K64 contraction, packed scales and BF16 epilogue are retained.
''' + source[end:]
    first = source.index('template <bool kAsyncStaging = false>')
    second = source.index('// Step 2c:', first)
    third = source.index('// Step 3: Shared Expert Gate & Up Projection', second)
    source = source[:first] + legacy_and_async(source[first:second]) + v2(source[second:third]) + source[third:]
    start = source.index('            // 4. Grouped Expert Gate & Up (Native NVFP4 Tensor Core MMA)')
    end = source.index('            CUDA_CHECK(hipGetLastError());', start)
    source = source[:start] + '''            // The same complete production gate/up phase is used by the
            // bounded packed-input oracle. All native schedules fit 64KiB LDS.
            const char* staging_env = std::getenv("NINFER_FLASH_NEXT_MOE_STAGING");
            const bool staging_legacy = staging_env && (std::strcmp(staging_env, "0") == 0 ||
                std::strcmp(staging_env, "legacy") == 0 || std::strcmp(staging_env, "old") == 0);
            const bool staging_v1 = staging_env && (std::strcmp(staging_env, "1") == 0 ||
                std::strcmp(staging_env, "v1") == 0);
            const auto schedule = staging_legacy ? NativeGateUpSchedule::Legacy
                : staging_v1 ? NativeGateUpSchedule::Async : NativeGateUpSchedule::V2;
            native_flash_next_moe_gate_up_launch(schedule, weights, workspace, stream);
''' + source[end:]
    anchor = 'void flash_next_moe_kernels_launch('
    source = replace(source, anchor, launch_function() + anchor)
    return '#include "models/amd_moe_gate_up.h"\n' + source


def report():
    return {'mechanism': 'Flash-Next NVFP4 gate/up CTA resource geometry',
            'replacement': 'M16 (8 gate/up pairs), N8/N16 native WMMA schedules with actual sizeof LDS layouts',
            'lds_bytes': {'legacy': 34656, 'async': 46208, 'v2': 46208},
            'guarantee': 'original NVFP4 code/scales, 40 K64 steps, expert/token/path maps, quantizers, epilogue casts and down phase unchanged',
            'qualification': 'resource/address audit only until production phase packed FP64 oracle runs'}
