#pragma once

#include "models/qwen3_8_flash_next/impl/model_view.h"
#include "models/qwen3_8_flash_next/impl/moe_workspace.h"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::detail {

template <int ActivationRows, int Buffers>
struct alignas(16) NativeGateUpLds {
    alignas(16) std::uint8_t weight_codes[16 * 1280];
    alignas(16) std::uint8_t weight_scales[16 * 164];
    struct alignas(16) Activation {
        alignas(16) std::uint8_t codes[ActivationRows * 1280];
        alignas(16) std::uint8_t scales[ActivationRows * 164];
    } activation[Buffers];
};
static_assert(sizeof(NativeGateUpLds<8, 1>) == 34656);
static_assert(sizeof(NativeGateUpLds<8, 2>) == 46208);
static_assert(sizeof(NativeGateUpLds<16, 1>) == 46208);
static_assert(sizeof(NativeGateUpLds<8, 1>) <= 65536);
static_assert(sizeof(NativeGateUpLds<8, 2>) <= 65536);
static_assert(sizeof(NativeGateUpLds<16, 1>) <= 65536);

enum class NativeGateUpSchedule { Legacy, Async, V2 };

// The complete original production gate/up phase. The full MoE launcher and
// its bounded packed-input FP64 qualification call this same entry point.
void native_flash_next_moe_gate_up_launch(NativeGateUpSchedule schedule,
                                         const MoeWeights& weights,
                                         const FlashNextMoeWorkspace& workspace,
                                         hipStream_t stream);

} // namespace ninfer::models::qwen3_8_flash_next::detail
