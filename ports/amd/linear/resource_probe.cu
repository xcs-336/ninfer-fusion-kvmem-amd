// Compile-time qualification of the real Q8 K-split storage type on the native source view.
// This does not launch a device or replace the original independent numerical suites.
#include "ops/linear/q8/q8_ksplit_mma.cuh"
#include "ops/linear/q8/q8_a16_sliced_k_mma.cuh"

namespace ninfer::ops::detail {
template <int Warps, int Tokens, Q8KSplitScaleAccess Access>
constexpr bool native_q8_storage_fits() {
    using Schedule = Q8KSplitSchedule<Warps, Tokens, 1, Access>;
    static_assert(sizeof(Q8KSplitSharedStorage<Schedule>) <= 65536);
    static_assert(sizeof(Q8KSplitSharedStorage<Schedule>) ==
                  Schedule::kNativeStagingBytes);
    static_assert(Schedule::kThreads == 32 * Schedule::kKWarps);
    static_assert(Schedule::kGroupK == 64 * Schedule::kKWarps);
    static_assert(Schedule::kRowsPerLoaderWarp * Schedule::kKWarps == 16);
    return true;
}

template <int Warps, int... Tokens>
constexpr bool native_q8_all_tiles_fit() {
    return ((native_q8_storage_fits<Warps, Tokens, Q8KSplitScaleAccess::Direct>() &&
             native_q8_storage_fits<Warps, Tokens, Q8KSplitScaleAccess::Shared>()) && ...);
}

static_assert(native_q8_all_tiles_fit<4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88>());
static_assert(native_q8_all_tiles_fit<8, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88>());
static_assert(native_q8_all_tiles_fit<16, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88>());
static_assert(Q8KSplitSchedule<16, 16, 1, Q8KSplitScaleAccess::Shared>::kKWarps == 16);
static_assert(Q8KSplitSchedule<16, 24, 1, Q8KSplitScaleAccess::Shared>::kKWarps == 8);
static_assert(Q8KSplitSchedule<8, 48, 1, Q8KSplitScaleAccess::Shared>::kKWarps == 8);
static_assert(Q8KSplitSchedule<8, 56, 1, Q8KSplitScaleAccess::Shared>::kKWarps == 4);

template <int Warps, int Tokens, int Stages, Q8ScaleAccess Access>
constexpr bool native_q8_sliced_storage_fits() {
    using Schedule = Q8A16SlicedKMmaSchedule<Tokens, Warps, Stages, 1, Access>;
    static_assert(sizeof(Q8SlicedKSharedStorage<Schedule>) <= 65536);
    static_assert(sizeof(Q8SlicedKSharedStorage<Schedule>) == Schedule::kSharedBytes);
    static_assert(Schedule::kThreads == 32 * Schedule::kKWarps);
    static_assert(Schedule::kBlockK == 64 * Schedule::kKWarps);
    static_assert(Schedule::kRowsPerLoaderWarp * Schedule::kKWarps == 16);
    return true;
}

template <int Warps, int... Tokens>
constexpr bool native_q8_all_sliced_tiles_fit() {
    return ((native_q8_sliced_storage_fits<Warps, Tokens, 1, Q8ScaleAccess::Direct>() &&
             native_q8_sliced_storage_fits<Warps, Tokens, 1, Q8ScaleAccess::Shared>() &&
             native_q8_sliced_storage_fits<Warps, Tokens, 2, Q8ScaleAccess::Direct>() &&
             native_q8_sliced_storage_fits<Warps, Tokens, 2, Q8ScaleAccess::Shared>()) && ...);
}

static_assert(native_q8_all_sliced_tiles_fit<2, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88>());
static_assert(native_q8_all_sliced_tiles_fit<4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88>());
static_assert(native_q8_all_sliced_tiles_fit<8, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88>());
static_assert(native_q8_all_sliced_tiles_fit<16, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88>());
} // namespace ninfer::ops::detail
