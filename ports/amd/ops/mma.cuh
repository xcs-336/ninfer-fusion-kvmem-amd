#pragma once
#include "ops/common/ldmatrix.cuh"
#include "ops/common/mma_bf16_gfx12.h"

namespace ninfer::ops {
// Existing production BF16 tile kernels keep their packed-register interface;
// the arithmetic is executed by native gfx1201 WMMA. Other MMA formats require
// their own native implementation and qualification before their routes build.
using ::ninfer::amd_candidate::mma_bf16;
} // namespace ninfer::ops
