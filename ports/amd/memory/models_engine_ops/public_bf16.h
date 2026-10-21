#pragma once
// NEW full-controller-Core ABI bridge. This is not the old partial-Core B1 target.
#include "ninfer/ops/softmax_attention.h"
#include <stdexcept>
#if !defined(NINFER_AMD_SEGMENTED_CONTROLLER_HANDOFF_ABI) || NINFER_AMD_SEGMENTED_CONTROLLER_HANDOFF_ABI != 1
#error "Models Engine Ops requires the same full controller-handoff Core ABI"
#endif
#define NINFER_AMD_SEGMENTED_ENGINE_OPS_ABI 1

namespace ninfer::ops {
void segmented_b1_causal_softmax_attention(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    AttentionHeadGeometry, float, PagedKVBatchLayerView, CausalAttentionExecutionEnvelope,
    WorkspaceArena&, Tensor&, hipStream_t, const Tensor* gate = nullptr);
std::size_t segmented_b1_causal_softmax_attention_workspace_capacity_bytes(
    AttentionHeadGeometry, KvCacheStorage, CausalAttentionExecutionEnvelope,
    std::int32_t, std::int32_t, std::int32_t);
int segmented_b1_causal_softmax_attention_route_family(
    AttentionHeadGeometry, KvCacheStorage, CausalAttentionExecutionEnvelope,
    std::int32_t, std::int32_t);

// Only Main Text calls use this router. A one-sided table is invalid, never an
// excuse to dispatch affine arithmetic against the other table-backed plane.
inline void models_engine_main_causal_softmax_attention(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, AttentionHeadGeometry geometry,
    float scale, PagedKVBatchLayerView cache, CausalAttentionExecutionEnvelope envelope,
    WorkspaceArena& workspace, Tensor& out, hipStream_t stream, const Tensor* gate = nullptr) {
    if ((cache.k_page_bases == nullptr) != (cache.v_page_bases == nullptr))
        throw std::invalid_argument("segmented Engine Main requires paired K/V page tables");
    if (cache.k_page_bases != nullptr) {
        segmented_b1_causal_softmax_attention(q, k, v, positions, valid_columns, table_rows,
            geometry, scale, cache, envelope, workspace, out, stream, gate);
    } else {
        causal_softmax_attention(q, k, v, positions, valid_columns, table_rows,
            geometry, scale, cache, envelope, workspace, out, stream, gate);
    }
}
} // namespace ninfer::ops
