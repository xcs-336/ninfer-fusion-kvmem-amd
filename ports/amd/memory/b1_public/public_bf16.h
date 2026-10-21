#pragma once
// SOURCE ONLY, default-OFF experimental public-wrapper ABI. This header extends
// no Core type: it requires the frozen isolated segmented Core view verbatim.
#include "ninfer/ops/softmax_attention.h"
#ifndef NINFER_AMD_SEGMENTED_POOL_ABI
#error "B1 table-aware wrapper requires the isolated segmented Core ABI"
#endif

namespace ninfer::ops {
void segmented_b1_causal_softmax_attention(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& kv_table_rows, AttentionHeadGeometry geometry,
    float scale, PagedKVBatchLayerView cache, CausalAttentionExecutionEnvelope envelope,
    WorkspaceArena& workspace, Tensor& out, hipStream_t stream, const Tensor* gate = nullptr);
std::size_t segmented_b1_causal_softmax_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, KvCacheStorage storage, CausalAttentionExecutionEnvelope envelope,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width);
int segmented_b1_causal_softmax_attention_route_family(
    AttentionHeadGeometry geometry, KvCacheStorage storage, CausalAttentionExecutionEnvelope envelope,
    std::int32_t batch_size, std::int32_t width);
} // namespace ninfer::ops
