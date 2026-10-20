#pragma once
#include <hip/hip_runtime_api.h>
#include <cstddef>

// Preserve both original CUDA overloads without touching any argument syntax.
// The three-argument form's third argument is flags, never an error-node pointer.
inline hipError_t ninfer_hip_graph_instantiate(hipGraphExec_t* execution,
                                               hipGraph_t graph,
                                               unsigned long long flags) {
    return hipGraphInstantiateWithFlags(execution, graph, flags);
}

// The five-argument form retains the legacy error-node and log-buffer outputs.
inline hipError_t ninfer_hip_graph_instantiate(hipGraphExec_t* execution,
                                               hipGraph_t graph,
                                               hipGraphNode_t* error_node,
                                               char* log_buffer,
                                               std::size_t buffer_bytes) {
    return hipGraphInstantiate(execution, graph, error_node, log_buffer, buffer_bytes);
}
