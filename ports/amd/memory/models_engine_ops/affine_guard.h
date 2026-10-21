#pragma once
#include "core/paged_kv_cache.h"
#include <stdexcept>
#include <string>
#if !defined(NINFER_AMD_SEGMENTED_CONTROLLER_HANDOFF_ABI) || NINFER_AMD_SEGMENTED_CONTROLLER_HANDOFF_ABI != 1
#error "Affine cache closure must be rebuilt against the full controller Core ABI"
#endif
namespace ninfer::amd::models_engine_ops {
template<class Cache>
inline void reject_table_backed_affine_entry(const Cache& cache, const char* operation) {
    if (cache.k_page_bases != nullptr || cache.v_page_bases != nullptr)
        throw std::invalid_argument(std::string(operation) +
            ": table-backed cache is outside this affine entry; use the admitted Main BF16 B1 bridge");
}
} // namespace ninfer::amd::models_engine_ops
