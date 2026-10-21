# SOURCE ONLY leaf. Root must explicitly include it AFTER SegmentedPool.cmake.
# Default OFF, no Engine/Model/production source rewrite and no automatic GPU test.
option(NINFER_AMD_SEGMENTED_BF16_OPS "Build isolated table-aware actual BF16 append/Attention candidate (unqualified)" OFF)
if(NOT NINFER_AMD_SEGMENTED_BF16_OPS)
  return()
endif()
if(NOT TARGET ninfer_segmented_pool_candidate_amd OR
   NOT DEFINED NINFER_AMD_SEGMENTED_POOL_VIEW OR NOT COMMAND ninfer_native_hip)
  message(FATAL_ERROR "Table-aware BF16 candidate requires explicit NINFER_AMD_SEGMENTED_POOL=ON and its isolated Core view")
endif()
set(_segmented_bf16_inputs
  "${CMAKE_CURRENT_LIST_FILE}"
  "${CMAKE_CURRENT_LIST_DIR}/segmented_bf16_ops_overlay.py"
  "${CMAKE_CURRENT_LIST_DIR}/segmented_bf16_view.cuh"
  "${CMAKE_CURRENT_LIST_DIR}/segmented_bf16_launch.cuh"
  "${CMAKE_CURRENT_LIST_DIR}/tests/kv_segmented_bf16_attention_test.cu"
  "${CMAKE_CURRENT_LIST_DIR}/tests/verify_segmented_bf16_map.py"
  "${CMAKE_CURRENT_LIST_DIR}/tests/verify_segmented_pool_map.py"
  "${_engine}/tests/ops/op_check.h"
  "${NINFER_AMD_SEGMENTED_POOL_VIEW}/segmented-source-receipt.json"
  "${_generated}/src/ops/kv_cache/append/kernel.cuh"
  "${_generated}/src/ops/kv_cache/append/geometry.cuh"
  "${_generated}/src/ops/softmax_attention/dense/causal_cache/small_t_bf16.cuh"
  "${_generated}/src/ops/softmax_attention/dense/causal_cache/small_t.cuh"
  "${_generated}/src/ops/softmax_attention/dense/causal_cache/prompt_bf16.cuh"
  "${_generated}/src/ops/softmax_attention/dense/causal_cache/prompt_common.cuh"
  "${_generated}/src/ops/softmax_attention/dense/causal_cache/geometry.cuh"
  "${_generated}/src/ops/kernel/paged_kv_address.cuh")
set(_segmented_bf16_recipe "isolated-actual-native-bf16-page-table-v1\n")
foreach(_segmented_bf16_input IN LISTS _segmented_bf16_inputs)
  file(SHA256 "${_segmented_bf16_input}" _segmented_bf16_digest)
  string(APPEND _segmented_bf16_recipe "${_segmented_bf16_digest}\n")
endforeach()
string(SHA256 _segmented_bf16_sha "${_segmented_bf16_recipe}")
set(NINFER_AMD_SEGMENTED_BF16_VIEW
  "${CMAKE_CURRENT_BINARY_DIR}/segmented-bf16-candidate/${_segmented_bf16_sha}")
execute_process(COMMAND "${NINFER_PYTHON}" -B "${CMAKE_CURRENT_LIST_DIR}/segmented_bf16_ops_overlay.py"
  --baseline "${_generated}" --output "${NINFER_AMD_SEGMENTED_BF16_VIEW}" --verify-existing
  RESULT_VARIABLE _segmented_bf16_prepared ERROR_VARIABLE _segmented_bf16_error)
if(NOT _segmented_bf16_prepared EQUAL 0)
  message(FATAL_ERROR "BF16 native body isolation refused: ${_segmented_bf16_error}")
endif()
add_executable(ninfer_amd_segmented_bf16_attention_test
  "${CMAKE_CURRENT_LIST_DIR}/tests/kv_segmented_bf16_attention_test.cu")
ninfer_native_hip(ninfer_amd_segmented_bf16_attention_test)
target_compile_features(ninfer_amd_segmented_bf16_attention_test PRIVATE cxx_std_20)
target_compile_options(ninfer_amd_segmented_bf16_attention_test PRIVATE
  -O3 -fno-fast-math -fhip-fp32-correctly-rounded-divide-sqrt -fms-runtime-lib=dll)
# Device bodies retain normal production contraction behavior. The independent
# FP64 oracle disables contraction locally with a host-function pragma.
target_include_directories(ninfer_amd_segmented_bf16_attention_test BEFORE PRIVATE
  "${NINFER_AMD_SEGMENTED_BF16_VIEW}"
  "${NINFER_AMD_SEGMENTED_POOL_VIEW}"
  "${NINFER_AMD_SEGMENTED_POOL_VIEW}/src"
  "${NINFER_AMD_SEGMENTED_POOL_VIEW}/amd-memory"
  "${_engine}/tests/ops")
target_link_libraries(ninfer_amd_segmented_bf16_attention_test PRIVATE ninfer_segmented_pool_candidate_amd)
target_link_options(ninfer_amd_segmented_bf16_attention_test PRIVATE
  "-Wl,/MAP:${CMAKE_CURRENT_BINARY_DIR}/ninfer_amd_segmented_bf16_attention_test.map")
add_custom_command(TARGET ninfer_amd_segmented_bf16_attention_test POST_BUILD
  COMMAND "${NINFER_PYTHON}" -B "${CMAKE_CURRENT_LIST_DIR}/tests/verify_segmented_bf16_map.py"
    --map "${CMAKE_CURRENT_BINARY_DIR}/ninfer_amd_segmented_bf16_attention_test.map"
  VERBATIM COMMENT "Reject old view Core/Op/test-support extraction from table-aware BF16 candidate")
# The actual final map still must prove that old Core paged_kv_cache.obj and all
# public Attention/append Op/test-support members were not extracted. Source
# preparation or a successful link alone is NOT mathematical/ABI qualification.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_segmented_bf16_inputs})
