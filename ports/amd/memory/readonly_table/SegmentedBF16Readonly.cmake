# NEW independent candidate. Include explicitly AFTER the frozen SegmentedBF16Ops leaf.
# No original option/source/view is changed and no GPU test is registered or run here.
option(NINFER_AMD_SEGMENTED_BF16_READONLY "Build ordinary U64 table-load six-case BF16 comparison (unqualified)" OFF)
if(NOT NINFER_AMD_SEGMENTED_BF16_READONLY)
  return()
endif()
if(NOT TARGET ninfer_amd_segmented_bf16_attention_test OR
   NOT TARGET ninfer_segmented_pool_candidate_amd OR
   NOT DEFINED NINFER_AMD_SEGMENTED_BF16_VIEW OR
   NOT DEFINED NINFER_AMD_SEGMENTED_POOL_VIEW OR
   NOT DEFINED _generated OR NOT DEFINED _engine OR NOT DEFINED NINFER_PYTHON)
  message(FATAL_ERROR "Readonly BF16 candidate needs the frozen isolated Core and BF16 options/source views")
endif()
get_filename_component(_readonly_bf16_memory "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(_readonly_bf16_inputs
  "${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_DIR}/prepare_readonly_bf16.py"
  "${_readonly_bf16_memory}/SegmentedBF16Ops.cmake"
  "${_readonly_bf16_memory}/segmented_bf16_ops_overlay.py"
  "${_readonly_bf16_memory}/segmented_bf16_view.cuh"
  "${_readonly_bf16_memory}/segmented_bf16_launch.cuh"
  "${_readonly_bf16_memory}/tests/kv_segmented_bf16_attention_test.cu"
  "${_readonly_bf16_memory}/tests/verify_segmented_bf16_map.py"
  "${_readonly_bf16_memory}/tests/verify_segmented_pool_map.py"
  "${_engine}/tests/ops/op_check.h"
  "${NINFER_AMD_SEGMENTED_BF16_VIEW}/segmented-bf16-source-receipt.json"
  "${NINFER_AMD_SEGMENTED_POOL_VIEW}/segmented-source-receipt.json")
foreach(_readonly_bf16_name IN ITEMS append small_t prompt view launch)
  list(APPEND _readonly_bf16_inputs "${NINFER_AMD_SEGMENTED_BF16_VIEW}/amd-memory/segmented_bf16_${_readonly_bf16_name}.cuh")
endforeach()
foreach(_readonly_bf16_name IN ITEMS
    src/core/paged_kv_cache.h src/core/paged_kv_cache.cpp
    amd-memory/kv_segmented_storage.h amd-memory/kv_segmented_storage.cpp
    amd-memory/kv_segmented_pool_control.h amd-memory/kv_segmented_pool_owner.h
    amd-memory/kv_segmented_pool_owner.cpp amd-memory/kv_segmented_loan_plan.h
    amd-memory/kv_segmented_loan_plan.cpp)
  list(APPEND _readonly_bf16_inputs "${NINFER_AMD_SEGMENTED_POOL_VIEW}/${_readonly_bf16_name}")
endforeach()
foreach(_readonly_bf16_name IN ITEMS
    src/core/paged_kv_cache.h src/core/paged_kv_cache.cpp
    src/ops/kv_cache/append/kernel.cuh src/ops/kv_cache/append/geometry.cuh
    src/ops/softmax_attention/dense/causal_cache/small_t_bf16.cuh
    src/ops/softmax_attention/dense/causal_cache/small_t.cuh
    src/ops/softmax_attention/dense/causal_cache/prompt_bf16.cuh
    src/ops/softmax_attention/dense/causal_cache/prompt_common.cuh
    src/ops/softmax_attention/dense/causal_cache/geometry.cuh
    src/ops/kernel/paged_kv_address.cuh)
  list(APPEND _readonly_bf16_inputs "${_generated}/${_readonly_bf16_name}")
endforeach()
set(_readonly_bf16_recipe "segmented-bf16-ordinary-table-read-v1\n")
foreach(_readonly_bf16_input IN LISTS _readonly_bf16_inputs)
  file(SHA256 "${_readonly_bf16_input}" _readonly_bf16_sha)
  string(APPEND _readonly_bf16_recipe "${_readonly_bf16_sha}\n")
endforeach()
string(SHA256 _readonly_bf16_recipe_sha "${_readonly_bf16_recipe}")
set(NINFER_AMD_SEGMENTED_BF16_READONLY_VIEW
    "${CMAKE_CURRENT_BINARY_DIR}/segmented-bf16-readonly/${_readonly_bf16_recipe_sha}")
execute_process(COMMAND "${NINFER_PYTHON}" -B "${CMAKE_CURRENT_LIST_DIR}/prepare_readonly_bf16.py"
  --native-root "${_generated}" --baseline-view "${NINFER_AMD_SEGMENTED_BF16_VIEW}"
  --pool-view "${NINFER_AMD_SEGMENTED_POOL_VIEW}" --engine-root "${_engine}"
  --output "${NINFER_AMD_SEGMENTED_BF16_READONLY_VIEW}" --verify-existing
  RESULT_VARIABLE _readonly_bf16_result ERROR_VARIABLE _readonly_bf16_error)
if(NOT _readonly_bf16_result EQUAL 0)
  message(FATAL_ERROR "Readonly BF16 source isolation refused: ${_readonly_bf16_error}")
endif()
# Same frozen source TU, independent object/executable, new include root first.
# EXCLUDE_FROM_ALL keeps this experiment out of ordinary all-target builds even ON.
add_executable(ninfer_amd_segmented_bf16_readonly_test EXCLUDE_FROM_ALL
  "${_readonly_bf16_memory}/tests/kv_segmented_bf16_attention_test.cu")
ninfer_native_hip(ninfer_amd_segmented_bf16_readonly_test)
target_compile_features(ninfer_amd_segmented_bf16_readonly_test PRIVATE cxx_std_20)
target_compile_options(ninfer_amd_segmented_bf16_readonly_test PRIVATE
  -O3 -fno-fast-math -fhip-fp32-correctly-rounded-divide-sqrt -fms-runtime-lib=dll)
target_include_directories(ninfer_amd_segmented_bf16_readonly_test BEFORE PRIVATE
  "${NINFER_AMD_SEGMENTED_BF16_READONLY_VIEW}"
  "${NINFER_AMD_SEGMENTED_POOL_VIEW}"
  "${NINFER_AMD_SEGMENTED_POOL_VIEW}/src"
  "${NINFER_AMD_SEGMENTED_POOL_VIEW}/amd-memory"
  "${_engine}/tests/ops")
target_link_libraries(ninfer_amd_segmented_bf16_readonly_test PRIVATE ninfer_segmented_pool_candidate_amd)
target_link_options(ninfer_amd_segmented_bf16_readonly_test PRIVATE
  "-Wl,/MAP:${CMAKE_CURRENT_BINARY_DIR}/ninfer_amd_segmented_bf16_readonly_test.map")
add_custom_command(TARGET ninfer_amd_segmented_bf16_readonly_test POST_BUILD
  COMMAND "${NINFER_PYTHON}" -B "${_readonly_bf16_memory}/tests/verify_segmented_bf16_map.py"
    --map "${CMAKE_CURRENT_BINARY_DIR}/ninfer_amd_segmented_bf16_readonly_test.map"
  VERBATIM COMMENT "Prove isolated Core/frozen fixture extraction and reject old Core/Op/support members")
# A map cannot identify an inlined volatile/nonvolatile instruction. The pinned
# include source plus independent actual SDK ISA and math evidence remain mandatory.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_readonly_bf16_inputs})
