# New standalone leaf, not included from Core or the frozen six-case leaf.
option(NINFER_AMD_SEGMENTED_B1_PUBLIC "Build isolated original-policy BF16 B1 wrapper candidate (unqualified)" OFF)
if(NOT NINFER_AMD_SEGMENTED_B1_PUBLIC)
  return()
endif()
if(NOT TARGET ninfer_segmented_pool_candidate_amd OR NOT TARGET ninfer_basic_ops_amd OR
   NOT COMMAND ninfer_native_hip OR NOT DEFINED NINFER_AMD_SEGMENTED_POOL_VIEW)
  message(FATAL_ERROR "Explicit isolated segmented Core and native Basic are required; never use old-view Ops")
endif()
set(_b1_inputs
  "${CMAKE_CURRENT_LIST_FILE}"
  "${CMAKE_CURRENT_LIST_DIR}/prepare_public_bf16.py"
  "${CMAKE_CURRENT_LIST_DIR}/public_bf16.h"
  "${CMAKE_CURRENT_LIST_DIR}/fixture_common.cuh"
  "${CMAKE_CURRENT_LIST_DIR}/test_public_bf16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/verify_map.py"
  "${CMAKE_CURRENT_LIST_DIR}/../segmented_bf16_ops_overlay.py"
  "${CMAKE_CURRENT_LIST_DIR}/../segmented_bf16_view.cuh"
  "${CMAKE_CURRENT_LIST_DIR}/../tests/verify_segmented_pool_map.py"
  "${CMAKE_CURRENT_LIST_DIR}/../tests/verify_segmented_bf16_map.py"
  "${NINFER_AMD_SEGMENTED_POOL_VIEW}/segmented-source-receipt.json"
  "${_engine}/tests/ops/op_check.h")
set(_b1_native
  src/ops/kv_cache/append/kernel.cuh src/ops/kv_cache/append/geometry.cuh
  src/ops/kernel/paged_kv_address.cuh
  src/ops/softmax_attention/dense/causal_cache/small_t_bf16.cuh
  src/ops/softmax_attention/dense/causal_cache/small_t.cuh
  src/ops/softmax_attention/dense/causal_cache/prompt_bf16.cuh
  src/ops/softmax_attention/dense/causal_cache/prompt_common.cuh
  src/ops/softmax_attention/dense/causal_cache/geometry.cuh
  src/ops/softmax_attention/dense/causal_cache/launch.h
  src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp
  src/ops/softmax_attention/dense/causal_cache/small_t.cu
  src/ops/softmax_attention/dense/causal_cache/prompt.cu src/ops/kv_cache/append/launch.cu)
foreach(_source IN LISTS _b1_native)
  list(APPEND _b1_inputs "${_generated}/${_source}")
endforeach()
# Ninja canonicalizes '..' in regeneration dependencies. Normalize here so the
# shared frozen inputs are not emitted twice under distinct lexical spellings.
set(_b1_normalized_inputs)
foreach(_input IN LISTS _b1_inputs)
  get_filename_component(_b1_normalized_input "${_input}" ABSOLUTE)
  list(APPEND _b1_normalized_inputs "${_b1_normalized_input}")
endforeach()
list(REMOVE_DUPLICATES _b1_normalized_inputs)
set(_b1_inputs ${_b1_normalized_inputs})
set(_b1_recipe "actual-public-b1-bf16-segmented-v1\n")
foreach(_input IN LISTS _b1_inputs)
  file(SHA256 "${_input}" _hash)
  string(APPEND _b1_recipe "${_hash}\n")
endforeach()
string(SHA256 _b1_identity "${_b1_recipe}")
set(NINFER_AMD_SEGMENTED_B1_VIEW "${CMAKE_CURRENT_BINARY_DIR}/segmented-b1-public/${_b1_identity}")
execute_process(COMMAND "${NINFER_PYTHON}" -B "${CMAKE_CURRENT_LIST_DIR}/prepare_public_bf16.py"
  --baseline "${_generated}" --output "${NINFER_AMD_SEGMENTED_B1_VIEW}" --verify-existing
  RESULT_VARIABLE _b1_result ERROR_VARIABLE _b1_error)
if(NOT _b1_result EQUAL 0)
  message(FATAL_ERROR "Original public B1 BF16 isolation refused: ${_b1_error}")
endif()
add_library(ninfer_segmented_b1_public_ops_amd STATIC
  "${NINFER_AMD_SEGMENTED_B1_VIEW}/amd-b1/public.cpp"
  "${NINFER_AMD_SEGMENTED_B1_VIEW}/amd-b1/small_t.cu"
  "${NINFER_AMD_SEGMENTED_B1_VIEW}/amd-b1/prompt.cu"
  "${NINFER_AMD_SEGMENTED_B1_VIEW}/amd-b1/append.cu")
ninfer_native_hip(ninfer_segmented_b1_public_ops_amd)
target_compile_features(ninfer_segmented_b1_public_ops_amd PUBLIC cxx_std_20)
target_compile_options(ninfer_segmented_b1_public_ops_amd PRIVATE
  -O3 -fno-fast-math -fhip-fp32-correctly-rounded-divide-sqrt -fms-runtime-lib=dll)
target_include_directories(ninfer_segmented_b1_public_ops_amd BEFORE PUBLIC
  "${NINFER_AMD_SEGMENTED_B1_VIEW}"
  "${NINFER_AMD_SEGMENTED_POOL_VIEW}"
  "${NINFER_AMD_SEGMENTED_POOL_VIEW}/src" "${NINFER_AMD_SEGMENTED_POOL_VIEW}/amd-memory")
# Basic extraction supplies only Tensor-only primitives such as sigmoid_mul.
# New isolated paged Core must resolve first; the actual final map rejects any
# old paged Core / original Attention or append / test-support member.
target_link_libraries(ninfer_segmented_b1_public_ops_amd PUBLIC
  ninfer_segmented_pool_candidate_amd ninfer_basic_ops_amd)
add_executable(ninfer_amd_segmented_b1_public_test "${CMAKE_CURRENT_LIST_DIR}/test_public_bf16.cu")
ninfer_native_hip(ninfer_amd_segmented_b1_public_test)
target_compile_features(ninfer_amd_segmented_b1_public_test PRIVATE cxx_std_20)
target_compile_options(ninfer_amd_segmented_b1_public_test PRIVATE
  -O3 -fno-fast-math -fhip-fp32-correctly-rounded-divide-sqrt -fms-runtime-lib=dll)
target_include_directories(ninfer_amd_segmented_b1_public_test BEFORE PRIVATE
  "${NINFER_AMD_SEGMENTED_B1_VIEW}" "${NINFER_AMD_SEGMENTED_POOL_VIEW}"
  "${NINFER_AMD_SEGMENTED_POOL_VIEW}/src"
  "${NINFER_AMD_SEGMENTED_POOL_VIEW}/amd-memory" "${_engine}/tests/ops")
target_link_libraries(ninfer_amd_segmented_b1_public_test PRIVATE ninfer_segmented_b1_public_ops_amd)
target_link_options(ninfer_amd_segmented_b1_public_test PRIVATE
  "-Wl,/MAP:${CMAKE_CURRENT_BINARY_DIR}/ninfer_amd_segmented_b1_public_test.map")
add_custom_command(TARGET ninfer_amd_segmented_b1_public_test POST_BUILD
  COMMAND "${NINFER_PYTHON}" -B "${CMAKE_CURRENT_LIST_DIR}/verify_map.py"
    --map "${CMAKE_CURRENT_BINARY_DIR}/ninfer_amd_segmented_b1_public_test.map"
  VERBATIM COMMENT "Require actual new public policy/launchers and isolated Core; reject old view members")
# No automatic test registration/admission. Root must explicitly launch --run
# under the shared byte0/host-budget/PID monitor in an owned resource window.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_b1_inputs})
