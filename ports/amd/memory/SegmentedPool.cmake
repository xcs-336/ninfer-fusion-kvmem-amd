# New SOURCE-ONLY leaf; root must explicitly include this file to expose the option.
# This never changes the production Core source list or enables the production gate.
option(NINFER_AMD_SEGMENTED_POOL "Build isolated original-Core segmented allocator candidate (unqualified)" OFF)
if(NOT NINFER_AMD_SEGMENTED_POOL)
  return()
endif()
if(NOT TARGET ninfer_core_amd OR NOT COMMAND ninfer_native_hip OR NOT DEFINED _generated OR NOT DEFINED NINFER_PYTHON)
  message(FATAL_ERROR "Segmented pool candidate requires the completed native Core source view")
endif()
set(_segmented_pool_inputs
  "${CMAKE_CURRENT_LIST_FILE}"
  "${CMAKE_CURRENT_LIST_DIR}/prepare_segmented_pool.py"
  "${CMAKE_CURRENT_LIST_DIR}/kv_segmented_pool_owner.h"
  "${CMAKE_CURRENT_LIST_DIR}/kv_segmented_pool_owner.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/kv_segmented_pool_control.h"
  "${CMAKE_CURRENT_LIST_DIR}/kv_segmented_storage.h"
  "${CMAKE_CURRENT_LIST_DIR}/kv_segmented_storage.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/kv_segmented_loan_plan.h"
  "${CMAKE_CURRENT_LIST_DIR}/kv_segmented_loan_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/tests/kv_segmented_pool_test.cu"
  "${CMAKE_CURRENT_LIST_DIR}/tests/verify_segmented_pool_map.py"
  "${_generated}/src/core/paged_kv_cache.h"
  "${_generated}/src/core/paged_kv_cache.cpp")
set(_segmented_pool_recipe "isolated-core-segmented-pool-abi1\n")
foreach(_segmented_pool_input IN LISTS _segmented_pool_inputs)
  file(SHA256 "${_segmented_pool_input}" _segmented_pool_digest)
  string(APPEND _segmented_pool_recipe "${_segmented_pool_digest}\n")
endforeach()
string(SHA256 _segmented_pool_recipe_sha "${_segmented_pool_recipe}")
set(NINFER_AMD_SEGMENTED_POOL_VIEW
    "${CMAKE_CURRENT_BINARY_DIR}/segmented-pool-candidate/${_segmented_pool_recipe_sha}")
execute_process(COMMAND "${NINFER_PYTHON}" -B "${CMAKE_CURRENT_LIST_DIR}/prepare_segmented_pool.py"
  --baseline "${_generated}" --output "${NINFER_AMD_SEGMENTED_POOL_VIEW}" --verify-existing
  RESULT_VARIABLE _segmented_pool_prepared
  OUTPUT_VARIABLE _segmented_pool_prepare_stdout ERROR_VARIABLE _segmented_pool_prepare_stderr)
if(NOT _segmented_pool_prepared EQUAL 0)
  message(FATAL_ERROR "Segmented pool isolation refused: ${_segmented_pool_prepare_stderr}")
endif()
add_library(ninfer_segmented_pool_candidate_amd STATIC
  "${NINFER_AMD_SEGMENTED_POOL_VIEW}/src/core/paged_kv_cache.cpp"
  "${NINFER_AMD_SEGMENTED_POOL_VIEW}/amd-memory/kv_segmented_loan_plan.cpp"
  "${NINFER_AMD_SEGMENTED_POOL_VIEW}/amd-memory/kv_segmented_storage.cpp"
  "${NINFER_AMD_SEGMENTED_POOL_VIEW}/amd-memory/kv_segmented_pool_owner.cpp")
ninfer_native_hip(ninfer_segmented_pool_candidate_amd)
target_compile_features(ninfer_segmented_pool_candidate_amd PUBLIC cxx_std_20)
target_compile_options(ninfer_segmented_pool_candidate_amd PRIVATE
  -O2 -fno-fast-math -ffp-contract=off -fms-runtime-lib=dll)
target_include_directories(ninfer_segmented_pool_candidate_amd BEFORE PUBLIC
  "${NINFER_AMD_SEGMENTED_POOL_VIEW}/src"
  "${NINFER_AMD_SEGMENTED_POOL_VIEW}/amd-memory"
  "${_generated}/src" "${_generated}/include")
# Normal archive resolution only. The future executable's actual host/device map must
# prove old paged_kv_cache.cpp.obj was NOT extracted. No whole-archive/force/test-support.
target_link_libraries(ninfer_segmented_pool_candidate_amd PUBLIC ninfer_core_amd)

add_executable(ninfer_amd_segmented_pool_test
  "${CMAKE_CURRENT_LIST_DIR}/tests/kv_segmented_pool_test.cu")
ninfer_native_hip(ninfer_amd_segmented_pool_test)
target_compile_features(ninfer_amd_segmented_pool_test PRIVATE cxx_std_20)
target_compile_options(ninfer_amd_segmented_pool_test PRIVATE
  -O2 -fno-fast-math -ffp-contract=off -fms-runtime-lib=dll)
target_include_directories(ninfer_amd_segmented_pool_test BEFORE PRIVATE
  "${NINFER_AMD_SEGMENTED_POOL_VIEW}/src"
  "${NINFER_AMD_SEGMENTED_POOL_VIEW}/amd-memory")
target_link_libraries(ninfer_amd_segmented_pool_test PRIVATE ninfer_segmented_pool_candidate_amd)
target_link_options(ninfer_amd_segmented_pool_test PRIVATE
  "-Wl,/MAP:${CMAKE_CURRENT_BINARY_DIR}/ninfer_amd_segmented_pool_test.map")
add_custom_command(TARGET ninfer_amd_segmented_pool_test POST_BUILD
  COMMAND "${NINFER_PYTHON}" -B "${CMAKE_CURRENT_LIST_DIR}/tests/verify_segmented_pool_map.py"
    --map "${CMAKE_CURRENT_BINARY_DIR}/ninfer_amd_segmented_pool_test.map"
  VERBATIM COMMENT "Prove candidate Core extraction and exclude old paged Core archive member")
if(NOT COMMAND ninfer_add_gpu_test)
  message(FATAL_ERROR "Segmented original-Core fixture needs the normal native GPU lock/test helper")
endif()
ninfer_add_gpu_test(native_kv_segmented_pool ninfer_amd_segmented_pool_test 120 --run)
set_tests_properties(native_kv_segmented_pool PROPERTIES
  LABELS "amd;experimental;original-core;segmented;gpu"
  ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${ROCM_ROOT}/bin")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_segmented_pool_inputs})
