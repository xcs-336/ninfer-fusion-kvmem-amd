# NEW independent leaf. Root Core and existing candidate leaves are untouched.
option(NINFER_AMD_MODELS_ENGINE_OPS "Build full-controller-Core ABI B1/ordinary cache Ops (unqualified)" OFF)
if(NOT NINFER_AMD_MODELS_ENGINE_OPS)
  return()
endif()
if(NOT WIN32 OR NOT TARGET ninfer_controller_handoff_core_amd OR
   NOT TARGET ninfer_basic_ops_amd OR NOT DEFINED NINFER_AMD_CONTROLLER_HANDOFF_VIEW)
  message(FATAL_ERROR "Models Engine Ops needs the exact full controller Core and actual Basic archive")
endif()
set(_meo_dir "${CMAKE_CURRENT_LIST_DIR}")
file(READ "${_meo_dir}/external-pins.json" _meo_pins)
file(GLOB _meo_owned CONFIGURE_DEPENDS "${_meo_dir}/*.py" "${_meo_dir}/*.h"
  "${_meo_dir}/*.cu" "${_meo_dir}/*.cmake" "${_meo_dir}/*.json" "${_meo_dir}/*.md")
set(_meo_inputs ${_meo_owned}
  "${NINFER_AMD_CONTROLLER_HANDOFF_VIEW}/controller-handoff-source-receipt.json")
string(JSON _meo_basic_manifest GET "${_meo_pins}" basic_archive manifest)
list(APPEND _meo_inputs "${_engine}/${_meo_basic_manifest}")
foreach(_group native_family_source_sha256 native_b1_extra_source_sha256 original_models_bridge_input_sha256)
  string(JSON _meo_count LENGTH "${_meo_pins}" ${_group})
  math(EXPR _meo_last "${_meo_count}-1")
  foreach(_i RANGE 0 ${_meo_last})
    string(JSON _source MEMBER "${_meo_pins}" ${_group} ${_i})
    list(APPEND _meo_inputs "${_generated}/${_source}")
  endforeach()
endforeach()
foreach(_group external_recipe_sha256 manifest_sha256)
  string(JSON _meo_count LENGTH "${_meo_pins}" ${_group})
  math(EXPR _meo_last "${_meo_count}-1")
  foreach(_i RANGE 0 ${_meo_last})
    string(JSON _source MEMBER "${_meo_pins}" ${_group} ${_i})
    if(_group STREQUAL "external_recipe_sha256")
      list(APPEND _meo_inputs "${_meo_dir}/../../../../${_source}")
    else()
      list(APPEND _meo_inputs "${_engine}/${_source}")
    endif()
  endforeach()
endforeach()
set(_meo_normalized)
foreach(_source IN LISTS _meo_inputs)
  get_filename_component(_source "${_source}" ABSOLUTE)
  list(APPEND _meo_normalized "${_source}")
endforeach()
list(REMOVE_DUPLICATES _meo_normalized)
set(_meo_inputs ${_meo_normalized})
set(_meo_recipe "full-controller-core-b1-original-cache-ops-v1\n")
foreach(_source IN LISTS _meo_inputs)
  file(SHA256 "${_source}" _sha)
  string(APPEND _meo_recipe "${_source}:${_sha}\n")
endforeach()
string(SHA256 _meo_identity "${_meo_recipe}")
set(NINFER_AMD_MODELS_ENGINE_OPS_VIEW "${CMAKE_CURRENT_BINARY_DIR}/models-engine-ops/${_meo_identity}")
execute_process(COMMAND "${NINFER_PYTHON}" -B "${_meo_dir}/prepare_models_engine_ops.py"
  --baseline "${_generated}" --engine "${_engine}"
  --controller-view "${NINFER_AMD_CONTROLLER_HANDOFF_VIEW}"
  --output "${NINFER_AMD_MODELS_ENGINE_OPS_VIEW}" --verify-existing
  RESULT_VARIABLE _meo_result ERROR_VARIABLE _meo_error)
if(NOT _meo_result EQUAL 0)
  message(FATAL_ERROR "New Models Engine Ops source-view refused: ${_meo_error}")
endif()

function(ninfer_models_engine_ops_native target)
  target_compile_definitions(${target} PUBLIC __HIP_PLATFORM_AMD__=1 NINFER_HIP=1 NOMINMAX WIN32_LEAN_AND_MEAN)
  target_compile_features(${target} PUBLIC cxx_std_20)
  target_compile_options(${target} PRIVATE -O3 -fno-fast-math
    -fhip-fp32-correctly-rounded-divide-sqrt -fms-runtime-lib=dll)
  target_include_directories(${target} BEFORE PUBLIC
    "${NINFER_AMD_MODELS_ENGINE_OPS_VIEW}"
    "${NINFER_AMD_CONTROLLER_HANDOFF_VIEW}/src" "${NINFER_AMD_CONTROLLER_HANDOFF_VIEW}/amd-memory"
    "${NINFER_AMD_MODELS_ENGINE_OPS_VIEW}/src"
    "${ROCM_ROOT}/include" "${_generated}/src" "${_generated}/include" "${_engine}/include")
  target_link_libraries(${target} PUBLIC "${ROCM_ROOT}/lib/amdhip64.lib")
  target_link_options(${target} PUBLIC --hip-link --offload-arch=gfx1201 "--rocm-path=${ROCM_ROOT}")
  get_target_property(_sources ${target} SOURCES)
  foreach(_source IN LISTS _sources)
    if(_source MATCHES "\\.cu$")
      set_source_files_properties("${_source}" PROPERTIES LANGUAGE CXX COMPILE_OPTIONS
        "-x;hip;--offload-arch=gfx1201;-mno-wavefrontsize64;-include;${NINFER_AMD_CONTROLLER_HANDOFF_VIEW}/src/core/hip_intrinsics.h;--rocm-path=${ROCM_ROOT};--rocm-device-lib-path=${ROCM_ROOT}/lib/llvm/amdgcn/bitcode")
    endif()
  endforeach()
endfunction()

set(_meo_cache_sources)
string(JSON _meo_count LENGTH "${_meo_pins}" compile_sources)
math(EXPR _meo_last "${_meo_count}-1")
foreach(_i RANGE 0 ${_meo_last})
  string(JSON _source GET "${_meo_pins}" compile_sources ${_i})
  list(APPEND _meo_cache_sources "${NINFER_AMD_MODELS_ENGINE_OPS_VIEW}/${_source}")
endforeach()
add_library(ninfer_models_engine_cache_ops_amd STATIC EXCLUDE_FROM_ALL ${_meo_cache_sources})
ninfer_models_engine_ops_native(ninfer_models_engine_cache_ops_amd)
add_library(ninfer_models_engine_b1_ops_amd STATIC EXCLUDE_FROM_ALL
  "${NINFER_AMD_MODELS_ENGINE_OPS_VIEW}/amd-models-b1/public.cpp"
  "${NINFER_AMD_MODELS_ENGINE_OPS_VIEW}/amd-models-b1/small_t.cu"
  "${NINFER_AMD_MODELS_ENGINE_OPS_VIEW}/amd-models-b1/prompt.cu"
  "${NINFER_AMD_MODELS_ENGINE_OPS_VIEW}/amd-models-b1/append.cu")
ninfer_models_engine_ops_native(ninfer_models_engine_b1_ops_amd)
foreach(_target ninfer_models_engine_cache_ops_amd ninfer_models_engine_b1_ops_amd)
  # Reuse only the actual Tensor-only Basic archive. Its PUBLIC old-Core link
  # interface must not become this closure's ABI fallback. Final maps still prove
  # which members LLD actually extracts; a link option is not that proof.
  target_link_libraries(${_target} PUBLIC ninfer_controller_handoff_core_amd
    "$<TARGET_FILE:ninfer_basic_ops_amd>")
  add_dependencies(${_target} ninfer_basic_ops_amd)
endforeach()
add_executable(ninfer_amd_models_engine_b1_ops_test EXCLUDE_FROM_ALL "${_meo_dir}/test_models_engine_ops.cu")
ninfer_models_engine_ops_native(ninfer_amd_models_engine_b1_ops_test)
target_include_directories(ninfer_amd_models_engine_b1_ops_test PRIVATE "${_engine}/tests/ops")
target_link_libraries(ninfer_amd_models_engine_b1_ops_test PRIVATE
  ninfer_models_engine_b1_ops_amd ninfer_models_engine_cache_ops_amd)
set(_meo_map "${CMAKE_CURRENT_BINARY_DIR}/ninfer_amd_models_engine_b1_ops_test.map")
target_link_options(ninfer_amd_models_engine_b1_ops_test PRIVATE "-Wl,/MAP:${_meo_map}")
add_custom_command(TARGET ninfer_amd_models_engine_b1_ops_test POST_BUILD
  COMMAND "${NINFER_PYTHON}" -B "${_meo_dir}/verify_map.py" --map "${_meo_map}"
  VERBATIM COMMENT "Require full new-Core/B1/cache members; reject mixed paged ABI")
# Default executable is plan-only; explicit --host-guards or --run is root-owned.
# No CTest registration, calibration grant, production switch or automatic build.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_meo_inputs})
