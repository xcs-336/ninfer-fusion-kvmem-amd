# NEW leaf only. Root Core.cmake and every frozen candidate leaf remain untouched.
option(NINFER_AMD_MODELS_ENGINE_WORKER "Build isolated single-payload public Engine worker candidate" OFF)
if(NOT NINFER_AMD_MODELS_ENGINE_WORKER)
  return()
endif()
if(NOT WIN32 OR NOT DEFINED _engine_required_ops)
  message(FATAL_ERROR "Worker candidate needs the actual Windows public Engine dependency inventory")
endif()
foreach(_required NINFER_AMD_CONTROLLER_HANDOFF_VIEW NINFER_AMD_MODELS_STORAGE_VIEW
    NINFER_AMD_MODELS_ENGINE_OPS_VIEW _generated _engine NINFER_PYTHON ROCM_ROOT)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "Worker candidate requires explicit ${_required}")
  endif()
endforeach()
foreach(_required ninfer_models_engine_b1_ops_amd ninfer_models_engine_cache_ops_amd
    ninfer_artifact_amd ninfer_runtime_support_amd ninfer_text_amd ninfer_media_decode_amd
    ninfer_jinja ninfer_xgrammar)
  if(NOT TARGET ${_required})
    message(FATAL_ERROR "Build-ready original/isolated dependency missing: ${_required}")
  endif()
endforeach()
set(_mew_dir "${CMAKE_CURRENT_LIST_DIR}")
file(GLOB _mew_owned CONFIGURE_DEPENDS "${_mew_dir}/*.py" "${_mew_dir}/*.h"
  "${_mew_dir}/*.inc" "${_mew_dir}/*.cpp" "${_mew_dir}/*.cmake" "${_mew_dir}/*.md")
file(GLOB _mew_json CONFIGURE_DEPENDS "${_mew_dir}/*.json")
list(APPEND _mew_owned ${_mew_json})
set(_mew_inputs ${_mew_owned}
  "${NINFER_AMD_CONTROLLER_HANDOFF_VIEW}/controller-handoff-source-receipt.json"
  "${NINFER_AMD_MODELS_STORAGE_VIEW}/models-storage-source-receipt.json"
  "${NINFER_AMD_MODELS_ENGINE_OPS_VIEW}/models-engine-ops-source-receipt.json"
  "${_generated}/engine-manifest.json"
  "${_mew_dir}/../models_engine/prepare_engine_storage_plan.py"
  "${_mew_dir}/../models_engine/segmented_planner_checks.cpp.inc"
  "${_mew_dir}/../models_engine_ops/bridge_models_calls.py")
file(READ "${_mew_dir}/dependency-pins.json" _mew_dependency_pins)
string(JSON _mew_pin_count LENGTH "${_mew_dependency_pins}" source_sha256)
math(EXPR _mew_pin_last "${_mew_pin_count}-1")
foreach(_index RANGE 0 ${_mew_pin_last})
  string(JSON _source MEMBER "${_mew_dependency_pins}" source_sha256 ${_index})
  list(APPEND _mew_inputs "${_mew_dir}/../../../../${_source}")
endforeach()
# Include every actual small copied text in the configure identity, not only TUs
# whose top-level functions changed. The generator independently bounds/verifies.
file(GLOB_RECURSE _mew_snapshot CONFIGURE_DEPENDS
  "${_generated}/src/models/*.h" "${_generated}/src/models/*.hpp"
  "${_generated}/src/models/*.cuh" "${_generated}/src/models/*.inc"
  "${_generated}/src/models/*.cpp" "${_generated}/src/models/*.cu"
  "${_generated}/src/runtime/*.h" "${_generated}/src/runtime/*.hpp"
  "${_generated}/src/runtime/*.cuh" "${_generated}/src/runtime/*.inc"
  "${_generated}/src/runtime/*.cpp" "${_generated}/src/runtime/*.cu"
  "${_generated}/include/ninfer/*.h" "${_generated}/include/ninfer/*.hpp")
list(APPEND _mew_inputs ${_mew_snapshot})
foreach(_view NINFER_AMD_CONTROLLER_HANDOFF_VIEW NINFER_AMD_MODELS_STORAGE_VIEW NINFER_AMD_MODELS_ENGINE_OPS_VIEW)
  file(GLOB_RECURSE _mew_view_files CONFIGURE_DEPENDS "${${_view}}/*")
  list(APPEND _mew_inputs ${_mew_view_files})
endforeach()
list(REMOVE_DUPLICATES _mew_inputs)
set(_mew_recipe "public-engine-single-native-pool-once-worker-v1\n")
foreach(_source IN LISTS _mew_inputs)
  file(SHA256 "${_source}" _sha)
  string(APPEND _mew_recipe "${_source}:${_sha}\n")
endforeach()
string(SHA256 _mew_sha "${_mew_recipe}")
set(NINFER_AMD_MODELS_ENGINE_WORKER_VIEW "${CMAKE_CURRENT_BINARY_DIR}/models-engine-worker/${_mew_sha}")
execute_process(COMMAND "${NINFER_PYTHON}" -B "${_mew_dir}/prepare_models_engine_worker.py"
  --native-root "${_generated}" --models-root "${NINFER_AMD_MODELS_STORAGE_VIEW}"
  --controller-root "${NINFER_AMD_CONTROLLER_HANDOFF_VIEW}" --ops-root "${NINFER_AMD_MODELS_ENGINE_OPS_VIEW}"
  --output "${NINFER_AMD_MODELS_ENGINE_WORKER_VIEW}" --verify-existing
  RESULT_VARIABLE _mew_result ERROR_VARIABLE _mew_error)
if(NOT _mew_result EQUAL 0)
  message(FATAL_ERROR "Full Engine worker source closure refused: ${_mew_error}")
endif()
file(READ "${NINFER_AMD_MODELS_ENGINE_WORKER_VIEW}/models-engine-worker-source-receipt.json" _mew_receipt)
function(ninfer_models_engine_worker_native target)
  target_compile_features(${target} PUBLIC cxx_std_20)
  target_compile_definitions(${target} PUBLIC __HIP_PLATFORM_AMD__=1 NINFER_HIP=1 NOMINMAX WIN32_LEAN_AND_MEAN)
  target_compile_options(${target} PRIVATE -O3 -fno-fast-math
    -fhip-fp32-correctly-rounded-divide-sqrt -fms-runtime-lib=dll)
  target_include_directories(${target} BEFORE PUBLIC
    "${NINFER_AMD_MODELS_ENGINE_WORKER_VIEW}/include"
    "${NINFER_AMD_MODELS_ENGINE_WORKER_VIEW}/src"
    "${NINFER_AMD_MODELS_ENGINE_WORKER_VIEW}/amd-memory"
    "${NINFER_AMD_MODELS_ENGINE_WORKER_VIEW}"
    "${NINFER_AMD_MODELS_ENGINE_WORKER_VIEW}/src/models/qwen3_8_flash_next/export"
    "${NINFER_AMD_MODELS_ENGINE_WORKER_VIEW}/src/models/qwen3_vision/export"
    "${ROCM_ROOT}/include" "${_generated}/src" "${_generated}/include"
    "${_engine}/include" "${_engine}/third_party")
  target_link_libraries(${target} PUBLIC "${ROCM_ROOT}/lib/amdhip64.lib" Threads::Threads advapi32)
  target_link_options(${target} PUBLIC --hip-link --offload-arch=gfx1201 "--rocm-path=${ROCM_ROOT}")
  get_target_property(_sources ${target} SOURCES)
  foreach(_source IN LISTS _sources)
    if(_source MATCHES "\\.cu$")
      set_source_files_properties("${_source}" PROPERTIES LANGUAGE CXX COMPILE_OPTIONS
        "-x;hip;--offload-arch=gfx1201;-mno-wavefrontsize64;-include;${NINFER_AMD_MODELS_ENGINE_WORKER_VIEW}/src/core/hip_intrinsics.h;--rocm-path=${ROCM_ROOT};--rocm-device-lib-path=${ROCM_ROOT}/lib/llvm/amdgcn/bitcode")
    endif()
  endforeach()
endfunction()
find_package(Threads REQUIRED)
function(_mew_sources output)
  string(JSON _count LENGTH "${_mew_receipt}" ${ARGN})
  math(EXPR _last "${_count}-1")
  set(_paths)
  foreach(_index RANGE 0 ${_last})
    string(JSON _source GET "${_mew_receipt}" ${ARGN} ${_index})
    list(APPEND _paths "${NINFER_AMD_MODELS_ENGINE_WORKER_VIEW}/${_source}")
  endforeach()
  set(${output} "${_paths}" PARENT_SCOPE)
endfunction()
_mew_sources(_mew_core_sources core_compile_sources)
add_library(ninfer_models_engine_worker_core_amd STATIC EXCLUDE_FROM_ALL ${_mew_core_sources})
ninfer_models_engine_worker_native(ninfer_models_engine_worker_core_amd)
foreach(_component model_loading model_runtime engine calibration)
  _mew_sources(_mew_component_sources components ${_component})
  add_library(ninfer_models_engine_worker_${_component}_amd STATIC EXCLUDE_FROM_ALL ${_mew_component_sources})
  ninfer_models_engine_worker_native(ninfer_models_engine_worker_${_component}_amd)
endforeach()

# Raw archive files do not propagate usage requirements. Restore only the actual
# Engine third-party headers and original text export definition, without any
# old target LINK_LIBRARIES/Core interface.
get_target_property(_mew_xgrammar_includes ninfer_xgrammar INTERFACE_INCLUDE_DIRECTORIES)
if(NOT _mew_xgrammar_includes OR _mew_xgrammar_includes MATCHES "-NOTFOUND$")
  message(FATAL_ERROR "Actual xgrammar public include inventory is missing")
endif()
target_include_directories(ninfer_models_engine_worker_engine_amd SYSTEM PRIVATE
  ${_mew_xgrammar_includes})
foreach(_component model_loading model_runtime engine)
  target_compile_definitions(ninfer_models_engine_worker_${_component}_amd PRIVATE UTF8PROC_STATIC)
endforeach()

# Follow actual existing Tensor-only dependency edges, but pass static archive
# FILES: never inherit an old Core/model/cache ABI through a PUBLIC target link.
# The new cache/B1 families completely replace the original Attention manifests.
set_property(GLOBAL PROPERTY NINFER_MEW_VISITED "")
set_property(GLOBAL PROPERTY NINFER_MEW_ARCHIVES "")
function(_mew_tensor_dependency dependency)
  if(dependency MATCHES "^\\$<LINK_ONLY:([^>]+)>$")
    set(dependency "${CMAKE_MATCH_1}")
  endif()
  if(dependency MATCHES "^ninfer_(core_amd|controller_handoff_core_amd|attention.*ops_amd|model_.*|engine_amd)$")
    return()
  endif()
  if(TARGET ${dependency})
    get_property(_visited GLOBAL PROPERTY NINFER_MEW_VISITED)
    if(dependency IN_LIST _visited)
      return()
    endif()
    set_property(GLOBAL APPEND PROPERTY NINFER_MEW_VISITED "${dependency}")
    get_target_property(_type ${dependency} TYPE)
    if(_type STREQUAL "STATIC_LIBRARY" OR _type STREQUAL "UNKNOWN_LIBRARY")
      set_property(GLOBAL APPEND PROPERTY NINFER_MEW_ARCHIVES "$<TARGET_FILE:${dependency}>")
    elseif(NOT _type STREQUAL "INTERFACE_LIBRARY")
      message(FATAL_ERROR "Unreviewed dependency type ${dependency}/${_type}")
    endif()
    foreach(_property LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
      get_target_property(_links ${dependency} ${_property})
      if(_links AND NOT _links MATCHES "-NOTFOUND$")
        # CMake wraps cross-directory link entries in ::@(directory-id);...;::@.
        # These are lookup-scope delimiters, not libraries. Keep every enclosed
        # dependency on the same strict recursion path; unresolved targets still
        # fail below instead of becoming an unreviewed archive fallback.
        set(_mew_directory_depth 0)
        foreach(_link IN LISTS _links)
          if(_link MATCHES "^::@\\([^()]+\\)$")
            math(EXPR _mew_directory_depth "${_mew_directory_depth}+1")
          elseif(_link STREQUAL "::@")
            if(_mew_directory_depth LESS 1)
              message(FATAL_ERROR "Unbalanced CMake link directory-scope close")
            endif()
            math(EXPR _mew_directory_depth "${_mew_directory_depth}-1")
          else()
            _mew_tensor_dependency("${_link}")
          endif()
        endforeach()
        if(NOT _mew_directory_depth EQUAL 0)
          message(FATAL_ERROR "Unclosed CMake link directory-scope marker")
        endif()
      endif()
    endforeach()
  elseif(dependency MATCHES "\\.lib$" OR dependency STREQUAL "advapi32")
    set_property(GLOBAL APPEND PROPERTY NINFER_MEW_ARCHIVES "${dependency}")
  elseif(dependency MATCHES "^\\$<")
    message(FATAL_ERROR "Unreviewed generator-expression dependency: ${dependency}")
  elseif(NOT dependency STREQUAL "")
    # System imported-interface flags must be reviewed rather than silently lost.
    message(FATAL_ERROR "Unreviewed non-target Tensor dependency: ${dependency}")
  endif()
endfunction()
foreach(_dependency IN LISTS _engine_required_ops)
  _mew_tensor_dependency("${_dependency}")
endforeach()
foreach(_dependency ninfer_artifact_amd ninfer_runtime_support_amd ninfer_text_amd
    ninfer_media_decode_amd ninfer_jinja ninfer_xgrammar)
  _mew_tensor_dependency("${_dependency}")
endforeach()
get_property(_mew_tensor_archives GLOBAL PROPERTY NINFER_MEW_ARCHIVES)
list(REMOVE_DUPLICATES _mew_tensor_archives)
add_library(ninfer_models_engine_worker_amd INTERFACE)
target_link_libraries(ninfer_models_engine_worker_amd INTERFACE
  ninfer_models_engine_worker_engine_amd ninfer_models_engine_worker_model_runtime_amd
  ninfer_models_engine_worker_model_loading_amd ninfer_models_engine_worker_calibration_amd
  "$<TARGET_FILE:ninfer_models_engine_b1_ops_amd>" "$<TARGET_FILE:ninfer_models_engine_cache_ops_amd>"
  ${_mew_tensor_archives} ninfer_models_engine_worker_core_amd)
add_executable(ninfer_amd_models_engine_worker_test EXCLUDE_FROM_ALL "${_mew_dir}/test_models_engine_worker.cpp")
ninfer_models_engine_worker_native(ninfer_amd_models_engine_worker_test)
target_link_libraries(ninfer_amd_models_engine_worker_test PRIVATE ninfer_models_engine_worker_amd)
add_dependencies(ninfer_amd_models_engine_worker_test ninfer_models_engine_b1_ops_amd ninfer_models_engine_cache_ops_amd)
get_property(_mew_depends GLOBAL PROPERTY NINFER_MEW_VISITED)
add_dependencies(ninfer_amd_models_engine_worker_test ${_mew_depends})
set(_mew_map "${CMAKE_CURRENT_BINARY_DIR}/ninfer_amd_models_engine_worker_test.map")
target_link_options(ninfer_amd_models_engine_worker_test PRIVATE "-Wl,/MAP:${_mew_map}")
add_custom_command(TARGET ninfer_amd_models_engine_worker_test POST_BUILD
  COMMAND "${NINFER_PYTHON}" -B "${_mew_dir}/verify_models_engine_worker_map.py" --map "${_mew_map}"
  VERBATIM COMMENT "Require full unique worker Core/Models/public Engine; reject every old paged ABI provider")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_mew_inputs})
# No automatic GPU CTest, calibration grant, public ordinary default change or Engine alias.
