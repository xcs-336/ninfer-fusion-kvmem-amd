# New, unwired, SOURCE ONLY. Root may include explicitly after native Core planning.
# Both targets are EXCLUDE_FROM_ALL even after the default-OFF option is enabled.
option(NINFER_AMD_CONTROLLER_HANDOFF "Build one-time isolated Core controller handoff candidate" OFF)
if(NOT NINFER_AMD_CONTROLLER_HANDOFF)
  return()
endif()
if(NOT WIN32 OR NOT DEFINED _core_sources OR NOT DEFINED _generated OR NOT DEFINED NINFER_PYTHON)
  message(FATAL_ERROR "Controller handoff needs the actual Windows prepared-native Core inventory")
endif()
set(NINFER_AMD_CONTROLLER_HANDOFF_BASE_VIEW "" CACHE PATH "Exact existing frozen isolated Core source view")
if(NOT IS_ABSOLUTE "${NINFER_AMD_CONTROLLER_HANDOFF_BASE_VIEW}")
  message(FATAL_ERROR "Explicit existing frozen Core source-view path required")
endif()
set(_handoff_dir "${CMAKE_CURRENT_LIST_DIR}")
set(_handoff_base_fixture "${_handoff_dir}/../tests/kv_segmented_pool_test.cu")
file(GLOB_RECURSE _handoff_native_text CONFIGURE_DEPENDS
  "${_generated}/src/core/*.h" "${_generated}/src/core/*.hpp"
  "${_generated}/src/core/*.cuh" "${_generated}/src/core/*.cpp" "${_generated}/src/core/*.cu")
set(_handoff_inputs ${_handoff_native_text} ${_core_sources}
  "${CMAKE_CURRENT_LIST_FILE}" "${_handoff_dir}/prepare_controller_handoff.py"
  "${_handoff_dir}/controller_handoff_types.h" "${_handoff_dir}/controller_handoff_methods.cpp.inc"
  "${_handoff_dir}/test_controller_handoff.cu" "${_handoff_dir}/verify_controller_handoff_map.py"
  "${_handoff_base_fixture}" "${NINFER_AMD_CONTROLLER_HANDOFF_BASE_VIEW}/segmented-source-receipt.json")
foreach(_f src/core/paged_kv_cache.h src/core/paged_kv_cache.cpp
    amd-memory/kv_segmented_storage.h amd-memory/kv_segmented_storage.cpp
    amd-memory/kv_segmented_pool_control.h amd-memory/kv_segmented_pool_owner.h
    amd-memory/kv_segmented_pool_owner.cpp amd-memory/kv_segmented_loan_plan.h amd-memory/kv_segmented_loan_plan.cpp)
  list(APPEND _handoff_inputs "${NINFER_AMD_CONTROLLER_HANDOFF_BASE_VIEW}/${_f}")
endforeach()
set(_handoff_recipe "full-prepared-native-core-once-controller-handoff-v1\n")
foreach(_input IN LISTS _handoff_inputs)
  file(SHA256 "${_input}" _digest)
  string(APPEND _handoff_recipe "${_input}:${_digest}\n")
endforeach()
string(SHA256 _handoff_sha "${_handoff_recipe}")
set(NINFER_AMD_CONTROLLER_HANDOFF_VIEW "${CMAKE_CURRENT_BINARY_DIR}/controller-handoff-candidate/${_handoff_sha}")
set(_handoff_source_manifest "${CMAKE_CURRENT_BINARY_DIR}/controller-handoff-native-${_handoff_sha}.txt")
set(_handoff_source_text "")
foreach(_source IN LISTS _core_sources)
  file(TO_CMAKE_PATH "${_source}" _normalized)
  string(APPEND _handoff_source_text "${_normalized}\n")
endforeach()
file(WRITE "${_handoff_source_manifest}" "${_handoff_source_text}")
execute_process(COMMAND "${NINFER_PYTHON}" -B "${_handoff_dir}/prepare_controller_handoff.py"
  --base-view "${NINFER_AMD_CONTROLLER_HANDOFF_BASE_VIEW}" --native-view "${_generated}"
  --native-sources "${_handoff_source_manifest}" --base-fixture "${_handoff_base_fixture}"
  --output "${NINFER_AMD_CONTROLLER_HANDOFF_VIEW}" --verify-existing
  RESULT_VARIABLE _handoff_prepared ERROR_VARIABLE _handoff_error)
if(NOT _handoff_prepared EQUAL 0)
  message(FATAL_ERROR "Controller handoff isolation refused: ${_handoff_error}")
endif()
file(READ "${NINFER_AMD_CONTROLLER_HANDOFF_VIEW}/controller-handoff-source-receipt.json" _handoff_receipt)
string(JSON _handoff_count LENGTH "${_handoff_receipt}" compile_sources)
math(EXPR _handoff_last "${_handoff_count}-1")
set(_handoff_sources)
foreach(_index RANGE 0 ${_handoff_last})
  string(JSON _source GET "${_handoff_receipt}" compile_sources ${_index})
  list(APPEND _handoff_sources "${NINFER_AMD_CONTROLLER_HANDOFF_VIEW}/${_source}")
endforeach()
function(ninfer_controller_handoff_native target)
  target_compile_definitions(${target} PUBLIC __HIP_PLATFORM_AMD__=1 NINFER_HIP=1 NOMINMAX WIN32_LEAN_AND_MEAN)
  target_compile_features(${target} PUBLIC cxx_std_20)
  target_compile_options(${target} PRIVATE -O2 -fno-fast-math -ffp-contract=off -fms-runtime-lib=dll)
  target_include_directories(${target} BEFORE PUBLIC
    "${NINFER_AMD_CONTROLLER_HANDOFF_VIEW}/src" "${NINFER_AMD_CONTROLLER_HANDOFF_VIEW}/amd-memory"
    "${ROCM_ROOT}/include" "${_generated}/include" "${_engine}/include")
  target_link_libraries(${target} PUBLIC "${ROCM_ROOT}/lib/amdhip64.lib" Threads::Threads advapi32)
  target_link_options(${target} PUBLIC --hip-link --offload-arch=gfx1201 "--rocm-path=${ROCM_ROOT}")
  get_target_property(_sources ${target} SOURCES)
  foreach(_source IN LISTS _sources)
    if(_source MATCHES "\\.cu$")
      set_source_files_properties("${_source}" PROPERTIES LANGUAGE CXX COMPILE_OPTIONS
        "-x;hip;--offload-arch=gfx1201;-mno-wavefrontsize64;-include;${NINFER_AMD_CONTROLLER_HANDOFF_VIEW}/src/core/hip_intrinsics.h;--rocm-path=${ROCM_ROOT};--rocm-device-lib-path=${ROCM_ROOT}/lib/llvm/amdgcn/bitcode")
    endif()
  endforeach()
endfunction()
find_package(Threads REQUIRED)
add_library(ninfer_controller_handoff_core_amd STATIC EXCLUDE_FROM_ALL ${_handoff_sources})
ninfer_controller_handoff_native(ninfer_controller_handoff_core_amd)
add_executable(ninfer_amd_controller_handoff_test EXCLUDE_FROM_ALL "${_handoff_dir}/test_controller_handoff.cu")
ninfer_controller_handoff_native(ninfer_amd_controller_handoff_test)
target_compile_definitions(ninfer_amd_controller_handoff_test PRIVATE
  "NINFER_CONTROLLER_ORIGINAL_CORE_FIXTURE=\"${NINFER_AMD_CONTROLLER_HANDOFF_VIEW}/test-support/original_pool_fixture.cu\"")
target_link_libraries(ninfer_amd_controller_handoff_test PRIVATE ninfer_controller_handoff_core_amd)
set(_handoff_map "${CMAKE_CURRENT_BINARY_DIR}/ninfer_amd_controller_handoff_test.map")
target_link_options(ninfer_amd_controller_handoff_test PRIVATE "-Wl,/MAP:${_handoff_map}")
add_custom_command(TARGET ninfer_amd_controller_handoff_test POST_BUILD
  COMMAND "${NINFER_PYTHON}" -B "${_handoff_dir}/verify_controller_handoff_map.py" --map "${_handoff_map}"
  VERBATIM COMMENT "Require full new-Core handoff extraction and reject old Core/candidate archives")
# No link to ninfer_core_amd / old segmented candidate; no automatic CTest/Engine hook.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_handoff_inputs})
