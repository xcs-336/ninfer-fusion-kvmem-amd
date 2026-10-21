# NEW standalone component, default OFF, root does not include it this turn.
option(NINFER_AMD_SEGMENTED_VISION_BROKER "Build real isolated segmented Vision Broker component" OFF)
if(NOT NINFER_AMD_SEGMENTED_VISION_BROKER)
  return()
endif()
if(NOT WIN32 OR NOT DEFINED NINFER_PYTHON OR NOT DEFINED ROCM_ROOT OR
    NOT DEFINED _generated OR NOT DEFINED _engine)
  message(FATAL_ERROR "Segmented Vision Broker needs the actual Windows native environment")
endif()
set(NINFER_AMD_SEGMENTED_VISION_CONTROLLER_VIEW "" CACHE PATH "Exact full frozen controller Core input")
if(NOT IS_ABSOLUTE "${NINFER_AMD_SEGMENTED_VISION_CONTROLLER_VIEW}")
  message(FATAL_ERROR "Explicit full controller Core view required")
endif()
set(_svb_dir "${CMAKE_CURRENT_LIST_DIR}")
file(GLOB_RECURSE _svb_core_inputs CONFIGURE_DEPENDS "${NINFER_AMD_SEGMENTED_VISION_CONTROLLER_VIEW}/*")
set(_svb_inputs ${_svb_core_inputs} "${CMAKE_CURRENT_LIST_FILE}"
  "${_generated}/src/models/qwen3_5/execution/vision_overlay.h"
  "${_generated}/src/models/qwen3_5/execution/vision_overlay.cpp")
foreach(_name prepare_segmented_vision_broker.py broker_methods.cpp.inc manifest.json README.md
    test_segmented_vision_broker.cu verify_segmented_vision_map.py)
  list(APPEND _svb_inputs "${_svb_dir}/${_name}")
endforeach()
foreach(_name prepare_vision_segmented_capacity.py manifest.json check_source.py README.md)
  list(APPEND _svb_inputs "${_svb_dir}/../vision_segmented_capacity/${_name}")
endforeach()
execute_process(COMMAND "${NINFER_PYTHON}" -B "${_svb_dir}/prepare_segmented_vision_broker.py"
  --controller-view "${NINFER_AMD_SEGMENTED_VISION_CONTROLLER_VIEW}" --native-view "${_generated}"
  --output-root "${CMAKE_CURRENT_BINARY_DIR}/segmented-vision-broker-candidate" --emit --verify-existing
  RESULT_VARIABLE _svb_result OUTPUT_VARIABLE _svb_plan ERROR_VARIABLE _svb_error)
if(NOT _svb_result EQUAL 0)
  message(FATAL_ERROR "Real Vision Broker component isolation refused: ${_svb_error}")
endif()
string(JSON NINFER_AMD_SEGMENTED_VISION_BROKER_VIEW GET "${_svb_plan}" output)
file(TO_CMAKE_PATH "${NINFER_AMD_SEGMENTED_VISION_BROKER_VIEW}" NINFER_AMD_SEGMENTED_VISION_BROKER_VIEW)
file(READ "${NINFER_AMD_SEGMENTED_VISION_BROKER_VIEW}/segmented-vision-broker-source-receipt.json" _svb_receipt)
string(JSON _svb_count LENGTH "${_svb_receipt}" compile_sources)
if(NOT _svb_count EQUAL 25)
  message(FATAL_ERROR "Exactly the new full25 Core source closure is required")
endif()
set(_svb_core_sources)
foreach(_i RANGE 0 24)
  string(JSON _rel GET "${_svb_receipt}" compile_sources ${_i})
  list(APPEND _svb_core_sources "${NINFER_AMD_SEGMENTED_VISION_BROKER_VIEW}/${_rel}")
endforeach()
find_package(Threads REQUIRED)
function(ninfer_segmented_vision_native target)
  target_compile_definitions(${target} PUBLIC __HIP_PLATFORM_AMD__=1 NINFER_HIP=1 NOMINMAX WIN32_LEAN_AND_MEAN)
  target_compile_features(${target} PUBLIC cxx_std_20)
  target_compile_options(${target} PRIVATE -O2 -fno-fast-math -ffp-contract=off -fms-runtime-lib=dll)
  target_include_directories(${target} BEFORE PUBLIC
    "${NINFER_AMD_SEGMENTED_VISION_BROKER_VIEW}" "${NINFER_AMD_SEGMENTED_VISION_BROKER_VIEW}/src"
    "${NINFER_AMD_SEGMENTED_VISION_BROKER_VIEW}/amd-memory"
    "${ROCM_ROOT}/include" "${_generated}/include" "${_engine}/include")
  target_link_libraries(${target} PUBLIC "${ROCM_ROOT}/lib/amdhip64.lib" Threads::Threads advapi32)
  target_link_options(${target} PUBLIC --hip-link --offload-arch=gfx1201 "--rocm-path=${ROCM_ROOT}")
  get_target_property(_sources ${target} SOURCES)
  foreach(_source IN LISTS _sources)
    if(_source MATCHES "\\.cu$")
      set_source_files_properties("${_source}" PROPERTIES LANGUAGE CXX COMPILE_OPTIONS
        "-x;hip;--offload-arch=gfx1201;-mno-wavefrontsize64;-include;${NINFER_AMD_SEGMENTED_VISION_BROKER_VIEW}/src/core/hip_intrinsics.h;--rocm-path=${ROCM_ROOT};--rocm-device-lib-path=${ROCM_ROOT}/lib/llvm/amdgcn/bitcode")
    endif()
  endforeach()
endfunction()
add_library(ninfer_segmented_vision_core_amd STATIC EXCLUDE_FROM_ALL ${_svb_core_sources})
ninfer_segmented_vision_native(ninfer_segmented_vision_core_amd)
add_library(ninfer_segmented_vision_broker_amd STATIC EXCLUDE_FROM_ALL
  "${NINFER_AMD_SEGMENTED_VISION_BROKER_VIEW}/amd-segmented-vision/vision_broker.cpp")
ninfer_segmented_vision_native(ninfer_segmented_vision_broker_amd)
target_link_libraries(ninfer_segmented_vision_broker_amd PUBLIC ninfer_segmented_vision_core_amd)
add_executable(ninfer_amd_segmented_vision_broker_test EXCLUDE_FROM_ALL "${_svb_dir}/test_segmented_vision_broker.cu")
ninfer_segmented_vision_native(ninfer_amd_segmented_vision_broker_test)
target_compile_definitions(ninfer_amd_segmented_vision_broker_test PRIVATE
  "NINFER_BROKER_ORIGINAL_CORE_FIXTURE=\"${NINFER_AMD_SEGMENTED_VISION_BROKER_VIEW}/test-support/original_pool_fixture.cu\"")
target_link_libraries(ninfer_amd_segmented_vision_broker_test PRIVATE ninfer_segmented_vision_broker_amd)
set(_svb_map "${CMAKE_CURRENT_BINARY_DIR}/ninfer_amd_segmented_vision_broker_test.map")
target_link_options(ninfer_amd_segmented_vision_broker_test PRIVATE "-Wl,/MAP:${_svb_map}")
set_property(TARGET ninfer_amd_segmented_vision_broker_test APPEND PROPERTY LINK_DEPENDS "${_svb_dir}/verify_segmented_vision_map.py")
add_custom_command(TARGET ninfer_amd_segmented_vision_broker_test POST_BUILD
  COMMAND "${NINFER_PYTHON}" -B "${_svb_dir}/verify_segmented_vision_map.py" --map "${_svb_map}"
  VERBATIM COMMENT "Require real Broker and one full new Core; reject old ModelRuntime/Core/Ops")
set(_svb_normalized)
foreach(_path IN LISTS _svb_inputs)
  get_filename_component(_absolute "${_path}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  file(TO_CMAKE_PATH "${_absolute}" _canonical)
  list(APPEND _svb_normalized "${_canonical}")
endforeach()
list(REMOVE_DUPLICATES _svb_normalized)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_svb_normalized})
# No Core.cmake wiring, Session, ModelRuntime, Engine, automatic CTest or production gate.
