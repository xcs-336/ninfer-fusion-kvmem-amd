# Independent NEW default-OFF leaf. Root does not include/configure it this turn.
option(NINFER_AMD_VISION_SEGMENTED_CAPACITY_TEST "Build real optional-capacity Core fixture" OFF)
if(NOT NINFER_AMD_VISION_SEGMENTED_CAPACITY_TEST)
  return()
endif()
if(NOT WIN32 OR NOT DEFINED NINFER_PYTHON OR NOT DEFINED ROCM_ROOT OR
    NOT DEFINED _generated OR NOT DEFINED _engine)
  message(FATAL_ERROR "Capacity fixture needs the actual Windows native build environment")
endif()
set(NINFER_AMD_VISION_SEGMENTED_CAPACITY_CONTROLLER_VIEW "" CACHE PATH "Exact frozen full controller Core view")
if(NOT IS_ABSOLUTE "${NINFER_AMD_VISION_SEGMENTED_CAPACITY_CONTROLLER_VIEW}")
  message(FATAL_ERROR "Explicit existing full controller view required")
endif()
set(_capacity_test_dir "${CMAKE_CURRENT_LIST_DIR}")
set(_capacity_recipe_dir "${_capacity_test_dir}/../vision_segmented_capacity")
file(SHA256 "${_capacity_recipe_dir}/prepare_vision_segmented_capacity.py" _capacity_preparer_sha)
if(NOT _capacity_preparer_sha STREQUAL "101de66a965b34a2fc2f48de12e1e55a39dd8076916653a532f1eb4a08cd30d1")
  message(FATAL_ERROR "Exact frozen capacity v2 path-guard recipe required")
endif()
file(GLOB_RECURSE _capacity_controller_inputs CONFIGURE_DEPENDS
  "${NINFER_AMD_VISION_SEGMENTED_CAPACITY_CONTROLLER_VIEW}/*")
set(_capacity_inputs ${_capacity_controller_inputs}
  "${CMAKE_CURRENT_LIST_FILE}" "${_capacity_test_dir}/test_vision_segmented_capacity.cu"
  "${_capacity_test_dir}/verify_capacity_map.py" "${_capacity_test_dir}/README.md"
  "${_capacity_recipe_dir}/prepare_vision_segmented_capacity.py"
  "${_capacity_recipe_dir}/manifest.json" "${_capacity_recipe_dir}/check_source.py"
  "${_capacity_recipe_dir}/README.md")
# Use the generator's source-content key. No duplicate CMake recipe implementation,
# generated source list file, or unconditional file(WRITE) can perturb no-op mtimes.
execute_process(COMMAND "${NINFER_PYTHON}" -B "${_capacity_recipe_dir}/prepare_vision_segmented_capacity.py"
  --controller-view "${NINFER_AMD_VISION_SEGMENTED_CAPACITY_CONTROLLER_VIEW}"
  --output-root "${CMAKE_CURRENT_BINARY_DIR}/vision-segmented-capacity-candidate"
  --emit --verify-existing
  RESULT_VARIABLE _capacity_result OUTPUT_VARIABLE _capacity_plan ERROR_VARIABLE _capacity_error)
if(NOT _capacity_result EQUAL 0)
  message(FATAL_ERROR "Optional-capacity source isolation refused: ${_capacity_error}")
endif()
string(JSON NINFER_AMD_VISION_SEGMENTED_CAPACITY_VIEW GET "${_capacity_plan}" output)
file(TO_CMAKE_PATH "${NINFER_AMD_VISION_SEGMENTED_CAPACITY_VIEW}" NINFER_AMD_VISION_SEGMENTED_CAPACITY_VIEW)
file(READ "${NINFER_AMD_VISION_SEGMENTED_CAPACITY_VIEW}/vision-segmented-capacity-source-receipt.json" _capacity_receipt)
string(JSON _capacity_count LENGTH "${_capacity_receipt}" compile_sources)
if(NOT _capacity_count EQUAL 25)
  message(FATAL_ERROR "Exact full 25-TU Core rebuild required")
endif()
set(_capacity_sources)
math(EXPR _capacity_last "${_capacity_count}-1")
foreach(_index RANGE 0 ${_capacity_last})
  string(JSON _relative GET "${_capacity_receipt}" compile_sources ${_index})
  list(APPEND _capacity_sources "${NINFER_AMD_VISION_SEGMENTED_CAPACITY_VIEW}/${_relative}")
endforeach()
find_package(Threads REQUIRED)
function(ninfer_capacity_native target)
  target_compile_definitions(${target} PUBLIC __HIP_PLATFORM_AMD__=1 NINFER_HIP=1 NOMINMAX WIN32_LEAN_AND_MEAN)
  target_compile_features(${target} PUBLIC cxx_std_20)
  target_compile_options(${target} PRIVATE -O2 -fno-fast-math -ffp-contract=off -fms-runtime-lib=dll)
  target_include_directories(${target} BEFORE PUBLIC
    "${NINFER_AMD_VISION_SEGMENTED_CAPACITY_VIEW}/src"
    "${NINFER_AMD_VISION_SEGMENTED_CAPACITY_VIEW}/amd-memory"
    "${ROCM_ROOT}/include" "${_generated}/include" "${_engine}/include")
  target_link_libraries(${target} PUBLIC "${ROCM_ROOT}/lib/amdhip64.lib" Threads::Threads advapi32)
  target_link_options(${target} PUBLIC --hip-link --offload-arch=gfx1201 "--rocm-path=${ROCM_ROOT}")
  get_target_property(_sources ${target} SOURCES)
  foreach(_source IN LISTS _sources)
    if(_source MATCHES "\\.cu$")
      set_source_files_properties("${_source}" PROPERTIES LANGUAGE CXX COMPILE_OPTIONS
        "-x;hip;--offload-arch=gfx1201;-mno-wavefrontsize64;-include;${NINFER_AMD_VISION_SEGMENTED_CAPACITY_VIEW}/src/core/hip_intrinsics.h;--rocm-path=${ROCM_ROOT};--rocm-device-lib-path=${ROCM_ROOT}/lib/llvm/amdgcn/bitcode")
    endif()
  endforeach()
endfunction()
add_library(ninfer_vision_segmented_capacity_core_amd STATIC EXCLUDE_FROM_ALL ${_capacity_sources})
ninfer_capacity_native(ninfer_vision_segmented_capacity_core_amd)
add_executable(ninfer_amd_vision_segmented_capacity_test EXCLUDE_FROM_ALL
  "${_capacity_test_dir}/test_vision_segmented_capacity.cu")
ninfer_capacity_native(ninfer_amd_vision_segmented_capacity_test)
target_compile_definitions(ninfer_amd_vision_segmented_capacity_test PRIVATE
  "NINFER_CAPACITY_ORIGINAL_CORE_FIXTURE=\"${NINFER_AMD_VISION_SEGMENTED_CAPACITY_VIEW}/test-support/original_pool_fixture.cu\"")
target_link_libraries(ninfer_amd_vision_segmented_capacity_test PRIVATE ninfer_vision_segmented_capacity_core_amd)
set(_capacity_map "${CMAKE_CURRENT_BINARY_DIR}/ninfer_amd_vision_segmented_capacity_test.map")
target_link_options(ninfer_amd_vision_segmented_capacity_test PRIVATE "-Wl,/MAP:${_capacity_map}")
set_property(TARGET ninfer_amd_vision_segmented_capacity_test APPEND PROPERTY LINK_DEPENDS
  "${_capacity_test_dir}/verify_capacity_map.py")
add_custom_command(TARGET ninfer_amd_vision_segmented_capacity_test POST_BUILD
  COMMAND "${NINFER_PYTHON}" -B "${_capacity_test_dir}/verify_capacity_map.py" --map "${_capacity_map}"
  VERBATIM COMMENT "Require new optional/throwing storage and original allocator from one full new Core")
# No Core.cmake include, old target link, provider, CTest auto-run or production gate.
set(_capacity_normalized_inputs)
foreach(_input IN LISTS _capacity_inputs)
  get_filename_component(_absolute "${_input}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  file(TO_CMAKE_PATH "${_absolute}" _canonical)
  list(APPEND _capacity_normalized_inputs "${_canonical}")
endforeach()
list(REMOVE_DUPLICATES _capacity_normalized_inputs)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_capacity_normalized_inputs})
