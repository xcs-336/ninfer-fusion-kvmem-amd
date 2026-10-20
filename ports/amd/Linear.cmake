# The actual public Linear wrapper, all registered native format dispatchers and their original
# mathematical tests. Common Core/BasicOps and the native GGUF kernel library are prerequisites.
if(NOT TARGET ninfer_basic_ops_amd OR NOT TARGET ninfer_ggml_quants_amd)
  message(FATAL_ERROR "NINFER_AMD_LINEAR requires NINFER_AMD_BASIC_OPS and its native GGUF closure")
endif()
file(GLOB_RECURSE _linear_port_inputs CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/linear/*.py")
file(GLOB_RECURSE _linear_original_manifests CONFIGURE_DEPENDS "${_engine}/src/ops/linear/sources.cmake"
  "${_engine}/src/ops/linear/*/sources.cmake")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_linear_port_inputs} ${_linear_original_manifests})
execute_process(COMMAND "${NINFER_PYTHON}" "${CMAKE_CURRENT_LIST_DIR}/linear/prepare_linear.py"
  --engine "${_engine}" --source-view "${_prepared}" --output "${_prepared}"
  RESULT_VARIABLE _linear_generation_status COMMAND_ECHO STDOUT)
if(NOT _linear_generation_status EQUAL 0)
  message(FATAL_ERROR "Native AMD Linear source preparation failed")
endif()
file(READ "${_prepared}/linear-manifest.json" _linear_manifest)
string(JSON _linear_source_count LENGTH "${_linear_manifest}" native_sources)
math(EXPR _linear_last_source "${_linear_source_count} - 1")
set(_linear_paths)
foreach(_index RANGE 0 ${_linear_last_source})
  string(JSON _source GET "${_linear_manifest}" native_sources ${_index})
  list(APPEND _linear_paths "${_generated}/${_source}")
endforeach()
add_library(ninfer_linear_ops_amd STATIC ${_linear_paths}
  "${CMAKE_CURRENT_LIST_DIR}/linear/resource_probe.cu")
ninfer_native_hip(ninfer_linear_ops_amd)
# NVFP4 activation encoding has explicit __fdiv_rn boundaries. The installed HIP wrapper is
# ordinary division; make its correct-rounding mode explicit only in the actual codec consumers.
foreach(_source IN LISTS _linear_paths)
  if(_source MATCHES "/nvfp4/.*\\.cu$")
    set_property(SOURCE "${_source}" APPEND PROPERTY COMPILE_OPTIONS
      -fhip-fp32-correctly-rounded-divide-sqrt)
  endif()
endforeach()
target_link_libraries(ninfer_linear_ops_amd PUBLIC ninfer_basic_ops_amd)

add_library(ninfer_linear_test_support_amd STATIC
  "${_generated}/tests/ops/linear/linear_test_common.cpp")
ninfer_native_hip(ninfer_linear_test_support_amd)
target_include_directories(ninfer_linear_test_support_amd PRIVATE "${_generated}/tests")
target_link_libraries(ninfer_linear_test_support_amd PUBLIC ninfer_linear_ops_amd)
target_compile_options(ninfer_linear_test_support_amd PRIVATE -fno-fast-math -ffp-contract=off)

foreach(_test q4_a16 q5_a16 q6_a16 q8_a16 t2_a16 t2_a8 nvfp4_a16 nvfp4_a4 fp8_a16 fp8_a8 bf16_a16)
  set(_target "ninfer_amd_linear_${_test}_test")
  add_executable(${_target} "${_generated}/tests/ops/linear/test_${_test}.cpp")
  ninfer_native_hip(${_target})
  target_include_directories(${_target} PRIVATE "${_generated}/tests")
  target_link_libraries(${_target} PRIVATE ninfer_linear_test_support_amd)
  target_compile_options(${_target} PRIVATE -fno-fast-math -ffp-contract=off)
  ninfer_add_gpu_test(${_target} ${_target} 1800)
endforeach()

# Bound the host fixture while exercising the same public dispatch and upstream FP64 oracle.
# These first-execution probes complement every unchanged original family suite above.
add_executable(ninfer_amd_linear_public_probe "${CMAKE_CURRENT_LIST_DIR}/linear/public_probe.cpp")
ninfer_native_hip(ninfer_amd_linear_public_probe)
target_include_directories(ninfer_amd_linear_public_probe PRIVATE "${_generated}/tests")
target_link_libraries(ninfer_amd_linear_public_probe PRIVATE ninfer_linear_test_support_amd)
target_compile_options(ninfer_amd_linear_public_probe PRIVATE -fno-fast-math -ffp-contract=off)
foreach(_family q4 q5 q6 q8 t2 fp8 nvfp4)
  ninfer_add_gpu_test(ninfer_amd_linear_${_family}_public_probe
    ninfer_amd_linear_public_probe 180 ${_family})
endforeach()
# test_gguf.cpp also exercises the complete public LinearAdd/LinearSwiGLU wrappers. Their full
# production families are required before this unchanged test can link; no GGUF-only wrapper or
# removed assertion substitutes for that dependency. The production GGUF Linear route is already
# linked through the reused BasicOps/Gguf.cmake closure above.
message(STATUS "Original Linear GGUF combined test pending complete LinearAdd/LinearSwiGLU closure")
# The original staging-race test also calls fused LinearSwiGLU and requires actual TMA hardware.
# It cannot qualify the non-TMA native path and is not registered as a passing or skipped substitute.
