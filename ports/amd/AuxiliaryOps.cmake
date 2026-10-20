# Complete original Ops with their real storage/math dependencies. Keep each
# family independent so a qualification result has an explicit scope.
if(NOT TARGET ninfer_basic_ops_amd OR NOT TARGET ninfer_linear_ops_amd)
  message(FATAL_ERROR "Auxiliary Ops require native BasicOps and public Linear closure")
endif()
file(GLOB_RECURSE _auxiliary_inputs CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/auxiliary/*.py" "${CMAKE_CURRENT_LIST_DIR}/auxiliary/*.cuh")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_auxiliary_inputs})
execute_process(COMMAND "${NINFER_PYTHON}"
  "${CMAKE_CURRENT_LIST_DIR}/auxiliary/prepare_auxiliary.py"
  --engine "${_engine}" --source-view "${_prepared}" --output "${_prepared}"
  RESULT_VARIABLE _auxiliary_generation_status COMMAND_ECHO STDOUT)
if(NOT _auxiliary_generation_status EQUAL 0)
  message(FATAL_ERROR "Native HIP auxiliary source preparation failed")
endif()
file(READ "${_prepared}/auxiliary-manifest.json" _auxiliary_manifest)
foreach(_family candidate_selector dynamic_grouped_conv kvmem sparse_moe)
  string(JSON _source_count LENGTH "${_auxiliary_manifest}" families ${_family})
  math(EXPR _last_source "${_source_count} - 1")
  set(_paths)
  foreach(_index RANGE 0 ${_last_source})
    string(JSON _source GET "${_auxiliary_manifest}" families ${_family} ${_index})
    list(APPEND _paths "${_generated}/${_source}")
  endforeach()
  set(_target "ninfer_${_family}_ops_amd")
  add_library(${_target} STATIC ${_paths})
  ninfer_native_hip(${_target})
  target_link_libraries(${_target} PUBLIC ninfer_basic_ops_amd)
  if(_family STREQUAL "dynamic_grouped_conv" OR _family STREQUAL "sparse_moe")
    # SparseMoE's original wrapper shares the production NVFP4 weight-format
    # validator owned by Linear, in addition to its own expert kernels.
    target_link_libraries(${_target} PUBLIC ninfer_linear_ops_amd)
  elseif(_family STREQUAL "kvmem")
    # Preserve the original imported qw3/ header spelling and attribution.
    target_include_directories(${_target} PRIVATE "${_generated}/src/ops/kvmem"
      "${_engine}/src/ops/kvmem")
  endif()
endforeach()

foreach(_test candidate_selector dynamic_grouped_conv_prepare linear_dynamic_grouped_conv_add
              sparse_moe sparse_moe_route_network)
  set(_target "ninfer_amd_${_test}_op_test")
  if(_test STREQUAL "sparse_moe_route_network")
    set(_extension cu)
  else()
    set(_extension cpp)
  endif()
  add_executable(${_target} "${_generated}/tests/ops/test_${_test}.${_extension}")
  ninfer_native_hip(${_target})
  target_include_directories(${_target} PRIVATE "${_generated}/tests")
  target_compile_options(${_target} PRIVATE -fno-fast-math -ffp-contract=off)
  if(_test STREQUAL "candidate_selector")
    target_link_libraries(${_target} PRIVATE ninfer_candidate_selector_ops_amd)
  elseif(_test MATCHES "dynamic_grouped_conv")
    target_link_libraries(${_target} PRIVATE ninfer_dynamic_grouped_conv_ops_amd)
  elseif(_test STREQUAL "sparse_moe_route_network")
    # This original exact oracle instantiates the real header-only sort on
    # device; it does not need to wait for all expert projection objects.
    target_link_libraries(${_target} PRIVATE ninfer_basic_ops_amd)
  else()
    target_link_libraries(${_target} PRIVATE ninfer_sparse_moe_ops_amd)
  endif()
  ninfer_add_gpu_test(${_target} ${_target} 1800)
endforeach()
ninfer_add_gpu_test(ninfer_amd_sparse_moe_wide
  ninfer_amd_sparse_moe_op_test 1800 --wide-only)
# KVMem has no original standalone in-tree oracle. This additional qualification
# directly invokes its original launchers, checking raw bit-copy, mean-K sum and
# count, and FP64 retrieval scoring. Compilation alone is not a math pass.
add_executable(ninfer_amd_kvmem_math_op_test
  "${CMAKE_CURRENT_LIST_DIR}/auxiliary/test_kvmem_math.cpp")
ninfer_native_hip(ninfer_amd_kvmem_math_op_test)
target_include_directories(ninfer_amd_kvmem_math_op_test PRIVATE "${_generated}/tests")
target_compile_options(ninfer_amd_kvmem_math_op_test PRIVATE -fno-fast-math -ffp-contract=off)
target_link_libraries(ninfer_amd_kvmem_math_op_test PRIVATE ninfer_kvmem_ops_amd)
ninfer_add_gpu_test(ninfer_amd_kvmem_math_op_test ninfer_amd_kvmem_math_op_test 60)
