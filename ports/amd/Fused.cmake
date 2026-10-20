# Production projection/fusion families. The public wrappers, weight formats,
# workspace ownership and original independent numerical tests remain intact.
set(NINFER_AMD_FUSED_FAMILIES
  "linear_add;linear_swiglu;gdn_input_proj;gdn_gating_proj;attn_input_proj;linear_pair;linear_topk;rmsnorm_rope"
  CACHE STRING "Production fusion families to build; each keeps its complete original source manifest")
set(_fused_needs_linear FALSE)
set(_fused_needs_aux FALSE)
set(_fused_aux_families linear_add linear_swiglu gdn_input_proj attn_input_proj)
foreach(_family IN LISTS NINFER_AMD_FUSED_FAMILIES)
  if(NOT _family STREQUAL "rmsnorm_rope" AND NOT _family STREQUAL "gdn_gating_proj")
    set(_fused_needs_linear TRUE)
  endif()
  if(_family IN_LIST _fused_aux_families)
    set(_fused_needs_aux TRUE)
  endif()
endforeach()
if(_fused_needs_linear AND NOT TARGET ninfer_linear_ops_amd)
  message(FATAL_ERROR "The selected NINFER_AMD_FUSED families require the actual NINFER_AMD_LINEAR closure")
endif()
file(GLOB_RECURSE _fused_port_inputs CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/fused/*.py")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_fused_port_inputs})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/linear/prepare_linear.py")
foreach(_family IN LISTS NINFER_AMD_FUSED_FAMILIES)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_engine}/src/ops/${_family}/sources.cmake")
endforeach()
execute_process(COMMAND "${NINFER_PYTHON}" "${CMAKE_CURRENT_LIST_DIR}/fused/prepare_fused.py"
  --engine "${_engine}" --source-view "${_prepared}" --output "${_prepared}"
  --families ${NINFER_AMD_FUSED_FAMILIES}
  RESULT_VARIABLE _fused_generation_status COMMAND_ECHO STDOUT)
if(NOT _fused_generation_status EQUAL 0)
  message(FATAL_ERROR "Native AMD fusion source preparation failed")
endif()
file(READ "${_prepared}/fused-manifest.json" _fused_manifest)

# These real integer-activation/BLAS implementations serve several public
# families. Give them one owner rather than duplicating symbols or introducing
# a dependency cycle between LinearAdd and LinearSwiGLU.
set(_fused_aux_relative
  src/ops/linear_swiglu/q4a8/q4a8_linear_swiglu.cu
  src/ops/linear_swiglu/q4a8/q5a8_linear_add.cu
  src/ops/linear_swiglu/q4cublas/w4_cublas_prefill.cu)
set(_fused_aux_paths)
foreach(_source IN LISTS _fused_aux_relative)
  list(APPEND _fused_aux_paths "${_generated}/${_source}")
endforeach()
if(_fused_needs_aux)
  add_library(ninfer_fused_aux_ops_amd STATIC ${_fused_aux_paths})
  ninfer_native_hip(ninfer_fused_aux_ops_amd)
  target_link_libraries(ninfer_fused_aux_ops_amd PUBLIC ninfer_linear_ops_amd
    "${ROCM_ROOT}/lib/hipblas.lib")
endif()
add_library(ninfer_fused_ops_amd INTERFACE)

foreach(_family IN LISTS NINFER_AMD_FUSED_FAMILIES)
  string(JSON _source_count LENGTH "${_fused_manifest}" families ${_family} native_sources)
  math(EXPR _last_source "${_source_count} - 1")
  set(_family_paths)
  foreach(_index RANGE 0 ${_last_source})
    string(JSON _source GET "${_fused_manifest}" families ${_family} native_sources ${_index})
    if(NOT _source IN_LIST _fused_aux_relative)
      list(APPEND _family_paths "${_generated}/${_source}")
    endif()
  endforeach()
  set(_library "ninfer_${_family}_ops_amd")
  add_library(${_library} STATIC ${_family_paths})
  ninfer_native_hip(${_library})
  foreach(_source IN LISTS _family_paths)
    if(_source MATCHES "/nvfp4/.*\\.cu$")
      # Preserve the original __fdiv_rn scale/activation encoding contract using
      # the compiler's native correct-divide path, without unavailable OCML APIs.
      set_property(SOURCE "${_source}" APPEND PROPERTY COMPILE_OPTIONS
        -fhip-fp32-correctly-rounded-divide-sqrt)
    endif()
  endforeach()
  if(_family STREQUAL "rmsnorm_rope" OR _family STREQUAL "gdn_gating_proj")
    target_link_libraries(${_library} PUBLIC ninfer_basic_ops_amd)
  else()
    target_link_libraries(${_library} PUBLIC ninfer_linear_ops_amd)
    if(_family IN_LIST _fused_aux_families)
      target_link_libraries(${_library} PUBLIC ninfer_fused_aux_ops_amd)
    endif()
  endif()
  target_link_libraries(ninfer_fused_ops_amd INTERFACE ${_library})
endforeach()

foreach(_family linear_add linear_swiglu linear_pair)
  if(NOT TARGET ninfer_${_family}_ops_amd)
    continue()
  endif()
  set(_support "ninfer_${_family}_test_support_amd")
  add_library(${_support} STATIC "${_generated}/tests/ops/${_family}/${_family}_test_common.cpp")
  ninfer_native_hip(${_support})
  target_include_directories(${_support} PRIVATE "${_generated}/tests")
  target_compile_options(${_support} PRIVATE -fno-fast-math -ffp-contract=off)
  target_link_libraries(${_support} PUBLIC ninfer_${_family}_ops_amd)
  if(_family STREQUAL "linear_add")
    set(_tests q4_a16 q5_a16 q5_small_t q8_a16 nvfp4 fp8 bf16_a16)
  elseif(_family STREQUAL "linear_swiglu")
    set(_tests q4_a16 q8_a16 q4_a8int q4a8_int nvfp4 fp8 w4_cublas_prefill)
  else()
    set(_tests q8_a16)
  endif()
  foreach(_test IN LISTS _tests)
    if(_test STREQUAL "w4_cublas_prefill" AND
       (NOT TARGET ninfer_gdn_input_proj_ops_amd OR NOT TARGET ninfer_attn_input_proj_ops_amd))
      message(STATUS "Unchanged fused BLAS suite pending complete GDN/attention projection families")
      continue()
    endif()
    set(_target "ninfer_amd_${_family}_${_test}_test")
    add_executable(${_target} "${_generated}/tests/ops/${_family}/test_${_test}.cpp")
    ninfer_native_hip(${_target})
    target_include_directories(${_target} PRIVATE "${_generated}/tests")
    target_compile_options(${_target} PRIVATE -fno-fast-math -ffp-contract=off)
    target_link_libraries(${_target} PRIVATE ${_support})
    if(_test STREQUAL "w4_cublas_prefill")
      target_link_libraries(${_target} PRIVATE ninfer_linear_add_ops_amd
        ninfer_gdn_input_proj_ops_amd ninfer_attn_input_proj_ops_amd)
    endif()
    ninfer_add_gpu_test(${_target} ${_target} 1800)
  endforeach()
endforeach()

foreach(_test gdn_input_proj gdn_input_proj_conv_record gdn_input_proj_conv_snapshot
  gdn_gating_proj attn_input_proj attn_input_proj_fused_rmsnorm linear_topk rmsnorm_rope)
  string(REGEX REPLACE "_conv_(record|snapshot)$|_fused_rmsnorm$" "" _family "${_test}")
  if(NOT TARGET ninfer_${_family}_ops_amd)
    continue()
  endif()
  set(_extension cpp)
  if(_test STREQUAL "linear_topk")
    set(_extension cu)
  endif()
  set(_target "ninfer_amd_${_test}_op_test")
  add_executable(${_target} "${_generated}/tests/ops/test_${_test}.${_extension}")
  ninfer_native_hip(${_target})
  target_include_directories(${_target} PRIVATE "${_generated}/tests")
  target_compile_options(${_target} PRIVATE -fno-fast-math -ffp-contract=off)
  target_link_libraries(${_target} PRIVATE ninfer_${_family}_ops_amd)
  ninfer_add_gpu_test(${_target} ${_target} 1800)
endforeach()

if(TARGET ninfer_linear_pair_ops_amd)
  add_executable(ninfer_amd_linear_pair_grouped_boundary_test
    "${CMAKE_CURRENT_LIST_DIR}/fused/tests/pair_grouped_boundary.cpp")
  ninfer_native_hip(ninfer_amd_linear_pair_grouped_boundary_test)
  target_include_directories(ninfer_amd_linear_pair_grouped_boundary_test PRIVATE "${_generated}/tests")
  target_compile_options(ninfer_amd_linear_pair_grouped_boundary_test PRIVATE -fno-fast-math -ffp-contract=off)
  target_link_libraries(ninfer_amd_linear_pair_grouped_boundary_test PRIVATE ninfer_linear_pair_ops_amd)
  ninfer_add_gpu_test(ninfer_amd_linear_pair_grouped_boundary_test
    ninfer_amd_linear_pair_grouped_boundary_test 180)
endif()

if(TARGET ninfer_gdn_ops_amd AND TARGET ninfer_gdn_input_proj_ops_amd)
  # The original complete replay-fold suite also exercises public fused
  # convolution snapshot/record projections; both actual families must link.
  add_executable(ninfer_amd_gdn_replay_fold_op_test
    "${_generated}/tests/ops/test_gdn_replay_fold.cpp")
  ninfer_native_hip(ninfer_amd_gdn_replay_fold_op_test)
  target_include_directories(ninfer_amd_gdn_replay_fold_op_test PRIVATE "${_generated}/tests")
  target_compile_options(ninfer_amd_gdn_replay_fold_op_test PRIVATE -fno-fast-math -ffp-contract=off)
  target_link_libraries(ninfer_amd_gdn_replay_fold_op_test PRIVATE
    ninfer_gdn_ops_amd ninfer_gdn_input_proj_ops_amd)
  ninfer_add_gpu_test(ninfer_amd_gdn_replay_fold_op_test ninfer_amd_gdn_replay_fold_op_test 1800)
endif()

if(TARGET ninfer_linear_add_ops_amd AND TARGET ninfer_linear_swiglu_ops_amd)
  # Unchanged combined GGUF oracle exercises public Linear, LinearAdd and both
  # LinearSwiGLU parent forms, with no GGUF-only wrapper or removed assertion.
  add_executable(ninfer_amd_linear_gguf_test "${_generated}/tests/ops/linear/test_gguf.cpp")
  ninfer_native_hip(ninfer_amd_linear_gguf_test)
  target_include_directories(ninfer_amd_linear_gguf_test PRIVATE "${_generated}/tests")
  target_compile_options(ninfer_amd_linear_gguf_test PRIVATE -fno-fast-math -ffp-contract=off)
  # The SDK's host float-to-__bf16 conversion uses compiler-rt on Windows.
  execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -print-resource-dir
    RESULT_VARIABLE _fused_gguf_resource_status
    OUTPUT_VARIABLE _fused_gguf_resource_dir
    ERROR_VARIABLE _fused_gguf_resource_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _fused_gguf_resource_status STREQUAL "0" OR
      _fused_gguf_resource_dir STREQUAL "")
    message(FATAL_ERROR "Cannot locate compiler-rt for the combined GGUF test: ${_fused_gguf_resource_error}")
  endif()
  file(TO_CMAKE_PATH "${_fused_gguf_resource_dir}" _fused_gguf_resource_dir)
  set(_fused_gguf_builtins
    "${_fused_gguf_resource_dir}/lib/windows/clang_rt.builtins-x86_64.lib")
  if(NOT IS_DIRECTORY "${_fused_gguf_resource_dir}" OR
      NOT EXISTS "${_fused_gguf_builtins}" OR IS_DIRECTORY "${_fused_gguf_builtins}")
    message(FATAL_ERROR "Missing native Windows compiler-rt builtins: ${_fused_gguf_builtins}")
  endif()
  target_link_libraries(ninfer_amd_linear_gguf_test PRIVATE ninfer_linear_test_support_amd
    ninfer_linear_add_ops_amd ninfer_linear_swiglu_ops_amd "${_fused_gguf_builtins}")
  ninfer_add_gpu_test(ninfer_amd_linear_gguf_test ninfer_amd_linear_gguf_test 1800)
endif()
